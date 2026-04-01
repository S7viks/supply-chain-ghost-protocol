import copy

import pytest

from env import SupplyChainEnv
from models import Action, ActionType
from tasks import TASK_EASY, TASK_HARD, TASK_MEDIUM, run_task


def test_reset_determinism() -> None:
    env1 = SupplyChainEnv(episode_length=7, seed=42, shock_config={})
    env2 = SupplyChainEnv(episode_length=7, seed=42, shock_config={})

    obs1 = env1.reset()
    obs2 = env2.reset()

    assert obs1.model_dump() == obs2.model_dump()


def test_step_advances_day() -> None:
    env = SupplyChainEnv(episode_length=3, seed=42, shock_config={})
    obs = env.reset()
    assert obs.day == 0

    obs, reward, done, info = env.step(Action(action_type=ActionType.NOOP))
    # Implementation reports the current day *after* advancing the clock.
    assert info["day"] == 1
    assert obs.day == 1
    assert done is False
    assert reward.done is False


def test_episode_termination() -> None:
    env = SupplyChainEnv(episode_length=2, seed=42, shock_config={})
    env.reset()

    _, _, done1, _ = env.step(Action(action_type=ActionType.NOOP))
    _, _, done2, _ = env.step(Action(action_type=ActionType.NOOP))
    assert done1 is False
    assert done2 is True


def test_bullwhip_amplification_present() -> None:
    env = TASK_HARD.build_env(seed=42)
    obs = env.reset()

    seen = False
    for _ in range(10):
        obs, _, done, _ = env.step(Action(action_type=ActionType.NOOP))
        if obs.bullwhip_index > 1.0:
            seen = True
            break
        if done:
            break

    assert seen is True


def test_demand_spike_resets_next_day() -> None:
    env = SupplyChainEnv(
        episode_length=3,
        seed=42,
        shock_config={"random_demand_spikes": True},
    )
    env.reset()

    fid = "FAC_TAIPEI"
    original = env._factory_demands[fid].daily_burn_rate  # type: ignore[attr-defined]

    class _RngStub:
        def __init__(self) -> None:
            self._calls = 0

        def random(self) -> float:
            self._calls += 1
            return 0.0 if self._calls == 1 else 1.0  # spike only once

        def uniform(self, a: float, b: float) -> float:
            return 1.5

        def choice(self, xs):
            return fid

    env._rng = _RngStub()  # type: ignore[assignment]

    env._apply_shocks()  # type: ignore[attr-defined]
    spiked = env._factory_demands[fid].daily_burn_rate  # type: ignore[attr-defined]
    assert spiked == pytest.approx(original * 1.5)

    env._day += 1  # type: ignore[attr-defined]
    env._apply_shocks()  # type: ignore[attr-defined]
    reset_rate = env._factory_demands[fid].daily_burn_rate  # type: ignore[attr-defined]
    assert reset_rate == pytest.approx(original)


def test_grader_score_range() -> None:
    noop = lambda obs: Action(action_type=ActionType.NOOP)
    for task in (TASK_EASY, TASK_MEDIUM, TASK_HARD):
        result, _ = run_task(task, noop, seed=42)
        assert 0.0 <= result.score <= 1.0


def test_difficulty_progression_noop() -> None:
    noop = lambda obs: Action(action_type=ActionType.NOOP)
    r_easy, _ = run_task(TASK_EASY, noop, seed=42)
    r_med, _ = run_task(TASK_MEDIUM, noop, seed=42)
    r_hard, _ = run_task(TASK_HARD, noop, seed=42)

    assert r_easy.score > r_med.score
    assert r_med.score > r_hard.score

