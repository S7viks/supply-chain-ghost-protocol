"""
Supply Chain Ghost Protocol — Gymnasium Wrapper
================================================
Gymnasium-compatible wrapper around SupplyChainEnv for use with
standard RL libraries (Stable-Baselines3, etc.).

Observation : 63-dimensional float32 vector
Action      : Discrete(21)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from env import SupplyChainEnv
from models import Action, ActionType, Observation, PortStatus, Ship, ShipStatus
from tasks import ALL_TASKS, Task

# ─── Canonical Node Ordering ─────────────────────────────────────────────────

NODE_ORDER = [
    "PORT_SINGAPORE", "PORT_ROTTERDAM", "PORT_SHANGHAI", "PORT_BUSAN", "PORT_HAMBURG",
    "WH_ASIA_HUB", "WH_EUROPE_HUB",
    "FAC_TAIPEI", "FAC_BERLIN", "FAC_AUSTIN",
]
PORT_IDS = NODE_ORDER[:5]
WAREHOUSE_IDS = NODE_ORDER[5:7]
FACTORY_IDS = NODE_ORDER[7:10]
NODE_INDEX = {name: i for i, name in enumerate(NODE_ORDER)}

EXPEDITE_DESTINATIONS = ["FAC_TAIPEI", "FAC_BERLIN", "FAC_AUSTIN", "WH_ASIA_HUB", "WH_EUROPE_HUB"]
BUFFER_TARGETS = ["FAC_TAIPEI", "FAC_BERLIN", "FAC_AUSTIN", "WH_ASIA_HUB", "WH_EUROPE_HUB"]

MAX_SHIPS = 5
OBS_DIM = 20 + 15 + 9 + 4 + (MAX_SHIPS * 3)  # 63
NUM_ACTIONS = 21  # 0=NOOP, 1-5=REROUTE, 6-10=EXP_ASIA, 11-15=EXP_EUR, 16-20=ADJUST

TASK_MAP: Dict[str, Task] = {t.task_id: t for t in ALL_TASKS}

_PORT_STATUS_INDEX = {"open": 0, "blocked": 1, "congested": 2}


# ─── Gymnasium Environment ───────────────────────────────────────────────────


class GymSupplyChainEnv(gymnasium.Env):
    """Gymnasium wrapper for the Supply Chain Ghost Protocol environment.

    Encodes structured Pydantic observations into a fixed-size float32 vector
    and maps a Discrete action space back to the environment's Action model.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "task_hard_ghost_protocol", seed: int = 42) -> None:
        super().__init__()

        if task_id not in TASK_MAP:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASK_MAP)}")

        self._task: Task = TASK_MAP[task_id]
        self._initial_seed = seed
        self._next_seed = seed
        self._env: SupplyChainEnv = self._task.build_env(seed=seed)
        self._last_obs: Optional[Observation] = None

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    # ─── Gymnasium API ────────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self._next_seed = seed

        self._env.seed = self._next_seed
        obs = self._env.reset()
        self._last_obs = obs
        self._next_seed += 1

        return self._encode_obs(obs), {"raw_obs": obs}

    def step(self, action_int: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = self._decode_action(action_int)
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs

        return (
            self._encode_obs(obs),
            float(reward.total),
            done,   # terminated
            False,  # truncated — env manages its own episode length
            {**info, "raw_obs": obs, "reward_components": reward.components},
        )

    # ─── Observation Encoding ─────────────────────────────────────────────

    def _encode_obs(self, obs: Observation) -> np.ndarray:
        """Flatten structured Observation into a fixed-size float32 vector.

        Layout (63 floats):
            [0:20]  per-node inventory  (10 nodes x 2)
            [20:35] port status one-hot (5 ports x 3)
            [35:44] factory demand      (3 factories x 3)
            [44:48] global scalars      (4)
            [48:63] ship summary        (top-5 ships x 3)
        """
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0
        episode_len = max(self._task.episode_length, 1)

        # Per-node inventory: stock utilisation + below-safety flag
        for node_id in NODE_ORDER:
            inv = obs.inventory_levels.get(node_id)
            if inv is not None:
                vec[idx] = inv.current_stock / max(inv.capacity, 1.0)
                vec[idx + 1] = float(inv.is_below_safety)
            idx += 2

        # Port status one-hot
        for port_id in PORT_IDS:
            status = obs.port_status.get(port_id, PortStatus.OPEN)
            vec[idx + _PORT_STATUS_INDEX.get(status.value, 0)] = 1.0
            idx += 3

        # Factory demand features
        for fac_id in FACTORY_IDS:
            fd = obs.daily_burn_rate.get(fac_id)
            if fd is not None:
                vec[idx] = fd.daily_burn_rate / 1000.0
                dtc = fd.days_until_critical if fd.days_until_critical is not None else 30.0
                vec[idx + 1] = float(np.clip(dtc / 30.0, 0.0, 1.0))
                vec[idx + 2] = fd.demand_volatility
            idx += 3

        # Global scalars
        vec[idx] = float(np.clip(obs.bullwhip_index / 5.0, 0.0, 1.0))
        vec[idx + 1] = obs.network_service_level
        vec[idx + 2] = obs.day / episode_len
        vec[idx + 3] = obs.episode_remaining_days / episode_len
        idx += 4

        # Ship summary — top 5 by soonest ETA, padded with zeros
        ships = sorted(obs.transit_queue, key=lambda s: s.eta_days)[:MAX_SHIPS]
        for ship in ships:
            vec[idx] = ship.cargo_volume / 10_000.0
            vec[idx + 1] = ship.eta_days / 10.0
            vec[idx + 2] = NODE_INDEX.get(ship.destination_port, 0) / 10.0
            idx += 3

        return vec

    # ─── Action Decoding ──────────────────────────────────────────────────

    def _decode_action(self, action_int: int) -> Action:
        """Map a discrete integer action to a structured Action model."""

        # 0: NOOP
        if action_int == 0:
            return Action(action_type=ActionType.NOOP)

        # 1-5: REROUTE first in-transit ship to port [0..4]
        if 1 <= action_int <= 5:
            target_port = PORT_IDS[action_int - 1]
            ship = self._find_first_transit_ship()
            if ship is None:
                return Action(action_type=ActionType.NOOP)
            return Action(
                action_type=ActionType.REROUTE_SHIP,
                ship_id=ship.ship_id,
                new_port=target_port,
            )

        # 6-10: EXPEDITE 500 units from WH_ASIA_HUB to destination [0..4]
        if 6 <= action_int <= 10:
            dest = EXPEDITE_DESTINATIONS[action_int - 6]
            if dest == "WH_ASIA_HUB":
                return Action(action_type=ActionType.NOOP)
            return Action(
                action_type=ActionType.EXPEDITE_ORDER,
                source_node="WH_ASIA_HUB",
                destination_node=dest,
                expedite_volume=500.0,
            )

        # 11-15: EXPEDITE 500 units from WH_EUROPE_HUB to destination [0..4]
        if 11 <= action_int <= 15:
            dest = EXPEDITE_DESTINATIONS[action_int - 11]
            if dest == "WH_EUROPE_HUB":
                return Action(action_type=ActionType.NOOP)
            return Action(
                action_type=ActionType.EXPEDITE_ORDER,
                source_node="WH_EUROPE_HUB",
                destination_node=dest,
                expedite_volume=500.0,
            )

        # 16-20: ADJUST_BUFFER at target [0..4] to safety_stock * 1.5
        if 16 <= action_int <= 20:
            target = BUFFER_TARGETS[action_int - 16]
            safety = self._get_safety_stock(target)
            return Action(
                action_type=ActionType.ADJUST_BUFFER,
                target_node=target,
                target_inventory=safety * 1.5,
            )

        return Action(action_type=ActionType.NOOP)

    # ─── Internal Helpers ─────────────────────────────────────────────────

    def _find_first_transit_ship(self) -> Optional[Ship]:
        if self._last_obs is None:
            return None
        for ship in self._last_obs.transit_queue:
            if ship.status in (ShipStatus.IN_TRANSIT, ShipStatus.REROUTING, ShipStatus.DELAYED):
                return ship
        return None

    def _get_safety_stock(self, node_id: str) -> float:
        if self._last_obs is None:
            return 0.0
        inv = self._last_obs.inventory_levels.get(node_id)
        return inv.safety_stock if inv is not None else 0.0


# ─── Gymnasium Registration ──────────────────────────────────────────────────

gymnasium.register(
    id="SupplyChain-v0",
    entry_point="gym_wrapper:GymSupplyChainEnv",
)
