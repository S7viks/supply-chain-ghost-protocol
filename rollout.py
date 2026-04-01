"""
Supply Chain Ghost Protocol — Rollout Recording
================================================
Captures full-state episode rollouts for 3D visualization, including
inventory per node, ship positions, port statuses, and reward breakdowns.

Usage:
    python rollout.py --task easy --seed 42 --policy heuristic --output rollouts/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from models import (
    Action,
    ActionType,
    Observation,
    PortStatus,
    RewardComponents,
    ShipStatus,
)
from tasks import (
    ALL_TASKS,
    GRADERS,
    TASK_CASCADE,
    TASK_EASY,
    TASK_HARD,
    TASK_MEDIUM,
    Task,
)


# ─── Rollout Data Models ─────────────────────────────────────────────────────


class RolloutStep(BaseModel):
    """Single time-step snapshot for 3D viewer consumption."""

    day: int
    action: Dict[str, Any] = Field(default_factory=dict)
    reward_total: float = 0.0
    reward_components: Dict[str, float] = Field(default_factory=dict)
    bullwhip_index: float = 1.0
    service_level: float = 1.0
    any_factory_stockout: bool = False
    inventory: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ships: List[Dict[str, Any]] = Field(default_factory=list)
    active_shocks: List[str] = Field(default_factory=list)
    demand_fulfilled: float = 0.0
    demand_total: float = 0.0


class Rollout(BaseModel):
    """Complete episode rollout for a single policy run."""

    task_id: str
    task_name: str
    difficulty: str
    seed: int
    policy_name: str
    episode_length: int
    final_score: float = 0.0
    final_service_level: float = 1.0
    steps: List[RolloutStep] = Field(default_factory=list)


# ─── Network Topology (for heuristic decisions) ─────────────────────────────

FACTORY_IDS = frozenset(("FAC_TAIPEI", "FAC_BERLIN", "FAC_AUSTIN"))

FACTORY_WAREHOUSE: Dict[str, str] = {
    "FAC_TAIPEI": "WH_ASIA_HUB",
    "FAC_AUSTIN": "WH_ASIA_HUB",
    "FAC_BERLIN": "WH_EUROPE_HUB",
}

PORT_WAREHOUSE: Dict[str, str] = {
    "PORT_SINGAPORE": "WH_ASIA_HUB",
    "PORT_SHANGHAI": "WH_ASIA_HUB",
    "PORT_BUSAN": "WH_ASIA_HUB",
    "PORT_ROTTERDAM": "WH_EUROPE_HUB",
    "PORT_HAMBURG": "WH_EUROPE_HUB",
}

ASIA_PORTS = ("PORT_BUSAN", "PORT_SHANGHAI", "PORT_SINGAPORE")
EUROPE_PORTS = ("PORT_HAMBURG", "PORT_ROTTERDAM")


# ─── State Snapshot Helpers ──────────────────────────────────────────────────


def _snapshot_inventory(obs: Observation) -> Dict[str, Dict[str, Any]]:
    """Extract per-node inventory with status annotations."""
    result: Dict[str, Dict[str, Any]] = {}
    for node_id, inv in obs.inventory_levels.items():
        status = "active"
        if node_id in obs.port_status:
            status = obs.port_status[node_id].value
        result[node_id] = {
            "current_stock": round(inv.current_stock, 2),
            "capacity": round(inv.capacity, 2),
            "safety_stock": round(inv.safety_stock, 2),
            "status": status,
        }
    return result


def _snapshot_ships(obs: Observation) -> List[Dict[str, Any]]:
    """Extract ship positions for visualization."""
    return [
        {
            "ship_id": s.ship_id,
            "origin": s.origin_port,
            "destination": s.destination_port,
            "cargo": round(s.cargo_volume, 2),
            "eta": round(s.eta_days, 2),
            "status": s.status.value,
        }
        for s in obs.transit_queue
    ]


def _snapshot_step(
    obs: Observation,
    action: Action,
    reward_total: float,
    reward_components: Optional[RewardComponents],
    demand_fulfilled: float,
    demand_total: float,
) -> RolloutStep:
    """Build a single RolloutStep from environment state."""
    any_stockout = any(
        obs.inventory_levels[fid].current_stock <= 0
        for fid in FACTORY_IDS
        if fid in obs.inventory_levels
    )

    comp_dict: Dict[str, float] = {}
    if reward_components is not None:
        comp_dict = {
            "service_level_bonus": reward_components.service_level_bonus,
            "stockout_penalty": reward_components.stockout_penalty,
            "holding_cost_penalty": reward_components.holding_cost_penalty,
            "reroute_cost_penalty": reward_components.reroute_cost_penalty,
            "expedite_cost_penalty": reward_components.expedite_cost_penalty,
            "bullwhip_penalty": reward_components.bullwhip_penalty,
            "stability_bonus": reward_components.stability_bonus,
        }

    return RolloutStep(
        day=obs.day,
        action=action.model_dump(exclude_none=True),
        reward_total=round(reward_total, 4),
        reward_components=comp_dict,
        bullwhip_index=round(obs.bullwhip_index, 4),
        service_level=round(obs.network_service_level, 4),
        any_factory_stockout=any_stockout,
        inventory=_snapshot_inventory(obs),
        ships=_snapshot_ships(obs),
        active_shocks=list(obs.active_shocks),
        demand_fulfilled=round(demand_fulfilled, 2),
        demand_total=round(demand_total, 2),
    )


# ─── Rollout Recorder ───────────────────────────────────────────────────────


def record_rollout(
    task: Task,
    agent_fn: Callable[..., Action],
    seed: int = 42,
    policy_name: str = "unknown",
) -> Rollout:
    """Run a full episode capturing every state snapshot for visualization.

    Produces a ``Rollout`` containing one ``RolloutStep`` per simulation day
    (including day 0 — the initial state before any action).  Also grades
    the episode using the task's registered grader.
    """
    env = task.build_env(seed=seed)
    obs = env.reset()

    rollout = Rollout(
        task_id=task.task_id,
        task_name=task.name,
        difficulty=task.difficulty,
        seed=seed,
        policy_name=policy_name,
        episode_length=task.episode_length,
    )

    noop = Action(action_type=ActionType.NOOP)
    rollout.steps.append(_snapshot_step(obs, noop, 0.0, None, 0.0, 0.0))

    done = False
    prev_step: Optional[Dict[str, Any]] = None
    trajectory: List[Dict[str, Any]] = []

    while not done:
        try:
            action = agent_fn(obs, prev_step, task)
        except TypeError:
            action = agent_fn(obs)

        next_obs, reward, done, info = env.step(action)

        any_stockout = any(
            next_obs.inventory_levels[fid].current_stock <= 0
            for fid in FACTORY_IDS
            if fid in next_obs.inventory_levels
        )

        step_data: Dict[str, Any] = {
            **info,
            "action_type": action.action_type.value,
            "reward_total": reward.total,
            "bullwhip_index": next_obs.bullwhip_index,
            "service_level": next_obs.network_service_level,
            "any_factory_stockout": any_stockout,
        }
        trajectory.append(step_data)
        prev_step = step_data

        rollout.steps.append(
            _snapshot_step(
                next_obs,
                action,
                reward.total,
                reward.components,
                info.get("demand_fulfilled", 0.0),
                info.get("demand_total", 0.0),
            )
        )
        obs = next_obs

    grader = GRADERS.get(task.task_id)
    if grader:
        result = grader(env, trajectory)
        rollout.final_score = result.score
    rollout.final_service_level = obs.network_service_level

    return rollout


# ─── Serialization ──────────────────────────────────────────────────────────


def save_rollout(rollout: Rollout, path: str | Path) -> Path:
    """Serialize rollout to JSON file. Returns the written path."""
    path = Path(path)
    if path.is_dir() or (path.suffix == "" and not path.exists()):
        filename = f"{rollout.task_id}_{rollout.policy_name}_s{rollout.seed}.json"
        path = path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rollout.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_rollout(path: str | Path) -> dict:
    """Deserialize rollout from JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ─── Agent Policies ─────────────────────────────────────────────────────────


class NaiveAgent:
    """Baseline agent that always returns NOOP."""

    def __call__(
        self,
        obs: Observation,
        prev_step: Optional[Dict[str, Any]] = None,
        task: Optional[Task] = None,
    ) -> Action:
        return Action(action_type=ActionType.NOOP)


class HeuristicAgent:
    """Rule-based agent with domain-aware supply chain heuristics.

    Decision priority (highest first):
        1. Expedite to any factory below 2 days of stock
        2. Reroute delayed ships stuck at blocked ports
        3. Buffer warehouses when their upstream ports are disrupted
        4. Buffer warehouses that have fallen below safety stock
        5. NOOP
    """

    def __call__(
        self,
        obs: Observation,
        prev_step: Optional[Dict[str, Any]] = None,
        task: Optional[Task] = None,
    ) -> Action:
        return (
            self._expedite_critical_factory(obs)
            or self._reroute_delayed_ships(obs)
            or self._buffer_disrupted_warehouse(obs)
            or self._buffer_low_warehouse(obs)
            or Action(action_type=ActionType.NOOP)
        )

    def _expedite_critical_factory(self, obs: Observation) -> Optional[Action]:
        """If any factory has < 2 days of stock, rush-ship from its warehouse."""
        critical: List[tuple[float, str, float]] = []
        for fid, demand in obs.daily_burn_rate.items():
            dtc = demand.days_until_critical
            if dtc is not None and dtc < 2.0:
                critical.append((dtc, fid, demand.daily_burn_rate))

        if not critical:
            return None

        critical.sort(key=lambda x: x[0])
        _, fid, burn_rate = critical[0]

        warehouse = FACTORY_WAREHOUSE.get(fid)
        if not warehouse:
            return None

        wh_inv = obs.inventory_levels.get(warehouse)
        if wh_inv is None or wh_inv.current_stock <= 0:
            return None

        volume = min(
            burn_rate * 3.0,
            wh_inv.current_stock * 0.4,
        )
        if volume <= 0:
            return None

        return Action(
            action_type=ActionType.EXPEDITE_ORDER,
            source_node=warehouse,
            destination_node=fid,
            expedite_volume=round(volume, 0),
        )

    def _reroute_delayed_ships(self, obs: Observation) -> Optional[Action]:
        """Reroute ships that are DELAYED (stuck at a blocked port) or
        heading toward a port that is currently blocked."""
        blocked_ports = frozenset(
            pid
            for pid, status in obs.port_status.items()
            if status == PortStatus.BLOCKED
        )
        if not blocked_ports:
            return None

        for ship in obs.transit_queue:
            is_stuck = ship.status == ShipStatus.DELAYED
            heading_to_blocked = ship.destination_port in blocked_ports

            if not (is_stuck or heading_to_blocked):
                continue
            if ship.status == ShipStatus.DOCKED:
                continue

            target_port = ship.destination_port
            region_wh = PORT_WAREHOUSE.get(target_port)
            candidates = ASIA_PORTS if region_wh == "WH_ASIA_HUB" else EUROPE_PORTS

            for port in candidates:
                if port == target_port:
                    continue
                port_status = obs.port_status.get(port)
                if port_status == PortStatus.OPEN:
                    return Action(
                        action_type=ActionType.REROUTE_SHIP,
                        ship_id=ship.ship_id,
                        new_port=port,
                    )

        return None

    def _buffer_disrupted_warehouse(self, obs: Observation) -> Optional[Action]:
        """If a key port is non-OPEN, proactively buffer its warehouse."""
        for port_id, status in obs.port_status.items():
            if status == PortStatus.OPEN:
                continue

            wh_id = PORT_WAREHOUSE.get(port_id)
            if not wh_id:
                continue

            inv = obs.inventory_levels.get(wh_id)
            if inv is None:
                continue

            if inv.current_stock < inv.capacity * 0.4:
                target = min(inv.safety_stock * 1.5, inv.capacity * 0.7)
                if target > inv.current_stock:
                    return Action(
                        action_type=ActionType.ADJUST_BUFFER,
                        target_node=wh_id,
                        target_inventory=round(target, 0),
                    )

        return None

    def _buffer_low_warehouse(self, obs: Observation) -> Optional[Action]:
        """If a warehouse is below safety stock, top it up."""
        worst_ratio = float("inf")
        worst_wh: Optional[str] = None

        for wid in ("WH_ASIA_HUB", "WH_EUROPE_HUB"):
            inv = obs.inventory_levels.get(wid)
            if inv and inv.is_below_safety:
                ratio = inv.current_stock / max(inv.safety_stock, 1.0)
                if ratio < worst_ratio:
                    worst_ratio = ratio
                    worst_wh = wid

        if worst_wh is None:
            return None

        inv = obs.inventory_levels[worst_wh]
        target = min(inv.safety_stock * 1.2, inv.capacity)
        if target <= inv.current_stock:
            return None

        return Action(
            action_type=ActionType.ADJUST_BUFFER,
            target_node=worst_wh,
            target_inventory=round(target, 0),
        )


# ─── Policy Registry ────────────────────────────────────────────────────────

POLICIES: Dict[str, Callable[..., Action]] = {
    "naive": NaiveAgent(),
    "heuristic": HeuristicAgent(),
}

TASK_MAP: Dict[str, Task] = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "cascade": TASK_CASCADE,
    "hard": TASK_HARD,
}


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supply Chain Ghost Protocol -- Rollout Recorder",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_MAP.keys()),
        default="easy",
        help="Task difficulty (default: easy)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--policy",
        choices=list(POLICIES.keys()),
        default="heuristic",
        help="Agent policy (default: heuristic)",
    )
    parser.add_argument(
        "--output",
        default="rollouts/",
        help="Output directory or file path (default: rollouts/)",
    )
    args = parser.parse_args()

    task = TASK_MAP[args.task]
    agent = POLICIES[args.policy]

    print(f"Recording rollout: task={args.task} seed={args.seed} policy={args.policy}")
    rollout = record_rollout(task, agent, seed=args.seed, policy_name=args.policy)

    out_path = save_rollout(rollout, args.output)
    print(f"Score: {rollout.final_score:.4f} | Service Level: {rollout.final_service_level:.2%}")
    print(f"Steps: {len(rollout.steps)} | Written to: {out_path}")


if __name__ == "__main__":
    main()
