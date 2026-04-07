"""
Supply Chain Ghost Protocol — Tasks & Graders
==============================================
Three difficulty-tiered tasks (Easy / Medium / Hard) with programmatic
graders that return deterministic scores in [0.0, 1.0].

Grader contract:
    - Deterministic given the same seed.
    - Returns a float in [0.0, 1.0], rounded to 4 decimal places.
    - 0.0 = complete failure, 1.0 = perfect.
    - Hard task: any stock-out event causes a significant score drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from env import SupplyChainEnv
from models import Action, ActionType, Observation


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class TaskResult:
    """Outcome of grading a single task run."""

    task_id: str
    score: float
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Hackathon Phase-2 requires scores strictly within (0, 1).
        # Enforce it centrally so individual graders can stay simple.
        # NOTE: scores are printed to 4 decimals in stdout, so eps must be >= 1e-4
        # to avoid rounding to 0.0000 / 1.0000.
        eps = 1e-4
        if not isinstance(self.score, (int, float)):
            raise TypeError(f"TaskResult.score must be float-like, got {type(self.score).__name__}")
        if self.score <= 0.0:
            self.score = eps
        elif self.score >= 1.0:
            self.score = 1.0 - eps


@dataclass
class Task:
    """Descriptor for a single evaluation task."""

    task_id: str
    name: str
    description: str
    difficulty: str
    episode_length: int
    shock_config: Dict[str, Any]
    success_threshold: float
    tags: List[str] = field(default_factory=list)

    def build_env(self, seed: int = 42) -> SupplyChainEnv:
        """Construct a fresh environment configured for this task."""
        return SupplyChainEnv(
            episode_length=self.episode_length,
            seed=seed,
            shock_config=self.shock_config,
        )


# ─── Task Definitions ────────────────────────────────────────────────────────

TASK_EASY = Task(
    task_id="task_easy_delay",
    name="The Delay",
    difficulty="easy",
    description=(
        "A single vessel (V-001) heading to PORT_SINGAPORE is delayed by 48 hours due to "
        "a weather event. The agent must reroute it to PORT_BUSAN within 1 day to prevent "
        "FAC_TAIPEI from running out of semiconductors before the end of the 7-day episode. "
        "Success requires maintaining >=95% service level across all factories."
    ),
    episode_length=7,
    shock_config={
        "scheduled": [
            {"day": 0, "port": "PORT_SINGAPORE", "status": "congested", "end_day": 2},
        ],
    },
    success_threshold=0.7,
    tags=["rerouting", "single-failure", "time-pressure"],
)

TASK_MEDIUM = Task(
    task_id="task_medium_blockage",
    name="The Blockage",
    difficulty="medium",
    description=(
        "PORT_SINGAPORE -- the largest regional hub -- closes for 5 days (days 2-7) due to a "
        "canal authority dispute, cutting off 40% of Asia-Pacific throughput. The agent must "
        "manage inventory across WH_ASIA_HUB and reroute ships to PORT_BUSAN / PORT_SHANGHAI "
        "to prevent FAC_TAIPEI from hitting a stock-out. Success requires >=0.85 service level "
        "over the 14-day episode with no factory idle day lasting more than 1 day."
    ),
    episode_length=14,
    shock_config={
        "scheduled": [
            {"day": 2, "port": "PORT_SINGAPORE", "status": "blocked", "end_day": 7},
        ],
    },
    success_threshold=0.6,
    tags=["hub-failure", "inventory-management", "multi-ship"],
)

TASK_REROUTE_DRILL = Task(
    task_id="task_easy_reroute_drill",
    name="Reroute Drill",
    difficulty="easy",
    description=(
        "Immediate Singapore blockage (days 0–2). The agent should quickly reroute ships away "
        "from PORT_SINGAPORE and maintain service levels over a short 5-day horizon."
    ),
    episode_length=5,
    shock_config={
        "scheduled": [
            {"day": 0, "port": "PORT_SINGAPORE", "status": "blocked", "end_day": 2},
        ],
        "random_demand_spikes": False,
    },
    success_threshold=0.75,
    tags=["rerouting", "reaction-time", "short-horizon"],
)

TASK_BUFFER_TUNING = Task(
    task_id="task_medium_buffer_tuning",
    name="Buffer Tuning",
    difficulty="medium",
    description=(
        "No port shocks. The agent should keep factories well-buffered without inducing "
        "excessive instability or panic ordering. Tests long-term steadiness and bullwhip control."
    ),
    episode_length=10,
    shock_config={
        "scheduled": [],
        "random_demand_spikes": False,
    },
    success_threshold=0.65,
    tags=["stability", "bullwhip", "inventory-policy"],
)

TASK_EUROPE_SHOCK = Task(
    task_id="task_medium_europe_shock",
    name="Europe Shock",
    difficulty="medium",
    description=(
        "PORT_ROTTERDAM blocks for 5 days (days 2–6), disrupting supply to WH_EUROPE_HUB. "
        "The agent must keep FAC_BERLIN online and maintain overall service levels."
    ),
    episode_length=12,
    shock_config={
        "scheduled": [
            {"day": 2, "port": "PORT_ROTTERDAM", "status": "blocked", "end_day": 6},
        ],
        "random_demand_spikes": False,
    },
    success_threshold=0.6,
    tags=["europe", "berlin", "port-failure"],
)

TASK_CASCADE = Task(
    task_id="task_cascade_dual_chokepoint",
    name="The Cascade",
    difficulty="hard",
    description=(
        "A mid-horizon dual-chokepoint scenario over 20 days. PORT_SINGAPORE blocks for 6 days "
        "(days 3-9), then PORT_ROTTERDAM congests for 6 days (days 10-16). No random demand "
        "spikes. The agent must pre-position inventory and reroute flow to keep factories "
        "online while avoiding extreme Bullwhip amplification."
    ),
    episode_length=20,
    shock_config={
        "scheduled": [
            {"day": 3, "port": "PORT_SINGAPORE", "status": "blocked", "end_day": 9},
            {"day": 10, "port": "PORT_ROTTERDAM", "status": "congested", "end_day": 16},
        ],
        "random_demand_spikes": False,
    },
    success_threshold=0.55,
    tags=["dual-failure", "planning", "bullwhip"],
)

TASK_MULTI_PORT_WHIPLASH = Task(
    task_id="task_hard_multiport_whiplash",
    name="Multi-Port Whiplash",
    difficulty="hard",
    description=(
        "Alternating port congestion across Asia and Europe creates repeated throughput swings. "
        "The agent must sustain service levels while preventing sustained bullwhip amplification."
    ),
    episode_length=18,
    shock_config={
        "scheduled": [
            {"day": 1, "port": "PORT_SINGAPORE", "status": "congested", "end_day": 3},
            {"day": 4, "port": "PORT_ROTTERDAM", "status": "congested", "end_day": 6},
            {"day": 7, "port": "PORT_SHANGHAI", "status": "congested", "end_day": 9},
            {"day": 10, "port": "PORT_HAMBURG", "status": "congested", "end_day": 12},
            {"day": 13, "port": "PORT_BUSAN", "status": "congested", "end_day": 15},
        ],
        "random_demand_spikes": False,
    },
    success_threshold=0.55,
    tags=["multi-port", "throughput-swings", "bullwhip"],
)

TASK_DEMAND_SURGE_NO_PORTS = Task(
    task_id="task_hard_demand_surge_no_ports",
    name="Demand Surge",
    difficulty="hard",
    description=(
        "No port disruptions, but random demand spikes occur throughout the episode. "
        "The agent must maintain service levels and suppress bullwhip under stochastic demand."
    ),
    episode_length=20,
    shock_config={
        "scheduled": [],
        "random_demand_spikes": True,
    },
    success_threshold=0.55,
    tags=["demand-spikes", "stochastic", "bullwhip"],
)

TASK_HARD = Task(
    task_id="task_hard_ghost_protocol",
    name="Ghost Protocol",
    difficulty="hard",
    description=(
        "The full crisis scenario. Multiple simultaneous failures cascade over 30 days: "
        "PORT_SINGAPORE closes on day 3, PORT_ROTTERDAM congests on day 7, and PORT_HAMBURG "
        "blocks on day 15. Additionally, demand spikes randomly (+/-80% volatility). "
        "The Bullwhip Effect will amplify through the network. "
        "The agent must maintain >=0.9 network service level across all 3 factories for the "
        "entire 30-day episode. Any stock-out event (inventory = 0) causes a 20% score penalty "
        "per occurrence. The Bullwhip Index must stay below 2.0."
    ),
    episode_length=30,
    shock_config={
        "scheduled": [
            {"day": 3,  "port": "PORT_SINGAPORE", "status": "blocked",   "end_day": 10},
            {"day": 7,  "port": "PORT_ROTTERDAM",  "status": "congested", "end_day": 14},
            {"day": 15, "port": "PORT_HAMBURG",    "status": "blocked",   "end_day": 22},
        ],
        "random_demand_spikes": True,
    },
    success_threshold=0.5,
    tags=["multi-failure", "bullwhip", "demand-volatility", "ghost-protocol"],
)

TASK_GHOST_PROTOCOL_PLUS = Task(
    task_id="task_expert_ghost_protocol_plus",
    name="Ghost Protocol Plus",
    difficulty="hard",
    description=(
        "An extended Ghost Protocol variant with longer disruption windows and random demand spikes. "
        "Designed to stress-test long-horizon planning and stability under compounding uncertainty."
    ),
    episode_length=30,
    shock_config={
        "scheduled": [
            {"day": 2,  "port": "PORT_SINGAPORE", "status": "blocked",   "end_day": 12},
            {"day": 6,  "port": "PORT_ROTTERDAM",  "status": "congested", "end_day": 16},
            {"day": 14, "port": "PORT_HAMBURG",    "status": "blocked",   "end_day": 24},
        ],
        "random_demand_spikes": True,
    },
    success_threshold=0.45,
    tags=["expert", "multi-failure", "bullwhip", "demand-volatility"],
)

ALL_TASKS: List[Task] = [
    TASK_EASY,
    TASK_REROUTE_DRILL,
    TASK_MEDIUM,
    TASK_BUFFER_TUNING,
    TASK_EUROPE_SHOCK,
    TASK_CASCADE,
    TASK_MULTI_PORT_WHIPLASH,
    TASK_DEMAND_SURGE_NO_PORTS,
    TASK_HARD,
    TASK_GHOST_PROTOCOL_PLUS,
]

_FACTORY_IDS = ("FAC_TAIPEI", "FAC_BERLIN", "FAC_AUSTIN")


# ─── Graders ─────────────────────────────────────────────────────────────────


def grade_easy(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Easy grader — service level (90 %) + reroute bonus (10 %).

    Score = min(service_level, 1.0) * 0.9 + (0.1 if any reroute else 0.0).
    """
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)

    service_level = total_fulfilled / max(total_demand, 1e-9)
    service_score = min(service_level, 1.0) * 0.9

    rerouted = any(t.get("action_type") == "reroute_ship" for t in trajectory)
    reroute_bonus = 0.1 if rerouted else 0.0

    score = round(service_score + reroute_bonus, 4)
    success = score >= TASK_EASY.success_threshold

    failure_reasons: List[str] = []
    if not success:
        failure_reasons.append(
            f"Service level {service_level:.2%} below threshold"
        )
        if not rerouted:
            failure_reasons.append("Agent never issued a reroute action")

    return TaskResult(
        task_id=TASK_EASY.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "rerouted_ship": rerouted,
        },
        failure_reasons=failure_reasons,
    )


def grade_medium(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Medium grader — service level (70 %) + uptime (30 %).

    Score = min(service_level, 1.0) * 0.7 + (1 - idle_fraction) * 0.3.
    Idle days = days where any factory had current_stock <= 0.
    """
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)

    service_level = total_fulfilled / max(total_demand, 1e-9)

    idle_days = sum(1 for t in trajectory if t.get("any_factory_stockout", False))
    idle_fraction = idle_days / max(len(trajectory), 1)

    service_score = min(service_level, 1.0) * 0.7
    idle_score = (1.0 - idle_fraction) * 0.3

    score = round(service_score + idle_score, 4)
    success = score >= TASK_MEDIUM.success_threshold

    failure_reasons: List[str] = []
    if not success:
        failure_reasons.append(
            f"Service level {service_level:.2%} below threshold"
        )
        if idle_days > 1:
            failure_reasons.append(
                f"Factory idle for {idle_days} days -- exceeds tolerance"
            )

    return TaskResult(
        task_id=TASK_MEDIUM.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "idle_days": idle_days,
            "idle_fraction": round(idle_fraction, 4),
        },
        failure_reasons=failure_reasons,
    )


def grade_hard(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Hard grader (Ghost Protocol) — strict multi-metric evaluation.

    base           = min(service_level, 1.0) * 0.8
    stockout_pen   = min(stockout_events * 0.20, 0.60)
    bullwhip_pen   = min(violation_events * 0.05, 0.25)   (violation = BWI > 2.0)
    service_bonus  = 0.10 if service_level >= 0.9
    score          = clamp(base - stockout_pen - bullwhip_pen + service_bonus, 0.0, 1.0)
    """
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)

    service_level = total_fulfilled / max(total_demand, 1e-9)
    base_score = min(service_level, 1.0) * 0.8

    # Count stock-out *events* (rising edges), not stock-out *days*.
    # This matches the task spec language "per occurrence" and avoids
    # penalizing a single prolonged outage as dozens of independent events.
    stockout_events = 0
    prev_stockout = False
    for t in trajectory:
        cur_stockout = bool(t.get("any_factory_stockout", False))
        if cur_stockout and not prev_stockout:
            stockout_events += 1
        prev_stockout = cur_stockout
    stockout_penalty = min(stockout_events * 0.20, 0.60)

    bullwhip_violation_events = 0
    prev_violation = False
    for t in trajectory:
        cur_violation = float(t.get("bullwhip_index", 1.0)) > 2.0
        if cur_violation and not prev_violation:
            bullwhip_violation_events += 1
        prev_violation = cur_violation
    bullwhip_penalty = min(bullwhip_violation_events * 0.05, 0.25)

    service_bonus = 0.10 if service_level >= 0.9 else 0.0

    raw_score = base_score - stockout_penalty - bullwhip_penalty + service_bonus
    score = round(max(0.0, min(1.0, raw_score)), 4)
    success = score >= TASK_HARD.success_threshold

    failure_reasons: List[str] = []
    if not success:
        if stockout_events > 0:
            failure_reasons.append(
                f"CRITICAL: {stockout_events} stock-out event(s) "
                f"-- {stockout_penalty:.0%} penalty applied"
            )
        if service_level < 0.9:
            failure_reasons.append(
                f"Service level {service_level:.2%} below 90% target"
            )
        if bullwhip_violation_events > 0:
            failure_reasons.append(
                f"Bullwhip Index exceeded 2.0 in {bullwhip_violation_events} period(s)"
            )

    return TaskResult(
        task_id=TASK_HARD.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "stockout_events": stockout_events,
            "bullwhip_violation_events": bullwhip_violation_events,
            "service_bonus_awarded": service_bonus > 0,
        },
        failure_reasons=failure_reasons,
    )


def grade_cascade(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Cascade grader — service level (70 %) + uptime (20 %) + bullwhip control (10 %)."""
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)
    service_level = total_fulfilled / max(total_demand, 1e-9)

    idle_days = sum(1 for t in trajectory if t.get("any_factory_stockout", False))
    idle_fraction = idle_days / max(len(trajectory), 1)

    max_bullwhip = max((float(t.get("bullwhip_index", 1.0)) for t in trajectory), default=1.0)
    bullwhip_score = 0.10 if max_bullwhip <= 2.0 else max(0.0, 0.10 - min(0.10, (max_bullwhip - 2.0) * 0.02))

    score = round(min(service_level, 1.0) * 0.70 + (1.0 - idle_fraction) * 0.20 + bullwhip_score, 4)
    score = max(0.0, min(1.0, score))
    success = score >= TASK_CASCADE.success_threshold

    failure_reasons: List[str] = []
    if not success:
        failure_reasons.append(f"Service level {service_level:.2%} below target")
        if idle_days > 0:
            failure_reasons.append(f"Factory stockout on {idle_days} day(s)")
        if max_bullwhip > 2.0:
            failure_reasons.append(f"Bullwhip Index exceeded 2.0 (max={max_bullwhip:.2f})")

    return TaskResult(
        task_id=TASK_CASCADE.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "idle_days": idle_days,
            "idle_fraction": round(idle_fraction, 4),
            "max_bullwhip_index": round(max_bullwhip, 4),
        },
        failure_reasons=failure_reasons,
    )


def grade_reroute_drill(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Reroute drill — reward fast reroute + service + avoid excessive reroutes."""
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)
    service_level = total_fulfilled / max(total_demand, 1e-9)

    first_reroute_step: Optional[int] = None
    reroute_count = 0
    for i, t in enumerate(trajectory):
        if t.get("action_type") == "reroute_ship":
            reroute_count += 1
            if first_reroute_step is None:
                first_reroute_step = i

    fast_bonus = 0.15 if first_reroute_step is not None and first_reroute_step <= 1 else 0.0
    late_bonus = 0.05 if first_reroute_step is not None and first_reroute_step <= 3 else 0.0
    reroute_pen = min(max(0, reroute_count - 1) * 0.05, 0.15)

    score = (
        min(service_level, 1.0) * 0.75
        + fast_bonus
        + late_bonus
        - reroute_pen
    )
    score = round(max(0.0, min(1.0, score)), 4)
    success = score >= TASK_REROUTE_DRILL.success_threshold

    failure_reasons: List[str] = []
    if not success:
        if first_reroute_step is None:
            failure_reasons.append("Agent never issued a reroute action")
        failure_reasons.append(f"Service level {service_level:.2%} below target")

    return TaskResult(
        task_id=TASK_REROUTE_DRILL.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "first_reroute_step": first_reroute_step,
            "reroute_count": reroute_count,
        },
        failure_reasons=failure_reasons,
    )


def grade_buffer_tuning(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Buffer tuning — service + stability + bullwhip control + action restraint."""
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)
    service_level = total_fulfilled / max(total_demand, 1e-9)

    stockout_days = sum(1 for t in trajectory if t.get("any_factory_stockout", False))
    idle_fraction = stockout_days / max(len(trajectory), 1)

    bullwhip_max = max((float(t.get("bullwhip_index", 1.0)) for t in trajectory), default=1.0)
    bullwhip_score = 0.15 if bullwhip_max <= 2.0 else max(0.0, 0.15 - min(0.15, (bullwhip_max - 2.0) * 0.02))

    adjust_count = sum(1 for t in trajectory if t.get("action_type") == "adjust_buffer")
    expedite_count = sum(1 for t in trajectory if t.get("action_type") == "expedite_order")
    reroute_count = sum(1 for t in trajectory if t.get("action_type") == "reroute_ship")
    action_penalty = min(adjust_count * 0.01 + expedite_count * 0.03 + reroute_count * 0.02, 0.20)

    score = (
        min(service_level, 1.0) * 0.70
        + (1.0 - idle_fraction) * 0.15
        + bullwhip_score
        - action_penalty
    )
    score = round(max(0.0, min(1.0, score)), 4)
    success = score >= TASK_BUFFER_TUNING.success_threshold

    failure_reasons: List[str] = []
    if not success:
        failure_reasons.append(f"Service level {service_level:.2%} below target")
        if stockout_days > 0:
            failure_reasons.append(f"Factory stockout on {stockout_days} day(s)")
        if bullwhip_max > 2.0:
            failure_reasons.append(f"Bullwhip Index too high (max={bullwhip_max:.2f})")

    return TaskResult(
        task_id=TASK_BUFFER_TUNING.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "stockout_days": stockout_days,
            "idle_fraction": round(idle_fraction, 4),
            "max_bullwhip_index": round(bullwhip_max, 4),
            "adjust_count": adjust_count,
            "expedite_count": expedite_count,
            "reroute_count": reroute_count,
        },
        failure_reasons=failure_reasons,
    )


def grade_europe_shock(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Europe shock — service + uptime, with extra weight on stability."""
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)
    service_level = total_fulfilled / max(total_demand, 1e-9)

    idle_days = sum(1 for t in trajectory if t.get("any_factory_stockout", False))
    idle_fraction = idle_days / max(len(trajectory), 1)

    bullwhip_max = max((float(t.get("bullwhip_index", 1.0)) for t in trajectory), default=1.0)
    bullwhip_bonus = 0.10 if bullwhip_max <= 2.0 else 0.0

    score = round(
        max(0.0, min(1.0, min(service_level, 1.0) * 0.70 + (1.0 - idle_fraction) * 0.20 + bullwhip_bonus)),
        4,
    )
    success = score >= TASK_EUROPE_SHOCK.success_threshold

    failure_reasons: List[str] = []
    if not success:
        failure_reasons.append(f"Service level {service_level:.2%} below target")
        if idle_days > 0:
            failure_reasons.append(f"Factory stockout on {idle_days} day(s)")

    return TaskResult(
        task_id=TASK_EUROPE_SHOCK.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "idle_days": idle_days,
            "idle_fraction": round(idle_fraction, 4),
            "max_bullwhip_index": round(bullwhip_max, 4),
        },
        failure_reasons=failure_reasons,
    )


def grade_multiport_whiplash(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Multiport whiplash — service + bullwhip + stockout-event control."""
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)
    service_level = total_fulfilled / max(total_demand, 1e-9)

    stockout_events = 0
    prev_stockout = False
    for t in trajectory:
        cur = bool(t.get("any_factory_stockout", False))
        if cur and not prev_stockout:
            stockout_events += 1
        prev_stockout = cur

    bullwhip_events = 0
    prev_violation = False
    for t in trajectory:
        cur_v = float(t.get("bullwhip_index", 1.0)) > 2.5
        if cur_v and not prev_violation:
            bullwhip_events += 1
        prev_violation = cur_v

    base = min(service_level, 1.0) * 0.75
    stock_pen = min(stockout_events * 0.10, 0.30)
    bw_pen = min(bullwhip_events * 0.05, 0.25)
    score = round(max(0.0, min(1.0, base - stock_pen - bw_pen)), 4)
    success = score >= TASK_MULTI_PORT_WHIPLASH.success_threshold

    failure_reasons: List[str] = []
    if not success:
        if stockout_events:
            failure_reasons.append(f"{stockout_events} stockout event(s)")
        if bullwhip_events:
            failure_reasons.append(f"{bullwhip_events} bullwhip violation period(s)")
        failure_reasons.append(f"Service level {service_level:.2%}")

    return TaskResult(
        task_id=TASK_MULTI_PORT_WHIPLASH.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "stockout_events": stockout_events,
            "bullwhip_violation_events": bullwhip_events,
        },
        failure_reasons=failure_reasons,
    )


def grade_demand_surge(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Demand surge — service robustness under spikes + bullwhip control."""
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)
    service_level = total_fulfilled / max(total_demand, 1e-9)

    stockout_events = 0
    prev_stockout = False
    for t in trajectory:
        cur = bool(t.get("any_factory_stockout", False))
        if cur and not prev_stockout:
            stockout_events += 1
        prev_stockout = cur

    bullwhip_max = max((float(t.get("bullwhip_index", 1.0)) for t in trajectory), default=1.0)
    bw_pen = 0.0 if bullwhip_max <= 2.5 else min((bullwhip_max - 2.5) * 0.03, 0.30)

    score = min(service_level, 1.0) * 0.80 - min(stockout_events * 0.15, 0.60) - bw_pen
    score = round(max(0.0, min(1.0, score)), 4)
    success = score >= TASK_DEMAND_SURGE_NO_PORTS.success_threshold

    failure_reasons: List[str] = []
    if not success:
        if stockout_events:
            failure_reasons.append(f"{stockout_events} stockout event(s)")
        if bullwhip_max > 2.5:
            failure_reasons.append(f"Bullwhip max {bullwhip_max:.2f}")
        failure_reasons.append(f"Service level {service_level:.2%}")

    return TaskResult(
        task_id=TASK_DEMAND_SURGE_NO_PORTS.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "stockout_events": stockout_events,
            "max_bullwhip_index": round(bullwhip_max, 4),
        },
        failure_reasons=failure_reasons,
    )


def grade_ghost_protocol_plus(env: SupplyChainEnv, trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Ghost Protocol Plus — harsher stability demands than TASK_HARD."""
    total_demand = sum(t["demand_total"] for t in trajectory)
    total_fulfilled = sum(t["demand_fulfilled"] for t in trajectory)
    service_level = total_fulfilled / max(total_demand, 1e-9)

    stockout_events = 0
    prev_stockout = False
    for t in trajectory:
        cur = bool(t.get("any_factory_stockout", False))
        if cur and not prev_stockout:
            stockout_events += 1
        prev_stockout = cur

    bullwhip_events = 0
    prev_violation = False
    for t in trajectory:
        cur_v = float(t.get("bullwhip_index", 1.0)) > 2.0
        if cur_v and not prev_violation:
            bullwhip_events += 1
        prev_violation = cur_v

    base = min(service_level, 1.0) * 0.85
    stock_pen = min(stockout_events * 0.25, 0.75)
    bw_pen = min(bullwhip_events * 0.05, 0.30)
    score = round(max(0.0, min(1.0, base - stock_pen - bw_pen)), 4)
    success = score >= TASK_GHOST_PROTOCOL_PLUS.success_threshold

    failure_reasons: List[str] = []
    if not success:
        failure_reasons.append(f"Service level {service_level:.2%}")
        if stockout_events:
            failure_reasons.append(f"{stockout_events} stockout event(s)")
        if bullwhip_events:
            failure_reasons.append(f"{bullwhip_events} bullwhip violation period(s)")

    return TaskResult(
        task_id=TASK_GHOST_PROTOCOL_PLUS.task_id,
        score=score,
        success=success,
        metrics={
            "service_level": round(service_level, 4),
            "stockout_events": stockout_events,
            "bullwhip_violation_events": bullwhip_events,
        },
        failure_reasons=failure_reasons,
    )


GRADERS: Dict[str, Callable[..., TaskResult]] = {
    TASK_EASY.task_id: grade_easy,
    TASK_REROUTE_DRILL.task_id: grade_reroute_drill,
    TASK_MEDIUM.task_id: grade_medium,
    TASK_BUFFER_TUNING.task_id: grade_buffer_tuning,
    TASK_EUROPE_SHOCK.task_id: grade_europe_shock,
    TASK_CASCADE.task_id: grade_cascade,
    TASK_MULTI_PORT_WHIPLASH.task_id: grade_multiport_whiplash,
    TASK_DEMAND_SURGE_NO_PORTS.task_id: grade_demand_surge,
    TASK_HARD.task_id: grade_hard,
    TASK_GHOST_PROTOCOL_PLUS.task_id: grade_ghost_protocol_plus,
}


# ─── Task Runner ─────────────────────────────────────────────────────────────


def run_task(
    task: Task,
    agent_fn: Callable[..., Action],
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[TaskResult, List[Dict[str, Any]]]:
    """Run a full episode and grade it.

    Parameters
    ----------
    task : Task
        The task descriptor to evaluate.
    agent_fn : Callable[[Observation], Action]
        Agent policy — receives an Observation, returns an Action.
    seed : int
        RNG seed for deterministic replay.
    verbose : bool
        If True, prints a one-line summary per step.

    Returns
    -------
    result : TaskResult
        Graded outcome including score, metrics, and failure reasons.
    trajectory : list[dict]
        Per-step info dicts suitable for analysis and grading.
    """
    env = task.build_env(seed=seed)
    obs = env.reset()
    trajectory: List[Dict[str, Any]] = []
    done = False
    prev_step: Optional[Dict[str, Any]] = None

    while not done:
        try:
            action = agent_fn(obs, prev_step, task)
        except TypeError:
            action = agent_fn(obs)
        next_obs, reward, done, info = env.step(action)

        any_stockout = any(
            next_obs.inventory_levels[fid].current_stock <= 0
            for fid in _FACTORY_IDS
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

        if verbose:
            print(
                f"Day {info['day']:2d} | "
                f"Reward: {reward.total:+.1f} | "
                f"SL: {next_obs.network_service_level:.2%} | "
                f"BWE: {next_obs.bullwhip_index:.2f} | "
                f"Stockout: {any_stockout}"
            )

        obs = next_obs

    grader = GRADERS[task.task_id]
    result = grader(env, trajectory)
    return result, trajectory
