"""
Supply Chain Ghost Protocol — Foundation Models
================================================
Pydantic v2 data-model layer for the OpenEnv-compliant RL environment
simulating global semiconductor supply chain crisis management.

This file is the **single source of truth** for every data structure in the
project.  It imports *only* from the stdlib and ``pydantic``.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─── Enums ────────────────────────────────────────────────────────────────────


class PortStatus(str, Enum):
    """Operational status of a maritime port node."""

    OPEN = "open"
    BLOCKED = "blocked"
    CONGESTED = "congested"


class ShipStatus(str, Enum):
    """Current logistics state of a cargo vessel."""

    IN_TRANSIT = "in_transit"
    DOCKED = "docked"
    REROUTING = "rerouting"
    DELAYED = "delayed"


class ActionType(str, Enum):
    """Discrete action primitives available to the RL agent."""

    REROUTE_SHIP = "reroute_ship"
    EXPEDITE_ORDER = "expedite_order"
    ADJUST_BUFFER = "adjust_buffer"
    NOOP = "noop"


# ─── Sub-Models ───────────────────────────────────────────────────────────────


class Ship(BaseModel):
    """A cargo vessel transporting semiconductor components between ports."""

    model_config = ConfigDict(frozen=False)

    ship_id: str = Field(
        ...,
        description="Unique identifier for the vessel.",
    )
    cargo_volume: float = Field(
        ...,
        ge=0,
        description="Volume of cargo on board (units).",
    )
    origin_port: str = Field(
        ...,
        description="Port ID where the shipment originated.",
    )
    destination_port: str = Field(
        ...,
        description="Port ID the vessel is heading toward.",
    )
    eta_days: float = Field(
        ...,
        ge=0,
        description="Estimated time of arrival in days.",
    )
    status: ShipStatus = Field(
        default=ShipStatus.IN_TRANSIT,
        description="Current logistics state of the vessel.",
    )
    reroute_cost: float = Field(
        default=0.0,
        ge=0,
        description="Incremental cost incurred if this ship is rerouted.",
    )


class NodeInventory(BaseModel):
    """Inventory snapshot for a single supply-chain node (warehouse / fab)."""

    model_config = ConfigDict(frozen=False)

    node_id: str = Field(
        ...,
        description="Unique identifier for the inventory node.",
    )
    current_stock: float = Field(
        ...,
        ge=0,
        description="Units currently held in stock.",
    )
    capacity: float = Field(
        ...,
        gt=0,
        description="Maximum storage capacity (units).  Must be > 0.",
    )
    safety_stock: float = Field(
        ...,
        ge=0,
        description="Minimum desired stock level before triggering alerts.",
    )
    holding_cost_per_unit_day: float = Field(
        default=0.5,
        ge=0,
        description="Cost per unit per day of holding inventory.",
    )

    # -- computed properties --------------------------------------------------

    @property
    def utilization(self) -> float:
        """Fraction of capacity currently in use (0.0 – 1.0+)."""
        return self.current_stock / self.capacity

    @property
    def is_below_safety(self) -> bool:
        """``True`` when current stock has dropped below the safety threshold."""
        return self.current_stock < self.safety_stock


class FactoryDemand(BaseModel):
    """Demand profile for a downstream semiconductor fab / factory."""

    model_config = ConfigDict(frozen=False)

    factory_id: str = Field(
        ...,
        description="Unique identifier for the factory.",
    )
    daily_burn_rate: float = Field(
        ...,
        ge=0,
        description="Units consumed per day under normal operations.",
    )
    demand_volatility: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Normalised volatility coefficient (0 = deterministic, 1 = max).",
    )
    stockout_cost_per_unit: float = Field(
        default=500.0,
        ge=0,
        description="Penalty cost per unit of unmet demand.",
    )
    days_until_critical: Optional[float] = Field(
        default=None,
        description="Days until the factory hits a critical stockout (None = safe).",
    )


# ─── Core OpenEnv Models ─────────────────────────────────────────────────────


class Observation(BaseModel):
    """
    Full environment observation returned to the agent each step.

    Encodes the current state of the global semiconductor supply chain:
    inventories, ships in transit, port conditions, demand profiles, and
    aggregate health metrics.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "day": 5,
                    "episode_remaining_days": 25,
                    "inventory_levels": {
                        "node_tw01": {
                            "node_id": "node_tw01",
                            "current_stock": 800.0,
                            "capacity": 2000.0,
                            "safety_stock": 400.0,
                            "holding_cost_per_unit_day": 0.5,
                        }
                    },
                    "transit_queue": [
                        {
                            "ship_id": "ship_alpha",
                            "cargo_volume": 500.0,
                            "origin_port": "port_shanghai",
                            "destination_port": "port_la",
                            "eta_days": 12.0,
                            "status": "in_transit",
                            "reroute_cost": 0.0,
                        }
                    ],
                    "port_status": {
                        "port_shanghai": "open",
                        "port_la": "congested",
                    },
                    "daily_burn_rate": {
                        "fab_intel_az": {
                            "factory_id": "fab_intel_az",
                            "daily_burn_rate": 120.0,
                            "demand_volatility": 0.15,
                            "stockout_cost_per_unit": 500.0,
                            "days_until_critical": 10.0,
                        }
                    },
                    "bullwhip_index": 1.35,
                    "network_service_level": 0.92,
                    "total_holding_cost_today": 400.0,
                    "active_shocks": ["typhoon_pacific"],
                }
            ]
        }
    )

    day: int = Field(
        ...,
        ge=0,
        description="Current simulation day (0-indexed).",
    )
    episode_remaining_days: int = Field(
        ...,
        ge=0,
        description="Number of days remaining in this episode.",
    )
    inventory_levels: Dict[str, NodeInventory] = Field(
        ...,
        description="Mapping of node_id → NodeInventory for every tracked node.",
    )
    transit_queue: List[Ship] = Field(
        default_factory=list,
        description="Ships currently in transit across the network.",
    )
    port_status: Dict[str, PortStatus] = Field(
        ...,
        description="Mapping of port_id → current PortStatus.",
    )
    daily_burn_rate: Dict[str, FactoryDemand] = Field(
        ...,
        description="Mapping of factory_id → FactoryDemand demand profile.",
    )
    bullwhip_index: float = Field(
        default=1.0,
        ge=0,
        description="Bullwhip-effect multiplier (1.0 = no amplification).",
    )
    network_service_level: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="Fraction of total demand met on time (0.0 – 1.0).",
    )
    total_holding_cost_today: float = Field(
        default=0.0,
        ge=0,
        description="Aggregate holding cost accrued across all nodes today.",
    )
    active_shocks: List[str] = Field(
        default_factory=list,
        description="IDs of supply-chain disruption events currently active.",
    )


class Action(BaseModel):
    """
    An agent action submitted to the environment for execution.

    The ``action_type`` determines which optional fields are required:
    - ``REROUTE_SHIP`` → ``ship_id`` and ``new_port`` are mandatory.
    - ``EXPEDITE_ORDER`` → ``source_node``, ``destination_node``, and
      ``expedite_volume`` are expected.
    - ``ADJUST_BUFFER`` → ``target_node`` and ``target_inventory`` are
      expected.
    - ``NOOP`` → no additional fields required.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "action_type": "reroute_ship",
                    "ship_id": "ship_alpha",
                    "new_port": "port_busan",
                },
                {
                    "action_type": "noop",
                },
            ]
        }
    )

    action_type: ActionType = Field(
        ...,
        description="The discrete action primitive to execute.",
    )
    ship_id: Optional[str] = Field(
        default=None,
        description="Target ship ID (required for REROUTE_SHIP).",
    )
    new_port: Optional[str] = Field(
        default=None,
        description="Destination port to reroute to (required for REROUTE_SHIP).",
    )
    source_node: Optional[str] = Field(
        default=None,
        description="Source inventory node for EXPEDITE_ORDER.",
    )
    expedite_volume: Optional[float] = Field(
        default=None,
        ge=0,
        description="Volume of units to expedite (EXPEDITE_ORDER).",
    )
    destination_node: Optional[str] = Field(
        default=None,
        description="Destination inventory node for EXPEDITE_ORDER.",
    )
    target_node: Optional[str] = Field(
        default=None,
        description="Target inventory node for ADJUST_BUFFER.",
    )
    target_inventory: Optional[float] = Field(
        default=None,
        ge=0,
        description="Desired inventory level for ADJUST_BUFFER.",
    )

    @model_validator(mode="after")
    def _require_reroute_fields(self) -> "Action":
        """``ship_id`` and ``new_port`` must be provided for REROUTE_SHIP."""
        if self.action_type == ActionType.REROUTE_SHIP:
            if self.ship_id is None:
                raise ValueError("ship_id is required when action_type is 'reroute_ship'")
            if self.new_port is None:
                raise ValueError("new_port is required when action_type is 'reroute_ship'")
        return self


class RewardComponents(BaseModel):
    """Itemised breakdown of the scalar reward signal."""

    model_config = ConfigDict(frozen=True)

    service_level_bonus: float = Field(
        ...,
        description="Bonus awarded for maintaining high network service level.",
    )
    stockout_penalty: float = Field(
        ...,
        description="Penalty incurred for unmet downstream demand (stockouts).",
    )
    holding_cost_penalty: float = Field(
        ...,
        description="Penalty proportional to aggregate inventory holding costs.",
    )
    reroute_cost_penalty: float = Field(
        ...,
        description="Penalty for costs incurred by rerouting vessels.",
    )
    expedite_cost_penalty: float = Field(
        ...,
        description="Penalty for costs incurred by expediting orders.",
    )
    bullwhip_penalty: float = Field(
        ...,
        description="Penalty proportional to the bullwhip-effect index.",
    )
    stability_bonus: float = Field(
        ...,
        description="Bonus rewarding smooth, stable ordering patterns.",
    )


class Reward(BaseModel):
    """
    Composite reward object returned by the environment after each step.

    Contains the scalar ``total`` reward, a full ``components`` breakdown, an
    episode-termination flag, and an arbitrary ``info`` dict for diagnostics.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "total": 42.5,
                    "components": {
                        "service_level_bonus": 80.0,
                        "stockout_penalty": -20.0,
                        "holding_cost_penalty": -5.0,
                        "reroute_cost_penalty": -2.5,
                        "expedite_cost_penalty": 0.0,
                        "bullwhip_penalty": -15.0,
                        "stability_bonus": 5.0,
                    },
                    "done": False,
                    "info": {"step": 5},
                }
            ]
        }
    )

    total: float = Field(
        ...,
        description="Scalar reward value for the current step.",
    )
    components: RewardComponents = Field(
        ...,
        description="Itemised breakdown of the total reward.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has terminated.",
    )
    info: Dict = Field(
        default_factory=dict,
        description="Auxiliary diagnostic information (free-form).",
    )
