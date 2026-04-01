"""
Supply Chain Ghost Protocol — Environment
==========================================
OpenEnv-compliant RL environment for semiconductor supply chain crisis
management.  Implements the full step / reset / state API with an embedded
Bullwhip Effect model, shock system, and multi-component reward function.

Network topology:  5 ports → 2 warehouses → 3 factories.
Each upstream node amplifies demand volatility (Bullwhip Effect).
The agent must maintain service levels while minimising cost and
order-variance amplification across a 30-day episode.

Dependencies: stdlib only (random, math, copy) + models.py.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action,
    ActionType,
    FactoryDemand,
    NodeInventory,
    Observation,
    PortStatus,
    Reward,
    RewardComponents,
    Ship,
    ShipStatus,
)


# ─── Network Topology Constants ──────────────────────────────────────────────

PORTS: Dict[str, Dict[str, Any]] = {
    "PORT_SINGAPORE": {"throughput_capacity": 5000},
    "PORT_ROTTERDAM":  {"throughput_capacity": 4000},
    "PORT_SHANGHAI":   {"throughput_capacity": 6000},
    "PORT_BUSAN":      {"throughput_capacity": 3500},
    "PORT_HAMBURG":    {"throughput_capacity": 3000},
}

WAREHOUSES: Dict[str, Dict[str, Any]] = {
    "WH_ASIA_HUB":   {"capacity": 20_000, "safety_stock": 4000},
    "WH_EUROPE_HUB": {"capacity": 15_000, "safety_stock": 3000},
}

FACTORIES: Dict[str, Dict[str, Any]] = {
    "FAC_TAIPEI": {"daily_burn": 600, "capacity": 12_000, "safety_stock": 1200},
    "FAC_BERLIN": {"daily_burn": 400, "capacity":  8_000, "safety_stock":  800},
    "FAC_AUSTIN": {"daily_burn": 300, "capacity":  6_000, "safety_stock":  600},
}

TRANSIT_TIMES: Dict[Tuple[str, str], float] = {
    ("PORT_SINGAPORE", "WH_ASIA_HUB"):   1.0,
    ("PORT_SHANGHAI",  "WH_ASIA_HUB"):   2.0,
    ("PORT_BUSAN",     "WH_ASIA_HUB"):   1.5,
    ("PORT_ROTTERDAM", "WH_EUROPE_HUB"): 1.0,
    ("PORT_HAMBURG",   "WH_EUROPE_HUB"): 0.5,
    ("WH_ASIA_HUB",   "FAC_TAIPEI"):     1.0,
    ("WH_EUROPE_HUB", "FAC_BERLIN"):     1.5,
    ("WH_ASIA_HUB",   "FAC_AUSTIN"):     3.0,
}


# ─── Cost Constants ──────────────────────────────────────────────────────────

REROUTE_COST_PER_SHIP      = 15_000
EXPEDITE_COST_MULTIPLIER   = 2.0
BASE_SHIPPING_COST_PER_UNIT = 5.0
STOCKOUT_COST_PER_UNIT     = 500.0
HOLDING_COST_PER_UNIT_DAY  = 0.5
REWARD_SCALE               = 0.01
SERVICE_LEVEL_BONUS        = 200.0
STABILITY_BONUS            = 50.0


# ─── SupplyChainEnv ──────────────────────────────────────────────────────────


class SupplyChainEnv:
    """OpenEnv-compliant semiconductor supply-chain environment.

    Simulates global logistics across a network of 5 maritime ports,
    2 regional warehouses, and 3 semiconductor factories over discrete
    daily time-steps.

    **Bullwhip Effect**
        Warehouse ordering amplifies downstream demand variance by factor
        k = 1 + (lead_time / review_period).  With default parameters
        (lead_time=2, review_period=1), k=3 — warehouses order ~3x the
        actual demand signal, causing upstream variance explosion.

    **Shock system**
        Supports deterministic scheduled port disruptions and stochastic
        demand spikes, configurable via ``shock_config``.

    **Reward**
        Dense, multi-component signal balancing service-level maintenance,
        cost efficiency, order-variance dampening, and factory stability.

    Parameters
    ----------
    episode_length : int
        Number of simulation days per episode.
    seed : int
        RNG seed for full trajectory reproducibility.
    shock_config : dict | None
        Optional disruption configuration.  Supports keys:
        ``"scheduled"`` — list of ``{day, port, status, end_day}`` dicts,
        ``"random_demand_spikes"`` — bool enabling stochastic burn-rate
        spikes (15 % daily probability).
    """

    def __init__(
        self,
        episode_length: int = 30,
        seed: int = 42,
        shock_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.episode_length = episode_length
        self.seed = seed
        self.shock_config = shock_config or {}
        self._rng = random.Random(seed)

        # --- mutable state (fully initialised in reset) ---
        self._day: int = 0
        self._inventory: Dict[str, NodeInventory] = {}
        self._transit_queue: List[Ship] = []
        self._port_status: Dict[str, PortStatus] = {}
        self._factory_demands: Dict[str, FactoryDemand] = {}
        self._ship_counter: int = 0

        # Bullwhip rolling history
        self._demand_history: List[float] = []
        self._order_history: List[float] = []

        # Service-level tracking
        self._cumulative_service_hits: int = 0
        self._cumulative_service_checks: int = 0

        # Financial accumulators
        self._episode_reroute_cost: float = 0.0
        self._episode_expedite_cost: float = 0.0

        # Random demand spikes are one-day shocks; store originals for reset next day.
        self._pending_spike_reset: Dict[str, float] = {}

    # ─── OpenEnv API ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment to its initial state and return the first
        observation.

        Determinism guarantee: calling ``reset()`` always produces the same
        starting state for a given ``seed``, regardless of how many episodes
        have been run previously.

        The method performs the following in order:

        1. Re-seed the internal RNG.
        2. Zero all counters and history buffers.
        3. Initialise port statuses (all OPEN).
        4. Initialise inventory for every node (ports at 30 % throughput
           capacity, warehouses at 50 % capacity, factories at 60 %
           capacity).
        5. Initialise factory demand profiles.
        6. Spawn three ships already in transit.
        7. Apply any Day-0 shocks from ``shock_config``.

        Returns
        -------
        Observation
            Full state snapshot at day 0.
        """
        self._rng = random.Random(self.seed)
        self._day = 0
        self._ship_counter = 0
        self._demand_history = []
        self._order_history = []
        self._cumulative_service_hits = 0
        self._cumulative_service_checks = 0
        self._episode_reroute_cost = 0.0
        self._episode_expedite_cost = 0.0
        self._pending_spike_reset = {}

        # Port statuses
        self._port_status = {pid: PortStatus.OPEN for pid in PORTS}

        # Node inventories
        self._inventory = {}
        for pid, pcfg in PORTS.items():
            cap = pcfg["throughput_capacity"]
            self._inventory[pid] = NodeInventory(
                node_id=pid,
                current_stock=cap * 0.3,
                capacity=float(cap),
                safety_stock=cap * 0.1,
            )
        for wid, wcfg in WAREHOUSES.items():
            self._inventory[wid] = NodeInventory(
                node_id=wid,
                current_stock=wcfg["capacity"] * 0.5,
                capacity=float(wcfg["capacity"]),
                safety_stock=float(wcfg["safety_stock"]),
            )
        for fid, fcfg in FACTORIES.items():
            self._inventory[fid] = NodeInventory(
                node_id=fid,
                current_stock=fcfg["capacity"] * 0.6,
                capacity=float(fcfg["capacity"]),
                safety_stock=float(fcfg["safety_stock"]),
            )

        # Factory demand profiles
        self._factory_demands = {}
        for fid, fcfg in FACTORIES.items():
            self._factory_demands[fid] = FactoryDemand(
                factory_id=fid,
                daily_burn_rate=float(fcfg["daily_burn"]),
                demand_volatility=0.1,
            )

        # Seed initial ships
        self._transit_queue = []
        self._spawn_initial_ships()

        # Day-0 shocks
        self._apply_shocks()

        return self.state()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Advance the simulation by one day.

        Execution order is fixed and must not be reordered:

        1. Execute the agent's action (reroute / expedite / adjust / noop).
        2. Advance all ships by 1 day; deliver arrived cargo.
        3. Consume factory demand — Bullwhip index is computed here.
        4. Apply any scheduled or random shocks for the new day.
        5. Run the automatic (s, S) warehouse replenishment policy.
        6. Compute the multi-component reward signal.
        7. Increment the day counter; determine episode termination.

        Parameters
        ----------
        action : Action
            The agent's chosen action for this time-step.

        Returns
        -------
        observation : Observation
            Post-step state snapshot.
        reward : Reward
            Scalar total plus itemised component breakdown.
        done : bool
            ``True`` when the episode has ended.
        info : dict
            Diagnostic metadata for the step (day, demand metrics,
            bullwhip index, arrived volume, reorder cost).

        Raises
        ------
        RuntimeError
            If the episode has already ended (day >= episode_length).
        """
        if self._day >= self.episode_length:
            raise RuntimeError("Episode is done. Call reset().")

        # 1. Execute agent action
        action_cost = self._execute_action(action)

        # 2. Advance ships
        arrived_volume = self._advance_ships()

        # 3. Factory demand consumption (Bullwhip core)
        demand_fulfilled, demand_total, bullwhip_index = self._consume_factory_demand()

        # 4. Shocks
        self._apply_shocks()

        # 5. Auto-replenishment
        reorder_cost = self._auto_replenish()

        # 6. Reward
        reward = self._compute_reward(
            demand_fulfilled, demand_total, bullwhip_index, action, action_cost,
        )

        # 7. Day counter + termination
        self._day += 1
        done = self._day >= self.episode_length

        info: Dict[str, Any] = {
            "day": self._day,
            "demand_fulfilled": demand_fulfilled,
            "demand_total": demand_total,
            "bullwhip_index": bullwhip_index,
            "arrived_volume": arrived_volume,
            "reorder_cost": reorder_cost,
        }

        return self.state(), reward, done, info

    def state(self) -> Observation:
        """Return a full snapshot of the current environment state.

        This method is side-effect free with respect to simulation
        progression — it does not advance the day counter.  It **does**
        update derived fields (``days_until_critical`` for each factory)
        so the observation is always consistent.

        Used internally after ``reset()`` and ``step()``, and also
        exposed to the HTTP server for ``GET /state/{session_id}``.

        Returns
        -------
        Observation
            Complete state including inventories, transit queue, port
            statuses, demand profiles, bullwhip index, service level,
            holding cost, and active shocks.  Mutable fields are
            deep-copied to prevent aliasing.
        """
        for fid, fd in self._factory_demands.items():
            inv = self._inventory[fid]
            if fd.daily_burn_rate > 0:
                fd.days_until_critical = inv.current_stock / fd.daily_burn_rate
            else:
                fd.days_until_critical = float("inf")

        bullwhip = self._compute_bullwhip_index()

        service_level = (
            self._cumulative_service_hits / self._cumulative_service_checks
            if self._cumulative_service_checks > 0
            else 1.0
        )

        holding_cost = sum(
            inv.current_stock * HOLDING_COST_PER_UNIT_DAY
            for inv in self._inventory.values()
        )

        active_shocks = [
            f"{pid} is {status.value}"
            for pid, status in self._port_status.items()
            if status != PortStatus.OPEN
        ]

        return Observation(
            day=self._day,
            episode_remaining_days=max(0, self.episode_length - self._day),
            inventory_levels=deepcopy(self._inventory),
            transit_queue=deepcopy(self._transit_queue),
            port_status=deepcopy(self._port_status),
            daily_burn_rate=deepcopy(self._factory_demands),
            bullwhip_index=bullwhip,
            network_service_level=service_level,
            total_holding_cost_today=holding_cost,
            active_shocks=active_shocks,
        )

    # ─── Bullwhip Effect ─────────────────────────────────────────────────

    def _consume_factory_demand(self) -> Tuple[float, float, float]:
        """Simulate one day of factory consumption and compute the Bullwhip
        index from the resulting order signals.

        For each factory:
        1. Realised demand = burn_rate + N(0, volatility * burn_rate).
        2. Fulfilled volume = min(factory_stock, realised_demand).
        3. Warehouse order signal = realised_demand * k + safety_stock_gap,
           where k = 1 + (lead_time / review_period) = 3.0.

        Returns
        -------
        total_fulfilled : float
        total_demand : float
        bullwhip_index : float
        """
        total_demand = 0.0
        total_fulfilled = 0.0
        total_orders_placed = 0.0

        for fid, fd in self._factory_demands.items():
            noise = self._rng.gauss(0, fd.demand_volatility * fd.daily_burn_rate)
            realized_demand = max(0.0, fd.daily_burn_rate + noise)
            total_demand += realized_demand

            inv = self._inventory[fid]
            fulfilled = min(inv.current_stock, realized_demand)
            inv.current_stock -= fulfilled
            total_fulfilled += fulfilled

            k = 1.0 + (2.0 / 1.0)  # lead_time=2, review_period=1
            warehouse_order = realized_demand * k + max(0.0, inv.safety_stock - inv.current_stock)
            total_orders_placed += warehouse_order

        self._demand_history.append(total_demand)
        self._order_history.append(total_orders_placed)

        self._cumulative_service_checks += 1
        if total_demand > 0 and (total_fulfilled / total_demand) >= 0.95:
            self._cumulative_service_hits += 1

        return total_fulfilled, total_demand, self._compute_bullwhip_index()

    def _compute_bullwhip_index(self) -> float:
        """Rolling 7-day Bullwhip Index = Var(orders) / Var(demand).

        Uses population variance.  Returns 1.0 when fewer than 2 data
        points are available or when demand variance is near zero.
        """
        window = 7
        d = self._demand_history[-window:]
        o = self._order_history[-window:]

        if len(d) < 2:
            return 1.0

        def _var(xs: List[float]) -> float:
            mu = sum(xs) / len(xs)
            return sum((x - mu) ** 2 for x in xs) / len(xs)

        var_demand = _var(d)
        var_orders = _var(o)

        if var_demand < 1e-9:
            return 1.0

        return round(var_orders / var_demand, 4)

    # ─── Action Execution ────────────────────────────────────────────────

    def _execute_action(self, action: Action) -> float:
        """Dispatch the agent's action to the appropriate handler.

        Returns the direct monetary cost incurred by the action.
        """
        if action.action_type == ActionType.REROUTE_SHIP:
            return self._reroute_ship(action.ship_id, action.new_port)
        if action.action_type == ActionType.EXPEDITE_ORDER:
            return self._expedite_order(
                action.source_node, action.destination_node, action.expedite_volume,
            )
        if action.action_type == ActionType.ADJUST_BUFFER:
            return self._adjust_buffer(action.target_node, action.target_inventory)
        return 0.0  # NOOP

    def _reroute_ship(
        self, ship_id: Optional[str], new_port: Optional[str],
    ) -> float:
        """Reroute *ship_id* to *new_port*.

        Applies a flat reroute fee plus 1–2 days of random additional
        transit time.  Returns 0.0 if the ship or port is not found.
        """
        if ship_id is None or new_port is None:
            return 0.0

        for ship in self._transit_queue:
            if ship.ship_id == ship_id:
                if new_port not in self._port_status:
                    return 0.0
                ship.destination_port = new_port
                ship.status = ShipStatus.REROUTING
                ship.eta_days += self._rng.uniform(1.0, 2.0)
                ship.reroute_cost += REROUTE_COST_PER_SHIP
                self._episode_reroute_cost += REROUTE_COST_PER_SHIP
                return float(REROUTE_COST_PER_SHIP)

        return 0.0

    def _expedite_order(
        self,
        source: Optional[str],
        destination: Optional[str],
        volume: Optional[float],
    ) -> float:
        """Place a rush order from *source* to *destination*.

        The order ships at 2x base cost with a 1-day ETA.  Actual volume
        is clamped by source stock and destination remaining capacity.
        """
        if not all([source, destination, volume]):
            return 0.0

        source_inv = self._inventory.get(source)  # type: ignore[arg-type]
        dest_inv = self._inventory.get(destination)  # type: ignore[arg-type]
        if source_inv is None or dest_inv is None:
            return 0.0

        actual_volume = min(
            volume,  # type: ignore[arg-type]
            source_inv.current_stock,
            dest_inv.capacity - dest_inv.current_stock,
        )
        if actual_volume <= 0:
            return 0.0

        cost = actual_volume * BASE_SHIPPING_COST_PER_UNIT * EXPEDITE_COST_MULTIPLIER
        source_inv.current_stock -= actual_volume
        ship = self._create_ship(source, destination, actual_volume, eta_days=1.0)  # type: ignore[arg-type]
        self._transit_queue.append(ship)
        self._episode_expedite_cost += cost
        return cost

    def _adjust_buffer(
        self, target_node: Optional[str], target_inventory: Optional[float],
    ) -> float:
        """Purchase inventory to raise *target_node* stock toward
        *target_inventory*.

        Creates a standard-cost shipment from PORT_SINGAPORE with a
        2-day ETA.  Volume is capped by remaining node capacity.
        """
        if target_node is None or target_inventory is None:
            return 0.0

        inv = self._inventory.get(target_node)
        if inv is None:
            return 0.0

        gap = target_inventory - inv.current_stock
        if gap <= 0:
            return 0.0

        actual_purchase = min(gap, inv.capacity - inv.current_stock)
        cost = actual_purchase * BASE_SHIPPING_COST_PER_UNIT
        ship = self._create_ship("PORT_SINGAPORE", target_node, actual_purchase, eta_days=2.0)
        self._transit_queue.append(ship)
        return cost

    # ─── Ship & Replenishment ────────────────────────────────────────────

    def _advance_ships(self) -> float:
        """Advance every ship by 1 day and deliver arrived cargo.

        Ships arriving at a BLOCKED port hold offshore with a 1-day retry
        loop (status → DELAYED).  Successfully delivered ships are marked
        DOCKED and removed from the transit queue.

        Returns total delivered volume.
        """
        still_in_transit: List[Ship] = []
        total_arrived = 0.0

        for ship in self._transit_queue:
            ship.eta_days -= 1.0

            if ship.eta_days <= 0:
                dest_status = self._port_status.get(ship.destination_port)
                dest_inv = self._inventory.get(ship.destination_port)

                if dest_inv is not None and dest_status != PortStatus.BLOCKED:
                    space = dest_inv.capacity - dest_inv.current_stock
                    delivered = min(ship.cargo_volume, space)
                    dest_inv.current_stock += delivered
                    total_arrived += delivered
                    ship.status = ShipStatus.DOCKED
                else:
                    ship.eta_days = 1.0
                    ship.status = ShipStatus.DELAYED
                    still_in_transit.append(ship)
            else:
                still_in_transit.append(ship)

        self._transit_queue = still_in_transit
        return total_arrived

    def _auto_replenish(self) -> float:
        """(s, S) replenishment: auto-order from the first OPEN port when a
        warehouse drops below its safety-stock threshold.

        This background ordering is a primary contributor to the Bullwhip
        Effect — it runs even when the agent takes NOOP actions.

        Returns total replenishment shipping cost.
        """
        total_cost = 0.0

        for wid in WAREHOUSES:
            wh_inv = self._inventory[wid]
            if wh_inv.current_stock >= wh_inv.safety_stock:
                continue

            open_ports = [
                pid for pid, status in self._port_status.items()
                if status == PortStatus.OPEN
            ]
            if not open_ports:
                continue

            source_port = open_ports[0]
            port_inv = self._inventory[source_port]
            order_volume = min(
                wh_inv.capacity - wh_inv.current_stock,
                port_inv.current_stock,
            )
            if order_volume <= 0:
                continue

            cost = order_volume * BASE_SHIPPING_COST_PER_UNIT
            eta = TRANSIT_TIMES.get((source_port, wid), 2.0)
            ship = self._create_ship(source_port, wid, order_volume, eta)
            self._transit_queue.append(ship)
            port_inv.current_stock -= order_volume
            total_cost += cost

        return total_cost

    def _spawn_initial_ships(self) -> None:
        """Pre-populate the transit queue with three ships at episode start."""
        initial_routes: List[Tuple[str, str, float, float]] = [
            ("PORT_SINGAPORE", "WH_ASIA_HUB",   2000.0, 1.0),
            ("PORT_ROTTERDAM", "WH_EUROPE_HUB", 1500.0, 0.5),
            ("PORT_SHANGHAI",  "WH_ASIA_HUB",   2500.0, 2.0),
        ]
        for origin, destination, volume, eta in initial_routes:
            ship = self._create_ship(origin, destination, volume, eta)
            self._transit_queue.append(ship)

    def _create_ship(
        self, origin: str, destination: str, volume: float, eta_days: float,
    ) -> Ship:
        """Factory method — mint a new Ship with an auto-incrementing ID."""
        self._ship_counter += 1
        return Ship(
            ship_id=f"V-{self._ship_counter:03d}",
            cargo_volume=volume,
            origin_port=origin,
            destination_port=destination,
            eta_days=eta_days,
        )

    # ─── Shock System ────────────────────────────────────────────────────

    def _apply_shocks(self) -> None:
        """Process scheduled port disruptions and random demand spikes.

        Scheduled shocks have a ``day`` (onset) and optional ``end_day``
        (resolution).  Random demand spikes fire with 15 % probability
        per day when enabled, multiplying a random factory's burn rate
        by 1.2–1.8x for one day.
        """
        if self._pending_spike_reset:
            for fid, original_rate in self._pending_spike_reset.items():
                if fid in self._factory_demands:
                    self._factory_demands[fid].daily_burn_rate = original_rate
            self._pending_spike_reset = {}

        for shock in self.shock_config.get("scheduled", []):
            if shock["day"] == self._day:
                self._port_status[shock["port"]] = PortStatus(shock["status"])

        for shock in self.shock_config.get("scheduled", []):
            if shock.get("end_day") == self._day:
                self._port_status[shock["port"]] = PortStatus.OPEN

        if self.shock_config.get("random_demand_spikes") and self._rng.random() < 0.15:
            fid = self._rng.choice(list(self._factory_demands.keys()))
            original_rate = self._factory_demands[fid].daily_burn_rate
            self._pending_spike_reset[fid] = original_rate
            self._factory_demands[fid].daily_burn_rate = original_rate * self._rng.uniform(1.2, 1.8)

    # ─── Reward Computation ──────────────────────────────────────────────

    def _compute_reward(
        self,
        demand_fulfilled: float,
        demand_total: float,
        bullwhip_index: float,
        action: Action,
        action_cost: float,
    ) -> Reward:
        """Dense, multi-component reward with asymmetric penalty structure.

        Stockout cost >> holding cost (500:0.5 per unit) to teach the
        agent that availability always dominates cost efficiency.

        Components:
            service_level_bonus  — proportional to daily fill rate
            stockout_penalty     — per unfulfilled unit (scaled)
            holding_cost_penalty — aggregate across all nodes (scaled)
            reroute_cost_penalty — flat fee when action is REROUTE_SHIP
            expedite_cost_penalty— 2x shipping when action is EXPEDITE_ORDER
            bullwhip_penalty     — activates above index 1.5
            stability_bonus      — binary; +50 when all factories have stock
        """
        service_rate = demand_fulfilled / max(demand_total, 1e-9)
        service_bonus = SERVICE_LEVEL_BONUS * service_rate

        unfulfilled = max(0.0, demand_total - demand_fulfilled)
        stockout_penalty = -unfulfilled * STOCKOUT_COST_PER_UNIT * REWARD_SCALE

        holding_cost = sum(
            inv.current_stock * HOLDING_COST_PER_UNIT_DAY
            for inv in self._inventory.values()
        )
        holding_penalty = -holding_cost * REWARD_SCALE

        reroute_penalty = (
            -action_cost * REWARD_SCALE
            if action.action_type == ActionType.REROUTE_SHIP
            else 0.0
        )
        expedite_penalty = (
            -action_cost * REWARD_SCALE
            if action.action_type == ActionType.EXPEDITE_ORDER
            else 0.0
        )

        bullwhip_penalty = 0.0
        if bullwhip_index > 1.5:
            bullwhip_penalty = -(bullwhip_index - 1.5) * 20.0

        all_stable = all(
            self._inventory[fid].current_stock > 0 for fid in FACTORIES
        )
        stability = STABILITY_BONUS if all_stable else 0.0

        total = (
            service_bonus
            + stockout_penalty
            + holding_penalty
            + reroute_penalty
            + expedite_penalty
            + bullwhip_penalty
            + stability
        )

        components = RewardComponents(
            service_level_bonus=round(service_bonus, 2),
            stockout_penalty=round(stockout_penalty, 2),
            holding_cost_penalty=round(holding_penalty, 2),
            reroute_cost_penalty=round(reroute_penalty, 2),
            expedite_cost_penalty=round(expedite_penalty, 2),
            bullwhip_penalty=round(bullwhip_penalty, 2),
            stability_bonus=round(stability, 2),
        )

        done = (self._day + 1) >= self.episode_length

        return Reward(
            total=round(total, 4),
            components=components,
            done=done,
            info={
                "service_rate": round(service_rate, 4),
                "bullwhip_index": bullwhip_index,
                "all_factories_stable": all_stable,
            },
        )
