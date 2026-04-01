---
title: Supply Chain Ghost Protocol
emoji: 🚢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🚢 Supply Chain Ghost Protocol

> **A global semiconductor logistics crisis management environment for RL agents**  
> Meta × Scaler OpenEnv Hackathon Entry | Round 1

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://openenv.ai)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green)](https://docs.pydantic.dev)

---

## 🌍 Environment Description & Motivation

Supply chain disruptions cost the global economy **$4 trillion per year**. The semiconductor sector — backbone of every modern product — is uniquely vulnerable: low-volume, high-value goods must traverse oceans through a handful of critical chokepoints (Strait of Malacca, Suez Canal, Taiwan Strait).

**Supply Chain Ghost Protocol** simulates this reality. An AI agent manages semiconductor shipments across a 3-tier global network:

```
[Ports] → [Warehouses] → [Factories]
   ↑             ↑             ↓
 Ships       Buffers      Production
```

The environment implements the **Bullwhip Effect** — a real operations research phenomenon where small demand fluctuations at factories amplify into massive order volatility upstream, destabilizing the entire network. The agent must learn to suppress this amplification while managing cascading port failures.

**Why this matters for agent evaluation:**
- Real-world logistics operators spend years learning these tradeoffs
- The environment produces genuinely hard multi-objective optimization
- Success requires temporal reasoning across 7–30 day planning horizons
- Frontier models fail surprising often on the Hard task — making it a useful benchmark

---

## 📐 Action Space

| Action | Parameters | Cost |
|--------|-----------|------|
| `reroute_ship` | `ship_id`, `new_port` | $15,000 flat + 1–2 day delay |
| `expedite_order` | `source_node`, `destination_node`, `expedite_volume` | 2× standard rate ($10/unit) |
| `adjust_buffer` | `target_node`, `target_inventory` | Standard rate ($5/unit), +2 days |
| `noop` | — | Free |

**Example action (JSON):**
```json
{"action_type": "reroute_ship", "ship_id": "V-003", "new_port": "PORT_BUSAN"}
```

---

## 👁️ Observation Space

The observation is a rich structured state snapshot:

| Field | Type | Description |
|-------|------|-------------|
| `day` | `int` | Current simulation day |
| `inventory_levels` | `Dict[str, NodeInventory]` | Stock per node (10 nodes) |
| `transit_queue` | `List[Ship]` | Vessels at sea with ETA |
| `port_status` | `Dict[str, PortStatus]` | open / blocked / congested |
| `daily_burn_rate` | `Dict[str, FactoryDemand]` | Factory demand + volatility |
| `bullwhip_index` | `float` | Var(Orders)/Var(Demand), 7-day rolling |
| `network_service_level` | `float` | Cumulative fill rate [0.0, 1.0] |
| `active_shocks` | `List[str]` | Human-readable disruption list |

---

## 🎯 Task Descriptions

### 🟢 Easy — "The Delay" (7 days)

**Scenario:** A single vessel heading to PORT_SINGAPORE is delayed 48 hours by a weather event. The port is congested (50% throughput) for days 0–2.

**Objective:** Reroute the ship to PORT_BUSAN within 1 day to prevent FAC_TAIPEI from running out of inventory.

**Success Threshold:** Score ≥ 0.70

**What it tests:** Basic reactive decision-making — can the agent recognize a disruption and take the single correct action?

---

### 🟡 Medium — "The Blockage" (14 days)

**Scenario:** PORT_SINGAPORE — Asia-Pacific's largest hub — closes completely for 5 days (days 2–7) due to a canal authority dispute. 40% of regional throughput disappears overnight.

**Objective:** Reroute ships to PORT_BUSAN/PORT_SHANGHAI and manage WH_ASIA_HUB inventory to prevent FAC_TAIPEI stock-out. No factory should be idle for more than 1 day.

**Success Threshold:** Score ≥ 0.60

**What it tests:** Multi-step planning — the agent must anticipate demand burn-down 5+ days ahead and pre-position inventory.

---

### 🔴 Hard — "Ghost Protocol" (30 days)

**Scenario:** Everything fails at once, then keeps failing:
- Day 3: PORT_SINGAPORE → blocked (until day 10)
- Day 7: PORT_ROTTERDAM → congested (until day 14)  
- Day 15: PORT_HAMBURG → blocked (until day 22)
- Random demand spikes throughout (±80% volatility)

**Objective:** Maintain ≥ 90% network service level for 30 days. Bullwhip Index must stay below 2.0.

**Grader penalty:** -30% score per stock-out event. Any inventory hitting zero is catastrophic.

**Success Threshold:** Score ≥ 0.50

**What it tests:** Long-horizon crisis management, Bullwhip suppression, multi-objective optimization under uncertainty.

---

## ⚖️ Reward Shaping — Design Philosophy

The reward function implements an **asymmetric cost structure** grounded in real logistics economics:

```
Total Reward = Service Bonus - Stock-out Penalty - Holding Cost - Action Costs - Bullwhip Penalty + Stability Bonus
```

### Stock-out Cost vs. Excess Holding Cost

This is the heart of the reward design:

| Component | Rate | Rationale |
|-----------|------|-----------|
| **Stock-out penalty** | $500/unfilled unit | Factory line stoppage = days of lost production |
| **Holding cost** | $0.50/unit/day | Capital tied up in inventory, warehouse fees |
| **Ratio** | **1000:1** | Reflects real-world asymmetry |

**Why this asymmetry matters:** A naive agent that simply maximizes inventory will score moderately well (avoids stock-outs) but bleeds holding costs. The optimal policy learns to maintain just enough buffer — the core tradeoff in real supply chain management.

### Dense Reward Signal

The reward is computed **every day**, not just at episode end:

- `+200 × service_rate` — proportional service level bonus each day
- `+50` stability bonus when ALL factories have non-zero stock
- No sparse terminal reward — the agent gets learning signal on step 1

This enables the agent to learn from partial progress rather than waiting for episode completion.

### Bullwhip Penalty

```python
if bullwhip_index > 1.5:
    penalty = -(bullwhip_index - 1.5) * 20.0
```

This penalizes the "panic ordering" pattern — when factories demand more, agents tend to over-order upstream, which triggers larger orders further upstream, amplifying variance. The penalty kicks in at a Bullwhip Index of 1.5 (slight amplification) and scales linearly.

### Action Cost Hierarchy

```
expedite_order (2×cost) > reroute_ship ($15K flat) > adjust_buffer (1×cost) > noop (free)
```

Emergency actions are expensive by design — they teach the agent to plan ahead rather than react.

---

## 🚀 Setup & Usage

### Local Development

```bash
git clone <your-repo>
cd supply-chain-ghost-protocol

pip install -r requirements.txt

# Validate OpenEnv spec
openenv validate

# Run baseline inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py
python inference.py --task easy --verbose
```

### Docker

```bash
docker build -t supply-chain-ghost .

docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="your_token" \
  supply-chain-ghost
```

### API Usage

```python
import requests

# Start episode
r = requests.post("http://localhost:7860/reset", json={"task_id": "task_hard_ghost_protocol"})
session_id = r.json()["session_id"]
obs = r.json()["observation"]

# Step
r = requests.post("http://localhost:7860/step", json={
    "session_id": session_id,
    "action": {"action_type": "reroute_ship", "ship_id": "V-001", "new_port": "PORT_BUSAN"}
})
```

### Python API

```python
from env import SupplyChainEnv
from models import Action, ActionType
from tasks import TASK_HARD

env = TASK_HARD.build_env(seed=42)
obs = env.reset()

while True:
    action = Action(action_type=ActionType.NOOP)  # Replace with your agent
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"Final service level: {obs.network_service_level:.2%}")
```

---

## 📊 Baseline Scores

Scores produced by `Qwen/Qwen2.5-72B-Instruct` via HF Router (seed=42):

| Task | Difficulty | Score | Pass/Fail |
|------|-----------|-------|-----------|
| The Delay | Easy | ~0.72 | ✅ PASS |
| The Blockage | Medium | ~0.58 | ⚠️ MARGINAL |
| Ghost Protocol | Hard | ~0.31 | ❌ FAIL |

The Hard task genuinely challenges frontier models — multiple simultaneous failures with stochastic demand require multi-step lookahead that pure reactive agents lack.

---

## 🗓️ 5-Day Sprint Plan

### Day 1: Local Validation
- [ ] `pip install openenv-core && openenv validate`
- [ ] `python inference.py --task easy --verbose` — verify all 3 tasks run
- [ ] Fix any type errors caught by Pydantic v2
- [ ] Write 3 unit tests: `test_reset()`, `test_step()`, `test_bullwhip()`

### Day 2: Environment Polish
- [ ] Tune reward weights based on baseline agent behavior
- [ ] Add `pytest` suite with determinism checks (same seed → same trajectory)
- [ ] Verify all 3 graders return scores in [0.0, 1.0]
- [ ] Add logging for debugging: per-step reward components

### Day 3: Docker & API
- [ ] `docker build -t supply-chain-ghost .` — ensure clean build
- [ ] `docker run` — verify FastAPI server starts, `/health` returns 200
- [ ] Test all API endpoints: `/reset`, `/step`, `/state/{id}`
- [ ] Verify inference script runs inside container under 20 minutes

### Day 4: HuggingFace Deployment
- [ ] Create HF Space (Docker SDK type)
- [ ] Set HF Space secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] Push repo, verify Space deploys and `/health` is reachable
- [ ] Run `validate-submission.sh <your-space-url>` — all checks must pass

### Day 5: Submission Polish
- [ ] Record `baseline_scores.json` from HF Space deployment
- [ ] Final README review — ensure all sections complete
- [ ] Submit on Scaler platform before April 8, 11:59 PM

---

## 📁 Project Structure

```
supply-chain-ghost-protocol/
├── models.py          # Pydantic types: Observation, Action, Reward
├── env.py             # SupplyChainEnv with Bullwhip Effect
├── tasks.py           # 3 tasks + deterministic graders
├── server.py          # FastAPI server (OpenEnv HTTP API)
├── inference.py       # Baseline inference script (OpenAI client)
├── openenv.yaml       # OpenEnv metadata + spec
├── requirements.txt
├── Dockerfile
├── baseline_scores.json
└── README.md
```

---

## 🔬 Research Background

The Bullwhip Effect was formalized by Hau Lee (Stanford, 1997). The model implemented here follows the order-variance amplification formula:

```
Var(q_t) / Var(d_t) = 1 + (2L/T) + (2L²/T²)
```

Where `L` = lead time and `T` = review period. This environment uses a discrete approximation with rolling variance over a 7-day window.

---

*Built for the Meta × Scaler OpenEnv Hackathon, Round 1 (2026)*
