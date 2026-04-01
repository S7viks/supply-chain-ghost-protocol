"""
Supply Chain Ghost Protocol — Inference
========================================
Baseline inference script using the OpenAI client (mandatory per hackathon spec).

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face / API key

Usage:
    python inference.py
    python inference.py --task easy     # Run only the easy task
    python inference.py --verbose       # Verbose step output
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any, Dict, Optional

from openai import OpenAI

from models import Action, ActionType, Observation
from tasks import ALL_TASKS, run_task


# ─── Configuration ───────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS   = 30
TEMPERATURE = 0.1
MAX_TOKENS  = 512
SEED        = 42

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a supply chain crisis manager for a global semiconductor logistics network.
You control shipments across ports, warehouses, and factories.

YOUR GOAL: Maintain high service levels (factories never run out of inventory)
while minimizing costs. Stock-outs are catastrophically expensive ($500/unit).
Excess inventory is merely annoying ($0.50/unit/day).

NETWORK NODES:
  Ports: PORT_SINGAPORE, PORT_ROTTERDAM, PORT_SHANGHAI, PORT_BUSAN, PORT_HAMBURG
  Warehouses: WH_ASIA_HUB, WH_EUROPE_HUB
  Factories: FAC_BERLIN, FAC_TAIPEI, FAC_AUSTIN

AVAILABLE ACTIONS (respond with exactly one as valid JSON):
  {"action_type": "reroute_ship", "ship_id": "<id>", "new_port": "<port>"}
  {"action_type": "expedite_order", "source_node": "<node>", "destination_node": "<node>", "expedite_volume": <float>}
  {"action_type": "adjust_buffer", "target_node": "<node>", "target_inventory": <float>}
  {"action_type": "noop"}

DECISION PRINCIPLES:
1. If any factory has < 2 days of stock, expedite immediately — cost is secondary.
2. If a port is blocked, reroute ships heading there to the nearest open port.
3. Keep Bullwhip Index below 2.0 — avoid panic ordering.
4. Maintain safety stock (safety_stock field) at each node as minimum buffer.
5. Prefer adjust_buffer for long-term rebalancing; expedite only for emergencies.

Respond ONLY with a single valid JSON action object. No explanation, no markdown.
""").strip()


# ─── Observation Serialisation ───────────────────────────────────────────────


def obs_to_prompt(obs: Observation) -> str:
    """Convert an Observation to a compact LLM-readable state string."""
    lines = [
        f"DAY {obs.day} | Remaining: {obs.episode_remaining_days} days",
        f"Service Level: {obs.network_service_level:.2%} | Bullwhip Index: {obs.bullwhip_index:.2f}",
        f"Active Disruptions: {', '.join(obs.active_shocks) or 'None'}",
        "",
        "FACTORY INVENTORY (critical nodes):",
    ]

    for fid, demand in obs.daily_burn_rate.items():
        inv = obs.inventory_levels.get(fid)
        if inv:
            days_left = demand.days_until_critical or 0
            if days_left < 2:
                urgency = "CRITICAL"
            elif days_left < 4:
                urgency = "LOW"
            else:
                urgency = "OK"
            lines.append(
                f"  {fid}: stock={inv.current_stock:.0f} | "
                f"burn={demand.daily_burn_rate:.0f}/day | "
                f"~{days_left:.1f} days left [{urgency}]"
            )

    lines.append("\nWAREHOUSE INVENTORY:")
    for wid in ("WH_ASIA_HUB", "WH_EUROPE_HUB"):
        inv = obs.inventory_levels.get(wid)
        if inv:
            lines.append(
                f"  {wid}: {inv.current_stock:.0f} / {inv.capacity:.0f} "
                f"({inv.utilization:.0%} full)"
            )

    lines.append("\nPORT STATUS:")
    for pid, status in obs.port_status.items():
        lines.append(f"  {pid}: {status.value.upper()}")

    lines.append(f"\nSHIPS IN TRANSIT ({len(obs.transit_queue)}):")
    for ship in obs.transit_queue[:6]:
        lines.append(
            f"  {ship.ship_id}: {ship.cargo_volume:.0f} units -> {ship.destination_port} "
            f"(ETA: {ship.eta_days:.1f} days, status: {ship.status.value})"
        )
    if len(obs.transit_queue) > 6:
        lines.append(f"  ... and {len(obs.transit_queue) - 6} more vessels")

    return "\n".join(lines)


# ─── Action Parsing ──────────────────────────────────────────────────────────


def parse_action(response_text: str) -> Action:
    """Parse LLM response into a typed Action. Falls back to noop on error."""
    text = response_text.strip()

    if text.startswith("```"):
        inner_lines = text.split("\n")
        text = "\n".join(inner_lines[1:-1]) if len(inner_lines) > 2 else text

    try:
        data = json.loads(text)
        return Action(**data)
    except Exception as exc:
        print(f"[WARN] Action parse failed: {exc}. Defaulting to noop.")
        return Action(action_type=ActionType.NOOP)


# ─── LLM Agent ───────────────────────────────────────────────────────────────


def llm_agent(obs: Observation) -> Action:
    """Query the LLM with the current observation and return a typed action."""
    user_message = obs_to_prompt(obs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        action_text = response.choices[0].message.content or ""
        return parse_action(action_text)
    except Exception as exc:
        print(f"[ERROR] LLM call failed: {exc}")
        return Action(action_type=ActionType.NOOP)


# ─── Task Runner ─────────────────────────────────────────────────────────────


def run_all_tasks(
    task_filter: Optional[str] = None, verbose: bool = False,
) -> Dict[str, Any]:
    """Run inference against all tasks (or a filtered subset)."""
    results: Dict[str, Any] = {}
    all_pass = True

    print("=" * 60)
    print("Supply Chain Ghost Protocol -- Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print("=" * 60)

    for task in ALL_TASKS:
        if task_filter and task.difficulty != task_filter:
            continue

        print(f"\n[TASK] {task.name} ({task.difficulty.upper()}) -- {task.episode_length} days")
        print(f"       {task.description[:120]}...")

        result, trajectory = run_task(
            task=task,
            agent_fn=llm_agent,
            seed=SEED,
            verbose=verbose,
        )

        status = "PASS" if result.success else "FAIL"
        print(f"\n{status} | Score: {result.score:.4f} (threshold: {task.success_threshold})")
        print(f"Metrics: {json.dumps(result.metrics, indent=2)}")
        if result.failure_reasons:
            print("Failure reasons:")
            for reason in result.failure_reasons:
                print(f"  - {reason}")

        results[task.task_id] = {
            "task": task.task_id,
            "difficulty": task.difficulty,
            "score": result.score,
            "success": result.success,
            "metrics": result.metrics,
            "failure_reasons": result.failure_reasons,
        }

        if not result.success:
            all_pass = False

    print("\n" + "=" * 60)
    label = "ALL TASKS PASSED" if all_pass else "SOME TASKS FAILED"
    print(f"OVERALL: {label}")
    scores = [r["score"] for r in results.values()]
    if scores:
        print(f"Average score: {sum(scores) / len(scores):.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supply Chain Ghost Protocol -- Inference",
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run only a specific task difficulty",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-step details")
    args = parser.parse_args()

    results = run_all_tasks(task_filter=args.task, verbose=args.verbose)

    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults written to baseline_scores.json")
