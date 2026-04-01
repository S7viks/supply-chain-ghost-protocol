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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from models import Action, ActionType, Observation
from rollout import HeuristicAgent
from tasks import ALL_TASKS, Task, run_task

_heuristic = HeuristicAgent()


# ─── Configuration ───────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
FALLBACK_MODELS = [
    os.getenv("FALLBACK_MODEL", "google/gemma-3-27b-it"),
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

if not API_KEY:
    import sys
    print(
        "FATAL: HF_TOKEN (or API_KEY) is not set. "
        "Export it before running inference:\n"
        "  export HF_TOKEN='hf_...'\n"
        "On HF Spaces, add it under Settings -> Repository secrets.",
        file=sys.stderr,
    )
    sys.exit(1)

MAX_STEPS   = 30
TEMPERATURE = 0.1
MAX_TOKENS  = 512
SEED        = 42

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

_active_model = MODEL_NAME


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

def _task_brief(task: Task) -> str:
    scheduled = task.shock_config.get("scheduled", [])
    spikes = bool(task.shock_config.get("random_demand_spikes", False))
    lines = [
        f"TASK: {task.name} ({task.difficulty.upper()}) | Episode length: {task.episode_length} days",
        "KNOWN UPCOMING DISRUPTIONS (public task conditions):",
    ]
    if scheduled:
        for s in scheduled:
            end = s.get("end_day")
            if end is None:
                lines.append(f"- Day {s['day']}: {s['port']} -> {s['status']}")
            else:
                lines.append(f"- Day {s['day']}-{end}: {s['port']} -> {s['status']}")
    else:
        lines.append("- None")
    lines.append(f"Random demand spikes enabled: {spikes}")
    return "\n".join(lines)


def _prev_outcome_brief(prev_step: Optional[Dict[str, Any]]) -> str:
    if not prev_step:
        return "PREVIOUS OUTCOME: None (episode start)"
    parts: List[str] = [
        f"PREVIOUS OUTCOME: day={prev_step.get('day')} action={prev_step.get('action_type')}",
        f"reward={prev_step.get('reward_total')} service_level={prev_step.get('service_level')} bullwhip={prev_step.get('bullwhip_index')}",
        f"stockout={prev_step.get('any_factory_stockout')}",
    ]
    return " | ".join(parts)


@dataclass
class _AgentMemory:
    history: List[Tuple[str, str]] = field(default_factory=list)  # (user, assistant_json)

    def append(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append((user_msg, assistant_msg))
        self.history = self.history[-4:]  # keep last 4 turns

    def to_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for u, a in self.history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        return messages


def _llm_call(model: str, messages: List[Dict[str, str]]) -> str:
    """Single LLM completion call. Returns the response text or raises."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
    )
    return (response.choices[0].message.content or "").strip()


def make_llm_agent(task: Task) -> Any:
    """Build a stateful LLM agent closure for a given task.

    Model cascade on failure:
        1. Primary model (MODEL_NAME)
        2. Fallback models (FALLBACK_MODELS) — tried in order
        3. HeuristicAgent — domain-aware rule-based policy
    """
    global _active_model
    mem = _AgentMemory()
    brief = _task_brief(task)

    def llm_agent(obs: Observation, prev_step: Optional[Dict[str, Any]] = None, _task: Optional[Task] = None) -> Action:
        """Query the LLM with model cascade; fall back to heuristic as last resort."""
        global _active_model
        user_message = "\n\n".join([brief, _prev_outcome_brief(prev_step), "CURRENT OBSERVATION:", obs_to_prompt(obs)])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *mem.to_messages(),
            {"role": "user", "content": user_message},
        ]

        models_to_try = [_active_model] if _active_model != MODEL_NAME else [MODEL_NAME]
        for fb in FALLBACK_MODELS:
            if fb not in models_to_try:
                models_to_try.append(fb)
        if MODEL_NAME not in models_to_try:
            models_to_try.insert(0, MODEL_NAME)

        for model in models_to_try:
            try:
                action_text = _llm_call(model, messages)
                action = parse_action(action_text)
                mem.append(user_message, action.model_dump_json())
                if model != _active_model:
                    print(f"[MODEL] Switched to {model}")
                    _active_model = model
                return action
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status in (401, 402, 429):
                    continue
                print(f"[ERROR] {model}: {type(exc).__name__}: {exc}")
                continue

        print("[FALLBACK] All models exhausted, using heuristic agent")
        return _heuristic(obs, prev_step, _task or task)

    return llm_agent


# ─── Task Runner ─────────────────────────────────────────────────────────────


def run_all_tasks(
    task_filter: Optional[str] = None, verbose: bool = False,
) -> Dict[str, Any]:
    """Run inference against all tasks (or a filtered subset)."""
    results: Dict[str, Any] = {}
    all_pass = True

    print("=" * 60)
    print("Supply Chain Ghost Protocol -- Baseline Inference")
    print(f"Model:     {MODEL_NAME}")
    print(f"Fallbacks: {', '.join(FALLBACK_MODELS)}")
    print(f"API:       {API_BASE_URL}")
    print("=" * 60)

    for task in ALL_TASKS:
        if task_filter and task.difficulty != task_filter:
            continue

        print(f"\n[TASK] {task.name} ({task.difficulty.upper()}) -- {task.episode_length} days")
        print(f"       {task.description[:120]}...")

        agent = make_llm_agent(task)
        result, trajectory = run_task(
            task=task,
            agent_fn=agent,
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
