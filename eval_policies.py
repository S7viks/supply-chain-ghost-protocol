"""
Supply Chain Ghost Protocol — Policy Evaluation Harness
========================================================
Runs multiple policies across all tasks and seeds, collecting performance
metrics for side-by-side comparison.

Usage:
    python eval_policies.py --seeds 5 --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from rollout import HeuristicAgent, NaiveAgent, record_rollout
from tasks import ALL_TASKS, Task


# ─── Evaluation Core ─────────────────────────────────────────────────────────


def evaluate_policy(
    policy_name: str,
    agent_fn: Any,
    tasks: List[Task],
    seeds: List[int],
) -> List[Dict[str, Any]]:
    """Run a policy across all tasks and seeds, collecting metrics."""
    results: List[Dict[str, Any]] = []

    for task in tasks:
        for seed in seeds:
            rollout = record_rollout(
                task, agent_fn, seed=seed, policy_name=policy_name,
            )

            stockout_count = sum(
                1 for step in rollout.steps if step.any_factory_stockout
            )
            max_bullwhip = max(
                (step.bullwhip_index for step in rollout.steps), default=1.0,
            )
            avg_bullwhip = (
                sum(s.bullwhip_index for s in rollout.steps) / len(rollout.steps)
                if rollout.steps
                else 1.0
            )
            total_reward = sum(s.reward_total for s in rollout.steps)

            results.append({
                "policy": policy_name,
                "task_id": task.task_id,
                "task_name": task.name,
                "difficulty": task.difficulty,
                "seed": seed,
                "score": rollout.final_score,
                "service_level": rollout.final_service_level,
                "stockout_days": stockout_count,
                "max_bullwhip": round(max_bullwhip, 4),
                "avg_bullwhip": round(avg_bullwhip, 4),
                "total_reward": round(total_reward, 4),
                "episode_length": task.episode_length,
            })

    return results


# ─── Summary Aggregation ────────────────────────────────────────────────────


def build_summary(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-run results into a nested summary keyed by policy and task."""
    summary: Dict[str, Any] = {"policies": {}}

    policies = sorted(set(r["policy"] for r in all_results))
    tasks = sorted(set(r["task_id"] for r in all_results))

    for policy in policies:
        policy_data: Dict[str, Any] = {"tasks": {}, "overall": {}}
        policy_results = [r for r in all_results if r["policy"] == policy]

        for task_id in tasks:
            task_results = [r for r in policy_results if r["task_id"] == task_id]
            if not task_results:
                continue

            scores = [r["score"] for r in task_results]
            sls = [r["service_level"] for r in task_results]
            stockouts = [r["stockout_days"] for r in task_results]
            bullwhips = [r["max_bullwhip"] for r in task_results]
            rewards = [r["total_reward"] for r in task_results]
            n = len(scores)

            policy_data["tasks"][task_id] = {
                "task_name": task_results[0]["task_name"],
                "difficulty": task_results[0]["difficulty"],
                "n_seeds": n,
                "score_mean": round(sum(scores) / n, 4),
                "score_min": round(min(scores), 4),
                "score_max": round(max(scores), 4),
                "service_level_mean": round(sum(sls) / n, 4),
                "stockout_days_mean": round(sum(stockouts) / n, 2),
                "max_bullwhip_mean": round(sum(bullwhips) / n, 4),
                "total_reward_mean": round(sum(rewards) / n, 2),
            }

        all_scores = [r["score"] for r in policy_results]
        all_sls = [r["service_level"] for r in policy_results]
        n_total = len(all_scores)
        if n_total:
            policy_data["overall"] = {
                "n_runs": n_total,
                "score_mean": round(sum(all_scores) / n_total, 4),
                "service_level_mean": round(sum(all_sls) / n_total, 4),
            }

        summary["policies"][policy] = policy_data

    return summary


# ─── Table Rendering ────────────────────────────────────────────────────────

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def print_comparison_table(
    summary: Dict[str, Any],
    all_results: List[Dict[str, Any]],
) -> None:
    """Print a formatted comparison table to stdout."""
    policies = sorted(summary["policies"].keys())
    tasks = sorted(
        set(r["task_id"] for r in all_results),
        key=lambda t: _DIFFICULTY_ORDER.get(
            next((r["difficulty"] for r in all_results if r["task_id"] == t), ""),
            3,
        ),
    )

    col_w = 12
    policy_block_w = col_w * 3 + 4

    sep = "=" * (30 + len(policies) * (policy_block_w + 2))
    print(sep)
    print("POLICY COMPARISON")
    print(sep)

    header = f"{'Task':<30}"
    for p in policies:
        header += f"| {p:^{policy_block_w}}"
    print(header)

    sub = f"{'':30}"
    for _ in policies:
        sub += f"| {'Score':>{col_w}} {'SL':>{col_w}} {'Stockouts':>{col_w}}"
    print(sub)
    print("-" * len(sep))

    for task_id in tasks:
        task_name = next(
            (r["task_name"] for r in all_results if r["task_id"] == task_id),
            task_id,
        )
        difficulty = next(
            (r["difficulty"] for r in all_results if r["task_id"] == task_id),
            "",
        )
        label = f"{task_name} ({difficulty})"
        row = f"{label:<30}"

        for policy in policies:
            pdata = (
                summary["policies"]
                .get(policy, {})
                .get("tasks", {})
                .get(task_id)
            )
            if pdata:
                row += (
                    f"| {pdata['score_mean']:>{col_w}.4f}"
                    f" {pdata['service_level_mean']:>{col_w}.2%}"
                    f" {pdata['stockout_days_mean']:>{col_w}.1f}"
                )
            else:
                row += f"| {'N/A':>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}}"

        print(row)

    print("-" * len(sep))

    overall_row = f"{'OVERALL':<30}"
    for policy in policies:
        overall = summary["policies"].get(policy, {}).get("overall", {})
        if overall:
            overall_row += (
                f"| {overall['score_mean']:>{col_w}.4f}"
                f" {overall['service_level_mean']:>{col_w}.2%}"
                f" {'':>{col_w}}"
            )
        else:
            overall_row += f"| {'N/A':>{col_w}} {'N/A':>{col_w}} {'':>{col_w}}"
    print(overall_row)
    print(sep)


# ─── CLI ─────────────────────────────────────────────────────────────────────


AVAILABLE_POLICIES: Dict[str, Any] = {
    "naive": NaiveAgent(),
    "heuristic": HeuristicAgent(),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supply Chain Ghost Protocol -- Policy Evaluation",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to evaluate (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Output JSON file (default: eval_results.json)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["naive", "heuristic"],
        help="Policies to evaluate (default: naive heuristic)",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    print("=" * 60)
    print("Supply Chain Ghost Protocol -- Policy Evaluation")
    print(f"Seeds: {seeds}")
    print(f"Policies: {args.policies}")
    print(f"Tasks: {len(ALL_TASKS)}")
    print("=" * 60)

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for policy_name in args.policies:
        agent = AVAILABLE_POLICIES.get(policy_name)
        if agent is None:
            print(f"Unknown policy: {policy_name}, skipping")
            continue

        print(f"\nEvaluating: {policy_name}")
        results = evaluate_policy(policy_name, agent, ALL_TASKS, seeds)
        all_results.extend(results)

        n = len(results)
        avg_score = sum(r["score"] for r in results) / max(n, 1)
        print(f"  {n} runs | avg score: {avg_score:.4f}")

    elapsed = time.time() - t0

    summary = build_summary(all_results)
    summary["metadata"] = {
        "seeds": seeds,
        "n_tasks": len(ALL_TASKS),
        "policies": args.policies,
        "elapsed_seconds": round(elapsed, 2),
    }
    summary["runs"] = all_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nCompleted in {elapsed:.1f}s")
    print_comparison_table(summary, all_results)
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
