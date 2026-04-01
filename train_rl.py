"""
Supply Chain Ghost Protocol — PPO Training Script
==================================================
Train a PPO agent on the Gymnasium-wrapped supply chain environment
using Stable-Baselines3.

Usage:
    python train_rl.py --task hard --timesteps 50000 --output models/ppo_supply_chain
    python train_rl.py --eval-only models/ppo_supply_chain --task hard
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from gym_wrapper import GymSupplyChainEnv
from models import Action, Observation
from tasks import (
    TASK_CASCADE,
    TASK_EASY,
    TASK_HARD,
    TASK_MEDIUM,
    Task,
    run_task,
)

TASK_ALIASES: Dict[str, str] = {
    "easy": "task_easy_delay",
    "medium": "task_medium_blockage",
    "cascade": "task_cascade_dual_chokepoint",
    "hard": "task_hard_ghost_protocol",
}

TASK_OBJECTS: Dict[str, Task] = {
    "task_easy_delay": TASK_EASY,
    "task_medium_blockage": TASK_MEDIUM,
    "task_cascade_dual_chokepoint": TASK_CASCADE,
    "task_hard_ghost_protocol": TASK_HARD,
}


def make_env(task_id: str, seed: int = 42) -> Monitor:
    """Create a Monitor-wrapped GymSupplyChainEnv."""
    return Monitor(GymSupplyChainEnv(task_id=task_id, seed=seed))


def train(
    task_id: str,
    total_timesteps: int = 50_000,
    output_path: str = "models/ppo_supply_chain",
    seed: int = 42,
    eval_freq: int = 5000,
) -> PPO:
    """Train a PPO agent and save the model checkpoint."""
    env = make_env(task_id, seed=seed)
    eval_env = make_env(task_id, seed=seed + 1000)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    eval_log_dir = os.path.join(output_dir, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
    )

    print(f"Training PPO on task '{task_id}' for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(output_path)
    print(f"Model saved to {output_path}")

    env.close()
    eval_env.close()
    return model


def load_rl_agent(model_path: str, task_id: str) -> Callable[..., Action]:
    """Load a trained PPO model and return an agent_fn compatible with run_task.

    Returned signature: ``agent_fn(obs, prev_step, task) -> Action``
    """
    wrapper = GymSupplyChainEnv(task_id=task_id)
    model = PPO.load(model_path, env=wrapper)

    def agent_fn(
        obs: Observation,
        prev_step: Optional[Dict[str, Any]] = None,
        task: Optional[Task] = None,
    ) -> Action:
        obs_vec = wrapper._encode_obs(obs)
        wrapper._last_obs = obs
        action_int, _ = model.predict(obs_vec, deterministic=True)
        return wrapper._decode_action(int(action_int))

    return agent_fn


def evaluate(model_path: str, task_id: str, seed: int = 42) -> None:
    """Run one full episode with the grader and print results."""
    task_obj = TASK_OBJECTS.get(task_id)
    if task_obj is None:
        print(f"No grader found for task '{task_id}', skipping evaluation.")
        return

    agent_fn = load_rl_agent(model_path, task_id)
    result, _ = run_task(task_obj, agent_fn, seed=seed, verbose=True)

    print("\n--- Evaluation Results ---")
    print(f"Task:    {result.task_id}")
    print(f"Score:   {result.score:.4f}")
    print(f"Success: {result.success}")
    print(f"Metrics: {result.metrics}")
    if result.failure_reasons:
        print(f"Failures: {result.failure_reasons}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Supply Chain Ghost Protocol")
    parser.add_argument(
        "--task", type=str, default="hard",
        choices=list(TASK_ALIASES) + list(TASK_OBJECTS),
        help="Task difficulty alias or full task_id (default: hard)",
    )
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--output", type=str, default="models/ppo_supply_chain", help="Model save path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluate every N steps")
    parser.add_argument(
        "--eval-only", type=str, default=None, metavar="MODEL_PATH",
        help="Skip training; evaluate an existing model",
    )
    args = parser.parse_args()

    task_id = TASK_ALIASES.get(args.task, args.task)

    if args.eval_only:
        evaluate(args.eval_only, task_id, seed=args.seed)
        return

    train(
        task_id=task_id,
        total_timesteps=args.timesteps,
        output_path=args.output,
        seed=args.seed,
        eval_freq=args.eval_freq,
    )
    evaluate(args.output, task_id, seed=args.seed)


if __name__ == "__main__":
    main()
