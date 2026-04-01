"""
Supply Chain Ghost Protocol — HTTP Server
==========================================
FastAPI server exposing the OpenEnv step()/reset()/state() API.
Required for Hugging Face Spaces deployment.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import SupplyChainEnv
from models import Action, Observation, Reward
from tasks import ALL_TASKS, TASK_EASY

app = FastAPI(
    title="Supply Chain Ghost Protocol",
    description="OpenEnv-compliant supply chain crisis management RL environment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, Dict[str, Any]] = {}


# ─── Request / Response Schemas ──────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy_delay"
    seed: Optional[int] = 42


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation


class StepRequest(BaseModel):
    session_id: str
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "environment": "supply-chain-ghost-protocol"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "episode_length": t.episode_length,
                "description": t.description,
            }
            for t in ALL_TASKS
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    """Initialise a new episode. Returns session_id and initial observation."""
    task = next((t for t in ALL_TASKS if t.task_id == request.task_id), TASK_EASY)
    env = task.build_env(seed=request.seed or 42)
    obs = env.reset()

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"env": env, "task": task, "done": False}

    return ResetResponse(session_id=session_id, observation=obs)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Advance the environment by one time-step."""
    session = _sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    if session["done"]:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset.")

    env: SupplyChainEnv = session["env"]
    obs, reward, done, info = env.step(request.action)
    session["done"] = done

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state/{session_id}", response_model=Observation)
def state(session_id: str) -> Observation:
    """Return current state without advancing the simulation."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    env: SupplyChainEnv = session["env"]
    return env.state()


@app.delete("/session/{session_id}")
def close_session(session_id: str) -> Dict[str, str]:
    """Clean up a session."""
    _sessions.pop(session_id, None)
    return {"status": "closed"}
