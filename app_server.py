"""
Supply Chain Ghost Protocol — HTTP Server
==========================================
FastAPI server exposing the OpenEnv step()/reset()/state() API.
Required for Hugging Face Spaces deployment.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env import SupplyChainEnv
from models import Action, Observation, Reward
from openenv.core.env_server.types import State as OpenEnvState
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


@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page for HF Spaces app tab."""
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Supply Chain Ghost Protocol</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;
display:flex;align-items:center;justify-content:center;min-height:100vh}
.c{max-width:640px;text-align:center;padding:2rem}
h1{font-size:2rem;margin-bottom:.5rem}
p{color:#94a3b8;margin-bottom:1.5rem;line-height:1.6}
.links{display:flex;gap:1rem;justify-content:center;flex-wrap:wrap}
a{color:#0f172a;background:#38bdf8;padding:.6rem 1.2rem;border-radius:6px;
text-decoration:none;font-weight:600;font-size:.9rem}
a:hover{background:#7dd3fc}
a.sec{background:transparent;color:#38bdf8;border:1px solid #38bdf8}
a.sec:hover{background:#1e293b}
.tag{display:inline-block;background:#1e293b;color:#38bdf8;padding:.2rem .6rem;
border-radius:4px;font-size:.75rem;margin-bottom:1rem}
</style></head><body><div class="c">
<div class="tag">OpenEnv-Compliant</div>
<h1>Supply Chain Ghost Protocol</h1>
<p>Semiconductor supply chain crisis management RL environment with
Bullwhip Effect simulation. 10 tasks across 3 difficulty tiers.</p>
<div class="links">
<a href="/docs">API Docs</a>
<a href="/viewer" class="sec">3D Viewer</a>
<a href="/health" class="sec">Health</a>
<a href="/tasks" class="sec">Tasks</a>
</div></div></body></html>"""


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "environment": "supply-chain-ghost-protocol"}


@app.get("/metadata")
def metadata() -> Dict[str, str]:
    return {
        "name": "supply-chain-ghost-protocol",
        "description": (
            "OpenEnv-compliant supply chain crisis management RL environment "
            "with Bullwhip Effect simulation"
        ),
        "version": "1.0.0",
        "author": "Ch Sai Sathvik",
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": OpenEnvState.model_json_schema(),
    }


@app.post("/mcp")
def mcp_endpoint(request: dict) -> Dict[str, Any]:
    """Minimal MCP JSON-RPC 2.0 endpoint."""
    method = request.get("method", "")
    request_id = request.get("id", 1)

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {"name": "reset", "description": "Reset the environment"},
                    {"name": "step", "description": "Take a step in the environment"},
                    {"name": "state", "description": "Get current environment state"},
                ]
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


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


@app.get("/state")
def state_default() -> Dict[str, Any]:
    """OpenEnv-compatible stateless state endpoint."""
    return {"episode_id": None, "step_count": 0}


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


# ─── 3D Viewer & Rollout Endpoints ───────────────────────────────────────────

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_ROLLOUTS_DIR = Path(__file__).resolve().parent / "rollouts"


@app.get("/viewer")
def viewer():
    """Serve the 3D rollout viewer."""
    html_path = _STATIC_DIR / "viewer.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="viewer.html not found")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/api/rollouts")
def list_rollouts() -> Dict[str, Any]:
    """List available recorded rollouts."""
    if not _ROLLOUTS_DIR.exists():
        return {"rollouts": []}
    results = []
    for f in sorted(_ROLLOUTS_DIR.glob("*.json")):
        entry: Dict[str, Any] = {"filename": f.name}
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            entry.update({
                "task_name": data.get("task_name", ""),
                "difficulty": data.get("difficulty", ""),
                "policy_name": data.get("policy_name", ""),
                "final_score": data.get("final_score"),
                "episode_length": data.get("episode_length"),
            })
        except (json.JSONDecodeError, OSError):
            pass
        results.append(entry)
    return {"rollouts": results}


@app.get("/api/rollouts/{filename}")
def get_rollout(filename: str):
    """Serve a specific rollout JSON file."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = _ROLLOUTS_DIR / filename
    if not filepath.exists() or filepath.suffix != ".json":
        raise HTTPException(status_code=404, detail="Rollout not found")
    resolved = filepath.resolve()
    if not str(resolved).startswith(str(_ROLLOUTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return FileResponse(str(resolved), media_type="application/json")


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
