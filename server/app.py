"""
server.py — FastAPI server exposing the Print Farm Scheduler environment
for the OpenEnv hackathon automated validator.

Endpoints:
    POST /reset   — reset environment, returns Observation
    POST /step    — step with an action, returns (Observation, Reward, done, info)
    GET  /state   — returns current state dict
    GET  /health  — liveness probe
    GET  /        — basic info page
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is on the import path
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import PrintFarmEnv, Observation, Action, Reward
from tasks import TASKS


# ── Request/Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42

class StepRequest(BaseModel):
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict

class ResetResponse(BaseModel):
    observation: Observation


# ── Global Environment State ─────────────────────────────────────────────────

_envs: dict[str, PrintFarmEnv] = {}


# ── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-initialize environments for all tasks
    for task_name, task in TASKS.items():
        _envs[task_name] = PrintFarmEnv(
            seed=task.seed, difficulty=task_name, max_steps=task.max_steps
        )
        _envs[task_name].reset()
    yield
    _envs.clear()


app = FastAPI(
    title="Print Farm Scheduler — OpenEnv",
    description="3D Print Farm Logistics Scheduling RL Environment",
    version="2.1.1",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "print-farm-scheduler",
        "version": "2.1.1",
        "tasks": list(TASKS.keys()),
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    task_name = req.task
    if task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}",
        )
    task = TASKS[task_name]
    env = PrintFarmEnv(
        seed=req.seed, difficulty=task_name, max_steps=task.max_steps
    )
    obs = env.reset()
    _envs[task_name] = env
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    # Find the active environment (use the most recently reset one)
    env = None
    for task_name in ["hard", "medium", "easy"]:
        if task_name in _envs:
            env = _envs[task_name]
            break

    if env is None:
        raise HTTPException(status_code=400, detail="No environment active. Call /reset first.")

    try:
        obs, reward, done, info = env.step(req.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def get_state():
    for task_name in ["hard", "medium", "easy"]:
        if task_name in _envs:
            return _envs[task_name].state()
    raise HTTPException(status_code=400, detail="No environment active. Call /reset first.")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    """Start the uvicorn server. Used by [project.scripts] entry point."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
