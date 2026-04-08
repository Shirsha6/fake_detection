"""
FastAPI server for FakeNews Detection OpenEnv environment.
Exposes /reset, /step, /state endpoints per OpenEnv spec.
"""
from __future__ import annotations
import os
import sys

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from models import Action, ActionType, Label
from env import FakeNewsEnv
from tasks import list_tasks


# ─────────────────────────── App ───────────────────────────

app = FastAPI(
    title="FakeNews Detection OpenEnv",
    description=(
        "Social Media Fake News Detection Environment with Alert System. "
        "An OpenEnv-compliant RL environment for training and evaluating "
        "AI agents on multi-signal fake news detection."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── State ───────────────────────────

# One env instance per task_id
_envs: Dict[str, FakeNewsEnv] = {}


def _get_env(task_id: str) -> FakeNewsEnv:
    if task_id not in _envs:
        _envs[task_id] = FakeNewsEnv(task_id=task_id)
    return _envs[task_id]


# ─────────────────────────── Request/Response Models ───────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None
    reasoning: Optional[str] = None
    final_label: Optional[str] = None
    confidence: Optional[float] = None
    task_id: Optional[str] = "task_easy"


# ─────────────────────────── Endpoints ───────────────────────────

@app.get("/")
async def root():
    """Health check and environment info."""
    return {
        "name": "FakeNews Detection OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "available_tasks": list_tasks(),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment for a given task.
    Returns initial observation.
    """
    task_id = (request.task_id if request else None) or "task_easy"
    if task_id not in list_tasks():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {task_id}. Available: {list_tasks()}"
        )
    env = _get_env(task_id)
    result = env.reset()
    return result.model_dump()


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute one step in the environment.
    Returns observation, reward, done, info.
    """
    task_id = request.task_id or "task_easy"
    env = _get_env(task_id)

    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    # Parse action
    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type: {request.action_type}. "
                   f"Valid: {[a.value for a in ActionType]}"
        )

    final_label = None
    if request.final_label:
        try:
            final_label = Label(request.final_label)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid final_label: {request.final_label}. "
                       f"Valid: {[l.value for l in Label]}"
            )

    action = Action(
        action_type=action_type,
        target=request.target,
        reasoning=request.reasoning,
        final_label=final_label,
        confidence=request.confidence,
    )

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result.model_dump()


@app.post("/state")
async def get_state(task_id: str = Query(default="task_easy")):
    """Return current full environment state."""
    env = _get_env(task_id)
    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    return env.state().model_dump()


@app.get("/state")
async def get_state_get(task_id: str = Query(default="task_easy")):
    """Return current full environment state (GET)."""
    env = _get_env(task_id)
    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    return env.state().model_dump()


@app.get("/tasks")
async def get_tasks():
    """List all available tasks."""
    from tasks import TASKS
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "task_name": t["task_name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_steps": t["max_steps"],
                "pass_threshold": t["pass_threshold"],
            }
            for t in TASKS.values()
        ]
    }

def main():
    """Entry point for OpenEnv server."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()