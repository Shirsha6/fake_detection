"""
FastAPI server for FakeNews Detection OpenEnv environment.
Exposes /reset, /step, /state, /health endpoints per OpenEnv spec.
Compatible with openenv-core validate requirements.
"""
from __future__ import annotations
import os
import sys

# Add parent directory so all modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from models import Action, ActionType, Label
from env import FakeNewsEnv
from tasks import list_tasks, TASKS

# ─────────────────────────── App ───────────────────────────

app = FastAPI(
    title="FakeNews Detection OpenEnv",
    description=(
        "Social Media Fake News Detection Environment with Alert System. "
        "OpenEnv-compliant RL environment for training AI agents on fake news detection."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Environment Registry ───────────────────────────

_envs: Dict[str, FakeNewsEnv] = {}


def _get_env(task_id: str) -> FakeNewsEnv:
    """Get or create environment instance for a task."""
    if task_id not in _envs:
        available = list_tasks()
        if task_id not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id: '{task_id}'. Available: {available}"
            )
        _envs[task_id] = FakeNewsEnv(task_id=task_id)
    return _envs[task_id]


# ─────────────────────────── Request Models ───────────────────────────

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

@app.get("/", tags=["meta"])
async def root():
    """Root — environment info and available tasks."""
    return {
        "name": "FakeNews Detection OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "openenv_compatible": True,
        "available_tasks": list_tasks(),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.get("/health", tags=["meta"])
async def health():
    """Health check endpoint — must return 200."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset", tags=["openenv"])
async def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment.
    OpenEnv spec: POST /reset → returns initial observation.
    Accepts empty body {} — defaults to task_easy.
    """
    task_id = "task_easy"
    if request is not None and request.task_id:
        task_id = request.task_id

    available = list_tasks()
    if task_id not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: '{task_id}'. Available: {available}"
        )

    env = _get_env(task_id)
    result = env.reset()
    return result.model_dump()


@app.post("/step", tags=["openenv"])
async def step(request: StepRequest):
    """
    Execute one step.
    OpenEnv spec: POST /step → returns observation, reward, done, info.
    """
    task_id = request.task_id or "task_easy"

    try:
        env = _get_env(task_id)
    except HTTPException:
        raise

    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    if env._state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode already done. Call POST /reset to start a new episode."
        )

    # Validate action_type
    valid_action_types = [a.value for a in ActionType]
    if request.action_type not in valid_action_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type: '{request.action_type}'. Valid: {valid_action_types}"
        )

    # Validate final_label if provided
    final_label = None
    if request.final_label:
        valid_labels = [l.value for l in Label]
        if request.final_label not in valid_labels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid final_label: '{request.final_label}'. Valid: {valid_labels}"
            )
        final_label = Label(request.final_label)

    action = Action(
        action_type=ActionType(request.action_type),
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


@app.get("/state", tags=["openenv"])
@app.post("/state", tags=["openenv"])
async def get_state(task_id: str = Query(default="task_easy")):
    """
    Return full environment state.
    OpenEnv spec: GET or POST /state → returns current EnvState.
    """
    try:
        env = _get_env(task_id)
    except HTTPException:
        raise

    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    return env.state().model_dump()


@app.get("/tasks", tags=["meta"])
async def get_tasks():
    """List all available tasks with metadata."""
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


# ─────────────────────────── Exception Handlers ───────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ─────────────────────────── Entry Point ───────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main():
    """Main entry point — referenced by [project.scripts] in pyproject.toml."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")