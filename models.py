"""
Pydantic models for the FakeNews Detection OpenEnv environment.
Typed Observation, Action, Reward models as required by OpenEnv spec.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────── Enums ───────────────────────────

class Label(str, Enum):
    REAL = "real"
    FAKE = "fake"
    LIKELY_FAKE = "likely_fake"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"


class AlertLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class ActionType(str, Enum):
    ANALYZE_CLAIM = "analyze_claim"
    CHECK_SOURCE = "check_source"
    CROSS_VERIFY = "cross_verify"
    RAISE_ALERT = "raise_alert"
    MARK_SAFE = "mark_safe"


# ─────────────────────────── Action ───────────────────────────

class Action(BaseModel):
    """Agent action model."""
    action_type: ActionType = Field(
        ...,
        description="Type of reasoning step the agent takes."
    )
    target: Optional[str] = Field(
        None,
        description="Target of the action (e.g. claim text, source name, or alert level)."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Agent's reasoning for this action (used in confidence calibration)."
    )
    final_label: Optional[Label] = Field(
        None,
        description="Final label if action_type is raise_alert or mark_safe."
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this action (0.0–1.0)."
    )


# ─────────────────────────── Observation ───────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""
    post_id: str = Field(..., description="Unique identifier for the social media post.")
    post_text: str = Field(..., description="Full text of the social media post.")
    task_description: str = Field(..., description="What the agent needs to accomplish.")
    step_number: int = Field(..., description="Current step number in the episode.")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode.")

    # Evidence accumulated so far
    claims_extracted: List[str] = Field(
        default_factory=list,
        description="Claims extracted so far by the agent."
    )
    sources_checked: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources checked so far with credibility info."
    )
    patterns_detected: List[str] = Field(
        default_factory=list,
        description="Fake-news linguistic patterns detected."
    )
    cross_verifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Cross-verification results from knowledge base."
    )

    # Signals
    current_fake_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated fake score so far (0=real, 1=fake)."
    )
    current_alert: AlertLevel = Field(
        AlertLevel.GREEN,
        description="Current alert level based on evidence gathered."
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Actions available at this step."
    )
    message: str = Field(
        "",
        description="System message/hint for the agent."
    )


# ─────────────────────────── Reward ───────────────────────────

class Reward(BaseModel):
    """Reward model with breakdown."""
    total: float = Field(..., ge=-1.0, le=1.0, description="Total reward for this step.")
    detection_accuracy: float = Field(0.0, description="Reward for correct label detection.")
    alert_correctness: float = Field(0.0, description="Reward for correct alert level.")
    efficiency: float = Field(0.0, description="Penalty for wasting steps.")
    confidence_calibration: float = Field(0.0, description="Reward for calibrated confidence.")
    explanation: str = Field("", description="Human-readable reward breakdown.")


# ─────────────────────────── State ───────────────────────────

class EnvState(BaseModel):
    """Full environment state (returned by state() endpoint)."""
    task_id: str
    task_name: str
    post_id: str
    post_text: str
    ground_truth_label: Label
    ground_truth_alert: AlertLevel
    step_number: int
    max_steps: int
    done: bool
    claims_extracted: List[str] = Field(default_factory=list)
    sources_checked: List[Dict[str, Any]] = Field(default_factory=list)
    patterns_detected: List[str] = Field(default_factory=list)
    cross_verifications: List[Dict[str, Any]] = Field(default_factory=list)
    current_fake_score: float = 0.0
    current_alert: AlertLevel = AlertLevel.GREEN
    final_label: Optional[Label] = None
    final_alert: Optional[AlertLevel] = None
    episode_rewards: List[float] = Field(default_factory=list)
    total_reward: float = 0.0
    agent_actions: List[Dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────── Step Result ───────────────────────────

class StepResult(BaseModel):
    """Result returned after env.step()."""
    observation: Observation
    reward: float
    reward_details: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result returned after env.reset()."""
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)