"""
models.py — Member A
All Pydantic models for the Fake News Detection OpenEnv Environment.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# ENUMS / LITERALS
# ─────────────────────────────────────────────

# What label the agent can assign to a post
ClassificationLabel = Literal["real", "fake", "likely_fake", "suspicious", "unknown"]

# Alert level the agent can raise
AlertLevel = Literal["GREEN", "YELLOW", "RED"]

# All valid step-based reasoning actions
ActionType = Literal[
    "analyze_claim",
    "check_source",
    "cross_verify",
    "raise_alert",
    "mark_safe",
]


# ─────────────────────────────────────────────
# ACTION MODEL  (what the agent sends each step)
# ─────────────────────────────────────────────

class Action(BaseModel):
    """
    The action the agent takes at each step.

    Fields
    ------
    action_type : one of the 5 reasoning actions
    classification : the label the agent assigns (optional — only required on final actions)
    alert_level   : alert to raise (optional — used with raise_alert / mark_safe)
    reasoning     : free-text explanation from the agent (used for partial scoring)
    target_claim  : specific claim being investigated (optional — used with analyze_claim / cross_verify)
    source_name   : source being checked (optional — used with check_source)
    """

    action_type: ActionType = Field(
        ...,
        description="The reasoning step the agent is performing."
    )
    classification: Optional[ClassificationLabel] = Field(
        default=None,
        description="Agent's classification of the post. Required on raise_alert or mark_safe."
    )
    alert_level: Optional[AlertLevel] = Field(
        default=None,
        description="Alert level to set. Required on raise_alert or mark_safe."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's explanation for this action step."
    )
    target_claim: Optional[str] = Field(
        default=None,
        description="The specific claim the agent is analyzing or cross-verifying."
    )
    source_name: Optional[str] = Field(
        default=None,
        description="The source name the agent is checking credibility for."
    )


# ─────────────────────────────────────────────
# OBSERVATION MODEL  (what the env returns each step)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """
    Everything the agent can observe after each action.

    Fields
    ------
    post_text         : the original social media post
    extracted_claims  : list of claims extracted from the post
    source_info       : credibility info for sources mentioned in the post
    pattern_flags     : list of suspicious phrase/pattern flags found
    virality_risk     : risk label — "low", "medium", "high"
    known_claims      : which extracted claims are in the knowledge base
    current_label     : current classification label in env state
    current_alert     : current alert level in env state
    confidence        : current confidence score 0.0–1.0
    fake_score        : multi-signal fake score 0.0–1.0
    step_feedback     : natural language feedback on the last action
    step_number       : current step index
    done_hint         : True if env thinks episode should end soon
    available_actions : which action types are still meaningful to call
    """

    post_text: str = Field(..., description="The full social media post text.")

    extracted_claims: List[str] = Field(
        default_factory=list,
        description="Claims extracted from the post."
    )
    source_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source name → credibility dict (score, tier, known)."
    )
    pattern_flags: List[str] = Field(
        default_factory=list,
        description="Suspicious language patterns detected in the post."
    )
    virality_risk: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Estimated virality risk of the post."
    )
    known_claims: Dict[str, bool] = Field(
        default_factory=dict,
        description="claim → True if found in knowledge base, False if not."
    )
    current_label: Optional[ClassificationLabel] = Field(
        default=None,
        description="The agent's current classification label."
    )
    current_alert: Optional[AlertLevel] = Field(
        default=None,
        description="The current alert level set in the environment."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent's current confidence in its classification."
    )
    fake_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Multi-signal fake score computed by detection engine."
    )
    step_feedback: str = Field(
        default="",
        description="Feedback message from the environment for the last step."
    )
    step_number: int = Field(
        default=0,
        description="Current step index (0-based)."
    )
    done_hint: bool = Field(
        default=False,
        description="True when the env recommends ending the episode."
    )
    available_actions: List[ActionType] = Field(
        default_factory=lambda: [
            "analyze_claim",
            "check_source",
            "cross_verify",
            "raise_alert",
            "mark_safe",
        ],
        description="Action types still valid at this step."
    )


# ─────────────────────────────────────────────
# STEP RESULT MODEL  (returned by step())
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """
    The full result returned by env.step(action).

    Fields
    ------
    observation : updated observation after the action
    reward      : immediate reward for this step (can be negative)
    done        : True if episode is complete
    info        : extra debug/metadata dict
    """

    observation: Observation
    reward: float = Field(
        ...,
        description="Immediate step reward. Range roughly -1.0 to +1.0."
    )
    done: bool = Field(
        ...,
        description="Whether the episode has ended."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata: cumulative reward, error messages, grader hints, etc."
    )


# ─────────────────────────────────────────────
# EPISODE STATE MODEL  (returned by state())
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
    """
    Full internal state of the environment (returned by state()).

    Fields
    ------
    task_id           : which task is loaded ("easy", "medium", "hard")
    post_text         : the original post
    ground_truth      : the true label (hidden from agent, used by grader)
    ground_truth_alert: the correct alert level
    step_number       : how many steps have been taken
    max_steps         : step limit before forced termination
    cumulative_reward : total reward accumulated so far
    actions_taken     : history of all actions taken this episode
    current_label     : current label set by agent
    current_alert     : current alert set by agent
    confidence        : current confidence
    fake_score        : last computed fake score
    done              : whether episode is complete
    """

    task_id: str = Field(..., description="Task difficulty: easy / medium / hard.")
    post_text: str = Field(..., description="The social media post being analyzed.")

    ground_truth: ClassificationLabel = Field(
        ...,
        description="True label — used by grader only, never revealed in observation."
    )
    ground_truth_alert: AlertLevel = Field(
        ...,
        description="Correct alert level — used by grader only."
    )

    step_number: int = Field(default=0)
    max_steps: int = Field(default=10)
    cumulative_reward: float = Field(default=0.0)

    actions_taken: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of all actions taken this episode."
    )

    current_label: Optional[ClassificationLabel] = Field(default=None)
    current_alert: Optional[AlertLevel] = Field(default=None)
    confidence: float = Field(default=0.0)
    fake_score: float = Field(default=0.0)
    done: bool = Field(default=False)
