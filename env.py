"""
env.py — Member A
Core environment class for the Fake News Detection OpenEnv Environment.
Detection logic will be injected in Step 4 (integration phase).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from models import (
    Action,
    ActionType,
    AlertLevel,
    ClassificationLabel,
    EpisodeState,
    Observation,
    StepResult,
)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_STEPS = 10  # Maximum steps per episode before forced termination

# Alert level → numeric severity (used for penalty math)
ALERT_SEVERITY: Dict[AlertLevel, int] = {
    "GREEN": 0,
    "YELLOW": 1,
    "RED": 2,
}

# Correct alert for each label (used in step reward shaping)
LABEL_TO_EXPECTED_ALERT: Dict[ClassificationLabel, AlertLevel] = {
    "real":        "GREEN",
    "fake":        "RED",
    "likely_fake": "RED",
    "suspicious":  "YELLOW",
    "unknown":     "YELLOW",
}


# ─────────────────────────────────────────────
# TASK REGISTRY  (populated by tasks.py in Step 5)
# ─────────────────────────────────────────────

# Each task is a dict with these keys:
# {
#   "task_id"             : str,
#   "post_text"           : str,
#   "ground_truth"        : ClassificationLabel,
#   "ground_truth_alert"  : AlertLevel,
#   "extracted_claims"    : List[str],
#   "sources"             : List[str],   # source names present in the post
#   "virality_risk"       : "low" | "medium" | "high",
# }
TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}  # filled by tasks.py


def register_task(task: Dict[str, Any]) -> None:
    """Register a task into the global task registry."""
    TASK_REGISTRY[task["task_id"]] = task


# ─────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class FakeNewsEnv:
    """
    OpenEnv-compliant environment for Social Media Fake News Detection.

    The agent interacts through step-based reasoning actions:
        analyze_claim  → inspect a specific claim
        check_source   → look up a source's credibility
        cross_verify   → cross-check a claim across the knowledge base
        raise_alert    → set a classification + alert level (RED / YELLOW)
        mark_safe      → mark the post as real/safe (GREEN)

    The episode ends when:
        - Agent calls raise_alert or mark_safe  (terminal actions)
        - step_number reaches MAX_STEPS          (forced termination)
    """

    def __init__(self) -> None:
        self._state: Optional[EpisodeState] = None
        self._task_data: Optional[Dict[str, Any]] = None

        # Detection engine will be attached in Step 4
        # self._detector = FakeNewsDetector()   ← injected after Step 3

    # ──────────────────────────────────────────
    # PUBLIC API — required by OpenEnv spec
    # ──────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> Observation:
        """
        Start a fresh episode for the given task.

        Parameters
        ----------
        task_id : "easy" | "medium" | "hard"

        Returns
        -------
        Observation — the initial observation (no actions taken yet)
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Task '{task_id}' not found. "
                f"Available tasks: {list(TASK_REGISTRY.keys())}"
            )

        task = TASK_REGISTRY[task_id]
        self._task_data = copy.deepcopy(task)

        # Initialise episode state
        self._state = EpisodeState(
            task_id=task_id,
            post_text=task["post_text"],
            ground_truth=task["ground_truth"],
            ground_truth_alert=task["ground_truth_alert"],
            step_number=0,
            max_steps=MAX_STEPS,
            cumulative_reward=0.0,
            actions_taken=[],
            current_label=None,
            current_alert=None,
            confidence=0.0,
            fake_score=0.0,
            done=False,
        )

        return self._build_initial_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process one agent action and advance the episode.

        Parameters
        ----------
        action : Action — the agent's chosen reasoning step

        Returns
        -------
        StepResult — (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        # Increment step counter
        self._state.step_number += 1

        # Log the action in history
        self._state.actions_taken.append(action.model_dump())

        # ── Route to action handler ──────────────────────
        reward, feedback = self._handle_action(action)

        # ── Update cumulative reward ──────────────────────
        self._state.cumulative_reward += reward

        # ── Check terminal conditions ──────────────────────
        is_terminal_action = action.action_type in ("raise_alert", "mark_safe")
        step_limit_reached = self._state.step_number >= self._state.max_steps

        if is_terminal_action or step_limit_reached:
            self._state.done = True

            # Penalty for hitting step limit without a decision
            if step_limit_reached and not is_terminal_action:
                timeout_penalty = -0.3
                reward += timeout_penalty
                self._state.cumulative_reward += timeout_penalty
                feedback += " | ⚠️ Step limit reached without a final decision — penalty applied."

        # ── Build and return result ──────────────────────
        observation = self._build_observation(feedback)
        return StepResult(
            observation=observation,
            reward=round(reward, 4),
            done=self._state.done,
            info={
                "step_number": self._state.step_number,
                "cumulative_reward": round(self._state.cumulative_reward, 4),
                "task_id": self._state.task_id,
                "is_terminal": self._state.done,
            },
        )

    def state(self) -> EpisodeState:
        """
        Return the full internal state of the environment.
        Note: ground_truth is included here — used by the grader, not the agent.

        Returns
        -------
        EpisodeState — deep copy of current state
        """
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._state)

    # ──────────────────────────────────────────
    # ACTION HANDLERS  (stubs — logic injected in Step 4)
    # ──────────────────────────────────────────

    def _handle_action(self, action: Action) -> tuple[float, str]:
        """
        Route action to the correct handler.
        Returns (reward, feedback_string).
        """
        handlers = {
            "analyze_claim":  self._handle_analyze_claim,
            "check_source":   self._handle_check_source,
            "cross_verify":   self._handle_cross_verify,
            "raise_alert":    self._handle_raise_alert,
            "mark_safe":      self._handle_mark_safe,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return -0.1, f"Unknown action type: {action.action_type}"
        return handler(action)

    def _handle_analyze_claim(self, action: Action) -> tuple[float, str]:
        """
        Agent is analyzing a specific claim.
        Stub — detection logic injected in Step 4.
        """
        # Step 4 will:
        #   1. Call detector.check_claim(action.target_claim)
        #   2. Update self._state.fake_score
        #   3. Return appropriate partial reward
        reward = 0.05  # small exploration reward for taking a reasoning step
        feedback = (
            f"Claim analysis noted: '{action.target_claim or 'unspecified'}'. "
            "Detection engine not yet integrated."
        )
        return reward, feedback

    def _handle_check_source(self, action: Action) -> tuple[float, str]:
        """
        Agent is checking a source's credibility.
        Stub — detection logic injected in Step 4.
        """
        # Step 4 will:
        #   1. Call detector.get_source_credibility(action.source_name)
        #   2. Update fake_score signal
        #   3. Return reward based on source relevance
        reward = 0.05
        feedback = (
            f"Source check noted: '{action.source_name or 'unspecified'}'. "
            "Detection engine not yet integrated."
        )
        return reward, feedback

    def _handle_cross_verify(self, action: Action) -> tuple[float, str]:
        """
        Agent is cross-verifying a claim across the knowledge base.
        Stub — detection logic injected in Step 4.
        """
        # Step 4 will:
        #   1. Call detector.cross_verify(action.target_claim)
        #   2. Return known_claims result + reward
        reward = 0.05
        feedback = (
            f"Cross-verification noted for: '{action.target_claim or 'unspecified'}'. "
            "Detection engine not yet integrated."
        )
        return reward, feedback

    def _handle_raise_alert(self, action: Action) -> tuple[float, str]:
        """
        Agent is raising an alert — terminal action.
        Classifies the post as fake/suspicious and sets alert level.
        Stub — grading logic injected in Step 4.
        """
        # Validate required fields
        if action.classification is None or action.alert_level is None:
            return (
                -0.2,
                "raise_alert requires both 'classification' and 'alert_level'. "
                "No decision recorded.",
            )

        # Update state
        self._state.current_label = action.classification
        self._state.current_alert = action.alert_level

        # Step 4 will compute real reward vs ground truth
        reward = 0.0
        feedback = (
            f"Alert raised: {action.alert_level} | "
            f"Classification: {action.classification}. "
            "Final scoring pending detection engine integration."
        )
        return reward, feedback

    def _handle_mark_safe(self, action: Action) -> tuple[float, str]:
        """
        Agent is marking the post as safe/real — terminal action.
        Stub — grading logic injected in Step 4.
        """
        # Default to GREEN + real when marking safe
        label = action.classification or "real"
        alert = action.alert_level or "GREEN"

        self._state.current_label = label
        self._state.current_alert = alert

        # Step 4 will compute real reward vs ground truth
        reward = 0.0
        feedback = (
            f"Post marked safe: {alert} | Classification: {label}. "
            "Final scoring pending detection engine integration."
        )
        return reward, feedback

    # ──────────────────────────────────────────
    # OBSERVATION BUILDERS
    # ──────────────────────────────────────────

    def _build_initial_observation(self) -> Observation:
        """Build the observation returned by reset() — no actions taken yet."""
        task = self._task_data
        return Observation(
            post_text=task["post_text"],
            extracted_claims=task.get("extracted_claims", []),
            source_info={},           # filled by check_source in Step 4
            pattern_flags=[],         # filled by detection engine in Step 4
            virality_risk=task.get("virality_risk", "low"),
            known_claims={},          # filled by cross_verify in Step 4
            current_label=None,
            current_alert=None,
            confidence=0.0,
            fake_score=0.0,
            step_feedback=(
                "Episode started. Analyze the post using available actions: "
                "analyze_claim, check_source, cross_verify, raise_alert, mark_safe."
            ),
            step_number=0,
            done_hint=False,
            available_actions=[
                "analyze_claim",
                "check_source",
                "cross_verify",
                "raise_alert",
                "mark_safe",
            ],
        )

    def _build_observation(self, feedback: str) -> Observation:
        """Build the observation returned after each step()."""
        s = self._state
        task = self._task_data

        # Compute available actions (terminal actions remove investigation options)
        if s.done:
            available: List[ActionType] = []
        else:
            available = ["analyze_claim", "check_source", "cross_verify"]
            # Only offer terminal actions after at least 1 investigation step
            if s.step_number >= 1:
                available += ["raise_alert", "mark_safe"]

        # done_hint fires at 80% of max steps
        done_hint = s.step_number >= int(0.8 * s.max_steps)

        return Observation(
            post_text=task["post_text"],
            extracted_claims=task.get("extracted_claims", []),
            source_info=task.get("source_info", {}),   # updated in Step 4
            pattern_flags=task.get("pattern_flags", []),  # updated in Step 4
            virality_risk=task.get("virality_risk", "low"),
            known_claims=task.get("known_claims", {}),    # updated in Step 4
            current_label=s.current_label,
            current_alert=s.current_alert,
            confidence=s.confidence,
            fake_score=s.fake_score,
            step_feedback=feedback,
            step_number=s.step_number,
            done_hint=done_hint,
            available_actions=available,
        )