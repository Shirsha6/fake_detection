"""
env.py — 

Actions:
    analyze_claim  — investigate a specific claim
    check_source   — look up source credibility
    cross_verify   — cross-check all claims
    raise_alert    — terminal: classify as fake/suspicious
    mark_safe      — terminal: classify as real/safe

Fully self-contained — only depends on models.py and rewards.py.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from models import (
    Action,
    AlertLevel,
    ClassificationLabel,
    EpisodeState,
    Observation,
    StepResult,
)
from rewards import FakeNewsDetector, RewardComputer

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_STEPS = 10   # Step 4 original value

ALERT_SEVERITY: Dict[AlertLevel, int] = {
    "GREEN": 0,
    "YELLOW": 1,
    "RED": 2,
}

LABEL_TO_EXPECTED_ALERT: Dict[ClassificationLabel, AlertLevel] = {
    "real":        "GREEN",
    "fake":        "RED",
    "likely_fake": "RED",
    "suspicious":  "YELLOW",
    "unknown":     "YELLOW",
}


# ─────────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_task(task: Dict[str, Any]) -> None:
    """Register a task into the global task registry."""
    TASK_REGISTRY[task["task_id"]] = task


# ─────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class FakeNewsEnv:
    """
    OpenEnv-compliant Fake News Detection Environment.

    Implements the three required OpenEnv methods:
        reset(task_id)  → Observation
        step(action)    → StepResult
        state()         → EpisodeState

    No external dependencies beyond models.py and rewards.py.
    """

    def __init__(self) -> None:
        self._state:     Optional[EpisodeState]  = None
        self._task_data: Optional[Dict[str, Any]] = None

        # Core detection + reward engines
        self._detector = FakeNewsDetector()
        self._rewarder = RewardComputer()

        # Per-episode tracking (reset on each reset() call)
        self._analyzed_claims:   List[str]       = []
        self._checked_sources:   List[str]       = []
        self._cross_verified:    bool             = False

        # Live observation caches
        self._live_source_info:   Dict[str, Any]  = {}
        self._live_pattern_flags: List[str]       = []
        self._live_known_claims:  Dict[str, bool] = {}

    # ──────────────────────────────────────────
    # PUBLIC API — required by OpenEnv spec
    # ──────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> Observation:
        """
        Start a fresh episode for the given task.

        Parameters
        ----------
        task_id : str  — one of the registered task ids (easy / medium / hard …)

        Returns
        -------
        Observation  — initial observation (step 0, no actions taken)
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Task '{task_id}' not found. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )

        task = TASK_REGISTRY[task_id]
        self._task_data = copy.deepcopy(task)

        # Reset all per-episode state
        self._analyzed_claims    = []
        self._checked_sources    = []
        self._cross_verified     = False
        self._live_source_info   = {}
        self._live_pattern_flags = []
        self._live_known_claims  = {}

        # Passive pattern detection on reset (no reward, just pre-populate flags)
        pattern_result = self._detector.detect_patterns(task["post_text"])
        self._live_pattern_flags = pattern_result["flags"]

        # Build EpisodeState — handles both old models.py (3 tasks) and
        # upgraded models.py (with extra fields like fake_intensity etc.)
        state_kwargs: Dict[str, Any] = dict(
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

        # Safely add upgraded fields if the model supports them
        # (so this file works with BOTH the old and new models.py)
        try:
            from models import EpisodeState as _ES
            _fields = _ES.model_fields
            if "fake_intensity"             in _fields: state_kwargs["fake_intensity"]             = 0.0
            if "viral_score"                in _fields: state_kwargs["viral_score"]                = 0.0
            if "reports_count"              in _fields: state_kwargs["reports_count"]              = task.get("reports_count", 0)
            if "wait_count"                 in _fields: state_kwargs["wait_count"]                 = 0
            if "virality_penalty_accumulator" in _fields: state_kwargs["virality_penalty_accumulator"] = 0.0
            if "link_scanned"               in _fields: state_kwargs["link_scanned"]               = False
        except Exception:
            pass  # old models.py — skip extra fields

        self._state = EpisodeState(**state_kwargs)

        return self._build_initial_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process one agent action and advance the episode.

        Parameters
        ----------
        action : Action

        Returns
        -------
        StepResult — (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode already done. Call reset().")

        self._state.step_number += 1
        self._state.actions_taken.append(action.model_dump())

        reward, feedback = self._handle_action(action)
        self._state.cumulative_reward += reward

        is_terminal = action.action_type in ("raise_alert", "mark_safe")
        step_limit  = self._state.step_number >= self._state.max_steps

        if is_terminal or step_limit:
            self._state.done = True
            if step_limit and not is_terminal:
                timeout_penalty = -0.3
                reward += timeout_penalty
                self._state.cumulative_reward += timeout_penalty
                feedback += " | Step limit reached without decision — penalty applied."

        observation = self._build_observation(feedback)
        return StepResult(
            observation=observation,
            reward=round(reward, 4),
            done=self._state.done,
            info={
                "step_number":       self._state.step_number,
                "cumulative_reward": round(self._state.cumulative_reward, 4),
                "task_id":           self._state.task_id,
                "is_terminal":       self._state.done,
            },
        )

    def state(self) -> EpisodeState:
        """
        Return the full internal state.
        Includes ground_truth — used by grader, never exposed to agent in obs.

        Returns
        -------
        EpisodeState — deep copy of current state
        """
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._state)

    # ──────────────────────────────────────────
    # ACTION HANDLERS
    # ──────────────────────────────────────────

    def _handle_action(self, action: Action) -> tuple[float, str]:
        """Route action to the correct handler."""
        handlers = {
            "analyze_claim": self._handle_analyze_claim,
            "check_source":  self._handle_check_source,
            "cross_verify":  self._handle_cross_verify,
            "raise_alert":   self._handle_raise_alert,
            "mark_safe":     self._handle_mark_safe,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return -0.1, f"Unknown action type: {action.action_type}"
        return handler(action)

    def _handle_analyze_claim(self, action: Action) -> tuple[float, str]:
        """Check a specific claim against the knowledge base."""
        target       = action.target_claim or ""
        claim_result = self._detector.check_claim(target)

        reward, feedback = self._rewarder.reward_analyze_claim(
            claim_result=claim_result,
            already_analyzed=self._analyzed_claims,
            target_claim=target,
        )

        if target and target not in self._analyzed_claims:
            self._analyzed_claims.append(target)

        if claim_result["found"]:
            self._live_known_claims[target] = True

        self._recompute_fake_score()
        self._update_confidence()
        return reward, feedback

    def _handle_check_source(self, action: Action) -> tuple[float, str]:
        """Look up the credibility of a named source."""
        source        = action.source_name or ""
        source_result = self._detector.get_source_credibility(source)

        reward, feedback = self._rewarder.reward_check_source(
            source_result=source_result,
            already_checked=self._checked_sources,
            source_name=source,
        )

        if source and source not in self._checked_sources:
            self._checked_sources.append(source)
            self._live_source_info[source] = source_result

        self._recompute_fake_score()
        self._update_confidence()
        return reward, feedback

    def _handle_cross_verify(self, action: Action) -> tuple[float, str]:
        """Cross-verify all extracted claims at once."""
        target    = action.target_claim or ""
        extracted = self._task_data.get("extracted_claims", [])

        cross_result = self._detector.cross_verify(target, extracted)

        reward, feedback = self._rewarder.reward_cross_verify(
            cross_result=cross_result,
            already_cross_verified=self._cross_verified,
        )

        self._cross_verified = True

        for r in cross_result.get("results", []):
            self._live_known_claims[r["claim"]] = r["found"]

        self._recompute_fake_score()
        self._update_confidence()
        return reward, feedback

    def _handle_raise_alert(self, action: Action) -> tuple[float, str]:
        """Terminal action — classify as fake/suspicious and raise alert."""
        if action.classification is None or action.alert_level is None:
            return -0.2, "raise_alert requires both 'classification' and 'alert_level'."

        self._state.current_label = action.classification
        self._state.current_alert = action.alert_level

        if action.reasoning:
            self._state.confidence = min(self._state.confidence + 0.05, 1.0)

        reward, feedback = self._rewarder.reward_terminal(
            agent_label=action.classification,
            agent_alert=action.alert_level,
            ground_truth=self._state.ground_truth,
            ground_truth_alert=self._state.ground_truth_alert,
            step_number=self._state.step_number,
            max_steps=self._state.max_steps,
            confidence=self._state.confidence,
        )
        return reward, feedback

    def _handle_mark_safe(self, action: Action) -> tuple[float, str]:
        """Terminal action — classify as real/safe."""
        label: ClassificationLabel = action.classification or "real"
        alert: AlertLevel          = action.alert_level or "GREEN"

        self._state.current_label = label
        self._state.current_alert = alert

        reward, feedback = self._rewarder.reward_terminal(
            agent_label=label,
            agent_alert=alert,
            ground_truth=self._state.ground_truth,
            ground_truth_alert=self._state.ground_truth_alert,
            step_number=self._state.step_number,
            max_steps=self._state.max_steps,
            confidence=self._state.confidence,
        )
        return reward, feedback

    # ──────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────

    def _recompute_fake_score(self) -> None:
        """Recompute multi-signal fake score after every investigation step."""
        task = self._task_data

        # Check if reward_terminal accepts reports_count (upgraded rewards.py)
        try:
            result = self._detector.compute_fake_score(
                post_text=task["post_text"],
                extracted_claims=self._analyzed_claims or task.get("extracted_claims", []),
                sources=self._checked_sources or task.get("sources", []),
                virality_risk=task.get("virality_risk", "low"),
                reports_count=task.get("reports_count", 0),
            )
        except TypeError:
            # Old rewards.py without reports_count param
            result = self._detector.compute_fake_score(
                post_text=task["post_text"],
                extracted_claims=self._analyzed_claims or task.get("extracted_claims", []),
                sources=self._checked_sources or task.get("sources", []),
                virality_risk=task.get("virality_risk", "low"),
            )

        self._state.fake_score    = result["fake_score"]
        self._live_pattern_flags  = result["flags"]

        # Set fake_intensity if the state model supports it
        if hasattr(self._state, "fake_intensity"):
            self._state.fake_intensity = result["fake_score"]

        classification = self._detector.classify(
            fake_score=result["fake_score"],
            claim_results=result["claim_results"],
            source_info=result["source_info"],
            flags=result["flags"],
        )
        self._state.confidence = classification["confidence"]

    def _update_confidence(self) -> None:
        """Grow confidence as more signals are gathered."""
        signals = (
            len(self._analyzed_claims) +
            len(self._checked_sources) +
            (1 if self._cross_verified else 0)
        )
        investigation_confidence = min(signals * 0.08, 0.90)
        self._state.confidence   = round(
            max(self._state.confidence, investigation_confidence), 3
        )

    # ──────────────────────────────────────────
    # OBSERVATION BUILDERS
    # ──────────────────────────────────────────

    def _build_initial_observation(self) -> Observation:
        """Build the observation returned by reset() — no actions taken yet."""
        task = self._task_data

        # Base kwargs that every version of Observation supports
        obs_kwargs: Dict[str, Any] = dict(
            post_text=task["post_text"],
            extracted_claims=task.get("extracted_claims", []),
            source_info={},
            pattern_flags=self._live_pattern_flags,
            virality_risk=task.get("virality_risk", "low"),
            known_claims={},
            current_label=None,
            current_alert=None,
            confidence=0.0,
            fake_score=0.0,
            step_feedback=(
                "Episode started. Use analyze_claim, check_source, "
                "cross_verify to investigate. Then raise_alert or mark_safe."
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

        # Add upgraded Observation fields only if the model supports them
        self._add_upgraded_obs_fields(obs_kwargs, step=0)

        return Observation(**obs_kwargs)

    def _build_observation(self, feedback: str) -> Observation:
        """Build the observation returned after each step()."""
        s    = self._state
        task = self._task_data

        if s.done:
            available = []
        else:
            available = ["analyze_claim", "check_source", "cross_verify"]
            if s.step_number >= 1:
                available += ["raise_alert", "mark_safe"]

        done_hint = s.step_number >= int(0.8 * s.max_steps)

        obs_kwargs: Dict[str, Any] = dict(
            post_text=task["post_text"],
            extracted_claims=task.get("extracted_claims", []),
            source_info=self._live_source_info,
            pattern_flags=self._live_pattern_flags,
            virality_risk=task.get("virality_risk", "low"),
            known_claims=self._live_known_claims,
            current_label=s.current_label,
            current_alert=s.current_alert,
            confidence=s.confidence,
            fake_score=s.fake_score,
            step_feedback=feedback,
            step_number=s.step_number,
            done_hint=done_hint,
            available_actions=available,
        )

        self._add_upgraded_obs_fields(obs_kwargs, step=s.step_number)

        return Observation(**obs_kwargs)

    def _add_upgraded_obs_fields(
        self, obs_kwargs: Dict[str, Any], step: int
    ) -> None:
        """
        Safely add upgraded Observation fields (fake_intensity, viral_score,
        reports_count, safe_share_warning, scan_result, fields_unlocked)
        only if the current models.py Observation class supports them.
        This ensures env.py works with BOTH old and upgraded models.py.
        """
        try:
            from models import Observation as _Obs
            _fields = _Obs.model_fields

            if "fake_intensity" in _fields:
                obs_kwargs["fake_intensity"] = getattr(self._state, "fake_intensity", 0.0) if self._state else 0.0

            if "viral_score" in _fields:
                obs_kwargs["viral_score"] = getattr(self._state, "viral_score", 0.0) if self._state else 0.0

            if "reports_count" in _fields:
                obs_kwargs["reports_count"] = getattr(self._state, "reports_count", 0) if self._state else 0

            if "safe_share_warning" in _fields:
                fake_score = getattr(self._state, "fake_score", 0.0) if self._state else 0.0
                obs_kwargs["safe_share_warning"] = fake_score >= 0.40

            if "scan_result" in _fields:
                obs_kwargs["scan_result"] = None

            if "fields_unlocked" in _fields:
                unlocked = ["post_text", "virality_risk"]
                if step >= 1:
                    unlocked.append("pattern_flags")
                if self._live_known_claims:
                    unlocked.append("known_claims")
                if self._live_source_info:
                    unlocked.append("source_info")
                obs_kwargs["fields_unlocked"] = unlocked

        except Exception:
            pass  # old models.py — no upgraded fields, skip silently