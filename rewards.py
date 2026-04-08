"""
Reward system for the FakeNews Detection Environment.
Provides partial rewards, penalties, and continuous scoring signals.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from models import Label, AlertLevel, Reward, EnvState, Action, ActionType


# ─────────────────────────── Constants ───────────────────────────

LABEL_COMPATIBILITY: Dict[Tuple[Label, Label], float] = {
    # (ground_truth, predicted): compatibility score
    (Label.FAKE, Label.FAKE): 1.0,
    (Label.FAKE, Label.LIKELY_FAKE): 0.5,
    (Label.FAKE, Label.SUSPICIOUS): 0.3,
    (Label.FAKE, Label.UNKNOWN): 0.0,
    (Label.FAKE, Label.REAL): -1.0,

    (Label.LIKELY_FAKE, Label.LIKELY_FAKE): 1.0,
    (Label.LIKELY_FAKE, Label.FAKE): 0.6,
    (Label.LIKELY_FAKE, Label.SUSPICIOUS): 0.5,
    (Label.LIKELY_FAKE, Label.UNKNOWN): 0.1,
    (Label.LIKELY_FAKE, Label.REAL): -0.8,

    (Label.SUSPICIOUS, Label.SUSPICIOUS): 1.0,
    (Label.SUSPICIOUS, Label.LIKELY_FAKE): 0.7,
    (Label.SUSPICIOUS, Label.UNKNOWN): 0.3,
    (Label.SUSPICIOUS, Label.FAKE): 0.2,
    (Label.SUSPICIOUS, Label.REAL): -0.5,

    (Label.REAL, Label.REAL): 1.0,
    (Label.REAL, Label.SUSPICIOUS): -0.3,
    (Label.REAL, Label.LIKELY_FAKE): -0.7,
    (Label.REAL, Label.FAKE): -1.0,
    (Label.REAL, Label.UNKNOWN): 0.0,

    (Label.UNKNOWN, Label.UNKNOWN): 0.5,
    (Label.UNKNOWN, Label.SUSPICIOUS): 0.4,
    (Label.UNKNOWN, Label.REAL): 0.2,
    (Label.UNKNOWN, Label.LIKELY_FAKE): 0.2,
    (Label.UNKNOWN, Label.FAKE): 0.1,
}

ALERT_COMPATIBILITY: Dict[Tuple[AlertLevel, AlertLevel], float] = {
    (AlertLevel.RED, AlertLevel.RED): 1.0,
    (AlertLevel.RED, AlertLevel.YELLOW): 0.3,
    (AlertLevel.RED, AlertLevel.GREEN): -1.0,

    (AlertLevel.YELLOW, AlertLevel.YELLOW): 1.0,
    (AlertLevel.YELLOW, AlertLevel.RED): 0.4,
    (AlertLevel.YELLOW, AlertLevel.GREEN): -0.5,

    (AlertLevel.GREEN, AlertLevel.GREEN): 1.0,
    (AlertLevel.GREEN, AlertLevel.YELLOW): -0.3,
    (AlertLevel.GREEN, AlertLevel.RED): -1.0,
}

# Action value at each step (partial progress signal)
ACTION_STEP_REWARDS: Dict[ActionType, float] = {
    ActionType.ANALYZE_CLAIM: 0.08,
    ActionType.CHECK_SOURCE: 0.06,
    ActionType.CROSS_VERIFY: 0.10,
    ActionType.RAISE_ALERT: 0.0,   # Evaluated separately
    ActionType.MARK_SAFE: 0.0,     # Evaluated separately
}

# Penalties
REPEATED_ACTION_PENALTY = -0.05
UNNECESSARY_STEP_PENALTY = -0.03
FALSE_POSITIVE_PENALTY = -0.40    # Raising RED when GREEN
FALSE_NEGATIVE_PENALTY = -0.50    # Marking safe when RED
WRONG_ALERT_PENALTY = -0.20


# ─────────────────────────── Reward Calculator ───────────────────────────

class RewardCalculator:
    """
    Calculates step-level and episode-level rewards.
    All calculations are deterministic.
    """

    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.ground_truth_label: Label = task["ground_truth_label"]
        self.ground_truth_alert: AlertLevel = task["ground_truth_alert"]
        self.max_steps: int = task["max_steps"]
        self.action_history: List[ActionType] = []

    def step_reward(
        self,
        action: Action,
        state: EnvState,
        action_result: Dict[str, Any],
    ) -> Reward:
        """
        Calculate reward for a single step.
        Provides partial progress signals throughout the episode.
        """
        base_reward = 0.0
        detection_accuracy = 0.0
        alert_correctness = 0.0
        efficiency = 0.0
        confidence_cal = 0.0
        explanation_parts = []

        # ── 1. Base action reward (partial progress) ──
        action_value = ACTION_STEP_REWARDS.get(action.action_type, 0.0)

        # Penalize repeated identical actions
        recent = self.action_history[-2:] if len(self.action_history) >= 2 else []
        if all(a == action.action_type for a in recent):
            action_value += REPEATED_ACTION_PENALTY
            explanation_parts.append("Penalized for repeating the same action.")
        else:
            explanation_parts.append(f"Step reward for {action.action_type.value}: +{action_value:.2f}")

        self.action_history.append(action.action_type)
        base_reward += action_value

        # ── 2. Quality of action result ──
        if action.action_type == ActionType.ANALYZE_CLAIM:
            if action_result.get("kb_verdict"):
                base_reward += 0.05
                explanation_parts.append("Claim found in knowledge base: +0.05")
            if action_result.get("patterns_found"):
                n_patterns = len(action_result["patterns_found"])
                pattern_bonus = min(0.03 * n_patterns, 0.12)
                base_reward += pattern_bonus
                explanation_parts.append(f"Detected {n_patterns} fake patterns: +{pattern_bonus:.2f}")

        elif action.action_type == ActionType.CHECK_SOURCE:
            credibility = action_result.get("credibility", 0.5)
            tier = action_result.get("tier", "unknown")
            if tier in ("misinformation", "fake_news_site", "conspiracy"):
                base_reward += 0.08
                explanation_parts.append(f"Identified low-credibility source ({tier}): +0.08")
            elif tier == "anonymous":
                base_reward += 0.04
                explanation_parts.append("Identified anonymous source: +0.04")
            elif credibility > 0.85:
                base_reward += 0.03
                explanation_parts.append("High-credibility source identified: +0.03")

        elif action.action_type == ActionType.CROSS_VERIFY:
            if action_result.get("verified"):
                base_reward += 0.06
                explanation_parts.append("Successful cross-verification: +0.06")
            if action_result.get("contradiction_found"):
                base_reward += 0.08
                explanation_parts.append("Contradiction found — strong evidence: +0.08")

        # ── 3. Final verdict actions ──
        elif action.action_type in (ActionType.RAISE_ALERT, ActionType.MARK_SAFE):
            predicted_label = action.final_label
            predicted_alert = self._infer_alert_from_action(action)

            if predicted_label is not None:
                label_score = LABEL_COMPATIBILITY.get(
                    (self.ground_truth_label, predicted_label), 0.0
                )
                detection_accuracy = label_score * 0.4
                base_reward += detection_accuracy
                explanation_parts.append(
                    f"Label accuracy ({self.ground_truth_label.value} vs {predicted_label.value}): "
                    f"{detection_accuracy:+.2f}"
                )

            if predicted_alert is not None:
                alert_score = ALERT_COMPATIBILITY.get(
                    (self.ground_truth_alert, predicted_alert), 0.0
                )
                alert_correctness = alert_score * 0.3
                base_reward += alert_correctness
                explanation_parts.append(
                    f"Alert accuracy ({self.ground_truth_alert.value} vs {predicted_alert.value}): "
                    f"{alert_correctness:+.2f}"
                )

                # Severe penalties
                if (self.ground_truth_alert == AlertLevel.GREEN and
                        predicted_alert == AlertLevel.RED):
                    base_reward += FALSE_POSITIVE_PENALTY
                    explanation_parts.append(f"False positive penalty: {FALSE_POSITIVE_PENALTY}")
                elif (self.ground_truth_alert == AlertLevel.RED and
                      predicted_alert == AlertLevel.GREEN):
                    base_reward += FALSE_NEGATIVE_PENALTY
                    explanation_parts.append(f"False negative penalty: {FALSE_NEGATIVE_PENALTY}")

            # Confidence calibration
            if action.confidence is not None:
                confidence_cal = self._calibrate_confidence(
                    action.confidence, detection_accuracy, alert_correctness
                )
                base_reward += confidence_cal
                explanation_parts.append(
                    f"Confidence calibration: {confidence_cal:+.2f}"
                )

        # ── 4. Efficiency ──
        step_fraction = state.step_number / self.max_steps
        if step_fraction > 0.8 and action.action_type not in (
            ActionType.RAISE_ALERT, ActionType.MARK_SAFE
        ):
            efficiency = UNNECESSARY_STEP_PENALTY
            base_reward += efficiency
            explanation_parts.append(f"Late-step inefficiency penalty: {efficiency}")

        # Clamp
        total = max(-1.0, min(1.0, base_reward))

        return Reward(
            total=total,
            detection_accuracy=detection_accuracy,
            alert_correctness=alert_correctness,
            efficiency=efficiency,
            confidence_calibration=confidence_cal,
            explanation=" | ".join(explanation_parts),
        )

    def _infer_alert_from_action(self, action: Action) -> Optional[AlertLevel]:
        """Infer alert level from action type and final label."""
        if action.action_type == ActionType.MARK_SAFE:
            return AlertLevel.GREEN
        if action.action_type == ActionType.RAISE_ALERT:
            label = action.final_label
            if label in (Label.FAKE,):
                return AlertLevel.RED
            elif label in (Label.LIKELY_FAKE, Label.SUSPICIOUS):
                return AlertLevel.YELLOW
            elif label == Label.REAL:
                return AlertLevel.GREEN
        return None

    def _calibrate_confidence(
        self,
        confidence: float,
        label_score: float,
        alert_score: float
    ) -> float:
        """
        Reward well-calibrated confidence.
        If the agent is correct and confident → positive reward.
        If the agent is wrong and confident → negative reward (overconfidence penalty).
        If the agent is uncertain (0.4–0.6) regardless of correctness → small neutral reward.
        """
        correctness = (label_score + alert_score) / 2
        is_correct = correctness > 0.0

        if is_correct:
            # Reward proportional to confidence × correctness
            return 0.1 * confidence * correctness
        else:
            # Penalize overconfidence on wrong answers
            return -0.1 * confidence * abs(correctness)

    def episode_bonus(self, state: EnvState) -> float:
        """
        End-of-episode bonus reward.
        Rewards completing with fewer steps than max.
        """
        if not state.done:
            return 0.0
        steps_used = state.step_number
        step_efficiency = 1.0 - (steps_used / self.max_steps)
        return 0.05 * step_efficiency