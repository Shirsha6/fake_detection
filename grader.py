"""
Deterministic grader for FakeNews Detection Environment.
Scores agent performance on each task 0.0–1.0.
All grading logic is deterministic and reproducible.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from models import Label, AlertLevel, EnvState


# Compatibility maps defined inline (deterministic, no external dependency)
LABEL_COMPAT: Dict = {
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

ALERT_COMPAT: Dict = {
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


class TaskGrader:
    """
    Deterministic grader for a completed task episode.
    Outputs a score in [0.0, 1.0].
    """

    def grade(self, state: EnvState, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grade a completed episode.
        Returns a dict with score (0.0–1.0) and breakdown.
        """
        ground_truth_label: Label = task["ground_truth_label"]
        ground_truth_alert: AlertLevel = task["ground_truth_alert"]
        required_actions: List[str] = task.get("required_actions", [])
        max_steps: int = task["max_steps"]
        difficulty: str = task["difficulty"]

        score_components: Dict[str, float] = {}
        breakdown: List[str] = []

        # ── 1. Final label accuracy (40%) ──
        final_label = state.final_label
        if final_label is not None:
            label_compat = LABEL_COMPAT.get((ground_truth_label, final_label), 0.0)
            label_score = max(0.0, label_compat)  # no negative contribution to total
            score_components["label_accuracy"] = label_score * 0.40
            breakdown.append(
                f"Label: {final_label.value} (truth={ground_truth_label.value}) "
                f"→ compat={label_compat:.2f}, score={score_components['label_accuracy']:.3f}"
            )
        else:
            score_components["label_accuracy"] = 0.0
            breakdown.append("Label: Not provided → 0.0")

        # ── 2. Alert level accuracy (25%) ──
        final_alert = state.final_alert
        if final_alert is not None:
            alert_compat = ALERT_COMPAT.get((ground_truth_alert, final_alert), 0.0)
            alert_score = max(0.0, alert_compat)
            score_components["alert_accuracy"] = alert_score * 0.25
            breakdown.append(
                f"Alert: {final_alert.value} (truth={ground_truth_alert.value}) "
                f"→ compat={alert_compat:.2f}, score={score_components['alert_accuracy']:.3f}"
            )
        else:
            score_components["alert_accuracy"] = 0.0
            breakdown.append("Alert: Not provided → 0.0")

        # ── 3. Evidence gathering quality (20%) ──
        evidence_score = self._grade_evidence(state, task)
        score_components["evidence_quality"] = evidence_score * 0.20
        breakdown.append(f"Evidence quality: {evidence_score:.2f} → score={score_components['evidence_quality']:.3f}")

        # ── 4. Required action coverage (10%) ──
        action_coverage = self._grade_action_coverage(state, required_actions)
        score_components["action_coverage"] = action_coverage * 0.10
        breakdown.append(f"Action coverage: {action_coverage:.2f} → score={score_components['action_coverage']:.3f}")

        # ── 5. Efficiency (5%) ──
        steps_used = state.step_number
        efficiency = max(0.0, 1.0 - (steps_used / max_steps))
        score_components["efficiency"] = efficiency * 0.05
        breakdown.append(f"Efficiency ({steps_used}/{max_steps} steps): {score_components['efficiency']:.3f}")

        # ── Total ──
        total = sum(score_components.values())

        # Difficulty multiplier: hard tasks that are well-solved get bonus
        difficulty_multiplier = {"easy": 1.0, "medium": 1.0, "hard": 1.05}[difficulty]
        total = min(1.0, total * difficulty_multiplier)

        # Severe error check: only penalize if BOTH label AND alert are completely wrong
        # (not just one of them — this prevents over-penalizing partial mistakes)
        if (final_label == Label.REAL and ground_truth_label == Label.FAKE) and \
           (final_alert == AlertLevel.GREEN and ground_truth_alert == AlertLevel.RED):
            total = min(total, 0.25)
            breakdown.append("SEVERE ERROR: Marked clearly fake post as safe → capped at 0.25")
        elif final_label == Label.REAL and ground_truth_label == Label.FAKE:
            total = min(total, 0.35)
            breakdown.append("ERROR: Wrong label (real vs fake) → capped at 0.35")

        # Clamp
        total = max(0.0, min(1.0, total))

        passed = total >= task.get("pass_threshold", 0.60)

        return {
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "difficulty": difficulty,
            "score": round(total, 4),
            "passed": passed,
            "pass_threshold": task.get("pass_threshold", 0.60),
            "components": {k: round(v, 4) for k, v in score_components.items()},
            "breakdown": breakdown,
            "ground_truth": {
                "label": ground_truth_label.value,
                "alert": ground_truth_alert.value,
            },
            "agent_output": {
                "label": final_label.value if final_label else None,
                "alert": final_alert.value if final_alert else None,
            },
        }

    def _grade_evidence(self, state: EnvState, task: Dict[str, Any]) -> float:
        """
        Score the quality of evidence gathered (0.0–1.0).
        Checks if agent found key claims, checked key source, detected patterns.
        """
        score = 0.0
        checks = 0

        key_claims = task.get("key_claims", [])
        if key_claims:
            found_claims = len(state.claims_extracted)
            claim_ratio = min(1.0, found_claims / max(1, len(key_claims)))
            score += claim_ratio * 0.4
            checks += 1

        key_source = task.get("key_source", "")
        if key_source:
            source_checked = any(
                key_source.lower() in str(s).lower()
                for s in state.sources_checked
            )
            score += (0.3 if source_checked else 0.0)
            checks += 1

        expected_patterns = task.get("expected_patterns", [])
        if expected_patterns:
            detected_count = len(state.patterns_detected)
            pattern_ratio = min(1.0, detected_count / max(1, len(expected_patterns)))
            score += pattern_ratio * 0.3
            checks += 1

        return score if checks > 0 else 0.5

    def _grade_action_coverage(
        self,
        state: EnvState,
        required_actions: List[str]
    ) -> float:
        """Check if agent used all required action types."""
        if not required_actions:
            return 1.0
        used_types = set()
        for action_dict in state.agent_actions:
            atype = action_dict.get("action_type", "")
            if atype:
                used_types.add(atype)
        required_set = set(required_actions)
        if not required_set:
            return 1.0
        covered = len(used_types & required_set)
        return covered / len(required_set)


# ─────────────────────────── Grader singleton ───────────────────────────

_grader = TaskGrader()


def grade_episode(state: EnvState, task: Dict[str, Any]) -> Dict[str, Any]:
    """Grade a completed episode. Returns score dict with 'score' key (0.0–1.0)."""
    return _grader.grade(state, task)