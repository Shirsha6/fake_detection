"""
grader.py — 
Deterministic grader for all 3 tasks.
Scores agent performance 0.0–1.0 using EpisodeState.
Never uses randomness — same input always returns same score.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from models import AlertLevel, ClassificationLabel, EpisodeState

# ─────────────────────────────────────────────
# SCORING WEIGHTS
# ─────────────────────────────────────────────

WEIGHT_LABEL        = 0.40   # Was the classification label correct?
WEIGHT_ALERT        = 0.25   # Was the alert level correct?
WEIGHT_EFFICIENCY   = 0.15   # How efficiently did the agent reach a decision?
WEIGHT_INVESTIGATION= 0.10   # Did the agent use investigation actions?
WEIGHT_CONFIDENCE   = 0.10   # Was the confidence calibrated correctly?

# Alert level numeric ranks for distance scoring
ALERT_RANK: Dict[AlertLevel, int] = {
    "GREEN": 0,
    "YELLOW": 1,
    "RED": 2,
}

# Correct alert for each ground truth label
LABEL_TO_CORRECT_ALERT: Dict[ClassificationLabel, AlertLevel] = {
    "real":        "GREEN",
    "fake":        "RED",
    "likely_fake": "RED",
    "suspicious":  "YELLOW",
    "unknown":     "YELLOW",
}


# ═══════════════════════════════════════════════════════════════════
# GRADER CLASS
# ═══════════════════════════════════════════════════════════════════

class FakeNewsGrader:
    """
    Deterministic grader that evaluates a completed episode.

    Input  : EpisodeState (from env.state() after done=True)
    Output : score (float 0.0–1.0), breakdown (dict), explanation (str)
    """

    def grade(self, episode_state: EpisodeState) -> Dict[str, Any]:
        """
        Main grading entry point.

        Parameters
        ----------
        episode_state : EpisodeState — must be a completed episode (done=True)

        Returns
        -------
        dict with keys:
            score        : float 0.0–1.0
            breakdown    : dict of component scores
            explanation  : human-readable scoring explanation
            passed       : bool (score >= 0.5)
        """
        s = episode_state

        # ── Guard: episode must be complete ──────────────────────────
        if not s.done:
            return {
                "score": 0.0,
                "breakdown": {},
                "explanation": "Episode not yet complete — cannot grade.",
                "passed": False,
            }

        # ── Guard: agent must have made a decision ───────────────────
        if s.current_label is None or s.current_alert is None:
            return {
                "score": 0.0,
                "breakdown": {"label": 0.0, "alert": 0.0},
                "explanation": "Agent never made a final classification decision.",
                "passed": False,
            }

        # ── Component scores ─────────────────────────────────────────
        label_score        = self._score_label(s.current_label, s.ground_truth)
        alert_score        = self._score_alert(s.current_alert, s.ground_truth_alert)
        efficiency_score   = self._score_efficiency(s.step_number, s.max_steps, label_score)
        investigation_score= self._score_investigation(s.actions_taken)
        confidence_score   = self._score_confidence(s.confidence, label_score)

        # ── Weighted total ───────────────────────────────────────────
        total = (
            label_score         * WEIGHT_LABEL +
            alert_score         * WEIGHT_ALERT +
            efficiency_score    * WEIGHT_EFFICIENCY +
            investigation_score * WEIGHT_INVESTIGATION +
            confidence_score    * WEIGHT_CONFIDENCE
        )
        total = round(min(max(total, 0.0), 1.0), 4)

        breakdown = {
            "label_score":         round(label_score, 4),
            "alert_score":         round(alert_score, 4),
            "efficiency_score":    round(efficiency_score, 4),
            "investigation_score": round(investigation_score, 4),
            "confidence_score":    round(confidence_score, 4),
            "weighted_total":      total,
        }

        explanation = self._build_explanation(
            s, label_score, alert_score,
            efficiency_score, investigation_score,
            confidence_score, total,
        )

        return {
            "score": total,
            "breakdown": breakdown,
            "explanation": explanation,
            "passed": total >= 0.5,
        }

    # ──────────────────────────────────────────
    # COMPONENT SCORERS
    # ──────────────────────────────────────────

    def _score_label(
        self,
        agent_label: ClassificationLabel,
        ground_truth: ClassificationLabel,
    ) -> float:
        """
        Graded label accuracy with partial credit.

        Scoring:
            Exact match                    → 1.00
            fake ↔ likely_fake             → 0.70  (same danger zone)
            suspicious ↔ likely_fake       → 0.40
            suspicious ↔ unknown           → 0.30
            real ↔ unknown                 → 0.20
            real ↔ suspicious              → 0.15
            fake ↔ unknown                 → 0.10
            fake ↔ real  (opposite poles)  → 0.00
            likely_fake ↔ real             → 0.00
        """
        if agent_label == ground_truth:
            return 1.00

        pair = tuple(sorted([agent_label, ground_truth]))

        partial_map: Dict[tuple, float] = {
            ("fake", "likely_fake"):       0.70,
            ("likely_fake", "suspicious"): 0.40,
            ("suspicious", "unknown"):     0.30,
            ("real", "unknown"):           0.20,
            ("real", "suspicious"):        0.15,
            ("fake", "unknown"):           0.10,
            ("likely_fake", "unknown"):    0.10,
            ("fake", "suspicious"):        0.10,
            ("fake", "real"):              0.00,
            ("likely_fake", "real"):       0.00,
        }
        return partial_map.get(pair, 0.05)

    def _score_alert(
        self,
        agent_alert: AlertLevel,
        correct_alert: AlertLevel,
    ) -> float:
        """
        Alert level scoring based on distance from correct level.

        Distance 0 → 1.00
        Distance 1 → 0.40  (one level off)
        Distance 2 → 0.00  (complete opposite — e.g. GREEN on a fake post)
        """
        distance = abs(ALERT_RANK[agent_alert] - ALERT_RANK[correct_alert])
        scores = {0: 1.00, 1: 0.40, 2: 0.00}
        return scores.get(distance, 0.0)

    def _score_efficiency(
        self,
        step_number: int,
        max_steps: int,
        label_score: float,
    ) -> float:
        """
        Efficiency score — rewards reaching a correct decision in fewer steps.
        Only meaningful if the label was at least partially correct (label_score > 0.10).

        step_fraction = steps_used / max_steps
            0.0–0.30 → 1.00  (very fast)
            0.31–0.50 → 0.80
            0.51–0.70 → 0.60
            0.71–0.90 → 0.40
            0.91–1.00 → 0.20
        Multiplied by label_score so wrong-but-fast = low efficiency reward.
        """
        if label_score <= 0.10:
            return 0.0  # Wrong answer quickly = no efficiency credit

        fraction = step_number / max_steps
        if fraction <= 0.30:
            base = 1.00
        elif fraction <= 0.50:
            base = 0.80
        elif fraction <= 0.70:
            base = 0.60
        elif fraction <= 0.90:
            base = 0.40
        else:
            base = 0.20

        return round(base * label_score, 4)

    def _score_investigation(self, actions_taken: list) -> float:
        """
        Reward for using diverse investigation actions before deciding.

        Scoring:
            Used analyze_claim  → +0.30
            Used check_source   → +0.30
            Used cross_verify   → +0.40
            Max: 1.00

        Penalise redundant actions (same action type > 2 times): -0.10 each.
        """
        action_types = [a.get("action_type") for a in actions_taken]
        investigation_types = ["analyze_claim", "check_source", "cross_verify"]

        score = 0.0
        type_weights = {
            "analyze_claim": 0.30,
            "check_source":  0.30,
            "cross_verify":  0.40,
        }

        used = set()
        for atype in action_types:
            if atype in investigation_types and atype not in used:
                score += type_weights[atype]
                used.add(atype)

        # Penalty for over-repetition
        from collections import Counter
        counts = Counter(action_types)
        for atype, count in counts.items():
            if atype in investigation_types and count > 2:
                score -= 0.10 * (count - 2)

        return round(min(max(score, 0.0), 1.0), 4)

    def _score_confidence(
        self,
        confidence: float,
        label_score: float,
    ) -> float:
        """
        Confidence calibration score.
        A well-calibrated agent should have:
            - High confidence when correct
            - Low confidence when wrong

        Scoring:
            Correct (label_score >= 0.70) + confidence >= 0.70 → 1.00
            Correct + confidence 0.40–0.69                     → 0.70
            Correct + confidence < 0.40                        → 0.40
            Wrong (label_score < 0.30) + confidence < 0.50    → 0.80  (correctly uncertain)
            Wrong + confidence 0.50–0.70                       → 0.40
            Wrong + confidence > 0.70                          → 0.00  (overconfident + wrong)
        """
        correct = label_score >= 0.70

        if correct:
            if confidence >= 0.70:
                return 1.00
            elif confidence >= 0.40:
                return 0.70
            else:
                return 0.40
        else:
            if confidence < 0.50:
                return 0.80
            elif confidence <= 0.70:
                return 0.40
            else:
                return 0.00

    # ──────────────────────────────────────────
    # EXPLANATION BUILDER
    # ──────────────────────────────────────────

    def _build_explanation(
        self,
        s: EpisodeState,
        label_score: float,
        alert_score: float,
        efficiency_score: float,
        investigation_score: float,
        confidence_score: float,
        total: float,
    ) -> str:
        lines = [
            f"=== GRADER REPORT — Task: {s.task_id.upper()} ===",
            f"Agent label   : {s.current_label}  (ground truth: {s.ground_truth})",
            f"Agent alert   : {s.current_alert}  (correct: {LABEL_TO_CORRECT_ALERT[s.ground_truth]})",
            f"Steps used    : {s.step_number} / {s.max_steps}",
            f"Confidence    : {s.confidence:.2f}",
            f"",
            f"SCORES (weighted):",
            f"  Label accuracy   : {label_score:.4f} x {WEIGHT_LABEL}  = {label_score * WEIGHT_LABEL:.4f}",
            f"  Alert accuracy   : {alert_score:.4f} x {WEIGHT_ALERT}  = {alert_score * WEIGHT_ALERT:.4f}",
            f"  Efficiency       : {efficiency_score:.4f} x {WEIGHT_EFFICIENCY}  = {efficiency_score * WEIGHT_EFFICIENCY:.4f}",
            f"  Investigation    : {investigation_score:.4f} x {WEIGHT_INVESTIGATION}  = {investigation_score * WEIGHT_INVESTIGATION:.4f}",
            f"  Confidence calib.: {confidence_score:.4f} x {WEIGHT_CONFIDENCE}  = {confidence_score * WEIGHT_CONFIDENCE:.4f}",
            f"",
            f"FINAL SCORE: {total:.4f}  ({'PASS' if total >= 0.5 else 'FAIL'})",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def grade_episode(episode_state: EpisodeState) -> Dict[str, Any]:
    """Module-level convenience wrapper around FakeNewsGrader.grade()."""
    return FakeNewsGrader().grade(episode_state)