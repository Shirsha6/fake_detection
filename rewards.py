"""
rewards.py — Member B (FINAL CLEAN VERSION)

Detection engine + reward system for Fake News Environment.
Fully deterministic. No external dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models import AlertLevel, ClassificationLabel


# ═══════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "the earth orbits the sun": {
        "verdict": "true",
        "explanation": "Basic established astronomical fact.",
        "confidence": 1.0,
    },
    "vaccines cause autism": {
        "verdict": "false",
        "explanation": "Debunked by scientific consensus.",
        "confidence": 1.0,
    },
    "5g towers spread covid-19": {
        "verdict": "false",
        "explanation": "Viruses do not spread via radio waves.",
        "confidence": 1.0,
    },
    "climate change is supported by scientific consensus": {
        "verdict": "true",
        "explanation": "Overwhelming scientific agreement.",
        "confidence": 0.98,
    },
}


# ═══════════════════════════════════════════════════════════════════
# SOURCE CREDIBILITY
# ═══════════════════════════════════════════════════════════════════

SOURCE_CREDIBILITY: Dict[str, Dict[str, Any]] = {
    "reuters": {"score": 0.95, "tier": "high", "known": True},
    "bbc": {"score": 0.90, "tier": "high", "known": True},
    "infowars": {"score": 0.05, "tier": "unreliable", "known": True},
}

UNKNOWN_SOURCE_DEFAULT = {
    "score": 0.30,
    "tier": "low",
    "known": False,
}


# ═══════════════════════════════════════════════════════════════════
# PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════

FAKE_PATTERNS: Dict[str, float] = {
    "wake up sheeple": 0.9,
    "share before they delete": 0.9,
    "miracle cure": 0.85,
    "you won't believe": 0.6,
}


# ═══════════════════════════════════════════════════════════════════
# VIRALITY
# ═══════════════════════════════════════════════════════════════════

VIRALITY_RISK_MULTIPLIER = {
    "low": 1.0,
    "medium": 1.15,
    "high": 1.3,
}


# ═══════════════════════════════════════════════════════════════════
# DETECTOR
# ═══════════════════════════════════════════════════════════════════

class FakeNewsDetector:

    def check_claim(self, claim: str) -> Dict[str, Any]:
        key = claim.lower().strip()

        if key in KNOWLEDGE_BASE:
            entry = KNOWLEDGE_BASE[key]
            return {"found": True, "claim": claim, **entry}

        return {
            "found": False,
            "claim": claim,
            "verdict": "unknown",
            "explanation": "Not found in KB",
            "confidence": 0.0,
        }

    def cross_verify(self, claim: str, extracted_claims: List[str]) -> Dict[str, Any]:
        claims = [claim] + extracted_claims
        results = [self.check_claim(c) for c in claims]

        score = sum(
            1.0 if r["verdict"] == "false"
            else 0.5 if r["verdict"] == "partial"
            else 0.3 if r["verdict"] == "unknown"
            else 0.0
            for r in results
        ) / max(len(results), 1)

        return {
            "results": results,
            "aggregate_fake_signal": round(score, 4),
            "summary": f"{len(results)} claims checked",
        }

    def get_source_credibility(self, source: str) -> Dict[str, Any]:
        key = source.lower().strip()
        entry = SOURCE_CREDIBILITY.get(key, UNKNOWN_SOURCE_DEFAULT)
        return {"source": source, **entry}

    def detect_patterns(self, text: str) -> Dict[str, Any]:
        text = text.lower()
        flags = [p for p in FAKE_PATTERNS if p in text]

        score = min(len(flags) * 0.2, 1.0)

        return {
            "flags": flags,
            "pattern_score": score,
        }

    def compute_fake_score(
        self,
        post_text: str,
        extracted_claims: List[str],
        sources: List[str],
        virality_risk: str,
    ) -> Dict[str, Any]:

        pattern = self.detect_patterns(post_text)
        pattern_score = pattern["pattern_score"]

        if sources:
            source_scores = [
                self.get_source_credibility(s)["score"] for s in sources
            ]
            source_score = 1 - sum(source_scores) / len(source_scores)
        else:
            source_score = 0.4

        if extracted_claims:
            cross = self.cross_verify(extracted_claims[0], extracted_claims[1:])
            claim_score = cross["aggregate_fake_signal"]
            claim_results = cross["results"]
        else:
            claim_score = 0.3
            claim_results = []

        raw = (pattern_score * 0.3 + source_score * 0.35 + claim_score * 0.35)

        multiplier = VIRALITY_RISK_MULTIPLIER.get(virality_risk, 1.0)

        fake_score = min(raw * multiplier, 1.0)

        return {
            "fake_score": round(fake_score, 4),
            "flags": pattern["flags"],
            "source_info": {},
            "claim_results": claim_results,
        }

    def classify(
        self,
        fake_score: float,
        claim_results: List[Dict[str, Any]],
        source_info: Dict[str, Any],
        flags: List[str],
    ) -> Dict[str, Any]:

        if fake_score > 0.75:
            return {"label": "fake", "alert_level": "RED", "confidence": 0.9}
        elif fake_score > 0.55:
            return {"label": "likely_fake", "alert_level": "RED", "confidence": 0.8}
        elif fake_score > 0.35:
            return {"label": "suspicious", "alert_level": "YELLOW", "confidence": 0.7}
        else:
            return {"label": "real", "alert_level": "GREEN", "confidence": 0.85}


# ═══════════════════════════════════════════════════════════════════
# REWARD SYSTEM
# ═══════════════════════════════════════════════════════════════════

ALERT_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}


class RewardComputer:

    def reward_analyze_claim(self, claim_result, already_analyzed, target_claim):
        if target_claim in already_analyzed:
            return -0.1, "Repeated claim"

        if claim_result["found"]:
            return 0.2, "Useful claim verified"
        return 0.05, "Unknown claim checked"

    def reward_check_source(self, source_result, already_checked, source_name):
        if source_name in already_checked:
            return -0.08, "Repeated source"

        tier = source_result.get("tier", "low")

        if tier == "unreliable":
            return 0.18, "Unreliable source found"
        elif tier == "high":
            return 0.15, "Credible source verified"
        return 0.1, "Source checked"

    def reward_cross_verify(self, cross_result, already_cross_verified):
        if already_cross_verified:
            return -0.05, "Already cross verified"

        signal = cross_result.get("aggregate_fake_signal", 0)

        if signal > 0.7:
            return 0.22, "Strong fake signal"
        elif signal > 0.4:
            return 0.15, "Moderate fake signal"
        return 0.1, "Weak signal"

    def reward_terminal(
        self,
        agent_label,
        agent_alert,
        ground_truth,
        ground_truth_alert,
        step_number,
        max_steps,
        confidence,
    ):

        label_score = 0.5 if agent_label == ground_truth else -0.2

        alert_diff = abs(ALERT_RANK[agent_alert] - ALERT_RANK[ground_truth_alert])

        if alert_diff == 0:
            alert_score = 0.25
        elif alert_diff == 1:
            alert_score = 0.1
        else:
            alert_score = -0.1

        efficiency = 0.15 * (1 - step_number / max_steps)

        total = label_score + alert_score + efficiency

        return round(total, 4), f"Final reward: {total:.3f}"