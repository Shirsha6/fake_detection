"""
rewards.py — Member B
Detection engine for the Fake News Detection OpenEnv Environment.

Contains:
- Controlled knowledge base (deterministic)
- Pattern detection (fake phrases / clickbait)
- Source credibility system
- Virality risk analysis
- Multi-signal fake score calculation
- Multi-class classification logic
- Alert system (GREEN / YELLOW / RED)
- Confidence scoring
- Explanation generation
- Reward computation (partial + terminal)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models import AlertLevel, ClassificationLabel


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — KNOWLEDGE BASE  (fully deterministic, no external calls)
# ═══════════════════════════════════════════════════════════════════

# Structure:
#   claim_text (lowercase) → {
#       "verdict"     : "true" | "false" | "partial" | "unknown",
#       "explanation" : str,
#       "confidence"  : float (0.0–1.0),
#   }

KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {

    # ── REAL / TRUE claims ──────────────────────────────────────────
    "the earth orbits the sun": {
        "verdict": "true",
        "explanation": "Basic established astronomical fact.",
        "confidence": 1.0,
    },
    "vaccines undergo clinical trials before approval": {
        "verdict": "true",
        "explanation": "Standard regulatory process globally.",
        "confidence": 1.0,
    },
    "the who declared covid-19 a pandemic in march 2020": {
        "verdict": "true",
        "explanation": "WHO declared pandemic on March 11, 2020.",
        "confidence": 1.0,
    },
    "climate change is supported by scientific consensus": {
        "verdict": "true",
        "explanation": "Over 97% of climate scientists agree on human-caused warming.",
        "confidence": 0.98,
    },
    "drinking water is essential for human survival": {
        "verdict": "true",
        "explanation": "Biological fact — humans cannot survive without water.",
        "confidence": 1.0,
    },
    "the human body has 206 bones": {
        "verdict": "true",
        "explanation": "Standard anatomical fact for adult humans.",
        "confidence": 1.0,
    },
    "5g networks use radio waves": {
        "verdict": "true",
        "explanation": "5G uses radio frequency spectrum, same electromagnetic family as 4G.",
        "confidence": 1.0,
    },
    "elon musk is the ceo of tesla": {
        "verdict": "true",
        "explanation": "Elon Musk is the CEO of Tesla Inc.",
        "confidence": 0.95,
    },

    # ── FAKE / FALSE claims ─────────────────────────────────────────
    "5g towers spread covid-19": {
        "verdict": "false",
        "explanation": "Viruses cannot travel on radio waves. Thoroughly debunked.",
        "confidence": 1.0,
    },
    "vaccines cause autism": {
        "verdict": "false",
        "explanation": "The original Wakefield study was retracted. No credible study confirms this.",
        "confidence": 1.0,
    },
    "bill gates is microchipping people through vaccines": {
        "verdict": "false",
        "explanation": "No evidence. Thoroughly debunked by multiple fact-checkers.",
        "confidence": 1.0,
    },
    "the moon landing was faked": {
        "verdict": "false",
        "explanation": "Multiple independent verifications confirm Apollo missions were real.",
        "confidence": 1.0,
    },
    "drinking bleach cures covid-19": {
        "verdict": "false",
        "explanation": "Bleach is toxic. This claim is dangerous and false.",
        "confidence": 1.0,
    },
    "george soros controls world governments": {
        "verdict": "false",
        "explanation": "No credible evidence. Common conspiracy theory.",
        "confidence": 0.98,
    },
    "the earth is flat": {
        "verdict": "false",
        "explanation": "Earth is an oblate spheroid. Confirmed by centuries of science.",
        "confidence": 1.0,
    },
    "covid-19 vaccines contain microchips": {
        "verdict": "false",
        "explanation": "No microchips in vaccines. Physically impossible at that scale.",
        "confidence": 1.0,
    },
    "migrants are replacing native populations deliberately": {
        "verdict": "false",
        "explanation": "Replacement theory is a debunked extremist conspiracy.",
        "confidence": 0.97,
    },

    # ── PARTIAL TRUTH claims ────────────────────────────────────────
    "sugar causes hyperactivity in children": {
        "verdict": "partial",
        "explanation": "Studies show no direct link. Effect may be psychological (parental expectation).",
        "confidence": 0.85,
    },
    "eating carrots improves night vision": {
        "verdict": "partial",
        "explanation": "True only if vitamin A deficient. Does not give night-vision beyond normal.",
        "confidence": 0.80,
    },
    "coffee stunts growth": {
        "verdict": "partial",
        "explanation": "No strong evidence, but heavy caffeine may affect sleep which affects growth.",
        "confidence": 0.75,
    },
    "humans only use 10 percent of their brain": {
        "verdict": "false",
        "explanation": "Neuroscience shows virtually all brain regions are active.",
        "confidence": 0.99,
    },
    "social media companies sell your data": {
        "verdict": "partial",
        "explanation": "They share/monetize data with advertisers, though 'sell' is technically disputed.",
        "confidence": 0.80,
    },
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — SOURCE CREDIBILITY SYSTEM
# ═══════════════════════════════════════════════════════════════════

# Structure:
#   source_name (lowercase) → {
#       "score" : float 0.0–1.0  (higher = more credible),
#       "tier"  : "high" | "medium" | "low" | "unreliable",
#       "known" : bool,
#       "note"  : str,
#   }

SOURCE_CREDIBILITY: Dict[str, Dict[str, Any]] = {

    # ── High credibility ────────────────────────────────────────────
    "reuters": {
        "score": 0.95, "tier": "high", "known": True,
        "note": "International wire service with strong editorial standards.",
    },
    "associated press": {
        "score": 0.95, "tier": "high", "known": True,
        "note": "AP is one of the most trusted news agencies globally.",
    },
    "bbc": {
        "score": 0.90, "tier": "high", "known": True,
        "note": "Public broadcaster with strong editorial oversight.",
    },
    "who": {
        "score": 0.92, "tier": "high", "known": True,
        "note": "World Health Organization — authoritative on health claims.",
    },
    "cdc": {
        "score": 0.92, "tier": "high", "known": True,
        "note": "Centers for Disease Control — authoritative on health.",
    },
    "nature": {
        "score": 0.97, "tier": "high", "known": True,
        "note": "Peer-reviewed scientific journal.",
    },
    "new york times": {
        "score": 0.85, "tier": "high", "known": True,
        "note": "Major newspaper with strong fact-checking, though editorial bias exists.",
    },
    "the guardian": {
        "score": 0.83, "tier": "high", "known": True,
        "note": "Reputable British newspaper.",
    },
    "snopes": {
        "score": 0.88, "tier": "high", "known": True,
        "note": "Dedicated fact-checking organisation.",
    },
    "politifact": {
        "score": 0.87, "tier": "high", "known": True,
        "note": "Pulitzer Prize-winning fact-checker.",
    },

    # ── Medium credibility ──────────────────────────────────────────
    "fox news": {
        "score": 0.55, "tier": "medium", "known": True,
        "note": "Large US network with documented editorial slant.",
    },
    "daily mail": {
        "score": 0.45, "tier": "medium", "known": True,
        "note": "High readership but history of sensationalist reporting.",
    },
    "buzzfeed news": {
        "score": 0.60, "tier": "medium", "known": True,
        "note": "Has broken real stories but mixed track record.",
    },
    "huffpost": {
        "score": 0.62, "tier": "medium", "known": True,
        "note": "Left-leaning with generally acceptable fact standards.",
    },

    # ── Low / Unreliable credibility ────────────────────────────────
    "naturalcures.com": {
        "score": 0.10, "tier": "unreliable", "known": True,
        "note": "Known pseudoscience and alternative medicine misinformation site.",
    },
    "infowars": {
        "score": 0.05, "tier": "unreliable", "known": True,
        "note": "Repeatedly flagged for conspiracy theories and misinformation.",
    },
    "beforeitsnews": {
        "score": 0.08, "tier": "unreliable", "known": True,
        "note": "Citizen journalism site with no editorial standards.",
    },
    "yournewswire": {
        "score": 0.06, "tier": "unreliable", "known": True,
        "note": "Known fake news site, now rebranded as newspunch.",
    },
    "zerohedge": {
        "score": 0.20, "tier": "low", "known": True,
        "note": "Financial blog with history of sensationalism and conspiracy content.",
    },
    "unknown blog": {
        "score": 0.15, "tier": "low", "known": False,
        "note": "Unverified blog with no editorial accountability.",
    },
}

# Default for sources not in the registry
UNKNOWN_SOURCE_DEFAULT: Dict[str, Any] = {
    "score": 0.30,
    "tier": "low",
    "known": False,
    "note": "Source not found in credibility database. Treat with caution.",
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — PATTERN DETECTION  (fake/clickbait phrase library)
# ═══════════════════════════════════════════════════════════════════

# Each pattern → weight contribution to fake_score (0.0–1.0)
FAKE_PATTERNS: Dict[str, float] = {
    # Extreme urgency / panic
    "share before they delete": 0.90,
    "they don't want you to know": 0.88,
    "wake up sheeple": 0.92,
    "the mainstream media won't tell you": 0.85,
    "banned from social media": 0.80,
    "censored by": 0.78,
    "deep state": 0.75,
    "new world order": 0.82,
    "illuminati": 0.80,
    "shadow government": 0.79,

    # Medical misinformation signals
    "miracle cure": 0.88,
    "doctors don't want you to know": 0.90,
    "big pharma hiding": 0.85,
    "natural cure for": 0.65,
    "cures cancer": 0.80,
    "cures covid": 0.85,
    "detox your body": 0.55,
    "boosts immunity instantly": 0.70,

    # Conspiracy signals
    "false flag": 0.78,
    "government is lying": 0.72,
    "crisis actor": 0.85,
    "plandemic": 0.90,
    "great reset": 0.70,
    "globalist agenda": 0.80,
    "population control": 0.72,
    "chemtrails": 0.85,
    "microchipped": 0.88,

    # Clickbait / sensationalism signals
    "you won't believe": 0.60,
    "shocking truth": 0.65,
    "what they're not telling you": 0.72,
    "this will blow your mind": 0.58,
    "100% proven": 0.70,
    "scientists baffled": 0.65,
    "exposed!": 0.68,
    "breaking:": 0.40,         # lower — legitimate news also uses this
    "urgent:": 0.45,

    # Emotional manipulation
    "share with everyone you love": 0.62,
    "repost immediately": 0.65,
    "before it's too late": 0.60,
    "going viral": 0.35,        # lower — can be neutral
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — VIRALITY RISK TABLE
# ═══════════════════════════════════════════════════════════════════

# Virality risk amplifies the danger of a fake post but is NOT
# a signal of fakeness itself — it only affects alert severity.
VIRALITY_RISK_MULTIPLIER: Dict[str, float] = {
    "low":    1.0,
    "medium": 1.15,
    "high":   1.30,
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — DETECTION ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════

class FakeNewsDetector:
    """
    Deterministic multi-signal fake news detection engine.

    All methods are pure functions of their inputs — no randomness,
    no external API calls. Given the same input, always returns the
    same output.
    """

    # ── 5.1  Knowledge Base Lookup ───────────────────────────────────

    def check_claim(self, claim: str) -> Dict[str, Any]:
        """
        Look up a claim in the knowledge base.

        Parameters
        ----------
        claim : str — the claim to look up (case-insensitive)

        Returns
        -------
        dict with keys: found, verdict, explanation, confidence, claim
        """
        key = claim.strip().lower()

        # Exact match
        if key in KNOWLEDGE_BASE:
            entry = KNOWLEDGE_BASE[key]
            return {
                "found": True,
                "claim": claim,
                "verdict": entry["verdict"],
                "explanation": entry["explanation"],
                "confidence": entry["confidence"],
            }

        # Partial / substring match (scan all KB entries)
        for kb_key, entry in KNOWLEDGE_BASE.items():
            if kb_key in key or key in kb_key:
                return {
                    "found": True,
                    "claim": claim,
                    "verdict": entry["verdict"],
                    "explanation": f"[Partial match to '{kb_key}'] " + entry["explanation"],
                    "confidence": entry["confidence"] * 0.85,  # slight confidence penalty
                }

        # Not found
        return {
            "found": False,
            "claim": claim,
            "verdict": "unknown",
            "explanation": "Claim not found in knowledge base. Cannot verify.",
            "confidence": 0.0,
        }

    def cross_verify(self, claim: str, extracted_claims: List[str]) -> Dict[str, Any]:
        """
        Cross-verify a claim by checking it and all extracted claims,
        then returning an aggregate verdict.

        Returns
        -------
        dict with: results (list), aggregate_fake_signal (float), summary (str)
        """
        targets = [claim] if claim else []
        targets += [c for c in extracted_claims if c != claim]

        results = [self.check_claim(c) for c in targets]

        false_count = sum(1 for r in results if r["verdict"] == "false")
        partial_count = sum(1 for r in results if r["verdict"] == "partial")
        true_count = sum(1 for r in results if r["verdict"] == "true")
        unknown_count = sum(1 for r in results if r["verdict"] == "unknown")
        total = len(results) if results else 1

        # Aggregate fake signal: false → 1.0, partial → 0.5, true → 0.0, unknown → 0.3
        fake_signal = (
            (false_count * 1.0) +
            (partial_count * 0.5) +
            (unknown_count * 0.3)
        ) / total

        summary_parts = []
        if false_count:
            summary_parts.append(f"{false_count} false claim(s)")
        if partial_count:
            summary_parts.append(f"{partial_count} partially true claim(s)")
        if true_count:
            summary_parts.append(f"{true_count} verified true claim(s)")
        if unknown_count:
            summary_parts.append(f"{unknown_count} unknown claim(s)")

        summary = "Cross-verify results: " + ", ".join(summary_parts) if summary_parts else "No claims verified."

        return {
            "results": results,
            "aggregate_fake_signal": round(fake_signal, 4),
            "summary": summary,
        }

    # ── 5.2  Source Credibility ──────────────────────────────────────

    def get_source_credibility(self, source_name: str) -> Dict[str, Any]:
        """
        Look up source credibility from the registry.

        Returns
        -------
        dict with: source, score, tier, known, note
        """
        key = source_name.strip().lower()
        entry = SOURCE_CREDIBILITY.get(key, UNKNOWN_SOURCE_DEFAULT)
        return {
            "source": source_name,
            "score": entry["score"],
            "tier": entry["tier"],
            "known": entry["known"],
            "note": entry["note"],
        }

    def check_all_sources(self, sources: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Check credibility for a list of sources.

        Returns
        -------
        dict: source_name → credibility_info
        """
        return {src: self.get_source_credibility(src) for src in sources}

    # ── 5.3  Pattern Detection ───────────────────────────────────────

    def detect_patterns(self, text: str) -> Dict[str, Any]:
        """
        Scan post text for fake/clickbait patterns.

        Returns
        -------
        dict with: flags (list of matched patterns), pattern_score (float),
                   explanation (str)
        """
        text_lower = text.lower()
        flags: List[str] = []
        total_weight = 0.0

        for pattern, weight in FAKE_PATTERNS.items():
            if pattern in text_lower:
                flags.append(pattern)
                total_weight += weight

        # Normalise — cap at 1.0 even if multiple patterns match
        pattern_score = min(total_weight / max(len(FAKE_PATTERNS) * 0.3, 1.0), 1.0)
        pattern_score = round(pattern_score, 4)

        if not flags:
            explanation = "No suspicious language patterns detected."
        elif len(flags) <= 2:
            explanation = f"Mild pattern flags detected: {', '.join(flags)}."
        else:
            explanation = f"Multiple suspicious patterns detected: {', '.join(flags)}. High fake signal."

        return {
            "flags": flags,
            "pattern_score": pattern_score,
            "explanation": explanation,
        }

    # ── 5.4  Multi-Signal Fake Score ─────────────────────────────────

    def compute_fake_score(
        self,
        post_text: str,
        extracted_claims: List[str],
        sources: List[str],
        virality_risk: str = "low",
    ) -> Dict[str, Any]:
        """
        Compute a composite fake score from multiple signals.

        Signals and weights:
            pattern_score        → 30%
            source_score         → 35%  (inverted: low credibility = high fake signal)
            claim_score          → 35%  (from cross-verify aggregate)

        Virality risk multiplies the final score slightly.

        Returns
        -------
        dict with: fake_score, pattern_score, source_score, claim_score,
                   flags, source_info, claim_results, explanation
        """

        # Signal 1: Pattern detection
        pattern_result = self.detect_patterns(post_text)
        pattern_score = pattern_result["pattern_score"]

        # Signal 2: Source credibility (inverted — low credibility = high fake signal)
        if sources:
            source_info = self.check_all_sources(sources)
            avg_source_credibility = sum(
                v["score"] for v in source_info.values()
            ) / len(source_info)
            source_score = round(1.0 - avg_source_credibility, 4)  # invert
        else:
            # No source cited → moderate suspicion
            source_info = {}
            source_score = 0.45

        # Signal 3: Claim verification
        if extracted_claims:
            cross_result = self.cross_verify(extracted_claims[0], extracted_claims[1:])
            claim_score = cross_result["aggregate_fake_signal"]
            claim_results = cross_result["results"]
            claim_summary = cross_result["summary"]
        else:
            claim_score = 0.3   # no claims extracted → mild suspicion
            claim_results = []
            claim_summary = "No claims extracted for verification."

        # Weighted composite
        raw_score = (
            (pattern_score * 0.30) +
            (source_score  * 0.35) +
            (claim_score   * 0.35)
        )

        # Virality risk amplifier (does NOT exceed 1.0)
        multiplier = VIRALITY_RISK_MULTIPLIER.get(virality_risk, 1.0)
        fake_score = round(min(raw_score * multiplier, 1.0), 4)

        # Explanation
        explanation = (
            f"Fake score: {fake_score:.2f} | "
            f"Pattern: {pattern_score:.2f} (30%) | "
            f"Source: {source_score:.2f} (35%) | "
            f"Claim: {claim_score:.2f} (35%) | "
            f"Virality multiplier: x{multiplier}"
        )

        return {
            "fake_score": fake_score,
            "pattern_score": pattern_score,
            "source_score": source_score,
            "claim_score": claim_score,
            "flags": pattern_result["flags"],
            "source_info": source_info,
            "claim_results": claim_results,
            "claim_summary": claim_summary,
            "explanation": explanation,
        }

    # ── 5.5  Classification Logic ────────────────────────────────────

    def classify(
        self,
        fake_score: float,
        claim_results: List[Dict[str, Any]],
        source_info: Dict[str, Any],
        flags: List[str],
    ) -> Dict[str, Any]:
        """
        Map a fake_score + signals to a multi-class label and alert level.

        Thresholds:
            fake_score >= 0.75                  → fake       / RED
            fake_score >= 0.55                  → likely_fake / RED
            fake_score >= 0.38 OR partial claims → suspicious / YELLOW
            fake_score < 0.20 AND no false claims→ real       / GREEN
            else (0.20–0.37, no clear signals)  → unknown    / YELLOW

        Returns
        -------
        dict with: label, alert_level, confidence, explanation
        """

        has_false_claim = any(r["verdict"] == "false" for r in claim_results)
        has_partial_claim = any(r["verdict"] == "partial" for r in claim_results)
        has_true_claim = any(r["verdict"] == "true" for r in claim_results)
        has_unreliable_source = any(
            v.get("tier") == "unreliable" for v in source_info.values()
        )

        label: ClassificationLabel
        alert: AlertLevel
        confidence: float

        if fake_score >= 0.75 or (has_false_claim and has_unreliable_source):
            label = "fake"
            alert = "RED"
            confidence = round(min(0.60 + fake_score * 0.40, 0.99), 3)

        elif fake_score >= 0.55 or has_false_claim:
            label = "likely_fake"
            alert = "RED"
            confidence = round(min(0.50 + fake_score * 0.40, 0.95), 3)

        elif fake_score >= 0.38 or has_partial_claim or len(flags) >= 2:
            label = "suspicious"
            alert = "YELLOW"
            confidence = round(min(0.40 + fake_score * 0.35, 0.85), 3)

        elif fake_score < 0.20 and not has_false_claim and has_true_claim:
            label = "real"
            alert = "GREEN"
            confidence = round(min(0.70 + (1.0 - fake_score) * 0.25, 0.97), 3)

        else:
            # Ambiguous zone — cannot determine with available signals
            label = "unknown"
            alert = "YELLOW"
            confidence = round(0.30 + fake_score * 0.20, 3)

        explanation = (
            f"Classification: {label} | Alert: {alert} | Confidence: {confidence:.2f} | "
            f"Score: {fake_score:.2f} | "
            f"False claims: {has_false_claim} | "
            f"Partial claims: {has_partial_claim} | "
            f"Unreliable source: {has_unreliable_source}"
        )

        return {
            "label": label,
            "alert_level": alert,
            "confidence": confidence,
            "explanation": explanation,
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — REWARD COMPUTER
# ═══════════════════════════════════════════════════════════════════

# Correct alert for each ground truth label
LABEL_TO_CORRECT_ALERT: Dict[ClassificationLabel, AlertLevel] = {
    "real":        "GREEN",
    "fake":        "RED",
    "likely_fake": "RED",
    "suspicious":  "YELLOW",
    "unknown":     "YELLOW",
}

# Alert level numeric for distance math
ALERT_RANK: Dict[AlertLevel, int] = {
    "GREEN": 0,
    "YELLOW": 1,
    "RED": 2,
}


class RewardComputer:
    """
    Computes step-level and terminal rewards for the environment.

    Design principles:
    - Partial rewards for good investigative actions
    - Efficiency bonus for reaching correct decision quickly
    - Graded penalty for wrong terminal decisions
    - Penalties for redundant / repeated actions
    - No binary-only rewards
    """

    # ── 6.1  Investigation Step Rewards ─────────────────────────────

    def reward_analyze_claim(
        self,
        claim_result: Dict[str, Any],
        already_analyzed: List[str],
        target_claim: Optional[str],
    ) -> Tuple[float, str]:
        """
        Reward for analyze_claim action.
        - Bonus if claim found in KB and verdict is informative
        - Small penalty for re-analyzing already checked claims
        """
        if not target_claim:
            return -0.05, "No target_claim specified — lost investigative value."

        if target_claim.lower() in [c.lower() for c in already_analyzed]:
            return -0.10, f"Claim '{target_claim}' already analyzed — redundant action."

        verdict = claim_result.get("verdict", "unknown")
        found = claim_result.get("found", False)

        if not found:
            return 0.05, "Claim not in knowledge base — unknown territory, mild reward for checking."

        verdict_rewards = {
            "false":   0.20,
            "true":    0.15,
            "partial": 0.12,
            "unknown": 0.05,
        }
        reward = verdict_rewards.get(verdict, 0.05)
        feedback = (
            f"Claim analyzed: verdict='{verdict}' | "
            f"Reward: +{reward:.2f} | "
            f"{claim_result.get('explanation', '')}"
        )
        return reward, feedback

    def reward_check_source(
        self,
        source_result: Dict[str, Any],
        already_checked: List[str],
        source_name: Optional[str],
    ) -> Tuple[float, str]:
        """
        Reward for check_source action.
        - Bonus for checking relevant/known sources
        - Penalty for re-checking the same source
        """
        if not source_name:
            return -0.05, "No source_name specified — no investigative value."

        if source_name.lower() in [s.lower() for s in already_checked]:
            return -0.08, f"Source '{source_name}' already checked — redundant action."

        tier = source_result.get("tier", "low")
        known = source_result.get("known", False)

        if not known:
            return 0.07, f"Unknown source checked: '{source_name}' — moderate caution signal."

        tier_rewards = {
            "high":       0.15,
            "medium":     0.10,
            "low":        0.10,
            "unreliable": 0.18,  # High reward — detecting unreliable source is very useful
        }
        reward = tier_rewards.get(tier, 0.08)
        feedback = (
            f"Source checked: tier='{tier}' | score={source_result.get('score', 0):.2f} | "
            f"Reward: +{reward:.2f} | {source_result.get('note', '')}"
        )
        return reward, feedback

    def reward_cross_verify(
        self,
        cross_result: Dict[str, Any],
        already_cross_verified: bool,
    ) -> Tuple[float, str]:
        """
        Reward for cross_verify action.
        - Good reward for first cross-verify (aggregates multiple claims)
        - Diminishing returns for repeat calls
        """
        if already_cross_verified:
            return -0.05, "Cross-verification already performed — diminishing returns."

        fake_signal = cross_result.get("aggregate_fake_signal", 0.0)

        # Higher fake signal found = more investigative value
        if fake_signal >= 0.7:
            reward = 0.22
        elif fake_signal >= 0.4:
            reward = 0.15
        else:
            reward = 0.10

        feedback = (
            f"Cross-verify complete | aggregate_fake_signal={fake_signal:.2f} | "
            f"Reward: +{reward:.2f} | {cross_result.get('summary', '')}"
        )
        return reward, feedback

    # ── 6.2  Terminal Action Rewards ─────────────────────────────────

    def reward_terminal(
        self,
        agent_label: ClassificationLabel,
        agent_alert: AlertLevel,
        ground_truth: ClassificationLabel,
        ground_truth_alert: AlertLevel,
        step_number: int,
        max_steps: int,
        confidence: float,
    ) -> Tuple[float, str]:
        """
        Compute terminal reward when agent calls raise_alert or mark_safe.

        Components:
        1. Label accuracy     → 0.0–0.50
        2. Alert accuracy     → 0.0–0.25
        3. Efficiency bonus   → 0.0–0.15  (fewer steps = higher bonus)
        4. Confidence penalty → up to -0.10 (overconfidence on wrong answer)

        Total range: roughly -0.35 to +0.90
        """

        # ── Label accuracy ───────────────────────────────────────────
        label_score = self._label_accuracy_score(agent_label, ground_truth)

        # ── Alert accuracy ───────────────────────────────────────────
        correct_alert = LABEL_TO_CORRECT_ALERT[ground_truth]
        alert_distance = abs(
            ALERT_RANK[agent_alert] - ALERT_RANK[correct_alert]
        )
        if alert_distance == 0:
            alert_score = 0.25
        elif alert_distance == 1:
            alert_score = 0.10   # one level off
        else:
            alert_score = -0.10  # completely wrong (GREEN on a fake post)

        # ── Efficiency bonus ─────────────────────────────────────────
        step_fraction = step_number / max_steps
        if label_score >= 0.40:
            # Only reward efficiency if the decision is mostly correct
            efficiency_bonus = round(0.15 * (1.0 - step_fraction), 3)
        else:
            efficiency_bonus = 0.0

        # ── Confidence penalty ───────────────────────────────────────
        if label_score < 0.20 and confidence > 0.80:
            confidence_penalty = -0.10  # very wrong + very confident = bad
        elif label_score < 0.20 and confidence > 0.60:
            confidence_penalty = -0.05
        else:
            confidence_penalty = 0.0

        total = round(
            label_score + alert_score + efficiency_bonus + confidence_penalty,
            4
        )
        # Clamp
        total = max(min(total, 1.0), -0.5)

        feedback = (
            f"Terminal reward: {total:+.3f} | "
            f"Label: {agent_label} (truth={ground_truth}, score={label_score:.2f}) | "
            f"Alert: {agent_alert} (correct={correct_alert}, score={alert_score:.2f}) | "
            f"Efficiency: +{efficiency_bonus:.3f} | "
            f"Confidence penalty: {confidence_penalty:.2f}"
        )
        return total, feedback

    def _label_accuracy_score(
        self,
        agent_label: ClassificationLabel,
        ground_truth: ClassificationLabel,
    ) -> float:
        """
        Graded label accuracy scoring — partial credit for close guesses.

        Scoring matrix:
            Exact match                          → 0.50
            fake ↔ likely_fake                   → 0.35  (same danger zone)
            suspicious ↔ likely_fake             → 0.20
            suspicious ↔ unknown                 → 0.15
            real → fake (or fake → real)         → -0.20 (opposite poles)
            All other mismatches                 → 0.05
        """
        if agent_label == ground_truth:
            return 0.50

        pair = tuple(sorted([agent_label, ground_truth]))

        partial_credit_map: Dict[tuple, float] = {
            ("fake", "likely_fake"):       0.35,
            ("likely_fake", "suspicious"): 0.20,
            ("suspicious", "unknown"):     0.15,
            ("likely_fake", "unknown"):    0.10,
            ("fake", "suspicious"):        0.10,
            ("real", "unknown"):           0.08,
            ("real", "suspicious"):        0.05,
            ("fake", "real"):             -0.20,
            ("likely_fake", "real"):      -0.15,
        }
        return partial_credit_map.get(pair, 0.05)