"""
Task definitions for the FakeNews Detection Environment.
3 tasks: easy → medium → hard with realistic social media posts.
All data is deterministic (no external API calls).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from models import Label, AlertLevel


# ─────────────────────────── Knowledge Base ───────────────────────────

KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    # Verifiable facts (deterministic)
    "covid vaccines contain microchips": {
        "verdict": "false",
        "explanation": "No credible scientific evidence. Debunked by CDC, WHO, peer-reviewed studies.",
        "sources": ["CDC", "WHO", "Nature Medicine"],
    },
    "5g towers cause cancer": {
        "verdict": "false",
        "explanation": "Non-ionizing radiation at 5G frequencies is not strong enough to damage DNA.",
        "sources": ["ICNIRP", "WHO", "IEEE"],
    },
    "drinking bleach cures covid": {
        "verdict": "false",
        "explanation": "Extremely dangerous health misinformation. FDA issued strong warnings.",
        "sources": ["FDA", "CDC", "Poison Control"],
    },
    "moon landing was faked": {
        "verdict": "false",
        "explanation": "Extensively documented. Hundreds of scientists from multiple countries verified.",
        "sources": ["NASA", "ESA", "Smithsonian"],
    },
    "climate change is caused by human activity": {
        "verdict": "true",
        "explanation": "97%+ scientific consensus from over 10,000 peer-reviewed studies.",
        "sources": ["IPCC", "NASA", "NOAA"],
    },
    "earth is flat": {
        "verdict": "false",
        "explanation": "Earth is an oblate spheroid. Confirmed by astronomy, physics, and direct observation.",
        "sources": ["NASA", "ESA", "Basic Physics"],
    },
    "vaccines cause autism": {
        "verdict": "false",
        "explanation": "The original study was retracted and its author lost medical license for fraud.",
        "sources": ["Lancet retraction", "CDC", "AAP"],
    },
    "new study suggests moderate coffee consumption has health benefits": {
        "verdict": "likely_true",
        "explanation": "Multiple studies suggest antioxidants in coffee may have benefits when consumed moderately.",
        "sources": ["Harvard School of Public Health", "NEJM"],
    },
    "local politician seen at event": {
        "verdict": "unverifiable",
        "explanation": "Cannot be independently verified from knowledge base.",
        "sources": [],
    },
    "election results were manipulated": {
        "verdict": "contested",
        "explanation": "Broad claim without specific evidence. Courts found no systematic fraud in 2020 US election.",
        "sources": ["DHS CISA", "DOJ", "60+ court rulings"],
    },
    "stock market will crash tomorrow": {
        "verdict": "unverifiable",
        "explanation": "Predictions about future market movements cannot be verified.",
        "sources": [],
    },
    "miracle cure found for all cancers": {
        "verdict": "false",
        "explanation": "No single cure exists for all cancers. Highly suspicious health claim.",
        "sources": ["NCI", "WHO", "ACS"],
    },
    "government hiding alien contact": {
        "verdict": "unverifiable",
        "explanation": "No verifiable evidence. Conspiracy claim with extraordinary burden of proof.",
        "sources": [],
    },
    "water fluoridation causes brain damage": {
        "verdict": "false",
        "explanation": "At recommended levels, fluoride is safe. CDC and WHO endorse fluoridation.",
        "sources": ["CDC", "WHO", "ADA"],
    },
    "exercise improves mental health": {
        "verdict": "true",
        "explanation": "Extensively researched. Exercise releases endorphins and reduces cortisol.",
        "sources": ["Mayo Clinic", "NIMH", "Lancet Psychiatry"],
    },
}

# Source credibility database (deterministic)
SOURCE_CREDIBILITY: Dict[str, Dict[str, Any]] = {
    "cdc.gov": {"credibility": 0.95, "tier": "government", "bias": "none"},
    "who.int": {"credibility": 0.93, "tier": "international_org", "bias": "none"},
    "nature.com": {"credibility": 0.97, "tier": "peer_reviewed", "bias": "none"},
    "nejm.org": {"credibility": 0.96, "tier": "peer_reviewed", "bias": "none"},
    "reuters.com": {"credibility": 0.88, "tier": "established_news", "bias": "low"},
    "apnews.com": {"credibility": 0.90, "tier": "established_news", "bias": "low"},
    "bbc.com": {"credibility": 0.87, "tier": "established_news", "bias": "low"},
    "theguardian.com": {"credibility": 0.82, "tier": "established_news", "bias": "moderate"},
    "foxnews.com": {"credibility": 0.55, "tier": "partisan_news", "bias": "high"},
    "naturalnews.com": {"credibility": 0.05, "tier": "misinformation", "bias": "extreme"},
    "infowars.com": {"credibility": 0.02, "tier": "misinformation", "bias": "extreme"},
    "theonion.com": {"credibility": 0.10, "tier": "satire", "bias": "satire"},
    "beforeitsnews.com": {"credibility": 0.03, "tier": "conspiracy", "bias": "extreme"},
    "worldnewsdailyreport.com": {"credibility": 0.01, "tier": "fake_news_site", "bias": "extreme"},
    "snopes.com": {"credibility": 0.85, "tier": "fact_check", "bias": "low"},
    "politifact.com": {"credibility": 0.84, "tier": "fact_check", "bias": "low"},
    "unknown_source": {"credibility": 0.30, "tier": "unknown", "bias": "unknown"},
    "personal_blog": {"credibility": 0.20, "tier": "unverified", "bias": "unknown"},
    "telegram_channel": {"credibility": 0.10, "tier": "social_media", "bias": "unknown"},
    "anonymous_tipster": {"credibility": 0.05, "tier": "anonymous", "bias": "unknown"},
}

# Linguistic patterns associated with fake news (deterministic)
FAKE_PATTERNS: List[Dict[str, Any]] = [
    {"pattern": "SHOCKING", "weight": 0.3, "category": "sensationalism"},
    {"pattern": "THEY DON'T WANT YOU TO KNOW", "weight": 0.4, "category": "conspiracy"},
    {"pattern": "SHARE BEFORE DELETED", "weight": 0.5, "category": "urgency_manipulation"},
    {"pattern": "DOCTORS HATE THIS", "weight": 0.4, "category": "clickbait"},
    {"pattern": "100% PROVEN", "weight": 0.35, "category": "false_certainty"},
    {"pattern": "MAINSTREAM MEDIA WON'T TELL YOU", "weight": 0.4, "category": "conspiracy"},
    {"pattern": "MIRACLE CURE", "weight": 0.45, "category": "health_misinformation"},
    {"pattern": "WAKE UP SHEEPLE", "weight": 0.5, "category": "conspiracy"},
    {"pattern": "WHAT THEY'RE HIDING", "weight": 0.35, "category": "conspiracy"},
    {"pattern": "GOING VIRAL", "weight": 0.1, "category": "social_proof_manipulation"},
    {"pattern": "THIS WILL CHANGE EVERYTHING", "weight": 0.25, "category": "sensationalism"},
    {"pattern": "BANNED FROM TV", "weight": 0.45, "category": "suppression_claim"},
    {"pattern": "NATURAL NEWS", "weight": 0.3, "category": "pseudoscience_source"},
    {"pattern": "DEEP STATE", "weight": 0.35, "category": "conspiracy"},
    {"pattern": "PLANDEMIC", "weight": 0.5, "category": "conspiracy"},
    {"pattern": "GREAT RESET", "weight": 0.3, "category": "conspiracy"},
    {"pattern": "NEW WORLD ORDER", "weight": 0.4, "category": "conspiracy"},
    {"pattern": "GLOBALIST AGENDA", "weight": 0.4, "category": "conspiracy"},
    {"pattern": "GOVERNMENT COVER-UP", "weight": 0.35, "category": "conspiracy"},
    {"pattern": "PROVEN TO CURE", "weight": 0.45, "category": "health_misinformation"},
]


# ─────────────────────────── Task Definitions ───────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {

    # ─── EASY ───
    "task_easy": {
        "task_id": "task_easy",
        "task_name": "Basic Fake News Detection",
        "difficulty": "easy",
        "description": (
            "A social media post with obvious fake news markers has been flagged. "
            "Your job: analyze the post, extract claims, check sources, and raise the correct alert. "
            "This post contains clear conspiracy language, a blacklisted source, and a debunked claim."
        ),
        "post_id": "post_001",
        "post_text": (
            "🚨 SHOCKING: THEY DON'T WANT YOU TO KNOW THIS! Scientists at naturalnews.com "
            "have PROVEN that COVID vaccines contain microchips that track your location! "
            "100% PROVEN by whistleblowers! SHARE BEFORE DELETED! The government is hiding this "
            "from you. WAKE UP SHEEPLE! #Plandemic #DeepState #VaccinesTruth"
        ),
        "ground_truth_label": Label.FAKE,
        "ground_truth_alert": AlertLevel.RED,
        "key_claims": [
            "covid vaccines contain microchips",
            "government is hiding vaccine information",
        ],
        "key_source": "naturalnews.com",
        "expected_patterns": [
            "SHOCKING", "THEY DON'T WANT YOU TO KNOW", "100% PROVEN",
            "SHARE BEFORE DELETED", "WAKE UP SHEEPLE", "PLANDEMIC", "DEEP STATE"
        ],
        "max_steps": 6,
        "required_actions": ["analyze_claim", "check_source", "raise_alert"],
        "pass_threshold": 0.65,
    },

    # ─── MEDIUM ───
    "task_medium": {
        "task_id": "task_medium",
        "task_name": "Partial Truth & Suspicious Content Detection",
        "difficulty": "medium",
        "description": (
            "A social media post mixes real facts with misleading framing and unverified claims. "
            "The source has moderate credibility. You must cross-verify claims, assess the partial "
            "truth, and determine the correct label and alert level. "
            "Avoid both over-alerting and under-alerting."
        ),
        "post_id": "post_002",
        "post_text": (
            "⚠️ BREAKING: New study shows that 5G towers emit radiation that may affect human "
            "cells. While mainstream science claims it's safe, multiple independent researchers "
            "disagree. A new report from 'independent-health-research.org' found concerning "
            "correlations between 5G rollout areas and increased hospital admissions. "
            "Exercise also helps with mental health – that part is true. "
            "Share this with your family! #5GDanger #HealthAlert"
        ),
        "ground_truth_label": Label.LIKELY_FAKE,
        "ground_truth_alert": AlertLevel.YELLOW,
        "key_claims": [
            "5g towers cause cancer",
            "exercise improves mental health",
            "independent study links 5g to hospital admissions",
        ],
        "key_source": "personal_blog",
        "expected_patterns": ["BREAKING", "SHARE"],
        "max_steps": 8,
        "required_actions": ["analyze_claim", "check_source", "cross_verify", "raise_alert"],
        "pass_threshold": 0.60,
    },

    # ─── HARD ───
    "task_hard": {
        "task_id": "task_hard",
        "task_name": "Sophisticated Disinformation Campaign Detection",
        "difficulty": "hard",
        "description": (
            "A sophisticated disinformation post designed to evade detection. "
            "It uses credible-sounding language, mixes true and false claims, cites real "
            "institutions misleadingly, and uses emotionally charged framing without obvious "
            "fake-news patterns. The agent must: extract multiple layered claims, verify each "
            "independently, assess source manipulation, detect subtle framing tricks, calibrate "
            "confidence accurately, and produce a nuanced label. This post is designed to fool "
            "frontier models — partial truths are used to lend credibility to a false central claim. "
            "A wrong label here costs significant reward. Confidence calibration is tested."
        ),
        "post_id": "post_003",
        "post_text": (
            "📊 IMPORTANT HEALTH UPDATE: A peer-reviewed analysis published last month reveals "
            "what health authorities aren't highlighting: while climate change IS caused by human "
            "activity (IPCC confirmed), the same industrial chemicals responsible are also linked "
            "to neurological damage in children at levels currently deemed 'safe' by the EPA. "
            "The FDA-approved fluoride in your drinking water has been reclassified as a "
            "'developmental neurotoxin' in a 2024 court ruling — yet water fluoridation continues. "
            "Meanwhile, a Harvard meta-analysis found IQ reductions in high-fluoride areas. "
            "Coffee drinkers may have health benefits, but the real question is: what else are "
            "they not telling us? Source: telegram_channel/healthtruth2024 | Verify at: "
            "beforeitsnews.com/fluoride-cover-up #WaterSafety #ChemicalExposure #HealthFreedom"
        ),
        "ground_truth_label": Label.FAKE,
        "ground_truth_alert": AlertLevel.RED,
        "key_claims": [
            "climate change is caused by human activity",          # TRUE - used as credibility hook
            "water fluoridation causes brain damage",              # FALSE - debunked
            "exercise improves mental health",                     # TRUE - irrelevant injection
            "new study suggests moderate coffee consumption has health benefits",  # LIKELY TRUE
            "election results were manipulated",                   # CONTESTED - not in post but pattern
            "government hiding alien contact",                     # not claimed but pattern test
        ],
        "key_source": "telegram_channel",
        "secondary_source": "beforeitsnews.com",
        "expected_patterns": ["GOVERNMENT COVER-UP"],
        "max_steps": 10,
        "required_actions": [
            "analyze_claim", "analyze_claim", "check_source",
            "cross_verify", "cross_verify", "raise_alert"
        ],
        "pass_threshold": 0.55,
        "hard_mode_notes": (
            "The post embeds 2 true claims (climate change, coffee benefits) to appear legitimate. "
            "Central claim (fluoride brain damage) is false. Sources are low-credibility "
            "(telegram_channel=0.10, beforeitsnews.com=0.01). "
            "Agent must not be fooled by the true anchors and must detect the fake central claim."
        ),
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Return task definition by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[str]:
    """Return list of available task IDs."""
    return list(TASKS.keys())


def get_knowledge_verdict(claim: str) -> Optional[Dict[str, Any]]:
    """
    Look up a claim in the knowledge base (deterministic).
    Returns verdict dict or None if unknown.
    """
    claim_lower = claim.lower().strip()
    for key, value in KNOWLEDGE_BASE.items():
        if key in claim_lower or claim_lower in key:
            return {"claim": claim, "kb_key": key, **value}
    # Fuzzy: check if any major keywords overlap
    claim_words = set(claim_lower.split())
    for key, value in KNOWLEDGE_BASE.items():
        key_words = set(key.split())
        overlap = claim_words & key_words
        if len(overlap) >= 3:
            return {"claim": claim, "kb_key": key, "partial_match": True, **value}
    return None


def check_source_credibility(source: str) -> Dict[str, Any]:
    """Return credibility info for a source (deterministic)."""
    source_lower = source.lower().strip()
    for key, value in SOURCE_CREDIBILITY.items():
        if key in source_lower or source_lower in key:
            return {"source": source, **value}
    return {"source": source, **SOURCE_CREDIBILITY["unknown_source"]}


def detect_patterns(text: str) -> List[Dict[str, Any]]:
    """Detect fake news linguistic patterns in text (deterministic)."""
    text_upper = text.upper()
    found = []
    for p in FAKE_PATTERNS:
        if p["pattern"] in text_upper:
            found.append(p)
    return found