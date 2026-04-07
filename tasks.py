"""
tasks.py — Member B
Defines 3 tasks (easy / medium / hard) with realistic social media posts.
Registers all tasks into env.TASK_REGISTRY on import.
"""

from __future__ import annotations
from env import register_task

# ═══════════════════════════════════════════════════════════════════
# TASK 1 — EASY
# Post contains obvious fake patterns + known false claim + unreliable source
# Ground truth: fake / RED
# ═══════════════════════════════════════════════════════════════════

TASK_EASY = {
    "task_id": "easy",
    "difficulty": "easy",
    "post_text": (
        "SHARE BEFORE THEY DELETE THIS!!! Scientists have PROVEN that 5G towers spread "
        "COVID-19 and the government is lying about it. They don't want you to know the "
        "truth. A doctor from naturalcures.com confirmed that 5G activates the virus in "
        "your body. Wake up sheeple! Repost immediately before it gets censored by Big Tech!"
    ),
    "ground_truth": "fake",
    "ground_truth_alert": "RED",
    "extracted_claims": [
        "5g towers spread covid-19",
        "the government is lying",
        "5g activates the virus in your body",
    ],
    "sources": ["naturalcures.com"],
    "virality_risk": "high",
    # Hints for grader
    "expected_patterns": [
        "share before they delete",
        "they don't want you to know",
        "wake up sheeple",
        "the government is lying",
        "censored by",
        "repost immediately",
    ],
    "expected_source_tier": "unreliable",
    "description": (
        "A straightforward fake post containing multiple red-flag patterns, "
        "a clearly false claim (5G-COVID link), and an unreliable source. "
        "An agent should be able to detect this with minimal investigation steps."
    ),
}

# ═══════════════════════════════════════════════════════════════════
# TASK 2 — MEDIUM
# Post contains partial truths + mixed source credibility + subtle manipulation
# Ground truth: likely_fake / RED
# ═══════════════════════════════════════════════════════════════════

TASK_MEDIUM = {
    "task_id": "medium",
    "difficulty": "medium",
    "post_text": (
        "New study reveals that sugar causes hyperactivity in children — doctors have "
        "been hiding this from parents for years! The Daily Mail and ZeroHedge both "
        "reported that Big Pharma is suppressing natural cures for ADHD. "
        "Social media companies sell your data to pharmaceutical companies to push "
        "medication on your kids. Shocking truth: 100% proven by independent researchers. "
        "Share with everyone you love before this gets taken down!"
    ),
    "ground_truth": "likely_fake",
    "ground_truth_alert": "RED",
    "extracted_claims": [
        "sugar causes hyperactivity in children",
        "big pharma is suppressing natural cures",
        "social media companies sell your data",
        "doctors have been hiding this from parents",
    ],
    "sources": ["daily mail", "zerohedge"],
    "virality_risk": "medium",
    "expected_patterns": [
        "big pharma hiding",
        "shocking truth",
        "100% proven",
        "share with everyone you love",
    ],
    "expected_source_tier": "mixed",  # daily mail=medium, zerohedge=low
    "description": (
        "A medium-difficulty post that mixes partial truths (sugar-hyperactivity is debated, "
        "social media data sharing is partially true) with clear misinformation "
        "(suppressed cures). Sources are of mixed credibility. "
        "Agent must cross-verify claims carefully to distinguish partial truths."
    ),
}

# ═══════════════════════════════════════════════════════════════════
# TASK 3 — HARD
# Post appears credible — cites real sources, uses measured language,
# but embeds a dangerous false claim. No clickbait patterns.
# Ground truth: suspicious / YELLOW
# ═══════════════════════════════════════════════════════════════════

TASK_HARD = {
    "task_id": "hard",
    "difficulty": "hard",
    "post_text": (
        "According to a Reuters analysis and WHO data, climate change is supported by "
        "scientific consensus. However, a growing number of independent researchers "
        "published in Nature argue that vaccines cause autism in genetically predisposed "
        "children — a finding that has been quietly acknowledged by the CDC but not yet "
        "publicly disclosed. The human body has 206 bones and drinking water is essential "
        "for survival — these are facts everyone agrees on. The question is: why is the "
        "CDC withholding vaccine-autism data from the public?"
    ),
    "ground_truth": "suspicious",
    "ground_truth_alert": "YELLOW",
    "extracted_claims": [
        "climate change is supported by scientific consensus",
        "vaccines cause autism in genetically predisposed children",
        "the cdc is withholding vaccine-autism data",
        "the human body has 206 bones",
        "drinking water is essential for human survival",
    ],
    "sources": ["reuters", "who", "nature", "cdc"],
    "virality_risk": "medium",
    "expected_patterns": [],   # No obvious clickbait — this is the hard part
    "expected_source_tier": "high",  # Legitimate sources cited — misleading!
    "description": (
        "A hard post that uses legitimate high-credibility sources (Reuters, WHO, Nature, CDC) "
        "and mixes true verified claims with a dangerous false claim (vaccines-autism link). "
        "No clickbait patterns are present. The agent must cross-verify every claim carefully "
        "and not be misled by the credible sources cited alongside the false claim. "
        "Correct classification is 'suspicious' — not 'fake' — because the post "
        "contains both true and false claims, making it a partial-truth misinformation piece."
    ),
}

# ═══════════════════════════════════════════════════════════════════
# REGISTER ALL TASKS
# ═══════════════════════════════════════════════════════════════════

def load_all_tasks() -> None:
    """Register all tasks into the environment task registry."""
    register_task(TASK_EASY)
    register_task(TASK_MEDIUM)
    register_task(TASK_HARD)


# Auto-register on import
load_all_tasks()


# ═══════════════════════════════════════════════════════════════════
# TASK METADATA (used by openenv.yaml and README)
# ═══════════════════════════════════════════════════════════════════

TASK_METADATA = [
    {
        "task_id": "easy",
        "difficulty": "easy",
        "ground_truth": "fake",
        "alert": "RED",
        "virality": "high",
        "num_claims": 3,
        "num_sources": 1,
        "description": TASK_EASY["description"],
    },
    {
        "task_id": "medium",
        "difficulty": "medium",
        "ground_truth": "likely_fake",
        "alert": "RED",
        "virality": "medium",
        "num_claims": 4,
        "num_sources": 2,
        "description": TASK_MEDIUM["description"],
    },
    {
        "task_id": "hard",
        "difficulty": "hard",
        "ground_truth": "suspicious",
        "alert": "YELLOW",
        "virality": "medium",
        "num_claims": 5,
        "num_sources": 4,
        "description": TASK_HARD["description"],
    },
]