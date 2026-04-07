"""
manual_test.py
Interactive CLI for manually testing the Fake News Detection pipeline.

Usage:
    python manual_test.py

Features:
    - Difficulty selection menu (easy / medium / hard / custom)
    - Multiple sample posts per difficulty from test_cases.py
    - Custom post input mode
    - Full detection pipeline (same as env.py — no RL loop)
    - Structured output: label, alert, confidence, reasons
    - Loop until user exits
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# ── Allow imports from project root ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rewards import FakeNewsDetector
from test_cases import TEST_CASES, get_by_difficulty

# ═══════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════

# Alert level → coloured label for terminal
ALERT_DISPLAY: Dict[str, str] = {
    "RED":    "🔴 RED",
    "YELLOW": "🟡 YELLOW",
    "GREEN":  "🟢 GREEN",
}

LABEL_DISPLAY: Dict[str, str] = {
    "fake":        "FAKE",
    "likely_fake": "LIKELY FAKE",
    "suspicious":  "SUSPICIOUS",
    "real":        "REAL",
    "unknown":     "UNKNOWN",
}

DIFFICULTY_DISPLAY = {
    "easy":        "🟢 Easy",
    "medium":      "🟡 Medium",
    "hard":        "🔴 Hard",
    "edge":        "⚠️  Edge Case",
    "adversarial": "🧨 Adversarial",
}


def _separator(char: str = "─", width: int = 65) -> str:
    return char * width


def _header(title: str) -> None:
    print("\n" + _separator("═"))
    print(f"  {title}")
    print(_separator("═"))


def _section(title: str) -> None:
    print("\n" + _separator())
    print(f"  {title}")
    print(_separator())


def _print_result(
    post_text: str,
    label: str,
    alert: str,
    confidence: float,
    fake_score: float,
    flags: List[str],
    source_info: Dict[str, Any],
    claim_results: List[Dict[str, Any]],
    claim_summary: str,
    pattern_explanation: str,
    virality_risk: str,
) -> None:
    """Print the full structured detection result."""

    _section("DETECTION RESULT")

    # ── POST ──────────────────────────────────────────────────────
    print(f"\n  📄 POST:")
    # Wrap long text at 60 chars
    words = post_text.split()
    line = "     "
    for word in words:
        if len(line) + len(word) + 1 > 65:
            print(line)
            line = "     " + word
        else:
            line += (" " if line.strip() else "") + word
    if line.strip():
        print(line)

    # ── RESULT ───────────────────────────────────────────────────
    print(f"\n  🏷️  RESULT:")
    print(f"     Label      : {LABEL_DISPLAY.get(label, label.upper())}")
    print(f"     Alert      : {ALERT_DISPLAY.get(alert, alert)}")
    print(f"     Confidence : {confidence:.0%}")
    print(f"     Fake Score : {fake_score:.2f} / 1.00")
    print(f"     Virality   : {virality_risk.upper()}")

    # ── REASONS ──────────────────────────────────────────────────
    print(f"\n  📋 REASONS:")
    reasons: List[str] = []

    # Pattern flags
    if flags:
        for f in flags:
            reasons.append(f"Suspicious pattern detected: '{f}'")
    else:
        reasons.append("No suspicious language patterns detected")

    # Source credibility
    if source_info:
        for src, info in source_info.items():
            tier = info.get("tier", "unknown")
            score = info.get("score", 0.0)
            known = info.get("known", False)
            if not known:
                reasons.append(f"Source '{src}' is unknown / unverified")
            elif tier == "unreliable":
                reasons.append(f"Source '{src}' is UNRELIABLE (credibility: {score:.0%})")
            elif tier == "low":
                reasons.append(f"Source '{src}' has LOW credibility ({score:.0%})")
            elif tier in ("medium",):
                reasons.append(f"Source '{src}' has MEDIUM credibility ({score:.0%})")
            else:
                reasons.append(f"Source '{src}' is credible — tier: {tier} ({score:.0%})")
    else:
        reasons.append("No source cited — moderate suspicion applied")

    # Claim results
    if claim_results:
        for r in claim_results:
            verdict = r.get("verdict", "unknown")
            claim = r.get("claim", "")[:60]
            explanation = r.get("explanation", "")
            if verdict == "false":
                reasons.append(f"❌ Claim DEBUNKED: '{claim}'")
                reasons.append(f"   → {explanation}")
            elif verdict == "true":
                reasons.append(f"✅ Claim VERIFIED: '{claim}'")
            elif verdict == "partial":
                reasons.append(f"⚠️  Claim PARTIALLY TRUE: '{claim}'")
                reasons.append(f"   → {explanation}")
            else:
                reasons.append(f"❓ Claim UNKNOWN (not in knowledge base): '{claim}'")
    else:
        reasons.append("No specific claims found in knowledge base")

    # Virality
    if virality_risk == "high":
        reasons.append("High virality risk — potential for rapid misinformation spread")
    elif virality_risk == "medium":
        reasons.append("Moderate virality risk")

    # Print all reasons
    for i, reason in enumerate(reasons, 1):
        prefix = "   " if reason.startswith("   →") else f"  {i:2d}."
        print(f"{prefix} {reason}")

    # ── CLAIM SUMMARY ────────────────────────────────────────────
    if claim_summary:
        print(f"\n  📊 CLAIM SUMMARY: {claim_summary}")

    print()


# ═══════════════════════════════════════════════════════════════════
# DETECTION RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_detection(
    post_text: str,
    sources: List[str],
    claims: List[str],
    virality_risk: str = "low",
    user_handle: str = "",
    followers: int = 0,
    likes: int = 0,
    shares: int = 0,
) -> None:
    """
    Run the full detection pipeline and print structured output.
    Reuses the exact same FakeNewsDetector used by env.py.
    No RL loop — direct pipeline call.
    """
    detector = FakeNewsDetector()

    # Step 1: Compute fake score (multi-signal)
    score_result = detector.compute_fake_score(
        post_text=post_text,
        extracted_claims=claims,
        sources=sources,
        virality_risk=virality_risk,
    )

    # Step 2: Classify
    classification = detector.classify(
        fake_score=score_result["fake_score"],
        claim_results=score_result["claim_results"],
        source_info=score_result["source_info"],
        flags=score_result["flags"],
    )

    # Step 3: Print structured output
    _print_result(
        post_text=post_text,
        label=classification["label"],
        alert=classification["alert_level"],
        confidence=classification["confidence"],
        fake_score=score_result["fake_score"],
        flags=score_result["flags"],
        source_info=score_result["source_info"],
        claim_results=score_result["claim_results"],
        claim_summary=score_result.get("claim_summary", ""),
        pattern_explanation=score_result.get("explanation", ""),
        virality_risk=virality_risk,
    )

    # Extra context if user info provided
    if user_handle or followers:
        print(f"  👤 Posted by : {user_handle or 'unknown'}")
        print(f"     Followers  : {followers:,}")
        print(f"     Likes      : {likes:,}")
        print(f"     Shares     : {shares:,}")
        if followers > 100000 and classification["label"] in ("fake", "likely_fake"):
            print(
                f"\n  ⚠️  HIGH-REACH FAKE: This post has {followers:,} followers "
                f"and is classified as {classification['label'].upper()}. "
                f"High misinformation risk."
            )
        print()


# ═══════════════════════════════════════════════════════════════════
# MENU HELPERS
# ═══════════════════════════════════════════════════════════════════

def _prompt(text: str) -> str:
    """Print a prompt and return stripped input."""
    try:
        return input(text).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Exiting. Goodbye!")
        sys.exit(0)


def _choose_number(prompt: str, min_val: int, max_val: int) -> int:
    """Loop until user enters a valid integer in [min_val, max_val]."""
    while True:
        raw = _prompt(prompt)
        if raw.isdigit():
            val = int(raw)
            if min_val <= val <= max_val:
                return val
        print(f"  ⚠️  Please enter a number between {min_val} and {max_val}.")


def _show_difficulty_menu() -> str:
    """Show difficulty selection menu. Returns chosen difficulty key."""
    _header("FAKE NEWS DETECTOR — MANUAL TEST")
    print("""
  Select difficulty level:

    1. 🟢 Easy         (obvious fake / obvious real)
    2. 🟡 Medium       (ambiguous claims, mixed sources)
    3. 🔴 Hard         (multi-claim, misleading headlines)
    4. ⚠️  Edge Cases   (empty post, satire, outdated news)
    5. 🧨 Adversarial  (emotional manipulation, source impersonation)
    6. ✏️  Custom Input  (enter your own post)
    """)
    choice = _choose_number("  Your choice (1–6): ", 1, 6)
    mapping = {
        1: "easy",
        2: "medium",
        3: "hard",
        4: "edge",
        5: "adversarial",
        6: "custom",
    }
    return mapping[choice]


def _show_post_menu(difficulty: str) -> Optional[Dict[str, Any]]:
    """
    Show available test posts for a difficulty.
    Returns the chosen test case dict, or None if user picks all.
    """
    cases = get_by_difficulty(difficulty)
    if not cases:
        print(f"\n  No test cases found for difficulty: {difficulty}")
        return None

    _section(f"Available test posts — {DIFFICULTY_DISPLAY.get(difficulty, difficulty)}")
    print()

    for i, tc in enumerate(cases, 1):
        # Truncate post text for preview
        preview = tc["input"]["post_text"][:80].replace("\n", " ")
        if len(tc["input"]["post_text"]) > 80:
            preview += "..."
        print(f"  {i}. [{tc['category']}]")
        print(f"     \"{preview}\"")
        print()

    print(f"  {len(cases)+1}. Run ALL posts in this difficulty")
    print()

    choice = _choose_number(f"  Choose a post (1–{len(cases)+1}): ", 1, len(cases) + 1)

    if choice == len(cases) + 1:
        return None  # Caller will run all
    return cases[choice - 1]


def _get_custom_input() -> Dict[str, Any]:
    """Prompt user to enter a custom post with all fields."""
    _section("CUSTOM INPUT MODE")
    print("  Enter details for your post:\n")

    post_text = _prompt("  Post text: ")
    user_handle = _prompt("  User handle (e.g. @user123) [optional]: ")
    followers_raw = _prompt("  Followers count [0]: ")
    likes_raw = _prompt("  Likes count [0]: ")
    shares_raw = _prompt("  Shares count [0]: ")
    timestamp = _prompt("  Timestamp (e.g. 2025-01-01T10:00:00) [optional]: ")

    print("\n  Sources (comma-separated, e.g. reuters,bbc) [leave blank if none]:")
    sources_raw = _prompt("  Sources: ")
    sources = [s.strip().lower() for s in sources_raw.split(",") if s.strip()]

    print("\n  Claims to verify (comma-separated) [leave blank to auto-detect]:")
    claims_raw = _prompt("  Claims: ")
    claims = [c.strip().lower() for c in claims_raw.split(",") if c.strip()]

    print("\n  Virality risk:")
    print("    1. Low")
    print("    2. Medium")
    print("    3. High")
    virality_choice = _choose_number("  Choice (1–3): ", 1, 3)
    virality_map = {1: "low", 2: "medium", 3: "high"}
    virality_risk = virality_map[virality_choice]

    def _safe_int(val: str) -> int:
        try:
            return int(val)
        except ValueError:
            return 0

    return {
        "input": {
            "post_text": post_text,
            "user_handle": user_handle,
            "followers": _safe_int(followers_raw),
            "likes": _safe_int(likes_raw),
            "shares": _safe_int(shares_raw),
            "timestamp": timestamp,
            "sources": sources,
            "claims": claims,
            "virality_risk": virality_risk,
        },
        "name": "Custom Input",
        "difficulty": "custom",
        "category": "custom",
        "id": 0,
    }


# ═══════════════════════════════════════════════════════════════════
# SINGLE TEST CASE RUNNER
# ═══════════════════════════════════════════════════════════════════

def _run_test_case(tc: Dict[str, Any], show_expected: bool = True) -> None:
    """Run detection on a single test case and print result + expected."""
    inp = tc["input"]

    # Show metadata
    print(f"\n  📌 Test #{tc['id']} — {tc['name']}")
    print(f"     Difficulty : {DIFFICULTY_DISPLAY.get(tc['difficulty'], tc['difficulty'])}")
    print(f"     Category   : {tc['category']}")
    if inp.get("user_handle"):
        print(f"     Handle     : {inp['user_handle']}  |  "
              f"Followers: {inp.get('followers', 0):,}  |  "
              f"Shares: {inp.get('shares', 0):,}")

    # Run detection
    run_detection(
        post_text=inp["post_text"],
        sources=inp.get("sources", []),
        claims=inp.get("claims", []),
        virality_risk=inp.get("virality_risk", "low"),
        user_handle=inp.get("user_handle", ""),
        followers=inp.get("followers", 0),
        likes=inp.get("likes", 0),
        shares=inp.get("shares", 0),
    )

    # Show expected (for non-custom cases)
    if show_expected and "expected" in tc:
        exp = tc["expected"]
        print(f"  🎯 EXPECTED:")
        print(f"     Label : {LABEL_DISPLAY.get(exp['label'], exp['label'].upper())}")
        print(f"     Alert : {ALERT_DISPLAY.get(exp['alert'], exp['alert'])}")
        if exp.get("reason_contains"):
            print(f"     Hints : {', '.join(exp['reason_contains'])}")
        print()


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    """Main interactive loop."""

    while True:
        # ── Step 1: Difficulty selection ──────────────────────────
        difficulty = _show_difficulty_menu()

        if difficulty == "custom":
            # ── Step 2a: Custom input mode ────────────────────────
            tc = _get_custom_input()
            _run_test_case(tc, show_expected=False)

        else:
            # ── Step 2b: Predefined post selection ────────────────
            chosen_tc = _show_post_menu(difficulty)

            if chosen_tc is None:
                # Run ALL posts for this difficulty
                all_cases = get_by_difficulty(difficulty)
                print(f"\n  Running all {len(all_cases)} {difficulty} test cases...\n")
                for tc in all_cases:
                    _run_test_case(tc, show_expected=True)
                    cont = _prompt("  Press Enter for next post (or 'q' to stop): ")
                    if cont.lower() == "q":
                        break
            else:
                # Run just the chosen one
                _run_test_case(chosen_tc, show_expected=True)

        # ── Step 5: Loop ──────────────────────────────────────────
        again = _prompt("\n  Do you want to test another post? (y/n): ")
        if again.lower() not in ("y", "yes"):
            print("\n  Thanks for testing. Goodbye! 👋\n")
            break


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()