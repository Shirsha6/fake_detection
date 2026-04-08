"""
Manual test script for FakeNews Detection OpenEnv.
Run this locally to verify everything works before submission.
Usage: python manual_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FakeNewsEnv
from models import Action, ActionType, Label
from grader import grade_episode
from tasks import list_tasks, get_task


def test_task(task_id: str) -> None:
    print(f"\n{'='*60}")
    print(f"Testing task: {task_id}")
    print('='*60)

    env = FakeNewsEnv(task_id=task_id)
    task = get_task(task_id)

    # 1. Reset
    reset_result = env.reset()
    obs = reset_result.observation
    print(f"[reset] post_id={obs.post_id}")
    print(f"[reset] post_text (first 80 chars): {obs.post_text[:80]}...")
    print(f"[reset] message: {obs.message}")
    assert obs.step_number == 0
    assert obs.current_alert.value == "GREEN"

    # 2. Step 1: analyze_claim
    action1 = Action(
        action_type=ActionType.ANALYZE_CLAIM,
        target=obs.post_text,
        reasoning="Analyzing the full post for claims and patterns.",
    )
    result1 = env.step(action1)
    print(f"\n[step 1 - analyze_claim] reward={result1.reward:.4f}")
    print(f"  patterns: {result1.observation.patterns_detected}")
    print(f"  claims: {result1.observation.claims_extracted}")
    assert result1.reward >= 0

    # 3. Step 2: check_source
    action2 = Action(
        action_type=ActionType.CHECK_SOURCE,
        target=task.get("key_source", "unknown_source"),
        reasoning="Checking the credibility of the source.",
    )
    result2 = env.step(action2)
    print(f"\n[step 2 - check_source] reward={result2.reward:.4f}")
    print(f"  sources: {result2.observation.sources_checked}")

    # 4. Step 3: cross_verify
    action3 = Action(
        action_type=ActionType.CROSS_VERIFY,
        target=task["key_claims"][0] if task.get("key_claims") else obs.post_text[:100],
        reasoning="Cross-verifying key claim against knowledge base.",
    )
    result3 = env.step(action3)
    print(f"\n[step 3 - cross_verify] reward={result3.reward:.4f}")
    print(f"  cross_verifications: {result3.observation.cross_verifications}")
    print(f"  current_fake_score: {result3.observation.current_fake_score:.2f}")
    print(f"  current_alert: {result3.observation.current_alert.value}")

    # 5. Final: raise_alert or mark_safe
    gt_label = task["ground_truth_label"]
    if gt_label.value in ("fake", "likely_fake", "suspicious"):
        action4 = Action(
            action_type=ActionType.RAISE_ALERT,
            final_label=gt_label,
            confidence=0.85,
            reasoning="Evidence strongly suggests fake content.",
        )
    else:
        action4 = Action(
            action_type=ActionType.MARK_SAFE,
            final_label=Label.REAL,
            confidence=0.80,
            reasoning="No evidence of misinformation found.",
        )

    result4 = env.step(action4)
    print(f"\n[step 4 - final verdict] reward={result4.reward:.4f} done={result4.done}")
    assert result4.done is True

    # 6. Check grading
    final_state = env.state()
    grade = env.get_grade_result()
    print(f"\n[grade] score={grade['score']:.4f} passed={grade['passed']}")
    print(f"  components: {grade['components']}")
    print(f"  breakdown:")
    for line in grade['breakdown']:
        print(f"    - {line}")

    assert 0.0 <= grade['score'] <= 1.0
    print(f"\n✓ Task {task_id} test PASSED (score={grade['score']:.4f})")


def test_state() -> None:
    print(f"\n{'='*60}")
    print("Testing state() endpoint")
    print('='*60)
    env = FakeNewsEnv(task_id="task_easy")
    env.reset()
    state = env.state()
    print(f"[state] task_id={state.task_id}")
    print(f"[state] done={state.done}")
    assert state.task_id == "task_easy"
    print("✓ state() test PASSED")


def test_models() -> None:
    print(f"\n{'='*60}")
    print("Testing Pydantic models")
    print('='*60)
    from models import Action, ActionType, Label, AlertLevel, Observation, Reward, EnvState

    a = Action(action_type=ActionType.ANALYZE_CLAIM, target="test claim")
    assert a.action_type == ActionType.ANALYZE_CLAIM
    print("✓ Action model OK")

    r = Reward(total=0.5, explanation="test")
    assert r.total == 0.5
    print("✓ Reward model OK")

    print("✓ All models PASSED")


def test_available_tasks() -> None:
    tasks = list_tasks()
    assert "task_easy" in tasks
    assert "task_medium" in tasks
    assert "task_hard" in tasks
    print(f"✓ Available tasks: {tasks}")


if __name__ == "__main__":
    print("FakeNews Detection OpenEnv — Manual Test Suite")
    print("=" * 60)

    test_models()
    test_available_tasks()
    test_state()

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        test_task(task_id)

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)