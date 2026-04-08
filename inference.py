"""
Inference script for FakeNews Detection OpenEnv.
Runs an LLM agent against all 3 tasks and produces structured logs.

CRITICAL: Prints [START], [STEP], [END] blocks to stdout with flush=True.
Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN env variables.
"""
import os
import sys
import json
import asyncio
import time
from typing import Any, Dict, List, Optional

# ── Path setup ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

# ─────────────────────────── Config ───────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "hf_placeholder"))

BENCHMARK = "fakenews-detection-openenv-v1"
MAX_STEPS = 8
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["task_easy", "task_medium", "task_hard"]


# ─────────────────────────── Structured Logging ───────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Print [START] block — required by OpenEnv evaluator."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Print [STEP] block — required by OpenEnv evaluator."""
    error_str = f" error={error}" if error else ""
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.4f} done={done}{error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Print [END] block — required by OpenEnv evaluator."""
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards=[{rewards_str}]",
        flush=True,
    )


# ─────────────────────────── Environment Client ───────────────────────────

class EnvClient:
    """
    HTTP client for the FakeNews Detection OpenEnv server.
    Calls the server API or runs env directly (for local runs).
    """

    def __init__(self, task_id: str, use_direct: bool = True):
        self.task_id = task_id
        self.use_direct = use_direct
        self._env = None

    def reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        if self.use_direct:
            from env import FakeNewsEnv
            self._env = FakeNewsEnv(task_id=self.task_id)
            result = self._env.reset()
            return result.model_dump()
        else:
            import urllib.request
            url = f"{os.environ.get('ENV_URL', 'http://localhost:7860')}/reset"
            data = json.dumps({"task_id": self.task_id}).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step."""
        if self.use_direct:
            from models import Action, ActionType, Label
            action = Action(
                action_type=ActionType(action_dict["action_type"]),
                target=action_dict.get("target"),
                reasoning=action_dict.get("reasoning"),
                final_label=Label(action_dict["final_label"]) if action_dict.get("final_label") else None,
                confidence=action_dict.get("confidence"),
            )
            result = self._env.step(action)
            return result.model_dump()
        else:
            import urllib.request
            action_dict["task_id"] = self.task_id
            url = f"{os.environ.get('ENV_URL', 'http://localhost:7860')}/step"
            data = json.dumps(action_dict).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())


# ─────────────────────────── LLM Agent ───────────────────────────

SYSTEM_PROMPT = """You are a fake news detection agent working inside an RL environment.

At each step, you must output a JSON object specifying your next action.

Available action_types:
- analyze_claim: Extract and analyze a claim from the post. Set target to the claim text.
- check_source: Check credibility of a source mentioned in the post. Set target to the source name/URL.
- cross_verify: Cross-verify a claim against the knowledge base. Set target to the claim text.
- raise_alert: Raise a final alert. Set final_label to one of: fake, likely_fake, suspicious, unknown. Set confidence (0.0-1.0).
- mark_safe: Mark the post as real/safe. Set final_label to "real". Set confidence (0.0-1.0).

IMPORTANT RULES:
1. You must eventually call raise_alert or mark_safe to end the episode.
2. Gather evidence first: analyze_claim, check_source, cross_verify.
3. Then make your final decision with raise_alert or mark_safe.
4. Be precise with confidence — overconfidence on wrong answers is penalized.
5. Output ONLY valid JSON. No markdown, no explanation outside JSON.

Output format:
{
  "action_type": "analyze_claim",
  "target": "the specific claim text",
  "reasoning": "why you chose this action",
  "final_label": null,
  "confidence": null
}

For final verdict:
{
  "action_type": "raise_alert",
  "target": null,
  "reasoning": "evidence summary",
  "final_label": "fake",
  "confidence": 0.85
}
"""


def get_agent_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """
    Ask LLM to decide next action given current observation.
    Returns action dict. Falls back to deterministic action on failure.
    """
    post_text = obs.get("post_text", "")
    message = obs.get("message", "")
    fake_score = obs.get("current_fake_score", 0.0)
    alert = obs.get("current_alert", "GREEN")
    claims = obs.get("claims_extracted", [])
    sources = obs.get("sources_checked", [])
    patterns = obs.get("patterns_detected", [])
    cross = obs.get("cross_verifications", [])
    max_steps = obs.get("max_steps", 8)

    user_content = f"""STEP {step}/{max_steps}

POST TO ANALYZE:
{post_text}

CURRENT STATE:
- Fake score: {fake_score:.2f}
- Alert level: {alert}
- Claims extracted ({len(claims)}): {claims}
- Sources checked ({len(sources)}): {[s.get('source', '') for s in sources]}
- Patterns detected ({len(patterns)}): {patterns}
- Cross-verifications ({len(cross)}): {[c.get('verdict', '') + ': ' + c.get('claim', '')[:50] for c in cross]}

SYSTEM: {message}

HISTORY:
{chr(10).join(history[-3:]) if history else 'No history yet.'}

Decide your next action. Output ONLY valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Clean up possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().strip("```").strip()

        action_dict = json.loads(raw)

        # Validate required fields
        if "action_type" not in action_dict:
            raise ValueError("Missing action_type")
        valid_types = [
            "analyze_claim", "check_source", "cross_verify",
            "raise_alert", "mark_safe"
        ]
        if action_dict["action_type"] not in valid_types:
            raise ValueError(f"Invalid action_type: {action_dict['action_type']}")

        return action_dict

    except Exception as exc:
        print(f"[DEBUG] LLM action parse failed at step {step}: {exc}", flush=True)
        # Deterministic fallback based on step number
        return _fallback_action(step, obs, max_steps)


def _fallback_action(step: int, obs: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    """
    Deterministic fallback action when LLM fails.
    Ensures the episode always completes with a reasonable action.
    """
    fake_score = obs.get("current_fake_score", 0.0)
    sources = obs.get("sources_checked", [])
    claims = obs.get("claims_extracted", [])

    if step == 1:
        return {
            "action_type": "analyze_claim",
            "target": obs.get("post_text", "")[:200],
            "reasoning": "Fallback: analyzing main post text.",
            "final_label": None,
            "confidence": None,
        }
    elif step == 2:
        # Check the key source if we can find one
        post = obs.get("post_text", "")
        source = "unknown_source"
        for domain_hint in ["naturalnews", "beforeitsnews", "telegram", "infowars", ".com", ".org"]:
            if domain_hint in post.lower():
                # Extract rough source
                idx = post.lower().find(domain_hint)
                source = post[max(0, idx-5):idx+20].split()[0]
                break
        return {
            "action_type": "check_source",
            "target": source,
            "reasoning": "Fallback: checking source credibility.",
            "final_label": None,
            "confidence": None,
        }
    elif step == 3 and claims:
        return {
            "action_type": "cross_verify",
            "target": claims[0],
            "reasoning": "Fallback: cross-verifying first claim.",
            "final_label": None,
            "confidence": None,
        }
    elif step >= max_steps - 1 or step >= 5:
        # Final verdict based on accumulated evidence
        if fake_score >= 0.6:
            label = "fake"
        elif fake_score >= 0.35:
            label = "likely_fake"
        elif fake_score >= 0.2:
            label = "suspicious"
        else:
            label = "real"

        action_type = "raise_alert" if label != "real" else "mark_safe"
        return {
            "action_type": action_type,
            "target": None,
            "reasoning": f"Fallback: final verdict based on fake_score={fake_score:.2f}.",
            "final_label": label,
            "confidence": 0.70,
        }
    else:
        return {
            "action_type": "cross_verify",
            "target": obs.get("post_text", "")[:100],
            "reasoning": "Fallback: cross-verifying post content.",
            "final_label": None,
            "confidence": None,
        }


# ─────────────────────────── Task Runner ───────────────────────────

def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """
    Run a single task episode with the LLM agent.
    Emits [START], [STEP], [END] to stdout.
    Returns result dict.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"[INFO] Starting task: {task_id}", flush=True)
    print(f"{'='*60}", flush=True)

    env = EnvClient(task_id=task_id, use_direct=True)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        reset_result = env.reset()
        obs = reset_result.get("observation", reset_result)

        print(f"[INFO] Post: {obs.get('post_text', '')[:100]}...", flush=True)

        done = False
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get agent action
            action_dict = get_agent_action(client, obs, step, history)
            action_str = json.dumps(action_dict)

            # Execute step
            error = None
            try:
                step_result = env.step(action_dict)
                obs = step_result.get("observation", {})
                reward = float(step_result.get("reward", 0.0))
                done = bool(step_result.get("done", False))
            except Exception as e:
                error = str(e)
                reward = 0.0
                done = True
                print(f"[DEBUG] Step error: {e}", flush=True)

            rewards.append(reward)
            steps_taken = step

            # ── REQUIRED: [STEP] log ──
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_dict.get('action_type')} → reward {reward:+.4f}"
            )

            if done:
                break

        # Compute episode score
        total_reward = sum(rewards)
        score = min(max(total_reward / max(len(rewards), 1), 0.0), 1.0)

        # Try to get grade result from env
        if env._env is not None and hasattr(env._env, 'get_grade_result'):
            grade = env._env.get_grade_result()
            if grade:
                score = grade.get("score", score)
                success = grade.get("passed", score >= SUCCESS_SCORE_THRESHOLD)
                print(
                    f"[INFO] Grade result: score={score:.4f} passed={success} "
                    f"label={grade.get('agent_output', {}).get('label', 'N/A')}",
                    flush=True
                )
            else:
                success = score >= SUCCESS_SCORE_THRESHOLD
        else:
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        success = False
        score = 0.0

    finally:
        # ── REQUIRED: [END] log ──
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


# ─────────────────────────── Main ───────────────────────────

def main() -> None:
    print("=" * 60, flush=True)
    print("FakeNews Detection OpenEnv — Inference Script", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API Base: {API_BASE_URL}", flush=True)
    print(f"Tasks: {TASKS}", flush=True)
    print("=" * 60, flush=True)

    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    all_results = []

    for task_id in TASKS:
        result = run_task(client, task_id)
        all_results.append(result)
        # Brief pause between tasks
        time.sleep(1)

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("INFERENCE COMPLETE — SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        status = "✓ PASSED" if r["success"] else "✗ FAILED"
        print(
            f"  {status} | {r['task_id']:15s} | score={r['score']:.4f} | steps={r['steps']}",
            flush=True
        )
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    passed = sum(1 for r in all_results if r["success"])
    print(f"\n  Average score: {avg_score:.4f}", flush=True)
    print(f"  Tasks passed:  {passed}/{len(all_results)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()