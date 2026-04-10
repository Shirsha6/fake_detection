"""
Core FakeNews Detection Environment.
Implements OpenEnv spec: reset(), step(), state()
with typed Pydantic models and deterministic logic.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import copy

from models import (
    Action, ActionType, Observation, Reward, EnvState,
    StepResult, ResetResult, Label, AlertLevel
)
from tasks import (
    get_task, detect_patterns, get_knowledge_verdict,
    check_source_credibility, list_tasks
)
from rewards import RewardCalculator
from grader import grade_episode


# ─────────────────────────── Alert Logic ───────────────────────────

def _compute_alert(fake_score: float, sources_checked: List[Dict]) -> AlertLevel:
    """
    Deterministic alert level from fake score and source signals.
    GREEN  → fake_score < 0.35 and no blacklisted sources
    YELLOW → 0.35 ≤ fake_score < 0.65 or suspicious sources
    RED    → fake_score ≥ 0.65 or blacklisted sources found
    """
    blacklisted_tiers = {"misinformation", "fake_news_site", "conspiracy", "anonymous"}
    has_blacklisted = any(
        s.get("tier", "") in blacklisted_tiers
        for s in sources_checked
    )

    if fake_score >= 0.65 or (has_blacklisted and fake_score >= 0.4):
        return AlertLevel.RED
    elif fake_score >= 0.35 or has_blacklisted:
        return AlertLevel.YELLOW
    else:
        return AlertLevel.GREEN


def _compute_fake_score(
    patterns: List[Dict],
    sources: List[Dict],
    verifications: List[Dict],
    claims: List[str],
) -> float:
    """
    Aggregate fake score from all signals (deterministic, 0.0–1.0).
    """
    score = 0.0
    components = 0

    # Pattern signal (max weight 0.5)
    if patterns:
        pattern_weight = min(1.0, sum(p["weight"] for p in patterns))
        score += pattern_weight * 0.35
        components += 1

    # Source credibility signal (inverted, low credibility = high fake score)
    if sources:
        avg_cred = sum(s.get("credibility", 0.5) for s in sources) / len(sources)
        source_signal = 1.0 - avg_cred
        score += source_signal * 0.35
        components += 1

    # Knowledge base verification signal
    if verifications:
        false_count = sum(
            1 for v in verifications
            if v.get("verdict") in ("false",)
        )
        true_count = sum(
            1 for v in verifications
            if v.get("verdict") in ("true", "likely_true")
        )
        if false_count + true_count > 0:
            kb_signal = false_count / (false_count + true_count + 0.001)
            score += kb_signal * 0.30
            components += 1

    if components == 0:
        return 0.3  # Default uncertain

    return min(1.0, score)


# ─────────────────────────── Environment ───────────────────────────

class FakeNewsEnv:
    """
    Social Media Fake News Detection Environment.
    
    OpenEnv-compliant environment for training and evaluating
    AI agents on multi-signal fake news detection tasks.
    """

    AVAILABLE_ACTIONS = [
        ActionType.ANALYZE_CLAIM.value,
        ActionType.CHECK_SOURCE.value,
        ActionType.CROSS_VERIFY.value,
        ActionType.RAISE_ALERT.value,
        ActionType.MARK_SAFE.value,
    ]

    def __init__(self, task_id: str = "task_easy"):
        self._task_id = task_id
        self._task: Dict[str, Any] = get_task(task_id)
        self._state: Optional[EnvState] = None
        self._reward_calc: Optional[RewardCalculator] = None
        self._grade_result: Optional[Dict[str, Any]] = None

    # ─── OpenEnv Interface ───

    def reset(self) -> ResetResult:
        """
        Reset the environment to initial state.
        Returns initial observation.
        """
        task = self._task
        self._reward_calc = RewardCalculator(task)
        self._grade_result = None

        self._state = EnvState(
            task_id=task["task_id"],
            task_name=task["task_name"],
            post_id=task["post_id"],
            post_text=task["post_text"],
            ground_truth_label=task["ground_truth_label"],
            ground_truth_alert=task["ground_truth_alert"],
            step_number=0,
            max_steps=task["max_steps"],
            done=False,
            claims_extracted=[],
            sources_checked=[],
            patterns_detected=[],
            cross_verifications=[],
            current_fake_score=0.0,
            current_alert=AlertLevel.GREEN,
            final_label=None,
            final_alert=None,
            episode_rewards=[],
            total_reward=0.0,
            agent_actions=[],
        )

        obs = self._build_observation()
        return ResetResult(
            observation=obs,
            info={"task_id": task["task_id"], "difficulty": task["difficulty"]},
        )

    def step(self, action: Action) -> StepResult:
        """
        Execute one step in the environment.
        Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_number += 1
        action_result = self._execute_action(action)
        reward_obj = self._reward_calc.step_reward(action, self._state, action_result)

        # Update state
        self._state.total_reward += reward_obj.total
        self._state.episode_rewards.append(reward_obj.total)
        self._state.agent_actions.append({
            "action_type": action.action_type.value,
            "target": action.target,
            "reasoning": action.reasoning,
            "final_label": action.final_label.value if action.final_label else None,
            "confidence": action.confidence,
            "step": self._state.step_number,
        })

        # Check done conditions
        done = self._check_done(action)
        self._state.done = done

        if done:
            self._grade_result = grade_episode(self._state, self._task)
            # Add final episode bonus
            bonus = self._reward_calc.episode_bonus(self._state)
            self._state.total_reward += bonus
            self._state.episode_rewards.append(bonus)

        obs = self._build_observation()

        info = {
            "action_result": action_result,
            "reward_breakdown": reward_obj.model_dump(),
            "step_number": self._state.step_number,
            "current_fake_score": self._state.current_fake_score,
            "current_alert": self._state.current_alert.value,
        }
        if done and self._grade_result:
            info["grade_result"] = self._grade_result

        return StepResult(
            observation=obs,
            reward=reward_obj.total,
            reward_details=reward_obj,
            done=done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return current full state of the environment."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return copy.deepcopy(self._state)

    # ─── Action Execution ───

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute the action and update internal state. Deterministic."""
        result: Dict[str, Any] = {"action_type": action.action_type.value}

        if action.action_type == ActionType.ANALYZE_CLAIM:
            result.update(self._do_analyze_claim(action))

        elif action.action_type == ActionType.CHECK_SOURCE:
            result.update(self._do_check_source(action))

        elif action.action_type == ActionType.CROSS_VERIFY:
            result.update(self._do_cross_verify(action))

        elif action.action_type == ActionType.RAISE_ALERT:
            result.update(self._do_raise_alert(action))

        elif action.action_type == ActionType.MARK_SAFE:
            result.update(self._do_mark_safe(action))

        # Recompute fake score and alert
        self._state.current_fake_score = _compute_fake_score(
            [{"weight": 0.3} for _ in self._state.patterns_detected],
            self._state.sources_checked,
            self._state.cross_verifications,
            self._state.claims_extracted,
        )
        self._state.current_alert = _compute_alert(
            self._state.current_fake_score,
            self._state.sources_checked,
        )

        return result

    def _do_analyze_claim(self, action: Action) -> Dict[str, Any]:
        """Extract and analyze a claim from the post."""
        post = self._state.post_text
        target = action.target or post

        # Extract key claim text
        claim = target[:200] if len(target) > 200 else target

        # Detect patterns in the full post
        patterns = detect_patterns(post)
        new_pattern_names = [p["pattern"] for p in patterns]
        for p in new_pattern_names:
            if p not in self._state.patterns_detected:
                self._state.patterns_detected.append(p)

        # Look up claim in knowledge base
        kb_verdict = get_knowledge_verdict(claim)

        if claim not in self._state.claims_extracted:
            self._state.claims_extracted.append(claim)

        return {
            "claim_analyzed": claim,
            "patterns_found": new_pattern_names,
            "kb_verdict": kb_verdict,
            "patterns_count": len(new_pattern_names),
        }

    def _do_check_source(self, action: Action) -> Dict[str, Any]:
        """Check credibility of a source."""
        source = action.target or self._task.get("key_source", "unknown_source")
        cred_info = check_source_credibility(source)

        # Avoid duplicate source checks
        already_checked = any(
            s.get("source", "") == source
            for s in self._state.sources_checked
        )
        if not already_checked:
            self._state.sources_checked.append(cred_info)

        return {
            "source_checked": source,
            "credibility": cred_info.get("credibility", 0.5),
            "tier": cred_info.get("tier", "unknown"),
            "bias": cred_info.get("bias", "unknown"),
        }

    def _do_cross_verify(self, action: Action) -> Dict[str, Any]:
        """Cross-verify a claim against the knowledge base."""
        target = action.target or (
            self._state.claims_extracted[-1]
            if self._state.claims_extracted else self._state.post_text
        )
        kb_verdict = get_knowledge_verdict(target)

        verified = False
        contradiction_found = False

        if kb_verdict:
            verified = True
            if kb_verdict.get("verdict") in ("false",):
                contradiction_found = True
            verification_entry = {
                "claim": target,
                "verdict": kb_verdict.get("verdict", "unverifiable"),
                "sources": kb_verdict.get("sources", []),
                "explanation": kb_verdict.get("explanation", ""),
                "contradiction_found": contradiction_found,
            }
            # Avoid duplicates
            existing = [v.get("claim") for v in self._state.cross_verifications]
            if target not in existing:
                self._state.cross_verifications.append(verification_entry)
        else:
            verification_entry = {
                "claim": target,
                "verdict": "unverifiable",
                "sources": [],
                "explanation": "Claim not found in knowledge base.",
                "contradiction_found": False,
            }
            existing = [v.get("claim") for v in self._state.cross_verifications]
            if target not in existing:
                self._state.cross_verifications.append(verification_entry)

        return {
            "verified": verified,
            "contradiction_found": contradiction_found,
            "verdict": kb_verdict.get("verdict") if kb_verdict else "unverifiable",
            "kb_result": kb_verdict,
        }

    def _do_raise_alert(self, action: Action) -> Dict[str, Any]:
        """Raise a final alert with label."""
        label = action.final_label or Label.SUSPICIOUS
        alert = AlertLevel.RED if label == Label.FAKE else \
                AlertLevel.YELLOW if label in (Label.LIKELY_FAKE, Label.SUSPICIOUS) else \
                AlertLevel.GREEN

        self._state.final_label = label
        self._state.final_alert = alert
        self._state.current_alert = alert

        return {
            "final_label": label.value,
            "final_alert": alert.value,
            "action": "alert_raised",
        }

    def _do_mark_safe(self, action: Action) -> Dict[str, Any]:
        """Mark the post as safe/real."""
        label = action.final_label or Label.REAL
        self._state.final_label = label
        self._state.final_alert = AlertLevel.GREEN
        self._state.current_alert = AlertLevel.GREEN

        return {
            "final_label": label.value,
            "final_alert": AlertLevel.GREEN.value,
            "action": "marked_safe",
        }

    # ─── Done Logic ───

    def _check_done(self, action: Action) -> bool:
        """Episode ends when agent raises alert/marks safe, or max steps reached."""
        if action.action_type in (ActionType.RAISE_ALERT, ActionType.MARK_SAFE):
            return True
        if self._state.step_number >= self._state.max_steps:
            # Auto-mark if no final verdict
            if self._state.final_label is None:
                # Use current signal to guess
                score = self._state.current_fake_score
                if score >= 0.65:
                    self._state.final_label = Label.FAKE
                    self._state.final_alert = AlertLevel.RED
                elif score >= 0.35:
                    self._state.final_label = Label.SUSPICIOUS
                    self._state.final_alert = AlertLevel.YELLOW
                else:
                    self._state.final_label = Label.UNKNOWN
                    self._state.final_alert = AlertLevel.GREEN
            return True
        return False

    # ─── Observation Builder ───

    def _build_observation(self) -> Observation:
        """Build observation from current state."""
        s = self._state
        step = s.step_number
        max_s = s.max_steps

        if step == 0:
            msg = (
                f"New task: {s.task_name}. "
                f"Analyze the following post and determine if it is fake news. "
                f"You have {max_s} steps. Start with analyze_claim or check_source."
            )
        elif s.done:
            msg = (
                f"Episode complete. Final label: {s.final_label.value if s.final_label else 'N/A'}. "
                f"Alert: {s.final_alert.value if s.final_alert else 'N/A'}. "
                f"Total reward: {s.total_reward:.3f}."
            )
        else:
            remaining = max_s - step
            msg = (
                f"Step {step}/{max_s} ({remaining} remaining). "
                f"Current fake score: {s.current_fake_score:.2f}. "
                f"Alert: {s.current_alert.value}. "
                f"Claims extracted: {len(s.claims_extracted)}. "
                f"Sources checked: {len(s.sources_checked)}. "
                f"Use raise_alert or mark_safe to finalize."
            )

        return Observation(
            post_id=s.post_id,
            post_text=s.post_text,
            task_description=self._task["description"],
            step_number=step,
            max_steps=max_s,
            claims_extracted=list(s.claims_extracted),
            sources_checked=list(s.sources_checked),
            patterns_detected=list(s.patterns_detected),
            cross_verifications=list(s.cross_verifications),
            current_fake_score=s.current_fake_score,
            current_alert=s.current_alert,
            available_actions=self.AVAILABLE_ACTIONS,
            message=msg,
        )

    # ─── Utilities ───

    @classmethod
    def available_tasks(cls) -> List[str]:
        return list_tasks()

    def get_grade_result(self) -> Optional[Dict[str, Any]]:
        return self._grade_result