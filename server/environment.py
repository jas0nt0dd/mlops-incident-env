"""
environment.py — MLOps Incident Response Environment v2
Upgrades over v1:
  - Graders now receive investigation_path + step_count for process-gated scoring
  - Red herring components in all scenarios mislead weak agents
  - SLA countdown visible in observation (urgency pressure)
  - Cascade failure task (4th task) — multi-root-cause requiring synthesis
  - Step efficiency tracked and passed to graders
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"

TASK_MAX_STEPS: Dict[str, int] = {
    "easy": 10,
    "medium": 15,
    "hard": 25,
    "cascade": 30,
}

VALID_ACTIONS = [
    "inspect", "querylogs", "checkmetrics",
    "compareconfigs", "checkfeaturedrift",
    "submitdiagnosis", "requestrollback",
]


@dataclass
class ObsPayload:
    goal: str
    alert_summary: str
    component_status: Dict[str, str] = field(default_factory=dict)
    recent_logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    action_feedback: str = ""
    step_count: int = 0
    sla_minutes_remaining: Optional[int] = None
    reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    final_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "alert_summary": self.alert_summary,
            "component_status": self.component_status,
            "recent_logs": self.recent_logs,
            "metrics_snapshot": self.metrics_snapshot,
            "action_feedback": self.action_feedback,
            "step_count": self.step_count,
            "sla_minutes_remaining": self.sla_minutes_remaining,
            "reward": self.reward,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
            "final_score": self.final_score,
            "score_breakdown": self.score_breakdown,
        }


@dataclass
class EpisodeState:
    episode_id: str
    task_id: str
    true_root_cause: str
    root_cause_keywords: List[str] = field(default_factory=list)
    investigation_path: List[str] = field(default_factory=list)
    partial_score: float = 0.0
    step_count: int = 0
    max_steps: int = 20
    sla_steps: Optional[int] = None          # steps before SLA breach warning
    diagnosis_submitted: bool = False
    relevant_components_visited: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "investigation_path": self.investigation_path,
            "partial_score": self.partial_score,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "diagnosis_submitted": self.diagnosis_submitted,
            "relevant_components_visited": self.relevant_components_visited,
        }


class MLOpsEnvironment:
    """Simulates a production ML system under incident conditions."""

    def __init__(self) -> None:
        self.scenario: Dict[str, Any] = {}
        self.state: EpisodeState = EpisodeState("", "", "")
        self.cumulative_reward: float = 0.0
        self.seen_actions: set = set()

    # ─── Public OpenEnv interface ────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> ObsPayload:
        if task_id not in TASK_MAX_STEPS:
            task_id = "easy"
        scenario_path = DATA_DIR / f"{task_id}_scenario.json"
        with open(scenario_path, encoding="utf-8") as fh:
            self.scenario = json.load(fh)

        self.state = EpisodeState(
            episode_id=str(uuid.uuid4())[:8],
            task_id=task_id,
            true_root_cause=self.scenario["root_cause"],
            root_cause_keywords=self.scenario.get("root_cause_keywords", []),
            investigation_path=[],
            partial_score=0.0,
            step_count=0,
            max_steps=TASK_MAX_STEPS[task_id],
            sla_steps=self.scenario.get("sla_steps"),
            diagnosis_submitted=False,
            relevant_components_visited=[],
        )
        self.cumulative_reward = 0.0
        self.seen_actions = set()

        components = self.scenario.get("components", {})
        return ObsPayload(
            goal=self.scenario["goal"],
            alert_summary=self.scenario["alert_summary"],
            component_status={k: v["status"] for k, v in components.items()},
            recent_logs=[],
            metrics_snapshot=self.scenario.get("global_metrics", {}),
            action_feedback=(
                f"Incident opened. Begin your investigation.\n"
                f"Available components: {', '.join(components.keys())}\n"
                f"Available actions: {', '.join(VALID_ACTIONS)}"
            ),
            step_count=0,
            sla_minutes_remaining=self.scenario.get("sla_minutes"),
            reward=0.0,
            cumulative_reward=0.0,
            done=False,
        )

    def step(self, action_type: str, target: str, parameters: Dict[str, Any]) -> ObsPayload:
        self.state.step_count += 1
        action_key = f"{action_type}:{target}"
        self.state.investigation_path.append(action_key)

        reward = 0.0
        done = False
        feedback = ""
        logs: List[Dict] = []
        metrics: Dict = {}
        final_score = None
        score_breakdown = None
        reward_cfg = self.scenario.get("reward_config", {})

        # SLA countdown
        sla_remaining = None
        if self.scenario.get("sla_minutes") and self.state.sla_steps:
            steps_left = self.state.sla_steps - self.state.step_count
            sla_remaining = max(0, steps_left * 2)  # 2 min per step approx

        # Loop detection
        if action_key in self.seen_actions and action_type != "submitdiagnosis":
            reward = reward_cfg.get("loop_penalty", -0.15)
            feedback = f"⚠ Already ran {action_type}:{target}. No new info. Explore a different component."
        else:
            self.seen_actions.add(action_key)
            if action_type == "inspect":
                reward, feedback, logs, metrics = self._do_inspect(target, reward_cfg)
            elif action_type == "querylogs":
                reward, feedback, logs, metrics = self._do_query_logs(target, reward_cfg)
            elif action_type == "checkmetrics":
                reward, feedback, logs, metrics = self._do_check_metrics(target, reward_cfg)
            elif action_type == "compareconfigs":
                reward, feedback, logs, metrics = self._do_compare_configs(target, reward_cfg)
            elif action_type == "checkfeaturedrift":
                reward, feedback, logs, metrics = self._do_feature_drift(reward_cfg)
            elif action_type == "submitdiagnosis":
                reward, feedback, done, final_score, score_breakdown = self._do_diagnosis(
                    target, parameters, reward_cfg
                )
            elif action_type == "requestrollback":
                reward, feedback = self._do_rollback(target)
            else:
                feedback = f"Unknown action '{action_type}'. Valid: {', '.join(VALID_ACTIONS)}"

        # Step limit
        if self.state.step_count >= self.state.max_steps and not done:
            reward += reward_cfg.get("step_over_max_penalty", -0.05)
            done = True
            feedback += f"\n⏰ Max steps ({self.state.max_steps}) reached. Episode ending."

        self.cumulative_reward += reward
        self.state.partial_score = round(self.cumulative_reward, 4)

        components = self.scenario.get("components", {})
        return ObsPayload(
            goal=self.scenario["goal"],
            alert_summary=self.scenario["alert_summary"],
            component_status={k: v["status"] for k, v in components.items()},
            recent_logs=logs,
            metrics_snapshot=metrics if metrics else self.scenario.get("global_metrics", {}),
            action_feedback=feedback,
            step_count=self.state.step_count,
            sla_minutes_remaining=sla_remaining,
            reward=round(reward, 4),
            cumulative_reward=round(self.cumulative_reward, 4),
            done=done,
            final_score=final_score,
            score_breakdown=score_breakdown,
        )

    @property
    def state_info(self) -> EpisodeState:
        return self.state

    # ─── Action handlers ─────────────────────────────────────────────────────

    def _do_inspect(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self.scenario.get("components", {})
        if target not in components:
            return 0.0, f"Unknown component '{target}'. Available: {', '.join(components.keys())}", [], {}
        comp = components[target]
        reward = 0.0
        relevant = reward_cfg.get("relevant_components", [])
        if target in relevant and target not in self.state.relevant_components_visited:
            reward = reward_cfg.get("explore_relevant_reward", 0.05)
            self.state.relevant_components_visited.append(target)

        has_config = "config_diff" in self.scenario and self.scenario["config_diff"].get("service") == target
        has_drift = "feature_drift_data" in self.scenario
        extra = ", compareconfigs" if has_config else ""
        feedback = (
            f"COMPONENT: {target.upper()}\n"
            f"Status: {comp['status'].upper()}\n"
            f"Description: {comp['description']}\n"
            f"Last updated: {comp.get('last_updated', 'unknown')}\n"
            f"Next steps: querylogs:{target}, checkmetrics:{target}{extra}"
            + (", checkfeaturedrift:feature_store" if has_drift else "")
        )
        return reward, feedback, comp.get("logs", [])[-2:], comp.get("metrics", {})

    def _do_query_logs(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self.scenario.get("components", {})
        if target not in components:
            return 0.0, f"Unknown component '{target}'. Available: {', '.join(components.keys())}", [], {}
        comp = components[target]
        logs = comp.get("logs", [])
        reward = 0.0
        key_evidence_logs = reward_cfg.get("key_evidence_logs", [])
        if target in key_evidence_logs:
            reward = reward_cfg.get("find_key_evidence_reward", 0.10)
        elif target in reward_cfg.get("relevant_components", []):
            reward = reward_cfg.get("explore_relevant_reward", 0.05) * 0.5

        formatted = "\n".join(
            f"  [{e.get('time','?')}] {e.get('level','INFO'):8} {e.get('msg','')}"
            for e in logs
        ) or "  (no logs available)"
        feedback = f"LOGS: {target} ({len(logs)} entries)\n{formatted}"
        return reward, feedback, logs, comp.get("metrics", {})

    def _do_check_metrics(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self.scenario.get("components", {})
        if target not in components:
            return 0.0, f"Unknown component '{target}'.", [], {}
        comp = components[target]
        metrics = comp.get("metrics", {})
        reward = reward_cfg.get("explore_relevant_reward", 0.05) * 0.5
        rows = "\n".join(f"  {k}: {v}" for k, v in metrics.items()) or "  (no metrics)"
        feedback = f"METRICS: {target}\n{rows}"
        return reward, feedback, [], metrics

    def _do_compare_configs(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        if "config_diff" not in self.scenario:
            return 0.0, "No config diff recorded for this incident.", [], {}
        diff = self.scenario["config_diff"]
        reward = reward_cfg.get("find_config_diff_reward", 0.15)
        prev = diff.get("previous_config", {})
        curr = diff.get("current_config", {})
        all_keys = sorted(set(list(prev.keys()) + list(curr.keys())))
        changes = [
            f"  {k}: {prev.get(k,'—')} → {curr.get(k,'—')}"
            for k in all_keys if prev.get(k) != curr.get(k)
        ]
        feedback = (
            f"CONFIG DIFF: {diff.get('service', target)}\n"
            f"Deployed: {diff.get('deployment_time','?')} by {diff.get('author','?')}\n"
            f"PR: {diff.get('pr_title','N/A')}\n"
            f"Changes:\n" + ("\n".join(changes) or "  (no changes detected)") +
            f"\nNote: {diff.get('note','')}"
        )
        return reward, feedback, [], {}

    def _do_feature_drift(self, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        if "feature_drift_data" not in self.scenario:
            return 0.0, "Feature drift data not available for this task.", [], {}
        drift = self.scenario["feature_drift_data"]
        features = drift.get("features", {})
        importance = drift.get("model_feature_importance", {})
        reward = reward_cfg.get("find_feature_drift_reward", 0.15)
        critical = [f for f, d in features.items() if d.get("status") == "CRITICAL_DRIFT"]
        if critical:
            reward = reward_cfg.get("find_critical_feature_reward", 0.20)
        header = f"{'Feature':<42} {'PSI':>6} {'Status':<22} {'Importance':>10}"
        rows = "\n".join(
            f"  {fname:<42} {fdata['psi']:>6.2f} {fdata.get('status','?'):<22} {importance.get(fname, 0):>10.2f}"
            for fname, fdata in features.items()
        )
        feedback = (
            f"FEATURE DRIFT REPORT\n{drift.get('summary','')}\n"
            f"{header}\n{'-'*84}\n{rows}\n"
            f"Threshold: PSI > 0.20 = significant drift | PSI > 0.25 = CRITICAL"
        )
        return reward, feedback, [], {}

    def _do_diagnosis(self, target: str, parameters: Dict[str, Any], reward_cfg: Dict
                      ) -> Tuple[float, str, bool, float, Dict]:
        if self.state.diagnosis_submitted:
            return (reward_cfg.get("false_diagnosis_penalty", -0.10),
                    "Diagnosis already submitted. Only ONE submission per episode.", True, 0.0, {})

        self.state.diagnosis_submitted = True

        try:
            from server.tasks import GRADERS
        except ImportError:
            from tasks import GRADERS

        # Pass investigation context to grader for process-gated scoring
        result = GRADERS[self.state.task_id].grade(
            target, parameters,
            investigation_path=self.state.investigation_path,
            step_count=self.state.step_count,
        )
        final_score = result["total"]
        breakdown = result["breakdown"]

        if final_score >= 0.9:
            reward = reward_cfg.get("correct_diagnosis_reward", 0.50)
            verdict = "🎯 EXCELLENT — Root cause correctly and fully identified!"
        elif final_score >= 0.5:
            reward = reward_cfg.get("partial_diagnosis_reward", 0.25)
            verdict = "⚡ PARTIAL — Correct direction but missing key details."
        else:
            reward = reward_cfg.get("false_diagnosis_penalty", -0.10)
            verdict = "❌ INCORRECT — Review the evidence more carefully."

        breakdown_text = "\n".join(f"  {k}: {v:.4f}" for k, v in breakdown.items())
        feedback = (
            f"DIAGNOSIS SUBMITTED\n{verdict}\n"
            f"Final score: {final_score:.4f}\n"
            f"Steps used: {self.state.step_count}/{self.state.max_steps}\n"
            f"Score breakdown:\n{breakdown_text}"
        )
        return reward, feedback, True, final_score, breakdown

    def _do_rollback(self, target: str) -> Tuple[float, str]:
        return 0.05, (
            f"ROLLBACK: {target} — Simulation restoring to previous config.\n"
            f"Note: You must still submitdiagnosis to close the episode."
        )
