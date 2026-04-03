
"""
environment.py — Core MLOps Incident Response Environment.

Implements the OpenEnv Environment interface:
  reset(task_id)  → ObsPayload   (initial state)
  step(action)    → ObsPayload   (new state + reward + done)
  state           → EpisodeState (current episode metadata)
"""
from __future__ import annotations
from tasks import GRADERS
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"

TASK_MAX_STEPS: Dict[str, int] = {
    "easy":   10,
    "medium": 15,
    "hard":   25,
}

VALID_ACTIONS = [
    "inspect",
    "query_logs",
    "check_metrics",
    "compare_configs",
    "check_feature_drift",
    "submit_diagnosis",
    "request_rollback",
]


# ── Internal dataclasses (server-side only) ───────────────────────────────────
@dataclass
class ObsPayload:
    goal:              str                       = ""
    alert_summary:     str                       = ""
    component_status:  Dict[str, str]            = field(default_factory=dict)
    recent_logs:       List[Dict[str, Any]]      = field(default_factory=list)
    metrics_snapshot:  Dict[str, Any]            = field(default_factory=dict)
    action_feedback:   str                       = ""
    step_count:        int                       = 0
    reward:            float                     = 0.0
    cumulative_reward: float                     = 0.0
    done:              bool                      = False
    final_score:       Optional[float]           = None
    score_breakdown:   Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal":              self.goal,
            "alert_summary":     self.alert_summary,
            "component_status":  self.component_status,
            "recent_logs":       self.recent_logs,
            "metrics_snapshot":  self.metrics_snapshot,
            "action_feedback":   self.action_feedback,
            "step_count":        self.step_count,
            "reward":            self.reward,
            "cumulative_reward": self.cumulative_reward,
            "done":              self.done,
            "final_score":       self.final_score,
            "score_breakdown":   self.score_breakdown,
        }


@dataclass
class EpisodeState:
    episode_id:                  str       = ""
    task_id:                     str       = ""
    true_root_cause:             str       = ""
    root_cause_keywords:         List[str] = field(default_factory=list)
    investigation_path:          List[str] = field(default_factory=list)
    partial_score:               float     = 0.0
    step_count:                  int       = 0
    max_steps:                   int       = 20
    diagnosis_submitted:         bool      = False
    relevant_components_visited: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id":                  self.episode_id,
            "task_id":                     self.task_id,
            "investigation_path":          self.investigation_path,
            "partial_score":               self.partial_score,
            "step_count":                  self.step_count,
            "max_steps":                   self.max_steps,
            "diagnosis_submitted":         self.diagnosis_submitted,
            "relevant_components_visited": self.relevant_components_visited,
            # true_root_cause intentionally excluded
        }


# ── Main Environment Class ────────────────────────────────────────────────────
class MLOpsEnvironment:
    """
    Simulates a production ML system under incident.
    Loads scenario data from JSON files and drives the episode.
    """

    def __init__(self) -> None:
        self._scenario:          Dict[str, Any] = {}
        self._state:             EpisodeState   = EpisodeState()
        self._cumulative_reward: float          = 0.0
        self._seen_actions:      set            = set()

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, task_id: str = "easy") -> ObsPayload:
        if task_id not in TASK_MAX_STEPS:
            task_id = "easy"

        scenario_path = DATA_DIR / f"{task_id}_scenario.json"
        with open(scenario_path, encoding="utf-8") as fh:
            self._scenario = json.load(fh)

        self._state = EpisodeState(
            episode_id=str(uuid.uuid4())[:8],
            task_id=task_id,
            true_root_cause=self._scenario["root_cause"],
            root_cause_keywords=self._scenario.get("root_cause_keywords", []),
            investigation_path=[],
            partial_score=0.0,
            step_count=0,
            max_steps=TASK_MAX_STEPS[task_id],
            diagnosis_submitted=False,
            relevant_components_visited=[],
        )
        self._cumulative_reward = 0.0
        self._seen_actions = set()

        components = self._scenario.get("components", {})

        return ObsPayload(
            goal=self._scenario["goal"],
            alert_summary=self._scenario["alert_summary"],
            component_status={k: v["status"] for k, v in components.items()},
            recent_logs=[],
            metrics_snapshot=self._scenario.get("global_metrics", {}),
            action_feedback=(
                "🚨 Incident opened. Begin your investigation.\n"
                f"Available components: {', '.join(components.keys())}\n"
                f"Available actions: {', '.join(VALID_ACTIONS)}"
            ),
            step_count=0,
            reward=0.0,
            cumulative_reward=0.0,
            done=False,
        )

    # ── step ──────────────────────────────────────────────────────────────────
    def step(
        self,
        action_type: str,
        target: str,
        parameters: Dict[str, Any],
    ) -> ObsPayload:
        self._state.step_count += 1
        self._state.investigation_path.append(f"{action_type}({target})")

        reward          = 0.0
        done            = False
        feedback        = ""
        logs: List[Dict] = []
        metrics: Dict    = {}
        final_score      = None
        score_breakdown  = None

        reward_cfg = self._scenario.get("reward_config", {})
        action_key = f"{action_type}::{target}"

        # ── Loop detection ────────────────────────────────────────────────────
        if action_key in self._seen_actions and action_type != "submit_diagnosis":
            reward   = reward_cfg.get("loop_penalty", -0.15)
            feedback = (
                f"⚠️  Already ran {action_type}({target}). "
                "No new info. Try a different component or action."
            )
        else:
            self._seen_actions.add(action_key)

            if action_type == "inspect":
                reward, feedback, logs, metrics = self._do_inspect(target, reward_cfg)

            elif action_type == "query_logs":
                reward, feedback, logs, metrics = self._do_query_logs(target, reward_cfg)

            elif action_type == "check_metrics":
                reward, feedback, logs, metrics = self._do_check_metrics(target, reward_cfg)

            elif action_type == "compare_configs":
                reward, feedback, logs, metrics = self._do_compare_configs(target, reward_cfg)

            elif action_type == "check_feature_drift":
                reward, feedback, logs, metrics = self._do_feature_drift(reward_cfg)

            elif action_type == "submit_diagnosis":
                reward, feedback, done, final_score, score_breakdown = (
                    self._do_diagnosis(target, parameters, reward_cfg)
                )

            elif action_type == "request_rollback":
                reward, feedback = self._do_rollback(target)

            else:
                feedback = (
                    f"❌ Unknown action '{action_type}'. "
                    f"Valid: {', '.join(VALID_ACTIONS)}"
                )

        # ── Step-limit enforcement ────────────────────────────────────────────
        if self._state.step_count >= self._state.max_steps and not done:
            reward  += reward_cfg.get("step_over_max_penalty", -0.05)
            done     = True
            feedback += f"\n\n⏱️  Max steps ({self._state.max_steps}) reached. Episode ending."

        self._cumulative_reward    += reward
        self._state.partial_score   = round(self._cumulative_reward, 4)

        components = self._scenario.get("components", {})

        return ObsPayload(
            goal=self._scenario["goal"],
            alert_summary=self._scenario["alert_summary"],
            component_status={k: v["status"] for k, v in components.items()},
            recent_logs=logs,
            metrics_snapshot=metrics if metrics else self._scenario.get("global_metrics", {}),
            action_feedback=feedback,
            step_count=self._state.step_count,
            reward=round(reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=done,
            final_score=final_score,
            score_breakdown=score_breakdown,
        )

    # ── state property ────────────────────────────────────────────────────────
    @property
    def state(self) -> EpisodeState:
        return self._state

    # ═════════════════════════════════════════════════════════════════════════
    # Private action handlers
    # ═════════════════════════════════════════════════════════════════════════

    def _do_inspect(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self._scenario.get("components", {})
        if target not in components:
            return (
                0.0,
                f"❌ Unknown component '{target}'.\n"
                f"Available: {', '.join(components.keys())}",
                [], {},
            )

        comp   = components[target]
        reward = 0.0

        relevant = reward_cfg.get("relevant_components", [])
        if target in relevant and target not in self._state.relevant_components_visited:
            reward = reward_cfg.get("explore_relevant_reward", 0.05)
            self._state.relevant_components_visited.append(target)

        has_config = (
            "config_diff" in self._scenario
            and self._scenario["config_diff"].get("service") == target
        )
        has_drift = "feature_drift_data" in self._scenario
        extra     = ", compare_configs" if has_config else ""

        feedback = (
            f"📋  COMPONENT: {target.upper()}\n"
            f"Status     : {comp['status'].upper()}\n"
            f"Description: {comp['description']}\n"
            f"Next steps : query_logs({target}), check_metrics({target}){extra}"
            + (", check_feature_drift(feature_store)" if has_drift else "")
        )

        return reward, feedback, comp.get("logs", [])[-2:], comp.get("metrics", {})

    def _do_query_logs(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self._scenario.get("components", {})
        if target not in components:
            return (
                0.0,
                f"❌ Unknown component '{target}'.\n"
                f"Available: {', '.join(components.keys())}",
                [], {},
            )

        comp   = components[target]
        logs   = comp.get("logs", [])
        reward = 0.0

        if target in reward_cfg.get("key_evidence_logs", []):
            reward = reward_cfg.get("find_key_evidence_reward", 0.10)
        elif target in reward_cfg.get("relevant_components", []):
            reward = reward_cfg.get("explore_relevant_reward", 0.05) * 0.5

        formatted = "\n".join(
            f"  [{e.get('time','?')}] [{e.get('level','INFO')}] {e.get('msg','')}"
            for e in logs
        ) or "  (no logs available)"

        feedback = f"📜  LOGS — {target} ({len(logs)} entries)\n{formatted}"
        return reward, feedback, logs, comp.get("metrics", {})

    def _do_check_metrics(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self._scenario.get("components", {})
        if target not in components:
            return (
                0.0,
                f"❌ Unknown component '{target}'.\n"
                f"Available: {', '.join(components.keys())}",
                [], {},
            )

        comp    = components[target]
        metrics = comp.get("metrics", {})
        reward  = reward_cfg.get("explore_relevant_reward", 0.05) * 0.5

        rows     = "\n".join(f"  {k}: {v}" for k, v in metrics.items()) or "  (no metrics)"
        feedback = f"📊  METRICS — {target}\n{rows}"
        return reward, feedback, [], metrics

    def _do_compare_configs(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        if "config_diff" not in self._scenario:
            return 0.0, "ℹ️  No config diff recorded for this incident.", [], {}

        diff   = self._scenario["config_diff"]
        reward = reward_cfg.get("find_config_diff_reward", 0.15)
        prev   = diff.get("previous_config", {})
        curr   = diff.get("current_config", {})

        all_keys = sorted(set(list(prev.keys()) + list(curr.keys())))
        changes  = [
            f"  {k}: {prev.get(k, 'N/A')}  →  {curr.get(k, 'N/A')}"
            for k in all_keys
            if prev.get(k) != curr.get(k)
        ]

        feedback = (
            f"🔧  CONFIG DIFF — {diff.get('service', target)}\n"
            f"Deployed : {diff.get('deployment_time','?')} by {diff.get('author','?')}\n"
            f"PR title : {diff.get('pr_title','N/A')}\n"
            f"Changes  :\n" + "\n".join(changes or ["  (no changes detected)"]) +
            f"\n\nNote: {diff.get('note','')}"
        )
        return reward, feedback, [], {}

    def _do_feature_drift(self, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        if "feature_drift_data" not in self._scenario:
            return 0.0, "ℹ️  Feature drift data not available for this task.", [], {}

        drift      = self._scenario["feature_drift_data"]
        features   = drift.get("features", {})
        importance = drift.get("model_feature_importance", {})
        reward     = reward_cfg.get("find_feature_drift_reward", 0.15)

        critical = [f for f, d in features.items() if d.get("status") == "CRITICAL_DRIFT"]
        if critical:
            reward += reward_cfg.get("find_critical_feature_reward", 0.20)

        header = f"{'Feature':<42} {'PSI':>6}  {'Status':<22} {'Importance':>10}\n" + "-" * 84
        rows   = "\n".join(
            f"{fname:<42} {fdata['psi']:>6.2f}  {fdata.get('status','?'):<22} "
            f"{importance.get(fname, 0):>10.2f}"
            for fname, fdata in features.items()
        )

        feedback = (
            f"🔍  FEATURE DRIFT REPORT\n"
            f"{drift.get('summary','')}\n\n"
            f"{header}\n{rows}\n\n"
            f"⚠️  Threshold: PSI > 0.20 = significant drift | PSI > 0.25 = CRITICAL"
        )
        return reward, feedback, [], {}

    def _do_diagnosis(
        self,
        target: str,
        parameters: Dict[str, Any],
        reward_cfg: Dict,
    ) -> Tuple[float, str, bool, float, Dict]:
        if self._state.diagnosis_submitted:
            return (
                reward_cfg.get("false_diagnosis_penalty", -0.10),
                "⚠️  Diagnosis already submitted. Only ONE submission per episode.",
                True, 0.0, {},
            )

        self._state.diagnosis_submitted = True

        from tasks import GRADERS
        result      = GRADERS[self._state.task_id].grade(target, parameters)
        final_score = result["total"]
        breakdown   = result["breakdown"]

        if final_score >= 0.9:
            reward  = reward_cfg.get("correct_diagnosis_reward", 0.50)
            verdict = "🎉 EXCELLENT — Root cause correctly and fully identified!"
        elif final_score >= 0.5:
            reward  = reward_cfg.get("partial_diagnosis_reward", 0.25)
            verdict = "✅ PARTIAL — Correct direction but missing some key details."
        else:
            reward  = reward_cfg.get("false_diagnosis_penalty", -0.10)
            verdict = "❌ INCORRECT — Review the evidence more carefully."

        breakdown_text = "\n".join(f"  {k}: {v:.2f}" for k, v in breakdown.items())
        feedback = (
            f"📋  DIAGNOSIS SUBMITTED\n"
            f"Verdict     : {verdict}\n"
            f"Final score : {final_score:.4f}\n"
            f"Breakdown   :\n{breakdown_text}"
        )
        return reward, feedback, True, final_score, breakdown

    def _do_rollback(self, target: str) -> Tuple[float, str]:
        feedback = (
            f"🔄  ROLLBACK requested for: {target}\n"
            f"Simulation: service restoring to previous config.\n"
            f"Note: You must still submit_diagnosis to close the episode."
        )
        return 0.05, feedback