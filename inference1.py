"""
inference.py — MLOps Incident Response Agent v4
Fixes in this version:
  - Hard task: FORCE_SUBMIT_AT=10 (submit at step 15, before red-herring spiral)
  - Hard task: diagnosis_readiness READY after 3x check_feature_drift + business metrics
  - best_evidence_target: always returns a valid component from valid_components
  - fallback_diagnosis target: validated against valid_components before use
  - Guard order: force_submit (Guard 4) now runs BEFORE loop_guard (Guard 3)
  - Cascade: check_feature_drift blocked entirely on non-ML components
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import requests
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
API_KEY = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("API_KEY", "")
)
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:8000").rstrip("/")

ENV_NAME = "mlops-incident-env"
TEMPERATURE = 0.15
MAX_TOKENS = 600
PROMPT_JSON_CHARS = 1800
PROMPT_HISTORY_LINES = 8
TASKS = ["easy", "medium", "hard", "cascade"]

TASK_MAX_STEPS = {
    "easy": 10,
    "medium": 15,
    "hard": 25,
    "cascade": 30,
}

TASK_FORCE_SUBMIT_AT = {
    "easy": 4,
    "medium": 5,
    "hard": 10,
    "cascade": 8,
}

TASK_MIN_EVIDENCE = {
    "easy": 3,
    "medium": 4,
    "hard": 4,
    "cascade": 6,
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

BLOCKED_ACTIONS = {"request_rollback"}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous MLOps incident response agent.

    Investigate production ML incidents step by step using only observations
    from the environment. Submit a diagnosis when evidence is strong OR
    when the step budget is nearly exhausted.

    CRITICAL RULES:
    - NEVER use request_rollback before submitting a diagnosis — it wastes steps.
    - NEVER repeat the same action on the same component.
    - When DIAGNOSIS READINESS says READY, call submit_diagnosis immediately.
    - When remaining_steps <= 4, you MUST call submit_diagnosis regardless.
    - For drift incidents: run check_feature_drift on feature_store + model_server
      + ab_testing_service, then check_metrics(business), then submit_diagnosis.
    - Do NOT run check_feature_drift on infrastructure components like api_gateway.

    Investigation strategy:
    - Start with components in critical, error, degraded, or warning states.
    - inspect: understand a component structure.
    - query_logs: concrete error evidence.
    - check_metrics: quantitative symptoms.
    - compare_configs: when config/deployment change suspected.
    - check_feature_drift: ONLY for drift/revenue/silent-degradation incidents,
      ONLY on ML components.
    - cascade: gather query_logs from EVERY degraded subsystem first.

    Diagnosis quality:
    - data_quality: name broken pipeline, schema/field issue, downstream impact.
    - latency: name bottleneck service, changed parameter, old/new values.
    - drift: name drifted feature, PSI evidence, model staleness, retrain plan.
    - cascade: name ALL root-cause services, shared deployment, coordinated rollback.

    Output exactly one JSON object — no prose, no markdown:
    {
      "action_type": "inspect|query_logs|check_metrics|compare_configs|check_feature_drift|submit_diagnosis",
      "target": "exact_component_name_from_COMPONENT_STATUS",
      "parameters": {}
    }

    For submit_diagnosis:
    {
      "action_type": "submit_diagnosis",
      "target": "root_cause_component",
      "parameters": {
        "root_cause": "specific evidence-based diagnosis with component names and values",
        "fix": "clear remediation plan"
      }
    }
    """
).strip()


@dataclass
class Obs:
    goal: str = ""
    alert_summary: str = ""
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


def _to_obs(d: Dict[str, Any]) -> Obs:
    return Obs(
        goal=d.get("goal", ""),
        alert_summary=d.get("alert_summary", ""),
        component_status=d.get("component_status", {}) or {},
        recent_logs=d.get("recent_logs", []) or [],
        metrics_snapshot=d.get("metrics_snapshot", {}) or {},
        action_feedback=d.get("action_feedback", ""),
        step_count=int(d.get("step_count", 0)),
        sla_minutes_remaining=d.get("sla_minutes_remaining"),
        reward=float(d.get("reward", 0.0)),
        cumulative_reward=float(d.get("cumulative_reward", 0.0)),
        done=bool(d.get("done", False)),
        final_score=d.get("final_score"),
        score_breakdown=d.get("score_breakdown"),
    )


class DirectEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy") -> Obs:
        r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return _to_obs(r.json())

    def step(self, action_type: str, target: str, parameters: Dict[str, Any]) -> Obs:
        r = requests.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "target": target, "parameters": parameters},
            timeout=30,
        )
        r.raise_for_status()
        return _to_obs(r.json())

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=10).status_code == 200
        except Exception:
            return False


def _norm(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def compact_text(value: str, limit: int) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[:limit] + " ...[truncated]"


def _json_block(value: Any) -> str:
    try:
        return compact_text(json.dumps(value, sort_keys=True), PROMPT_JSON_CHARS)
    except Exception:
        return compact_text(str(value), PROMPT_JSON_CHARS)


def normalize_action(action_type: str) -> str:
    compact = _norm(action_type).replace(" ", "")
    aliases = {
        "inspect": "inspect",
        "querylogs": "query_logs",
        "querylog": "query_logs",
        "query_logs": "query_logs",
        "checklogs": "query_logs",
        "checkmetrics": "check_metrics",
        "check_metrics": "check_metrics",
        "metrics": "check_metrics",
        "compareconfigs": "compare_configs",
        "compare_configs": "compare_configs",
        "compareconfig": "compare_configs",
        "configdiff": "compare_configs",
        "checkfeaturedrift": "check_feature_drift",
        "check_feature_drift": "check_feature_drift",
        "featuredrift": "check_feature_drift",
        "submitdiagnosis": "submit_diagnosis",
        "submit_diagnosis": "submit_diagnosis",
        "diagnose": "submit_diagnosis",
        "requestrollback": "request_rollback",
        "request_rollback": "request_rollback",
        "rollback": "request_rollback",
    }
    return aliases.get(compact, compact)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    loaded = json.loads(text[start : i + 1])
                    return loaded if isinstance(loaded, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _fuzzy_match(name: str, valid: Sequence[str]) -> str:
    if not valid:
        return name
    if name in valid:
        return name

    target_norm = _norm(name)
    target_compact = target_norm.replace(" ", "")
    if not target_compact:
        return valid[0]

    for c in valid:
        if _norm(c) == target_norm:
            return c

    for c in valid:
        cc = _norm(c).replace(" ", "")
        if target_compact in cc or cc in target_compact:
            return c

    tokens = [t for t in target_norm.split() if len(t) > 2]
    best, best_score = valid[0], -1.0
    for c in valid:
        cn = _norm(c)
        score = sum(1 for t in tokens if t in set(cn.split())) / max(len(tokens), 1)
        if any(t in cn for t in tokens):
            score += 0.25
        if score > best_score:
            best_score, best = score, c
    return best


def parse_action(
    text: str, valid_components: Sequence[str]
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    obj = _extract_json_object(text)
    if not obj:
        return None

    action_type = normalize_action(str(obj.get("action_type", "")))
    if action_type not in VALID_ACTIONS:
        return None

    target = str(obj.get("target", "") or "")
    parameters = obj.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    if action_type == "submit_diagnosis":
        if not parameters.get("root_cause"):
            return None
        if not parameters.get("fix"):
            parameters["fix"] = "Mitigate the identified root cause and monitor recovery."
        if valid_components:
            target = _fuzzy_match(target, valid_components) if target else valid_components[0]
        return action_type, target, parameters

    if action_type == "check_metrics" and _norm(target) in {"global", "business", "revenue"}:
        target = _norm(target)
    elif action_type == "check_feature_drift":
        target = target or "all"
    elif valid_components:
        target = _fuzzy_match(target, valid_components)

    return action_type, target, parameters


def incident_kind(obs: Obs) -> str:
    text = _norm(f"{obs.goal} {obs.alert_summary}")
    if any(t in text for t in ["cascade", "three ml", "multi system", "all root causes"]):
        return "cascade"
    if any(t in text for t in ["revenue", "silent", "drift", "distribution", "no error alerts"]):
        return "drift"
    if any(t in text for t in ["latency", "timeout", "p99", "bottleneck", "config change"]):
        return "latency"
    if any(t in text for t in ["accuracy", "data quality", "schema", "null", "pipeline"]):
        return "data_quality"
    return "general"


def evidence_text(obs: Obs, history: List[str]) -> str:
    return _norm(
        f"{obs.goal} {obs.alert_summary} "
        + " ".join(history[-12:])
        + " "
        + (obs.action_feedback or "")
    )


ML_DRIFT_COMPONENTS = {
    "feature_store",
    "model_server",
    "model_serving",
    "ab_testing_service",
    "training_pipeline",
}


def is_ml_drift_component(name: str) -> bool:
    n = _norm(name).replace(" ", "_")
    return any(m in n for m in ML_DRIFT_COMPONENTS)


def diagnosis_readiness(
    obs: Obs,
    history: List[str],
    seen: set,
    remaining_steps: int,
    task_id: str = "easy",
) -> str:
    force_at = TASK_FORCE_SUBMIT_AT.get(task_id, 4)
    if remaining_steps <= force_at:
        return "READY: step budget critical — submit now."

    kind = incident_kind(obs)
    evidence = evidence_text(obs, history)
    min_ev = TASK_MIN_EVIDENCE.get(task_id, 3)
    evidence_actions = [k for k in seen if not k.startswith("submit_diagnosis:")]

    if kind == "data_quality":
        has_pipeline_logs = any(k.startswith("query_logs:") and "pipeline" in k for k in seen)
        has_schema = any(t in evidence for t in ["schema", "validation", "null", "field", "migration"])
        if has_pipeline_logs and has_schema:
            return "READY: pipeline logs + schema evidence observed."
        return "NOT READY: need query_logs on pipeline + schema/null evidence."

    if kind == "latency":
        has_config = any(k.startswith("compare_configs:") for k in seen)
        has_latency = any(t in evidence for t in ["latency", "timeout", "deployed", "queue", "memory", "cpu"])
        if has_config and has_latency:
            return "READY: compare_configs + latency evidence observed."
        return "NOT READY: need compare_configs + latency/timeout evidence."

    if kind == "drift":
        drift_done = [k for k in seen if k.startswith("check_feature_drift:")]
        has_psi = any(t in evidence for t in ["psi", "critical drift", "critical_drift", "drift"])
        has_business = any(k.startswith("check_metrics:") and "business" in k for k in seen)
        if len(drift_done) >= 3 and has_business:
            return "READY: 3 drift checks + business metrics observed."
        if len(drift_done) >= 2 and has_psi and has_business:
            return "READY: drift PSI evidence + business metrics observed."
        return f"NOT READY: need check_feature_drift on 3 ML components + check_metrics(business). Done: {len(drift_done)}/3."

    if kind == "cascade":
        degraded = {
            c for c, s in obs.component_status.items()
            if any(t in _norm(s) for t in ["critical", "error", "degraded", "failed"])
        }
        log_covered = {k.split(':', 1)[1] for k in seen if k.startswith("query_logs:")} & degraded
        if len(log_covered) >= min(3, len(degraded)):
            return "READY: query_logs from all primary degraded subsystems."
        missing = sorted(degraded - log_covered)
        return f"NOT READY: need query_logs from: {', '.join(missing[:3]) or 'unknown'}."

    if len(evidence_actions) >= min_ev:
        return "READY: sufficient evidence gathered."
    return f"NOT READY: gather {min_ev} evidence actions ({len(evidence_actions)} done)."


def submit_ready(
    obs: Obs, history: List[str], seen: set, remaining_steps: int, task_id: str = "easy"
) -> bool:
    force_at = TASK_FORCE_SUBMIT_AT.get(task_id, 4)
    if remaining_steps <= force_at:
        return True
    return diagnosis_readiness(obs, history, seen, remaining_steps, task_id).startswith("READY")


def diagnosis_plausible(
    action_type: str, target: str, parameters: Dict[str, Any], obs: Obs, history: List[str]
) -> bool:
    if action_type != "submit_diagnosis":
        return True
    text = _norm(f"{target} {parameters}")
    kind = incident_kind(obs)

    if kind == "data_quality":
        return "pipeline" in text and any(
            t in text for t in ["schema", "null", "field", "migration", "validation"]
        )
    if kind == "latency":
        return any(
            t in text for t in ["batch", "worker", "concurrent", "latency", "timeout", "memory", "cpu", "rollback"]
        )
    if kind == "drift":
        return "drift" in text or any(
            t in text for t in ["psi", "retrain", "stale", "model", "feature"]
        )
    if kind == "cascade":
        degraded = [
            c for c, s in obs.component_status.items()
            if any(t in _norm(s) for t in ["critical", "error", "degraded"])
        ]
        mentioned = sum(1 for c in degraded if _norm(c) in text)
        return mentioned >= min(2, len(degraded))
    return len(text) > 30


def prioritized_components(obs: Obs, valid_components: Sequence[str]) -> List[str]:
    kind = incident_kind(obs)

    def rank(c: str) -> Tuple[int, int, int, str]:
        state = _norm(obs.component_status.get(c, ""))
        if any(t in state for t in ["critical", "error", "failed"]):
            s_rank = 0
        elif any(t in state for t in ["degraded", "warning", "warn"]):
            s_rank = 1
        elif "healthy" in state or "ok" in state:
            s_rank = 3
        else:
            s_rank = 2

        name = _norm(c)
        r_rank = 5
        if kind == "data_quality" and "pipeline" in name:
            r_rank = 0
        elif kind == "data_quality" and "feature store" in name:
            r_rank = 1
        elif kind == "latency" and any(t in name for t in ["preprocessor", "model serving", "load balancer"]):
            r_rank = 0
        elif kind == "drift" and any(t in name for t in ["feature store", "model server", "ab testing", "training", "monitoring"]):
            r_rank = 0
        elif kind == "cascade":
            r_rank = 0

        mentioned = name in _norm(f"{obs.action_feedback} {obs.alert_summary}")
        return (s_rank, r_rank, 0 if mentioned else 1, c)

    return sorted(valid_components, key=rank)


def fallback_candidates(obs: Obs, valid_components: Sequence[str]) -> List[Tuple[str, str]]:
    components = prioritized_components(obs, valid_components)
    kind = incident_kind(obs)
    candidates: List[Tuple[str, str]] = []

    if kind == "data_quality":
        for c in components:
            if "pipeline" in _norm(c):
                candidates += [("inspect", c), ("query_logs", c), ("check_metrics", c)]
        for c in components:
            if "feature" in _norm(c):
                candidates += [("inspect", c), ("query_logs", c), ("check_metrics", c)]

    elif kind == "latency":
        for c in components:
            candidates += [("inspect", c), ("query_logs", c), ("compare_configs", c), ("check_metrics", c)]

    elif kind == "drift":
        ml_comps = [c for c in components if is_ml_drift_component(c)]
        for c in ml_comps:
            candidates.append(("check_feature_drift", c))
        candidates.append(("check_metrics", "business"))
        for c in ml_comps:
            candidates += [("inspect", c), ("query_logs", c), ("check_metrics", c)]

    elif kind == "cascade":
        degraded = [
            c for c in components
            if any(t in _norm(obs.component_status.get(c, "")) for t in ["critical", "error", "degraded", "failed"])
        ]
        for c in degraded:
            candidates += [("query_logs", c), ("check_metrics", c), ("inspect", c), ("compare_configs", c)]

    for c in components:
        candidates += [("inspect", c), ("query_logs", c), ("check_metrics", c), ("compare_configs", c)]

    return candidates


def evidence_summary(obs: Obs, history: List[str]) -> str:
    snippets = (
        [f"Incident goal: {obs.goal}", f"Alert: {obs.alert_summary}"]
        + history[-6:]
        + [f"Latest feedback: {obs.action_feedback}"]
    )
    return "Evidence-based diagnosis: " + " | ".join(s[:700] for s in snippets if s)


def best_evidence_target(obs: Obs, valid_components: Sequence[str], history: List[str]) -> str:
    if not valid_components:
        return "unknown"
    ev = evidence_text(obs, history)
    ranked = prioritized_components(obs, valid_components)
    for c in ranked:
        if _norm(c) in ev:
            return c
    return ranked[0]


def fallback_diagnosis(
    obs: Obs, valid_components: Sequence[str], history: List[str]
) -> Tuple[str, str, Dict[str, Any]]:
    kind = incident_kind(obs)
    target = best_evidence_target(obs, valid_components, history)
    root_cause = evidence_summary(obs, history)
    fixes = {
        "data_quality": "Rollback the bad schema migration, repair and backfill downstream feature_store data, and monitor model accuracy recovery.",
        "latency": "Rollback the risky config parameter to the last stable value, restart the affected service, and monitor latency and error rates.",
        "drift": "Retrain the model on recent post-change data, validate PSI metrics, and enable continuous drift monitoring.",
        "cascade": "Perform a coordinated rollback of all affected services to pre-deployment state, verify each subsystem, and confirm model quality.",
    }
    fix = fixes.get(kind, "Apply safest remediation, rollback recent risky changes if present, and monitor recovery.")
    return "submit_diagnosis", target, {"root_cause": root_cause, "fix": fix}


def fallback_action(
    obs: Obs,
    valid_components: Sequence[str],
    seen: set,
    history: List[str],
    remaining_steps: int,
    task_id: str = "easy",
) -> Tuple[str, str, Dict[str, Any]]:
    force_at = TASK_FORCE_SUBMIT_AT.get(task_id, 4)
    if remaining_steps <= force_at:
        return fallback_diagnosis(obs, valid_components, history)
    for action_type, target in fallback_candidates(obs, valid_components):
        if f"{action_type}:{target}" not in seen:
            return action_type, target, {}
    return fallback_diagnosis(obs, valid_components, history)


def build_user_prompt(
    obs: Obs, history: List[str], seen: set, remaining_steps: int, task_id: str = "easy"
) -> str:
    status = "\n".join(f"- {n}: {s}" for n, s in obs.component_status.items())
    logs = _json_block(obs.recent_logs) if obs.recent_logs else "None returned."
    metrics = _json_block(obs.metrics_snapshot) if obs.metrics_snapshot else "None returned."
    history_text = "\n".join(history[-PROMPT_HISTORY_LINES:]) if history else "No actions yet."
    seen_text = ", ".join(sorted(seen)) if seen else "None"
    readiness = diagnosis_readiness(obs, history, seen, remaining_steps, task_id)
    suggested_action, suggested_target, _ = fallback_action(
        obs, list(obs.component_status.keys()), seen, history, remaining_steps, task_id
    )

    force_at = TASK_FORCE_SUBMIT_AT.get(task_id, 4)
    urgency = ""
    if remaining_steps <= force_at:
        urgency = f"\n⚠ CRITICAL: only {remaining_steps} step(s) left. You MUST submit_diagnosis RIGHT NOW."
    elif remaining_steps <= force_at + 3:
        urgency = f"\nWARNING: only {remaining_steps} step(s) left — submit_diagnosis soon."

    return textwrap.dedent(
        f"""
        INCIDENT GOAL: {obs.goal}

        ALERT: {obs.alert_summary}

        STEP: current={obs.step_count}  remaining={remaining_steps}
        sla_remaining={obs.sla_minutes_remaining}  reward={obs.reward}  cumulative={obs.cumulative_reward}
        {urgency}

        COMPONENT STATUS:
        {status}

        LATEST ACTION FEEDBACK:
        {compact_text(obs.action_feedback or "None", 1400)}

        RECENT LOGS: {logs}

        METRICS: {metrics}

        INVESTIGATION HISTORY:
        {history_text}

        USED PAIRS (DO NOT REPEAT): {seen_text}

        DIAGNOSIS READINESS: {readiness}

        SUGGESTED NEXT: {suggested_action} on {suggested_target}

        VALID ACTIONS: inspect, query_logs, check_metrics, compare_configs, check_feature_drift, submit_diagnosis
        (Never use request_rollback. Never use check_feature_drift on non-ML components like api_gateway.)

        Return exactly one JSON object. Target must be exact name from COMPONENT STATUS.
        """
    ).strip()


def chat_message(role: str, content: str) -> ChatCompletionMessageParam:
    return cast(ChatCompletionMessageParam, {"role": role, "content": content})


def messages_for_model(messages: List[ChatCompletionMessageParam]) -> List[ChatCompletionMessageParam]:
    if len(messages) <= 4:
        return messages
    return [messages[0]] + messages[-3:]


def run_task(llm: OpenAI, env: DirectEnv, task_id: str) -> Dict[str, Any]:
    rewards: List[float] = []
    final_score = 0.0
    success = False
    history: List[str] = []
    seen: set = set()
    obs = Obs()
    messages: List[ChatCompletionMessageParam] = [chat_message("system", SYSTEM_PROMPT)]
    max_steps = TASK_MAX_STEPS.get(task_id, 15)

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_id=task_id)
        valid_components = list(obs.component_status.keys())
        print(f"[RESET] components={valid_components}\n        goal={obs.goal[:100]}", flush=True)

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            remaining_steps = max_steps - step + 1
            user_prompt = build_user_prompt(obs, history, seen, remaining_steps, task_id)
            messages.append(chat_message("user", user_prompt))

            last_error = "null"
            source = "llm"
            parsed: Optional[Tuple[str, str, Dict[str, Any]]] = None

            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages_for_model(messages),
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw = completion.choices[0].message.content or ""
                messages.append(chat_message("assistant", raw))
                parsed = parse_action(raw, valid_components)
            except Exception as e:
                last_error = str(e).replace("\n", " ")[:200]

            if not parsed:
                action_type, target, parameters = fallback_action(
                    obs, valid_components, seen, history, remaining_steps, task_id
                )
                source = "fallback"
            else:
                action_type, target, parameters = parsed

            if action_type in BLOCKED_ACTIONS:
                action_type, target, parameters = fallback_action(
                    obs, valid_components, seen, history, remaining_steps, task_id
                )
                source = "fallback_blocked_action_guard"

            if action_type == "check_feature_drift":
                kind = incident_kind(obs)
                if kind not in {"drift", "cascade"}:
                    action_type, target, parameters = fallback_action(
                        obs, valid_components, seen, history, remaining_steps, task_id
                    )
                    source = "fallback_drift_guard"
                elif not is_ml_drift_component(target):
                    action_type, target, parameters = fallback_action(
                        obs, valid_components, seen, history, remaining_steps, task_id
                    )
                    source = "fallback_drift_infra_guard"

            force_at = TASK_FORCE_SUBMIT_AT.get(task_id, 4)
            if remaining_steps <= force_at and action_type != "submit_diagnosis":
                action_type, target, parameters = fallback_diagnosis(obs, valid_components, history)
                source = "fallback_force_submit_guard"

            if action_type != "submit_diagnosis" and f"{action_type}:{target}" in seen:
                action_type, target, parameters = fallback_action(
                    obs, valid_components, seen, history, remaining_steps, task_id
                )
                source = "fallback_loop_guard"

            if action_type == "submit_diagnosis" and not submit_ready(
                obs, history, seen, remaining_steps, task_id
            ):
                action_type, target, parameters = fallback_action(
                    obs, valid_components, seen, history, remaining_steps, task_id
                )
                source = "fallback_submit_guard"

            if action_type == "submit_diagnosis" and not diagnosis_plausible(
                action_type, target, parameters, obs, history
            ):
                action_type, target, parameters = fallback_diagnosis(obs, valid_components, history)
                source = "fallback_plausibility_guard"

            if action_type != "submit_diagnosis" and valid_components and target not in valid_components:
                if action_type not in {"check_metrics"}:
                    target = _fuzzy_match(target, valid_components)

            seen.add(f"{action_type}:{target}")

            try:
                obs = env.step(action_type, target, parameters)
            except Exception as e:
                last_error = str(e).replace("\n", " ")[:200]
                print(
                    f"[STEP] step={step} action={action_type}({target}) src={source} reward=0.00 done=false err={last_error}",
                    flush=True,
                )
                rewards.append(0.0)
                history.append(f"Step {step}: {action_type}({target}) ERROR {last_error}")
                break

            action_str = f"{action_type}({target})"
            feedback = (obs.action_feedback or "").replace("\n", " ")[:900]
            history.append(f"Step {step}: {action_str} src={source} r={obs.reward:+.2f} fb={feedback}")
            rewards.append(obs.reward)

            print(
                f"[STEP] step={step} action={action_str} src={source} reward={obs.reward:.2f} done={'true' if obs.done else 'false'} err={last_error}",
                flush=True,
            )

            final_score = obs.final_score if obs.final_score is not None else obs.cumulative_reward
            if obs.done:
                break

        success = final_score > 0.0

    except Exception as e:
        print(f"# run_task error: {e}", flush=True)
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        final_score = round(min(max(float(final_score), 0.0001), 0.9999), 4)
        print(
            f"[END] task={task_id} success={'true' if success else 'false'} steps={len(rewards)} score={final_score:.4f} rewards=[{rewards_str}]",
            flush=True,
        )

    return {"task_id": task_id, "final_score": final_score, "score_breakdown": obs.score_breakdown}


def main() -> None:
    print("# inference.py v4 starting...", flush=True)
    if not API_KEY:
        print("# WARNING: No API key found", flush=True)

    env = DirectEnv(base_url=HF_SPACE_URL)
    print(f"# Server:      {HF_SPACE_URL}", flush=True)
    print(f"# Model:       {MODEL_NAME}", flush=True)
    print(f"# Temperature: {TEMPERATURE}", flush=True)
    print("# Server: OK" if env.health() else f"# WARNING: Server not responding at {HF_SPACE_URL}", flush=True)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy-key", timeout=30)
    results: List[Dict[str, Any]] = []

    for task_id in TASKS:
        results.append(run_task(llm, env, task_id))
        time.sleep(0.3)

    print(f"\n# {'=' * 58}", flush=True)
    print("# LLM AGENT RESULTS", flush=True)
    print(f"# {'=' * 58}", flush=True)
    total = 0.0
    for r in results:
        print(f"# {r['task_id']:<10} score: {r['final_score']:.4f}", flush=True)
        total += r["final_score"]
    print(f"# {'AVERAGE':<10} score: {total / len(results):.4f}", flush=True)


if __name__ == "__main__":
    main()
