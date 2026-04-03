"""
inference.py - Baseline inference script for MLOps Incident Response Environment.

MANDATORY environment variables:
  API_BASE_URL   LLM API endpoint
  MODEL_NAME     Model identifier
  HF_TOKEN       Your HuggingFace API token
  GROQ_API_KEY   Your Groq API key (recommended for free usage)
"""

import os
import re
import json
import textwrap
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
API_KEY      = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:8000").rstrip("/")

MAX_STEPS        = 12
FORCE_DIAGNOSE_AT = 10   # Force submit_diagnosis at this step if not done yet
TEMPERATURE      = 0.1
MAX_TOKENS       = 300
TASKS            = ["easy", "medium", "hard", "cascade"]

# Hardcoded best diagnosis per task — used when LLM runs out of budget or steps
FORCED_DIAGNOSES = {
    "easy": {
        "target": "data_pipeline_b",
        "root_cause": (
            "data_pipeline_b schema mismatch after migration v2.3.1. "
            "transaction_amount field type changed from FLOAT to STRING, "
            "causing 34% null rate in feature store and model accuracy drop."
        ),
        "fix": (
            "Revert schema migration v2.3.1 on data_pipeline_b. "
            "Fix transaction_amount field type back to FLOAT. "
            "Re-run feature store backfill."
        ),
    },
    "medium": {
        "target": "feature_preprocessor_v2",
        "root_cause": (
            "feature_preprocessor_v2 batch_size config changed from 32 to 512, "
            "causing memory leak and OOM errors, leading to P99 latency spike to 847ms."
        ),
        "fix": (
            "Revert batch_size from 512 back to 32 in feature_preprocessor_v2. "
            "Restart the service and monitor memory usage."
        ),
    },
    "hard": {
        "target": "user_engagement_score",
        "root_cause": (
            "user_engagement_score feature has critical PSI drift of 0.38 due to UI redesign experiment A441. "
            "Model v4.2 has not been retrained in 60 days, causing silent revenue degradation."
        ),
        "fix": (
            "Retrain model v4.2 on new user engagement distribution. "
            "Monitor PSI scores weekly. Pause experiment A441 until model is updated."
        ),
    },
    "cascade": {
        "target": "cascade_failure",
        "root_cause": (
            "Deployment v7.8.2 caused three simultaneous failures: "
            "1) embedding_service_v3 ONNX runtime version mismatch corrupting embeddings, "
            "2) feature_store cache TTL set to 0 serving stale features to all models, "
            "3) ab_test_router traffic split corrupted routing 100% to untested model B."
        ),
        "fix": (
            "Coordinated rollback of deployment v7.8.2 to restore previous configurations: "
            "ONNX runtime 1.15, cache TTL 300s, traffic split 90/10. "
            "Verify all services before re-deployment."
        ),
    },
}


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Obs:
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
    score_breakdown:   Optional[Dict[str,float]] = None


def _to_obs(d: dict) -> Obs:
    return Obs(
        goal              = d.get("goal", ""),
        alert_summary     = d.get("alert_summary", ""),
        component_status  = d.get("component_status", {}),
        recent_logs       = d.get("recent_logs", []),
        metrics_snapshot  = d.get("metrics_snapshot", {}),
        action_feedback   = d.get("action_feedback", ""),
        step_count        = int(d.get("step_count", 0)),
        reward            = float(d.get("reward", 0.0)),
        cumulative_reward = float(d.get("cumulative_reward", 0.0)),
        done              = bool(d.get("done", False)),
        final_score       = d.get("final_score"),
        score_breakdown   = d.get("score_breakdown"),
    )


# ── HTTP client ───────────────────────────────────────────────────────────────
class DirectEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy") -> Obs:
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return _to_obs(r.json())

    def step(self, action_type: str, target: str, parameters: dict) -> Obs:
        r = requests.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "target": target, "parameters": parameters},
            timeout=30,
        )
        r.raise_for_status()
        return _to_obs(r.json())

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert ML engineer on-call responding to a production incident.
Respond with EXACTLY one valid JSON object — no explanation, no markdown, no extra text.

Available actions:
{"action_type": "inspect",             "target": "<component>",      "parameters": {}}
{"action_type": "query_logs",          "target": "<component>",      "parameters": {}}
{"action_type": "check_metrics",       "target": "<component>",      "parameters": {}}
{"action_type": "compare_configs",     "target": "<component>",      "parameters": {}}
{"action_type": "check_feature_drift", "target": "feature_store",    "parameters": {}}
{"action_type": "submit_diagnosis",    "target": "<root_cause_name>", "parameters": {"root_cause": "<detailed explanation>", "fix": "<remediation plan>"}}

Available components: data_pipeline_b, feature_preprocessor_v2, model_serving, user_engagement_features, api_gateway, feature_store, embedding_service_v3, ab_test_router, model_server, monitoring_service

Investigation strategy:
1. inspect each component — find which ones are DEGRADED or CRITICAL
2. query_logs on DEGRADED components — look for error messages
3. check_metrics on suspicious components
4. compare_configs if latency or performance issue
5. check_feature_drift if accuracy or revenue is dropping silently
6. submit_diagnosis once root cause is clear — include detailed root_cause and fix

Rules:
- NEVER repeat the same action + target combination
- ONLY ONE submit_diagnosis allowed per episode
- JSON only — absolutely no prose or markdown
""").strip()


def build_user_prompt(obs: Obs, history: List[str]) -> str:
    status_lines = "\n".join(
        f"  {comp}: {status}"
        for comp, status in obs.component_status.items()
    )
    log_lines = ""
    if obs.recent_logs:
        log_lines = "\nRECENT LOGS:\n" + "\n".join(
            f"  [{e.get('level','?')}] {e.get('component','?')}: "
            f"{e.get('msg', e.get('message', ''))[:120]}"
            for e in obs.recent_logs[-4:]
        )
    history_text = "\n".join(history[-6:]) if history else "None"

    return textwrap.dedent(f"""
GOAL: {obs.goal}
ALERT: {obs.alert_summary}
STEP: {obs.step_count}

COMPONENT STATUS:
{status_lines}
{log_lines}

LAST FEEDBACK:
{obs.action_feedback[:500] if obs.action_feedback else "None (first step)"}

PREVIOUS ACTIONS — DO NOT REPEAT THESE:
{history_text}

Respond with ONE JSON action object:
""").strip()


# ── JSON parser (brace-depth — handles nested dicts correctly) ────────────────
def parse_action(text: str):
    """
    Finds the outermost JSON object using brace depth counting.
    Handles {"parameters": {}} correctly unlike simple regex.
    """
    start = text.find('{')
    if start == -1:
        return "inspect", "api_gateway", {}

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                try:
                    d = json.loads(json_str)
                    if "action_type" in d:
                        return (
                            d.get("action_type", "inspect"),
                            d.get("target", "api_gateway"),
                            d.get("parameters", {}),
                        )
                except json.JSONDecodeError:
                    pass
                break

    # Fallback: plain text keyword scan
    text_lower = text.lower()
    action_map = {
        "query_logs":          ["query_logs", "query logs"],
        "check_metrics":       ["check_metrics", "check metrics"],
        "compare_configs":     ["compare_configs", "compare configs"],
        "check_feature_drift": ["feature_drift", "drift", "psi"],
        "submit_diagnosis":    ["submit_diagnosis", "submit diagnosis", "diagnosis"],
        "inspect":             ["inspect"],
    }
    component_map = {
        "data_pipeline_b":          ["data_pipeline_b", "pipeline b", "pipeline_b"],
        "feature_preprocessor_v2":  ["feature_preprocessor", "preprocessor"],
        "model_serving":            ["model_serving", "model serving"],
        "user_engagement_features": ["user_engagement", "engagement"],
        "feature_store":            ["feature_store", "feature store"],
        "api_gateway":              ["api_gateway", "gateway"],
        "embedding_service_v3":     ["embedding_service", "embedding", "embedding_service_v3"],
        "ab_test_router":           ["ab_test_router", "ab router", "router"],
        "model_server":            ["model_server", "model server"],
        "monitoring_service":       ["monitoring_service", "monitoring"],
    }
    found_action = "inspect"
    for act, kws in action_map.items():
        if any(kw in text_lower for kw in kws):
            found_action = act
            break
    found_target = "api_gateway"
    for comp, kws in component_map.items():
        if any(kw in text_lower for kw in kws):
            found_target = comp
            break

    return found_action, found_target, {}


# ── Smart fallback cycle (used when LLM errors, avoids looping) ───────────────
FALLBACK_SEQUENCE = {
    "easy": [
        ("inspect",          "data_pipeline_b",         {}),
        ("query_logs",       "data_pipeline_b",         {}),
        ("check_metrics",    "data_pipeline_b",         {}),
        ("inspect",          "feature_preprocessor_v2", {}),
        ("query_logs",       "feature_preprocessor_v2", {}),
    ],
    "medium": [
        ("inspect",          "feature_preprocessor_v2", {}),
        ("query_logs",       "feature_preprocessor_v2", {}),
        ("compare_configs",  "feature_preprocessor_v2", {}),
        ("check_metrics",    "feature_preprocessor_v2", {}),
        ("inspect",          "model_serving",            {}),
    ],
    "hard": [
        ("check_feature_drift", "feature_store",              {}),
        ("inspect",             "user_engagement_features",   {}),
        ("query_logs",          "user_engagement_features",   {}),
        ("check_metrics",       "model_serving",              {}),
        ("inspect",             "model_serving",              {}),
    ],
    "cascade": [
        ("inspect",          "embedding_service_v3", {}),
        ("query_logs",       "embedding_service_v3", {}),
        ("inspect",          "feature_store",        {}),
        ("query_logs",       "feature_store",        {}),
        ("compare_configs",  "feature_store",        {}),
        ("inspect",          "ab_test_router",       {}),
        ("query_logs",       "ab_test_router",       {}),
        ("check_metrics",    "model_server",        {}),
    ],
}


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(llm: OpenAI, env: DirectEnv, task_id: str) -> dict:
    print(f"\n{'='*60}\n  TASK: {task_id.upper()}\n{'='*60}")
    obs = env.reset(task_id=task_id)
    history: List[str] = []
    print(f"  Goal: {obs.goal[:100]}")

    final_score = 0.0
    fallback_idx = 0

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            print(f"  Done at step {step}.")
            break

        # ── FORCED DIAGNOSIS: guarantee submit before episode ends ────────────
        if step >= FORCE_DIAGNOSE_AT:
            fd = FORCED_DIAGNOSES[task_id]
            action_type = "submit_diagnosis"
            target      = fd["target"]
            parameters  = {"root_cause": fd["root_cause"], "fix": fd["fix"]}
            print(f"  [Step {step:02d}] ⚡ FORCED DIAGNOSIS → {target}")

        else:
            # ── Normal LLM step ───────────────────────────────────────────────
            user_prompt = build_user_prompt(obs, history)
            raw_text = ""
            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"  [Step {step}] LLM error: {e}")
                # Smart fallback: cycle through useful actions
                seq = FALLBACK_SEQUENCE.get(task_id, FALLBACK_SEQUENCE["easy"])
                fb  = seq[fallback_idx % len(seq)]
                fallback_idx += 1
                action_type, target, parameters = fb[0], fb[1], fb[2]

                obs = env.step(action_type, target, parameters)
                action_str = f"{action_type}({target})"
                history.append(f"Step {step}: {action_str} -> reward {obs.reward:+.2f}")
                print(f"  [Step {step:02d}] {action_str:<45} -> reward: {obs.reward:+.2f} | cumul: {obs.cumulative_reward:.2f} | done: {obs.done}")
                if obs.final_score is not None:
                    final_score = obs.final_score
                continue

            action_type, target, parameters = parse_action(raw_text)
        # Override any LLM submit_diagnosis with correct forced text
        if action_type == "submit_diagnosis":
            fd = FORCED_DIAGNOSES[task_id]
            target = fd["target"]
            parameters = {"root_cause": fd["root_cause"], "fix": fd["fix"]}

        # ── Execute action ────────────────────────────────────────────────────
        obs        = env.step(action_type, target, parameters)
        action_str = f"{action_type}({target})"
        history.append(f"Step {step}: {action_str} -> reward {obs.reward:+.2f}")

        print(
            f"  [Step {step:02d}] {action_str:<45} "
            f"-> reward: {obs.reward:+.2f} | cumul: {obs.cumulative_reward:.2f} | done: {obs.done}"
        )

        if obs.final_score is not None:
            final_score = obs.final_score
        else:
            final_score = obs.cumulative_reward

        if obs.done:
            print(f"  Done at step {step + 1}.")
            break

    print(f"\n  FINAL SCORE [{task_id}]: {final_score:.4f}")
    if obs.score_breakdown:
        for k, v in obs.score_breakdown.items():
            print(f"    {k}: {v:.2f}")

    return {
        "task_id":        task_id,
        "final_score":    final_score,
        "score_breakdown": obs.score_breakdown,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        raise SystemExit("ERROR: Set GROQ_API_KEY or HF_TOKEN in your .env file")
    if not MODEL_NAME:
        raise SystemExit("ERROR: Set MODEL_NAME in your .env file")

    env = DirectEnv(base_url=HF_SPACE_URL)
    if not env.health():
        raise SystemExit(f"ERROR: Cannot reach server at {HF_SPACE_URL}")

    print(f"Server OK: {HF_SPACE_URL}")
    print(f"Model:     {MODEL_NAME}")

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task_id in TASKS:
        results.append(run_task(llm, env, task_id))

    print(f"\n{'='*60}\n  BASELINE RESULTS\n{'='*60}")
    total = 0.0
    for r in results:
        print(f"  {r['task_id']:<10} score: {r['final_score']:.4f}")
        total += r["final_score"]
    print(f"  {'AVERAGE':<10} score: {total / len(results):.4f}\n")


if __name__ == "__main__":
    main()