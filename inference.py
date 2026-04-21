"""
inference.py - Baseline inference script for MLOps Incident Response Environment.
Emits: [START] [STEP] [END] structured stdout logs per OpenEnv spec.
"""

import os
import json
import textwrap
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
API_KEY      = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME")   or "llama-3.1-8b-instant"
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:8000").rstrip("/")

ENV_NAME          = "mlops-incident-env"
MAX_STEPS         = 12
FORCE_DIAGNOSE_AT = 10
TEMPERATURE       = 0.1
MAX_TOKENS        = 400
TASKS             = ["easy", "medium", "hard", "cascade"]


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Obs:
    goal:              str                        = ""
    alert_summary:     str                        = ""
    component_status:  Dict[str, str]             = field(default_factory=dict)
    recent_logs:       List[Dict[str, Any]]       = field(default_factory=list)
    metrics_snapshot:  Dict[str, Any]             = field(default_factory=dict)
    action_feedback:   str                        = ""
    step_count:        int                        = 0
    reward:            float                      = 0.0
    cumulative_reward: float                      = 0.0
    done:              bool                       = False
    final_score:       Optional[float]            = None
    score_breakdown:   Optional[Dict[str, float]] = None


def _to_obs(d: dict) -> Obs:
    return Obs(
        goal             = d.get("goal", ""),
        alert_summary    = d.get("alert_summary", ""),
        component_status = d.get("component_status", {}),
        recent_logs      = d.get("recent_logs", []),
        metrics_snapshot = d.get("metrics_snapshot", {}),
        action_feedback  = d.get("action_feedback", ""),
        step_count       = int(d.get("step_count", 0)),
        reward           = float(d.get("reward", 0.0)),
        cumulative_reward= float(d.get("cumulative_reward", 0.0)),
        done             = bool(d.get("done", False)),
        final_score      = d.get("final_score"),
        score_breakdown  = d.get("score_breakdown"),
    )


# ── HTTP client ───────────────────────────────────────────────────────────────
class DirectEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy") -> Obs:
        r = requests.post(
            f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30
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
            return requests.get(f"{self.base_url}/health", timeout=10).status_code == 200
        except Exception:
            return False


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert ML engineer on-call responding to a production incident.
Respond with EXACTLY one valid JSON object — no explanation, no markdown, no extra text.

Available actions:
{"action_type": "inspect",             "target": "<component>", "parameters": {}}
{"action_type": "query_logs",          "target": "<component>", "parameters": {}}
{"action_type": "check_metrics",       "target": "<component>", "parameters": {}}
{"action_type": "compare_configs",     "target": "<component>", "parameters": {}}
{"action_type": "check_feature_drift", "target": "feature_store", "parameters": {}}
{"action_type": "submit_diagnosis",    "target": "<root_cause_component>", "parameters": {"root_cause": "<detailed explanation>", "fix": "<remediation plan>"}}

Investigation strategy:
1. FIRST: inspect components that show ERROR or CRITICAL status
2. query_logs on those ERROR/CRITICAL components — logs contain the exact root cause
3. check_metrics on suspicious components
4. compare_configs if latency spike incident
5. check_feature_drift if revenue/accuracy drop with no visible errors
6. submit_diagnosis once you have strong evidence — name the EXACT component and EXACT config/field/feature

Rules:
- NEVER repeat the same action+target combination — you will be penalised -0.10 each time
- If you already ran an action on a target, pick a DIFFERENT target or action
- ONE submit_diagnosis per episode
- JSON only, no prose, no markdown
""").strip()


# ── Prompt builders ───────────────────────────────────────────────────────────
def build_user_prompt(obs: Obs, history: List[str]) -> str:
    status_lines = "\n".join(f"  {c}: {s}" for c, s in obs.component_status.items())
    log_lines = ""
    if obs.recent_logs:
        log_lines = "\nRECENT LOGS:\n" + "\n".join(
            f"  [{e.get('level','?')}] {e.get('component','?')}: "
            f"{e.get('msg', e.get('message',''))[:120]}"
            for e in obs.recent_logs[-4:]
        )
    return textwrap.dedent(f"""
GOAL: {obs.goal}
ALERT: {obs.alert_summary}
STEP: {obs.step_count}

COMPONENT STATUS:
{status_lines}
{log_lines}

LAST FEEDBACK:
{obs.action_feedback[:600] if obs.action_feedback else "None (first step)"}

PREVIOUS ACTIONS — DO NOT REPEAT:
{chr(10).join(history[-6:]) if history else "None"}

Respond with ONE JSON action:
""").strip()


def build_force_diagnose_prompt(obs: Obs, history: List[str]) -> str:
    """Used at FORCE_DIAGNOSE_AT — extract all clues from feedback and force diagnosis."""
    degraded = [
        f"{comp} ({status})"
        for comp, status in obs.component_status.items()
        if status.lower() in ("error", "critical", "degraded")
    ]
    status_lines = "\n".join(f"  {c}: {s}" for c, s in obs.component_status.items())
    full_history = "\n".join(history) if history else "None"

    return textwrap.dedent(f"""
GOAL: {obs.goal}
ALERT: {obs.alert_summary}
STEP: {obs.step_count} — SUBMIT YOUR DIAGNOSIS NOW. This is your final action.

COMPONENT STATUS:
{status_lines}

DEGRADED / ERROR COMPONENTS: {', '.join(degraded) if degraded else 'check status above'}

ALL ACTIONS TAKEN SO FAR (your full investigation):
{full_history}

LAST FEEDBACK RECEIVED (contains key evidence — read carefully):
{obs.action_feedback[:800] if obs.action_feedback else "None"}

INSTRUCTIONS:
- Read the last feedback carefully — it contains the root cause clues
- If you saw CRITICAL_DRIFT in feature drift data, name that EXACT feature as the target
- If you saw a schema error or type mismatch, name that EXACT pipeline as the target
- If you saw a config change (batch_size, worker_threads, max_concurrent_requests etc), name that service
- If cascade: name ALL failed services in root_cause text
- Do NOT submit a generic component like "model_server" unless that is the exact root cause
- Be specific: include the exact field name, config param, or feature name in root_cause

Respond with ONE JSON — action_type MUST be "submit_diagnosis":
""").strip()


# ── Action parser ─────────────────────────────────────────────────────────────
def parse_action(text: str):
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
                try:
                    d = json.loads(text[start:i + 1])
                    if "action_type" in d:
                        return (
                            d.get("action_type", "inspect"),
                            d.get("target", "api_gateway"),
                            d.get("parameters", {}),
                        )
                except json.JSONDecodeError:
                    pass
                break
    tl = text.lower()
    action = next((a for a, kws in {
        "query_logs":          ["query_logs", "query logs"],
        "check_metrics":       ["check_metrics", "check metrics"],
        "compare_configs":     ["compare_configs", "compare configs"],
        "check_feature_drift": ["feature_drift", "drift", "psi"],
        "submit_diagnosis":    ["submit_diagnosis", "diagnosis"],
        "inspect":             ["inspect"],
    }.items() if any(kw in tl for kw in kws)), "inspect")
    target = next((c for c, kws in {
        "data_pipeline_b":          ["data_pipeline_b", "pipeline b", "pipeline_b"],
        "data_pipeline_a":          ["data_pipeline_a", "pipeline a", "pipeline_a"],
        "data_pipeline_c":          ["data_pipeline_c", "pipeline c", "pipeline_c"],
        "feature_preprocessor_v2":  ["feature_preprocessor", "preprocessor"],
        "model_serving":            ["model_serving", "model serving"],
        "model_server":             ["model_server", "model server"],
        "user_engagement_features": ["user_engagement", "engagement"],
        "feature_store":            ["feature_store", "feature store"],
        "embedding_service_v3":     ["embedding_service", "embedding"],
        "ab_test_router":           ["ab_test_router", "ab router", "router"],
        "ab_testing_service":       ["ab_testing_service", "ab testing"],
        "monitoring_service":       ["monitoring_service", "monitoring"],
        "load_balancer":            ["load_balancer", "load balancer"],
        "api_gateway":              ["api_gateway", "gateway"],
    }.items() if any(kw in tl for kw in kws)), "api_gateway")
    return action, target, {}


# ── Fallback sequences ────────────────────────────────────────────────────────
FALLBACK_SEQUENCE = {
    "easy": [
        ("inspect",       "data_pipeline_b",        {}),
        ("query_logs",    "data_pipeline_b",         {}),
        ("check_metrics", "data_pipeline_b",         {}),
        ("inspect",       "feature_store",           {}),
        ("query_logs",    "feature_store",           {}),
    ],
    "medium": [
        ("inspect",         "feature_preprocessor_v2", {}),
        ("query_logs",      "feature_preprocessor_v2", {}),
        ("compare_configs", "feature_preprocessor_v2", {}),
        ("check_metrics",   "feature_preprocessor_v2", {}),
        ("inspect",         "model_serving",            {}),
    ],
    "hard": [
        ("check_feature_drift", "feature_store",            {}),
        ("inspect",             "model_server",             {}),
        ("query_logs",          "model_server",             {}),
        ("inspect",             "ab_testing_service",       {}),
        ("query_logs",          "ab_testing_service",       {}),
    ],
    "cascade": [
        ("inspect",         "embedding_service_v3", {}),
        ("query_logs",      "embedding_service_v3", {}),
        ("inspect",         "feature_store",        {}),
        ("query_logs",      "feature_store",        {}),
        ("compare_configs", "feature_store",        {}),
        ("inspect",         "ab_test_router",       {}),
        ("query_logs",      "ab_test_router",       {}),
        ("check_metrics",   "model_server",         {}),
    ],
}


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(llm: OpenAI, env: DirectEnv, task_id: str) -> dict:
    rewards:      List[float] = []
    final_score:  float       = 0.0
    success:      bool        = False
    fallback_idx: int         = 0
    history:      List[str]   = []
    obs:          Obs         = Obs()

    # ══════════════════════════════════════════════════════════════════════════
    # [START] MUST be the very first print — before ANY network call
    # ══════════════════════════════════════════════════════════════════════════
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_type: str  = ""
            target:      str  = ""
            parameters:  dict = {}
            last_error:  str  = "null"

            # ── Choose prompt based on step ───────────────────────────────
            if step >= FORCE_DIAGNOSE_AT:
                user_content = build_force_diagnose_prompt(obs, history)
            else:
                user_content = build_user_prompt(obs, history)

            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw_text = completion.choices[0].message.content or ""
                action_type, target, parameters = parse_action(raw_text)

            except Exception as e:
                last_error   = str(e).replace("\n", " ")[:120]
                seq          = FALLBACK_SEQUENCE.get(task_id, FALLBACK_SEQUENCE["easy"])
                fb           = seq[fallback_idx % len(seq)]
                fallback_idx += 1
                action_type, target, parameters = fb

            try:
                obs = env.step(action_type, target, parameters)
            except Exception as e:
                last_error = str(e).replace("\n", " ")[:120]
                print(
                    f"[STEP] step={step} action={action_type}({target}) "
                    f"reward=0.00 done=false error={last_error}",
                    flush=True,
                )
                rewards.append(0.0)
                history.append(f"Step {step}: {action_type}({target}) ERROR")
                break

            action_str = f"{action_type}({target})"
            history.append(f"Step {step}: {action_str} -> reward {obs.reward:+.2f}")
            rewards.append(obs.reward)
            done_str = "true" if obs.done else "false"

            # ══════════════════════════════════════════════════════════════════
            # [STEP] — exact spec format, one line per step
            # ══════════════════════════════════════════════════════════════════
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={obs.reward:.2f} done={done_str} error={last_error}",
                flush=True,
            )

            if obs.final_score is not None:
                final_score = obs.final_score
            else:
                final_score = obs.cumulative_reward

            if obs.done:
                break

        success = final_score > 0.0

    except Exception as e:
        print(f"# run_task error: {e}", flush=True)
        success = False

    finally:
        # ══════════════════════════════════════════════════════════════════════
        # [END] ALWAYS emitted — even on exception (spec requirement)
        # ══════════════════════════════════════════════════════════════════════
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        success_str = "true" if success else "false"
        final_score = round(min(max(final_score, 0.0001), 0.9999), 4)
        print(
            f"[END] success={success_str} steps={len(rewards)} "
            f"score={final_score:.4f} rewards={rewards_str}",
            flush=True,
        )

    return {
        "task_id":         task_id,
        "final_score":     final_score,
        "score_breakdown": obs.score_breakdown,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        print("# WARNING: No API key — LLM calls will fail; fallbacks will run", flush=True)
    if not MODEL_NAME:
        print("# WARNING: No MODEL_NAME set", flush=True)

    env = DirectEnv(base_url=HF_SPACE_URL)
    if not env.health():
        print(f"# WARNING: Cannot reach {HF_SPACE_URL} — tasks will fail gracefully", flush=True)

    print(f"# Server: {HF_SPACE_URL}", flush=True)
    print(f"# Model:  {MODEL_NAME}",   flush=True)

    llm     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy-key")
    results = []

    for task_id in TASKS:
        results.append(run_task(llm, env, task_id))

    print(f"\n# {'='*58}", flush=True)
    print(f"# BASELINE RESULTS",       flush=True)
    print(f"# {'='*58}",               flush=True)
    total = 0.0
    for r in results:
        print(f"#  {r['task_id']:<10} score: {r['final_score']:.4f}", flush=True)
        total += r["final_score"]
    print(f"#  {'AVERAGE':<10} score: {total / len(results):.4f}", flush=True)


if __name__ == "__main__":
    main()
