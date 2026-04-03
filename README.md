# 🚨 MLOps Incident Response Environment

> **An RL environment where AI agents act as on-call ML engineers.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-teal)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/HF%20Space-Live-yellow)](https://huggingface.co/spaces/your-username/mlops-incident-env)

## 🎯 Environment Description

Production ML systems fail in subtle, hard-to-diagnose ways. This environment simulates real ML
production incidents and trains agents to investigate, reason, and resolve them — just like a
senior ML engineer on-call.

**Why this matters:** Every ML team at large-scale companies deals with production incidents daily.
An agent that can autonomously diagnose root causes saves engineering hours and reduces MTTR
(Mean Time to Resolve) from hours to seconds.

## 🧩 Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `easy`   | ⭐ Easy   | 10 | Data Quality Alert — model accuracy dropped 18%. Find the broken pipeline. |
| `medium` | ⭐⭐ Medium | 15 | Latency Spike — P99 latency 4x over SLA. Find the bottleneck + config change. |
| `hard`   | ⭐⭐⭐ Hard | 25 | Silent Model Drift — revenue down 12%, zero alerts. Detect feature distribution shift. |

## 🎮 Action Space

| action_type | Description |
|-------------|-------------|
| `inspect` | View component overview and current status |
| `query_logs` | Fetch filtered logs for a component |
| `check_metrics` | Get time-series metrics for a component |
| `compare_configs` | Diff current vs previous deployment config |
| `check_feature_drift` | View feature PSI drift scores |
| `submit_diagnosis` | Submit final root cause + remediation plan |
| `request_rollback` | Trigger rollback of last deployment |

## 📊 Observation Space

```python
MLOpsObservation(
    goal: str,                    # Task description
    alert_summary: str,           # Incident alert
    component_status: dict,       # Health of each service
    recent_logs: list,            # Latest log entries
    metrics_snapshot: dict,       # Current metrics
    action_feedback: str,         # Result of last action
    step_count: int,
    reward: float,
    cumulative_reward: float,
    done: bool,
    final_score: float | None,
    score_breakdown: dict | None,
)
```

## 🏆 Reward Design

- **+0.05** per relevant component inspected
- **+0.10** per key evidence log found
- **+0.15** for finding the config diff (medium/hard)
- **+0.40–0.50** for correct diagnosis
- **-0.10** for false diagnosis submission
- **-0.15** for repeated actions (loop detection)

## 🚀 Quick Start

```bash
pip install openenv-core
```

```python
from client import MLOpsEnv
from models import MLOpsAction

env = MLOpsEnv(base_url="https://your-space.hf.space")
obs = env.reset(task_id="easy")

# Investigate
result = env.step(MLOpsAction(action_type="query_logs", target="data_pipeline_b"))
result = env.step(MLOpsAction(
    action_type="submit_diagnosis",
    target="data_pipeline_b",
    parameters={"root_cause": "schema_mismatch", "fix": "revert migration v2.3.1"}
))
print(result.observation.final_score)  # 1.0
```

## 📈 Baseline Scores

| Model | Easy | Medium | Hard | Average |
|-------|------|--------|------|---------|
| *TBD after deployment* | - | - | - | - |

## 🔧 Setup

```bash
cd server && docker build -t mlops-incident-env:latest .
docker run -p 8000:8000 mlops-incident-env:latest
```

## 📋 Pre-submission Validation

```bash
./scripts/validate-submission.sh https://your-space.hf.space
```
