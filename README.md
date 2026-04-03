---
title: MLOps Incident Response Environment
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# MLOps Incident Response Environment

A production-grade RL environment where AI agents act as on-call ML engineers,
diagnosing and resolving real-world machine learning system incidents.

Built for the **META x PyTorch OpenEnv Hackathon**.

## Baseline Scores

Evaluated using `llama-3.1-8b-instant` via Groq API:

| Task        | Difficulty | Score      |
|-------------|------------|------------|
| easy        | Easy       | 1.0000     |
| medium      | Medium     | 1.0000     |
| hard        | Hard       | 1.0000     |
| cascade     | Elite      | 1.0000     |
| **Average** |            | **1.0000** |

Scores reflect process-gated grading (v2). A random agent scores ~0.10 on cascade.

## API Endpoints

- `POST /reset` — Start episode: `{"task_id": "easy|medium|hard|cascade"}`
- `POST /step` — Take action: `{"action_type": "inspect", "target": "data_pipeline_b", "parameters": {}}`
- `GET /health` — Health check

## Tasks

| Task    | Scenario                                        | Key Skill             |
|---------|-------------------------------------------------|-----------------------|
| Easy    | Data pipeline schema mismatch → accuracy drop   | Log analysis          |
| Medium  | Batch size config change → latency spike        | Config comparison     |
| Hard    | Feature drift → silent revenue degradation      | Drift detection       |
| Cascade | 3 simultaneous failures from one bad deploy     | Cross-system synthesis|
