"""
medium_task.py — Grader for Task 2: Latency Spike Investigation.

Root cause: feature_preprocessor_v2 — batch_size 32→512, memory leak → OOM

Scoring (max 1.0):
  - Identifies feature_preprocessor_v2  → 0.50
  - Mentions batch_size / "512"         → 0.30
  - Mentions memory / OOM / leak        → 0.20
"""
from __future__ import annotations
from typing import Any, Dict


class MediumTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any]) -> Dict[str, Any]:
        text = (
            str(diagnosis_target) + " " + str(diagnosis_params)
        ).lower().replace("_", " ").replace("-", " ")

        score = 0.0
        breakdown: Dict[str, float] = {}

        if (
            "feature preprocessor v2" in text
            or "preprocessor v2" in text
            or "preprocessorv2" in text
            or ("preprocessor" in text and "v2" in text)
        ):
            breakdown["correct_component_preprocessor_v2"] = 0.50
            score += 0.50
        else:
            breakdown["correct_component_preprocessor_v2"] = 0.0

        if "batch size" in text or "batch_size" in text or "512" in text:
            breakdown["correct_config_batch_size"] = 0.30
            score += 0.30
        else:
            breakdown["correct_config_batch_size"] = 0.0

        if any(kw in text for kw in ["memory", "oom", "out of memory", "leak", "ram"]):
            breakdown["mentions_memory_issue"] = 0.20
            score += 0.20
        else:
            breakdown["mentions_memory_issue"] = 0.0

        return {"total": round(min(score, 1.0), 4), "breakdown": breakdown}