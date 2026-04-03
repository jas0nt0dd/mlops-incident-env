"""
easy_task.py — Grader for Task 1: Data Quality Alert.

Root cause: data_pipeline_b schema mismatch.
  - transaction_amount field changed FLOAT → STRING after schema migration v2.3.1
  - Causes 34% null rate in feature_store → model accuracy drops 0.87 → 0.69

Scoring (max 1.0):
  - Identifies data_pipeline_b       → 0.60  (required)
  - Mentions "schema" or "migration" → +0.20
  - Mentions "transaction_amount"    → +0.20
"""
from __future__ import annotations
from typing import Any, Dict


class EasyTaskGrader:
    REQUIRED_KEYWORD = "data_pipeline_b"
    BONUS_KEYWORDS = {
        "schema":             0.10,
        "migration":          0.10,
        "transaction_amount": 0.20,
    }

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any]) -> Dict[str, Any]:
        text = (
            str(diagnosis_target) + " " + str(diagnosis_params)
        ).lower().replace("_", " ").replace("-", " ")

        score = 0.0
        breakdown: Dict[str, float] = {}

        if "data pipeline b" in text or "data_pipeline_b" in text or "pipeline b" in text:
            breakdown["identified_data_pipeline_b"] = 0.60
            score += 0.60
        else:
            breakdown["identified_data_pipeline_b"] = 0.0

        for keyword, points in self.BONUS_KEYWORDS.items():
            kw_normalized = keyword.replace("_", " ")
            if kw_normalized in text or keyword in text:
                breakdown[f"mentioned_{keyword}"] = points
                score += points
            else:
                breakdown[f"mentioned_{keyword}"] = 0.0

        return {"total": round(min(score, 1.0), 4), "breakdown": breakdown}