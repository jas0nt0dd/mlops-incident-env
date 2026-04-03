"""
hard_task.py — Grader for Task 3: Silent Model Drift.

Root cause: user_engagement_score PSI=0.38 (critical drift) due to UI redesign
            + model v4.2 stale (60 days not retrained)

Scoring (max 1.0):
  - Identifies user_engagement_score drift   → 0.35
  - Mentions PSI / distribution shift        → 0.15
  - Connects to UI redesign / experiment     → 0.15
  - Mentions model staleness / 60 days       → 0.20
  - Provides remediation (retrain)           → 0.15
"""
from __future__ import annotations
from typing import Any, Dict


class HardTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any]) -> Dict[str, Any]:
        text = (
            str(diagnosis_target) + " " + str(diagnosis_params)
        ).lower().replace("_", " ").replace("-", " ")

        score = 0.0
        breakdown: Dict[str, float] = {}

        if any(kw in text for kw in [
            "user engagement score", "user engagement", "engagement score", "engagement_score",
        ]):
            breakdown["identifies_user_engagement_score_drift"] = 0.35
            score += 0.35
        else:
            breakdown["identifies_user_engagement_score_drift"] = 0.0

        if any(kw in text for kw in ["psi", "distribution", "drift", "shift", "covariate"]):
            breakdown["mentions_drift_evidence"] = 0.15
            score += 0.15
        else:
            breakdown["mentions_drift_evidence"] = 0.0

        if any(kw in text for kw in [
            "ui redesign", "redesign", "experiment", "a441", "a/b", "ab test", "new ui",
        ]):
            breakdown["connects_to_ui_redesign_experiment"] = 0.15
            score += 0.15
        else:
            breakdown["connects_to_ui_redesign_experiment"] = 0.0

        if any(kw in text for kw in [
            "stale", "60 day", "60days", "not retrained", "outdated model", "old model",
        ]):
            breakdown["mentions_model_staleness_60_days"] = 0.20
            score += 0.20
        else:
            breakdown["mentions_model_staleness_60_days"] = 0.0

        if any(kw in text for kw in [
            "retrain", "retraining", "fine tune", "fine-tune", "update model", "remediat",
        ]):
            breakdown["provides_remediation_retrain"] = 0.15
            score += 0.15
        else:
            breakdown["provides_remediation_retrain"] = 0.0

        return {"total": round(min(score, 1.0), 4), "breakdown": breakdown}