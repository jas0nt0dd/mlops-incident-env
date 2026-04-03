"""
hard_task.py — Grader v2 for Task 3: Silent Model Drift
Deep multi-criteria: drift ID + root cause chain + business impact + remediation plan + process depth
No shortcuts — must demonstrate full investigation chain to score high
"""
from __future__ import annotations
from typing import Any, Dict, List


class HardTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None) -> Dict[str, Any]:
        text = (str(diagnosis_target) + " " + str(diagnosis_params)).lower().replace("_", " ").replace("-", " ")
        path = [p.lower() for p in (investigation_path or [])]
        score = 0.0
        breakdown: Dict[str, float] = {}

        # 1. Drift identification (0.30)
        eng_score = 0.20 if any(kw in text for kw in ["user engagement score", "user engagement", "engagement score"]) else 0.0
        psi_score = 0.10 if any(kw in text for kw in ["psi", "population stability", "distribution shift", "0.38", "covariate"]) else 0.0
        breakdown["identifies_correct_drifted_feature"] = eng_score
        breakdown["mentions_psi_evidence"] = psi_score
        score += eng_score + psi_score

        # 2. Root cause chain (0.30)
        redesign_score = 0.15 if any(kw in text for kw in ["ui redesign", "redesign", "a4 41", "a441", "ab test", "experiment", "interface change"]) else 0.0
        stale_score = 0.15 if any(kw in text for kw in ["stale", "60 day", "not retrained", "outdated model", "old model", "model age"]) else 0.0
        breakdown["connects_ui_redesign_experiment"] = redesign_score
        breakdown["mentions_model_staleness"] = stale_score
        score += redesign_score + stale_score

        # 3. Business impact (0.15)
        biz_score = 0.08 if any(kw in text for kw in ["revenue", "business impact", "conversion", "downstream"]) else 0.0
        quant_score = 0.07 if any(kw in text for kw in ["12%", "12 percent", "3 day", "three day", "gradual", "silent"]) else 0.0
        breakdown["connects_drift_to_revenue"] = biz_score
        breakdown["quantifies_impact"] = quant_score
        score += biz_score + quant_score

        # 4. Remediation plan (0.15) — all 3 needed for full marks
        retrain_score = 0.05 if any(kw in text for kw in ["retrain", "fine tune", "fine-tune", "update model"]) else 0.0
        window_score = 0.05 if any(kw in text for kw in ["data window", "training window", "cutoff", "post redesign", "recent data"]) else 0.0
        monitor_score = 0.05 if any(kw in text for kw in ["monitor", "drift detection", "psi threshold", "continuous", "schedule"]) else 0.0
        breakdown["mentions_retraining"] = retrain_score
        breakdown["specifies_data_window"] = window_score
        breakdown["proposes_ongoing_monitoring"] = monitor_score
        score += retrain_score + window_score + monitor_score

        # 5. Investigation depth (0.10)
        drift_used = 0.05 if any("checkfeaturedrift" in p for p in path) else 0.0
        biz_checked = 0.05 if any("checkmetrics" in p for p in path) else 0.0
        breakdown["used_checkfeaturedrift_action"] = drift_used
        breakdown["checked_metrics_during_investigation"] = biz_checked
        score += drift_used + biz_checked

        # 6. Red herring / wrong direction penalties
        infra_wrong = any(kw in text for kw in ["oom", "out of memory", "latency spike", "pipeline error", "schema mismatch"])
        if infra_wrong and eng_score == 0.0:
            breakdown["wrong_direction_infra_penalty"] = -0.20
            score -= 0.20
        else:
            breakdown["wrong_direction_infra_penalty"] = 0.0

        wrong_feat = any(kw in text for kw in ["transaction amount", "click through rate", "page load", "api response time"])
        if wrong_feat and eng_score == 0.0:
            breakdown["wrong_feature_penalty"] = -0.10
            score -= 0.10
        else:
            breakdown["wrong_feature_penalty"] = 0.0

        return {"total": round(min(max(score, 0.0), 1.0), 4), "breakdown": breakdown}
