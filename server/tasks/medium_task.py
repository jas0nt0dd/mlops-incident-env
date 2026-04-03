"""
medium_task.py — Grader v2 for Task 2: Latency Spike Investigation
Multi-criteria: component ID + config depth + fix quality + process bonus + penalties
"""
from __future__ import annotations
from typing import Any, Dict, List


class MediumTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None) -> Dict[str, Any]:
        text = (str(diagnosis_target) + " " + str(diagnosis_params)).lower().replace("_", " ").replace("-", " ")
        path = [p.lower() for p in (investigation_path or [])]
        score = 0.0
        breakdown: Dict[str, float] = {}

        # Gate: component identification
        if any(kw in text for kw in ["feature preprocessor v2", "preprocessor v2", "preprocessorv2"]):
            breakdown["correct_component"] = 0.40
            score += 0.40
        elif "preprocessor" in text:
            breakdown["correct_component"] = 0.15  # partial
            score += 0.15
        else:
            breakdown["correct_component"] = 0.0

        # Root cause depth
        has_correct = breakdown["correct_component"] > 0
        batch_score = 0.20 if any(kw in text for kw in ["batch size", "batchsize", "512"]) else 0.0
        memory_score = 0.15 if any(kw in text for kw in ["memory", "oom", "out of memory", "leak"]) else 0.0
        breakdown["batch_size_identified"] = batch_score
        breakdown["memory_issue_mentioned"] = memory_score
        score += batch_score + memory_score

        # Fix quality
        rollback_score = 0.08 if any(kw in text for kw in ["rollback", "revert", "restore", "previous config"]) else 0.0
        correct_size = 0.07 if ("32" in text and "batch" in text) else 0.0
        breakdown["proposes_rollback"] = rollback_score
        breakdown["correct_batch_size_32"] = correct_size
        score += rollback_score + correct_size

        # Investigation process bonus
        cmp_bonus = 0.05 if any("compareconfigs" in p for p in path) else 0.0
        met_bonus = 0.05 if any("checkmetrics" in p and "preprocessor" in p for p in path) else 0.0
        breakdown["used_compareconfigs"] = cmp_bonus
        breakdown["checked_metrics"] = met_bonus
        score += cmp_bonus + met_bonus

        # Red herring penalty (only if wrong component identified)
        if not has_correct:
            ms_pen = -0.15 if "model server" in text else 0.0
            ag_pen = -0.10 if "api gateway" in text else 0.0
            breakdown["model_server_blame_penalty"] = ms_pen
            breakdown["api_gateway_blame_penalty"] = ag_pen
            score += ms_pen + ag_pen
        else:
            breakdown["model_server_blame_penalty"] = 0.0
            breakdown["api_gateway_blame_penalty"] = 0.0

        # Efficiency bonus
        eff = 0.10 if (step_count and step_count <= 6) else (0.05 if (step_count and step_count <= 9) else 0.0)
        breakdown["efficiency_bonus"] = round(eff, 4)
        score += eff

        return {"total": round(min(max(score, 0.0), 1.0), 4), "breakdown": breakdown}
