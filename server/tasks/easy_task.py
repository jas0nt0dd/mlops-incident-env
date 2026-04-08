"""
easy_task.py — Grader v2 for Task 1: Data Quality Alert
Process-gated scoring: investigation quality + efficiency bonus + red herring penalty
"""
from __future__ import annotations
from typing import Any, Dict, List


class EasyTaskGrader:
    RED_HERRINGS = {"api_gateway": -0.15, "data_pipeline_a": -0.10}

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None) -> Dict[str, Any]:
        text = (str(diagnosis_target) + " " + str(diagnosis_params)).lower().replace("_", " ").replace("-", " ")
        path = [p.lower() for p in (investigation_path or [])]
        score = 0.0
        breakdown: Dict[str, float] = {}

        # Gate: correct component (required for bonus points)
        if any(kw in text for kw in ["data pipeline b", "datapipelineb", "pipeline b"]):
            breakdown["identified_data_pipeline_b"] = 0.40
            score += 0.40
            # Bonus keywords only count if gate passed
            for kw, pts in [("schema", 0.10), ("migration", 0.10), ("transaction amount", 0.10)]:
                if kw in text:
                    breakdown[f"mentioned_{kw.replace(' ','_')}"] = pts
                    score += pts
                else:
                    breakdown[f"mentioned_{kw.replace(' ','_')}"] = 0.0
        else:
            breakdown["identified_data_pipeline_b"] = 0.0
            for kw, pts in [("schema", 0.10), ("migration", 0.10), ("transaction amount", 0.10)]:
                breakdown[f"mentioned_{kw.replace(' ','_')}"] = 0.0
            # Red herring penalties only if wrong answer
            for rh, pen in self.RED_HERRINGS.items():
                if rh.replace("_", " ") in text:
                    breakdown[f"blamed_{rh}_penalty"] = pen
                    score += pen
                else:
                    breakdown[f"blamed_{rh}_penalty"] = 0.0

        # Investigation quality bonus
        iq = 0.0
        iq += 0.10 if any("querylogs" in p and "pipeline" in p for p in path) else 0.0
        iq += 0.08 if any("checkmetrics" in p and "feature" in p for p in path) else 0.0
        iq += 0.07 if any("inspect" in p and "pipeline" in p for p in path) else 0.0
        breakdown["investigation_quality_bonus"] = round(iq, 4)
        score += iq

        # Efficiency bonus
        eff = 0.15 if (step_count and step_count <= 4) else (0.08 if (step_count and step_count <= 6) else (0.03 if (step_count and step_count <= 8) else 0.0))
        breakdown["efficiency_bonus"] = round(eff, 4)
        score += eff

        return {"total": round(min(max(score, 0.0001), 0.9999), 4), "breakdown": breakdown}
