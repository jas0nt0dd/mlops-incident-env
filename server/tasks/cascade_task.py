"""
cascade_task.py — Grader for Task 4: Cascade Failure (ELITE difficulty)

Scenario: 3 simultaneous failures triggered by a single deployment
  Root causes (ALL must be identified for full score):
    1. embedding_service_v3 — ONNX runtime version mismatch → corrupted embeddings
    2. feature_store         — cache TTL set to 0s → serving stale features to ALL models
    3. ab_test_router        — traffic split config corrupted → 100% traffic to model B (untested)

  This task simulates a real production cascade where one bad deploy causes 3 downstream failures.
  An agent must synthesize evidence across ALL 3 subsystems to score > 0.7.

Scoring (max 1.0) — must demonstrate cross-system synthesis:
  Identifies embedding_service issue        +0.20
  Identifies feature_store cache issue      +0.20
  Identifies ab_test_router misconfiguration +0.20
  Connects all 3 to single deployment       +0.15
  Proposes coordinated rollback plan        +0.15
  Investigation covered all 3 subsystems    +0.10 (process bonus)

  Partial penalties:
  - Only finds 1 root cause                 capped at 0.35
  - Only finds 2 root causes                capped at 0.70
"""
from __future__ import annotations
from typing import Any, Dict, List


class CascadeTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None) -> Dict[str, Any]:
        text = (str(diagnosis_target) + " " + str(diagnosis_params)).lower().replace("_", " ").replace("-", " ")
        path = [p.lower() for p in (investigation_path or [])]
        score = 0.0
        breakdown: Dict[str, float] = {}
        causes_found = 0

        # Root cause 1: embedding service ONNX mismatch
        emb_score = 0.0
        if any(kw in text for kw in ["embedding service", "embedding", "onnx", "runtime mismatch", "corrupted embedding"]):
            emb_score = 0.20
            causes_found += 1
        breakdown["identifies_embedding_service_issue"] = emb_score
        score += emb_score

        # Root cause 2: feature store cache TTL
        fs_score = 0.0
        if any(kw in text for kw in ["feature store", "featurestore", "cache", "ttl", "stale feature", "cache ttl"]):
            fs_score = 0.20
            causes_found += 1
        breakdown["identifies_feature_store_cache_issue"] = fs_score
        score += fs_score

        # Root cause 3: ab test router
        ab_score = 0.0
        if any(kw in text for kw in ["ab test router", "ab router", "traffic split", "model b", "routing config", "ab test"]):
            ab_score = 0.20
            causes_found += 1
        breakdown["identifies_ab_router_misconfiguration"] = ab_score
        score += ab_score

        # Synthesis: connects all 3 to single deployment
        synthesis_score = 0.0
        if causes_found >= 3 and any(kw in text for kw in ["single deployment", "same deploy", "deployment v", "deploy rollout", "one change caused"]):
            synthesis_score = 0.15
        elif causes_found >= 2:
            synthesis_score = 0.05
        breakdown["connects_all_causes_to_deployment"] = synthesis_score
        score += synthesis_score

        # Remediation: coordinated rollback
        rollback_score = 0.0
        if any(kw in text for kw in ["rollback", "revert", "coordinated", "all services", "full rollback", "restore all"]):
            rollback_score = 0.15 if causes_found >= 3 else 0.07
        breakdown["proposes_coordinated_rollback"] = rollback_score
        score += rollback_score

        # Investigation breadth bonus
        investigated_embedding = any("embedding" in p for p in path)
        investigated_featurestore = any("feature" in p for p in path)
        investigated_ab = any("ab" in p or "router" in p for p in path)
        inv_breadth = sum([investigated_embedding, investigated_featurestore, investigated_ab])
        inv_bonus = round((inv_breadth / 3) * 0.10, 4)
        breakdown["investigation_breadth_bonus"] = inv_bonus
        score += inv_bonus

        # Cap score based on number of causes found (prevents partial guessing)
        if causes_found == 0:
            score = min(score, 0.10)
        elif causes_found == 1:
            score = min(score, 0.35)
        elif causes_found == 2:
            score = min(score, 0.70)

        breakdown["causes_found"] = float(causes_found)
        return {"total": round(min(max(score, 0.0), 1.0), 4), "breakdown": breakdown}
