"""Label-free F3 strict crossover and Section CE confirmation rules for HUY_D1_F3_SECTION_BOUNDARY_V1."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.gemini.huy_d1_f3_section_boundary_v1.common import (
    CAL_FROZEN_SECTION_CACHE_PATH,
    F3_FEASIBILITY_PATH,
    F3_MODEL_PATH,
    F3_SCORES_PATH,
    F3_SOURCE_PATH,
    RESULTS_DIR,
    ROOT,
    sha256_file,
)


def load_frozen_expert_caches() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Load frozen F3 diagonal metric scores and frozen Section-CE CV scores."""
    f3_scores = pickle.loads(F3_SCORES_PATH.read_bytes())
    sec_raw = pickle.loads(CAL_FROZEN_SECTION_CACHE_PATH.read_bytes())
    sec_scores = sec_raw.get("scores", sec_raw)
    return f3_scores, sec_scores


def evaluate_f3_coverage(
    all_ids: List[str],
    d1_rankings: Dict[str, List[str]],
    extended: Dict[str, List[str]],
    f3_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Verify F3 coverage on exact D1 rank 5 and rank 6."""
    r5_covered = 0
    r6_covered = 0
    total_cands = 0
    covered_cands = 0

    for q in all_ids:
        q_f3 = f3_scores.get(q, {})
        def_doc = d1_rankings[q][4]
        chal_doc = d1_rankings[q][5]

        if def_doc in q_f3:
            r5_covered += 1
        if chal_doc in q_f3:
            r6_covered += 1

        cands = extended[q]
        total_cands += len(cands)
        covered_cands += sum(1 for d in cands if d in q_f3)

    r5_cov_ratio = float(r5_covered / len(all_ids))
    r6_cov_ratio = float(r6_covered / len(all_ids))
    full_cand_cov_ratio = float(covered_cands / max(1, total_cands))

    coverage_passed = (r5_covered == len(all_ids)) and (r6_covered == len(all_ids))

    coverage_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_f3_section_boundary_v1.f3_coverage.v1",
        "experiment_id": "HUY_D1_F3_SECTION_BOUNDARY_V1",
        "status": "PASS" if coverage_passed else "FAIL",
        "historical_f3": {
            "source_path": str(F3_SOURCE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256_file(F3_SOURCE_PATH),
            "model_path": str(F3_MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": sha256_file(F3_MODEL_PATH),
            "score_cache_path": str(F3_SCORES_PATH.relative_to(ROOT)).replace("\\", "/"),
            "score_cache_sha256": sha256_file(F3_SCORES_PATH),
            "feasibility_path": str(F3_FEASIBILITY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "feasibility_sha256": sha256_file(F3_FEASIBILITY_PATH),
            "training_query_count": 1046,
            "cal_overlap": 0,
            "hard_negative_count": 17574,
        },
        "coverage": {
            "total_queries": len(all_ids),
            "rank5_covered_count": r5_covered,
            "rank5_coverage_ratio": r5_cov_ratio,
            "rank6_covered_count": r6_covered,
            "rank6_coverage_ratio": r6_cov_ratio,
            "coverage_gate_passed": coverage_passed,
            "descriptive_full_candidate_coverage": {
                "total_candidate_pairs": total_cands,
                "covered_candidate_pairs": covered_cands,
                "candidate_coverage_ratio": full_cand_cov_ratio,
            },
        },
    }

    out_path = RESULTS_DIR / "F3_PROVENANCE_AND_COVERAGE.json"
    out_path.write_text(json.dumps(coverage_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    return coverage_doc


def compute_label_free_proposals(
    all_ids: List[str],
    d1_rankings: Dict[str, List[str]],
    extended: Dict[str, List[str]],
    f3_scores: Dict[str, Dict[str, float]],
    sec_scores: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Any], List[dict], List[str]]:
    """Compute label-free proposals across all 600 queries.
    
    STRICT LEAKAGE RULE: Zero gold labels are read, materialized, or evaluated here.
    """
    records = []
    action_qids = []
    crossover_qids = []

    for q in all_ids:
        ranking = d1_rankings[q]
        defender = ranking[4]    # D1 rank 5
        challenger = ranking[5]  # D1 rank 6

        q_cands = extended[q]
        q_f3 = f3_scores.get(q, {})
        q_sec = sec_scores.get(q, {})

        # 1. F3 ranking over all candidates: higher score = better, tie-break by doc ID
        f3_ranked = sorted(q_cands, key=lambda d: (-q_f3.get(d, -1e9), d))
        f3_def_rank = f3_ranked.index(defender) + 1 if defender in f3_ranked else 999
        f3_chal_rank = f3_ranked.index(challenger) + 1 if challenger in f3_ranked else 999

        f3_def_score = float(q_f3.get(defender, -1e9))
        f3_chal_score = float(q_f3.get(challenger, -1e9))

        # F3 strict boundary crossover
        cond_f3_score = f3_chal_score > f3_def_score
        cond_f3_chal_top5 = f3_chal_rank <= 5
        cond_f3_def_outside = f3_def_rank > 5
        f3_crossover = bool(cond_f3_score and cond_f3_chal_top5 and cond_f3_def_outside)

        if f3_crossover:
            crossover_qids.append(q)

        # 2. Section CE confirmation
        sec_def_score = float(q_sec.get(defender, -1e9))
        sec_chal_score = float(q_sec.get(challenger, -1e9))
        sec_has_scores = (defender in q_sec) and (challenger in q_sec)

        sec_ranked = sorted(q_sec.keys(), key=lambda d: (-q_sec[d], d))
        sec_chal_rank = sec_ranked.index(challenger) + 1 if challenger in sec_ranked else 999
        sec_def_rank = sec_ranked.index(defender) + 1 if defender in sec_ranked else 999

        cond_sec_score = sec_chal_score > sec_def_score
        cond_sec_top5 = sec_chal_rank <= 5

        sec_confirmed = bool(sec_has_scores and cond_sec_score and cond_sec_top5)

        # 3. Final action
        action_fire = bool(f3_crossover and sec_confirmed)
        if action_fire:
            action_qids.append(q)

        records.append({
            "qid": q,
            "defender_doc_id": defender,
            "challenger_doc_id": challenger,
            "d1_rank_defender": 5,
            "d1_rank_challenger": 6,
            "f3_scores": {
                "defender": f3_def_score,
                "challenger": f3_chal_score,
                "delta": float(f3_chal_score - f3_def_score),
            },
            "f3_ranks": {
                "defender": f3_def_rank,
                "challenger": f3_chal_rank,
            },
            "section_scores": {
                "defender": sec_def_score,
                "challenger": sec_chal_score,
                "delta": float(sec_chal_score - sec_def_score),
                "both_exist": sec_has_scores,
            },
            "section_ranks": {
                "defender": sec_def_rank,
                "challenger": sec_chal_rank,
            },
            "f3_crossover_conditions": {
                "challenger_score_higher": cond_f3_score,
                "challenger_f3_rank_le_5": cond_f3_chal_top5,
                "defender_f3_rank_gt_5": cond_f3_def_outside,
                "f3_crossover_passed": f3_crossover,
            },
            "section_confirmation_conditions": {
                "both_scores_exist": sec_has_scores,
                "challenger_score_higher": cond_sec_score,
                "challenger_in_section_top5": cond_sec_top5,
                "section_confirmation_passed": sec_confirmed,
            },
            "final_action_fire": action_fire,
        })

    proposals_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_f3_section_boundary_v1.proposals_label_free.v1",
        "experiment_id": "HUY_D1_F3_SECTION_BOUNDARY_V1",
        "summary": {
            "total_queries": len(all_ids),
            "f3_crossover_count": len(crossover_qids),
            "section_confirmed_actions_count": len(action_qids),
            "action_fraction": float(len(action_qids) / max(1, len(all_ids))),
        },
        "proposals": records,
    }

    out_path = RESULTS_DIR / "F3_SECTION_PROPOSALS_LABEL_FREE.json"
    out_path.write_text(json.dumps(proposals_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    return proposals_doc, records, action_qids
