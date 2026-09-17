"""Label-free dual crossover action rules and ambiguity abstention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.gemini.huy_d1_aiteam_novel_consensus_v1.common import (
    RESULTS_DIR,
    sha256_file,
)


def compute_novel_consensus_actions(
    all_ids: List[str],
    d1_top5: Dict[str, List[str]],
    novel_map: Dict[str, List[str]],
    aiteam20_rankings: Dict[str, List[str]],
    jina_scores: Dict[str, Dict[str, float]],
    sec_scores: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, List[str]], Dict[str, Any], str]:
    """Execute label-free dual crossover rules on U(q) and apply ambiguity abstention.
    
    STRICT LEAKAGE RULE: Zero gold labels are read, materialized, or evaluated here.
    """
    print("--- COMPUTING LABEL-FREE DUAL CROSSOVER ACTIONS & SEAL ---", flush=True)

    records = []
    repaired_preds: Dict[str, List[str]] = {}
    action_qids = []
    ambiguity_abstain_qids = []

    total_novel_cands = 0
    jina_crossover_count = 0
    sec_crossover_count = 0
    dual_eligible_count = 0

    for q in all_ids:
        defender = d1_top5[q][4]  # exact D1 rank 5
        novels = novel_map[q]
        aiteam20 = aiteam20_rankings.get(q, [])

        u_docs = list(dict.fromkeys(d1_top5[q] + novels))

        # Rank U(q) by Jina-FT: higher score first, tie break by doc ID
        q_jina = jina_scores[q]
        jina_ranked = sorted(u_docs, key=lambda d: (-q_jina.get(d, -1e9), str(d)))
        def_jina_rank = jina_ranked.index(defender) + 1
        def_jina_score = float(q_jina.get(defender, -1e9))

        # Rank U(q) by Section CE: higher score first, tie break by doc ID
        q_sec = sec_scores[q]
        sec_ranked = sorted(u_docs, key=lambda d: (-q_sec.get(d, -1e9), str(d)))
        def_sec_rank = sec_ranked.index(defender) + 1
        def_sec_score = float(q_sec.get(defender, -1e9))

        candidate_evaluations = []
        eligible_cands = []

        for c in novels:
            total_novel_cands += 1
            aiteam_rank = aiteam20.index(c) + 1 if c in aiteam20 else 999

            c_jina_score = float(q_jina.get(c, -1e9))
            c_jina_rank = jina_ranked.index(c) + 1

            cond_j_score = c_jina_score > def_jina_score
            cond_j_c_top5 = c_jina_rank <= 5
            cond_j_def_out = def_jina_rank > 5
            jina_crossover = bool(cond_j_score and cond_j_c_top5 and cond_j_def_out)
            if jina_crossover:
                jina_crossover_count += 1

            c_sec_score = float(q_sec.get(c, -1e9))
            c_sec_rank = sec_ranked.index(c) + 1

            cond_s_score = c_sec_score > def_sec_score
            cond_s_c_top5 = c_sec_rank <= 5
            cond_s_def_out = def_sec_rank > 5
            sec_crossover = bool(cond_s_score and cond_s_c_top5 and cond_s_def_out)
            if sec_crossover:
                sec_crossover_count += 1

            is_eligible = bool(jina_crossover and sec_crossover)
            if is_eligible:
                dual_eligible_count += 1
                eligible_cands.append(c)

            candidate_evaluations.append({
                "candidate_doc_id": c,
                "aiteam_rank": aiteam_rank,
                "jina": {
                    "score": c_jina_score,
                    "rank_in_u": c_jina_rank,
                    "score_gt_defender": cond_j_score,
                    "rank_le_5": cond_j_c_top5,
                    "defender_rank_gt_5": cond_j_def_out,
                    "crossover_passed": jina_crossover,
                },
                "section": {
                    "score": c_sec_score,
                    "rank_in_u": c_sec_rank,
                    "score_gt_defender": cond_s_score,
                    "rank_le_5": cond_s_c_top5,
                    "defender_rank_gt_5": cond_s_def_out,
                    "crossover_passed": sec_crossover,
                },
                "eligible": is_eligible,
            })

        # Ambiguity Abstention Rule:
        # len(ELIGIBLE) == 1 -> Swap exact D1 rank 5
        # Otherwise -> ABSTAIN / KEEP
        n_eligible = len(eligible_cands)
        action_fire = (n_eligible == 1)
        ambiguity_abstain = (n_eligible > 1)

        if action_fire:
            selected_challenger = eligible_cands[0]
            new_top5 = d1_top5[q][:4] + [selected_challenger]
            action_qids.append(q)
        else:
            selected_challenger = None
            new_top5 = list(d1_top5[q])
            if ambiguity_abstain:
                ambiguity_abstain_qids.append(q)

        repaired_preds[q] = new_top5

        records.append({
            "qid": q,
            "d1_top5": d1_top5[q],
            "defender_doc_id": defender,
            "defender_jina": {
                "score": def_jina_score,
                "rank_in_u": def_jina_rank,
            },
            "defender_section": {
                "score": def_sec_score,
                "rank_in_u": def_sec_rank,
            },
            "novel_count": len(novels),
            "novel_candidates": candidate_evaluations,
            "eligible_count": n_eligible,
            "eligible_candidate_doc_ids": eligible_cands,
            "ambiguity_abstain": ambiguity_abstain,
            "action_fire": action_fire,
            "selected_challenger": selected_challenger,
            "new_top5": new_top5,
        })

    actions_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.actions_label_free.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
        "summary": {
            "total_queries": len(all_ids),
            "total_novel_candidates": total_novel_cands,
            "jina_crossover_count": jina_crossover_count,
            "section_crossover_count": sec_crossover_count,
            "dual_consensus_eligible_count": dual_eligible_count,
            "unique_action_query_count": len(action_qids),
            "ambiguity_abstention_query_count": len(ambiguity_abstain_qids),
            "action_fraction": float(len(action_qids) / len(all_ids)),
        },
        "actions": records,
    }

    out_path = RESULTS_DIR / "NOVEL_CONSENSUS_ACTIONS_LABEL_FREE.json"
    out_path.write_text(json.dumps(actions_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    action_seal_sha = sha256_file(out_path)
    print(f"Wrote and sealed {out_path} (SHA256: {action_seal_sha})", flush=True)
    print(f"  Total Novel Candidates:          {total_novel_cands}", flush=True)
    print(f"  Jina Crossover Count:            {jina_crossover_count}", flush=True)
    print(f"  Section Crossover Count:         {sec_crossover_count}", flush=True)
    print(f"  Dual Consensus Eligible Count:   {dual_eligible_count}", flush=True)
    print(f"  Unique Action Query Count:       {len(action_qids)}", flush=True)
    print(f"  Ambiguity Abstention Count:      {len(ambiguity_abstain_qids)}", flush=True)

    return repaired_preds, actions_doc, action_seal_sha
