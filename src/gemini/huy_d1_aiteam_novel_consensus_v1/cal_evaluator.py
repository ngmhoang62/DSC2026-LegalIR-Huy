"""CAL evaluation, outside-pool rescue diagnostic, and promotion gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.gemini.huy_d1_aiteam_novel_consensus_v1.common import (
    EXPECTED_D1_R5,
    RESULTS_DIR,
)


def evaluate_novel_consensus_cal(
    all_ids: List[str],
    blocks: Dict[str, List[str]],
    extended: Dict[str, List[str]],
    d1_preds: Dict[str, List[str]],
    repaired_preds: Dict[str, List[str]],
    aiteam20_rankings: Dict[str, List[str]],
    actions_records: List[dict],
    gold: Dict[str, Set[str]],
    d1_parity_passed: bool,
) -> Tuple[Dict[str, Any], str]:
    """Evaluate CAL metrics, perform rescue diagnostics, and assess promotion gates."""
    print("--- EVALUATING CAL600 UTILITY AND PROMOTION GATES ---", flush=True)

    # Base R0 vs Candidate R1 recalls & precisions
    base_recalls = [len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in all_ids]
    cand_recalls = [len(set(repaired_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in all_ids]

    base_precisions = [len(set(d1_preds[q][:5]) & gold[q]) / 5.0 for q in all_ids]
    cand_precisions = [len(set(repaired_preds[q][:5]) & gold[q]) / 5.0 for q in all_ids]

    mean_base_recall = float(np.mean(base_recalls))
    mean_cand_recall = float(np.mean(cand_recalls))
    delta_recall = float(mean_cand_recall - mean_base_recall)

    mean_base_prec = float(np.mean(base_precisions))
    mean_cand_prec = float(np.mean(cand_precisions))
    delta_prec = float(mean_cand_prec - mean_base_prec)

    # Single vs Multi gold
    single_ids = [q for q in all_ids if len(gold[q]) == 1]
    multi_ids = [q for q in all_ids if len(gold[q]) > 1]

    single_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in single_ids]))
    single_cand = float(np.mean([len(set(repaired_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in single_ids]))
    single_delta = float(single_cand - single_base)

    multi_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in multi_ids]))
    multi_cand = float(np.mean([len(set(repaired_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in multi_ids]))
    multi_delta = float(multi_cand - multi_base)

    # Block metrics
    block_metrics = {}
    for b in sorted(blocks.keys()):
        b_ids = blocks[b]
        b_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in b_ids]))
        b_cand = float(np.mean([len(set(repaired_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in b_ids]))
        block_metrics[b] = {
            "baseline": b_base,
            "candidate": b_cand,
            "delta": float(b_cand - b_base),
        }

    # Paired metrics, actions, churn
    wins = 0
    losses = 0
    ties = 0
    beneficial_actions = 0
    harmful_actions = 0
    neutral_actions = 0
    gold_in = 0
    gold_out = 0
    ordered_churn = 0
    set_churn = 0

    action_details = []

    for rec in actions_records:
        q = rec["qid"]
        b_top5 = d1_preds[q][:5]
        c_top5 = repaired_preds[q][:5]

        b_set = set(b_top5)
        c_set = set(c_top5)
        g = gold[q]

        r_base = len(b_set & g) / max(1, len(g))
        r_cand = len(c_set & g) / max(1, len(g))
        diff = r_cand - r_base

        if b_top5 != c_top5:
            ordered_churn += 1
        if b_set != c_set:
            set_churn += 1

            in_docs = list(c_set - b_set)
            out_docs = list(b_set - c_set)
            for d in in_docs:
                if d in g:
                    gold_in += 1
            for d in out_docs:
                if d in g:
                    gold_out += 1

            if diff > 1e-9:
                beneficial_actions += 1
                effect = "BENEFICIAL"
                wins += 1
            elif diff < -1e-9:
                harmful_actions += 1
                effect = "HARMFUL"
                losses += 1
            else:
                neutral_actions += 1
                effect = "NEUTRAL"
                ties += 1

            action_details.append({
                "qid": q,
                "entered_doc": in_docs[0] if in_docs else None,
                "evicted_defender": out_docs[0] if out_docs else None,
                "effect": effect,
                "recall_delta": diff,
            })
        else:
            ties += 1

    total_actions = beneficial_actions + harmful_actions + neutral_actions

    # Outside-pool rescue diagnostic
    print("Computing outside-pool rescue diagnostic...", flush=True)
    outside_pool_golds_in_aiteam20 = 0
    rescued_unique_eligible = 0
    rescued_entered_top5 = 0
    rejected_by_jina = 0
    rejected_by_section = 0
    rejected_by_ambiguity = 0
    diagnostic_details = []

    for rec in actions_records:
        q = rec["qid"]
        d1_pool = set(extended[q])
        g = gold[q]
        outside_gold = [d for d in g if d not in d1_pool]

        for og in outside_gold:
            if og in aiteam20_rankings.get(q, []):
                outside_pool_golds_in_aiteam20 += 1

                # Find candidate evaluation
                c_eval = next((c for c in rec["novel_candidates"] if c["candidate_doc_id"] == og), None)
                if c_eval:
                    j_pass = c_eval["jina"]["crossover_passed"]
                    s_pass = c_eval["section"]["crossover_passed"]
                    is_elig = c_eval["eligible"]

                    if is_elig:
                        if rec["action_fire"]:
                            rescued_unique_eligible += 1
                            rescued_entered_top5 += 1
                            outcome = "RESCUED_ENTERED_TOP5"
                        else:
                            rejected_by_ambiguity += 1
                            outcome = "REJECTED_BY_AMBIGUITY_ABSTENTION"
                    else:
                        if not j_pass and not s_pass:
                            rejected_by_jina += 1
                            rejected_by_section += 1
                            outcome = "REJECTED_BY_BOTH_JINA_AND_SECTION"
                        elif not j_pass:
                            rejected_by_jina += 1
                            outcome = "REJECTED_BY_JINA"
                        else:
                            rejected_by_section += 1
                            outcome = "REJECTED_BY_SECTION"

                    diagnostic_details.append({
                        "qid": q,
                        "gold_doc_id": og,
                        "outcome": outcome,
                        "jina_crossover": j_pass,
                        "section_crossover": s_pass,
                        "eligible_count_in_query": rec["eligible_count"],
                    })

    rescue_diagnostic = {
        "outside_pool_gold_occurrences_in_aiteam20": outside_pool_golds_in_aiteam20,
        "rescued_unique_eligible_count": rescued_unique_eligible,
        "rescued_entered_top5_count": rescued_entered_top5,
        "rejected_by_jina_count": rejected_by_jina,
        "rejected_by_section_count": rejected_by_section,
        "rejected_by_ambiguity_count": rejected_by_ambiguity,
        "details": diagnostic_details,
    }

    # Promotion Gates Evaluation
    gate_d1_parity = d1_parity_passed
    gate_actions_ge_2 = total_actions >= 2
    gate_actions_le_12 = total_actions <= 12
    gate_beneficial_ge_2 = beneficial_actions >= 2
    gate_harmful_eq_0 = harmful_actions == 0
    gate_recall_ge_0015 = mean_cand_recall >= (EXPECTED_D1_R5 + 0.0015 - 1e-9)
    gate_prec_ge_base = mean_cand_prec >= (mean_base_prec - 1e-9)
    gate_losses_eq_0 = losses == 0
    gate_no_block_regress = all(block_metrics[b]["delta"] >= -1e-9 for b in block_metrics)
    gate_single_no_decrease = single_delta >= -1e-9
    gate_multi_no_decrease = multi_delta >= -1e-9

    promotion_gates = {
        "gate_01_d1_parity_pass": gate_d1_parity,
        "gate_02_total_actions_ge_2": gate_actions_ge_2,
        "gate_03_total_actions_le_12": gate_actions_le_12,
        "gate_04_beneficial_ge_2": gate_beneficial_ge_2,
        "gate_05_harmful_eq_0": gate_harmful_eq_0,
        "gate_06_recall_gain_ge_0015": gate_recall_ge_0015,
        "gate_07_precision_ge_baseline": gate_prec_ge_base,
        "gate_08_paired_losses_eq_0": gate_losses_eq_0,
        "gate_09_no_block_regression": gate_no_block_regress,
        "gate_10_single_gold_no_decrease": gate_single_no_decrease,
        "gate_11_multi_gold_no_decrease": gate_multi_no_decrease,
    }

    all_passed = all(promotion_gates.values())
    final_verdict = "LOCAL_PROMOTE_AITEAM_NOVEL_CONSENSUS_V1" if all_passed else "KILL_AITEAM_NOVEL_CONSENSUS_V1"

    cal_report_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.cal_report.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
        "metrics": {
            "recall_at_5": {
                "baseline": mean_base_recall,
                "candidate": mean_cand_recall,
                "delta": delta_recall,
            },
            "precision_at_5": {
                "baseline": mean_base_prec,
                "candidate": mean_cand_prec,
                "delta": delta_prec,
            },
            "single_gold_recall": {
                "baseline": single_base,
                "candidate": single_cand,
                "delta": single_delta,
            },
            "multi_gold_recall": {
                "baseline": multi_base,
                "candidate": multi_cand,
                "delta": multi_delta,
            },
            "block_recalls": block_metrics,
        },
        "pairwise": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "actions": {
            "total": total_actions,
            "beneficial": beneficial_actions,
            "harmful": harmful_actions,
            "neutral": neutral_actions,
            "action_details": action_details,
        },
        "churn": {
            "ordered_churn": ordered_churn,
            "set_churn": set_churn,
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
        },
        "outside_pool_rescue_diagnostic": rescue_diagnostic,
        "promotion_gates": promotion_gates,
        "promotion_passed": all_passed,
        "final_verdict": final_verdict,
    }

    out_path = RESULTS_DIR / "NOVEL_CONSENSUS_CAL_REPORT.json"
    out_path.write_text(json.dumps(cal_report_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} (Final Verdict: {final_verdict})", flush=True)

    return cal_report_doc, final_verdict
