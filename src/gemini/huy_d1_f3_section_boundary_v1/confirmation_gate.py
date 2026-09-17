"""Confirmation set evaluation and gating for HUY_D1_F3_SECTION_BOUNDARY_V1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.gemini.huy_d1_f3_section_boundary_v1.common import (
    CAL600_5FOLD_PATH,
    EXPECTED_5FOLD_SPLIT_SHA256,
    RESULTS_DIR,
    sha256_file,
)


def load_confirmation_split() -> Tuple[Dict[str, List[str]], List[str], Dict[str, Any]]:
    """Load stratified 5-fold split and extract F3_CONFIRMATION_240 (fold_3 + fold_4)."""
    split_sha = sha256_file(CAL600_5FOLD_PATH)
    if split_sha != EXPECTED_5FOLD_SPLIT_SHA256:
        raise RuntimeError(
            f"5-fold split SHA mismatch! Expected {EXPECTED_5FOLD_SPLIT_SHA256}, got {split_sha}"
        )

    split_raw = json.loads(CAL600_5FOLD_PATH.read_text(encoding="utf-8"))
    folds = split_raw["folds"]
    conf_240_ids = folds["fold_3"] + folds["fold_4"]

    if len(conf_240_ids) != 240:
        raise ValueError(f"Expected 240 confirmation queries, got {len(conf_240_ids)}")

    return folds, conf_240_ids, split_raw


def evaluate_confirmation_gate(
    folds: Dict[str, List[str]],
    conf_ids: List[str],
    d1_preds: Dict[str, List[str]],
    action_qids: List[str],
    proposals_records: List[dict],
    gold: Dict[str, Set[str]],
) -> Tuple[Dict[str, Any], bool, str]:
    """Evaluate primary rule on F3_CONFIRMATION_240 and test confirmation gates."""
    print("=== EVALUATING F3_CONFIRMATION_240 GATES ===", flush=True)

    action_set = set(action_qids)
    conf_set = set(conf_ids)

    # Proposals in confirmation set
    conf_proposals = [r for r in proposals_records if r["qid"] in conf_set and r["f3_crossover_conditions"]["f3_crossover_passed"]]
    conf_actions = [r for r in proposals_records if r["qid"] in conf_set and r["final_action_fire"]]

    # Construct candidate predictions on confirmation set
    cand_preds = {}
    for q in conf_ids:
        top5 = list(d1_preds[q][:5])
        if q in action_set:
            # Swap rank 5 with rank 6
            chal = next(r["challenger_doc_id"] for r in proposals_records if r["qid"] == q)
            top5 = top5[:4] + [chal]
        cand_preds[q] = top5

    # Compute baseline and candidate metrics on confirmation set
    base_recalls = [len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in conf_ids]
    cand_recalls = [len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in conf_ids]

    base_precisions = [len(set(d1_preds[q][:5]) & gold[q]) / 5.0 for q in conf_ids]
    cand_precisions = [len(set(cand_preds[q][:5]) & gold[q]) / 5.0 for q in conf_ids]

    mean_base_recall = float(np.mean(base_recalls))
    mean_cand_recall = float(np.mean(cand_recalls))
    recall_delta = float(mean_cand_recall - mean_base_recall)

    mean_base_prec = float(np.mean(base_precisions))
    mean_cand_prec = float(np.mean(cand_precisions))
    prec_delta = float(mean_cand_prec - mean_base_prec)

    # Paired wins/losses/ties
    wins = 0
    losses = 0
    ties = 0
    beneficial = 0
    harmful = 0
    neutral = 0

    for q in conf_ids:
        r_b = len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q]))
        r_c = len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q]))
        diff = r_c - r_b

        if diff > 1e-9:
            wins += 1
            if q in action_set:
                beneficial += 1
        elif diff < -1e-9:
            losses += 1
            if q in action_set:
                harmful += 1
        else:
            ties += 1
            if q in action_set:
                neutral += 1

    # Single vs Multi gold on confirmation set
    single_ids = [q for q in conf_ids if len(gold[q]) == 1]
    multi_ids = [q for q in conf_ids if len(gold[q]) > 1]

    single_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in single_ids]))
    single_cand = float(np.mean([len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in single_ids]))
    single_delta = float(single_cand - single_base)

    multi_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in multi_ids]))
    multi_cand = float(np.mean([len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in multi_ids]))
    multi_delta = float(multi_cand - multi_base)

    # Per fold metrics for fold_3 and fold_4
    fold3_ids = folds["fold_3"]
    fold4_ids = folds["fold_4"]

    fold3_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in fold3_ids]))
    fold3_cand = float(np.mean([len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in fold3_ids]))
    fold3_delta = float(fold3_cand - fold3_base)

    fold4_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in fold4_ids]))
    fold4_cand = float(np.mean([len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in fold4_ids]))
    fold4_delta = float(fold4_cand - fold4_base)

    # Confirmation Gate conditions:
    n_actions = len(conf_actions)
    gate_actions_ge_1 = n_actions >= 1
    gate_actions_le_8 = n_actions <= 8
    gate_harmful_eq_0 = harmful == 0
    gate_beneficial_ge_1 = beneficial >= 1
    gate_recall_gt_0 = recall_delta > 0
    gate_prec_no_decrease = prec_delta >= -1e-9
    gate_single_no_decrease = single_delta >= -1e-9
    gate_multi_no_decrease = multi_delta >= -1e-9
    gate_folds_no_decrease = (fold3_delta >= -1e-9) and (fold4_delta >= -1e-9)

    all_gates = {
        "gate_01_actions_ge_1": gate_actions_ge_1,
        "gate_02_actions_le_8": gate_actions_le_8,
        "gate_03_harmful_eq_0": gate_harmful_eq_0,
        "gate_04_beneficial_ge_1": gate_beneficial_ge_1,
        "gate_05_recall_delta_gt_0": gate_recall_gt_0,
        "gate_06_precision_no_decrease": gate_prec_no_decrease,
        "gate_07_single_gold_no_decrease": gate_single_no_decrease,
        "gate_08_multi_gold_no_decrease": gate_multi_no_decrease,
        "gate_09_neither_fold_decreases": gate_folds_no_decrease,
    }

    confirmation_passed = all(all_gates.values())
    conf_verdict = "PASS_F3_SECTION_BOUNDARY_CONFIRMATION" if confirmation_passed else "KILL_F3_SECTION_BOUNDARY_CONFIRMATION"

    report_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_f3_section_boundary_v1.confirmation_report.v1",
        "experiment_id": "HUY_D1_F3_SECTION_BOUNDARY_V1",
        "confirmation_set_size": len(conf_ids),
        "confirmation_folds": ["fold_3", "fold_4"],
        "proposals_count": len(conf_proposals),
        "actions_count": n_actions,
        "action_breakdown": {
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
        },
        "metrics": {
            "recall_at_5": {
                "baseline": mean_base_recall,
                "candidate": mean_cand_recall,
                "delta": recall_delta,
            },
            "precision_at_5": {
                "baseline": mean_base_prec,
                "candidate": mean_cand_prec,
                "delta": prec_delta,
            },
            "pairwise_vs_baseline": {
                "wins": wins,
                "losses": losses,
                "ties": ties,
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
            "fold_3_recall": {
                "baseline": fold3_base,
                "candidate": fold3_cand,
                "delta": fold3_delta,
            },
            "fold_4_recall": {
                "baseline": fold4_base,
                "candidate": fold4_cand,
                "delta": fold4_delta,
            },
        },
        "gates_evaluation": all_gates,
        "confirmation_passed": confirmation_passed,
        "verdict": conf_verdict,
    }

    out_path = RESULTS_DIR / "F3_CONFIRMATION_240_REPORT.json"
    out_path.write_text(json.dumps(report_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    return report_doc, confirmation_passed, conf_verdict


def evaluate_full_cal(
    all_ids: List[str],
    blocks: Dict[str, List[str]],
    d1_preds: Dict[str, List[str]],
    action_qids: List[str],
    proposals_records: List[dict],
    gold: Dict[str, Set[str]],
) -> Tuple[Dict[str, Any], str]:
    """Evaluate full CAL600 metrics ONLY IF confirmation passed."""
    print("=== EVALUATING FULL CAL600 PROMOTION GATES ===", flush=True)
    action_set = set(action_qids)

    cand_preds = {}
    for q in all_ids:
        top5 = list(d1_preds[q][:5])
        if q in action_set:
            chal = next(r["challenger_doc_id"] for r in proposals_records if r["qid"] == q)
            top5 = top5[:4] + [chal]
        cand_preds[q] = top5

    # Metrics
    base_recalls = [len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in all_ids]
    cand_recalls = [len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in all_ids]

    base_precisions = [len(set(d1_preds[q][:5]) & gold[q]) / 5.0 for q in all_ids]
    cand_precisions = [len(set(cand_preds[q][:5]) & gold[q]) / 5.0 for q in all_ids]

    single_ids = [q for q in all_ids if len(gold[q]) == 1]
    multi_ids = [q for q in all_ids if len(gold[q]) > 1]

    single_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in single_ids]))
    single_cand = float(np.mean([len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in single_ids]))

    multi_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in multi_ids]))
    multi_cand = float(np.mean([len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in multi_ids]))

    block_metrics = {}
    for b in sorted(blocks.keys()):
        b_ids = blocks[b]
        b_base = float(np.mean([len(set(d1_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in b_ids]))
        b_cand = float(np.mean([len(set(cand_preds[q][:5]) & gold[q]) / max(1, len(gold[q])) for q in b_ids]))
        block_metrics[b] = {
            "baseline": b_base,
            "candidate": b_cand,
            "delta": b_cand - b_base,
        }

    wins = 0
    losses = 0
    ties = 0
    beneficial = 0
    harmful = 0
    neutral = 0
    gold_in = 0
    gold_out = 0

    for q in all_ids:
        b_set = set(d1_preds[q][:5])
        c_set = set(cand_preds[q][:5])
        diff = (len(c_set & gold[q]) - len(b_set & gold[q])) / max(1, len(gold[q]))

        if diff > 1e-9:
            wins += 1
            if q in action_set:
                beneficial += 1
        elif diff < -1e-9:
            losses += 1
            if q in action_set:
                harmful += 1
        else:
            ties += 1
            if q in action_set:
                neutral += 1

        in_g = sum(1 for d in (c_set - b_set) if d in gold[q])
        out_g = sum(1 for d in (b_set - c_set) if d in gold[q])
        gold_in += in_g
        gold_out += out_g

    mean_base_recall = float(np.mean(base_recalls))
    mean_cand_recall = float(np.mean(cand_recalls))
    mean_base_prec = float(np.mean(base_precisions))
    mean_cand_prec = float(np.mean(cand_precisions))

    # Full CAL Promotion Gates
    gate_actions_ge_2 = len(action_qids) >= 2
    gate_actions_le_15 = len(action_qids) <= 15
    gate_beneficial_ge_2 = beneficial >= 2
    gate_harmful_eq_0 = harmful == 0
    gate_recall_ge_0015 = mean_cand_recall >= mean_base_recall + 0.0015
    gate_prec_ge_base = mean_cand_prec >= mean_base_prec
    gate_paired_losses_eq_0 = losses == 0
    gate_no_block_decreases = all(block_metrics[b]["delta"] >= -1e-9 for b in block_metrics)
    gate_single_no_decrease = single_cand >= single_base - 1e-9
    gate_multi_no_decrease = multi_cand >= multi_base - 1e-9

    all_gates = {
        "gate_01_total_actions_ge_2": gate_actions_ge_2,
        "gate_02_total_actions_le_15": gate_actions_le_15,
        "gate_03_beneficial_ge_2": gate_beneficial_ge_2,
        "gate_04_harmful_eq_0": gate_harmful_eq_0,
        "gate_05_recall_gain_ge_0015": gate_recall_ge_0015,
        "gate_06_precision_no_decrease": gate_prec_ge_base,
        "gate_07_paired_losses_eq_0": gate_paired_losses_eq_0,
        "gate_08_no_block_decreases": gate_no_block_decreases,
        "gate_09_single_gold_no_decrease": gate_single_no_decrease,
        "gate_10_multi_gold_no_decrease": gate_multi_no_decrease,
    }

    full_pass = all(all_gates.values())
    final_verdict = "LOCAL_PROMOTE_F3_SECTION_BOUNDARY_V1" if full_pass else "KILL_F3_SECTION_BOUNDARY_V1"

    full_cal_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_f3_section_boundary_v1.full_cal_report.v1",
        "experiment_id": "HUY_D1_F3_SECTION_BOUNDARY_V1",
        "metrics": {
            "recall_at_5": {
                "baseline": mean_base_recall,
                "candidate": mean_cand_recall,
                "delta": mean_cand_recall - mean_base_recall,
            },
            "precision_at_5": {
                "baseline": mean_base_prec,
                "candidate": mean_cand_prec,
                "delta": mean_cand_prec - mean_base_prec,
            },
            "single_gold_recall": {
                "baseline": single_base,
                "candidate": single_cand,
                "delta": single_cand - single_base,
            },
            "multi_gold_recall": {
                "baseline": multi_base,
                "candidate": multi_cand,
                "delta": multi_cand - multi_base,
            },
            "block_recalls": block_metrics,
        },
        "actions": {
            "total": len(action_qids),
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
            "action_qids": action_qids,
        },
        "pairwise_vs_baseline": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "churn": {
            "ordered_churn": len(action_qids),
            "set_churn": len(action_qids),
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
        },
        "promotion_gates": all_gates,
        "full_cal_pass": full_pass,
        "final_verdict": final_verdict,
    }

    out_path = RESULTS_DIR / "F3_SECTION_BOUNDARY_FULL_CAL.json"
    out_path.write_text(json.dumps(full_cal_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    return full_cal_doc, final_verdict
