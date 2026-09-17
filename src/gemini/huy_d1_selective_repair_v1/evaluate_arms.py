"""Evaluation of Selective Repair Arms R0, R1, R2, R3 and Production Gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .common import (
    EXPECTED_BLOCK_RECALLS,
    EXPECTED_D1_R5,
)


def compute_query_recalls(
    predictions: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    all_ids: List[str],
) -> Dict[str, float]:
    """Compute Recall@5 for each query."""
    return {
        q: float(len(set(predictions[q][:5]) & gold[q]) / max(1, len(gold[q])))
        for q in all_ids
    }


def compute_arm_metrics(
    arm_name: str,
    predictions: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    all_ids: List[str],
    blocks: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute standard metrics for an arm on CAL600."""
    recalls = [
        len(set(predictions[q][:5]) & gold[q]) / max(1, len(gold[q]))
        for q in all_ids
    ]
    precisions = [
        len(set(predictions[q][:5]) & gold[q]) / 5.0
        for q in all_ids
    ]

    single_ids = [q for q in all_ids if len(gold[q]) == 1]
    multi_ids = [q for q in all_ids if len(gold[q]) > 1]

    single_recalls = [
        len(set(predictions[q][:5]) & gold[q]) / max(1, len(gold[q]))
        for q in single_ids
    ]
    multi_recalls = [
        len(set(predictions[q][:5]) & gold[q]) / max(1, len(gold[q]))
        for q in multi_ids
    ]

    block_recalls = {}
    for b in sorted(blocks.keys()):
        b_ids = blocks[b]
        block_recalls[b] = float(np.mean([
            len(set(predictions[q][:5]) & gold[q]) / max(1, len(gold[q]))
            for q in b_ids
        ]))

    return {
        "arm": arm_name,
        "recall_at_5": float(np.mean(recalls)),
        "precision_at_5": float(np.mean(precisions)),
        "single_gold_recall_at_5": float(np.mean(single_recalls)),
        "multi_gold_recall_at_5": float(np.mean(multi_recalls)),
        "block_recalls": block_recalls,
    }


def compute_intervention_churn_and_utility(
    baseline_preds: Dict[str, List[str]],
    repaired_preds: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    all_ids: List[str],
) -> Dict[str, Any]:
    """Compute intervention counts, utility, and churn vs baseline."""
    total_interventions = 0
    beneficial = 0
    harmful = 0
    neutral = 0
    gold_crossings_in = 0
    gold_crossings_out = 0
    set_churn_count = 0
    ordered_churn_count = 0
    net_utility = 0.0

    wins = 0
    losses = 0
    ties = 0

    action_details = []

    for q in all_ids:
        b_top5 = baseline_preds[q][:5]
        r_top5 = repaired_preds[q][:5]

        b_set = set(b_top5)
        r_set = set(r_top5)
        g = gold[q]

        if b_top5 != r_top5:
            ordered_churn_count += 1
        if b_set != r_set:
            set_churn_count += 1
            total_interventions += 1

            entered = list(r_set - b_set)
            left = list(b_set - r_set)

            r_base = len(b_set & g) / max(1, len(g))
            r_rep = len(r_set & g) / max(1, len(g))
            delta_q = r_rep - r_base
            net_utility += delta_q

            in_g = sum(1 for d in entered if d in g)
            out_g = sum(1 for d in left if d in g)
            gold_crossings_in += in_g
            gold_crossings_out += out_g

            if delta_q > 1e-9:
                beneficial += 1
                effect = "BENEFICIAL"
                wins += 1
            elif delta_q < -1e-9:
                harmful += 1
                effect = "HARMFUL"
                losses += 1
            else:
                neutral += 1
                effect = "NEUTRAL"
                ties += 1

            action_details.append({
                "qid": q,
                "entered_docs": entered,
                "left_docs": left,
                "effect": effect,
                "baseline_hits": len(b_set & g),
                "repaired_hits": len(r_set & g),
                "recall_delta": float(delta_q),
            })
        else:
            ties += 1

    return {
        "total_interventions": total_interventions,
        "beneficial_interventions": beneficial,
        "harmful_interventions": harmful,
        "neutral_interventions": neutral,
        "gold_crossings_in": gold_crossings_in,
        "gold_crossings_out": gold_crossings_out,
        "top5_set_churn": set_churn_count,
        "ordered_top5_churn": ordered_churn_count,
        "beneficial_fraction": float(beneficial / max(1, total_interventions)),
        "net_recall_utility": float(net_utility),
        "net_recall_utility_per_action": float(net_utility / max(1, total_interventions)),
        "pairwise_vs_r0": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "action_details": action_details,
    }


def evaluate_parity(r0_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Verify exact parity of R0 baseline against frozen D1 champion."""
    r5_match = abs(r0_metrics["recall_at_5"] - EXPECTED_D1_R5) < 1e-12
    block_matches = {
        b: abs(r0_metrics["block_recalls"][b] - EXPECTED_BLOCK_RECALLS[b]) < 1e-12
        for b in EXPECTED_BLOCK_RECALLS
    }
    all_blocks_match = all(block_matches.values())
    parity_pass = r5_match and all_blocks_match

    return {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.d1_parity.v1",
        "status": "PASS" if parity_pass else "FAIL",
        "recall_at_5": {
            "reproduced": r0_metrics["recall_at_5"],
            "expected": EXPECTED_D1_R5,
            "match": r5_match,
        },
        "block_recalls": {
            b: {
                "reproduced": r0_metrics["block_recalls"][b],
                "expected": EXPECTED_BLOCK_RECALLS[b],
                "match": block_matches[b],
            }
            for b in EXPECTED_BLOCK_RECALLS
        },
        "parity_pass": parity_pass,
    }
