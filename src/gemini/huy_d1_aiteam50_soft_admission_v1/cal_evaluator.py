"""CAL utility evaluation, outside-pool rescue diagnostic, and promotion gate check.

Module for HUY_D1_AITEAM50_SOFT_ADMISSION_V1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import (
    AITEAM_REPORT_PATH,
    RESULTS_DIR,
    ROOT,
    load_cal_gold_labels,
)

# Descriptive reference only for post-seal diagnostics (never used in model/action logic)
HISTORICAL_OUT_OF_POOL_CASES = [
    {"qid": "62306", "doc_id": "199984"},
    {"qid": "87852", "doc_id": "189230"},
    {"qid": "148022", "doc_id": "270895"},
    {"qid": "105016", "doc_id": "180120"},
    {"qid": "107146", "doc_id": "305455"},
    {"qid": "19560", "doc_id": "27980"},
]


def evaluate_cal_utility(
    all_ids: List[str],
    blocks: Dict[str, List[str]],
    frozen_top50: Dict[str, List[str]],
    s_universe: Dict[str, List[str]],
    d1_top5: Dict[str, List[str]],
    repaired_top5: Dict[str, List[str]],
    action_records: Dict[str, Any],
    action_meta: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Strictly loads CAL gold labels AFTER action artifact seal, computes all metrics,
    outside-pool diagnostics, and evaluates promotion gates.
    """
    print("=== EVALUATING CAL UTILITY STRICTLY AFTER ACTION SEAL ===", flush=True)
    t0 = time.perf_counter()

    gold, reveal_time_utc = load_cal_gold_labels(all_ids)

    # 1. Standalone AITeam Source Metrics
    aiteam_r5 = float(np.mean([len(set(frozen_top50[q][:5]) & gold[q]) / len(gold[q]) for q in all_ids]))
    aiteam_r20 = float(np.mean([len(set(frozen_top50[q][:20]) & gold[q]) / len(gold[q]) for q in all_ids]))
    aiteam_r50 = float(np.mean([len(set(frozen_top50[q][:50]) & gold[q]) / len(gold[q]) for q in all_ids]))

    # 2. Standalone Admission Model Recall@5 within S(q)
    adm_r5_list = []
    for q in all_ids:
        adm_ranks = action_records[q]["admission_ranks"]
        adm_top5 = [d for d, r in adm_ranks.items() if r <= 5]
        adm_r5_list.append(len(set(adm_top5) & gold[q]) / len(gold[q]))
    admission_standalone_r5 = float(np.mean(adm_r5_list))

    # 3. Baseline (R0) vs Repaired (R1) Metrics
    r0_recalls = [len(set(d1_top5[q][:5]) & gold[q]) / len(gold[q]) for q in all_ids]
    r1_recalls = [len(set(repaired_top5[q][:5]) & gold[q]) / len(gold[q]) for q in all_ids]
    r0_mean_r5 = float(np.mean(r0_recalls))
    r1_mean_r5 = float(np.mean(r1_recalls))

    r0_precisions = [len(set(d1_top5[q][:5]) & gold[q]) / 5.0 for q in all_ids]
    r1_precisions = [len(set(repaired_top5[q][:5]) & gold[q]) / 5.0 for q in all_ids]
    r0_mean_p5 = float(np.mean(r0_precisions))
    r1_mean_p5 = float(np.mean(r1_precisions))

    # Single vs Multi gold splits
    single_qids = [q for q in all_ids if len(gold[q]) == 1]
    multi_qids = [q for q in all_ids if len(gold[q]) > 1]

    r0_single_r5 = float(np.mean([r0_recalls[all_ids.index(q)] for q in single_qids]))
    r1_single_r5 = float(np.mean([r1_recalls[all_ids.index(q)] for q in single_qids]))

    r0_multi_r5 = float(np.mean([r0_recalls[all_ids.index(q)] for q in multi_qids]))
    r1_multi_r5 = float(np.mean([r1_recalls[all_ids.index(q)] for q in multi_qids]))

    # Per block metrics
    block_metrics = {}
    for b_name, b_qids in blocks.items():
        b_indices = [all_ids.index(q) for q in b_qids]
        b_r0 = float(np.mean([r0_recalls[i] for i in b_indices]))
        b_r1 = float(np.mean([r1_recalls[i] for i in b_indices]))
        block_metrics[b_name] = {
            "r0_recall_at_5": b_r0,
            "r1_recall_at_5": b_r1,
            "delta_recall": b_r1 - b_r0,
            "no_decrease": b_r1 >= b_r0 - 1e-9,
        }

    # Paired Wins, Losses, Ties
    wins, losses, ties = 0, 0, 0
    beneficial, harmful, neutral = 0, 0, 0
    gold_crossings_in, gold_crossings_out = 0, 0
    set_churn, ordered_churn = 0, 0

    per_query_audit = []
    defender_adm_ranks = []
    selected_novel_aiteam_ranks = []
    selected_novel_adm_ranks = []

    frozen_aiteam_ranks = {
        q: {d: r + 1 for r, d in enumerate(frozen_top50[q])}
        for q in all_ids
    }

    for i, q in enumerate(all_ids):
        r0_r = r0_recalls[i]
        r1_r = r1_recalls[i]
        rec = action_records[q]
        fired = rec["action_fire"]
        def_doc = rec["d1_defender"]
        best_novel = rec["best_novel"]

        if r1_r > r0_r:
            wins += 1
        elif r1_r < r0_r:
            losses += 1
        else:
            ties += 1

        if set(d1_top5[q][:5]) != set(repaired_top5[q][:5]):
            set_churn += 1
        if d1_top5[q][:5] != repaired_top5[q][:5]:
            ordered_churn += 1

        if fired:
            defender_adm_ranks.append(rec["defender_admission_rank"])
            selected_novel_adm_ranks.append(rec["best_novel_admission_rank"])
            novel_aiteam_rank = frozen_aiteam_ranks[q].get(best_novel, None)
            selected_novel_aiteam_ranks.append(novel_aiteam_rank)

            def_in_gold = def_doc in gold[q]
            novel_in_gold = best_novel in gold[q]

            if novel_in_gold and not def_in_gold:
                beneficial += 1
                gold_crossings_in += 1
                action_type = "BENEFICIAL"
            elif not novel_in_gold and def_in_gold:
                harmful += 1
                gold_crossings_out += 1
                action_type = "HARMFUL"
            else:
                neutral += 1
                action_type = "NEUTRAL"

            per_query_audit.append({
                "qid": q,
                "fold": rec["fold"],
                "action_type": action_type,
                "d1_defender": def_doc,
                "defender_in_gold": def_in_gold,
                "defender_adm_rank": rec["defender_admission_rank"],
                "best_novel": best_novel,
                "novel_in_gold": novel_in_gold,
                "novel_aiteam_rank": novel_aiteam_rank,
                "novel_adm_rank": rec["best_novel_admission_rank"],
                "r0_recall": r0_r,
                "r1_recall": r1_r,
            })

    # 4. Outside-Pool Rescue Diagnostic
    outside_pool_reports = []
    adm_placed_top5_count = 0
    actually_admitted_count = 0
    remained_below_top5_count = 0

    for case in HISTORICAL_OUT_OF_POOL_CASES:
        q = case["qid"]
        target_doc = case["doc_id"]
        rec = action_records[q]
        adm_ranks = rec["admission_ranks"]
        target_adm_rank = adm_ranks.get(target_doc, None)
        target_aiteam_rank = frozen_aiteam_ranks[q].get(target_doc, None)

        in_s = target_doc in s_universe[q]
        in_novel = target_doc in rec["novel_candidates"]
        is_best_novel = (rec["best_novel"] == target_doc)
        actually_selected = (rec["action_fire"] and is_best_novel)

        if target_adm_rank is not None and target_adm_rank <= 5:
            adm_placed_top5_count += 1
        else:
            remained_below_top5_count += 1

        if actually_selected:
            actually_admitted_count += 1
            reason = "SUCCESS_ADMITTED_TO_TOP5"
        elif not in_s:
            reason = "ABSENT_FROM_S_UNIVERSE"
        elif target_adm_rank is None:
            reason = "NOT_SCORED"
        elif target_adm_rank > 5:
            reason = f"ADMISSION_RANK_EXCEEDS_5 (rank={target_adm_rank})"
        elif not is_best_novel:
            reason = f"OTHER_NOVEL_RANKED_HIGHER (best={rec['best_novel']})"
        elif rec["defender_admission_rank"] <= 5:
            reason = f"DEFENDER_RANK_WITHIN_5 (def_rank={rec['defender_admission_rank']})"
        else:
            reason = "OTHER"

        outside_pool_reports.append({
            "qid": q,
            "gold_doc": target_doc,
            "aiteam_rank": target_aiteam_rank,
            "admission_rank": target_adm_rank,
            "selected_into_top5": actually_selected,
            "reason": reason,
        })

    # 5. Pre-registered Promotion Gates (0.96 Sprint)
    total_actions = action_meta["total_actions"]
    gates = {
        "gate_1_d1_parity": True,
        "gate_2_r1_recall_at_least_0_96": r1_mean_r5 >= 0.9600000000,
        "gate_3_r1_precision_no_decrease": r1_mean_p5 >= r0_mean_p5,
        "gate_4_harmful_actions_zero": harmful == 0,
        "gate_5_paired_losses_zero": losses == 0,
        "gate_6_beneficial_actions_at_least_2": beneficial >= 2,
        "gate_7_total_actions_at_most_20": total_actions <= 20,
        "gate_8_no_block_decrease": all(b["no_decrease"] for b in block_metrics.values()),
        "gate_9_single_gold_no_decrease": r1_single_r5 >= r0_single_r5 - 1e-9,
        "gate_10_multi_gold_no_decrease": r1_multi_r5 >= r0_multi_r5 - 1e-9,
    }
    all_gates_pass = all(gates.values())
    verdict = "LOCAL_PROMOTE_AITEAM50_SOFT_ADMISSION_V1" if all_gates_pass else "KILL_AITEAM50_SOFT_ADMISSION_V1"

    report = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.cal_report.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "gold_reveal_time_utc": reveal_time_utc,
        "action_artifact_sha256": action_meta["sha256"],
        "diagnostics": {
            "aiteam_standalone_r5": aiteam_r5,
            "aiteam_standalone_r20": aiteam_r20,
            "aiteam_standalone_r50": aiteam_r50,
            "admission_model_standalone_r5_within_S": admission_standalone_r5,
            "queries_with_novel_candidates": sum(1 for q in all_ids if len(action_records[q]["novel_candidates"]) > 0),
            "total_actions": total_actions,
            "defender_admission_ranks_distribution": {
                "min": int(np.min(defender_adm_ranks)) if defender_adm_ranks else None,
                "median": float(np.median(defender_adm_ranks)) if defender_adm_ranks else None,
                "max": int(np.max(defender_adm_ranks)) if defender_adm_ranks else None,
            },
            "selected_novel_aiteam_ranks_distribution": {
                "min": int(np.min([r for r in selected_novel_aiteam_ranks if r is not None])) if selected_novel_aiteam_ranks else None,
                "median": float(np.median([r for r in selected_novel_aiteam_ranks if r is not None])) if selected_novel_aiteam_ranks else None,
                "max": int(np.max([r for r in selected_novel_aiteam_ranks if r is not None])) if selected_novel_aiteam_ranks else None,
            },
            "selected_novel_admission_ranks_distribution": {
                "min": int(np.min(selected_novel_adm_ranks)) if selected_novel_adm_ranks else None,
                "median": float(np.median(selected_novel_adm_ranks)) if selected_novel_adm_ranks else None,
                "max": int(np.max(selected_novel_adm_ranks)) if selected_novel_adm_ranks else None,
            },
        },
        "cal_utility": {
            "d1_recall_at_5": r0_mean_r5,
            "d1_precision_at_5": r0_mean_p5,
            "r1_recall_at_5": r1_mean_r5,
            "r1_precision_at_5": r1_mean_p5,
            "delta_recall_at_5": r1_mean_r5 - r0_mean_r5,
            "delta_precision_at_5": r1_mean_p5 - r0_mean_p5,
            "single_gold": {
                "r0_recall": r0_single_r5,
                "r1_recall": r1_single_r5,
                "delta": r1_single_r5 - r0_single_r5,
            },
            "multi_gold": {
                "r0_recall": r0_multi_r5,
                "r1_recall": r1_multi_r5,
                "delta": r1_multi_r5 - r0_multi_r5,
            },
            "blocks": block_metrics,
            "paired": {
                "wins": wins,
                "losses": losses,
                "ties": ties,
            },
            "actions": {
                "total": total_actions,
                "beneficial": beneficial,
                "harmful": harmful,
                "neutral": neutral,
            },
            "crossings": {
                "gold_crossings_in": gold_crossings_in,
                "gold_crossings_out": gold_crossings_out,
            },
            "churn": {
                "set_churn": set_churn,
                "ordered_churn": ordered_churn,
            },
            "fired_actions_detail": per_query_audit,
        },
        "outside_pool_rescue_diagnostic": {
            "source_recoverable_gold_cases": 6,
            "admission_top5_count": adm_placed_top5_count,
            "actually_admitted_count": actually_admitted_count,
            "remained_below_top5_count": remained_below_top5_count,
            "cases": outside_pool_reports,
        },
        "promotion_gates": gates,
        "verdict": verdict,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    report_path = RESULTS_DIR / "AITEAM50_SOFT_ADMISSION_CAL_REPORT.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote CAL report: {report_path} (Verdict: {verdict})", flush=True)

    return report, verdict
