"""Stage 5: Evaluate exactly three local arms (A0 48D, A1 50D, A2 50D) via 4-block LOBO and evaluate scientific gates."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tune_expanded_fusion_selection import ltr_features

from .common import (
    D1_VIEWS,
    EXPECTED_BLOCK_RECALLS,
    EXPECTED_D1_R5,
    EXPECTED_FROZEN_SECTION_R5,
    FROZEN_SECTION_CACHE_PATH,
    RESULTS_DIR,
    get_git_status,
    load_cal_data,
    seed_everything,
)
from .score_cal_adapted_section_ce import CAL_ADAPTED_CACHE_PATH


def evaluate_arm(
    arm_name: str,
    channels: Dict[str, Any],
    local_views: Dict[str, Any],
    extended: Dict[str, List[str]],
    type_rows: Dict[str, Any],
    cite_rows: Dict[str, Any],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    gold: Dict[str, Set[str]],
) -> Tuple[
    Dict[str, List[str]],
    int,
    Dict[str, Any],
    Dict[str, np.ndarray],
    Dict[str, List[str]],
    Dict[str, Dict[str, float]],
]:
    preds: Dict[str, List[str]] = {}
    scores_dict: Dict[str, np.ndarray] = {}
    full_rankings: Dict[str, List[str]] = {}
    full_scores: Dict[str, Dict[str, float]] = {}
    feature_dim = 0
    block_metrics = {}

    for held in sorted(blocks.keys()):
        train_ids = sum((blocks[n] for n in blocks if n != held), [])
        test_ids = blocks[held]
        eval_ids = train_ids + test_ids

        eval_rows, eval_groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, channels
        )
        for q in eval_rows:
            eval_rows[q] = np.concatenate(
                [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        feature_dim = eval_rows[all_ids[0]].shape[1]

        X_train = np.vstack([eval_rows[q] for q in train_ids])
        y_train = np.concatenate(
            [[d in gold[q] for d in eval_groups[q]] for q in train_ids]
        ).astype(np.int8)

        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X_train), y_train)

        b_recalls = []
        for q in test_ids:
            X_test = scaler.transform(eval_rows[q])
            dec_scores = model.decision_function(X_test)
            order = sorted(
                range(len(dec_scores)), key=lambda i: dec_scores[i], reverse=True
            )
            ranked_cands = [eval_groups[q][i] for i in order]
            full_rankings[q] = ranked_cands
            full_scores[q] = {
                eval_groups[q][i]: float(dec_scores[i]) for i in range(len(dec_scores))
            }
            top5 = ranked_cands[:5]
            preds[q] = top5
            scores_dict[q] = dec_scores
            b_recalls.append(len(set(top5) & gold[q]) / max(1, len(gold[q])))

        block_metrics[held] = float(np.mean(b_recalls))

    pooled_recalls = [
        len(set(preds[q]) & gold[q]) / max(1, len(gold[q])) for q in all_ids
    ]
    pooled_precisions = [len(set(preds[q]) & gold[q]) / 5.0 for q in all_ids]
    single_gold_recalls = [
        len(set(preds[q]) & gold[q]) / max(1, len(gold[q]))
        for q in all_ids
        if len(gold[q]) == 1
    ]
    multi_gold_recalls = [
        len(set(preds[q]) & gold[q]) / max(1, len(gold[q]))
        for q in all_ids
        if len(gold[q]) > 1
    ]

    metrics = {
        "arm": arm_name,
        "feature_dim": feature_dim,
        "recall_at_5": float(np.mean(pooled_recalls)),
        "precision_at_5": float(np.mean(pooled_precisions)),
        "single_gold_recall_at_5": float(np.mean(single_gold_recalls)),
        "multi_gold_recall_at_5": float(np.mean(multi_gold_recalls)),
        "block_recalls": block_metrics,
    }
    return preds, feature_dim, metrics, scores_dict, full_rankings, full_scores


def compare_arms(
    all_ids: List[str],
    gold: Dict[str, Set[str]],
    preds_base: Dict[str, List[str]],
    preds_comp: Dict[str, List[str]],
) -> Dict[str, Any]:
    wins = 0
    losses = 0
    ties = 0
    set_churn = 0
    ord_churn = 0
    gold_in = 0
    gold_out = 0

    for q in all_ids:
        b_top5 = preds_base[q]
        c_top5 = preds_comp[q]
        g_set = gold[q]

        r_base = len(set(b_top5) & g_set) / max(1, len(g_set))
        r_comp = len(set(c_top5) & g_set) / max(1, len(g_set))

        if r_comp > r_base + 1e-9:
            wins += 1
        elif r_base > r_comp + 1e-9:
            losses += 1
        else:
            ties += 1

        if set(b_top5) != set(c_top5):
            set_churn += 1
        if b_top5 != c_top5:
            ord_churn += 1

        # Gold crossings
        entered_gold = (set(c_top5) - set(b_top5)) & g_set
        left_gold = (set(b_top5) - set(c_top5)) & g_set
        gold_in += len(entered_gold)
        gold_out += len(left_gold)

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "top5_set_churn": set_churn,
        "top5_set_churn_pct": set_churn / len(all_ids) * 100.0,
        "top5_ordered_churn": ord_churn,
        "top5_ordered_churn_pct": ord_churn / len(all_ids) * 100.0,
        "gold_crossings_into_top5": gold_in,
        "gold_crossings_out_of_top5": gold_out,
        "net_gold_crossings": gold_in - gold_out,
    }


def evaluate_local_three_arms() -> Tuple[Dict[str, Any], str]:
    print("=== STAGE 5: THREE-ARM LOBO EVALUATION & LOCAL GATES ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load CAL data and gold
    print("Loading CAL dataset and gold labels...", flush=True)
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, type_rows, cite_rows = load_cal_data()

    # 2. Load Frozen Section CE cache (A1)
    if not FROZEN_SECTION_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing frozen Section cache: {FROZEN_SECTION_CACHE_PATH}")
    cached_frozen = pickle.loads(FROZEN_SECTION_CACHE_PATH.read_bytes())
    frozen_scores = cached_frozen["scores"] if isinstance(cached_frozen, dict) and "scores" in cached_frozen else cached_frozen

    fl_frozen = min(v for q in frozen_scores for v in frozen_scores[q].values())
    aligned_frozen = {
        q: {d: frozen_scores.get(q, {}).get(d, fl_frozen) for d in extended[q]}
        for q in all_ids
    }

    # 3. Load Adapted Section CE cache (A2)
    if not CAL_ADAPTED_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing adapted Section cache: {CAL_ADAPTED_CACHE_PATH}")
    cached_adapted = pickle.loads(CAL_ADAPTED_CACHE_PATH.read_bytes())
    adapted_probs = cached_adapted["scores"]
    adapted_raw_logits = cached_adapted.get("raw_logits", {})

    fl_adapted = min(v for q in adapted_probs for v in adapted_probs[q].values())
    aligned_adapted = {
        q: {d: adapted_probs.get(q, {}).get(d, fl_adapted) for d in extended[q]}
        for q in all_ids
    }

    # 4. Construct channel configurations for the 3 arms
    channels_a0 = full_channels_cv
    channels_a1 = {**full_channels_cv, "legal_section_ce": aligned_frozen}
    channels_a2 = {**full_channels_cv, "legal_section_ce": aligned_adapted}

    # 5. Evaluate A0
    print("
--- Evaluating Arm A0: D1 Baseline (48D) ---", flush=True)
    preds_a0, dim_a0, metrics_a0, scores_a0, ranks_a0, full_sc_a0 = evaluate_arm(
        "A0_D1_BASELINE", channels_a0, local_views, extended, type_rows, cite_rows, blocks, all_ids, gold
    )
    print(f"A0 Recall@5: {metrics_a0['recall_at_5']:.6f} (Dim: {dim_a0})")

    # 6. Evaluate A1
    print("
--- Evaluating Arm A1: D1 + Frozen Section CE Control (50D) ---", flush=True)
    preds_a1, dim_a1, metrics_a1, scores_a1, ranks_a1, full_sc_a1 = evaluate_arm(
        "A1_D1_PLUS_FROZEN_SECTION_CE", channels_a1, local_views, extended, type_rows, cite_rows, blocks, all_ids, gold
    )
    print(f"A1 Recall@5: {metrics_a1['recall_at_5']:.6f} (Dim: {dim_a1})")

    # 7. Evaluate A2
    print("
--- Evaluating Arm A2: D1 + Adapted Section CE Probability (50D) ---", flush=True)
    preds_a2, dim_a2, metrics_a2, scores_a2, ranks_a2, full_sc_a2 = evaluate_arm(
        "A2_D1_PLUS_ADAPTED_SECTION_CE", channels_a2, local_views, extended, type_rows, cite_rows, blocks, all_ids, gold
    )
    print(f"A2 Recall@5: {metrics_a2['recall_at_5']:.6f} (Dim: {dim_a2})")

    # 8. Comparisons
    comp_a2_vs_a1 = compare_arms(all_ids, gold, preds_a1, preds_a2)
    comp_a2_vs_a0 = compare_arms(all_ids, gold, preds_a0, preds_a2)
    comp_a1_vs_a0 = compare_arms(all_ids, gold, preds_a0, preds_a1)

    delta_r5_a2_a1 = metrics_a2["recall_at_5"] - metrics_a1["recall_at_5"]
    delta_r5_a2_a0 = metrics_a2["recall_at_5"] - metrics_a0["recall_at_5"]
    delta_single_gold = metrics_a2["single_gold_recall_at_5"] - metrics_a1["single_gold_recall_at_5"]
    delta_multi_gold = metrics_a2["multi_gold_recall_at_5"] - metrics_a1["multi_gold_recall_at_5"]

    block_deltas = {
        b: metrics_a2["block_recalls"][b] - metrics_a1["block_recalls"][b]
        for b in ["A", "B", "C", "D"]
    }

    # 9. Evaluate Decision Gates
    # Check baseline parities first
    a0_parity = abs(metrics_a0["recall_at_5"] - EXPECTED_D1_R5) < 1e-12
    a1_parity = abs(metrics_a1["recall_at_5"] - EXPECTED_FROZEN_SECTION_R5) < 1e-4

    if not (a0_parity and a1_parity):
        verdict = "BLOCKED_BASELINE_PARITY"
        rationale = f"Baseline parity failed: A0 diff={abs(metrics_a0['recall_at_5'] - EXPECTED_D1_R5):.2e}, A1 diff={abs(metrics_a1['recall_at_5'] - EXPECTED_FROZEN_SECTION_R5):.2e}"
    elif (
        metrics_a2["recall_at_5"] <= metrics_a1["recall_at_5"]
        or comp_a2_vs_a1["wins"] <= comp_a2_vs_a1["losses"]
        or block_deltas["D"] < -1e-9
        or any(d < -0.001 - 1e-9 for d in block_deltas.values())
    ):
        verdict = "KILL_ADAPTED_SECTION_CHANNEL"
        rationale = (
            f"A2 did not beat A1 or violated safety gates: delta_r5={delta_r5_a2_a1:+.6f}, "
            f"wins={comp_a2_vs_a1['wins']}, losses={comp_a2_vs_a1['losses']}, "
            f"block_D_delta={block_deltas['D']:+.6f}, min_block_delta={min(block_deltas.values()):+.6f}"
        )
    elif delta_r5_a2_a1 < 0.0005:
        verdict = "KEEP_LOCAL_SIGNAL_NO_PUBLIC"
        rationale = f"A2 beat A1 but gain {delta_r5_a2_a1:+.6f} < +0.0005 threshold for public candidate."
    else:
        # Check single and multi gold safety
        safety_passed = (delta_single_gold >= -0.001 - 1e-9) and (delta_multi_gold >= -0.003 - 1e-9)
        if not safety_passed:
            verdict = "KILL_ADAPTED_SECTION_CHANNEL"
            rationale = f"Safety gate failed: single_gold_delta={delta_single_gold:+.6f}, multi_gold_delta={delta_multi_gold:+.6f}"
        elif metrics_a2["recall_at_5"] >= 0.960000:
            verdict = "STRONG_PUBLIC_CANDIDATE_ADAPTED_SECTION_CHANNEL"
            rationale = f"A2 >= 0.960000 ({metrics_a2['recall_at_5']:.6f}) and passed all safety gates."
        else:
            verdict = "PUBLIC_CANDIDATE_ADAPTED_SECTION_CHANNEL"
            rationale = f"A2 >= A1 + 0.0005 (gain: {delta_r5_a2_a1:+.6f}) and passed all safety gates."

    print(f"
=======================================================", flush=True)
    print(f"LOCAL SCIENTIFIC VERDICT: {verdict}", flush=True)
    print(f"RATIONALE: {rationale}", flush=True)
    print(f"=======================================================
", flush=True)

    # 10. Post-hoc Rank Diagnostics for Changed Recall Queries
    diagnostics: List[Dict[str, Any]] = []
    for q in all_ids:
        r_a1 = len(set(preds_a1[q]) & gold[q]) / max(1, len(gold[q]))
        r_a2 = len(set(preds_a2[q]) & gold[q]) / max(1, len(gold[q]))
        if abs(r_a2 - r_a1) > 1e-9:
            # Save boundary cands (ranks 3 to 8 in A2)
            cands_a2 = ranks_a2[q]
            boundary_cands = cands_a2[2:8] if len(cands_a2) >= 8 else cands_a2
            cand_details = []
            for did in boundary_cands:
                cand_details.append({
                    "did": did,
                    "is_gold": did in gold[q],
                    "rank_in_a1": ranks_a1[q].index(did) + 1 if did in ranks_a1[q] else -1,
                    "rank_in_a2": ranks_a2[q].index(did) + 1,
                    "d1_decision_score": full_sc_a0[q].get(did, 0.0),
                    "frozen_section_score": aligned_frozen[q].get(did, 0.0),
                    "adapted_raw_logit": adapted_raw_logits.get(q, {}).get(did, 0.0),
                    "adapted_prob": aligned_adapted[q].get(did, 0.0),
                })
            diagnostics.append({
                "qid": q,
                "gold_docs": list(gold[q]),
                "recall_a1": r_a1,
                "recall_a2": r_a2,
                "delta_recall": r_a2 - r_a1,
                "top5_a1": preds_a1[q],
                "top5_a2": preds_a2[q],
                "boundary_candidates": cand_details,
            })

    # Save reports
    report_data = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.local_three_arms.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "verdict": verdict,
        "rationale": rationale,
        "a0_d1_baseline": metrics_a0,
        "a1_frozen_section_control": metrics_a1,
        "a2_adapted_section_channel": metrics_a2,
        "deltas_a2_vs_a1": {
            "recall_at_5": delta_r5_a2_a1,
            "single_gold_recall_at_5": delta_single_gold,
            "multi_gold_recall_at_5": delta_multi_gold,
            "block_deltas": block_deltas,
        },
        "deltas_a2_vs_a0": {
            "recall_at_5": delta_r5_a2_a0,
        },
        "paired_comparison_a2_vs_a1": comp_a2_vs_a1,
        "paired_comparison_a2_vs_a0": comp_a2_vs_a0,
        "paired_comparison_a1_vs_a0": comp_a1_vs_a0,
        "total_changed_recall_queries": len(diagnostics),
    }

    report_path = RESULTS_DIR / "LOCAL_THREE_ARMS_REPORT.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)

    diag_path = RESULTS_DIR / "A2_VS_A1_RANK_DIAGNOSTICS.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Wrote {diag_path} ({len(diagnostics)} queries)", flush=True)

    decision_md = f"""# Local Scientific Decision: HUY_D1_ADAPTED_SECTION_CHANNEL_PUBLIC_V1

## 1. Scientific Verdict
- **Verdict**: `{verdict}`
- **Rationale**: {rationale}

---

## 2. Three-Arm Quantitative Summary (4-Block LOBO)

| Metric | A0 (D1 48D Baseline) | A1 (D1 + Frozen Section 50D) | A2 (D1 + Adapted Section 50D) | Delta (A2 - A1) | Delta (A2 - A0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pooled Recall@5** | **{metrics_a0['recall_at_5']:.6f}** | **{metrics_a1['recall_at_5']:.6f}** | **{metrics_a2['recall_at_5']:.6f}** | **{delta_r5_a2_a1:+.6f}** | **{delta_r5_a2_a0:+.6f}** |
| Precision@5 | {metrics_a0['precision_at_5']:.4f} | {metrics_a1['precision_at_5']:.4f} | {metrics_a2['precision_at_5']:.4f} | {metrics_a2['precision_at_5'] - metrics_a1['precision_at_5']:+.4f} | {metrics_a2['precision_at_5'] - metrics_a0['precision_at_5']:+.4f} |
| Single-Gold R@5 | {metrics_a0['single_gold_recall_at_5']:.6f} | {metrics_a1['single_gold_recall_at_5']:.6f} | {metrics_a2['single_gold_recall_at_5']:.6f} | {delta_single_gold:+.6f} | {metrics_a2['single_gold_recall_at_5'] - metrics_a0['single_gold_recall_at_5']:+.6f} |
| Multi-Gold R@5 | {metrics_a0['multi_gold_recall_at_5']:.6f} | {metrics_a1['multi_gold_recall_at_5']:.6f} | {metrics_a2['multi_gold_recall_at_5']:.6f} | {delta_multi_gold:+.6f} | {metrics_a2['multi_gold_recall_at_5'] - metrics_a0['multi_gold_recall_at_5']:+.6f} |
| Block A R@5 | {metrics_a0['block_recalls']['A']:.6f} | {metrics_a1['block_recalls']['A']:.6f} | {metrics_a2['block_recalls']['A']:.6f} | {block_deltas['A']:+.6f} | {metrics_a2['block_recalls']['A'] - metrics_a0['block_recalls']['A']:+.6f} |
| Block B R@5 | {metrics_a0['block_recalls']['B']:.6f} | {metrics_a1['block_recalls']['B']:.6f} | {metrics_a2['block_recalls']['B']:.6f} | {block_deltas['B']:+.6f} | {metrics_a2['block_recalls']['B'] - metrics_a0['block_recalls']['B']:+.6f} |
| Block C R@5 | {metrics_a0['block_recalls']['C']:.6f} | {metrics_a1['block_recalls']['C']:.6f} | {metrics_a2['block_recalls']['C']:.6f} | {block_deltas['C']:+.6f} | {metrics_a2['block_recalls']['C'] - metrics_a0['block_recalls']['C']:+.6f} |
| Block D R@5 | {metrics_a0['block_recalls']['D']:.6f} | {metrics_a1['block_recalls']['D']:.6f} | {metrics_a2['block_recalls']['D']:.6f} | {block_deltas['D']:+.6f} | {metrics_a2['block_recalls']['D'] - metrics_a0['block_recalls']['D']:+.6f} |

---

## 3. Query-Level Paired Comparisons

### A2 vs A1 (Adapted vs Frozen Control):
- **Wins**: `{comp_a2_vs_a1['wins']}` | **Losses**: `{comp_a2_vs_a1['losses']}` | **Ties**: `{comp_a2_vs_a1['ties']}` (Net: `{comp_a2_vs_a1['net_wins']}`)
- **Top-5 Set Churn**: `{comp_a2_vs_a1['top5_set_churn']}` / 600 ({comp_a2_vs_a1['top5_set_churn_pct']:.2f}%)
- **Top-5 Ordered Churn**: `{comp_a2_vs_a1['top5_ordered_churn']}` / 600 ({comp_a2_vs_a1['top5_ordered_churn_pct']:.2f}%)
- **Gold Crossings**: In=`{comp_a2_vs_a1['gold_crossings_into_top5']}` | Out=`{comp_a2_vs_a1['gold_crossings_out_of_top5']}` | Net=`{comp_a2_vs_a1['net_gold_crossings']}`

### A2 vs A0 (Adapted vs D1 Baseline):
- **Wins**: `{comp_a2_vs_a0['wins']}` | **Losses**: `{comp_a2_vs_a0['losses']}` | **Ties**: `{comp_a2_vs_a0['ties']}` (Net: `{comp_a2_vs_a0['net_wins']}`)
- **Top-5 Set Churn**: `{comp_a2_vs_a0['top5_set_churn']}` / 600 ({comp_a2_vs_a0['top5_set_churn_pct']:.2f}%)
- **Top-5 Ordered Churn**: `{comp_a2_vs_a0['top5_ordered_churn']}` / 600 ({comp_a2_vs_a0['top5_ordered_churn_pct']:.2f}%)
- **Gold Crossings**: In=`{comp_a2_vs_a0['gold_crossings_into_top5']}` | Out=`{comp_a2_vs_a0['gold_crossings_out_of_top5']}` | Net=`{comp_a2_vs_a0['net_gold_crossings']}`
"""
    decision_path = RESULTS_DIR / "DECISION_LOCAL.md"
    decision_path.write_text(decision_md, encoding="utf-8")
    print(f"Wrote {decision_path}", flush=True)

    return report_data, verdict


if __name__ == "__main__":
    evaluate_local_three_arms()
