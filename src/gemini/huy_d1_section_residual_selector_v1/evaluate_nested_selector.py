"""Nested cross-fitting, pairwise residual selector training, inference, and evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_expanded_fusion_selection import ltr_features
from src.gemini.huy_d1_section_residual_selector_v1.common import (
    D1_VIEWS,
    FEATURE_NAMES_13,
    OLD_JINA_CACHE_PKL,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    compute_13_pointwise_features,
    load_cal_data,
    load_pkl,
    seed_everything,
)


def evaluate_nested_residual_selector() -> Tuple[
    dict,
    dict,
    dict,
    dict,
    list,
    dict,
    dict,
    dict,
    dict,
]:
    seed_everything(2026)
    print("=== EVALUATING HUY_D1_SECTION_RESIDUAL_SELECTOR_V1 ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    ) = load_cal_data()

    s_sec = load_pkl(SECTION_CE_CACHE_PKL)
    s_old = load_pkl(OLD_JINA_CACHE_PKL)

    qid_to_block = {}
    for b, q_list in blocks.items():
        for q in q_list:
            qid_to_block[q] = b

    print("Precomputing 48D D1 features for all queries...", flush=True)
    eval_rows, eval_groups = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, full_channels_cv
    )
    for q in eval_rows:
        eval_rows[q] = np.concatenate(
            [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
        )

    r0_preds: Dict[str, List[str]] = {}
    r1_preds: Dict[str, List[str]] = {}
    r0_scores: Dict[str, Dict[str, float]] = {}

    stacking_integrity_records = []
    selector_training_records = {}
    fold_coefficients: Dict[str, List[float]] = {}
    fold_train_stds: Dict[str, List[float]] = {}
    swaps_diagnostic = []

    print("Running 4-block nested cross-fitting LOBO...", flush=True)
    for outer_held in sorted(blocks.keys()):
        outer_train_blocks = [b for b in sorted(blocks.keys()) if b != outer_held]
        print(f"\n--- OUTER HELD BLOCK: {outer_held} (Train Blocks: {outer_train_blocks}) ---", flush=True)

        # 1. Inner LOBO on outer_train_blocks to produce inner-OOF D1 scores
        oof_d1_scores: Dict[str, Dict[str, float]] = {}
        for inner_held in outer_train_blocks:
            inner_train_blocks = [b for b in outer_train_blocks if b != inner_held]
            inner_train_ids = sum([blocks[b] for b in inner_train_blocks], [])
            inner_test_ids = blocks[inner_held]

            # Verify zero label overlap
            overlap = set(inner_train_ids) & set(inner_test_ids)
            if overlap:
                print(f"FATAL: Label leakage in stacking! Overlap: {overlap}", flush=True)
                sys.exit("BLOCKED_STACKING_LEAKAGE")

            stacking_integrity_records.append({
                "outer_held_block": outer_held,
                "inner_held_block": inner_held,
                "d1_train_blocks": inner_train_blocks,
                "d1_train_query_count": len(inner_train_ids),
                "inner_held_query_count": len(inner_test_ids),
                "zero_overlap_verified": True,
            })

            X_tr = np.vstack([eval_rows[q] for q in inner_train_ids])
            y_tr = np.concatenate(
                [[d in gold[q] for d in eval_groups[q]] for q in inner_train_ids]
            ).astype(np.int8)

            scaler_d1 = StandardScaler().fit(X_tr)
            m_d1 = LogisticRegression(
                C=0.15,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=2026,
            )
            m_d1.fit(scaler_d1.transform(X_tr), y_tr)

            for q in inner_test_ids:
                scores = m_d1.decision_function(scaler_d1.transform(eval_rows[q]))
                oof_d1_scores[q] = {
                    eval_groups[q][i]: float(scores[i]) for i in range(len(scores))
                }

        # Verify all outer training queries have inner-OOF D1 scores
        outer_train_ids = sum([blocks[b] for b in outer_train_blocks], [])
        if set(oof_d1_scores.keys()) != set(outer_train_ids):
            print("FATAL: Mismatch in inner-OOF D1 queries!", flush=True)
            sys.exit("BLOCKED_STACKING_LEAKAGE")

        # 2. Build pairwise training data for residual selector
        X_pairs = []
        y_pairs = []
        w_pairs = []
        usable_queries = 0
        skipped_queries = 0
        raw_pairs_count = 0

        for q in outer_train_ids:
            feats_q, oof_d1_t5, sec_t5 = compute_13_pointwise_features(
                q, extended[q], oof_d1_scores[q], s_old[q], s_sec[q]
            )
            slate = list(dict.fromkeys(oof_d1_t5 + sec_t5))
            golds_in_slate = [d for d in slate if d in gold[q]]
            nongolds_in_slate = [d for d in slate if d not in gold[q]]

            if not golds_in_slate or not nongolds_in_slate:
                skipped_queries += 1
                continue

            usable_queries += 1
            n_pairs = len(golds_in_slate) * len(nongolds_in_slate)
            raw_pairs_count += n_pairs
            pair_wt = 1.0 / (2.0 * n_pairs)

            for p in golds_in_slate:
                for n in nongolds_in_slate:
                    delta = feats_q[p] - feats_q[n]
                    X_pairs.append(delta)
                    y_pairs.append(1)
                    w_pairs.append(pair_wt)

                    X_pairs.append(-delta)
                    y_pairs.append(0)
                    w_pairs.append(pair_wt)

        X_pairs = np.array(X_pairs, dtype=np.float64)
        y_pairs = np.array(y_pairs, dtype=np.int8)
        w_pairs = np.array(w_pairs, dtype=np.float64)

        # Scale pair-difference dimensions by training standard deviation (no mean centering)
        train_std = np.std(X_pairs, axis=0)
        train_std = np.where(train_std < 1e-9, 1.0, train_std)
        X_pairs_scaled = X_pairs / train_std

        selector_model = LogisticRegression(
            C=0.15,
            solver="liblinear",
            class_weight=None,
            fit_intercept=False,
            max_iter=3000,
            random_state=2026,
        )
        selector_model.fit(X_pairs_scaled, y_pairs, sample_weight=w_pairs)

        coefs = [float(c) for c in selector_model.coef_[0]]
        fold_coefficients[outer_held] = coefs
        fold_train_stds[outer_held] = [float(s) for s in train_std]

        selector_training_records[outer_held] = {
            "outer_held_block": outer_held,
            "outer_train_blocks": outer_train_blocks,
            "total_outer_train_queries": len(outer_train_ids),
            "usable_queries": usable_queries,
            "skipped_queries": skipped_queries,
            "raw_pairs_count": raw_pairs_count,
            "augmented_pairs_count": len(X_pairs),
            "total_sample_weight": float(np.sum(w_pairs)),
            "coefficients": dict(zip(FEATURE_NAMES_13, coefs)),
        }

        print(
            f"Selector Training ({outer_held}): usable={usable_queries}, skipped={skipped_queries}, "
            f"pairs={raw_pairs_count} (augmented={len(X_pairs)}), total_wt={np.sum(w_pairs):.1f}",
            flush=True,
        )

        # 3. Outer D1 fit on all outer_train_blocks to score outer_test_ids
        outer_test_ids = blocks[outer_held]
        X_outer_tr = np.vstack([eval_rows[q] for q in outer_train_ids])
        y_outer_tr = np.concatenate(
            [[d in gold[q] for d in eval_groups[q]] for q in outer_train_ids]
        ).astype(np.int8)

        scaler_outer = StandardScaler().fit(X_outer_tr)
        m_outer = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        m_outer.fit(scaler_outer.transform(X_outer_tr), y_outer_tr)

        # 4. Outer Inference: selective one-swap correction
        for q in outer_test_ids:
            scores = m_outer.decision_function(scaler_outer.transform(eval_rows[q]))
            d1_scores_q = {
                eval_groups[q][i]: float(scores[i]) for i in range(len(scores))
            }
            r0_scores[q] = d1_scores_q

            feats_q, d1_t5, sec_t5 = compute_13_pointwise_features(
                q, extended[q], d1_scores_q, s_old[q], s_sec[q]
            )
            r0_preds[q] = list(d1_t5)

            challengers = [c for c in sec_t5 if c not in d1_t5]
            defenders = list(d1_t5)

            best_pair = None
            best_prob = -1.0
            best_delta = None

            if challengers:
                for c in challengers:
                    for d in defenders:
                        delta = feats_q[c] - feats_q[d]
                        delta_scaled = (delta / train_std).reshape(1, -1)
                        prob = float(selector_model.predict_proba(delta_scaled)[0, 1])
                        if prob > best_prob:
                            best_prob = prob
                            best_pair = (c, d)
                            best_delta = delta

            if best_prob > 0.5 and best_pair is not None:
                c_star, d_star = best_pair
                new_t5 = list(d1_t5)
                slot_idx = new_t5.index(d_star)
                new_t5[slot_idx] = c_star
                r1_preds[q] = new_t5

                c_is_gold = c_star in gold[q]
                d_is_gold = d_star in gold[q]
                if c_is_gold and not d_is_gold:
                    effect = "BENEFICIAL"
                elif not c_is_gold and d_is_gold:
                    effect = "HARMFUL"
                else:
                    effect = "NEUTRAL"

                swaps_diagnostic.append({
                    "qid": q,
                    "block": outer_held,
                    "question": queries[q][0],
                    "challenger_doc": c_star,
                    "defender_doc": d_star,
                    "challenger_is_gold": c_is_gold,
                    "defender_is_gold": d_is_gold,
                    "swap_effect": effect,
                    "predicted_probability": best_prob,
                    "d1_scores": {"challenger": d1_scores_q[c_star], "defender": d1_scores_q[d_star]},
                    "old_jina_scores": {"challenger": s_old[q].get(c_star, -999.0), "defender": s_old[q].get(d_star, -999.0)},
                    "section_scores": {"challenger": s_sec[q].get(c_star, -999.0), "defender": s_sec[q].get(d_star, -999.0)},
                    "challenger_features": dict(zip(FEATURE_NAMES_13, [float(x) for x in feats_q[c_star]])),
                    "defender_features": dict(zip(FEATURE_NAMES_13, [float(x) for x in feats_q[d_star]])),
                    "delta_vector": dict(zip(FEATURE_NAMES_13, [float(x) for x in best_delta])),
                })
            else:
                r1_preds[q] = list(d1_t5)

    # 5. Global Metrics & Comparisons
    r0_recalls = [len(set(r0_preds[q]) & gold[q]) / max(1, len(gold[q])) for q in all_ids]
    r1_recalls = [len(set(r1_preds[q]) & gold[q]) / max(1, len(gold[q])) for q in all_ids]

    r0_precisions = [len(set(r0_preds[q]) & gold[q]) / 5.0 for q in all_ids]
    r1_precisions = [len(set(r1_preds[q]) & gold[q]) / 5.0 for q in all_ids]

    single_ids = [q for q in all_ids if len(gold[q]) == 1]
    multi_ids = [q for q in all_ids if len(gold[q]) > 1]

    r0_single_r5 = float(np.mean([r0_recalls[all_ids.index(q)] for q in single_ids]))
    r1_single_r5 = float(np.mean([r1_recalls[all_ids.index(q)] for q in single_ids]))

    r0_multi_r5 = float(np.mean([r0_recalls[all_ids.index(q)] for q in multi_ids]))
    r1_multi_r5 = float(np.mean([r1_recalls[all_ids.index(q)] for q in multi_ids]))

    block_metrics = {}
    block_deltas = {}
    for b in sorted(blocks.keys()):
        b_ids = blocks[b]
        b_r0 = float(np.mean([r0_recalls[all_ids.index(q)] for q in b_ids]))
        b_r1 = float(np.mean([r1_recalls[all_ids.index(q)] for q in b_ids]))
        block_metrics[b] = {"r0_recall_at_5": b_r0, "r1_recall_at_5": b_r1, "delta": b_r1 - b_r0}
        block_deltas[b] = b_r1 - b_r0

    r0_pooled_r5 = float(np.mean(r0_recalls))
    r1_pooled_r5 = float(np.mean(r1_recalls))
    delta_r5 = r1_pooled_r5 - r0_pooled_r5

    # Paired query breakdown
    wins = sum(1 for q in all_ids if r1_recalls[all_ids.index(q)] > r0_recalls[all_ids.index(q)])
    losses = sum(1 for q in all_ids if r1_recalls[all_ids.index(q)] < r0_recalls[all_ids.index(q)])
    ties = sum(1 for q in all_ids if r1_recalls[all_ids.index(q)] == r0_recalls[all_ids.index(q)])

    gold_crossings_in = sum(1 for s in swaps_diagnostic if s["challenger_is_gold"] and not s["defender_is_gold"])
    gold_crossings_out = sum(1 for s in swaps_diagnostic if not s["challenger_is_gold"] and s["defender_is_gold"])

    beneficial_swaps = sum(1 for s in swaps_diagnostic if s["swap_effect"] == "BENEFICIAL")
    harmful_swaps = sum(1 for s in swaps_diagnostic if s["swap_effect"] == "HARMFUL")
    neutral_swaps = sum(1 for s in swaps_diagnostic if s["swap_effect"] == "NEUTRAL")

    swaps_by_block = {}
    for b in sorted(blocks.keys()):
        b_swaps = [s for s in swaps_diagnostic if s["block"] == b]
        swaps_by_block[b] = {
            "total_swaps": len(b_swaps),
            "beneficial": sum(1 for s in b_swaps if s["swap_effect"] == "BENEFICIAL"),
            "harmful": sum(1 for s in b_swaps if s["swap_effect"] == "HARMFUL"),
            "neutral": sum(1 for s in b_swaps if s["swap_effect"] == "NEUTRAL"),
        }

    # 6. Coefficient Stability & Generalization Audit
    weight_matrix = np.array([fold_coefficients[b] for b in sorted(blocks.keys())])
    cosine_sim_matrix = {}
    block_list = sorted(blocks.keys())
    cos_sims = []
    for i in range(len(block_list)):
        for j in range(i + 1, len(block_list)):
            b1, b2 = block_list[i], block_list[j]
            v1, v2 = weight_matrix[i], weight_matrix[j]
            cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            cosine_sim_matrix[f"{b1}_vs_{b2}"] = cos_sim
            cos_sims.append(cos_sim)

    feature_stability = {}
    for idx, fn in enumerate(FEATURE_NAMES_13):
        vals = weight_matrix[:, idx]
        sign_consistent = (np.all(vals > 0) or np.all(vals < 0))
        feature_stability[fn] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "sign_consistent": bool(sign_consistent),
            "fold_values": {b: float(weight_matrix[b_idx, idx]) for b_idx, b in enumerate(block_list)},
        }

    coefficient_stability_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_section_residual_selector_v1.stability.v1",
        "experiment_id": "HUY_D1_SECTION_RESIDUAL_SELECTOR_V1",
        "feature_names": FEATURE_NAMES_13,
        "fold_coefficients": fold_coefficients,
        "pairwise_cosine_similarity": cosine_sim_matrix,
        "mean_cosine_similarity": float(np.mean(cos_sims)),
        "min_cosine_similarity": float(np.min(cos_sims)),
        "feature_stability": feature_stability,
    }

    # 7. 10,000 Paired Bootstrap Resamples
    print("\nRunning 10,000 paired bootstrap resamples...", flush=True)
    n_boot = 10000
    rng = np.random.default_rng(2026)
    delta_array = np.array(r1_recalls) - np.array(r0_recalls)

    # Ordinary bootstrap
    boot_deltas_ord = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(all_ids), size=len(all_ids))
        boot_deltas_ord.append(float(np.mean(delta_array[sample_idx])))
    boot_deltas_ord = np.array(boot_deltas_ord)

    # Block-stratified bootstrap
    block_indices = {b: [all_ids.index(q) for q in blocks[b]] for b in blocks}
    boot_deltas_strat = []
    for _ in range(n_boot):
        strat_samples = []
        for b, idxs in block_indices.items():
            b_sample = rng.choice(idxs, size=len(idxs), replace=True)
            strat_samples.extend(b_sample)
        boot_deltas_strat.append(float(np.mean(delta_array[strat_samples])))
    boot_deltas_strat = np.array(boot_deltas_strat)

    bootstrap_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_section_residual_selector_v1.bootstrap.v1",
        "experiment_id": "HUY_D1_SECTION_RESIDUAL_SELECTOR_V1",
        "n_samples": n_boot,
        "seed": 2026,
        "ordinary_bootstrap": {
            "mean_delta": float(np.mean(boot_deltas_ord)),
            "median_delta": float(np.median(boot_deltas_ord)),
            "ci_2_5": float(np.percentile(boot_deltas_ord, 2.5)),
            "ci_97_5": float(np.percentile(boot_deltas_ord, 97.5)),
            "p_delta_gt_zero": float(np.mean(boot_deltas_ord > 0)),
            "p_delta_ge_zero": float(np.mean(boot_deltas_ord >= 0)),
        },
        "block_stratified_bootstrap": {
            "mean_delta": float(np.mean(boot_deltas_strat)),
            "median_delta": float(np.median(boot_deltas_strat)),
            "ci_2_5": float(np.percentile(boot_deltas_strat, 2.5)),
            "ci_97_5": float(np.percentile(boot_deltas_strat, 97.5)),
            "p_delta_gt_zero": float(np.mean(boot_deltas_strat > 0)),
            "p_delta_ge_zero": float(np.mean(boot_deltas_strat >= 0)),
        },
    }

    # 8. Promotion Decision Logic
    block_d_regressed = block_deltas["D"] < -1e-9
    any_block_excess_regress = any(block_deltas[b] < -0.001 - 1e-9 for b in blocks)
    single_gold_regress = (r1_single_r5 - r0_single_r5) < -0.001 - 1e-9
    multi_gold_regress = (r1_multi_r5 - r0_multi_r5) < -0.003 - 1e-9
    safety_gates_pass = (
        (wins > losses)
        and (not block_d_regressed)
        and (not any_block_excess_regress)
        and (not single_gold_regress)
        and (not multi_gold_regress)
    )

    if r1_pooled_r5 >= 0.965000 and safety_gates_pass:
        verdict = "BREAK_0965_CAL_SECTION_SELECTOR"
    elif r1_pooled_r5 >= 0.960000 and safety_gates_pass:
        verdict = "PROMOTE_SECTION_SELECTOR"
    elif r1_pooled_r5 > r0_pooled_r5 and safety_gates_pass:
        verdict = "KEEP_SECTION_SELECTOR_SIGNAL"
    else:
        verdict = "KILL_SECTION_SELECTOR"

    eval_summary = {
        "verdict": verdict,
        "safety_gates": {
            "wins_gt_losses": wins > losses,
            "block_d_not_regressed": not block_d_regressed,
            "no_block_regress_gt_0001": not any_block_excess_regress,
            "single_gold_delta_ge_minus_0001": not single_gold_regress,
            "multi_gold_delta_ge_minus_0003": not multi_gold_regress,
            "all_safety_gates_pass": safety_gates_pass,
        },
        "r0_baseline": {
            "arm": "R0_D1_BASELINE",
            "feature_dim": 48,
            "recall_at_5": r0_pooled_r5,
            "precision_at_5": float(np.mean(r0_precisions)),
            "single_gold_recall_at_5": r0_single_r5,
            "multi_gold_recall_at_5": r0_multi_r5,
            "block_recalls": {b: block_metrics[b]["r0_recall_at_5"] for b in blocks},
        },
        "r1_section_selector": {
            "arm": "R1_SECTION_RESIDUAL_SELECTOR",
            "feature_dim": 48,
            "recall_at_5": r1_pooled_r5,
            "precision_at_5": float(np.mean(r1_precisions)),
            "single_gold_recall_at_5": r1_single_r5,
            "multi_gold_recall_at_5": r1_multi_r5,
            "block_recalls": {b: block_metrics[b]["r1_recall_at_5"] for b in blocks},
        },
        "deltas": {
            "recall_at_5": delta_r5,
            "precision_at_5": float(np.mean(r1_precisions)) - float(np.mean(r0_precisions)),
            "single_gold_recall_at_5": r1_single_r5 - r0_single_r5,
            "multi_gold_recall_at_5": r1_multi_r5 - r0_multi_r5,
            "block_deltas": block_deltas,
        },
        "paired_comparison": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_wins": wins - losses,
            "total_swaps": len(swaps_diagnostic),
            "beneficial_swaps": beneficial_swaps,
            "harmful_swaps": harmful_swaps,
            "neutral_swaps": neutral_swaps,
            "unchanged_queries": len(all_ids) - len(swaps_diagnostic),
            "gold_crossings_into_top5": gold_crossings_in,
            "gold_crossings_out_of_top5": gold_crossings_out,
            "net_gold_crossings": gold_crossings_in - gold_crossings_out,
            "swaps_by_block": swaps_by_block,
        },
    }

    print("\n========================= METRIC SUMMARY =========================", flush=True)
    print("Metric                 | R0 (Baseline 48D) | R1 (Selector 48D) | Delta", flush=True)
    print("-----------------------+-------------------+-------------------+----------", flush=True)
    print(f"Pooled Recall@5        | {r0_pooled_r5:.16f} | {r1_pooled_r5:.16f} | {delta_r5:+.16f}", flush=True)
    print(f"Block A Recall@5       | {block_metrics['A']['r0_recall_at_5']:.6f}          | {block_metrics['A']['r1_recall_at_5']:.6f}          | {block_deltas['A']:+.6f}", flush=True)
    print(f"Block B Recall@5       | {block_metrics['B']['r0_recall_at_5']:.6f}          | {block_metrics['B']['r1_recall_at_5']:.6f}          | {block_deltas['B']:+.6f}", flush=True)
    print(f"Block C Recall@5       | {block_metrics['C']['r0_recall_at_5']:.6f}          | {block_metrics['C']['r1_recall_at_5']:.6f}          | {block_deltas['C']:+.6f}", flush=True)
    print(f"Block D Recall@5       | {block_metrics['D']['r0_recall_at_5']:.6f}          | {block_metrics['D']['r1_recall_at_5']:.6f}          | {block_deltas['D']:+.6f}", flush=True)
    print(f"Single-gold Recall@5   | {r0_single_r5:.6f}          | {r1_single_r5:.6f}          | {r1_single_r5 - r0_single_r5:+.6f}", flush=True)
    print(f"Multi-gold Recall@5    | {r0_multi_r5:.6f}          | {r1_multi_r5:.6f}          | {r1_multi_r5 - r0_multi_r5:+.6f}", flush=True)
    print(f"Wins / Losses / Ties   | -                 | -                 | {wins} / {losses} / {ties} (Net: {wins - losses:+d})", flush=True)
    print(f"Total Swaps            | -                 | -                 | {len(swaps_diagnostic)} (Ben: {beneficial_swaps}, Harm: {harmful_swaps}, Neut: {neutral_swaps})", flush=True)
    print(f"Gold Crossings Top5    | -                 | -                 | In: {gold_crossings_in} / Out: {gold_crossings_out} (Net: {gold_crossings_in - gold_crossings_out:+d})", flush=True)
    print(f"Bootstrap 95% CI (Ord) | -                 | -                 | [{bootstrap_doc['ordinary_bootstrap']['ci_2_5']:+.6f}, {bootstrap_doc['ordinary_bootstrap']['ci_97_5']:+.6f}] (P>0: {bootstrap_doc['ordinary_bootstrap']['p_delta_gt_zero']:.3f})", flush=True)
    print(f"Bootstrap 95% CI (Blk) | -                 | -                 | [{bootstrap_doc['block_stratified_bootstrap']['ci_2_5']:+.6f}, {bootstrap_doc['block_stratified_bootstrap']['ci_97_5']:+.6f}] (P>0: {bootstrap_doc['block_stratified_bootstrap']['p_delta_gt_zero']:.3f})", flush=True)
    print(f"VERDICT                | -                 | -                 | {verdict}", flush=True)
    print("==================================================================\n", flush=True)

    stacking_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_section_residual_selector_v1.stacking_integrity.v1",
        "experiment_id": "HUY_D1_SECTION_RESIDUAL_SELECTOR_V1",
        "status": "PASS",
        "outer_held_folds_count": len(blocks),
        "total_inner_oof_runs": len(stacking_integrity_records),
        "zero_overlap_all_folds": True,
        "folds": stacking_integrity_records,
    }

    selector_train_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_section_residual_selector_v1.training.v1",
        "experiment_id": "HUY_D1_SECTION_RESIDUAL_SELECTOR_V1",
        "status": "PASS",
        "folds": selector_training_records,
    }

    return (
        eval_summary,
        r0_preds,
        r1_preds,
        r0_scores,
        swaps_diagnostic,
        stacking_doc,
        selector_train_doc,
        coefficient_stability_doc,
        bootstrap_doc,
    )
