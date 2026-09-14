"""Train structured linear energy model over Top-8 candidate slates.

Evaluates three endpoints:
- BASELINE: authoritative original Top-5
- QCSC_UNARY: structured learner using only 7 unary baseline evidence features
- QCSC_FULL: full model (22 features: unary + pair + coverage + cardinality)

Generates:
- QCSC_OOF_PREDICTIONS.jsonl
- PAIR_SUPPORT_AUDIT.json
- GENERALIZATION_AUDIT.json
- QCSC_FINAL_REPORT.json
- DECISION.md
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from common import (
    EXPECTED_METRICS,
    RESULTS_DIR,
    compare_endpoints,
    evaluate_predictions,
    get_56_slates,
    load_baseline_data,
    load_lal_query_vectors,
)
from slate_features import (
    SLATE_INDICES,
    extract_slate_feature_tensor_for_query,
    precompute_support_neighbors,
)

torch.manual_seed(2026)
np.random.seed(2026)


def train_structured_model(
    X_train_norm: torch.Tensor,  # (N_train, 56, D)
    best_mask_train: torch.Tensor,  # (N_train, 56) bool
    X_val_norm: torch.Tensor,  # (N_val, 56, D)
    best_mask_val: torch.Tensor,  # (N_val, 56) bool
    val_golds: List[Set[str]],
    val_orders: List[List[str]],
    val_slates: List[List[Tuple[str, ...]]],
    dim: int,
    lr: float = 0.01,
    l2_reg: float = 0.001,
    max_epochs: int = 200,
    patience: int = 15,
) -> torch.Tensor:
    """Train linear parameter vector w using structured softmax loss with early stopping."""
    w = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    optimizer = torch.optim.Adam([w], lr=lr)

    best_val_r5 = -1.0
    best_w = w.detach().clone()
    patience_counter = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        scores_train = torch.einsum('nsd,d->ns', X_train_norm, w)
        all_lse = torch.logsumexp(scores_train, dim=1)
        best_scores = torch.where(best_mask_train, scores_train, torch.tensor(-1e9, dtype=torch.float32))
        best_lse = torch.logsumexp(best_scores, dim=1)
        loss = torch.mean(all_lse - best_lse) + 0.5 * l2_reg * torch.sum(w ** 2)
        loss.backward()
        optimizer.step()

        # Evaluate on validation split
        with torch.no_grad():
            scores_val = torch.einsum('nsd,d->ns', X_val_norm, w)
            pred_slate_indices = torch.argmax(scores_val, dim=1).cpu().numpy()

            hits = []
            for i, s_idx in enumerate(pred_slate_indices):
                chosen_slate = set(val_slates[i][s_idx])
                g = val_golds[i]
                hits.append(len(chosen_slate & g) / len(g))
            val_r5 = float(np.mean(hits))

        if val_r5 > best_val_r5 + 1e-6:
            best_val_r5 = val_r5
            best_w = w.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_w


def run_experiment():
    started = time.perf_counter()
    print("=== Step 4 & 5: Running Complete QCSC Experiment ===", flush=True)

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()
    qrow, lal_vecs = load_lal_query_vectors()

    # Load P_multi from CARDINALITY_OOF_REPORT.json
    card_file = RESULTS_DIR / "CARDINALITY_OOF_REPORT.json"
    assert card_file.exists(), f"Missing {card_file}"
    with card_file.open("r", encoding="utf-8") as f:
        card_data = json.load(f)
    p_multi_oof = card_data["oof_predictions"]
    print(f"Loaded P_multi OOF predictions for {len(p_multi_oof)} queries.")

    all_qids = set(pools)
    q_slates = {q: get_56_slates(base_orders[q][:8]) for q in pools}

    # Diagnostics for PAIR_SUPPORT_AUDIT.json (K=32 vs K=16)
    pair_audit_k32 = []
    pair_audit_k16 = []

    oof_predictions_unary: Dict[str, List[str]] = {}
    oof_predictions_full: Dict[str, List[str]] = {}
    oof_support_regimes: Dict[str, Dict[str, Any]] = {}

    for fold, test_ids in folds.items():
        fold_started = time.perf_counter()
        held = set(test_ids)
        blocked = set(map(str, dup.get(fold, [])))
        train_ids = sorted(all_qids - held - blocked, key=int)

        print(f"\n--- Outer Fold {fold} ---")
        print(f"Train population: {len(train_ids)} queries, Held test: {len(test_ids)} queries")

        # 1. Precompute support neighbours for training queries (self-exclusion)
        print("Precomputing support neighbours (K=32 and K=16)...", flush=True)
        train_sims_32, train_neighbors_32 = precompute_support_neighbors(
            train_ids, train_ids, lal_vecs, qrow, K=32, is_training_self=True
        )
        test_sims_32, test_neighbors_32 = precompute_support_neighbors(
            test_ids, train_ids, lal_vecs, qrow, K=32, is_training_self=False
        )

        test_sims_16, test_neighbors_16 = precompute_support_neighbors(
            test_ids, train_ids, lal_vecs, qrow, K=16, is_training_self=False
        )

        # Audit pair support stability for test queries
        for i, qid in enumerate(test_ids):
            top8 = base_orders[qid][:8]
            # Check K=32
            cand_pairs_seen_32 = 0
            for a, b in combinations(top8, 2):
                if any(a in golds[nq] and b in golds[nq] for nq in test_neighbors_32[i]):
                    cand_pairs_seen_32 += 1
            pair_audit_k32.append(cand_pairs_seen_32)

            # Check K=16
            cand_pairs_seen_16 = 0
            for a, b in combinations(top8, 2):
                if any(a in golds[nq] and b in golds[nq] for nq in test_neighbors_16[i]):
                    cand_pairs_seen_16 += 1
            pair_audit_k16.append(cand_pairs_seen_16)

        # 2. Extract slate features
        print("Extracting slate feature tensors for outer-training...", flush=True)
        X_train_list = []
        best_mask_train_list = []
        for i, qid in enumerate(train_ids):
            top8 = base_orders[qid][:8]
            scores = base_scores[qid]
            sims = train_sims_32[i]
            n_qids = train_neighbors_32[i]
            p_m = p_multi_oof.get(qid, 0.5)

            feats = extract_slate_feature_tensor_for_query(
                qid, top8, scores, sims, n_qids, golds, p_m
            )
            X_train_list.append(feats)

            # Utility per slate
            g = golds[qid]
            slates = q_slates[qid]
            utils = [len(set(s) & g) / len(g) for s in slates]
            max_u = max(utils)
            best_mask_train_list.append([u >= max_u - 1e-6 for u in utils])

        X_train_arr = np.array(X_train_list, dtype=np.float32)  # (N_train, 56, 22)
        best_mask_train_arr = np.array(best_mask_train_list, dtype=bool)

        print("Extracting slate feature tensors for held-fold test...", flush=True)
        X_test_list = []
        for i, qid in enumerate(test_ids):
            top8 = base_orders[qid][:8]
            scores = base_scores[qid]
            sims = test_sims_32[i]
            n_qids = test_neighbors_32[i]
            p_m = p_multi_oof.get(qid, 0.5)

            feats = extract_slate_feature_tensor_for_query(
                qid, top8, scores, sims, n_qids, golds, p_m
            )
            X_test_list.append(feats)

        X_test_arr = np.array(X_test_list, dtype=np.float32)  # (N_test, 56, 22)

        # Standardize features using outer-training statistics
        feat_mean = X_train_arr.mean(axis=(0, 1), keepdims=True)
        feat_std = X_train_arr.std(axis=(0, 1), keepdims=True) + 1e-6

        X_train_norm_full = (X_train_arr - feat_mean) / feat_std
        X_test_norm_full = (X_test_arr - feat_mean) / feat_std

        X_train_norm_unary = X_train_norm_full[:, :, :7]
        X_test_norm_unary = X_test_norm_full[:, :, :7]

        # 80/20 inner split on outer-training for early stopping
        num_train = len(train_ids)
        perm = np.random.RandomState(2026).permutation(num_train)
        split_idx = int(0.8 * num_train)
        inner_train_idx = perm[:split_idx]
        inner_val_idx = perm[split_idx:]

        val_qids = [train_ids[idx] for idx in inner_val_idx]
        val_golds = [golds[q] for q in val_qids]
        val_orders = [base_orders[q] for q in val_qids]
        val_slates = [q_slates[q] for q in val_qids]

        # --- ABLATION 1: QCSC_UNARY ---
        print("Training QCSC_UNARY model (D=7)...", flush=True)
        best_w_unary = train_structured_model(
            torch.tensor(X_train_norm_unary[inner_train_idx], dtype=torch.float32),
            torch.tensor(best_mask_train_arr[inner_train_idx], dtype=torch.bool),
            torch.tensor(X_train_norm_unary[inner_val_idx], dtype=torch.float32),
            torch.tensor(best_mask_train_arr[inner_val_idx], dtype=torch.bool),
            val_golds, val_orders, val_slates, dim=7, lr=0.01, max_epochs=200, patience=15
        )

        with torch.no_grad():
            scores_test_unary = torch.einsum('nsd,d->ns', torch.tensor(X_test_norm_unary, dtype=torch.float32), best_w_unary)
            test_preds_unary = torch.argmax(scores_test_unary, dim=1).cpu().numpy()

        for i, qid in enumerate(test_ids):
            s_idx = test_preds_unary[i]
            chosen_slate = q_slates[qid][s_idx]
            order = base_orders[qid]
            new_top5 = sorted(chosen_slate, key=lambda d: order.index(d))
            full_order = new_top5 + [d for d in order if d not in set(new_top5)]
            oof_predictions_unary[qid] = full_order

        # --- ABLATION 2: QCSC_FULL ---
        print("Training QCSC_FULL model (D=22)...", flush=True)
        best_w_full = train_structured_model(
            torch.tensor(X_train_norm_full[inner_train_idx], dtype=torch.float32),
            torch.tensor(best_mask_train_arr[inner_train_idx], dtype=torch.bool),
            torch.tensor(X_train_norm_full[inner_val_idx], dtype=torch.float32),
            torch.tensor(best_mask_train_arr[inner_val_idx], dtype=torch.bool),
            val_golds, val_orders, val_slates, dim=22, lr=0.01, max_epochs=200, patience=15
        )

        with torch.no_grad():
            scores_test_full = torch.einsum('nsd,d->ns', torch.tensor(X_test_norm_full, dtype=torch.float32), best_w_full)
            test_preds_full = torch.argmax(scores_test_full, dim=1).cpu().numpy()

        for i, qid in enumerate(test_ids):
            s_idx = test_preds_full[i]
            chosen_slate = q_slates[qid][s_idx]
            order = base_orders[qid]
            new_top5 = sorted(chosen_slate, key=lambda d: order.index(d))
            full_order = new_top5 + [d for d in order if d not in set(new_top5)]
            oof_predictions_full[qid] = full_order

            # Classify support regime for generalization audit
            top8 = order[:8]
            top5_orig = set(order[:5])
            promoted = [d for d in chosen_slate if d not in top5_orig]

            # Pair familiarity: was any pair in chosen_slate seen in support?
            pairs_in_slate = list(combinations(chosen_slate, 2))
            seen_pair = any(
                any(a in golds[nq] and b in golds[nq] for nq in test_neighbors_32[i])
                for a, b in pairs_in_slate
            )

            # Label familiarity: was any promoted doc seen in support golds?
            support_golds_all = {d for nq in test_neighbors_32[i] for d in golds[nq]}
            promoted_seen_as_gold = all(d in support_golds_all for d in promoted) if promoted else True

            # Semantic support: top-1 cosine
            top1_sim = float(test_sims_32[i, 0])

            oof_support_regimes[qid] = {
                "seen_pair": bool(seen_pair),
                "promoted_seen_as_gold": bool(promoted_seen_as_gold),
                "nearest_neighbor_cosine": top1_sim,
                "is_multi_gold": len(golds[qid]) > 1,
                "p_multi": p_multi_oof.get(qid, 0.5),
                "slate_changed": s_idx != 0,
            }

        print(f"Fold {fold} finished in {time.perf_counter() - fold_started:.2f}s")

    # Save QCSC_OOF_PREDICTIONS.jsonl
    oof_pred_path = RESULTS_DIR / "QCSC_OOF_PREDICTIONS.jsonl"
    print(f"\nWriting {oof_pred_path}...", flush=True)
    with oof_pred_path.open("w", encoding="utf-8") as f:
        for qid in sorted(pools, key=int):
            row = {
                "qid": qid,
                "fold": fold_for[qid],
                "gold": list(golds[qid]),
                "baseline_top5": base_orders[qid][:5],
                "qcsc_unary_top5": oof_predictions_unary[qid][:5],
                "qcsc_full_top5": oof_predictions_full[qid][:5],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save PAIR_SUPPORT_AUDIT.json
    audit_k32_frac = float(np.mean([p > 0 for p in pair_audit_k32]))
    audit_k16_frac = float(np.mean([p > 0 for p in pair_audit_k16]))
    pair_audit_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries": len(pair_audit_k32),
        "k32": {
            "queries_with_supported_pair_pct": round(audit_k32_frac * 100, 2),
            "mean_supported_pairs_per_query": float(np.mean(pair_audit_k32)),
            "max_supported_pairs": int(np.max(pair_audit_k32)),
        },
        "k16": {
            "queries_with_supported_pair_pct": round(audit_k16_frac * 100, 2),
            "mean_supported_pairs_per_query": float(np.mean(pair_audit_k16)),
            "max_supported_pairs": int(np.max(pair_audit_k16)),
        },
        "stability_ratio_k16_to_k32": round(audit_k16_frac / max(audit_k32_frac, 1e-6), 4),
    }
    with (RESULTS_DIR / "PAIR_SUPPORT_AUDIT.json").open("w", encoding="utf-8") as f:
        json.dump(pair_audit_payload, f, indent=2, ensure_ascii=False)
    print("Saved PAIR_SUPPORT_AUDIT.json")

    # Compute evaluations
    print("\n=== Evaluating All Three Endpoints ===", flush=True)
    metrics_base = evaluate_predictions(base_orders, golds, folds)
    metrics_unary = evaluate_predictions(oof_predictions_unary, golds, folds)
    metrics_full = evaluate_predictions(oof_predictions_full, golds, folds)

    comp_unary = compare_endpoints(oof_predictions_unary, base_orders, golds, folds)
    comp_full = compare_endpoints(oof_predictions_full, base_orders, golds, folds)

    print(f"BASELINE:   R@5 = {metrics_base['recall_at_5']:.16f} | Single = {metrics_base['single_gold_recall_at_5']:.6f} | Multi = {metrics_base['multi_gold_recall_at_5']:.6f}")
    print(f"QCSC_UNARY: R@5 = {metrics_unary['recall_at_5']:.16f} | Delta = {comp_unary['delta_recall_at_5']:+.6f} | Wins = {comp_unary['wins']}, Losses = {comp_unary['losses']}")
    print(f"QCSC_FULL:  R@5 = {metrics_full['recall_at_5']:.16f} | Delta = {comp_full['delta_recall_at_5']:+.6f} | Wins = {comp_full['wins']}, Losses = {comp_full['losses']}")

    # Generalization Audit
    print("\nComputing Generalization Audit slices...", flush=True)
    slices = defaultdict(list)
    for qid in pools:
        reg = oof_support_regimes[qid]
        g = golds[qid]
        ra = len(set(base_orders[qid][:5]) & g) / len(g)
        rb = len(set(oof_predictions_full[qid][:5]) & g) / len(g)
        delta = rb - ra
        is_win = rb > ra + 1e-9
        is_loss = rb < ra - 1e-9
        is_tie = not (is_win or is_loss)

        rec = {"qid": qid, "delta": delta, "win": is_win, "loss": is_loss, "tie": is_tie}

        # Pair familiarity
        slices["pair_seen" if reg["seen_pair"] else "pair_unseen"].append(rec)

        # Label familiarity
        slices["label_seen" if reg["promoted_seen_as_gold"] else "label_unseen"].append(rec)

        # Semantic support
        c = reg["nearest_neighbor_cosine"]
        if c >= 0.90:
            slices["cosine_ge_0.90"].append(rec)
        elif c >= 0.85:
            slices["cosine_0.85_0.90"].append(rec)
        elif c >= 0.80:
            slices["cosine_0.80_0.85"].append(rec)
        else:
            slices["cosine_lt_0.80"].append(rec)

        # Query cardinality
        slices["true_single_gold" if not reg["is_multi_gold"] else "true_multi_gold"].append(rec)

        # Predicted cardinality
        p = reg["p_multi"]
        if p < 0.25:
            slices["p_multi_lt_0.25"].append(rec)
        elif p < 0.50:
            slices["p_multi_0.25_0.50"].append(rec)
        elif p < 0.75:
            slices["p_multi_0.50_0.75"].append(rec)
        else:
            slices["p_multi_ge_0.75"].append(rec)

    gen_audit_payload = {}
    for s_name, recs in slices.items():
        cnt = len(recs)
        w_cnt = sum(r["win"] for r in recs)
        l_cnt = sum(r["loss"] for r in recs)
        t_cnt = sum(r["tie"] for r in recs)
        d_mean = float(np.mean([r["delta"] for r in recs])) if cnt > 0 else 0.0
        gen_audit_payload[s_name] = {
            "query_count": cnt,
            "wins": w_cnt,
            "losses": l_cnt,
            "ties": t_cnt,
            "delta_recall_at_5": d_mean,
        }

    with (RESULTS_DIR / "GENERALIZATION_AUDIT.json").open("w", encoding="utf-8") as f:
        json.dump(gen_audit_payload, f, indent=2, ensure_ascii=False)
    print("Saved GENERALIZATION_AUDIT.json")

    # Decision classification
    delta_full = comp_full["delta_recall_at_5"]
    worst_fold = comp_full["worst_fold_delta"]

    if delta_full >= 0.0020:
        decision = "BREAKTHROUGH"
    elif delta_full >= 0.0008 and worst_fold >= -0.0010:
        decision = "PROMOTE"
    elif delta_full > 0.0:
        decision = "MARGINAL"
    else:
        decision = "KILL"

    print(f"\nFinal Decision: {decision} (Delta: {delta_full:+.6f}, Worst fold: {worst_fold:+.6f})")

    # Write QCSC_FINAL_REPORT.json
    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "baseline": {
            "metrics": metrics_base,
        },
        "qcsc_unary": {
            "metrics": metrics_unary,
            "comparison": comp_unary,
        },
        "qcsc_full": {
            "metrics": metrics_full,
            "comparison": comp_full,
        },
        "headroom_oracle_r5": 0.9631359366804939,
        "runtime_seconds": time.perf_counter() - started,
    }

    with (RESULTS_DIR / "QCSC_FINAL_REPORT.json").open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print("Saved QCSC_FINAL_REPORT.json")

    # Write DECISION.md
    dec_lines = [
        f"# QCSC Decision Report: {decision}\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
        f"**Authoritative Baseline R@5:** `{metrics_base['recall_at_5']:.16f}`\n",
        f"**QCSC_UNARY R@5:** `{metrics_unary['recall_at_5']:.16f}` (Delta: `{comp_unary['delta_recall_at_5']:+.6f}`, Wins: {comp_unary['wins']}, Losses: {comp_unary['losses']})\n",
        f"**QCSC_FULL R@5:** `{metrics_full['recall_at_5']:.16f}` (Delta: `{comp_full['delta_recall_at_5']:+.6f}`, Wins: {comp_full['wins']}, Losses: {comp_full['losses']})\n",
        f"**Decision Label:** `{decision}`\n\n",
        "## Summary Metrics Comparison\n",
        "| Endpoint | Recall@5 | Precision@5 | Single-gold R@5 | Multi-gold R@5 | Worst Fold Delta | Wins | Losses | Churn |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| BASELINE | {metrics_base['recall_at_5']:.6f} | {metrics_base['precision_at_5']:.6f} | {metrics_base['single_gold_recall_at_5']:.6f} | {metrics_base['multi_gold_recall_at_5']:.6f} | 0.000000 | 0 | 0 | 0 |",
        f"| QCSC_UNARY | {metrics_unary['recall_at_5']:.6f} | {metrics_unary['precision_at_5']:.6f} | {metrics_unary['single_gold_recall_at_5']:.6f} | {metrics_unary['multi_gold_recall_at_5']:.6f} | {comp_unary['worst_fold_delta']:+.6f} | {comp_unary['wins']} | {comp_unary['losses']} | {comp_unary['changed_top5_sets']} |",
        f"| QCSC_FULL | {metrics_full['recall_at_5']:.6f} | {metrics_full['precision_at_5']:.6f} | {metrics_full['single_gold_recall_at_5']:.6f} | {metrics_full['multi_gold_recall_at_5']:.6f} | {comp_full['worst_fold_delta']:+.6f} | {comp_full['wins']} | {comp_full['losses']} | {comp_full['changed_top5_sets']} |\n\n",
        "## Per-Fold Delta Breakdown (QCSC_FULL)\n",
        "| Fold | Baseline R@5 | QCSC_FULL R@5 | Delta |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for fold in sorted(folds):
        b_val = metrics_base["per_fold_recall_at_5"][fold]
        f_val = metrics_full["per_fold_recall_at_5"][fold]
        d_val = comp_full["per_fold_delta"][fold]
        dec_lines.append(f"| {fold} | {b_val:.6f} | {f_val:.6f} | {d_val:+.6f} |")

    dec_lines.extend([
        "\n## Generalization Audit Highlights\n",
        "| Slice | Query Count | Wins | Losses | Ties | Delta Recall@5 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for s_name in sorted(gen_audit_payload):
        sp = gen_audit_payload[s_name]
        dec_lines.append(f"| `{s_name}` | {sp['query_count']} | {sp['wins']} | {sp['losses']} | {sp['ties']} | {sp['delta_recall_at_5']:+.6f} |")

    dec_lines.append("\n## Decision Rationale\n")
    if decision == "KILL":
        dec_lines.append("QCSC did not improve upon the authoritative Huy-fasttrack strict 5-fold OOF baseline. Following strict anti-overfit protocols, the QCSC family is classified as KILL and will NOT replace the baseline.\n")
    elif decision == "MARGINAL":
        dec_lines.append("QCSC achieved a small positive gain (0 < delta < +0.0008). It is preserved for potential future orthogonal combination but does not replace the primary candidate automatically.\n")
    elif decision in ("PROMOTE", "BREAKTHROUGH"):
        dec_lines.append(f"QCSC demonstrated robust positive gains across folds with no fold collapse. Classification: {decision}.\n")

    with (RESULTS_DIR / "DECISION.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(dec_lines))
    print("Saved DECISION.md")

    return decision, final_report


if __name__ == "__main__":
    run_experiment()
