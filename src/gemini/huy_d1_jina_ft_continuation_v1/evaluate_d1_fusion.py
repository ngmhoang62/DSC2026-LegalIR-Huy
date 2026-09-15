"""Evaluate D1 Baseline Parity (J0) and Adapted Jina Continuation Replacement (J1) on CAL600."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

RES_DIR = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1"
RES_DIR.mkdir(parents=True, exist_ok=True)

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_D1_R5 = 0.9569444444444444

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_cal_inputs():
    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: set(queries[q][1]) for q in all_ids}

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned(rel) for name, rel in EXTRA_CV_PATHS.items()}

    full_channels_cv = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    return (
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    )


def run_lobo_eval(
    arm_name: str,
    view_names: List[str],
    views_map: Dict[str, Any],
    extended: Dict[str, List[str]],
    full_channels: Dict[str, Any],
    type_rows: Dict[str, Any],
    cite_rows: Dict[str, Any],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    gold: Dict[str, Set[str]],
) -> Tuple[Dict[str, List[str]], int, Dict[str, Any]]:
    preds: Dict[str, List[str]] = {}
    feature_dim = 0
    block_metrics = {}

    for held in sorted(blocks.keys()):
        train = sum((blocks[n] for n in blocks if n != held), [])
        test_ids = blocks[held]

        eval_ids = train + test_ids
        eval_rows, eval_groups = ltr_features(
            views_map, view_names, extended, eval_ids, full_channels
        )
        for q in eval_rows:
            eval_rows[q] = np.concatenate(
                [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        feature_dim = eval_rows[all_ids[0]].shape[1]

        X_train = np.vstack([eval_rows[q] for q in train])
        y_train = np.concatenate(
            [[d in gold[q] for d in eval_groups[q]] for q in train]
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
            scores = model.decision_function(X_test)
            order = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            top5 = [eval_groups[q][i] for i in order[:5]]
            preds[q] = top5
            b_recalls.append(len(set(top5) & gold[q]) / max(1, len(gold[q])))

        block_metrics[held] = float(np.mean(b_recalls))

    # Aggregate pooled metrics
    pooled_recalls = []
    pooled_precisions = []
    single_gold_recalls = []
    multi_gold_recalls = []

    for q in all_ids:
        top5 = preds[q]
        q_gold = gold[q]
        hits = len(set(top5) & q_gold)
        rec = hits / max(1, len(q_gold))
        prec = hits / 5.0
        pooled_recalls.append(rec)
        pooled_precisions.append(prec)
        if len(q_gold) == 1:
            single_gold_recalls.append(rec)
        else:
            multi_gold_recalls.append(rec)

    metrics = {
        "recall_at_5": float(np.mean(pooled_recalls)),
        "precision_at_5": float(np.mean(pooled_precisions)),
        "single_gold_recall_at_5": float(np.mean(single_gold_recalls)),
        "multi_gold_recall_at_5": float(np.mean(multi_gold_recalls)),
        "block_recalls": block_metrics,
    }

    return preds, feature_dim, metrics


def evaluate_both_arms() -> dict:
    print("Loading CAL inputs...", flush=True)
    (
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    ) = load_cal_inputs()

    views_map = {**local_views}

    # Arm J0: D1 Production baseline
    print("\n--- Evaluating Arm J0: D1 Baseline (48D) ---", flush=True)
    j0_preds, j0_dim, j0_metrics = run_lobo_eval(
        "J0_D1_CURRENT",
        D1_VIEWS,
        views_map,
        extended,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )

    print(f"J0 Recall@5: {j0_metrics['recall_at_5']:.16f} (expected: {EXPECTED_D1_R5:.16f})")
    print(f"J0 Blocks: {j0_metrics['block_recalls']}")
    print(f"J0 Feature Dim: {j0_dim}D")

    # Parity check
    parity_passed = (
        abs(j0_metrics["recall_at_5"] - EXPECTED_D1_R5) < 1e-9
        and j0_dim == 48
    )
    d1_parity = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "expected_recall_at_5": EXPECTED_D1_R5,
        "measured_recall_at_5": j0_metrics["recall_at_5"],
        "expected_blocks": {"A": 0.975, "B": 0.970, "C": 0.995, "D": 0.9338888888888888},
        "measured_blocks": j0_metrics["block_recalls"],
        "expected_feature_dim": 48,
        "measured_feature_dim": j0_dim,
        "status": "PASS" if parity_passed else "BLOCKED_D1_PARITY",
    }
    with open(RES_DIR / "D1_BASELINE_PARITY.json", "w", encoding="utf-8") as f:
        json.dump(d1_parity, f, indent=2)

    if not parity_passed:
        print("FATAL: D1 Parity Failed! Halting.", flush=True)
        return {"status": "BLOCKED_D1_PARITY"}

    # Arm J1: Replace jina_ft with adapted jina_ft_continued
    print("\n--- Evaluating Arm J1: D1 with Adapted Jina Continuation (48D) ---", flush=True)
    adapted_jina_pkl = RES_DIR / "jina_ft_continued_cv.pkl"
    with open(adapted_jina_pkl, "rb") as f:
        adapted_jina_raw = pickle.load(f)

    # Floor align
    fl = min(v for q in adapted_jina_raw for v in adapted_jina_raw[q].values())
    adapted_jina_aligned = {
        q: {d: adapted_jina_raw.get(q, {}).get(d, fl) for d in extended[q]}
        for q in all_ids
    }

    j1_channels = {**full_channels_cv, "jina_ft": adapted_jina_aligned}

    j1_preds, j1_dim, j1_metrics = run_lobo_eval(
        "J1_D1_REPLACE_JINA_FT",
        D1_VIEWS,
        views_map,
        extended,
        j1_channels,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )

    print(f"J1 Recall@5: {j1_metrics['recall_at_5']:.16f}")
    print(f"J1 Blocks: {j1_metrics['block_recalls']}")
    print(f"J1 Feature Dim: {j1_dim}D")

    # Detailed comparative metrics
    wins, losses, ties = 0, 0, 0
    changed_top5 = 0
    gold_entering = 0
    gold_leaving = 0

    for q in all_ids:
        j0_set = set(j0_preds[q])
        j1_set = set(j1_preds[q])
        q_gold = gold[q]

        if j0_set != j1_set:
            changed_top5 += 1

        hits_j0 = len(j0_set & q_gold)
        hits_j1 = len(j1_set & q_gold)

        if hits_j1 > hits_j0:
            wins += 1
        elif hits_j1 < hits_j0:
            losses += 1
        else:
            ties += 1

        new_golds = (j1_set - j0_set) & q_gold
        lost_golds = (j0_set - j1_set) & q_gold
        gold_entering += len(new_golds)
        gold_leaving += len(lost_golds)

    delta_r5 = j1_metrics["recall_at_5"] - j0_metrics["recall_at_5"]
    dist_to_096 = max(0.0, 0.96 - j1_metrics["recall_at_5"])

    # Section 20: Boundary Diagnostics
    print("\n--- Running Section 20 Boundary Diagnostics ---", flush=True)
    old_jina_scores = full_channels_cv["jina_ft"]
    boundary_cases = []
    rescued_10 = 0
    rescued_8 = 0
    rescued_5 = 0
    regressed_5 = 0

    for q in all_ids:
        q_gold = gold[q]
        cands = extended[q]
        j0_set = set(j0_preds[q])

        # Ranks under old Jina and adapted Jina
        old_jina_rank = {
            d: rank + 1
            for rank, d in enumerate(
                sorted(cands, key=lambda d: old_jina_scores[q].get(d, -1e9), reverse=True)
            )
        }
        new_jina_rank = {
            d: rank + 1
            for rank, d in enumerate(
                sorted(cands, key=lambda d: adapted_jina_aligned[q].get(d, -1e9), reverse=True)
            )
        }

        # Check J0 missed golds
        missed_golds = q_gold - j0_set
        for g in missed_golds:
            if g in old_jina_rank and g in new_jina_rank:
                r_old = old_jina_rank[g]
                r_new = new_jina_rank[g]
                boundary_cases.append({
                    "qid": q,
                    "doc_id": g,
                    "type": "missed_gold",
                    "old_jina_rank": r_old,
                    "new_jina_rank": r_new,
                })
                if r_old > 10 and r_new <= 10:
                    rescued_10 += 1
                if r_old > 8 and r_new <= 8:
                    rescued_8 += 1
                if r_old > 5 and r_new <= 5:
                    rescued_5 += 1

        # Check correctly ranked J0 golds that adapted Jina pushed out
        hit_golds = q_gold & j0_set
        for g in hit_golds:
            if g in old_jina_rank and g in new_jina_rank:
                r_old = old_jina_rank[g]
                r_new = new_jina_rank[g]
                if r_old <= 5 and r_new > 5:
                    regressed_5 += 1
                    boundary_cases.append({
                        "qid": q,
                        "doc_id": g,
                        "type": "regressed_gold",
                        "old_jina_rank": r_old,
                        "new_jina_rank": r_new,
                    })

    boundary_audit = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "j0_missed_golds_rescued_to_top10": rescued_10,
        "j0_missed_golds_rescued_to_top8": rescued_8,
        "j0_missed_golds_rescued_to_top5": rescued_5,
        "j0_correct_golds_pushed_past_top5": regressed_5,
        "boundary_events": boundary_cases,
    }
    with open(RES_DIR / "JINA_BOUNDARY_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(boundary_audit, f, indent=2)

    # Predictions jsonl
    with open(RES_DIR / "JINA_D1_CAL_PREDICTIONS.jsonl", "w", encoding="utf-8") as f:
        for q in all_ids:
            row = {
                "qid": q,
                "j0_top5": j0_preds[q],
                "j1_top5": j1_preds[q],
                "gold": list(gold[q]),
                "changed": j0_preds[q] != j1_preds[q],
            }
            f.write(json.dumps(row) + "\n")

    # Full CAL Report
    cal_report = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "feature_dim_j0": j0_dim,
        "feature_dim_j1": j1_dim,
        "j0_d1_current": j0_metrics,
        "j1_d1_replace_jina_ft": j1_metrics,
        "delta": {
            "recall_at_5": delta_r5,
            "precision_at_5": j1_metrics["precision_at_5"] - j0_metrics["precision_at_5"],
            "single_gold_recall_at_5": j1_metrics["single_gold_recall_at_5"] - j0_metrics["single_gold_recall_at_5"],
            "multi_gold_recall_at_5": j1_metrics["multi_gold_recall_at_5"] - j0_metrics["multi_gold_recall_at_5"],
            "block_deltas": {
                b: j1_metrics["block_recalls"][b] - j0_metrics["block_recalls"][b]
                for b in blocks.keys()
            },
        },
        "paired_counts": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "changed_top5_queries": changed_top5,
            "gold_entering_top5": gold_entering,
            "gold_leaving_top5": gold_leaving,
        },
        "distance_to_096": dist_to_096,
    }
    with open(RES_DIR / "JINA_D1_CAL_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(cal_report, f, indent=2)

    print(f"\nWrote JINA_D1_CAL_REPORT.json and JINA_BOUNDARY_AUDIT.json", flush=True)
    print(f"J1 R@5: {j1_metrics['recall_at_5']:.6f} vs J0: {j0_metrics['recall_at_5']:.6f} (delta: {delta_r5:+.6f})")
    print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}, Changed: {changed_top5}")
    return cal_report


if __name__ == "__main__":
    evaluate_both_arms()
