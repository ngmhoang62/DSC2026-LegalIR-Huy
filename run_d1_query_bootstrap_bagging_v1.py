#!/usr/bin/env python
"""
HUY_D1_QUERY_BOOTSTRAP_BAGGING_V1
=================================

Hypothesis
----------
Exact D1 may be variance-limited at the Top-5 boundary because each LOBO model
is one LogisticRegression fit on only 300-500 training queries.

This experiment changes NO features and adds NO heuristic:
  * exact same D1 48D features
  * exact same StandardScaler
  * exact same LogisticRegression(C=0.15, balanced, liblinear)
  * query-level bootstrap resampling of the training queries
  * average held-query decision scores across 64 bootstrap D1 models

The bootstrap unit is the QUERY, not individual candidate rows.

No QID-specific rules.
No threshold.
No new expert.
No CAL-derived feature engineering.
No candidate expansion.
No gold use at inference.

Run from Git Bash:
    python ../run_d1_query_bootstrap_bagging_v1.py \
      --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


N_BOOTSTRAPS = 64
MASTER_SEED = 20260917

EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_DIM = 48
EXPECTED_BLOCKS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pkl(root: Path, rel_path: str):
    obj = pickle.loads((root / rel_path).read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_aligned(
    root: Path,
    rel_path: str,
    candidate_pool: Dict[str, List[str]],
    all_ids: List[str],
    floor=None,
):
    obj = load_pkl(root, rel_path)
    if floor is None:
        values = [v for q in obj.values() for v in q.values()]
        if not values:
            raise RuntimeError(f"No score values in {rel_path}")
        floor = min(values)
    return {
        q: {
            d: obj.get(q, {}).get(d, floor)
            for d in candidate_pool[q]
        }
        for q in all_ids
    }


def metrics(
    pred: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    all_ids: List[str],
    blocks: Dict[str, List[str]],
) -> Dict[str, Any]:
    per_r = {
        q: len(set(pred[q][:5]) & gold[q]) / len(gold[q])
        for q in all_ids
    }
    per_p = {
        q: len(set(pred[q][:5]) & gold[q]) / 5.0
        for q in all_ids
    }

    single = [per_r[q] for q in all_ids if len(gold[q]) == 1]
    multi = [per_r[q] for q in all_ids if len(gold[q]) > 1]

    return {
        "recall_at_5": float(np.mean(list(per_r.values()))),
        "precision_at_5": float(np.mean(list(per_p.values()))),
        "single_gold_recall_at_5": float(np.mean(single)) if single else None,
        "multi_gold_recall_at_5": float(np.mean(multi)) if multi else None,
        "block_recalls": {
            b: float(np.mean([per_r[q] for q in ids]))
            for b, ids in blocks.items()
        },
        "per_query_recall": per_r,
    }


def reconstruct_feature_state(root: Path):
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    ctx = (
        root
        / "DSC2026-LegalIR-main"
        / "v4_run"
        / "public_test_dataset"
        / "selected-contexts"
    )
    docs = DocumentStore(sorted(ctx.glob("context_*.json")))

    (
        queries,
        blocks_raw,
        all_ids,
        extended,
        local_views,
        base_scores,
    ) = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )

    blocks = {str(k).upper(): list(v) for k, v in blocks_raw.items()}
    if set(blocks) != {"A", "B", "C", "D"}:
        raise RuntimeError(
            "Unexpected block schema: "
            f"raw={list(blocks_raw)}, normalized={list(blocks)}"
        )

    gold = {q: set(queries[q][1]) for q in all_ids}

    vnlegal_cv = load_pkl(
        root,
        "results/embedding_finetune/vnlegal_lal_cv_scores.pkl",
    )
    crossenc_cv = load_aligned(
        root,
        "results/crossenc_fullpool/cv_scores.pkl",
        extended,
        all_ids,
        -11.5,
    )
    extra_cv = {
        name: load_aligned(root, rel, extended, all_ids)
        for name, rel in EXTRA_CV_PATHS.items()
    }

    d1_channels = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(root, docs, all_ids, extended)
    type_rows = type_features(
        extended,
        type_table,
        queries,
        all_ids,
    )
    own, cited = build_citation_table(
        docs,
        all_ids,
        extended,
    )
    cite_rows = citation_features(
        extended,
        own,
        cited,
        all_ids,
    )

    rows, groups = ltr_features(
        local_views,
        D1_VIEWS,
        extended,
        all_ids,
        d1_channels,
    )
    for q in all_ids:
        rows[q] = np.concatenate(
            [rows[q], type_rows[q], cite_rows[q]],
            axis=1,
        )

    dim = int(rows[all_ids[0]].shape[1])
    if dim != EXPECTED_D1_DIM:
        raise RuntimeError(
            f"D1 feature dimension {dim}, expected {EXPECTED_D1_DIM}"
        )

    return {
        "queries": queries,
        "blocks": blocks,
        "all_ids": all_ids,
        "gold": gold,
        "rows": rows,
        "groups": groups,
        "feature_dim": dim,
        "score_channel_names": sorted(d1_channels),
    }


def fit_exact_d1(
    state: Dict[str, Any],
    train_ids: List[str],
):
    rows = state["rows"]
    groups = state["groups"]
    gold = state["gold"]

    X = np.vstack([rows[q] for q in train_ids])
    y = np.concatenate([
        [d in gold[q] for d in groups[q]]
        for q in train_ids
    ]).astype(np.int8)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    model.fit(scaler.transform(X), y)
    return scaler, model


def bootstrap_training_matrix(
    state: Dict[str, Any],
    sampled_qids: List[str],
):
    rows = state["rows"]
    groups = state["groups"]
    gold = state["gold"]

    X = np.vstack([rows[q] for q in sampled_qids])
    y = np.concatenate([
        [d in gold[q] for d in groups[q]]
        for q in sampled_qids
    ]).astype(np.int8)
    return X, y


def rank_from_scores(groups: List[str], score: np.ndarray) -> List[str]:
    # Match the authoritative D1 implementation exactly.
    order = np.argsort(-np.asarray(score))
    return [groups[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    if not (root / "tune_corpus_cap32_fusion.py").exists():
        raise RuntimeError(
            f"{root} does not look like the sota repo root. "
            "Use --repo-root /d/Study/DSC2026/sota in Git Bash."
        )

    out = root / "results/manual/huy_d1_query_bootstrap_bagging_v1"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[1/5] Building exact D1 48D feature state...", flush=True)
    state = reconstruct_feature_state(root)

    blocks = state["blocks"]
    all_ids = state["all_ids"]
    rows = state["rows"]
    groups = state["groups"]
    gold = state["gold"]

    baseline_ranked: Dict[str, List[str]] = {}
    bagged_ranked: Dict[str, List[str]] = {}

    # Label-free stability diagnostics.
    stability: Dict[str, Any] = {}

    print("[2/5] Exact D1 parity + query-bootstrap bagging...", flush=True)

    for held_index, held in enumerate(sorted(blocks)):
        held_ids = blocks[held]
        train_ids = [
            q
            for b in sorted(blocks)
            if b != held
            for q in blocks[b]
        ]

        print(
            f"  Fold {held}: train={len(train_ids)} held={len(held_ids)}",
            flush=True,
        )

        # Exact single-model D1.
        scaler0, model0 = fit_exact_d1(state, train_ids)
        for q in held_ids:
            s0 = model0.decision_function(
                scaler0.transform(rows[q])
            )
            baseline_ranked[q] = rank_from_scores(groups[q], s0)

        # Bagging accumulators.
        sum_scores = {
            q: np.zeros(len(groups[q]), dtype=np.float64)
            for q in held_ids
        }
        sumsq_scores = {
            q: np.zeros(len(groups[q]), dtype=np.float64)
            for q in held_ids
        }
        top5_counts = {
            q: np.zeros(len(groups[q]), dtype=np.int32)
            for q in held_ids
        }

        held_seed = MASTER_SEED + 100_000 * held_index
        rng = np.random.default_rng(held_seed)

        for bidx in range(N_BOOTSTRAPS):
            sampled = rng.choice(
                np.asarray(train_ids, dtype=object),
                size=len(train_ids),
                replace=True,
            ).tolist()

            Xb, yb = bootstrap_training_matrix(state, sampled)

            # Defensive class check. This should never fire given the corpus.
            if len(np.unique(yb)) != 2:
                raise RuntimeError(
                    f"Bootstrap {bidx} fold {held} has one-class labels"
                )

            scaler = StandardScaler().fit(Xb)
            model = LogisticRegression(
                C=0.15,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=2026,
            )
            model.fit(scaler.transform(Xb), yb)

            for q in held_ids:
                sb = np.asarray(
                    model.decision_function(
                        scaler.transform(rows[q])
                    ),
                    dtype=np.float64,
                )
                sum_scores[q] += sb
                sumsq_scores[q] += sb * sb

                idx = np.argsort(-sb)[:5]
                top5_counts[q][idx] += 1

            if (bidx + 1) % 8 == 0 or bidx + 1 == N_BOOTSTRAPS:
                print(
                    f"    bootstrap {bidx + 1:02d}/{N_BOOTSTRAPS}",
                    flush=True,
                )

        for q in held_ids:
            mean_score = sum_scores[q] / N_BOOTSTRAPS
            second_moment = sumsq_scores[q] / N_BOOTSTRAPS
            variance = np.maximum(
                second_moment - mean_score * mean_score,
                0.0,
            )
            std_score = np.sqrt(variance)

            bagged_ranked[q] = rank_from_scores(
                groups[q],
                mean_score,
            )

            # Save only the top region; this is label-free.
            union_top = list(dict.fromkeys(
                baseline_ranked[q][:10] + bagged_ranked[q][:10]
            ))
            gidx = {d: i for i, d in enumerate(groups[q])}
            stability[q] = {
                "qid": q,
                "held_block": held,
                "baseline_top10": baseline_ranked[q][:10],
                "bagged_top10": bagged_ranked[q][:10],
                "top_region": [
                    {
                        "doc_id": d,
                        "baseline_rank": (
                            baseline_ranked[q].index(d) + 1
                            if d in baseline_ranked[q]
                            else None
                        ),
                        "bagged_rank": (
                            bagged_ranked[q].index(d) + 1
                            if d in bagged_ranked[q]
                            else None
                        ),
                        "bootstrap_top5_frequency": (
                            float(top5_counts[q][gidx[d]])
                            / N_BOOTSTRAPS
                        ),
                        "mean_decision_score": float(
                            mean_score[gidx[d]]
                        ),
                        "std_decision_score": float(
                            std_score[gidx[d]]
                        ),
                    }
                    for d in union_top
                ],
            }

    print("[3/5] Checking exact single-model D1 parity...", flush=True)
    base_metrics = metrics(
        baseline_ranked,
        gold,
        all_ids,
        blocks,
    )

    parity_errors = []
    if abs(base_metrics["recall_at_5"] - EXPECTED_D1_R5) > 1e-12:
        parity_errors.append(
            f"R@5={base_metrics['recall_at_5']} "
            f"expected={EXPECTED_D1_R5}"
        )
    for b, expected in EXPECTED_BLOCKS.items():
        got = base_metrics["block_recalls"][b]
        if abs(got - expected) > 1e-12:
            parity_errors.append(
                f"block {b}={got} expected={expected}"
            )

    if parity_errors:
        raise RuntimeError(
            "BLOCKED_D1_PARITY:\n  - "
            + "\n  - ".join(parity_errors)
        )

    print(
        f"  PASS exact D1 R@5={base_metrics['recall_at_5']:.12f}",
        flush=True,
    )

    # Write label-free predictions/stability BEFORE utility comparison.
    predictions_payload = {
        "schema": "manual.d1_query_bootstrap_bagging_v1.predictions_label_free",
        "bootstraps": N_BOOTSTRAPS,
        "bootstrap_unit": "query",
        "master_seed": MASTER_SEED,
        "baseline_top5": {
            q: baseline_ranked[q][:5]
            for q in all_ids
        },
        "bagged_top5": {
            q: bagged_ranked[q][:5]
            for q in all_ids
        },
    }
    pred_path = out / "BAGGED_PREDICTIONS_LABEL_FREE.json"
    json_dump(pred_path, predictions_payload)

    stability_path = out / "BAGGING_STABILITY_LABEL_FREE.json"
    json_dump(
        stability_path,
        {
            "schema": "manual.d1_query_bootstrap_bagging_v1.stability_label_free",
            "bootstraps": N_BOOTSTRAPS,
            "rows": stability,
        },
    )

    seal = {
        "predictions_sha256": sha256_file(pred_path),
        "stability_sha256": sha256_file(stability_path),
    }
    json_dump(out / "LABEL_FREE_SEAL.json", seal)

    print(
        f"[4/5] Label-free seal written: "
        f"{seal['predictions_sha256'][:12]}...",
        flush=True,
    )

    # Gold utility only after seal.
    print("[5/5] Evaluating bagged D1...", flush=True)
    bag_metrics = metrics(
        bagged_ranked,
        gold,
        all_ids,
        blocks,
    )

    wins = losses = ties = 0
    set_churn = order_churn = 0
    gold_in = gold_out = 0
    changed_details = []

    for q in all_ids:
        r0 = base_metrics["per_query_recall"][q]
        r1 = bag_metrics["per_query_recall"][q]

        if r1 > r0:
            wins += 1
            utility = "WIN"
        elif r1 < r0:
            losses += 1
            utility = "LOSS"
        else:
            ties += 1
            utility = "TIE"

        b0 = baseline_ranked[q][:5]
        b1 = bagged_ranked[q][:5]

        if b0 != b1:
            order_churn += 1
        if set(b0) != set(b1):
            set_churn += 1
            gold_in += len((set(b1) - set(b0)) & gold[q])
            gold_out += len((set(b0) - set(b1)) & gold[q])
            changed_details.append({
                "qid": q,
                "block": next(
                    b for b, ids in blocks.items() if q in ids
                ),
                "utility": utility,
                "recall_before": r0,
                "recall_after": r1,
                "baseline_top5": b0,
                "bagged_top5": b1,
            })

    delta_r = (
        bag_metrics["recall_at_5"]
        - base_metrics["recall_at_5"]
    )
    delta_p = (
        bag_metrics["precision_at_5"]
        - base_metrics["precision_at_5"]
    )
    delta_single = (
        bag_metrics["single_gold_recall_at_5"]
        - base_metrics["single_gold_recall_at_5"]
    )
    delta_multi = (
        bag_metrics["multi_gold_recall_at_5"]
        - base_metrics["multi_gold_recall_at_5"]
    )
    block_delta = {
        b: (
            bag_metrics["block_recalls"][b]
            - base_metrics["block_recalls"][b]
        )
        for b in blocks
    }

    gates = {
        "recall_improves": delta_r > 0,
        "precision_no_decrease": delta_p >= -1e-12,
        "wins_gt_losses": wins > losses,
        "no_block_decrease": all(
            x >= -1e-12 for x in block_delta.values()
        ),
        "single_no_decrease": delta_single >= -1e-12,
        "multi_no_decrease": delta_multi >= -1e-12,
    }

    if bag_metrics["recall_at_5"] >= 0.96 and all(gates.values()):
        verdict = "STRONG_PROMOTE_D1_QUERY_BOOTSTRAP_BAGGING_V1"
    elif all(gates.values()):
        verdict = "PROMISING_D1_QUERY_BOOTSTRAP_BAGGING_V1"
    else:
        verdict = "KILL_D1_QUERY_BOOTSTRAP_BAGGING_V1"

    report = {
        "schema": "manual.d1_query_bootstrap_bagging_v1.report",
        "runtime_seconds": time.perf_counter() - started,
        "bootstraps": N_BOOTSTRAPS,
        "bootstrap_unit": "query",
        "feature_dim": state["feature_dim"],
        "score_channel_names": state["score_channel_names"],
        "label_free_seal": seal,
        "baseline": {
            k: v
            for k, v in base_metrics.items()
            if k != "per_query_recall"
        },
        "bagged": {
            k: v
            for k, v in bag_metrics.items()
            if k != "per_query_recall"
        },
        "delta": {
            "recall_at_5": delta_r,
            "precision_at_5": delta_p,
            "single_gold_recall_at_5": delta_single,
            "multi_gold_recall_at_5": delta_multi,
            "blocks": block_delta,
        },
        "paired": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "top5_set_churn": set_churn,
            "top5_order_churn": order_churn,
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
        },
        "promotion_gates": gates,
        "verdict": verdict,
        "changed_top5_details": changed_details,
    }
    json_dump(out / "FINAL_REPORT.json", report)

    print("=" * 80)
    print(
        f"D1       R@5={base_metrics['recall_at_5']:.10f} "
        f"P@5={base_metrics['precision_at_5']:.10f}"
    )
    print(
        f"Bagged   R@5={bag_metrics['recall_at_5']:.10f} "
        f"P@5={bag_metrics['precision_at_5']:.10f}"
    )
    print(
        f"Delta    R={delta_r:+.10f} "
        f"P={delta_p:+.10f}"
    )
    print(
        f"Single   {base_metrics['single_gold_recall_at_5']:.10f} "
        f"-> {bag_metrics['single_gold_recall_at_5']:.10f} "
        f"({delta_single:+.10f})"
    )
    print(
        f"Multi    {base_metrics['multi_gold_recall_at_5']:.10f} "
        f"-> {bag_metrics['multi_gold_recall_at_5']:.10f} "
        f"({delta_multi:+.10f})"
    )
    print(
        "Blocks   "
        + " ".join(
            f"{b}:{block_delta[b]:+.6f}"
            for b in sorted(block_delta)
        )
    )
    print(
        f"W/L/T    {wins}/{losses}/{ties} | "
        f"set churn={set_churn} | "
        f"gold in/out={gold_in}/{gold_out}"
    )
    print(f"Verdict  {verdict}")
    print(f"Report   {out / 'FINAL_REPORT.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
