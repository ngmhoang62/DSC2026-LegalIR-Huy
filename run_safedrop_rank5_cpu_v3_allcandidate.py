#!/usr/bin/env python
"""
SafeDrop Rank-5 CPU Probe V3 — All-Candidate Nonlinear Verifiers
================================================================

Why V3
------
V1/V2 trained the meta relevance models on only one rank-5 example/query.
That left only 3–13 positive rank-5 examples in some outer folds.

V3 instead trains two nonlinear relevance verifiers on ALL candidate documents
from the training blocks:
  * HistGradientBoostingClassifier
  * ExtraTreesClassifier

For calibration/inference we still look ONLY at D1 rank-5.  Verifier scores are
normalized within each query's candidate pool, so thresholds transfer across
models/folds better than raw probabilities.

Primary predeclared rule:
    D1 ROBUST_Z5 weak at lambda=0.25
    AND HGB normalized relevance below the outer-train gold-rank5 envelope
    AND ExtraTrees normalized relevance below the outer-train gold-rank5 envelope
    => DROP rank5

No GPU. No public labels. No public materialization.

Run:
  python ../run_safedrop_rank5_cpu_v3_allcandidate.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SEED = 2026
SCALAR_LAMBDA = 0.25


def load_finalize_module(script_dir: Path):
    p = script_dir / "run_adaptive_k_precision_finalize_v1.py"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Keep this script beside "
            "run_adaptive_k_precision_finalize_v1.py."
        )
    spec = importlib.util.spec_from_file_location("adaptive_finalize_v1", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def metrics(pred, gold, ids):
    recalls, precisions = [], []
    hits = returned = 0
    for q in ids:
        docs = pred[q]
        h = len(set(docs) & gold[q])
        recalls.append(h / len(gold[q]))
        precisions.append(h / len(docs))
        hits += h
        returned += len(docs)
    return {
        "recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "micro_precision": float(hits / returned),
        "mean_k": float(np.mean([len(pred[q]) for q in ids])),
        "hits": int(hits),
        "returned": int(returned),
    }


def balanced_weights(y):
    y = np.asarray(y, dtype=np.int8)
    n = len(y)
    n1 = max(int(np.sum(y == 1)), 1)
    n0 = max(int(np.sum(y == 0)), 1)
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))


def fit_d1(train_ids, test_ids, rows, groups, gold):
    X = np.vstack([rows[q] for q in train_ids])
    y = np.concatenate(
        [[d in gold[q] for d in groups[q]] for q in train_ids]
    ).astype(np.int8)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    ).fit(scaler.transform(X), y)

    ranking, scoremaps = {}, {}
    for q in test_ids:
        s = model.decision_function(scaler.transform(rows[q]))
        order = np.argsort(-s)
        ranking[q] = [str(groups[q][i]) for i in order]
        scoremaps[q] = {
            str(d): float(v) for d, v in zip(groups[q], s)
        }
    return ranking, scoremaps


class Verifiers:
    def __init__(self):
        self.hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=140,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=3.0,
            random_state=SEED,
        )
        self.et = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=10,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )

    def fit(self, X, y):
        w = balanced_weights(y)
        self.hgb.fit(X, y, sample_weight=w)
        self.et.fit(X, y)
        return self

    def score_all(self, X):
        return (
            self.hgb.predict_proba(X)[:, 1].astype(np.float64),
            self.et.predict_proba(X)[:, 1].astype(np.float64),
        )


def query_normalized_score(raw_scores, fin):
    x = np.asarray(raw_scores, dtype=np.float64)
    return (x - float(np.median(x))) / fin.robust_scale(x)


def score_rank5_with_verifiers(
    ids, ranking, rows, groups, verifiers, fin
):
    out = {}
    for q in ids:
        hgb_raw, et_raw = verifiers.score_all(rows[q])
        hgb_z = query_normalized_score(hgb_raw, fin)
        et_z = query_normalized_score(et_raw, fin)
        index = {str(d): i for i, d in enumerate(groups[q])}
        d5 = ranking[q][4]
        i5 = index[d5]
        out[q] = {
            "doc": d5,
            "hgb_z": float(hgb_z[i5]),
            "et_z": float(et_z[i5]),
        }
    return out


def d1_robust_z5(q, ranking, scoremaps, fin):
    vals = np.asarray(
        [scoremaps[q][d] for d in ranking[q]], dtype=np.float64
    )
    return float(
        (vals[4] - float(np.median(vals))) / fin.robust_scale(vals)
    )


def fit_verifiers_all_candidates(train_ids, rows, groups, gold):
    X = np.vstack([rows[q] for q in train_ids])
    y = np.concatenate(
        [[d in gold[q] for d in groups[q]] for q in train_ids]
    ).astype(np.int8)
    return Verifiers().fit(X, y)


def strict_gold_envelope(values, labels):
    """Largest safe lower-tail cutoff on calibration data, using strict <."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    pos = v[y == 1]
    if not len(pos):
        raise RuntimeError("No positive rank5 examples in calibration")
    return float(np.min(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    script_dir = Path(__file__).resolve().parent

    fin = load_finalize_module(script_dir)

    print("[1/5] Loading exact D1 feature world...", flush=True)
    (
        queries,
        blocks,
        all_ids,
        extended,
        views,
        full,
        gold,
        type_rows,
        cite_rows,
    ) = fin.load_inputs(root)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tune_expanded_fusion_selection import ltr_features

    base_rows, groups = ltr_features(
        views, D1_VIEWS, extended, all_ids, full
    )
    rows = {}
    for q in all_ids:
        rows[q] = np.concatenate(
            [base_rows[q], type_rows[q], cite_rows[q]], axis=1
        )
    if rows[all_ids[0]].shape[1] != 48:
        raise RuntimeError("Expected exact D1 48D features")
    print("  exact D1 pointwise features = 48D", flush=True)

    base_pred = {}
    dual_pred = {}
    hgb_pred = {}
    et_pred = {}

    dual_actions = []
    hgb_actions = []
    et_actions = []
    fold_reports = {}

    print("[2/5] Nested all-candidate verifier OOF...", flush=True)

    for outer_held in sorted(blocks):
        outer_train_blocks = [b for b in sorted(blocks) if b != outer_held]
        outer_train_ids = [
            q for b in outer_train_blocks for q in blocks[b]
        ]
        outer_test_ids = blocks[outer_held]

        # True outer baseline D1.
        outer_rank, outer_smaps = fit_d1(
            outer_train_ids, outer_test_ids, rows, groups, gold
        )

        # Inner OOF upstream D1 rank5 + verifier scores for outer-train calibration.
        cal = {}
        for inner_held in outer_train_blocks:
            inner_train_ids = [
                q
                for b in outer_train_blocks
                if b != inner_held
                for q in blocks[b]
            ]
            inner_test_ids = blocks[inner_held]

            inner_rank, inner_smaps = fit_d1(
                inner_train_ids, inner_test_ids, rows, groups, gold
            )
            inner_ver = fit_verifiers_all_candidates(
                inner_train_ids, rows, groups, gold
            )
            inner_scores = score_rank5_with_verifiers(
                inner_test_ids, inner_rank, rows, groups, inner_ver, fin
            )

            for q in inner_test_ids:
                d5 = inner_rank[q][4]
                cal[q] = {
                    "gold": int(d5 in gold[q]),
                    "d1_z": d1_robust_z5(
                        q, inner_rank, inner_smaps, fin
                    ),
                    "hgb_z": inner_scores[q]["hgb_z"],
                    "et_z": inner_scores[q]["et_z"],
                }

        if set(cal) != set(outer_train_ids):
            raise RuntimeError("Incomplete inner-OOF calibration world")

        labels = np.asarray(
            [cal[q]["gold"] for q in outer_train_ids], dtype=np.int8
        )
        d1z = np.asarray(
            [cal[q]["d1_z"] for q in outer_train_ids], dtype=np.float64
        )
        hgbz = np.asarray(
            [cal[q]["hgb_z"] for q in outer_train_ids], dtype=np.float64
        )
        etz = np.asarray(
            [cal[q]["et_z"] for q in outer_train_ids], dtype=np.float64
        )

        # Frozen scalar D1 gate: lambda .25.
        d1_pos = d1z[labels == 1]
        d1_thr = float(
            np.min(d1_pos)
            - SCALAR_LAMBDA * fin.robust_scale(d1_pos)
        )

        # No grid / no tuned margin here:
        # strict lower-than-all-observed-gold envelope for each nonlinear verifier.
        hgb_thr = strict_gold_envelope(hgbz, labels)
        et_thr = strict_gold_envelope(etz, labels)

        # Verify calibration has zero gold drops for each arm.
        cal_scalar = d1z <= d1_thr
        cal_hgb = cal_scalar & (hgbz < hgb_thr)
        cal_et = cal_scalar & (etz < et_thr)
        cal_dual = cal_scalar & (hgbz < hgb_thr) & (etz < et_thr)

        for name, mask in (
            ("HGB", cal_hgb),
            ("ET", cal_et),
            ("DUAL", cal_dual),
        ):
            loss = int(np.sum(mask & (labels == 1)))
            if loss:
                raise RuntimeError(
                    f"{name} calibration removed gold on outer={outer_held}"
                )

        # Final verifiers fit on all outer-train candidate docs.
        final_ver = fit_verifiers_all_candidates(
            outer_train_ids, rows, groups, gold
        )
        test_scores = score_rank5_with_verifiers(
            outer_test_ids, outer_rank, rows, groups, final_ver, fin
        )

        fold_dual = []
        fold_hgb = []
        fold_et = []

        for q in outer_test_ids:
            top5 = list(outer_rank[q][:5])
            base_pred[q] = top5
            dual_pred[q] = list(top5)
            hgb_pred[q] = list(top5)
            et_pred[q] = list(top5)

            d5 = top5[4]
            is_gold = d5 in gold[q]
            dz = d1_robust_z5(q, outer_rank, outer_smaps, fin)
            hz = test_scores[q]["hgb_z"]
            ez = test_scores[q]["et_z"]

            scalar = dz <= d1_thr
            hsafe = hz < hgb_thr
            esafe = ez < et_thr

            def row():
                return {
                    "qid": q,
                    "block": outer_held,
                    "removed_doc": d5,
                    "removed_was_gold": bool(is_gold),
                    "d1_z": float(dz),
                    "hgb_z": float(hz),
                    "et_z": float(ez),
                    "d1_thr": float(d1_thr),
                    "hgb_thr": float(hgb_thr),
                    "et_thr": float(et_thr),
                }

            if scalar and hsafe:
                hgb_pred[q] = top5[:-1]
                r = row()
                fold_hgb.append(r)
                hgb_actions.append(r)

            if scalar and esafe:
                et_pred[q] = top5[:-1]
                r = row()
                fold_et.append(r)
                et_actions.append(r)

            if scalar and hsafe and esafe:
                dual_pred[q] = top5[:-1]
                r = row()
                fold_dual.append(r)
                dual_actions.append(r)

        fold_reports[outer_held] = {
            "calibration_rank5_gold": int(np.sum(labels)),
            "calibration_rank5_nongold": int(len(labels) - np.sum(labels)),
            "thresholds": {
                "d1_robust_z5_lambda_0p25": d1_thr,
                "hgb_gold_envelope": hgb_thr,
                "et_gold_envelope": et_thr,
            },
            "calibration_actions": {
                "hgb": int(np.sum(cal_hgb)),
                "et": int(np.sum(cal_et)),
                "dual": int(np.sum(cal_dual)),
            },
            "outer": {
                "hgb": {
                    "actions": len(fold_hgb),
                    "gold_removed": sum(x["removed_was_gold"] for x in fold_hgb),
                },
                "et": {
                    "actions": len(fold_et),
                    "gold_removed": sum(x["removed_was_gold"] for x in fold_et),
                },
                "dual": {
                    "actions": len(fold_dual),
                    "gold_removed": sum(x["removed_was_gold"] for x in fold_dual),
                },
            },
        }

        print(
            f"  {outer_held}: rank5 gold/non={int(np.sum(labels))}/"
            f"{int(len(labels)-np.sum(labels))} | "
            f"DUAL={len(fold_dual)} loss="
            f"{fold_reports[outer_held]['outer']['dual']['gold_removed']} | "
            f"HGB={len(fold_hgb)} loss="
            f"{fold_reports[outer_held]['outer']['hgb']['gold_removed']} | "
            f"ET={len(fold_et)} loss="
            f"{fold_reports[outer_held]['outer']['et']['gold_removed']}",
            flush=True,
        )

    print("[3/5] Aggregate...", flush=True)
    base_m = metrics(base_pred, gold, all_ids)
    dual_m = metrics(dual_pred, gold, all_ids)
    hgb_m = metrics(hgb_pred, gold, all_ids)
    et_m = metrics(et_pred, gold, all_ids)

    if abs(base_m["recall"] - 0.9569444444444444) > 1e-12:
        raise RuntimeError(f"D1 parity failed: {base_m['recall']}")

    def summary(name, m, actions):
        loss = sum(x["removed_was_gold"] for x in actions)
        print(
            f"{name:5s} R={m['recall']:.10f} "
            f"P={m['macro_precision']:.10f} "
            f"meanK={m['mean_k']:.4f} "
            f"actions={len(actions)} gold_removed={loss}",
            flush=True,
        )
        return loss

    print(
        f"BASE  R={base_m['recall']:.10f} "
        f"P={base_m['macro_precision']:.10f} "
        f"meanK={base_m['mean_k']:.4f}",
        flush=True,
    )
    dual_loss = summary("DUAL", dual_m, dual_actions)
    hgb_loss = summary("HGB", hgb_m, hgb_actions)
    et_loss = summary("ET", et_m, et_actions)

    dual_delta_r = dual_m["recall"] - base_m["recall"]
    dual_delta_p = dual_m["macro_precision"] - base_m["macro_precision"]

    # Require action coverage in at least 3/4 blocks, besides aggregate zero loss.
    active_blocks = sum(
        fold_reports[b]["outer"]["dual"]["actions"] > 0 for b in fold_reports
    )

    promote = (
        abs(dual_delta_r) <= 1e-12
        and dual_loss == 0
        and len(dual_actions) > 33
        and active_blocks >= 3
        and dual_delta_p > 0
    )

    report = {
        "schema": "manual.safedrop_rank5_cpu_v3_allcandidate",
        "protocol": {
            "outer": "4-block LOBO",
            "verifier_training": "all candidate docs from training blocks",
            "verifiers": [
                "HistGradientBoosting",
                "ExtraTrees",
            ],
            "verifier_score_transfer": "within-query robust z-score",
            "verifier_calibration": (
                "strict lower-than-all-gold-rank5 inner-OOF envelope"
            ),
            "primary": (
                "ROBUST_Z5 lambda=.25 AND HGB envelope AND ET envelope"
            ),
            "gpu_inference": False,
            "public_labels": False,
        },
        "baseline": base_m,
        "dual": {
            "metrics": dual_m,
            "actions": len(dual_actions),
            "gold_removed": dual_loss,
            "delta_recall": dual_delta_r,
            "delta_macro_precision": dual_delta_p,
            "active_outer_blocks": active_blocks,
            "details": dual_actions,
        },
        "hgb_diagnostic": {
            "metrics": hgb_m,
            "actions": len(hgb_actions),
            "gold_removed": hgb_loss,
            "details": hgb_actions,
        },
        "et_diagnostic": {
            "metrics": et_m,
            "actions": len(et_actions),
            "gold_removed": et_loss,
            "details": et_actions,
        },
        "folds": fold_reports,
        "promotion_gate": {
            "exact_recall_preservation": abs(dual_delta_r) <= 1e-12,
            "zero_gold_removed": dual_loss == 0,
            "actions_gt_33": len(dual_actions) > 33,
            "active_blocks_ge_3": active_blocks >= 3,
            "precision_improved": dual_delta_p > 0,
            "pass": promote,
        },
        "verdict": (
            "PROMOTE_SAFEDROP_V3"
            if promote
            else "KILL_OR_REFINE_SAFEDROP_V3"
        ),
    }

    out = root / "results/manual/huy_safedrop_rank5_cpu_v3_allcandidate"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "DUAL_OOF_PREDICTIONS.json").write_text(
        json.dumps(dual_pred, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/5] Promotion gate")
    print(json.dumps(report["promotion_gate"], indent=2))
    print("[5/5] DONE")
    print("=" * 100)
    print(
        f"DUAL: R {base_m['recall']:.10f} -> {dual_m['recall']:.10f} "
        f"({dual_delta_r:+.10f})"
    )
    print(
        f"DUAL: P {base_m['macro_precision']:.10f} -> "
        f"{dual_m['macro_precision']:.10f} "
        f"({dual_delta_p:+.10f})"
    )
    print(
        f"DUAL actions={len(dual_actions)} gold_removed={dual_loss} "
        f"active_blocks={active_blocks}/4"
    )
    print("Verdict:", report["verdict"])
    print("Report:", report_path)
    print("GPU inference performed: FALSE")
    print("=" * 100)


if __name__ == "__main__":
    main()
