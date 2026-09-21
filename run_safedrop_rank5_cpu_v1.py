#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SEED = 2026
META_MARGIN = 0.50
SCALAR_LAMBDA = 0.25


def load_finalize_module(script_dir: Path):
    p = script_dir / "run_adaptive_k_precision_finalize_v1.py"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Keep this script beside run_adaptive_k_precision_finalize_v1.py."
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


def fit_d1(train_ids, test_ids, point_rows, groups, gold):
    X = np.vstack([point_rows[q] for q in train_ids])
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
    )
    model.fit(scaler.transform(X), y)

    ranking, scoremaps = {}, {}
    for q in test_ids:
        s = model.decision_function(scaler.transform(point_rows[q]))
        sm = {str(d): float(v) for d, v in zip(groups[q], s)}
        order_idx = np.argsort(-s)
        ranking[q] = [str(groups[q][i]) for i in order_idx]
        scoremaps[q] = sm
    return ranking, scoremaps


def view_support(doc, q, views, depth):
    return sum(doc in views[name][q][:depth] for name in D1_VIEWS) / len(D1_VIEWS)


def safe_features(q, ranking, scoremap, point_rows, groups, views, fin):
    d5 = ranking[q][4]
    d4 = ranking[q][3]
    d1 = ranking[q][0]

    index = {str(d): i for i, d in enumerate(groups[q])}
    row5 = np.asarray(point_rows[q][index[d5]], dtype=np.float64)

    ordered_scores = np.asarray(
        [scoremap[q][d] for d in ranking[q]], dtype=np.float64
    )
    scale = fin.robust_scale(ordered_scores)

    s1 = scoremap[q][d1]
    s4 = scoremap[q][d4]
    s5 = scoremap[q][d5]
    z5 = (s5 - float(np.median(ordered_scores))) / scale

    extra = np.asarray(
        [
            z5,
            (s4 - s5) / scale,
            (s1 - s5) / scale,
            view_support(d5, q, views, 5),
            view_support(d5, q, views, 10),
            view_support(d5, q, views, 20),
        ],
        dtype=np.float64,
    )
    return np.concatenate([row5, extra]), float(z5), d5


def class_balanced_weights(y):
    y = np.asarray(y, dtype=np.int8)
    n = len(y)
    n1 = max(int(np.sum(y == 1)), 1)
    n0 = max(int(np.sum(y == 0)), 1)
    return np.where(y == 1, n / (2 * n1), n / (2 * n0)).astype(np.float64)


class DualMeta:
    def __init__(self):
        self.scaler = None
        self.lr = None
        self.hgb = None
        self.constant = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int8)
        unique = np.unique(y)
        if len(unique) < 2:
            self.constant = float(unique[0])
            return self

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.lr = LogisticRegression(
            C=0.10,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=SEED,
        )
        self.lr.fit(Xs, y)

        self.hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=SEED,
        )
        self.hgb.fit(X, y, sample_weight=class_balanced_weights(y))
        return self

    def predict_parts(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.constant is not None:
            p = np.full(len(X), self.constant, dtype=np.float64)
            return p, p
        p_lr = self.lr.predict_proba(self.scaler.transform(X))[:, 1]
        p_hgb = self.hgb.predict_proba(X)[:, 1]
        return p_lr.astype(np.float64), p_hgb.astype(np.float64)


def threshold_from_meta_oof(scores, labels, fin):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    positives = scores[labels == 1]
    if positives.size == 0:
        return None
    return float(
        np.min(positives)
        - META_MARGIN * fin.robust_scale(positives)
    )


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

    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tune_expanded_fusion_selection import ltr_features

    base_rows, groups = ltr_features(
        views, D1_VIEWS, extended, all_ids, full
    )
    point_rows = {}
    for q in all_ids:
        point_rows[q] = np.concatenate(
            [base_rows[q], type_rows[q], cite_rows[q]], axis=1
        )
    dim = point_rows[all_ids[0]].shape[1]
    if dim != 48:
        raise RuntimeError(f"Expected exact D1 feature dim 48, got {dim}")
    print("  exact pointwise D1 feature dim=48", flush=True)

    final_base = {}
    primary_pred = {}
    meta_only_pred = {}
    outer_reports = {}
    primary_actions_all = []
    meta_actions_all = []

    print("[2/5] Nested outer-LOBO SafeDrop...", flush=True)

    for outer_held in sorted(blocks):
        outer_train_blocks = [b for b in sorted(blocks) if b != outer_held]
        outer_train_ids = [
            q for b in outer_train_blocks for q in blocks[b]
        ]
        outer_test_ids = blocks[outer_held]

        outer_rank, outer_smaps = fit_d1(
            outer_train_ids, outer_test_ids, point_rows, groups, gold
        )

        nested_rank, nested_smaps = {}, {}
        for inner_held in outer_train_blocks:
            d1_train = [
                q
                for b in outer_train_blocks
                if b != inner_held
                for q in blocks[b]
            ]
            r, s = fit_d1(
                d1_train,
                blocks[inner_held],
                point_rows,
                groups,
                gold,
            )
            nested_rank.update(r)
            nested_smaps.update(s)

        X_train = {}
        y_train = {}
        z_train = {}
        for q in outer_train_ids:
            feat, z5, d5 = safe_features(
                q, nested_rank, nested_smaps,
                point_rows, groups, views, fin
            )
            X_train[q] = feat
            y_train[q] = int(d5 in gold[q])
            z_train[q] = z5

        y_vec_all = np.asarray(
            [y_train[q] for q in outer_train_ids], dtype=np.int8
        )
        pos_count = int(np.sum(y_vec_all))
        neg_count = int(len(y_vec_all) - pos_count)

        meta_oof = {}
        for meta_held in outer_train_blocks:
            meta_fit_ids = [
                q
                for b in outer_train_blocks
                if b != meta_held
                for q in blocks[b]
            ]
            meta_test_ids = blocks[meta_held]

            Xf = np.vstack([X_train[q] for q in meta_fit_ids])
            yf = np.asarray([y_train[q] for q in meta_fit_ids], dtype=np.int8)
            Xt = np.vstack([X_train[q] for q in meta_test_ids])

            m = DualMeta().fit(Xf, yf)
            p_lr, p_hgb = m.predict_parts(Xt)
            joint = np.maximum(p_lr, p_hgb)
            for q, j in zip(meta_test_ids, joint):
                meta_oof[q] = float(j)

        oof_scores = np.asarray(
            [meta_oof[q] for q in outer_train_ids], dtype=np.float64
        )
        oof_labels = np.asarray(
            [y_train[q] for q in outer_train_ids], dtype=np.int8
        )
        meta_threshold = threshold_from_meta_oof(
            oof_scores, oof_labels, fin
        )
        if meta_threshold is None:
            raise RuntimeError(
                f"No gold rank5 in outer-train for held={outer_held}"
            )

        cal_drop = oof_scores <= meta_threshold
        cal_gold_removed = int(
            np.sum((oof_labels == 1) & cal_drop)
        )
        if cal_gold_removed != 0:
            raise RuntimeError(
                f"Meta calibration removed training OOF gold: held={outer_held}"
            )

        final_meta = DualMeta().fit(
            np.vstack([X_train[q] for q in outer_train_ids]),
            oof_labels,
        )

        z_gold = np.asarray(
            [z_train[q] for q in outer_train_ids if y_train[q] == 1],
            dtype=np.float64,
        )
        scalar_threshold = float(
            np.min(z_gold)
            - SCALAR_LAMBDA * fin.robust_scale(z_gold)
        )

        X_test_list = []
        test_features = {}
        for q in outer_test_ids:
            feat, z5, d5 = safe_features(
                q, outer_rank, outer_smaps,
                point_rows, groups, views, fin
            )
            X_test_list.append(feat)
            test_features[q] = (z5, d5)

        p_lr, p_hgb = final_meta.predict_parts(
            np.vstack(X_test_list)
        )
        joint = np.maximum(p_lr, p_hgb)

        fold_primary_actions = []
        fold_meta_actions = []

        for q, a, b, j in zip(
            outer_test_ids, p_lr, p_hgb, joint
        ):
            base5 = list(outer_rank[q][:5])
            final_base[q] = base5
            primary_pred[q] = list(base5)
            meta_only_pred[q] = list(base5)

            z5, d5 = test_features[q]
            is_gold = d5 in gold[q]

            meta_safe = float(j) <= meta_threshold
            scalar_weak = float(z5) <= scalar_threshold

            if meta_safe:
                meta_only_pred[q] = base5[:-1]
                row = {
                    "qid": q,
                    "block": outer_held,
                    "removed_doc": d5,
                    "removed_was_gold": bool(is_gold),
                    "joint_p_rel": float(j),
                    "p_rel_lr": float(a),
                    "p_rel_hgb": float(b),
                    "meta_threshold": float(meta_threshold),
                    "robust_z5": float(z5),
                    "scalar_threshold": float(scalar_threshold),
                }
                fold_meta_actions.append(row)
                meta_actions_all.append(row)

            if meta_safe and scalar_weak:
                primary_pred[q] = base5[:-1]
                row = {
                    "qid": q,
                    "block": outer_held,
                    "removed_doc": d5,
                    "removed_was_gold": bool(is_gold),
                    "joint_p_rel": float(j),
                    "p_rel_lr": float(a),
                    "p_rel_hgb": float(b),
                    "meta_threshold": float(meta_threshold),
                    "robust_z5": float(z5),
                    "scalar_threshold": float(scalar_threshold),
                }
                fold_primary_actions.append(row)
                primary_actions_all.append(row)

        outer_reports[outer_held] = {
            "outer_train_blocks": outer_train_blocks,
            "rank5_training_labels": {
                "gold": pos_count,
                "nongold": neg_count,
            },
            "meta_threshold": float(meta_threshold),
            "scalar_threshold_lambda_0p25": float(scalar_threshold),
            "meta_calibration_oof": {
                "actions": int(np.sum(cal_drop)),
                "gold_removed": cal_gold_removed,
                "gold_min_joint_probability": float(
                    np.min(oof_scores[oof_labels == 1])
                ),
            },
            "primary_outer": {
                "actions": len(fold_primary_actions),
                "gold_removed": sum(
                    a["removed_was_gold"]
                    for a in fold_primary_actions
                ),
            },
            "meta_only_outer": {
                "actions": len(fold_meta_actions),
                "gold_removed": sum(
                    a["removed_was_gold"]
                    for a in fold_meta_actions
                ),
            },
        }

        print(
            f"  {outer_held}: train rank5 gold/non={pos_count}/{neg_count} | "
            f"meta_thr={meta_threshold:.4f} scalar_thr={scalar_threshold:.4f} | "
            f"PRIMARY actions={len(fold_primary_actions)} "
            f"loss={outer_reports[outer_held]['primary_outer']['gold_removed']} | "
            f"META_ONLY actions={len(fold_meta_actions)} "
            f"loss={outer_reports[outer_held]['meta_only_outer']['gold_removed']}",
            flush=True,
        )

    print("[3/5] Aggregate OOF metrics...", flush=True)
    base_m = metrics(final_base, gold, all_ids)
    primary_m = metrics(primary_pred, gold, all_ids)
    meta_m = metrics(meta_only_pred, gold, all_ids)

    primary_gold_removed = sum(
        a["removed_was_gold"] for a in primary_actions_all
    )
    meta_gold_removed = sum(
        a["removed_was_gold"] for a in meta_actions_all
    )

    print(
        f"BASE       R={base_m['recall']:.10f} "
        f"P={base_m['macro_precision']:.10f} "
        f"meanK={base_m['mean_k']:.4f}",
        flush=True,
    )
    print(
        f"PRIMARY    R={primary_m['recall']:.10f} "
        f"P={primary_m['macro_precision']:.10f} "
        f"meanK={primary_m['mean_k']:.4f} "
        f"actions={len(primary_actions_all)} "
        f"gold_removed={primary_gold_removed}",
        flush=True,
    )
    print(
        f"META_ONLY  R={meta_m['recall']:.10f} "
        f"P={meta_m['macro_precision']:.10f} "
        f"meanK={meta_m['mean_k']:.4f} "
        f"actions={len(meta_actions_all)} "
        f"gold_removed={meta_gold_removed}",
        flush=True,
    )

    if abs(base_m["recall"] - 0.9569444444444444) > 1e-12:
        raise RuntimeError(
            f"Exact D1 parity failed: {base_m['recall']}"
        )

    primary_delta_r = primary_m["recall"] - base_m["recall"]
    primary_delta_p = (
        primary_m["macro_precision"] - base_m["macro_precision"]
    )

    promote = (
        abs(primary_delta_r) <= 1e-12
        and primary_gold_removed == 0
        and len(primary_actions_all) > 33
        and primary_delta_p > 0
    )

    report = {
        "schema": "manual.safedrop_rank5_cpu_v1",
        "status": "COMPLETE",
        "protocol": {
            "outer": "4-block LOBO",
            "upstream_train_features": "inner-OOF D1 rank5 on outer-training blocks",
            "meta_models": [
                "LogisticRegression C=0.10 balanced",
                "HistGradientBoosting max_leaf_nodes=7 min_samples_leaf=20 l2=2",
            ],
            "joint_relevance": "max(P_rel_LR, P_rel_HGB)",
            "meta_safety_margin": META_MARGIN,
            "primary_gate": (
                "joint_p_rel <= meta_threshold AND "
                "ROBUST_Z5 <= lambda0.25_threshold"
            ),
            "gpu_inference": False,
            "public_labels_used": False,
        },
        "baseline": base_m,
        "primary": {
            "metrics": primary_m,
            "delta_recall": primary_delta_r,
            "delta_macro_precision": primary_delta_p,
            "actions": len(primary_actions_all),
            "gold_removed": primary_gold_removed,
            "action_details": primary_actions_all,
        },
        "meta_only": {
            "metrics": meta_m,
            "delta_recall": meta_m["recall"] - base_m["recall"],
            "delta_macro_precision": (
                meta_m["macro_precision"] - base_m["macro_precision"]
            ),
            "actions": len(meta_actions_all),
            "gold_removed": meta_gold_removed,
            "action_details": meta_actions_all,
        },
        "folds": outer_reports,
        "promotion_gate": {
            "recall_exactly_preserved": abs(primary_delta_r) <= 1e-12,
            "zero_gold_removed": primary_gold_removed == 0,
            "actions_gt_33": len(primary_actions_all) > 33,
            "precision_improved": primary_delta_p > 0,
            "pass": promote,
        },
        "verdict": (
            "PROMOTE_SAFEDROP_CPU"
            if promote
            else "KILL_OR_REFINE_SAFEDROP_CPU"
        ),
    }

    out = root / "results/manual/huy_safedrop_rank5_cpu_v1"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "PRIMARY_OOF_PREDICTIONS.json").write_text(
        json.dumps(primary_pred, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/5] Promotion gate")
    print(json.dumps(report["promotion_gate"], indent=2))
    print("[5/5] DONE")
    print("=" * 100)
    print(
        f"PRIMARY: R {base_m['recall']:.10f} -> "
        f"{primary_m['recall']:.10f} "
        f"({primary_delta_r:+.10f})"
    )
    print(
        f"PRIMARY: P {base_m['macro_precision']:.10f} -> "
        f"{primary_m['macro_precision']:.10f} "
        f"({primary_delta_p:+.10f})"
    )
    print(
        f"PRIMARY actions={len(primary_actions_all)} "
        f"gold_removed={primary_gold_removed} "
        f"meanK={primary_m['mean_k']:.4f}"
    )
    print(
        f"META_ONLY actions={len(meta_actions_all)} "
        f"gold_removed={meta_gold_removed} "
        f"P={meta_m['macro_precision']:.10f}"
    )
    print("Verdict:", report["verdict"])
    print("Report:", report_path)
    print("GPU inference performed: FALSE")
    print("=" * 100)


if __name__ == "__main__":
    main()
