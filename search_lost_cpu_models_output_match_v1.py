#!/usr/bin/env python
"""
OUTPUT-MATCH LOST LARGE-LTR + EMPIRICAL-PAIRWISE MODELS
=======================================================

We already proved the historical FINAL stack config is exactly:
  weights = [robust=.40, pair=.30, profile=.15, graph=.15], rrf_k=0

So this script targets the two learned CPU artifacts that were missing and
had to be retrained:
  - burst_large_ltr/best_model.pkl
  - burst_empirical_pairwise/model.pkl

It recreates ALL historical candidate model configs from the original tuners
and selects by OUTPUT PARITY against the historical unlabeled artifact:
  results/burst_gpu_threeview/cpu_top20.pkl

No public labels are read.

Search A — Large-LTR:
  exact 6 historical XGBoost configs from tune_burst_large_ltr.py.
  Each candidate replaces ONLY the LARGE submodel inside robust;
  pair/profile/graph stay fixed.

Search B — Empirical pairwise:
  negative_depth ∈ {5,10,20,40,80}
  C ∈ {.003,.01,.03,.1,.3,1,3}
  = 35 exact historical model configs.
  Each candidate replaces ONLY pair branch;
  robust/profile/graph stay fixed.

Outputs:
  results/manual/huy_private_d1_rel_l0_v1/
    LOST_MODEL_OUTPUT_MATCH_V1.json
    recovered_large_model.pkl   (only if an exact/high-parity candidate improves)
    recovered_pair_model.pkl    (same)

Run:
  cd /d/Study/DSC2026/sota
  source dsc_env_huy/Scripts/activate
  python ../search_lost_cpu_models_output_match_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker


FINAL_WEIGHTS = (.40, .30, .15, .15)
FINAL_K = 0

XGB_CONFIGS = [
    (2, .03, 180, 10),
    (2, .05, 140, 10),
    (3, .03, 200, 10),
    (3, .05, 160, 10),
    (4, .03, 180, 10),
    (3, .03, 220, 20),
]
PAIR_DEPTHS = (5, 10, 20, 40, 80)
PAIR_CS = (.003, .01, .03, .1, .3, 1.0, 3.0)


def load_pickle(p: Path):
    return pickle.loads(p.read_bytes())


def save_pickle(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=5))
    tmp.replace(p)


def dump(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def weighted_rrf_one(branches, weights=FINAL_WEIGHTS, k=FINAL_K):
    rankmaps = [
        {d: i + 1 for i, d in enumerate(branch)}
        for branch in branches
    ]
    docs = set().union(*(r.keys() for r in rankmaps))
    return sorted(
        docs,
        key=lambda d: (
            -sum(
                w / (k + ranks.get(d, 100000))
                for w, ranks in zip(weights, rankmaps)
            ),
            d,
        ),
    )[:100]


def metrics(pred, target, ids):
    exact5 = exact20 = set5 = 0
    ov5 = ov20 = pos5 = 0.0
    for q in ids:
        a, b = pred[q], target[q]
        exact5 += a[:5] == b[:5]
        exact20 += a[:20] == b[:20]
        set5 += set(a[:5]) == set(b[:5])
        ov5 += len(set(a[:5]) & set(b[:5])) / 5
        ov20 += len(set(a[:20]) & set(b[:20])) / 20
        pos5 += sum(x == y for x, y in zip(a[:5], b[:5])) / 5
    n = len(ids)
    return {
        "exact_top5_order": int(exact5),
        "exact_top5_set": int(set5),
        "mean_top5_overlap": ov5 / n,
        "mean_top5_positional": pos5 / n,
        "exact_top20_order": int(exact20),
        "mean_top20_overlap": ov20 / n,
    }


def key(m):
    return (
        m["exact_top5_order"],
        m["exact_top5_set"],
        m["mean_top5_overlap"],
        m["mean_top20_overlap"],
        m["exact_top20_order"],
    )


def final_from_components(comp, robust=None, pair=None):
    robust = robust or comp["robust"]
    pair = pair or comp["pair"]
    out = {}
    for q in comp["ids"]:
        out[q] = weighted_rrf_one([
            robust[q],
            pair[q],
            comp["profile"][q],
            comp["graph"][q],
        ])
    return out


def load_public_retrieval(root, ids):
    for p in (
        root / "results/burst_multistage/public_retrieval.pkl",
        root / "results/burst_robust_fusion/public_retrieval.pkl",
        root / "results/burst_expanded_fusion/public_retrieval.pkl",
    ):
        if not p.is_file():
            continue
        obj = load_pickle(p)
        c = obj.get("cache", {})
        if all(q in c for q in ids):
            print(f"Historical public retrieval: {p}", flush=True)
            return c
    raise RuntimeError("No complete historical public retrieval cache")


def prepare_large_training(root, allq):
    from tune_burst_large_ltr import matrices

    qids = list(allq)
    train_ids = qids[:700] + qids[850:1150]
    tune_ids = qids[700:750]

    cache = load_pickle(
        root / "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl"
    )["cache"]

    train_features, trainX, trainy, train_groups = matrices(
        cache, allq, train_ids
    )
    tune_features, tuneX, tuney, tune_groups = matrices(
        cache, allq, tune_ids
    )
    return trainX, trainy, train_groups, tuneX, tuney, tune_groups


def build_public_large_feature_cache(root, comp, public, retrieval, cache_path):
    from tune_burst_score_ltr import score_features

    if cache_path.is_file():
        obj = load_pickle(cache_path)
        if obj.get("ids") == comp["ids"]:
            print(f"Reuse public large feature cache: {cache_path}", flush=True)
            return obj["features"], obj["candidates"]

    features, candidates = {}, {}
    for i, q in enumerate(comp["ids"], 1):
        cand, x = score_features(retrieval[q])
        candidates[q] = cand
        features[q] = x
        if i % 200 == 0:
            print(f"  public large features {i}/{len(comp['ids'])}", flush=True)

    save_pickle(
        cache_path,
        {"ids": comp["ids"], "features": features, "candidates": candidates},
    )
    return features, candidates


def robust_from_large_predictions(comp, large_rank):
    from tune_burst_pairwise import blend_rankings
    return {
        q: blend_rankings(
            {q: large_rank[q]},
            {q: comp["legal"][q][:10]},
            .275,
            40,
        )[q]
        for q in comp["ids"]
    }


def search_large(root, comp, target, allq, public, retrieval, out_dir):
    print("\n[A] Searching 6 historical XGBoost Large-LTR configs...", flush=True)

    trainX, trainy, train_groups, tuneX, tuney, tune_groups = (
        prepare_large_training(root, allq)
    )
    pub_features, pub_candidates = build_public_large_feature_cache(
        root,
        comp,
        public,
        retrieval,
        out_dir / "PUBLIC_LARGE_FEATURES.pkl",
    )

    trials = []
    best_model = None
    best_robust = None
    started = time.perf_counter()

    for idx, (depth, rate, trees, pairs) in enumerate(XGB_CONFIGS, 1):
        print(
            f"  XGB {idx}/6 depth={depth} rate={rate} "
            f"trees={trees} pairs={pairs}",
            flush=True,
        )
        model = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@5",
            tree_method="hist",
            n_estimators=trees,
            max_depth=depth,
            learning_rate=rate,
            min_child_weight=3,
            subsample=.85,
            colsample_bytree=.9,
            reg_lambda=5.0,
            n_jobs=2,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=pairs,
            random_state=2026,
        )
        model.fit(
            trainX,
            trainy,
            group=train_groups,
            eval_set=[(tuneX, tuney)],
            eval_group=[tune_groups],
            verbose=False,
        )

        large_rank = {}
        for q in comp["ids"]:
            s = model.predict(pub_features[q])
            order = np.argsort(-s)
            large_rank[q] = [pub_candidates[q][i] for i in order[:100]]

        robust = robust_from_large_predictions(comp, large_rank)
        pred = final_from_components(comp, robust=robust)
        m = metrics(pred, target, comp["ids"])
        item = {
            "config": {
                "depth": depth,
                "rate": rate,
                "trees": trees,
                "pairs": pairs,
            },
            "parity": m,
        }
        trials.append(item)
        print(
            f"    Top5={m['exact_top5_order']}/1000 "
            f"set={m['exact_top5_set']}/1000 "
            f"ov5={m['mean_top5_overlap']:.6f} "
            f"Top20={m['exact_top20_order']}/1000 "
            f"ov20={m['mean_top20_overlap']:.6f}",
            flush=True,
        )

        if best_model is None or key(m) > key(trials[0]["parity"]):
            pass

        # Save model/robust in temporary attrs by comparing after sorting.
        item["_model"] = model
        item["_robust"] = robust

    trials.sort(key=lambda x: key(x["parity"]), reverse=True)
    best = trials[0]
    best_model = best.pop("_model")
    best_robust = best.pop("_robust")
    for t in trials[1:]:
        t.pop("_model", None)
        t.pop("_robust", None)

    save_pickle(
        out_dir / "BEST_LARGE_OUTPUT_MATCH_MODEL.pkl",
        {
            "kind": "xgb",
            "params": best["config"],
            "model": best_model,
            "blend_alpha": .275,
            "blend_k": 40,
            "selection_basis": "unlabeled historical cpu_top20 output parity",
        },
    )
    save_pickle(
        out_dir / "BEST_LARGE_OUTPUT_MATCH_ROBUST.pkl",
        {"rankings": best_robust},
    )
    print(
        f"  BEST LARGE: {best['config']} "
        f"Top5={best['parity']['exact_top5_order']}/1000",
        flush=True,
    )
    return best, trials


def prepare_pair_training(root, allq):
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import features_for

    qids = list(allq)
    train_ids = qids[:700] + qids[850:1150]

    main = load_pickle(
        root / "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl"
    )["cache"]
    fresh1 = load_pickle(
        root / "results/burst_large_ltr/fresh_1151_1250_retrieval.pkl"
    )["cache"]
    fresh2 = load_pickle(
        root / "results/burst_large_ltr/fresh_1251_1350_retrieval.pkl"
    )["cache"]
    cache = main | fresh1 | fresh2

    freq = label_frequency(allq, set())
    train_features = features_for(
        cache, allq, train_ids, freq, True
    )
    return train_ids, train_features


def search_pair(root, comp, target, allq, public, retrieval, out_dir, base_robust):
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import (
        features_for,
        pairwise_matrix,
        rank,
    )

    print("\n[B] Searching 35 historical empirical-pairwise configs...", flush=True)

    train_ids, train_features = prepare_pair_training(root, allq)

    public_queries = {
        q: (public[q], set())
        for q in comp["ids"]
    }
    public_freq = label_frequency(allq, set())
    public_features = features_for(
        retrieval,
        public_queries,
        comp["ids"],
        public_freq,
    )

    trials = []
    best_model = None
    best_scaler = None
    best_pair_rank = None

    for depth in PAIR_DEPTHS:
        print(f"  building pair matrix depth={depth}...", flush=True)
        x, y, weights = pairwise_matrix(
            train_features, allq, train_ids, depth
        )
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)

        for c in PAIR_CS:
            model = LogisticRegression(
                C=c,
                solver="liblinear",
                max_iter=2000,
            )
            model.fit(z, y, sample_weight=weights)

            pair_rank = rank(
                model,
                scaler,
                public_features,
                comp["ids"],
            )
            pred = final_from_components(
                comp,
                robust=base_robust,
                pair=pair_rank,
            )
            m = metrics(pred, target, comp["ids"])
            item = {
                "config": {
                    "negative_depth": depth,
                    "C": c,
                },
                "parity": m,
                "_model": model,
                "_scaler": scaler,
                "_pair": pair_rank,
            }
            trials.append(item)
            print(
                f"    depth={depth:2d} C={c:<5} "
                f"Top5={m['exact_top5_order']}/1000 "
                f"set={m['exact_top5_set']}/1000 "
                f"ov5={m['mean_top5_overlap']:.6f} "
                f"ov20={m['mean_top20_overlap']:.6f}",
                flush=True,
            )

    trials.sort(key=lambda x: key(x["parity"]), reverse=True)
    best = trials[0]
    best_model = best.pop("_model")
    best_scaler = best.pop("_scaler")
    best_pair_rank = best.pop("_pair")
    for t in trials[1:]:
        t.pop("_model", None)
        t.pop("_scaler", None)
        t.pop("_pair", None)

    save_pickle(
        out_dir / "BEST_PAIR_OUTPUT_MATCH_MODEL.pkl",
        {
            "model": best_model,
            "scaler": best_scaler,
            "feature_version": "empirical-pairwise-v1",
            "config": best["config"],
            "selection_basis": "unlabeled historical cpu_top20 output parity",
        },
    )
    save_pickle(
        out_dir / "BEST_PAIR_OUTPUT_MATCH_RANKINGS.pkl",
        {"rankings": best_pair_rank},
    )
    print(
        f"  BEST PAIR: {best['config']} "
        f"Top5={best['parity']['exact_top5_order']}/1000",
        flush=True,
    )
    return best, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from run_burst_multistage_submission import load_metadata
    from tune_burst_kernel_posterior import load_queries

    out_dir = root / "results/manual/huy_private_d1_rel_l0_v1"
    comp_path = out_dir / "CPU_OUTPUT_MATCH_COMPONENTS.pkl"
    if not comp_path.is_file():
        raise FileNotFoundError(
            f"Run search_cpu_top20_output_match_v1.py first: {comp_path}"
        )
    comp = load_pickle(comp_path)

    target = load_pickle(
        root / "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]
    if set(comp["ids"]) != set(target):
        raise RuntimeError("Component/target qid population mismatch")

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    _, _, _, public = load_metadata(data)
    allq = load_queries(root)
    retrieval = load_public_retrieval(root, comp["ids"])

    current_pred = final_from_components(comp)
    current_m = metrics(current_pred, target, comp["ids"])
    print(
        f"CURRENT baseline: Top5={current_m['exact_top5_order']}/1000 "
        f"ov5={current_m['mean_top5_overlap']:.6f} "
        f"ov20={current_m['mean_top20_overlap']:.6f}",
        flush=True,
    )

    best_large, large_trials = search_large(
        root, comp, target, allq, public, retrieval, out_dir
    )

    recovered_robust = load_pickle(
        out_dir / "BEST_LARGE_OUTPUT_MATCH_ROBUST.pkl"
    )["rankings"]

    # Pair search is evaluated twice:
    #   1) with CURRENT robust (isolates pair only)
    #   2) with best recovered-large robust (tests joint explanation).
    best_pair_current, pair_current_trials = search_pair(
        root,
        comp,
        target,
        allq,
        public,
        retrieval,
        out_dir,
        comp["robust"],
    )

    # Re-evaluate the already fitted best pair ranking jointly with recovered robust.
    best_pair_rank = load_pickle(
        out_dir / "BEST_PAIR_OUTPUT_MATCH_RANKINGS.pkl"
    )["rankings"]
    joint_pred = final_from_components(
        comp,
        robust=recovered_robust,
        pair=best_pair_rank,
    )
    joint_m = metrics(joint_pred, target, comp["ids"])

    if joint_m["exact_top20_order"] == len(comp["ids"]):
        verdict = "EXACT_LOST_MODELS_RECOVERED"
    elif joint_m["exact_top5_order"] >= 950:
        verdict = "LOST_MODELS_EXPLAIN_MOST_DRIFT"
    elif joint_m["exact_top5_order"] > current_m["exact_top5_order"] + 100:
        verdict = "LOST_MODELS_EXPLAIN_SUBSTANTIAL_DRIFT"
    elif joint_m["exact_top5_order"] > current_m["exact_top5_order"]:
        verdict = "LOST_MODELS_EXPLAIN_PARTIAL_DRIFT"
    else:
        verdict = "LOST_MODEL_GRIDS_DO_NOT_EXPLAIN_DRIFT"

    report = {
        "schema": "manual.lost_cpu_models_output_match_v1",
        "labels_used_for_output_selection": False,
        "historical_target": "results/burst_gpu_threeview/cpu_top20.pkl",
        "current": current_m,
        "best_large": best_large,
        "large_trials": [
            {"config": t["config"], "parity": t["parity"]}
            for t in large_trials
        ],
        "best_pair_isolated_current_robust": best_pair_current,
        "pair_trials": [
            {"config": t["config"], "parity": t["parity"]}
            for t in pair_current_trials
        ],
        "joint_best_large_plus_best_pair": joint_m,
        "verdict": verdict,
    }
    report_path = out_dir / "LOST_MODEL_OUTPUT_MATCH_V1.json"
    dump(report_path, report)

    print("=" * 108)
    print("LOST CPU MODEL OUTPUT MATCH")
    print(
        f"CURRENT      Top5={current_m['exact_top5_order']}/1000 "
        f"ov5={current_m['mean_top5_overlap']:.6f} "
        f"ov20={current_m['mean_top20_overlap']:.6f}"
    )
    print(
        f"BEST LARGE   {best_large['config']} | "
        f"Top5={best_large['parity']['exact_top5_order']}/1000 "
        f"ov5={best_large['parity']['mean_top5_overlap']:.6f}"
    )
    print(
        f"BEST PAIR    {best_pair_current['config']} | "
        f"Top5={best_pair_current['parity']['exact_top5_order']}/1000 "
        f"ov5={best_pair_current['parity']['mean_top5_overlap']:.6f}"
    )
    print(
        f"JOINT        Top5={joint_m['exact_top5_order']}/1000 "
        f"set={joint_m['exact_top5_set']}/1000 "
        f"ov5={joint_m['mean_top5_overlap']:.6f} "
        f"Top20={joint_m['exact_top20_order']}/1000 "
        f"ov20={joint_m['mean_top20_overlap']:.6f}"
    )
    print("VERDICT:", verdict)
    print("Report:", report_path)
    print("=" * 108)


if __name__ == "__main__":
    main()
