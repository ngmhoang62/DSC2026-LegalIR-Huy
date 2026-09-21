#!/usr/bin/env python
"""
FINAL ATTEMPT: RECOVER HISTORICAL CPU BASE GENERATOR
====================================================

Why this script exists
----------------------
We already established:

1) The final multistage fusion itself is correct:
       weights = [robust=.40, pair=.30, profile=.15, graph=.15]
       rrf_k = 0

2) Rebuilding empirical-pairwise does NOT explain the drift.

3) Searching Large-LTR alone improved historical output parity, but the
   previous forensic script made one important approximation:
       it changed ROBUST while keeping GRAPH frozen.

   That is not the real production contract. In production:
       graph = graph_rerank(robust, ...)
   so every candidate robust ranking must generate its own graph branch.

This is therefore the LAST scientifically justified rebuild attempt.

Search space
------------
Historical Large-LTR XGBoost configs: 6
Historical Legal-LTR C values:       5
Historical Large/Legal RRF blend:
    alpha in np.linspace(0,1,21)
    k in {0,5,20,60}

Total robust candidates = 6 * 5 * 21 * 4 = 2520.

For EVERY candidate:
    large candidate
       +
    legal candidate
       -> robust candidate
       -> RECOMPUTE graph from that robust candidate
       -> fixed final stack:
            robust .40
            pair   .30
            profile .15
            graph   .15
            k=0

Pair/profile remain frozen because pairwise output-matching already showed
that the historical pairwise grid did not explain the drift.

Selection target
----------------
Only the unlabeled historical artifact:
    results/burst_gpu_threeview/cpu_top20.pkl

No public labels / leaderboard metrics are read.

Two-stage search
----------------
Stage A:
    all 2520 configs on a deterministic ~250-query sample.

Stage B:
    top N configs on all 1000 public queries.

Outputs
-------
results/manual/huy_private_d1_rel_l0_v1/final_cpu_recovery/
    LARGE_PUBLIC_RANKINGS.pkl
    LEGAL_PUBLIC_RANKINGS.pkl
    PUBLIC_LEGAL_FEATURES_DEPTH20.pkl
    FINAL_CPU_RECOVERY_SEARCH.json
    BEST_RECOVERED_PUBLIC_CPU_TOP20.pkl
    BEST_RECOVERED_LARGE_MODEL.pkl
    BEST_RECOVERED_LEGAL_MODEL.pkl
    BEST_RECOVERED_CONFIG.json

Interpretation
--------------
If this still cannot get close to the historical artifact, stop trying to
recover the exact private D1 base generator. Treat the original cpu_top20
generator as unrecoverable and do not spend more endgame time on it.

Run
---
cd /d/Study/DSC2026/sota
source dsc_env_huy/Scripts/activate

python ../final_rebuild_cpu_generator_v1.py \
  --repo-root /d/Study/DSC2026/sota \
  --sample 250 \
  --topn-full 30
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
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

LEGAL_CS = (.03, .1, .2, .5, 1.0)
ROBUST_ALPHAS = tuple(float(x) for x in np.linspace(0.0, 1.0, 21))
ROBUST_KS = (0, 5, 20, 60)

GRAPH_PARAMS = {
    "seed_depth": 3,
    "seed_power": .5,
    "mode": "conditional",
    "min_edge": 3,
    "alpha": .4,
    "base_k": 0,
}


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


def output_metrics(pred, target, qids):
    exact5 = exact20 = set5 = 0
    ov5 = ov20 = pos5 = 0.0
    for q in qids:
        a = pred[q]
        b = target[q]
        exact5 += int(a[:5] == b[:5])
        exact20 += int(a[:20] == b[:20])
        set5 += int(set(a[:5]) == set(b[:5]))
        ov5 += len(set(a[:5]) & set(b[:5])) / 5.0
        ov20 += len(set(a[:20]) & set(b[:20])) / 20.0
        pos5 += sum(x == y for x, y in zip(a[:5], b[:5])) / 5.0
    n = len(qids)
    return {
        "queries": n,
        "exact_top5_order": exact5,
        "exact_top5_set": set5,
        "mean_top5_overlap": ov5 / n,
        "mean_top5_positional": pos5 / n,
        "exact_top20_order": exact20,
        "mean_top20_overlap": ov20 / n,
    }


def metric_key(m):
    return (
        m["exact_top5_order"],
        m["exact_top5_set"],
        m["mean_top5_overlap"],
        m["mean_top20_overlap"],
        m["exact_top20_order"],
        m["mean_top5_positional"],
    )


def rrf_two(a, b, alpha, k):
    ra = {d: i + 1 for i, d in enumerate(a)}
    rb = {d: i + 1 for i, d in enumerate(b)}
    docs = set(ra) | set(rb)
    return sorted(
        docs,
        key=lambda d: (
            -(
                alpha / (k + ra.get(d, 100000))
                + (1.0 - alpha) / (k + rb.get(d, 100000))
            ),
            d,
        ),
    )[:100]


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


def load_public_retrieval(root: Path, ids):
    for p in (
        root / "results/burst_multistage/public_retrieval.pkl",
        root / "results/burst_robust_fusion/public_retrieval.pkl",
        root / "results/burst_expanded_fusion/public_retrieval.pkl",
    ):
        if not p.is_file():
            continue
        obj = load_pickle(p)
        cache = obj.get("cache", {})
        if all(q in cache for q in ids):
            print(f"Historical public retrieval: {p}", flush=True)
            return cache, p
    raise RuntimeError("No complete historical public retrieval cache found")


def prepare_large_training(root, allq):
    from tune_burst_large_ltr import matrices

    qids = list(allq)
    train_ids = qids[:700] + qids[850:1150]
    tune_ids = qids[700:750]

    cache_path = (
        root
        / "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl"
    )
    saved = load_pickle(cache_path)
    expected = train_ids + tune_ids + qids[750:850]
    if saved.get("qids") != expected:
        raise RuntimeError("Large-LTR training retrieval split drift")
    cache = saved["cache"]

    _, trainX, trainy, train_groups = matrices(
        cache, allq, train_ids
    )
    _, tuneX, tuney, tune_groups = matrices(
        cache, allq, tune_ids
    )
    return trainX, trainy, train_groups, tuneX, tuney, tune_groups


def build_public_large_features(root, retrieval, ids, out_path):
    from tune_burst_score_ltr import score_features

    if out_path.is_file():
        obj = load_pickle(out_path)
        if obj.get("ids") == ids:
            print(f"Reuse {out_path.name}", flush=True)
            return obj["candidates"], obj["features"]

    candidates = {}
    features = {}
    for i, q in enumerate(ids, 1):
        c, x = score_features(retrieval[q])
        candidates[q] = c
        features[q] = x
        if i % 200 == 0 or i == len(ids):
            print(f"  large public features {i}/{len(ids)}", flush=True)

    save_pickle(
        out_path,
        {
            "ids": ids,
            "candidates": candidates,
            "features": features,
        },
    )
    return candidates, features


def fit_all_large_models(
    root,
    allq,
    public_candidates,
    public_features,
    ids,
    out_dir,
):
    cache_path = out_dir / "LARGE_PUBLIC_RANKINGS.pkl"
    if cache_path.is_file():
        obj = load_pickle(cache_path)
        if obj.get("ids") == ids and len(obj.get("rankings", {})) == len(XGB_CONFIGS):
            print("Reuse all Large-LTR public rankings.", flush=True)
            return obj

    trainX, trainy, train_groups, tuneX, tuney, tune_groups = (
        prepare_large_training(root, allq)
    )

    rankings = {}
    model_blobs = {}
    for idx, (depth, rate, trees, pairs) in enumerate(XGB_CONFIGS):
        key = f"xgb_{idx}"
        print(
            f"Fit Large {idx+1}/{len(XGB_CONFIGS)}: "
            f"d={depth} lr={rate} trees={trees} pairs={pairs}",
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

        r = {}
        for q in ids:
            s = model.predict(public_features[q])
            order = np.argsort(-s)
            r[q] = [
                public_candidates[q][i]
                for i in order[:100]
            ]
        rankings[key] = r
        model_blobs[key] = pickle.dumps(model, protocol=5)

    obj = {
        "ids": ids,
        "configs": {
            f"xgb_{i}": {
                "depth": cfg[0],
                "rate": cfg[1],
                "trees": cfg[2],
                "pairs": cfg[3],
            }
            for i, cfg in enumerate(XGB_CONFIGS)
        },
        "rankings": rankings,
        "model_blobs": model_blobs,
    }
    save_pickle(cache_path, obj)
    return obj


def normalize_corpus(paths):
    from benchmark_burst_v4_full_sqlite import tokens

    normalized = {}
    print("Normalize corpus for Legal-LTR...", flush=True)
    for i, p in enumerate(paths, 1):
        row = json.loads(p.read_text(encoding="utf-8"))
        did = str(row["id"])
        normalized[did] = (
            " " + " ".join(tokens(row.get("passage") or "")) + " "
        )
        if i % 1500 == 0 or i == len(paths):
            print(f"  normalized {i}/{len(paths)}", flush=True)
    return normalized


def build_public_legal_features(
    root,
    paths,
    public,
    retrieval,
    ids,
    out_path,
):
    from tune_burst_legal_features import enhanced_features

    if out_path.is_file():
        obj = load_pickle(out_path)
        if obj.get("ids") == ids:
            print(f"Reuse {out_path.name}", flush=True)
            return obj["features"]

    normalized = normalize_corpus(paths)
    features = {}
    print("Build production Legal-LTR features (lexical_depth=20)...", flush=True)
    for i, q in enumerate(ids, 1):
        features[q] = enhanced_features(
            retrieval[q],
            public[q],
            normalized,
            lexical_depth=20,
        )
        if i % 100 == 0 or i == len(ids):
            print(f"  legal public features {i}/{len(ids)}", flush=True)

    save_pickle(
        out_path,
        {"ids": ids, "features": features},
    )
    return features


def fit_all_legal_models(root, public_features, ids, out_dir):
    from tune_burst_legal_features import fit_cached

    cache_path = out_dir / "LEGAL_PUBLIC_RANKINGS.pkl"
    if cache_path.is_file():
        obj = load_pickle(cache_path)
        if obj.get("ids") == ids and len(obj.get("rankings", {})) == len(LEGAL_CS):
            print("Reuse all Legal-LTR public rankings.", flush=True)
            return obj

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    raw = json.loads((data / "train.json").read_text(encoding="utf-8"))
    allq = {
        str(q): (
            str(v["question"]),
            {str(d) for d in v["answer"]},
        )
        for q, v in raw.items()
        if v.get("answer")
    }
    qids = list(allq)
    train_ids = qids[400:700]
    gold = {q: allq[q][1] for q in allq}

    train_feature_obj = load_pickle(
        root / "results/burst_legal_features/features_401_850.pkl"
    )
    expected = qids[400:850]
    if train_feature_obj.get("qids") != expected:
        raise RuntimeError("Legal feature-cache split drift")
    train_features = train_feature_obj["features"]

    rankings = {}
    model_blobs = {}
    for i, c in enumerate(LEGAL_CS):
        key = f"legal_{i}"
        print(f"Fit Legal {i+1}/{len(LEGAL_CS)} C={c}", flush=True)
        model = fit_cached(
            train_features,
            gold,
            train_ids,
            c,
        )
        r = {}
        for q in ids:
            cand, x = public_features[q]
            s = model.decision_function(x)
            order = np.argsort(-s)
            r[q] = [cand[j] for j in order[:100]]
        rankings[key] = r
        model_blobs[key] = pickle.dumps(model, protocol=5)

    obj = {
        "ids": ids,
        "configs": {
            f"legal_{i}": {"C": c}
            for i, c in enumerate(LEGAL_CS)
        },
        "rankings": rankings,
        "model_blobs": model_blobs,
    }
    save_pickle(cache_path, obj)
    return obj


def build_graph_state(train):
    from tune_burst_graph_posterior import build_graph
    train_ids = list(train)
    return build_graph(train, train_ids)[:2]


def evaluate_candidate(
    *,
    qids,
    target,
    large_rank,
    legal_rank,
    pair_rank,
    profile_rank,
    graph_state,
    robust_alpha,
    robust_k,
):
    from tune_burst_graph_posterior import graph_rerank

    frequency, adjacency = graph_state
    pred = {}

    for q in qids:
        robust = rrf_two(
            large_rank[q],
            legal_rank[q][:10],
            robust_alpha,
            robust_k,
        )
        graph = graph_rerank(
            robust,
            frequency,
            adjacency,
            GRAPH_PARAMS["seed_depth"],
            GRAPH_PARAMS["seed_power"],
            GRAPH_PARAMS["mode"],
            GRAPH_PARAMS["min_edge"],
            GRAPH_PARAMS["alpha"],
            GRAPH_PARAMS["base_k"],
        )
        pred[q] = weighted_rrf_one([
            robust,
            pair_rank[q],
            profile_rank[q],
            graph,
        ])

    return pred, output_metrics(pred, target, qids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--topn-full", type=int, default=30)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from run_burst_multistage_submission import load_metadata
    from tune_burst_kernel_posterior import load_queries

    base_out = root / "results/manual/huy_private_d1_rel_l0_v1"
    out_dir = base_out / "final_cpu_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_path = base_out / "CPU_OUTPUT_MATCH_COMPONENTS.pkl"
    if not comp_path.is_file():
        raise FileNotFoundError(
            f"Missing {comp_path}. Run search_cpu_top20_output_match_v1.py first."
        )
    comp = load_pickle(comp_path)
    ids = list(comp["ids"])

    target = load_pickle(
        root / "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    paths, _, train, public = load_metadata(data)
    allq = load_queries(root)

    retrieval, retrieval_path = load_public_retrieval(root, ids)

    print("[1/6] Build/reuse public Large-LTR feature matrices...")
    pub_large_cands, pub_large_feats = build_public_large_features(
        root,
        retrieval,
        ids,
        out_dir / "PUBLIC_LARGE_FEATURES.pkl",
    )

    print("[2/6] Fit/reuse all 6 historical Large-LTR candidates...")
    large_obj = fit_all_large_models(
        root,
        allq,
        pub_large_cands,
        pub_large_feats,
        ids,
        out_dir,
    )

    print("[3/6] Build/reuse production Legal-LTR feature matrices...")
    pub_legal_feats = build_public_legal_features(
        root,
        paths,
        public,
        retrieval,
        ids,
        out_dir / "PUBLIC_LEGAL_FEATURES_DEPTH20.pkl",
    )

    print("[4/6] Fit/reuse all 5 historical Legal-LTR candidates...")
    legal_obj = fit_all_legal_models(
        root,
        pub_legal_feats,
        ids,
        out_dir,
    )

    print("[5/6] Stage-A joint Large × Legal × robust blend search...")
    graph_state = build_graph_state(train)

    n_sample = min(args.sample, len(ids))
    sample_idx = np.linspace(
        0,
        len(ids) - 1,
        n_sample,
        dtype=int,
    )
    sample_ids = [ids[i] for i in sample_idx]

    combos = []
    for lk in large_obj["rankings"]:
        for gek in legal_obj["rankings"]:
            for alpha in ROBUST_ALPHAS:
                for k in ROBUST_KS:
                    combos.append((lk, gek, alpha, k))

    print(
        f"  candidates={len(combos)} | sample={len(sample_ids)}",
        flush=True,
    )

    trials = []
    started = time.perf_counter()

    for i, (lk, gek, alpha, k) in enumerate(combos, 1):
        _, m = evaluate_candidate(
            qids=sample_ids,
            target=target,
            large_rank=large_obj["rankings"][lk],
            legal_rank=legal_obj["rankings"][gek],
            pair_rank=comp["pair"],
            profile_rank=comp["profile"],
            graph_state=graph_state,
            robust_alpha=alpha,
            robust_k=k,
        )
        trials.append({
            "large_key": lk,
            "large_config": large_obj["configs"][lk],
            "legal_key": gek,
            "legal_config": legal_obj["configs"][gek],
            "robust_alpha": alpha,
            "robust_k": k,
            "sample": m,
        })

        if i % 100 == 0 or i == len(combos):
            best_now = max(
                trials,
                key=lambda x: metric_key(x["sample"]),
            )
            print(
                f"    {i}/{len(combos)} "
                f"best_sample_Top5="
                f"{best_now['sample']['exact_top5_order']}/{len(sample_ids)} "
                f"({time.perf_counter()-started:.1f}s)",
                flush=True,
            )

    trials.sort(
        key=lambda x: metric_key(x["sample"]),
        reverse=True,
    )

    print(
        f"[6/6] Full-1000 validation of top {args.topn_full} candidates...",
        flush=True,
    )
    finalists = []

    for rank_i, t in enumerate(trials[:args.topn_full], 1):
        pred, m = evaluate_candidate(
            qids=ids,
            target=target,
            large_rank=large_obj["rankings"][t["large_key"]],
            legal_rank=legal_obj["rankings"][t["legal_key"]],
            pair_rank=comp["pair"],
            profile_rank=comp["profile"],
            graph_state=graph_state,
            robust_alpha=t["robust_alpha"],
            robust_k=t["robust_k"],
        )
        item = {
            **t,
            "full": m,
        }
        finalists.append((item, pred))
        print(
            f"  #{rank_i:02d} "
            f"L={t['large_key']} {t['large_config']} | "
            f"Legal={t['legal_config']} | "
            f"a={t['robust_alpha']:.2f} k={t['robust_k']} | "
            f"Top5={m['exact_top5_order']}/1000 "
            f"set={m['exact_top5_set']}/1000 "
            f"ov5={m['mean_top5_overlap']:.6f} "
            f"Top20={m['exact_top20_order']}/1000 "
            f"ov20={m['mean_top20_overlap']:.6f}",
            flush=True,
        )

    finalists.sort(
        key=lambda x: metric_key(x[0]["full"]),
        reverse=True,
    )
    best, best_pred = finalists[0]

    current_pred = {}
    for q in ids:
        current_pred[q] = weighted_rrf_one([
            comp["robust"][q],
            comp["pair"][q],
            comp["profile"][q],
            comp["graph"][q],
        ])
    current = output_metrics(current_pred, target, ids)

    # Save recovered model objects for the best candidate.
    best_large_blob = large_obj["model_blobs"][best["large_key"]]
    best_legal_blob = legal_obj["model_blobs"][best["legal_key"]]

    save_pickle(
        out_dir / "BEST_RECOVERED_PUBLIC_CPU_TOP20.pkl",
        {"rankings": best_pred},
    )
    (out_dir / "BEST_RECOVERED_LARGE_MODEL.pkl").write_bytes(
        best_large_blob
    )
    (out_dir / "BEST_RECOVERED_LEGAL_MODEL.pkl").write_bytes(
        best_legal_blob
    )
    dump(
        out_dir / "BEST_RECOVERED_CONFIG.json",
        {
            "large_key": best["large_key"],
            "large_config": best["large_config"],
            "legal_key": best["legal_key"],
            "legal_config": best["legal_config"],
            "robust_alpha": best["robust_alpha"],
            "robust_k": best["robust_k"],
            "graph_params": GRAPH_PARAMS,
            "final_weights": FINAL_WEIGHTS,
            "final_k": FINAL_K,
            "full_parity": best["full"],
        },
    )

    # Deliberately strict endgame verdict.
    bm = best["full"]
    if bm["exact_top20_order"] == len(ids):
        verdict = "SUCCESS_EXACT_HISTORICAL_CPU_RECOVERED"
    elif (
        bm["exact_top5_order"] >= 950
        and bm["mean_top5_overlap"] >= .99
        and bm["mean_top20_overlap"] >= .98
    ):
        verdict = "SUCCESS_HIGH_PARITY_RECOVERY"
    else:
        verdict = "FINAL_RECOVERY_FAILED_STOP_REBUILDING"

    report = {
        "schema": "manual.final_cpu_recovery_search_v1",
        "labels_used_for_output_selection": False,
        "retrieval_path": str(retrieval_path),
        "historical_target": (
            "results/burst_gpu_threeview/cpu_top20.pkl"
        ),
        "search": {
            "large_configs": len(XGB_CONFIGS),
            "legal_configs": len(LEGAL_CS),
            "robust_alphas": list(ROBUST_ALPHAS),
            "robust_ks": list(ROBUST_KS),
            "total_candidates": len(combos),
            "sample_queries": len(sample_ids),
            "full_finalists": len(finalists),
            "graph_recomputed_for_every_candidate": True,
        },
        "current_rebuild": current,
        "best": best,
        "top_full": [x[0] for x in finalists[:20]],
        "top_sample": trials[:30],
        "verdict": verdict,
        "stop_rule": (
            "If verdict is FINAL_RECOVERY_FAILED_STOP_REBUILDING, "
            "do not spend further endgame time reconstructing historical cpu_top20."
        ),
    }
    report_path = out_dir / "FINAL_CPU_RECOVERY_SEARCH.json"
    dump(report_path, report)

    print("=" * 112)
    print("FINAL HISTORICAL CPU RECOVERY ATTEMPT")
    print(
        f"CURRENT Top5={current['exact_top5_order']}/1000 "
        f"ov5={current['mean_top5_overlap']:.6f} "
        f"ov20={current['mean_top20_overlap']:.6f}"
    )
    print(
        "BEST LARGE:", best["large_config"],
        "| LEGAL:", best["legal_config"],
        f"| robust alpha={best['robust_alpha']:.2f} k={best['robust_k']}"
    )
    print(
        f"BEST Top5={bm['exact_top5_order']}/1000 "
        f"set={bm['exact_top5_set']}/1000 "
        f"ov5={bm['mean_top5_overlap']:.6f} "
        f"Top20={bm['exact_top20_order']}/1000 "
        f"ov20={bm['mean_top20_overlap']:.6f}"
    )
    print("VERDICT:", verdict)
    print("Report:", report_path)
    print("=" * 112)


if __name__ == "__main__":
    main()
