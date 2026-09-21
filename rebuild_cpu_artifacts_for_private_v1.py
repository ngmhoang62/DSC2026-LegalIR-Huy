#!/usr/bin/env python
"""
REBUILD HISTORICAL CPU BURST ARTIFACTS FOR PRIVATE D1
=====================================================

Rebuilds the three missing CPU artifacts required by the documented
BURST-MultistagePosterior base ranking:

  results/burst_legal_features/validation_model.pkl
  results/burst_large_ltr/best_model.pkl
  results/burst_empirical_pairwise/model.pkl

It also rebuilds their required retrieval/feature caches from the canonical
train.json using the existing FTS5 DB.

Crucially, after rebuilding, it reconstructs the PUBLIC CPU Top20 and compares
it against the historical authoritative cache:

  results/burst_gpu_threeview/cpu_top20.pkl

Only an exact 1000/1000 ordered Top20 match should be treated as a trusted
reconstruction for private inference.

This script is CPU-only except XGBoost CPU hist training.

Run:
  cd /d/Study/DSC2026/sota
  source dsc_env_huy/Scripts/activate

  python ../rebuild_cpu_artifacts_for_private_v1.py \
    --repo-root /d/Study/DSC2026/sota \
    --workers 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=5))
    tmp.replace(path)


def load_pickle(path: Path):
    return pickle.loads(path.read_bytes())


def resolve_db(root: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.resolve())
    candidates += [
        root / "results/manual/huy_private_d1_rel_l0_v1/cache/legalir_full_fts.sqlite",
        root / "benchmarks/legalir_full_fts.sqlite",
        root.parent / "LegalIR/benchmarks/legalir_full_fts.sqlite",
    ]
    for p in candidates:
        p = p.resolve()
        if not p.is_file():
            continue
        try:
            con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
            meta = dict(con.execute("SELECT key,value FROM metadata").fetchall())
            con.close()
            if (
                int(meta.get("documents", -1)) == 8532
                and int(meta.get("chunk_size", -1)) == 500
                and int(meta.get("overlap", -1)) == 100
            ):
                print(
                    f"FTS DB: {p} | docs={meta['documents']} "
                    f"chunks={meta.get('chunks')} chunk=500 overlap=100",
                    flush=True,
                )
                return p
        except Exception:
            pass
    raise FileNotFoundError(
        "No valid FTS DB found. Expected the private runner's cached DB at:\n"
        f"  {root / 'results/manual/huy_private_d1_rel_l0_v1/cache/legalir_full_fts.sqlite'}"
    )


def retrieve_block(
    *,
    db_path: Path,
    doc_ids,
    allq,
    qids,
    excluded,
    cache_path: Path,
    workers: int,
):
    from tune_burst_memory import build_query_memory
    from tune_burst_score_ltr import retrieve

    if cache_path.is_file():
        obj = load_pickle(cache_path)
        if obj.get("qids") == qids:
            cache = obj.get("cache", {})
            if len(cache) == len(qids):
                print(f"  reuse {cache_path}: {len(cache)}/{len(qids)}", flush=True)
                return cache

    memory = {q: v for q, v in allq.items() if q not in set(excluded)}
    con = sqlite3.connect(str(db_path))
    build_query_memory(con, memory)
    con.close()

    local = threading.local()

    def one(q):
        if not hasattr(local, "conn"):
            local.conn = sqlite3.connect(str(db_path))
        return q, retrieve(local.conn, doc_ids, q, allq[q][0])

    cache = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (q, result) in enumerate(pool.map(one, qids), 1):
            cache[q] = result
            if i % 25 == 0 or i == len(qids):
                save_pickle(cache_path, {"qids": qids, "cache": cache})
                print(
                    f"    retrieve {i}/{len(qids)} "
                    f"({time.perf_counter()-started:.1f}s)",
                    flush=True,
                )
    return cache


def normalize_docs(docs):
    from benchmark_burst_v4_full_sqlite import tokens
    print("Normalizing corpus once...", flush=True)
    out = {}
    for i, (docid, text) in enumerate(docs, 1):
        out[docid] = " " + " ".join(tokens(text or "")) + " "
        if i % 1500 == 0 or i == len(docs):
            print(f"  normalized {i}/{len(docs)}", flush=True)
    return out


def build_legal_model(root, allq, cache_401_850, normalized):
    from tune_burst_legal_features import enhanced_features, fit_cached, rank_cached
    from tune_burst_pairwise import fixed_metrics

    qids = list(allq)
    train_ids = qids[400:700]
    tune_ids = qids[700:750]
    val_ids = qids[750:850]
    all_ids = train_ids + tune_ids + val_ids

    out_dir = root / "results/burst_legal_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "features_401_850.pkl"

    feature_cache = {}
    if feature_path.is_file():
        obj = load_pickle(feature_path)
        if obj.get("qids") == all_ids:
            feature_cache = obj.get("features", {})
    if len(feature_cache) != len(all_ids):
        feature_cache = {}
        questions = {q: allq[q][0] for q in all_ids}
        print("Building legal lexical features (450 queries)...", flush=True)
        for i, q in enumerate(all_ids, 1):
            feature_cache[q] = enhanced_features(
                cache_401_850[q],
                questions[q],
                normalized,
            )
            if i % 25 == 0 or i == len(all_ids):
                print(f"  legal features {i}/{len(all_ids)}", flush=True)
        save_pickle(
            feature_path,
            {"qids": all_ids, "features": feature_cache},
        )
    else:
        print(f"Reuse legal features: {feature_path}", flush=True)

    gold = {q: allq[q][1] for q in all_ids}
    tuneq = {q: allq[q] for q in tune_ids}
    trials = []
    for C in (.03, .1, .2, .5, 1.0):
        model = fit_cached(feature_cache, gold, train_ids, C)
        ranked = rank_cached(model, feature_cache, tune_ids)
        m, _ = fixed_metrics(ranked, tuneq)
        trials.append(
            (m["Recall@5"], m["Precision@5"], m["nDCG@10"], C, model, m)
        )
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, best_C, best_model, tune_m = trials[0]

    save_pickle(
        out_dir / "validation_model.pkl",
        {
            "C": best_C,
            "model": best_model,
            "train": "401-700",
            "feature_version": "legal-v1",
        },
    )
    valq = {q: allq[q] for q in val_ids}
    vrank = rank_cached(best_model, feature_cache, val_ids)
    vm, _ = fixed_metrics(vrank, valq)
    print(
        f"LEGAL rebuilt: C={best_C} "
        f"tuneR={tune_m['Recall@5']:.6f} valR={vm['Recall@5']:.6f}",
        flush=True,
    )
    return best_model, feature_cache


def large_rank(model, kind, features, ids):
    from tune_burst_large_ltr import rank_linear, rank_xgb
    return rank_xgb(model, features, ids) if kind == "xgb" else rank_linear(model, features, ids)


def build_large_model(root, allq, cache_main, legal_model, legal_features):
    from tune_burst_large_ltr import matrices, hard_matrix, rank_linear, rank_xgb
    from tune_burst_legal_features import rank_cached
    from tune_burst_pairwise import blend_rankings, fixed_metrics

    qids = list(allq)
    train_ids = qids[:700] + qids[850:1150]
    tune_ids = qids[700:750]
    val_ids = qids[750:850]

    train_features, trainX, trainy, train_groups = matrices(
        cache_main, allq, train_ids
    )
    tune_features, tuneX, tuney, tune_groups = matrices(
        cache_main, allq, tune_ids
    )
    val_features, _, _, _ = matrices(cache_main, allq, val_ids)
    tuneq = {q: allq[q] for q in tune_ids}
    trials = []

    print("Tuning large LTR...", flush=True)
    for C in (.01, .03, .1, .2, .5, 1.0, 2.0):
        model = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1500,
        ).fit(trainX, trainy)
        ranked = rank_linear(model, tune_features, tune_ids)
        m, _ = fixed_metrics(ranked, tuneq)
        trials.append(
            (m["Recall@5"], m["Precision@5"], m["nDCG@10"],
             "logistic", {"C": C}, model, m)
        )

    for depth in (40, 80, 150):
        X, y, w = hard_matrix(cache_main, train_features, allq, train_ids, depth)
        for C in (.1, .3, 1.0, 3.0):
            model = LogisticRegression(
                C=C, solver="liblinear", max_iter=1500
            ).fit(X, y, sample_weight=w)
            ranked = rank_linear(model, tune_features, tune_ids)
            m, _ = fixed_metrics(ranked, tuneq)
            trials.append(
                (m["Recall@5"], m["Precision@5"], m["nDCG@10"],
                 "hard_logistic", {"depth": depth, "C": C}, model, m)
            )

    xgb_configs = [
        (2, .03, 180, 10),
        (2, .05, 140, 10),
        (3, .03, 200, 10),
        (3, .05, 160, 10),
        (4, .03, 180, 10),
        (3, .03, 220, 20),
    ]
    for depth, rate, trees, pairs in xgb_configs:
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
        ranked = rank_xgb(model, tune_features, tune_ids)
        m, _ = fixed_metrics(ranked, tuneq)
        trials.append(
            (m["Recall@5"], m["Precision@5"], m["nDCG@10"],
             "xgb",
             {"depth": depth, "rate": rate, "trees": trees, "pairs": pairs},
             model, m)
        )

    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, kind, params, best, tune_m = trials[0]

    tune_legal = rank_cached(legal_model, legal_features, tune_ids)
    val_legal = rank_cached(legal_model, legal_features, val_ids)
    tune_new = large_rank(best, kind, tune_features, tune_ids)

    blends = []
    for alpha in np.linspace(0, 1, 21):
        for k in (0, 5, 20, 60):
            ranked = blend_rankings(tune_new, tune_legal, float(alpha), k)
            m, _ = fixed_metrics(ranked, tuneq)
            blends.append(
                (m["Recall@5"], m["Precision@5"], m["nDCG@10"],
                 float(alpha), k, m)
            )
    blends.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, alpha, k, blend_tune = blends[0]

    valq = {q: allq[q] for q in val_ids}
    val_new = large_rank(best, kind, val_features, val_ids)
    val_blend = blend_rankings(val_new, val_legal, alpha, k)
    val_m, _ = fixed_metrics(val_blend, valq)

    out_dir = root / "results/burst_large_ltr"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(
        out_dir / "best_model.pkl",
        {
            "kind": kind,
            "params": params,
            "model": best,
            "blend_alpha": alpha,
            "blend_k": k,
            "version": "large-ltr-v1",
        },
    )
    print(
        f"LARGE rebuilt: kind={kind} params={params} "
        f"tuneR={tune_m['Recall@5']:.6f} "
        f"blend(a={alpha},k={k}) valR={val_m['Recall@5']:.6f}",
        flush=True,
    )
    return best, kind


def build_fresh_rankings(
    *,
    root: Path,
    allq,
    cache,
    block_ids,
    tag: str,
    large_model,
    large_kind,
    legal_model,
    normalized,
):
    from tune_burst_large_ltr import matrices
    from tune_burst_legal_features import rank_enhanced

    features, _, _, _ = matrices(cache, allq, block_ids)
    new_rank = large_rank(large_model, large_kind, features, block_ids)
    questions = {q: allq[q][0] for q in block_ids}
    legal_rank = rank_enhanced(
        legal_model,
        cache,
        block_ids,
        questions,
        normalized,
    )
    path = root / f"results/burst_large_ltr/{tag}_rankings.pkl"
    save_pickle(
        path,
        {
            "qids": block_ids,
            "new": new_rank,
            "legal20": legal_rank,
        },
    )
    print(f"Fresh rankings saved: {path}", flush=True)


def build_empirical_pairwise(root, allq):
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import (
        features_for,
        pairwise_matrix,
        rank,
    )
    from tune_burst_kernel_posterior import robust_rankings
    from tune_burst_pairwise import blend_rankings, fixed_metrics

    qids = list(allq)
    train_ids = qids[:700] + qids[850:1150]
    tune_ids = qids[700:750]
    splits = {
        "validation_a": qids[750:850],
        "fresh_a": qids[1150:1250],
        "validation_b": qids[1250:1350],
    }

    large_dir = root / "results/burst_large_ltr"
    cache_main = load_pickle(
        large_dir / "retrieval_train1000_tune50_val100.pkl"
    )["cache"]
    fresh1 = load_pickle(
        large_dir / "fresh_1151_1250_retrieval.pkl"
    )["cache"]
    fresh2 = load_pickle(
        large_dir / "fresh_1251_1350_retrieval.pkl"
    )["cache"]
    cache = cache_main | fresh1 | fresh2

    eval_ids = tune_ids + sum(splits.values(), [])
    baseline = robust_rankings(root, allq, eval_ids)
    full_frequency = label_frequency(allq, set())

    print("Building empirical-pairwise features...", flush=True)
    train_features = features_for(
        cache, allq, train_ids, full_frequency, True
    )
    tune_features = features_for(
        cache,
        allq,
        tune_ids,
        label_frequency(allq, set(tune_ids)),
    )
    eval_features = {}
    for name, bids in splits.items():
        eval_features.update(
            features_for(
                cache,
                allq,
                bids,
                label_frequency(allq, set(bids)),
            )
        )

    tune_gold = {q: allq[q] for q in tune_ids}
    trials = []
    for negative_depth in (5, 10, 20, 40, 80):
        x, y, weights = pairwise_matrix(
            train_features, allq, train_ids, negative_depth
        )
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        print(
            f"  pair depth={negative_depth} rows={len(y):,}",
            flush=True,
        )
        for c in (.003, .01, .03, .1, .3, 1.0, 3.0):
            model = LogisticRegression(
                C=c,
                solver="liblinear",
                max_iter=2000,
            )
            model.fit(z, y, sample_weight=weights)
            tune_rank = rank(model, scaler, tune_features, tune_ids)
            metrics, _ = fixed_metrics(tune_rank, tune_gold)
            trials.append(
                (
                    metrics["Recall@5"],
                    metrics["Precision@5"],
                    metrics["nDCG@10"],
                    negative_depth,
                    c,
                    model,
                    scaler,
                    metrics,
                    tune_rank,
                )
            )

    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    fusion = []
    for trial in trials[:20]:
        _, _, _, depth, c, model, scaler, model_metrics, tune_rank = trial
        for alpha in (
            .02, .05, .08, .10, .15, .20, .25, .30, .40, .50, .65, .80, 1.0
        ):
            for k in (0, 2, 5, 10, 20, 40, 80):
                fused = blend_rankings(
                    tune_rank,
                    {q: baseline[q] for q in tune_ids},
                    alpha,
                    k,
                )
                metrics, _ = fixed_metrics(fused, tune_gold)
                fusion.append(
                    (
                        metrics["Recall@5"],
                        metrics["Precision@5"],
                        metrics["nDCG@10"],
                        depth,
                        c,
                        model,
                        scaler,
                        alpha,
                        k,
                        metrics,
                        model_metrics,
                    )
                )

    fusion.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    reports = []
    for trial in fusion[:300]:
        _, _, _, depth, c, model, scaler, alpha, k, tune_m, model_m = trial
        eval_rank = rank(
            model, scaler, eval_features, sum(splits.values(), [])
        )
        report = {
            "pairwise": {"negative_depth": depth, "C": c},
            "fusion": {"alpha": alpha, "rrf_k": k},
            "model_tune": model_m,
            "fusion_tune": tune_m,
            "splits": {},
        }
        safe = True
        gain = 0.0
        for name, bids in splits.items():
            gold = {q: allq[q] for q in bids}
            fused = blend_rankings(
                {q: eval_rank[q] for q in bids},
                {q: baseline[q] for q in bids},
                alpha,
                k,
            )
            bm, bp = fixed_metrics(
                {q: baseline[q] for q in bids}, gold
            )
            fm, fp = fixed_metrics(fused, gold)
            paired = {
                "wins": sum(a > b for a, b in zip(fp, bp)),
                "ties": sum(a == b for a, b in zip(fp, bp)),
                "losses": sum(a < b for a, b in zip(fp, bp)),
            }
            safe &= (
                paired["losses"] == 0
                and fm["Precision@5"] >= bm["Precision@5"]
            )
            gain += fm["Recall@5"] - bm["Recall@5"]
            report["splits"][name] = {
                "baseline": bm,
                "pairwise": fm,
                "paired": paired,
            }
        report["safe"] = bool(safe)
        report["total_validation_recall_gain"] = gain
        reports.append((report, model, scaler))

    reports.sort(
        key=lambda x: (
            x[0]["safe"],
            x[0]["total_validation_recall_gain"],
            x[0]["fusion_tune"]["Recall@5"],
            x[0]["fusion_tune"]["nDCG@10"],
        ),
        reverse=True,
    )
    best, model, scaler = reports[0]
    out = root / "results/burst_empirical_pairwise"
    out.mkdir(parents=True, exist_ok=True)
    save_pickle(
        out / "model.pkl",
        {
            "model": model,
            "scaler": scaler,
            "report": best,
            "feature_version": "empirical-pairwise-v1",
        },
    )
    print(
        "EMPIRICAL_PAIRWISE rebuilt: "
        f"depth={best['pairwise']['negative_depth']} "
        f"C={best['pairwise']['C']} "
        f"fusion={best['fusion']} safe={best['safe']} "
        f"gain={best['total_validation_recall_gain']:+.6f}",
        flush=True,
    )


def public_cpu_top20_parity(
    *,
    root: Path,
    db_path: Path,
    docs,
    doc_ids,
    allq,
    normalized,
    workers: int,
):
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
    from tune_burst_graph_posterior import build_graph, graph_rerank
    from tune_burst_legal_features import enhanced_features
    from tune_burst_memory import build_query_memory
    from tune_burst_multistage_posterior import weighted_rrf
    from tune_burst_pairwise import blend_rankings
    from tune_burst_score_ltr import retrieve, score_features
    from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    raw_public = json.loads(
        (data / "public-official.json").read_text(encoding="utf-8")
    )
    public = {
        str(q): str(v["question"])
        for q, v in raw_public.items()
    }
    ids = list(public)

    retr_path = root / "results/burst_robust_fusion/public_retrieval.pkl"
    retrieval = {}
    if retr_path.is_file():
        obj = load_pickle(retr_path)
        if obj.get("qids") == ids:
            retrieval = obj.get("cache", {})
    if len(retrieval) != len(ids):
        print("Historical public retrieval unavailable/incomplete; rebuilding...", flush=True)
        con = sqlite3.connect(str(db_path))
        build_query_memory(con, allq)
        con.close()
        local = threading.local()

        def one(q):
            if not hasattr(local, "conn"):
                local.conn = sqlite3.connect(str(db_path))
            return q, retrieve(local.conn, doc_ids, None, public[q])

        retrieval = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (q, result) in enumerate(pool.map(one, ids), 1):
                retrieval[q] = result
                if i % 50 == 0:
                    print(f"  public retrieval {i}/{len(ids)}", flush=True)

    large_saved = load_pickle(root / "results/burst_large_ltr/best_model.pkl")
    if large_saved.get("kind") != "xgb":
        raise RuntimeError(
            "Historical production runner calls large_model.predict(raw X); "
            f"rebuilt large kind={large_saved.get('kind')} is not XGB. "
            "Cannot claim exact CPU production parity."
        )
    large_model = large_saved["model"]
    legal_model = load_pickle(
        root / "results/burst_legal_features/validation_model.pkl"
    )["model"]
    pair_saved = load_pickle(
        root / "results/burst_empirical_pairwise/model.pkl"
    )

    train_ids = list(allq)
    profile_model = build_profiles(allq, train_ids)
    graph_frequency, graph_adjacency, _ = build_graph(allq, train_ids)
    frequency = label_frequency(allq, set())
    public_queries = {q: (public[q], set()) for q in ids}
    pair_features = features_for(
        retrieval, public_queries, ids, frequency
    )

    rebuilt = {}
    started = time.perf_counter()
    for i, q in enumerate(ids, 1):
        lists = retrieval[q]
        candidates, x = score_features(lists)
        large = [
            candidates[j]
            for j in np.argsort(-large_model.predict(x))[:100]
        ]

        candidates2, x2 = enhanced_features(
            lists, public[q], normalized, lexical_depth=20
        )
        legal = [
            candidates2[j]
            for j in np.argsort(-legal_model.decision_function(x2))[:100]
        ]
        robust = blend_rankings(
            {q: large},
            {q: legal[:10]},
            .275,
            40,
        )[q]

        pair = pairwise_rank(
            pair_saved["model"],
            pair_saved["scaler"],
            {q: pair_features[q]},
            [q],
        )[q]
        profile = profile_rank(
            public[q], profile_model, 2, 1.2, .75, .3
        )
        graph = graph_rerank(
            robust,
            graph_frequency,
            graph_adjacency,
            3,
            .5,
            "conditional",
            3,
            .4,
            0,
        )
        final = weighted_rrf(
            [{q: robust}, {q: pair}, {q: profile}, {q: graph}],
            (.40, .30, .15, .15),
            0,
        )[q]
        top20 = list(dict.fromkeys(final))[:20]
        for d in doc_ids:
            if len(top20) >= 20:
                break
            if d not in top20:
                top20.append(d)
        rebuilt[q] = top20

        if i % 50 == 0 or i == len(ids):
            print(
                f"  public CPU check {i}/{len(ids)} "
                f"({time.perf_counter()-started:.1f}s)",
                flush=True,
            )

    historical = load_pickle(
        root / "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]
    exact20 = sum(rebuilt[q] == historical[q] for q in ids)
    exact5 = sum(rebuilt[q][:5] == historical[q][:5] for q in ids)
    overlap5 = float(np.mean([
        len(set(rebuilt[q][:5]) & set(historical[q][:5])) / 5.0
        for q in ids
    ]))
    diffs = [
        {
            "qid": q,
            "rebuilt": rebuilt[q][:20],
            "historical": historical[q][:20],
        }
        for q in ids
        if rebuilt[q] != historical[q]
    ][:20]

    report = {
        "queries": len(ids),
        "exact_ordered_top20": exact20,
        "exact_ordered_top5": exact5,
        "mean_top5_set_overlap": overlap5,
        "sample_diffs": diffs,
        "verdict": (
            "PASS_EXACT_CPU_TOP20"
            if exact20 == len(ids)
            else "FAIL_CPU_TOP20_PARITY"
        ),
    }
    out = root / "results/manual/huy_private_d1_rel_l0_v1/CPU_REBUILD_PARITY.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("PUBLIC CPU TOP20 PARITY")
    print(
        f"exact Top20 = {exact20}/{len(ids)} | "
        f"exact Top5 = {exact5}/{len(ids)} | "
        f"mean Top5 overlap = {overlap5:.6f}"
    )
    print("VERDICT:", report["verdict"])
    print("Report:", out)
    print("=" * 96)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing rebuilt models and fit again.",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from benchmark_burst_v4_full_sqlite import load_dataset

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    docs, allq = load_dataset(data)
    doc_ids = [d for d, _ in docs]
    qids = list(allq)
    if len(allq) != 7000 or len(docs) != 8532:
        raise RuntimeError(
            f"Dataset drift: train={len(allq)} docs={len(docs)}"
        )

    db_path = resolve_db(root, args.db)
    normalized = normalize_docs(docs)

    print("\n[1/6] Rebuild retrieval_401_850 + legal_features model")
    ids_401_850 = qids[400:850]
    excluded_401_850 = qids[700:850]
    pair_cache_path = (
        root / "results/burst_pairwise/"
        "retrieval_401_850_exclude_701_850.pkl"
    )
    cache_401_850 = retrieve_block(
        db_path=db_path,
        doc_ids=doc_ids,
        allq=allq,
        qids=ids_401_850,
        excluded=excluded_401_850,
        cache_path=pair_cache_path,
        workers=args.workers,
    )
    legal_model, legal_features = build_legal_model(
        root, allq, cache_401_850, normalized
    )

    print("\n[2/6] Rebuild large-LTR training retrieval + model")
    train_large = qids[:700] + qids[850:1150]
    tune = qids[700:750]
    val = qids[750:850]
    needed_large = train_large + tune + val
    large_cache_path = (
        root / "results/burst_large_ltr/"
        "retrieval_train1000_tune50_val100.pkl"
    )
    cache_large = retrieve_block(
        db_path=db_path,
        doc_ids=doc_ids,
        allq=allq,
        qids=needed_large,
        excluded=set(tune) | set(val),
        cache_path=large_cache_path,
        workers=args.workers,
    )
    large_model, large_kind = build_large_model(
        root,
        allq,
        cache_large,
        legal_model,
        legal_features,
    )

    print("\n[3/6] Rebuild fresh retrieval/ranking blocks")
    fresh_specs = [
        ("fresh_1151_1250", qids[1150:1250]),
        ("fresh_1251_1350", qids[1250:1350]),
    ]
    for tag, bids in fresh_specs:
        cp = root / f"results/burst_large_ltr/{tag}_retrieval.pkl"
        cache = retrieve_block(
            db_path=db_path,
            doc_ids=doc_ids,
            allq=allq,
            qids=bids,
            excluded=set(bids),
            cache_path=cp,
            workers=args.workers,
        )
        build_fresh_rankings(
            root=root,
            allq=allq,
            cache=cache,
            block_ids=bids,
            tag=tag,
            large_model=large_model,
            large_kind=large_kind,
            legal_model=legal_model,
            normalized=normalized,
        )

    # robust_rankings() unconditionally opens this third file, even though
    # empirical_pairwise's qset never contains these ids.
    placeholder = (
        root / "results/burst_large_ltr/fresh_1351_1450_rankings.pkl"
    )
    save_pickle(
        placeholder,
        {
            "qids": qids[1350:1450],
            "new": {},
            "legal20": {},
            "note": "placeholder; block not used by empirical_pairwise selection",
        },
    )

    print("\n[4/6] Rebuild empirical_pairwise model")
    build_empirical_pairwise(root, allq)

    print("\n[5/6] Artifact summary")
    for rel in (
        "results/burst_legal_features/validation_model.pkl",
        "results/burst_large_ltr/best_model.pkl",
        "results/burst_empirical_pairwise/model.pkl",
    ):
        p = root / rel
        print(f"  {rel}: {'OK' if p.is_file() else 'MISSING'}", flush=True)
        if not p.is_file():
            raise FileNotFoundError(p)

    print("\n[6/6] Mandatory public CPU Top20 parity")
    report = public_cpu_top20_parity(
        root=root,
        db_path=db_path,
        docs=docs,
        doc_ids=doc_ids,
        allq=allq,
        normalized=normalized,
        workers=args.workers,
    )

    if report["verdict"] != "PASS_EXACT_CPU_TOP20":
        print(
            "\nCPU artifacts were rebuilt, but PUBLIC Top20 is not byte/exact "
            "equivalent to the historical cache. Do NOT resume private D1 yet; "
            "send CPU_REBUILD_PARITY.json / the printed parity counts for audit.",
            flush=True,
        )
        raise SystemExit(2)

    print(
        "\nCPU rebuild is exact. You can now rerun "
        "run_private_d1_rel_l0_v3_artifactfix.py; it will reuse these artifacts.",
        flush=True,
    )


if __name__ == "__main__":
    main()
