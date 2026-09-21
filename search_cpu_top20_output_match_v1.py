#!/usr/bin/env python
"""
BLACK-BOX OUTPUT MATCH FOR HISTORICAL cpu_top20.pkl
===================================================

This script does NOT use public labels.

It reconstructs the four CPU branches available in the current BURST code:
  0) robust  = large-LTR + legal blend
  1) pair    = empirical pairwise
  2) profile = supervised profile BM25
  3) graph   = graph rerank of robust

Then it searches the EXACT historical multistage tuner grid:

  pair/profile/graph weight in {0,.05,.10,.15,.20,.30,.40}
  sum(aux weights) <= .65
  robust weight = 1 - sum(aux)
  rrf_k in {0,2,5,10,20,40,80}

Unlike the tuner, this forensic search ALSO includes aux-sum=0, so robust-only
is testable. A branch weight of zero naturally tests two/three-view subsets.

Target:
  results/burst_gpu_threeview/cpu_top20.pkl

Objective is OUTPUT PARITY ONLY:
  - exact ordered Top5
  - exact Top5 set
  - mean Top5 set overlap
  - exact ordered Top20
  - mean Top20 set overlap

No labels / leaderboard metrics are read.

Outputs:
  results/manual/huy_private_d1_rel_l0_v1/
    CPU_OUTPUT_MATCH_COMPONENTS.pkl
    CPU_OUTPUT_MATCH_SEARCH_V1.json

Run:
  cd /d/Study/DSC2026/sota
  source dsc_env_huy/Scripts/activate

  python ../search_cpu_top20_output_match_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


AUX_GRID = (0.0, .05, .10, .15, .20, .30, .40)
K_GRID = (0, 2, 5, 10, 20, 40, 80)


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


def overlap(a, b, k):
    aa, bb = a[:k], b[:k]
    sa, sb = set(aa), set(bb)
    return {
        "exact_order": int(aa == bb),
        "exact_set": int(sa == sb),
        "overlap": len(sa & sb) / k,
        "positional": sum(x == y for x, y in zip(aa, bb)) / k,
    }


def weighted_rrf_one(branches, weights, k):
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


def build_components(root: Path, cache_path: Path):
    if cache_path.is_file():
        obj = load_pickle(cache_path)
        required = {"ids", "robust", "pair", "profile", "graph"}
        if required <= set(obj):
            print(f"Reusing component cache: {cache_path}", flush=True)
            return obj

    from benchmark_burst_v4_full_sqlite import tokens
    from run_burst_multistage_submission import load_metadata
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
    from tune_burst_graph_posterior import build_graph, graph_rerank
    from tune_burst_legal_features import enhanced_features
    from tune_burst_pairwise import blend_rankings
    from tune_burst_score_ltr import score_features
    from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    paths, doc_ids, train, public = load_metadata(data)
    ids = list(public)

    # Historical public retrieval cache.
    retrieval = None
    retrieval_path = None
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
            retrieval = c
            retrieval_path = p
            break
    if retrieval is None:
        raise RuntimeError("No complete historical public retrieval cache found")
    print(f"Historical retrieval: {retrieval_path}", flush=True)

    large_saved = load_pickle(
        root / "results/burst_large_ltr/best_model.pkl"
    )
    if large_saved.get("kind") != "xgb":
        raise RuntimeError(
            f"Expected rebuilt large-LTR kind=xgb, got {large_saved.get('kind')}"
        )
    large_model = large_saved["model"]
    legal_model = load_pickle(
        root / "results/burst_legal_features/validation_model.pkl"
    )["model"]
    pair_saved = load_pickle(
        root / "results/burst_empirical_pairwise/model.pkl"
    )

    print("Building train-dependent profile/graph/frequency...", flush=True)
    train_ids = list(train)
    profile_model = build_profiles(train, train_ids)
    graph_frequency, graph_adjacency, _ = build_graph(train, train_ids)
    frequency = label_frequency(train, set())

    print("Normalizing corpus...", flush=True)
    normalized = {}
    for i, p in enumerate(paths, 1):
        row = json.loads(p.read_text(encoding="utf-8"))
        did = str(row["id"])
        normalized[did] = (
            " " + " ".join(tokens(row.get("passage") or "")) + " "
        )
        if i % 1500 == 0 or i == len(paths):
            print(f"  normalized {i}/{len(paths)}", flush=True)

    public_queries = {q: (public[q], set()) for q in ids}
    pair_features = features_for(
        retrieval, public_queries, ids, frequency
    )

    robust, pair, profile, graph = {}, {}, {}, {}
    large_rank, legal_rank = {}, {}
    started = time.perf_counter()

    for i, q in enumerate(ids, 1):
        lists = retrieval[q]

        candidates, x = score_features(lists)
        large = [
            candidates[j]
            for j in np.argsort(-large_model.predict(x))[:100]
        ]

        c2, x2 = enhanced_features(
            lists,
            public[q],
            normalized,
            lexical_depth=20,
        )
        legal = [
            c2[j]
            for j in np.argsort(
                -legal_model.decision_function(x2)
            )[:100]
        ]

        r = blend_rankings(
            {q: large},
            {q: legal[:10]},
            .275,
            40,
        )[q]

        p_rank = pairwise_rank(
            pair_saved["model"],
            pair_saved["scaler"],
            {q: pair_features[q]},
            [q],
        )[q]

        prof = profile_rank(
            public[q],
            profile_model,
            2,
            1.2,
            .75,
            .3,
        )

        gr = graph_rerank(
            r,
            graph_frequency,
            graph_adjacency,
            3,
            .5,
            "conditional",
            3,
            .4,
            0,
        )

        large_rank[q] = large
        legal_rank[q] = legal
        robust[q] = r
        pair[q] = p_rank
        profile[q] = prof
        graph[q] = gr

        if i % 100 == 0 or i == len(ids):
            print(
                f"  components {i}/{len(ids)} "
                f"({time.perf_counter()-started:.1f}s)",
                flush=True,
            )

    obj = {
        "ids": ids,
        "large": large_rank,
        "legal": legal_rank,
        "robust": robust,
        "pair": pair,
        "profile": profile,
        "graph": graph,
        "large_config": {
            "kind": large_saved.get("kind"),
            "params": large_saved.get("params"),
            "blend_alpha_saved": large_saved.get("blend_alpha"),
            "blend_k_saved": large_saved.get("blend_k"),
        },
        "pair_report": pair_saved.get("report"),
        "retrieval_path": str(retrieval_path),
    }
    save_pickle(cache_path, obj)
    print(f"Saved component cache: {cache_path}", flush=True)
    return obj


def evaluate_config(comp, target, weights, k, qids):
    eo5 = es5 = eo20 = 0
    ov5 = ov20 = pos5 = 0.0

    for q in qids:
        pred = weighted_rrf_one(
            [
                comp["robust"][q],
                comp["pair"][q],
                comp["profile"][q],
                comp["graph"][q],
            ],
            weights,
            k,
        )
        tgt = target[q]

        m5 = overlap(pred, tgt, 5)
        m20 = overlap(pred, tgt, 20)
        eo5 += m5["exact_order"]
        es5 += m5["exact_set"]
        eo20 += m20["exact_order"]
        ov5 += m5["overlap"]
        ov20 += m20["overlap"]
        pos5 += m5["positional"]

    n = len(qids)
    return {
        "exact_top5_order": eo5,
        "exact_top5_set": es5,
        "mean_top5_overlap": ov5 / n,
        "mean_top5_positional": pos5 / n,
        "exact_top20_order": eo20,
        "mean_top20_overlap": ov20 / n,
    }


def score_key(m):
    return (
        m["exact_top5_order"],
        m["exact_top5_set"],
        m["mean_top5_overlap"],
        m["mean_top20_overlap"],
        m["exact_top20_order"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--sample",
        type=int,
        default=250,
        help="Stage-A query sample size for full-grid shortlist.",
    )
    ap.add_argument(
        "--topn-full",
        type=int,
        default=40,
        help="How many Stage-A configs to evaluate on all 1000 queries.",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    out_dir = root / "results/manual/huy_private_d1_rel_l0_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "CPU_OUTPUT_MATCH_COMPONENTS.pkl"

    print("[1/4] Build/reuse CPU component rankings...")
    comp = build_components(root, cache_path)
    ids = comp["ids"]

    target = load_pickle(
        root / "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]

    # Deterministic spread across all 1000 rather than the first N.
    n_sample = min(args.sample, len(ids))
    sample_idx = np.linspace(
        0, len(ids) - 1, n_sample, dtype=int
    )
    sample_ids = [ids[i] for i in sample_idx]

    print("[2/4] Sanity: current hard-coded stack...")
    current_weights = (.40, .30, .15, .15)
    current_k = 0
    current_sample = evaluate_config(
        comp, target, current_weights, current_k, sample_ids
    )
    current_full = evaluate_config(
        comp, target, current_weights, current_k, ids
    )
    print(
        "  current full: "
        f"Top5 exact={current_full['exact_top5_order']}/{len(ids)} "
        f"set={current_full['exact_top5_set']}/{len(ids)} "
        f"ov5={current_full['mean_top5_overlap']:.6f} "
        f"Top20 exact={current_full['exact_top20_order']}/{len(ids)} "
        f"ov20={current_full['mean_top20_overlap']:.6f}",
        flush=True,
    )

    print("[3/4] Stage-A exhaustive historical multistage grid on sample...")
    configs = []
    for pw, sw, gw in itertools.product(AUX_GRID, repeat=3):
        if pw + sw + gw > .6500000001:
            continue
        bw = 1.0 - pw - sw - gw
        if bw < -1e-12:
            continue
        weights = (
            round(bw, 12),
            float(pw),
            float(sw),
            float(gw),
        )
        for k in K_GRID:
            configs.append((weights, k))

    print(
        f"  configs={len(configs)} | sample_q={len(sample_ids)}",
        flush=True,
    )
    trials = []
    started = time.perf_counter()
    for i, (weights, k) in enumerate(configs, 1):
        m = evaluate_config(
            comp, target, weights, k, sample_ids
        )
        trials.append({
            "weights_robust_pair_profile_graph": list(weights),
            "rrf_k": k,
            "sample": m,
        })
        if i % 250 == 0 or i == len(configs):
            print(
                f"    grid {i}/{len(configs)} "
                f"({time.perf_counter()-started:.1f}s)",
                flush=True,
            )

    trials.sort(
        key=lambda x: score_key(x["sample"]),
        reverse=True,
    )

    print(
        f"[4/4] Full-1000 evaluation of top {args.topn_full} configs...",
        flush=True,
    )
    finalists = []
    for i, t in enumerate(trials[:args.topn_full], 1):
        weights = tuple(t["weights_robust_pair_profile_graph"])
        k = int(t["rrf_k"])
        full = evaluate_config(comp, target, weights, k, ids)
        finalists.append({
            **t,
            "full": full,
        })
        print(
            f"  #{i:02d} w={weights} k={k} | "
            f"Top5={full['exact_top5_order']}/{len(ids)} "
            f"set={full['exact_top5_set']}/{len(ids)} "
            f"ov5={full['mean_top5_overlap']:.6f} "
            f"Top20={full['exact_top20_order']}/{len(ids)} "
            f"ov20={full['mean_top20_overlap']:.6f}",
            flush=True,
        )

    finalists.sort(
        key=lambda x: score_key(x["full"]),
        reverse=True,
    )
    best = finalists[0]

    # Branch diagnostics: whether historical artifact resembles one component.
    branch_diag = {}
    for name in ("large", "legal", "robust", "pair", "profile", "graph"):
        fake = {
            "robust": comp[name],
            "pair": comp[name],
            "profile": comp[name],
            "graph": comp[name],
        }
        # all mass on robust copy -> exactly the selected branch order.
        branch_diag[name] = evaluate_config(
            fake,
            target,
            (1.0, 0.0, 0.0, 0.0),
            0,
            ids,
        )

    verdict = "NO_EXACT_MATCH_IN_FINAL_STACK_GRID"
    if best["full"]["exact_top20_order"] == len(ids):
        verdict = "EXACT_HISTORICAL_FINAL_STACK_FOUND"
    elif best["full"]["exact_top5_order"] == len(ids):
        verdict = "EXACT_TOP5_STACK_FOUND_TOP20_DIFFERS"
    elif best["full"]["exact_top5_order"] >= 950:
        verdict = "HIGH_PARITY_STACK_FOUND"
    elif best["full"]["exact_top5_order"] > current_full["exact_top5_order"]:
        verdict = "STACK_CONFIG_DRIFT_PARTIALLY_EXPLAINS_OUTPUT"
    else:
        verdict = "FINAL_STACK_GRID_DOES_NOT_EXPLAIN_DRIFT"

    report = {
        "schema": "manual.cpu_top20_output_match_search_v1",
        "labels_used": False,
        "target": str(
            root / "results/burst_gpu_threeview/cpu_top20.pkl"
        ),
        "component_cache": str(cache_path),
        "component_metadata": {
            "large_config": comp.get("large_config"),
            "pair_report": comp.get("pair_report"),
            "retrieval_path": comp.get("retrieval_path"),
        },
        "search_space": {
            "aux_weight_grid": list(AUX_GRID),
            "rrf_k_grid": list(K_GRID),
            "robust_weight": "1-pair-profile-graph",
            "aux_sum_max": 0.65,
            "configs": len(configs),
            "sample_queries": len(sample_ids),
            "full_finalists": len(finalists),
        },
        "current_hardcoded": {
            "weights": list(current_weights),
            "rrf_k": current_k,
            "sample": current_sample,
            "full": current_full,
        },
        "single_branch_diagnostics": branch_diag,
        "best": best,
        "top_full": finalists[:20],
        "top_sample": trials[:20],
        "verdict": verdict,
    }

    report_path = out_dir / "CPU_OUTPUT_MATCH_SEARCH_V1.json"
    dump(report_path, report)

    print("=" * 108)
    print("CPU TOP20 BLACK-BOX OUTPUT MATCH")
    print(
        f"CURRENT: Top5 exact={current_full['exact_top5_order']}/1000 "
        f"ov5={current_full['mean_top5_overlap']:.6f} "
        f"ov20={current_full['mean_top20_overlap']:.6f}"
    )
    print(
        "BEST:",
        best["weights_robust_pair_profile_graph"],
        "k=", best["rrf_k"],
    )
    print(
        f"BEST Top5 exact={best['full']['exact_top5_order']}/1000 "
        f"set={best['full']['exact_top5_set']}/1000 "
        f"ov5={best['full']['mean_top5_overlap']:.6f} "
        f"Top20 exact={best['full']['exact_top20_order']}/1000 "
        f"ov20={best['full']['mean_top20_overlap']:.6f}"
    )
    print("VERDICT:", verdict)
    print("Report:", report_path)
    print("=" * 108)


if __name__ == "__main__":
    main()
