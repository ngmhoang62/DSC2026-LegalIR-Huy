#!/usr/bin/env python
"""
AUDIT REBUILT CPU BASE IMPACT ON PUBLIC D1
==========================================

Purpose:
  We know the rebuilt BURST CPU Top20 is not byte-identical to historical
  cpu_top20.pkl. This audit asks the more relevant question:

      Does that CPU-base drift materially change the final production D1?

It does NOT use public labels. It compares predictions only against the
authoritative historical D1 public champion.

Checks:
  1) Rebuild current CPU public Top20 using the newly rebuilt CPU models.
  2) Compare ordered/set overlap vs historical cpu_top20.pkl.
  3) Re-train exact D1 on CAL600 (same 48D contract).
  4) Keep historical public candidate pool + all historical non-base channels,
     replace ONLY the "base" rank view with rebuilt CPU Top20.
  5) Compare resulting D1 Top5 to exact public D1 champion.
  6) Estimate candidate-pool membership drift if rebuilt CPU Top20 were used
     in candidate generation.

No leaderboard labels are read.

Run:
  python ../audit_rebuilt_cpu_impact_on_public_d1_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]


def load_pickle(p: Path):
    return pickle.loads(p.read_bytes())


def unwrap(obj):
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def rebuild_public_cpu_top20(root: Path):
    from benchmark_burst_v4_full_sqlite import tokens
    from run_burst_multistage_submission import load_metadata
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
    from tune_burst_graph_posterior import build_graph, graph_rerank
    from tune_burst_legal_features import enhanced_features
    from tune_burst_multistage_posterior import weighted_rrf
    from tune_burst_pairwise import blend_rankings
    from tune_burst_score_ltr import score_features
    from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    paths, doc_ids, train, public = load_metadata(data)
    ids = list(public)

    retr_path = root / "results/burst_robust_fusion/public_retrieval.pkl"
    if not retr_path.is_file():
        raise FileNotFoundError(retr_path)
    robj = load_pickle(retr_path)
    if robj.get("qids") != ids:
        raise RuntimeError("Historical public retrieval qids mismatch")
    retrieval = robj["cache"]

    large_saved = load_pickle(root / "results/burst_large_ltr/best_model.pkl")
    large_model = large_saved["model"]
    legal_model = load_pickle(
        root / "results/burst_legal_features/validation_model.pkl"
    )["model"]
    pair_saved = load_pickle(
        root / "results/burst_empirical_pairwise/model.pkl"
    )

    print("  building profile/graph/frequency...", flush=True)
    train_ids = list(train)
    profile_model = build_profiles(train, train_ids)
    graph_frequency, graph_adjacency, _ = build_graph(train, train_ids)
    frequency = label_frequency(train, set())

    print("  normalizing corpus...", flush=True)
    normalized = {}
    for i, p in enumerate(paths, 1):
        row = json.loads(p.read_text(encoding="utf-8"))
        did = str(row["id"])
        normalized[did] = " " + " ".join(tokens(row.get("passage") or "")) + " "
        if i % 1500 == 0:
            print(f"    {i}/{len(paths)}", flush=True)

    public_queries = {q: (public[q], set()) for q in ids}
    pair_features = features_for(
        retrieval, public_queries, ids, frequency
    )

    rebuilt = {}
    started = time.perf_counter()
    for i, q in enumerate(ids, 1):
        lists = retrieval[q]

        candidates, x = score_features(lists)
        # Historical production runner assumes XGB-style .predict().
        large = [
            candidates[j]
            for j in np.argsort(-large_model.predict(x))[:100]
        ]

        c2, x2 = enhanced_features(
            lists, public[q], normalized, lexical_depth=20
        )
        legal = [
            c2[j]
            for j in np.argsort(-legal_model.decision_function(x2))[:100]
        ]
        robust = blend_rankings(
            {q: large}, {q: legal[:10]}, .275, 40
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
            3, .5, "conditional", 3, .4, 0,
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

        if i % 100 == 0:
            print(
                f"    rebuilt CPU {i}/{len(ids)} "
                f"({time.perf_counter()-started:.1f}s)",
                flush=True,
            )

    return paths, doc_ids, train, public, ids, rebuilt


def overlap_stats(a, b, ids, k):
    exact_order = sum(a[q][:k] == b[q][:k] for q in ids)
    exact_set = sum(set(a[q][:k]) == set(b[q][:k]) for q in ids)
    set_overlap = np.mean([
        len(set(a[q][:k]) & set(b[q][:k])) / k
        for q in ids
    ])
    positional = np.mean([
        sum(x == y for x, y in zip(a[q][:k], b[q][:k])) / k
        for q in ids
    ])
    return {
        "k": k,
        "exact_order": exact_order,
        "exact_set": exact_set,
        "mean_set_overlap": float(set_overlap),
        "mean_positional_match": float(positional),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import (
        CORPUS_DEPTH,
        RERANK_CONFIG,
        DocumentStore,
    )
    from tune_expanded_fusion_selection import ltr_features
    from tune_doctype_features import build_type_table, type_features
    from tune_citation_graph import build_citation_table, citation_features

    print("[1/5] Rebuild public CPU Top20 with rebuilt CPU artifacts...")
    paths, doc_ids, train, public, ids, rebuilt = rebuild_public_cpu_top20(root)

    hist = load_pickle(
        root / "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]

    cpu_stats = {
        "top5": overlap_stats(rebuilt, hist, ids, 5),
        "top10": overlap_stats(rebuilt, hist, ids, 10),
        "top20": overlap_stats(rebuilt, hist, ids, 20),
    }
    print("  CPU overlap:")
    for k, s in cpu_stats.items():
        print(
            f"    {k}: exact_order={s['exact_order']}/{len(ids)} "
            f"exact_set={s['exact_set']}/{len(ids)} "
            f"set_overlap={s['mean_set_overlap']:.6f} "
            f"positional={s['mean_positional_match']:.6f}"
        )

    print("[2/5] Train exact 48D D1...")
    runner_path = root.parent / "run_private_d1_rel_l0_v3_artifactfix.py"
    if not runner_path.is_file():
        # v2 is also sufficient for train_exact_d1
        runner_path = root.parent / "run_private_d1_rel_l0_v2_ftsfix.py"
    if not runner_path.is_file():
        raise FileNotFoundError(
            "Need previous private runner at parent of repo: "
            "run_private_d1_rel_l0_v3_artifactfix.py"
        )
    pr = load_module(runner_path, "private_runner_for_cpu_audit")
    documents = DocumentStore(paths)
    scaler, model, d1_meta = pr.train_exact_d1(root, documents)

    print("[3/5] Reconstruct historical public D1 feature universe...")
    exp_obj = load_pickle(
        root / "results/burst_expanded_fusion/candidates.pkl"
    )
    hist_candidates = exp_obj["candidates"]
    expanded = exp_obj["expanded"]
    corpus_score = exp_obj["corpus_score"]
    expansion = exp_obj["expansion"]

    rerank = load_pickle(
        root / "results/burst_expanded_fusion/rerank_scores.pkl"
    )
    three = unwrap(load_pickle(
        root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl"
    ))
    vnlegal = unwrap(load_pickle(
        root / "results/burst_userft_maxrecall/vnlegal_scores.pkl"
    ))
    crossenc = unwrap(load_pickle(
        root / "results/crossenc_fullpool/public_scores.pkl"
    ))
    aiteam_ft = unwrap(load_pickle(
        root / "results/from_drive/aiteamvn_ft_public.pkl"
    ))
    jina_ft = unwrap(load_pickle(
        root / "results/from_drive/jina_ft_public.pkl"
    ))
    title = unwrap(load_pickle(
        root / "results/burst_fresh_block/title_embed_public.pkl"
    ))

    cv_ce = unwrap(load_pickle(
        root / "results/crossenc_fullpool/cv_scores.pkl"
    ))
    cv_ai = unwrap(load_pickle(
        root / "results/from_drive/aiteamvn_ft_cv.pkl"
    ))
    cv_jf = unwrap(load_pickle(
        root / "results/from_drive/jina_ft_cv.pkl"
    ))
    cv_title = unwrap(load_pickle(
        root / "results/burst_fresh_block/title_embed_scores.pkl"
    ))
    floors = {
        "crossenc": min(min(v.values()) for v in cv_ce.values() if v),
        "aiteamvn_ft": min(v for q in cv_ai for v in cv_ai[q].values()),
        "jina_ft": min(v for q in cv_jf for v in cv_jf[q].values()),
        "title_embed": min(v for q in cv_title for v in cv_title[q].values()),
    }

    # Estimate candidate-pool membership drift from swapping CPU base.
    alt_pools = {}
    pool_exact = 0
    pool_recall = []
    pool_precision = []
    added_counts = []
    removed_counts = []
    for q in ids:
        corpus_rank = sorted(
            corpus_score[q],
            key=lambda d: (-corpus_score[q][d], d),
        )
        alt = list(dict.fromkeys(
            rebuilt[q]
            + expanded[q][:RERANK_CONFIG["expanded_depth"]]
            + corpus_rank[:CORPUS_DEPTH]
        ))
        alt_pools[q] = alt
        hs = set(hist_candidates[q])
        aset = set(alt)
        pool_exact += (hs == aset)
        pool_recall.append(len(hs & aset) / max(1, len(hs)))
        pool_precision.append(len(hs & aset) / max(1, len(aset)))
        added_counts.append(len(aset - hs))
        removed_counts.append(len(hs - aset))

    pool_stats = {
        "exact_membership": int(pool_exact),
        "mean_historical_pool_coverage": float(np.mean(pool_recall)),
        "mean_alt_pool_covered_by_historical": float(np.mean(pool_precision)),
        "mean_added_docs": float(np.mean(added_counts)),
        "max_added_docs": int(max(added_counts)),
        "mean_removed_docs": float(np.mean(removed_counts)),
        "max_removed_docs": int(max(removed_counts)),
    }
    print(
        "  candidate pool drift: "
        f"exact={pool_stats['exact_membership']}/{len(ids)} "
        f"hist_coverage={pool_stats['mean_historical_pool_coverage']:.6f} "
        f"mean_added={pool_stats['mean_added_docs']:.3f} "
        f"mean_removed={pool_stats['mean_removed_docs']:.3f}",
        flush=True,
    )

    print("[4/5] Counterfactual D1: historical pool, rebuilt BASE rank view only...")
    view_rank = {
        "base": {q: rebuilt[q] for q in ids},
        "expanded": {q: list(expanded[q]) for q in ids},
        "jina": {
            q: sorted(
                hist_candidates[q],
                key=lambda d: (-rerank["jina"][q][d], d),
            )
            for q in ids
        },
        "dense": {
            q: sorted(
                hist_candidates[q],
                key=lambda d: (-rerank["dense"][q][d], d),
            )
            for q in ids
        },
        "corpus": {
            q: sorted(
                (d for d in hist_candidates[q] if d in corpus_score[q]),
                key=lambda d: (-corpus_score[q][d], d),
            )
            for q in ids
        },
    }
    scores = {
        "jina": rerank["jina"],
        "dense": rerank["dense"],
        "expansion": expansion,
        "e5": {q: three[q]["e5"] for q in ids},
        "corpus": {
            q: {
                d: corpus_score[q].get(d, -1.0)
                for d in hist_candidates[q]
            }
            for q in ids
        },
        "vnlegal_lal": vnlegal,
        "crossenc": {
            q: {
                d: crossenc.get(q, {}).get(d, floors["crossenc"])
                for d in hist_candidates[q]
            }
            for q in ids
        },
        "aiteamvn_ft": {
            q: {
                d: aiteam_ft.get(q, {}).get(d, floors["aiteamvn_ft"])
                for d in hist_candidates[q]
            }
            for q in ids
        },
        "jina_ft": {
            q: {
                d: jina_ft.get(q, {}).get(d, floors["jina_ft"])
                for d in hist_candidates[q]
            }
            for q in ids
        },
        "title_embed": {
            q: {
                d: title.get(q, {}).get(d, floors["title_embed"])
                for d in hist_candidates[q]
            }
            for q in ids
        },
    }

    qmeta = {q: (public[q], set()) for q in ids}
    type_table = build_type_table(root, documents, ids, hist_candidates)
    type_rows = type_features(
        hist_candidates, type_table, qmeta, ids
    )
    own, cited = build_citation_table(documents, ids, hist_candidates)
    cite_rows = citation_features(
        hist_candidates, own, cited, ids
    )

    rows0, groups = ltr_features(
        view_rank,
        D1_VIEWS,
        hist_candidates,
        ids,
        scores,
    )
    counter = {}
    for q in ids:
        X = np.concatenate(
            [rows0[q], type_rows[q], cite_rows[q]], axis=1
        )
        s = model.decision_function(scaler.transform(X))
        order = np.argsort(-s)
        counter[q] = [groups[q][i] for i in order[:5]]

    champ_raw = json.loads(
        (
            root
            / "results/gemini/huy_vnlegal_rank_ablation_v1/"
            "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
        ).read_text(encoding="utf-8")
    )
    champion = {
        str(q): [str(d) for d in row["answer"]]
        for q, row in champ_raw.items()
    }

    d1_stats = overlap_stats(counter, champion, ids, 5)
    rank5_same = sum(counter[q][4] == champion[q][4] for q in ids)
    first4_same = sum(counter[q][:4] == champion[q][:4] for q in ids)
    print(
        f"  D1 exact ordered Top5 = {d1_stats['exact_order']}/{len(ids)}\n"
        f"  D1 exact Top5 set     = {d1_stats['exact_set']}/{len(ids)}\n"
        f"  D1 mean set overlap   = {d1_stats['mean_set_overlap']:.6f}\n"
        f"  D1 first4 exact       = {first4_same}/{len(ids)}\n"
        f"  D1 rank5 same         = {rank5_same}/{len(ids)}",
        flush=True,
    )

    print("[5/5] Verdict + report...")
    # This is deliberately descriptive, not a hidden auto-approval threshold.
    if (
        d1_stats["exact_order"] >= 980
        and d1_stats["mean_set_overlap"] >= 0.995
        and pool_stats["mean_historical_pool_coverage"] >= 0.99
    ):
        verdict = "BASE_DRIFT_LOW_AT_D1_LEVEL"
    elif (
        d1_stats["exact_order"] >= 950
        and d1_stats["mean_set_overlap"] >= 0.99
    ):
        verdict = "BASE_DRIFT_MODERATE_AT_D1_LEVEL"
    else:
        verdict = "BASE_DRIFT_MATERIAL_AT_D1_LEVEL"

    report = {
        "schema": "manual.rebuilt_cpu_public_d1_impact_v1",
        "cpu_top20_overlap": cpu_stats,
        "candidate_pool_drift": pool_stats,
        "d1_rank_only_counterfactual": {
            **d1_stats,
            "first4_exact": first4_same,
            "rank5_same": rank5_same,
        },
        "verdict": verdict,
        "note": (
            "Counterfactual keeps historical candidate pool and all non-base "
            "public channels fixed; only D1 base rank view is replaced."
        ),
    }

    out_dir = root / "results/manual/huy_private_d1_rel_l0_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "REBUILT_CPU_PUBLIC_D1_IMPACT.json"
    p.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Save rebuilt ranking for later forensic work.
    with (out_dir / "REBUILT_PUBLIC_CPU_TOP20.pkl").open("wb") as f:
        pickle.dump({"rankings": rebuilt}, f, protocol=5)

    print("=" * 100)
    print("REBUILT CPU -> PUBLIC D1 IMPACT")
    print(
        f"CPU Top20 set overlap = {cpu_stats['top20']['mean_set_overlap']:.6f}"
    )
    print(
        f"Pool historical coverage = "
        f"{pool_stats['mean_historical_pool_coverage']:.6f}"
    )
    print(
        f"D1 exact Top5 = {d1_stats['exact_order']}/{len(ids)} | "
        f"D1 set overlap = {d1_stats['mean_set_overlap']:.6f} | "
        f"rank5 same = {rank5_same}/{len(ids)}"
    )
    print("VERDICT:", verdict)
    print("Report:", p)
    print("=" * 100)


if __name__ == "__main__":
    main()
