#!/usr/bin/env python
"""
HUY PIPELINE AUDIT C — CANDIDATE POOL DEPTH / SOURCE MARGINALS V1
================================================================

CPU-only. Uses immutable existing retrieval caches; no model inference.

Questions:
1) How much candidate recall comes from each source?
2) Is current expanded20 + corpus20 too shallow / too deep?
3) Which current-pool golds are uniquely contributed by expanded vs corpus?
4) How many novel docs are paid for per rescued gold occurrence at each depth?
5) Are deeper existing-source candidates actually capable of rescuing the
   current 15 outside-pool D1 gold occurrences?

This is an acquisition audit, NOT a Top-5 selector experiment.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP_DEPTHS = [0, 5, 10, 15, 20, 30, 40, 50, 75, 100]
CORPUS_DEPTHS = [0, 5, 10, 15, 20, 30, 40, 50, 75, 100]


def oracle(queries, pool, ids):
    return float(np.mean([
        len(set(pool[q]) & set(queries[q][1])) / len(queries[q][1])
        for q in ids
    ]))


def gold_occurrences(queries, pool, ids):
    return sum(
        len(set(pool[q]) & set(queries[q][1]))
        for q in ids
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from benchmark_expanded_rerank_holdouts import load_expanded
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_expanded_fusion_robust import TAGS, build_views

    print("[1/5] Loading frozen CAL600 candidate sources...", flush=True)

    # Current exact pool.
    queries, blocks, ids, current, _, _ = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
        expanded_depth=20,
    )

    # Base/expanded source independent of corpus depth.
    q2, b2, ids2, baseexp20, views = build_views(root, expanded_depth=20)
    if ids2 != ids:
        raise RuntimeError("CAL id order mismatch")

    # Full expanded ranking.
    expansion_scores = pickle.loads(
        (root / "results/dense_expansion/union50_scores.pkl").read_bytes()
    )["scores"]
    raw, dense_expanded, expanded_full = load_expanded(
        root,
        queries,
        {TAGS[k]: v for k, v in blocks.items()},
        expansion_scores,
    )

    dense_saved = pickle.loads(
        (root / "results/corpus_index/holdout_dense_rank_cap32.pkl").read_bytes()
    )
    corpus_rank = dense_saved["ranking"]

    # Base is the old multistage/Jina candidate list used before expansion.
    base = {q: list(views["base"][q]) for q in ids}

    print("[2/5] Verifying exact current pool reconstruction...", flush=True)
    reconstructed = {
        q: list(dict.fromkeys(
            base[q] + expanded_full[q][:20] + corpus_rank[q][:20]
        ))
        for q in ids
    }
    mismatch = [q for q in ids if set(reconstructed[q]) != set(current[q])]
    if mismatch:
        raise RuntimeError(
            f"Current pool reconstruction mismatch on {len(mismatch)} qids; "
            f"sample={mismatch[:5]}"
        )

    base_oracle = oracle(queries, base, ids)
    current_oracle = oracle(queries, current, ids)
    current_occ = gold_occurrences(queries, current, ids)

    current_missing = {
        (q, str(d))
        for q in ids
        for d in queries[q][1]
        if str(d) not in set(current[q])
    }

    print(
        f"  base oracle={base_oracle:.10f} | "
        f"current oracle={current_oracle:.10f} | "
        f"outside-pool gold occurrences={len(current_missing)}",
        flush=True,
    )

    print("[3/5] Source marginal accounting at current depth=20...", flush=True)

    pre_corpus = {
        q: list(dict.fromkeys(base[q] + expanded_full[q][:20]))
        for q in ids
    }
    base_plus_corpus = {
        q: list(dict.fromkeys(base[q] + corpus_rank[q][:20]))
        for q in ids
    }

    source_current = {
        "base_only": base,
        "base_plus_expanded20": pre_corpus,
        "base_plus_corpus20": base_plus_corpus,
        "current_union_exp20_corpus20": current,
    }

    source_summary = {}
    for name, pool in source_current.items():
        source_summary[name] = {
            "oracle_recall": oracle(queries, pool, ids),
            "gold_occurrences": gold_occurrences(queries, pool, ids),
            "mean_pool_size": float(np.mean([len(pool[q]) for q in ids])),
            "per_block_oracle": {
                b: oracle(queries, pool, qids)
                for b, qids in blocks.items()
            },
        }

    # Attribution of gold occurrences inside current pool.
    attribution = defaultdict(int)
    attribution_cases = defaultdict(list)

    for q in ids:
        g = set(map(str, queries[q][1]))
        b = set(base[q])
        e = set(expanded_full[q][:20])
        c = set(corpus_rank[q][:20])

        for d in g:
            key = (
                "base" if d in b else ""
            )
            if d not in b:
                in_e = d in e
                in_c = d in c
                if in_e and in_c:
                    key = "expanded_and_corpus_novel"
                elif in_e:
                    key = "expanded_only_novel"
                elif in_c:
                    key = "corpus_only_novel"
                else:
                    key = "outside_current_sources"
            attribution[key] += 1
            attribution_cases[key].append({"qid": q, "doc_id": d})

    print("[4/5] Full expanded-depth × corpus-depth acquisition grid...", flush=True)

    grid = []
    for ed in EXP_DEPTHS:
        for cd in CORPUS_DEPTHS:
            pool = {
                q: list(dict.fromkeys(
                    base[q]
                    + (expanded_full[q][:ed] if ed else [])
                    + (corpus_rank[q][:cd] if cd else [])
                ))
                for q in ids
            }

            rec = oracle(queries, pool, ids)
            occ = gold_occurrences(queries, pool, ids)
            mean_size = float(np.mean([len(pool[q]) for q in ids]))

            rescued_missing = [
                (q, d)
                for q, d in current_missing
                if d in set(pool[q])
            ]

            grid.append({
                "expanded_depth": ed,
                "corpus_depth": cd,
                "oracle_recall": rec,
                "delta_vs_current_oracle": rec - current_oracle,
                "gold_occurrences": occ,
                "delta_gold_occurrences_vs_current": occ - current_occ,
                "mean_pool_size": mean_size,
                "delta_mean_pool_size_vs_current": (
                    mean_size - np.mean([len(current[q]) for q in ids])
                ),
                "rescued_current_outside_gold_occurrences": len(rescued_missing),
                "per_block_oracle": {
                    b: oracle(queries, pool, qids)
                    for b, qids in blocks.items()
                },
            })

    # Pareto frontier: no other grid point has >= recall and <= pool size with
    # at least one strict advantage.
    frontier = []
    for a in grid:
        dominated = False
        for b in grid:
            if a is b:
                continue
            if (
                b["oracle_recall"] >= a["oracle_recall"] - 1e-15
                and b["mean_pool_size"] <= a["mean_pool_size"] + 1e-15
                and (
                    b["oracle_recall"] > a["oracle_recall"] + 1e-15
                    or b["mean_pool_size"] < a["mean_pool_size"] - 1e-15
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(a)

    frontier.sort(key=lambda r: (r["mean_pool_size"], -r["oracle_recall"]))

    # Most useful larger pools and cheapest equal/better pools.
    better = sorted(
        [r for r in grid if r["oracle_recall"] > current_oracle + 1e-15],
        key=lambda r: (
            -r["delta_vs_current_oracle"],
            r["mean_pool_size"],
        ),
    )
    equal_or_better_smaller = sorted(
        [
            r for r in grid
            if r["oracle_recall"] >= current_oracle - 1e-15
            and r["mean_pool_size"] < np.mean([len(current[q]) for q in ids]) - 1e-12
        ],
        key=lambda r: r["mean_pool_size"],
    )

    report = {
        "schema": "manual.candidate_pool_depth_ablation_v1",
        "current_contract": {
            "expanded_depth": 20,
            "corpus_depth": 20,
            "oracle_recall": current_oracle,
            "mean_pool_size": float(np.mean([len(current[q]) for q in ids])),
            "outside_pool_gold_occurrences": len(current_missing),
        },
        "source_summary": source_summary,
        "current_gold_source_attribution": dict(attribution),
        "current_gold_source_attribution_cases": dict(attribution_cases),
        "grid": grid,
        "pareto_frontier": frontier,
        "better_than_current_by_oracle": better,
        "equal_or_better_oracle_with_smaller_pool": equal_or_better_smaller,
        "interpretation": {
            "this_is_candidate_acquisition_only": True,
            "does_not_claim_top5_gain": True,
            "public_labels_used": False,
            "cal_gold_used": True,
            "warning": (
                "Depth selection on CAL is discovery. Any downstream ranking "
                "candidate must be independently validated."
            ),
        },
    }

    out = root / "results/manual/huy_candidate_pool_depth_ablation_v1"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT")
    print("=" * 112)
    for name, row in source_summary.items():
        print(
            f"{name:<34s} oracle={row['oracle_recall']:.10f} "
            f"pool={row['mean_pool_size']:.2f}"
        )

    print("Current gold source attribution:", dict(attribution))

    print("\nTop deeper grids by oracle gain:")
    for r in better[:10]:
        print(
            f"  exp={r['expanded_depth']:>3} corpus={r['corpus_depth']:>3} "
            f"R={r['oracle_recall']:.10f} "
            f"dR={r['delta_vs_current_oracle']:+.10f} "
            f"pool={r['mean_pool_size']:.1f} "
            f"rescued_outside={r['rescued_current_outside_gold_occurrences']}"
        )

    print("\nSmaller pools with >= current oracle:")
    if not equal_or_better_smaller:
        print("  NONE")
    else:
        for r in equal_or_better_smaller[:10]:
            print(
                f"  exp={r['expanded_depth']:>3} corpus={r['corpus_depth']:>3} "
                f"R={r['oracle_recall']:.10f} "
                f"pool={r['mean_pool_size']:.1f}"
            )

    print(f"Pareto points={len(frontier)}")
    print("Report:", report_path)
    print("=" * 112)


if __name__ == "__main__":
    main()
