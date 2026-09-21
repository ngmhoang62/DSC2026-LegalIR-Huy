#!/usr/bin/env python
"""
HUY PIPELINE AUDIT E — CAP32 vs CAP48 FULL-CORPUS DENSE V1
==========================================================

Run AFTER:
  python build_corpus_dense_index.py --cap 48 --batch-size 32 --name chunks
  python benchmark_corpus_dense_recall.py --cap 48

CPU-only comparison of the resulting retrieval caches.

Measures:
- dense standalone Recall@5/20/50/100;
- candidate ceiling when adding top20/top50 from cap32 vs cap48;
- cap48-only rescued gold occurrences;
- current-D1 outside-pool gold occurrences reached by cap48 but not cap32;
- top20 overlap / churn.

No public labels.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def macro_recall(queries, rank, ids, k):
    return float(np.mean([
        len(set(rank[q][:k]) & set(queries[q][1])) / len(queries[q][1])
        for q in ids
    ]))


def pool_oracle(queries, base, rank, ids, k):
    return float(np.mean([
        len((set(base[q]) | set(rank[q][:k])) & set(queries[q][1]))
        / len(queries[q][1])
        for q in ids
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from tune_expanded_fusion_robust import build_views
    from tune_corpus_cap32_fusion import build_training_cap

    print("[1/4] Loading pre-corpus candidate pool and cap32/cap48 rankings...", flush=True)

    queries, blocks, ids, pre_corpus, _ = build_views(root, expanded_depth=20)
    qcur, bcur, idcur, current, _, _ = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
        expanded_depth=20,
    )
    if ids != idcur:
        raise RuntimeError("CAL ids mismatch")

    ranks = {}
    for cap in (32, 48):
        p = root / f"results/corpus_index/holdout_dense_rank_cap{cap}.pkl"
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Build+benchmark cap{cap} first."
            )
        obj = pickle.loads(p.read_bytes())
        ranks[cap] = {
            q: [str(d) for d in obj["ranking"][q]]
            for q in ids
        }

    print("[2/4] Computing standalone dense and candidate-ceiling deltas...", flush=True)

    summary = {}
    for cap in (32, 48):
        summary[str(cap)] = {
            "standalone": {
                f"recall@{k}": macro_recall(queries, ranks[cap], ids, k)
                for k in (5, 20, 50, 100)
            },
            "candidate_ceiling": {
                f"precorpus_plus_top{k}": pool_oracle(
                    queries, pre_corpus, ranks[cap], ids, k
                )
                for k in (10, 20, 50, 100)
            },
            "per_block_dense_r20": {
                b: macro_recall(queries, ranks[cap], qids, 20)
                for b, qids in blocks.items()
            },
            "per_block_ceiling_plus20": {
                b: pool_oracle(queries, pre_corpus, ranks[cap], qids, 20)
                for b, qids in blocks.items()
            },
        }

    print("[3/4] Forensic cap48-only recovery / top20 churn...", flush=True)

    cap48_only_gold = []
    cap32_only_gold = []
    current_outside = {
        (q, str(d))
        for q in ids
        for d in queries[q][1]
        if str(d) not in set(current[q])
    }
    outside_reached_48 = []
    outside_reached_32 = []

    overlaps20 = []
    for q in ids:
        g = set(map(str, queries[q][1]))
        a = set(ranks[32][q][:20])
        b = set(ranks[48][q][:20])

        overlaps20.append(len(a & b) / 20.0)

        for d in sorted((b - a) & g):
            cap48_only_gold.append({"qid": q, "doc_id": d})
        for d in sorted((a - b) & g):
            cap32_only_gold.append({"qid": q, "doc_id": d})

    for q, d in sorted(current_outside):
        if d in set(ranks[48][q][:100]):
            outside_reached_48.append({
                "qid": q,
                "doc_id": d,
                "rank48": ranks[48][q].index(d) + 1,
                "rank32": (
                    ranks[32][q].index(d) + 1
                    if d in ranks[32][q]
                    else None
                ),
            })
        if d in set(ranks[32][q][:100]):
            outside_reached_32.append({
                "qid": q,
                "doc_id": d,
                "rank32": ranks[32][q].index(d) + 1,
            })

    delta = {
        "dense_r20": (
            summary["48"]["standalone"]["recall@20"]
            - summary["32"]["standalone"]["recall@20"]
        ),
        "ceiling_plus20": (
            summary["48"]["candidate_ceiling"]["precorpus_plus_top20"]
            - summary["32"]["candidate_ceiling"]["precorpus_plus_top20"]
        ),
        "per_block_dense_r20": {
            b: (
                summary["48"]["per_block_dense_r20"][b]
                - summary["32"]["per_block_dense_r20"][b]
            )
            for b in blocks
        },
        "per_block_ceiling_plus20": {
            b: (
                summary["48"]["per_block_ceiling_plus20"][b]
                - summary["32"]["per_block_ceiling_plus20"][b]
            )
            for b in blocks
        },
    }

    # Promotion to expensive extended rescoring only if acquisition signal exists.
    promote = bool(
        delta["ceiling_plus20"] > 1e-12
        or (
            delta["dense_r20"] > 0
            and all(x >= -1e-12 for x in delta["per_block_dense_r20"].values())
        )
    )

    report = {
        "schema": "manual.cap32_vs_cap48_dense_audit_v1",
        "summary": summary,
        "delta_cap48_minus_cap32": delta,
        "top20_overlap": {
            "mean": float(np.mean(overlaps20)),
            "p10": float(np.percentile(overlaps20, 10)),
            "p50": float(np.percentile(overlaps20, 50)),
            "p90": float(np.percentile(overlaps20, 90)),
        },
        "cap48_only_gold_in_top20_vs_cap32": cap48_only_gold,
        "cap32_only_gold_in_top20_vs_cap48": cap32_only_gold,
        "current_outside_pool_reached_by_cap48_top100": outside_reached_48,
        "current_outside_pool_reached_by_cap32_top100": outside_reached_32,
        "promotion_gate": {
            "verdict": (
                "PROMOTE_CAP48_TO_EXTENDED_RERANK_SCORING"
                if promote
                else "KILL_CAP48"
            ),
            "rule": (
                "promote iff cap48 improves candidate ceiling@20, OR improves "
                "dense R@20 with no block regression"
            ),
            "public_labels_used": False,
        },
    }

    out = root / "results/manual/huy_cap32_vs_cap48_dense_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] RESULT")
    print("=" * 112)
    print(
        f"Dense R@20: cap32={summary['32']['standalone']['recall@20']:.10f} "
        f"cap48={summary['48']['standalone']['recall@20']:.10f} "
        f"d={delta['dense_r20']:+.10f}"
    )
    print(
        f"Candidate ceiling + corpus20: "
        f"cap32={summary['32']['candidate_ceiling']['precorpus_plus_top20']:.10f} "
        f"cap48={summary['48']['candidate_ceiling']['precorpus_plus_top20']:.10f} "
        f"d={delta['ceiling_plus20']:+.10f}"
    )
    print("Block dense R@20 deltas:", delta["per_block_dense_r20"])
    print("Block ceiling deltas:", delta["per_block_ceiling_plus20"])
    print(
        f"Top20 overlap mean={np.mean(overlaps20):.4f}; "
        f"cap48-only gold={len(cap48_only_gold)} "
        f"cap32-only gold={len(cap32_only_gold)}"
    )
    print(
        f"Current outside-pool gold reached @100: "
        f"cap32={len(outside_reached_32)} cap48={len(outside_reached_48)}"
    )
    print("VERDICT:", report["promotion_gate"]["verdict"])
    print("Report:", path)
    print("=" * 112)


if __name__ == "__main__":
    main()
