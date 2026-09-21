#!/usr/bin/env python
"""
HUY PIPELINE AUDIT D — SPARSE / QUERY-MEMORY PROVENANCE V1
===========================================================

CPU-only. No model inference, no public labels.

Audits the four historical raw retrieval branches:
  0 FULL_FTS
  1 LOCAL_FTS
  2 TRIGRAM
  3 QUERY_MEMORY

Two goals:
1) Measure acquisition complementarity against exact D1 Top5 and current pool.
2) Audit query-memory leakage/provenance by reconstructing two contracts:
   CLEAN_BLOCK_EXCLUDED: entire held CAL block absent from labelled memory.
   SELF_ONLY_EXCLUDED: all labelled queries in memory; only current qid skipped.

For each held block we compare the cached memory branch against both. If cached
rankings are much closer to SELF_ONLY_EXCLUDED than CLEAN_BLOCK_EXCLUDED, that
is evidence of cross-query held-block label leakage. If they match CLEAN, good.
If neither, provenance remains unresolved.

Important: historical repo marks fresh retrieval cache generators unresolved, so
this reconstruction is materially useful.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


DEPTHS = [5, 10, 20, 50, 100]
BRANCH_NAMES = ["FULL_FTS", "LOCAL_FTS", "TRIGRAM", "QUERY_MEMORY"]


def docs_from_branch(branch):
    out = []
    for item in branch:
        if isinstance(item, (tuple, list)):
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return out


def prefix_overlap(a, b, k):
    aa = a[:k]
    bb = b[:k]
    if not aa and not bb:
        return 1.0
    return len(set(aa) & set(bb)) / max(1, k)


def exact_prefix(a, b, k):
    return a[:k] == b[:k]


def build_memory_db(build_query_memory, queries_subset):
    conn = sqlite3.connect(":memory:")
    build_query_memory(conn, queries_subset)
    return conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from benchmark_jina_reranker_holdouts import load_cache
    from tune_burst_kernel_posterior import load_queries
    from tune_burst_memory import build_query_memory
    from tune_burst_score_ltr import memory_scores
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_expanded_fusion_robust import TAGS

    print("[1/6] Loading CAL600 contract + exact D1 Top5...", flush=True)

    queries, blocks, ids, current_pool, _, _ = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    pred_path = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "S0_S1_CAL_PREDICTIONS.jsonl"
    )
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)

    d1 = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = str(row["qid"])
            if q in gold:
                d1[q] = [str(x) for x in row["s0_top5"]]

    if set(d1) != set(ids):
        raise RuntimeError(
            f"Exact D1 prediction population mismatch: {len(d1)} vs {len(ids)}"
        )

    d1_misses = {
        (q, d)
        for q in ids
        for d in gold[q]
        if d not in set(d1[q])
    }
    outside_pool = {
        (q, d)
        for q in ids
        for d in gold[q]
        if d not in set(current_pool[q])
    }

    print(
        f"  D1 missed gold occurrences={len(d1_misses)} | "
        f"outside current pool={len(outside_pool)}",
        flush=True,
    )

    print("[2/6] Loading immutable raw retrieval caches...", flush=True)

    branch_rank = {name: {} for name in BRANCH_NAMES}
    cache_paths = {}

    for block, qids in blocks.items():
        tag = TAGS[block]
        cache = load_cache(root, tag)
        cache_paths[block] = tag
        for q in qids:
            if q not in cache:
                raise RuntimeError(f"Missing retrieval cache q={q} tag={tag}")
            if len(cache[q]) != 4:
                raise RuntimeError(
                    f"Expected 4 retrieval branches q={q}, got {len(cache[q])}"
                )
            for i, name in enumerate(BRANCH_NAMES):
                branch_rank[name][q] = docs_from_branch(cache[q][i])

    print("[3/6] Measuring branch complementarity...", flush=True)

    complementarity = {}
    for name in BRANCH_NAMES:
        rows = {}
        for k in DEPTHS:
            rank = branch_rank[name]
            d1_rescued = {
                (q, d)
                for q, d in d1_misses
                if d in set(rank[q][:k])
            }
            outside_rescued = {
                (q, d)
                for q, d in outside_pool
                if d in set(rank[q][:k])
            }

            union_top5_oracle = float(np.mean([
                len(gold[q] & (set(d1[q]) | set(rank[q][:k]))) / len(gold[q])
                for q in ids
            ]))

            rows[str(k)] = {
                "d1_missed_gold_recovered": len(d1_rescued),
                "current_outside_pool_gold_recovered": len(outside_rescued),
                "oracle_d1_union_branch_depth_recall": union_top5_oracle,
                "d1_recovered_cases": [
                    {"qid": q, "doc_id": d}
                    for q, d in sorted(d1_rescued)
                ],
                "outside_recovered_cases": [
                    {"qid": q, "doc_id": d}
                    for q, d in sorted(outside_rescued)
                ],
            }
        complementarity[name] = rows

    print("[4/6] Reconstructing CLEAN vs SELF-ONLY query-memory contracts...", flush=True)

    all_queries = load_queries(root)
    if not all(q in all_queries for q in ids):
        raise RuntimeError("CAL ids are not all present in load_queries()")

    provenance = {}

    for bi, block in enumerate(sorted(blocks), 1):
        held = list(blocks[block])
        held_set = set(held)

        clean_memory = {
            q: v for q, v in all_queries.items()
            if q not in held_set
        }
        all_memory = dict(all_queries)

        print(
            f"  block {block} ({bi}/{len(blocks)}): "
            f"held={len(held)} clean_memory={len(clean_memory)} "
            f"all_memory={len(all_memory)}",
            flush=True,
        )

        clean_conn = build_memory_db(build_query_memory, clean_memory)
        self_conn = build_memory_db(build_query_memory, all_memory)

        clean_overlap = {k: [] for k in (5, 20, 100)}
        self_overlap = {k: [] for k in (5, 20, 100)}
        clean_exact = {k: 0 for k in (5, 20, 100)}
        self_exact = {k: 0 for k in (5, 20, 100)}

        per_query = []

        try:
            for i, q in enumerate(held, 1):
                text = all_queries[q][0]
                cached = branch_rank["QUERY_MEMORY"][q]

                clean = [
                    str(d) for d, _ in memory_scores(
                        clean_conn, text, exclude_qid=q, depth=100
                    )
                ]
                self_only = [
                    str(d) for d, _ in memory_scores(
                        self_conn, text, exclude_qid=q, depth=100
                    )
                ]

                row = {"qid": q}
                for k in (5, 20, 100):
                    co = prefix_overlap(cached, clean, k)
                    so = prefix_overlap(cached, self_only, k)
                    ce = exact_prefix(cached, clean, k)
                    se = exact_prefix(cached, self_only, k)

                    clean_overlap[k].append(co)
                    self_overlap[k].append(so)
                    clean_exact[k] += int(ce)
                    self_exact[k] += int(se)

                    row[f"cached_vs_clean_overlap@{k}"] = co
                    row[f"cached_vs_self_only_overlap@{k}"] = so

                per_query.append(row)

                if i % 25 == 0 or i == len(held):
                    print(
                        f"    memory audit {i}/{len(held)}",
                        flush=True,
                    )
        finally:
            clean_conn.close()
            self_conn.close()

        stats = {
            "held_queries": len(held),
            "cache_tag": cache_paths[block],
            "cached_vs_clean": {
                f"overlap@{k}_mean": float(np.mean(clean_overlap[k]))
                for k in (5, 20, 100)
            } | {
                f"exact_prefix@{k}_queries": clean_exact[k]
                for k in (5, 20, 100)
            },
            "cached_vs_self_only": {
                f"overlap@{k}_mean": float(np.mean(self_overlap[k]))
                for k in (5, 20, 100)
            } | {
                f"exact_prefix@{k}_queries": self_exact[k]
                for k in (5, 20, 100)
            },
            "per_query": per_query,
        }

        # Simple evidence classification, not a proof by itself.
        clean100 = stats["cached_vs_clean"]["overlap@100_mean"]
        self100 = stats["cached_vs_self_only"]["overlap@100_mean"]
        if clean100 >= .999:
            verdict = "CACHED_MEMORY_MATCHES_BLOCK_EXCLUDED_CONTRACT"
        elif self100 >= .999 and self100 > clean100 + .01:
            verdict = "LEAKAGE_RISK_MATCHES_SELF_ONLY_CONTRACT"
        elif self100 > clean100 + .05:
            verdict = "LEAKAGE_RISK_CLOSER_TO_SELF_ONLY"
        elif clean100 > self100 + .05:
            verdict = "CLOSER_TO_BLOCK_EXCLUDED"
        else:
            verdict = "UNRESOLVED_PROVENANCE"

        stats["verdict"] = verdict
        provenance[block] = stats

    print("[5/6] Summarizing leakage/generalization risk...", flush=True)

    block_verdicts = {b: x["verdict"] for b, x in provenance.items()}
    leakage_risk_blocks = [
        b for b, v in block_verdicts.items()
        if v.startswith("LEAKAGE_RISK")
    ]

    # Historical repository status of fresh cache generators is unresolved.
    known_repo_provenance = {
        "validation_a": (
            "main retrieval_train1000_tune50_val100 generator is committed and "
            "historically documented as holdout-excluded"
        ),
        "fresh_1251_1350": "historical repro graph marks generator unresolved",
        "fresh_1351_1450": "historical repro graph marks generator unresolved",
        "fresh_1451_1750": "historical repro graph marks generator unresolved",
    }

    report = {
        "schema": "manual.sparse_memory_provenance_audit_v1",
        "population": {
            "queries": len(ids),
            "d1_missed_gold_occurrences": len(d1_misses),
            "outside_current_pool_gold_occurrences": len(outside_pool),
        },
        "branch_complementarity": complementarity,
        "query_memory_provenance": provenance,
        "block_verdicts": block_verdicts,
        "leakage_risk_blocks": leakage_risk_blocks,
        "known_repo_provenance": known_repo_provenance,
        "interpretation": {
            "public_labels_used": False,
            "cal_gold_used_for_complementarity": True,
            "memory_contract_comparison_uses_no_gold": True,
            "warning": (
                "A mismatch alone is not proof of leakage because historical "
                "cache generation may have excluded a superset of the held block. "
                "A strong match to SELF_ONLY plus poor CLEAN match is the key risk signal."
            ),
        },
    }

    out = root / "results/manual/huy_sparse_memory_provenance_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[6/6] RESULT")
    print("=" * 116)

    for name in BRANCH_NAMES:
        r5 = complementarity[name]["5"]
        r20 = complementarity[name]["20"]
        r100 = complementarity[name]["100"]
        print(
            f"{name:<14s} "
            f"D1miss recovered @5/@20/@100 = "
            f"{r5['d1_missed_gold_recovered']}/"
            f"{r20['d1_missed_gold_recovered']}/"
            f"{r100['d1_missed_gold_recovered']} | "
            f"outside-pool @100={r100['current_outside_pool_gold_recovered']}"
        )

    print("\nQuery-memory provenance:")
    for b in sorted(provenance):
        x = provenance[b]
        print(
            f"  {b}: clean@100="
            f"{x['cached_vs_clean']['overlap@100_mean']:.4f} "
            f"self-only@100="
            f"{x['cached_vs_self_only']['overlap@100_mean']:.4f} "
            f"=> {x['verdict']}"
        )

    print("Leakage-risk blocks:", leakage_risk_blocks or "NONE")
    print("Report:", report_path)
    print("=" * 116)


if __name__ == "__main__":
    main()
