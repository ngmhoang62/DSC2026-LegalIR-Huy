"""Evaluate the fixed first-10-novel append policy on immutable source ranks."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_expanded_rerank_holdouts import load_expanded  # noqa: E402
from tune_corpus_cap32_fusion import build_training_cap  # noqa: E402
from tune_expanded_fusion_robust import TAGS  # noqa: E402


def novel(source, incumbent, quota=10):
    return [d for d in source if d not in set(incumbent)][:quota]


def oracle(queries, pool, ids):
    return float(np.mean([len(set(pool[q]) & queries[q][1]) / len(queries[q][1]) for q in ids]))


def main():
    queries, blocks, all_ids, current, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)
    expansion = pickle.loads((ROOT / "results/dense_expansion/union50_scores.pkl").read_bytes())["scores"]
    _, dense, _ = load_expanded(ROOT, queries, {TAGS[k]: v for k, v in blocks.items()}, expansion)
    corpus = pickle.loads((ROOT / "results/corpus_index/holdout_dense_rank_cap32.pkl").read_bytes())["ranking"]
    additions = {
        "dense_expanded_novel10": {q: novel(dense[q], current[q]) for q in all_ids},
        "corpus_dense_novel10": {q: novel(corpus[q], current[q]) for q in all_ids},
    }
    additions["union_both_novel10"] = {
        q: list(dict.fromkeys(additions["dense_expanded_novel10"][q] +
                              additions["corpus_dense_novel10"][q])) for q in all_ids
    }
    current_oracle = oracle(queries, current, all_ids)
    missing = {(q, d) for q in all_ids for d in queries[q][1] if d not in current[q]}
    report = {"status": "COMPLETE", "quota": 10, "current_oracle": current_oracle, "policies": {}}
    for name, add in additions.items():
        pool = {q: current[q] + add[q] for q in all_ids}
        rescued = {(q, d) for q, d in missing if d in add[q]}
        report["policies"][name] = {
            "mean_added": float(np.mean([len(add[q]) for q in all_ids])),
            "union_oracle": oracle(queries, pool, all_ids),
            "union_oracle_delta": oracle(queries, pool, all_ids) - current_oracle,
            "rescued_missing_gold_occurrences": len(rescued),
            "recovered": [{"qid": q, "doc_id": d} for q, d in sorted(rescued)],
            "per_block": {block: {
                "union_oracle": oracle(queries, pool, ids),
                "rescued": sum(q in set(ids) for q, _ in rescued),
            } for block, ids in blocks.items()},
        }
    path = ROOT / "results/sol_high_rl/bounded_novel_append_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
