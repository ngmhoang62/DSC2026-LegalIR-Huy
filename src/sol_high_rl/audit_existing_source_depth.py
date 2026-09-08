"""Trace current out-of-pool golds through immutable deeper source caches."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_expanded_rerank_holdouts import load_expanded  # noqa: E402
from benchmark_jina_reranker_holdouts import load_cache  # noqa: E402
from tune_corpus_cap32_fusion import build_training_cap  # noqa: E402
from tune_expanded_fusion_robust import TAGS  # noqa: E402


def main():
    queries, blocks, all_ids, current, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)
    expansion_scores = pickle.loads((ROOT / "results/dense_expansion/union50_scores.pkl").read_bytes())["scores"]
    tagged = {TAGS[name]: ids for name, ids in blocks.items()}
    raw, dense, expanded = load_expanded(ROOT, queries, tagged, expansion_scores)
    corpus = pickle.loads((ROOT / "results/corpus_index/holdout_dense_rank_cap32.pkl").read_bytes())["ranking"]
    branches = {i: {} for i in range(4)}
    for block, ids in blocks.items():
        cache = load_cache(ROOT, TAGS[block])
        for q in ids:
            for i, branch in enumerate(cache[q]):
                branches[i][q] = [str(item[0]) for item in branch]

    sources = {"raw_union": raw, "dense_expanded": dense,
               "expanded_rrf": expanded, "corpus_dense": corpus}
    sources.update({f"raw_branch_{i}": table for i, table in branches.items()})
    missing = [(q, d) for q in all_ids for d in queries[q][1] if d not in current[q]]
    details = []
    for q, d in missing:
        details.append({"qid": q, "doc_id": d,
                        "ranks": {name: (table[q].index(d) + 1 if d in table[q] else None)
                                  for name, table in sources.items()}})
    report = {"status": "COMPLETE", "out_of_pool_gold_occurrences": len(missing),
              "details": details, "depths": {}}
    for depth in (30, 40, 50, 100):
        report["depths"][str(depth)] = {}
        for name, table in sources.items():
            rescued = [(q, d) for q, d in missing if d in table[q][:depth]]
            novel = [sum(d not in current[q] for d in table[q][:depth]) for q in all_ids]
            union = {q: list(dict.fromkeys(current[q] + table[q][:depth])) for q in all_ids}
            current_oracle = float(np.mean([len(set(current[q]) & queries[q][1]) / len(queries[q][1]) for q in all_ids]))
            union_oracle = float(np.mean([len(set(union[q]) & queries[q][1]) / len(queries[q][1]) for q in all_ids]))
            report["depths"][str(depth)][name] = {
                "rescued_missing_gold_occurrences": len(rescued),
                "union_oracle_delta": union_oracle - current_oracle,
                "mean_novel_documents": float(np.mean(novel)),
            }
    path = ROOT / "results/sol_high_rl/existing_source_depth_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
