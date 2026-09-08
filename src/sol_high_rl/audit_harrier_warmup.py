"""Audit saved Harrier HNSW warmup output as an independent candidate source."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tune_corpus_cap32_fusion import build_training_cap  # noqa: E402


def oracle(queries, pool, ids):
    return float(np.mean([len(set(pool[q]) & queries[q][1]) / len(queries[q][1]) for q in ids]))


def main():
    queries, blocks, all_ids, current, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)
    raw = json.loads((ROOT / "fine_tune/vietlegal_finetuned_results_HNSW/warmup_finetuned_best.json")
                     .read_text(encoding="utf-8"))
    overlap = [q for q in all_ids if q in raw]
    report = {"status": "COMPLETE", "saved_queries": len(raw), "cv_overlap": len(overlap),
              "overlap_by_block": {k: sum(q in set(ids) for q in overlap) for k, ids in blocks.items()},
              "depths": {}}
    for depth in (10, 20):
        harrier = {}
        for q in overlap:
            parents = []
            for item in raw[q].get("results", []):
                doc = str(item["ctx_id"])
                if doc not in parents:
                    parents.append(doc)
                if len(parents) >= depth:
                    break
            harrier[q] = parents
        union = {q: list(dict.fromkeys(current[q] + harrier[q])) for q in overlap}
        missing = {(q, d) for q in overlap for d in queries[q][1] if d not in current[q]}
        rescued = {(q, d) for q, d in missing if d in harrier[q]}
        report["depths"][str(depth)] = {
            "current_oracle": oracle(queries, current, overlap) if overlap else None,
            "source_oracle": oracle(queries, harrier, overlap) if overlap else None,
            "union_oracle": oracle(queries, union, overlap) if overlap else None,
            "union_oracle_delta": (oracle(queries, union, overlap) - oracle(queries, current, overlap)) if overlap else None,
            "missing_gold_occurrences_on_overlap": len(missing),
            "unique_missing_gold_occurrences_recovered": len(rescued),
            "recovered": [{"qid": q, "doc_id": d} for q, d in sorted(rescued)],
            "mean_novel_parents": float(np.mean([sum(d not in current[q] for d in harrier[q]) for q in overlap])) if overlap else None,
        }
    path = ROOT / "results/sol_high_rl/harrier_warmup_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
