"""Headroom anatomy for the current 44D strict-V2 endpoint."""

from __future__ import annotations

import json
import numpy as np

import run_huy_5fold_fasttrack as core


def recall_at(pred, golds, k):
    return float(np.mean([
        len(set(pred[qid][:k]) & golds[qid]) / len(golds[qid])
        for qid in golds
    ]))


def main():
    folds, pools, questions, golds, e5_orders, *_ = core.load_inputs()
    current = {
        str(row["qid"]): list(map(str, row["order"]))
        for row in core.read_jsonl(core.OUT / "learner_prediction_locks/profile_memory_plus_sparse_rank_scores.jsonl")
    }
    jina_ce, _, _ = core.load_jina(pools)
    source = {}
    for name in ("e5", "lal", "jina", "bm25", "trigram"):
        source[name], _, _ = core.load_source_channel(name, pools)
    source["adapted_e5"] = e5_orders["adapted_e5"]
    source["huy_jina_ce"] = jina_ce
    union = {}
    for qid in golds:
        docs = set(current[qid][:5])
        for order in source.values():
            docs.update(order[qid][:5])
        union[qid] = list(docs)
    candidate_ceiling = float(np.mean([
        len(set(pools[qid]) & golds[qid]) / len(golds[qid]) for qid in golds
    ]))
    report = {
        "status": "COMPLETE",
        "current_recall_at": {str(k): recall_at(current, golds, k) for k in (5, 6, 7, 8, 10, 15, 20)},
        "native_source_top5_recall": {name: recall_at(order, golds, 5) for name, order in source.items()},
        "current_plus_native_top5_union_oracle": recall_at(union, golds, 100),
        "candidate_ceiling_recall": candidate_ceiling,
        "headroom_to_candidate_ceiling": candidate_ceiling - recall_at(current, golds, 5),
        "headroom_current_rank_6_10": recall_at(current, golds, 10) - recall_at(current, golds, 5),
        "headroom_current_rank_6_20": recall_at(current, golds, 20) - recall_at(current, golds, 5),
    }
    core.write_json(core.OUT / "CURRENT_HEADROOM_DIAGNOSTIC.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
