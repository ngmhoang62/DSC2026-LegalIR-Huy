"""Cheap deterministic RRF and normalized-score fusion screen."""

from __future__ import annotations

import json
import time

import numpy as np

import run_huy_5fold_fasttrack as core


def rank_rrf(pools, orders, k):
    result = {}
    for qid, docs in pools.items():
        ranks = [{d: i + 1 for i, d in enumerate(order[qid])} for order in orders]
        score = {d: sum(1.0 / (k + rank.get(d, 60)) for rank in ranks) for d in docs}
        result[qid] = sorted(docs, key=lambda d: (-score[d], d))
    return result


def rank_zblend(pools, score_maps):
    result = {}
    for qid, docs in pools.items():
        total = np.zeros(len(docs), dtype=np.float64)
        for channel in score_maps:
            values = np.asarray([channel[qid].get(d, np.nan) for d in docs], dtype=np.float64)
            present = values[~np.isnan(values)]
            mean = float(present.mean()) if present.size else 0.0
            std = (float(present.std()) or 1.0) if present.size else 1.0
            values = np.where(np.isnan(values), mean - 2 * std, values)
            total += (values - mean) / std
        result[qid] = [docs[i] for i in np.lexsort((np.asarray(docs), -total))]
    return result


def main():
    started = time.perf_counter()
    folds, pools, _, golds, e5_orders, e5_scores, _, _ = core.load_inputs()
    jina_order, jina_scores, _ = core.load_jina(pools)
    lal_order, lal_scores, _ = core.load_source_channel("lal", pools)
    reference = {
        str(row["qid"]): list(map(str, row["top5"]))
        for row in core.read_jsonl(core.OUT / "BEST_5FOLD_PREDICTIONS.jsonl")
    }
    orders = [jina_order, e5_orders["adapted_e5"], lal_order]
    score_maps = [jina_scores, e5_scores["adapted_e5"], lal_scores]
    predictions = {f"equal_rrf_k{k}": rank_rrf(pools, orders, k) for k in (0, 10, 32, 60)}
    predictions["equal_query_zscore_blend"] = rank_zblend(pools, score_maps)
    variants = {
        name: {
            "metrics": core.metrics(pred, golds, folds),
            "paired_vs_lr_best": core.compare(pred, reference, golds, folds),
        }
        for name, pred in predictions.items()
    }
    best = max(variants, key=lambda n: variants[n]["metrics"]["recall_at_5"])
    report = {
        "schema_version": "dsc2026.huy_fasttrack.simple_fusion_screen.v1",
        "status": "COMPLETE_CACHE_ONLY",
        "experts": ["Huy Jina-v2 lexical CE", "adapted VietLegal-E5", "LegalIR native LAL"],
        "variants": variants,
        "best_simple_fusion": best,
        "verdict": "KEEP" if variants[best]["paired_vs_lr_best"]["delta_recall_at_5"] >= .001 else "DROP",
        "runtime_seconds": time.perf_counter() - started,
    }
    core.write_json(core.OUT / "HUY_SIMPLE_FUSION_SCREEN.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
