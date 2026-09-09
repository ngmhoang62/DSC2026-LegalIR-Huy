"""Mandatory AITeamVN-FT inference-contract parity gate on CAL600."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_jina_reranker_holdouts import top_passages
from full_corpus_title_retrieval import load_model
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap


OUT = ROOT / "results/sol_high_rl/AITEAM_FT_INFERENCE_PARITY.json"


def main():
    started = time.perf_counter()
    queries, old_blocks, all_ids, candidates, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    split = json.loads((ROOT / "results/sol_high_rl/CAL600_STRATIFIED_5FOLD_SEED42.json").read_text(encoding="utf-8"))
    sample = []
    for fold, ids in split["folds"].items():
        for index in (0, 40, 80, 119):
            sample.append(ids[index])
    reference = pickle.loads((ROOT / "results/from_drive/aiteamvn_ft_cv.pkl").read_bytes())
    docs = DocumentStore(
        sorted((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")),
        cache_size=128,
    )
    model, tokenizer = load_model()
    rebuilt = {}
    for i, qid in enumerate(sample, 1):
        question = queries[qid][0]
        qvec = encode_cls(model, tokenizer, [question], 1, 512)[0]
        owners, passages = [], []
        for docid in candidates[qid]:
            for passage in top_passages(question, docs[docid], count=2):
                owners.append(docid)
                passages.append(passage)
        pvec = encode_cls(model, tokenizer, passages, 24, 512)
        row = {}
        for docid, score in zip(owners, pvec @ qvec):
            row[docid] = max(row.get(docid, -1e9), float(score))
        rebuilt[qid] = row
        print(f"parity scored {i}/{len(sample)} qid={qid} docs={len(row)}", flush=True)

    raw_reference, raw_rebuilt, per_query = [], [], []
    block_of = {q: block for block, ids in old_blocks.items() for q in ids}
    fold_of = {q: fold for fold, ids in split["folds"].items() for q in ids}
    for qid in sample:
        docs_q = candidates[qid]
        a = np.asarray([float(reference[qid][d]) for d in docs_q])
        b = np.asarray([float(rebuilt[qid][d]) for d in docs_q])
        raw_reference.extend(a.tolist())
        raw_rebuilt.extend(b.tolist())
        rank_a = sorted(docs_q, key=lambda d: (-float(reference[qid][d]), d))
        rank_b = sorted(docs_q, key=lambda d: (-float(rebuilt[qid][d]), d))
        per_query.append(
            {
                "qid": qid,
                "fold": fold_of[qid],
                "old_block": block_of[qid],
                "documents": len(docs_q),
                "pearson": float(pearsonr(a, b).statistic),
                "spearman": float(spearmanr(a, b).statistic),
                "mean_abs_diff": float(np.mean(np.abs(a - b))),
                "max_abs_diff": float(np.max(np.abs(a - b))),
                "top5_set_agreement": len(set(rank_a[:5]) & set(rank_b[:5])) / 5,
                "top10_set_agreement": len(set(rank_a[:10]) & set(rank_b[:10])) / 10,
                "top1_equal": rank_a[0] == rank_b[0],
            }
        )
    raw_reference = np.asarray(raw_reference)
    raw_rebuilt = np.asarray(raw_rebuilt)
    aggregate = {
        "score_pairs": len(raw_reference),
        "pearson": float(pearsonr(raw_reference, raw_rebuilt).statistic),
        "spearman": float(spearmanr(raw_reference, raw_rebuilt).statistic),
        "mean_abs_diff": float(np.mean(np.abs(raw_reference - raw_rebuilt))),
        "median_abs_diff": float(np.median(np.abs(raw_reference - raw_rebuilt))),
        "max_abs_diff": float(np.max(np.abs(raw_reference - raw_rebuilt))),
        "median_query_top5_set_agreement": float(np.median([x["top5_set_agreement"] for x in per_query])),
        "mean_query_top5_set_agreement": float(np.mean([x["top5_set_agreement"] for x in per_query])),
        "mean_query_top10_set_agreement": float(np.mean([x["top10_set_agreement"] for x in per_query])),
        "top1_equal_queries": sum(x["top1_equal"] for x in per_query),
    }
    thresholds = {
        "pearson_min": 0.999,
        "spearman_min": 0.995,
        "mean_abs_diff_max": 0.002,
        "median_query_top5_set_agreement_min": 0.8,
    }
    passed = (
        aggregate["pearson"] >= thresholds["pearson_min"]
        and aggregate["spearman"] >= thresholds["spearman_min"]
        and aggregate["mean_abs_diff"] <= thresholds["mean_abs_diff_max"]
        and aggregate["median_query_top5_set_agreement"] >= thresholds["median_query_top5_set_agreement_min"]
    )
    result = {
        "status": "PASS_CONTRACT" if passed else "BLOCKED_CONTRACT",
        "contract": {
            "checkpoint": "fine_tune/AITeamVN_Vietnamese_Embedding",
            "pooling": "CLS+L2",
            "prefixes": "none",
            "max_length": 512,
            "passage_selector": "exact Huy top_passages(count=2)",
            "document_aggregation": "max cosine",
        },
        "sample_policy": "four deterministic positions [0,40,80,119] from each sealed CAL600 fold; all current candidates",
        "sample_qids": sample,
        "thresholds_predeclared": thresholds,
        "aggregate": aggregate,
        "per_query": per_query,
        "runtime_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_query"}, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
