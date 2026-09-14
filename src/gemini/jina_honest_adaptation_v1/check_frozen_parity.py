"""Check frozen Jina baseline score and ranking parity against authoritative cache.
Generates results/gemini/jina_honest_adaptation_v1/FROZEN_JINA_PARITY.json.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    EXP_RESULTS,
    REPO_ROOT,
    get_git_info,
    log_execution_trace,
    sha256_file,
    sha256_text,
)

# Insert repo root and forensic source for exact top_passages
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src/research_v2_forensic"))

from benchmark_jina_reranker_holdouts import top_passages
import jina_v2_boundary_train as jb


def run_parity_check():
    start_time = time.perf_counter()
    jb.patch_transformers_v5()

    model_path = REPO_ROOT / "cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"
    print("Loading base Jina model on CUDA...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float16
    ).eval().to("cuda")
    model._tokenizer = tok

    # Load train data
    train_path = REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Load candidate pool
    pool_path = REPO_ROOT / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"
    pools = {}
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pools[str(row["qid"])] = [str(x) for x in row["doc_ids"]]

    # Load canonical contexts
    contexts_path = REPO_ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
    print("Loading canonical contexts...", flush=True)
    contexts = {}
    with open(contexts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                contexts[str(rec["doc_id"])] = str(rec["passage"] or "")

    # Open authoritative cache
    db_path = REPO_ROOT / "cache/research_v2_forensic/evidence_ab_scores.sqlite"
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    cur = conn.cursor()

    # Deterministic audit sample: 5 queries = 254 pairs (>= 200 requirement)
    sample_qids = sorted(pools.keys(), key=int)[:5]
    pairs_to_test = []
    for qid in sample_qids:
        for doc in pools[qid]:
            pairs_to_test.append((qid, doc))

    print(f"Testing parity on {len(sample_qids)} queries, {len(pairs_to_test)} pairs...", flush=True)

    cache_scores = {}
    for qid, doc in pairs_to_test:
        sc = cur.execute(
            "SELECT score FROM scores WHERE arm='lexical' AND qid=? AND doc_id=?", (qid, doc)
        ).fetchone()
        if sc is None:
            raise RuntimeError(f"Missing cache score for {qid}, {doc}")
        cache_scores[(qid, doc)] = float(sc[0])

    # Prepare passages and verify passage hashing
    all_items = []
    pair_indices = []
    passage_hashes = []
    for qid, doc in pairs_to_test:
        qtext = train_data[qid]["question"]
        dtext = contexts[doc]
        passages = top_passages(qtext, dtext, count=2, window=220, overlap=70)
        for p in passages:
            all_items.append((qtext, p))
            pair_indices.append((qid, doc))
            passage_hashes.append(sha256_text(p))

    # Compute scores with exact batch_size=16, max_length=512
    with torch.no_grad():
        raw_scores = model.compute_score(all_items, batch_size=16, max_length=512)

    recomputed_parent = {}
    for (qid, doc), score in zip(pair_indices, raw_scores):
        recomputed_parent[(qid, doc)] = max(recomputed_parent.get((qid, doc), -1e9), float(score))

    # Analysis
    diffs = []
    rank_correlations = []
    top1_matches = 0
    top5_matches = 0

    for qid in sample_qids:
        docs = pools[qid]
        c_scores = [cache_scores[(qid, d)] for d in docs]
        r_scores = [recomputed_parent[(qid, d)] for d in docs]
        for c, r in zip(c_scores, r_scores):
            diffs.append(abs(c - r))

        # Sort docs by cached vs recomputed
        c_order = [d for _, d in sorted(zip(c_scores, docs), reverse=True)]
        r_order = [d for _, d in sorted(zip(r_scores, docs), reverse=True)]

        if c_order[0] == r_order[0]:
            top1_matches += 1
        if set(c_order[:5]) == set(r_order[:5]):
            top5_matches += 1

        corr, _ = scipy.stats.spearmanr(c_scores, r_scores)
        rank_correlations.append(float(corr))

    max_diff = float(max(diffs))
    mean_diff = float(sum(diffs) / len(diffs))
    mismatches_1e4 = sum(1 for d in diffs if d > 1e-4)
    mismatches_1e3 = sum(1 for d in diffs if d > 1e-3)
    mean_spearman = float(sum(rank_correlations) / len(rank_correlations))

    parity_passed = max_diff < 0.001 and mean_spearman > 0.9999 and top1_matches == len(sample_qids)

    report = {
        "schema_version": "dsc2026.gemini.frozen_jina_parity.v1",
        "status": "PASS" if parity_passed else "FAIL",
        "audit_queries_count": len(sample_qids),
        "audit_pairs_count": len(pairs_to_test),
        "passages_scored_count": len(all_items),
        "tolerance": {
            "max_absolute_difference": max_diff,
            "mean_absolute_difference": mean_diff,
            "mismatches_gt_1e4": mismatches_1e4,
            "mismatches_gt_1e3": mismatches_1e3,
            "mean_spearman_rank_correlation": mean_spearman,
            "top1_ranking_match_rate": top1_matches / len(sample_qids),
            "top5_set_match_rate": top5_matches / len(sample_qids),
        },
        "sample_comparisons": [
            {
                "qid": qid,
                "doc_id": doc,
                "cached_score": cache_scores[(qid, doc)],
                "recomputed_score": recomputed_parent[(qid, doc)],
                "abs_diff": abs(cache_scores[(qid, doc)] - recomputed_parent[(qid, doc)]),
            }
            for qid, doc in pairs_to_test[:10]
        ],
        "evidence_contract": {
            "renderer": "R0_HUY_LOCKED",
            "count": 2,
            "window": 220,
            "overlap": 70,
            "max_length": 512,
            "parent_aggregation": "max",
        },
        "model_fingerprint": {
            "path": str(model_path),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "sha256_first_passage": passage_hashes[0],
        },
        "git": get_git_info(),
        "runtime_sec": round(time.perf_counter() - start_time, 3),
    }

    out_path = EXP_RESULTS / "FROZEN_JINA_PARITY.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Parity status: {report['status']}, max_diff: {max_diff:.6e}, mean_spearman: {mean_spearman:.6f}")

    log_execution_trace(
        stage_name="check_frozen_parity",
        command=f"python {__file__}",
        code_hash=sha256_file(Path(__file__)),
        model_checkpoint_hash=sha256_file(model_path / "model.safetensors")
        if (model_path / "model.safetensors").exists()
        else "N/A",
        train_query_count=0,
        val_query_count=len(sample_qids),
        optimizer_steps=0,
        gpu_info=torch.cuda.get_device_name(0),
        runtime_sec=time.perf_counter() - start_time,
        output_hashes={"FROZEN_JINA_PARITY.json": sha256_file(out_path)},
        status=report["status"],
        extra={"max_diff": max_diff, "mean_spearman": mean_spearman},
    )
    conn.close()


if __name__ == "__main__":
    run_parity_check()
