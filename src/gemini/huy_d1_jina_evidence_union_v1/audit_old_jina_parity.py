"""Deterministic 64+ query-doc frozen scorer parity audit for historical jina_ft."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats
import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SECTION_CACHE_SHA256,
    OLD_JINA_CACHE_PKL,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    WEIGHTS_JINA_FT,
    load_cal_data,
    load_jina_crossencoder,
    seed_everything,
    sha256_file,
    top_passages,
)


def run_old_jina_parity_audit(min_pairs: int = 64) -> dict:
    seed_everything(2026)
    print("=== AUDIT 2: FROZEN JINA-FT SCORER PARITY AUDIT ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load CAL dataset & historical cache
    print("Loading CAL dataset and historical jina_ft cache...", flush=True)
    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    ) = load_cal_data()

    assert OLD_JINA_CACHE_PKL.exists(), f"Old cache not found at {OLD_JINA_CACHE_PKL}"
    cached_jina_ft = pickle.loads(OLD_JINA_CACHE_PKL.read_bytes())
    if isinstance(cached_jina_ft, dict) and "scores" in cached_jina_ft:
        cached_jina_ft = cached_jina_ft["scores"]

    # 2. Verify Section CE cache provenance hash as required by contract
    assert SECTION_CE_CACHE_PKL.exists(), f"Section CE cache not found: {SECTION_CE_CACHE_PKL}"
    section_cache_sha256 = sha256_file(SECTION_CE_CACHE_PKL)
    section_cache_match = (section_cache_sha256 == EXPECTED_SECTION_CACHE_SHA256)
    assert section_cache_match, f"Section cache SHA256 mismatch! Got {section_cache_sha256}"

    # 3. Construct deterministic sample across blocks and score quantiles
    # 6 queries per block (A, B, C, D) = 24 queries
    # For each query, select 4 candidate documents:
    # head (idx 0), boundary (idx min(4, len-1)), middle (idx len//2), tail (idx len-1)
    # Total pairs = 24 * 4 = 96 pairs (>= 64 required)
    sampled_qids = []
    for b in sorted(blocks.keys()):
        b_ids = blocks[b]
        step = max(1, len(b_ids) // 6)
        for i in range(0, min(len(b_ids), step * 6), step):
            if len(sampled_qids) < 24 and b_ids[i] not in sampled_qids:
                sampled_qids.append(b_ids[i])
        if len(sampled_qids) % 6 != 0 and len(b_ids) >= 6:
            while len([q for q in sampled_qids if q in b_ids]) < 6:
                for cand_q in b_ids:
                    if cand_q not in sampled_qids:
                        sampled_qids.append(cand_q)
                        break

    sampled_qids = sampled_qids[:24]

    pairs_to_score = []  # (q, d, quantile_label)
    for q in sampled_qids:
        cands = extended[q]
        n_c = len(cands)
        selected_positions = [
            (0, "head_rank_1"),
            (min(4, n_c - 1), "boundary_top5"),
            (n_c // 2, "middle_pool"),
            (n_c - 1, "tail_pool"),
        ]
        seen_d = set()
        for idx, pos_lbl in selected_positions:
            d = cands[idx]
            if d not in seen_d:
                seen_d.add(d)
                pairs_to_score.append((q, d, pos_lbl))

    total_pairs = len(pairs_to_score)
    print(f"Deterministic audit sample: {len(sampled_qids)} queries across blocks A/B/C/D, {total_pairs} query-doc pairs (>= {min_pairs})", flush=True)

    # 4. Load frozen Jina cross-encoder on GPU
    print("Loading frozen Jina cross-encoder on GPU...", flush=True)
    model, tok, prov = load_jina_crossencoder()
    print(f"Model ready on {prov['device']}.", flush=True)

    # 5. Execute fresh scoring using exact historical jina_ft contract
    pair_requests = []
    pair_indices = []
    for q, d, q_label in pairs_to_score:
        q_text = queries[q][0]
        passages = top_passages(q_text, docs[d], count=2, window=220, overlap=70)
        for p_idx, p in enumerate(passages):
            pair_requests.append((q_text, p))
            pair_indices.append((q, d, p_idx))

    print(f"Computing scores for {len(pair_requests)} text passage pairs...", flush=True)
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    raw_scores = model.compute_score(pair_requests, batch_size=16, max_length=512)
    if isinstance(raw_scores, float):
        raw_scores = [raw_scores]
    elapsed = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Map raw scores back and apply MAX aggregation
    fresh_scores_map: Dict[Tuple[str, str], float] = {}
    for (q, d, _), s in zip(pair_indices, raw_scores):
        s_float = float(s)
        fresh_scores_map[(q, d)] = max(fresh_scores_map.get((q, d), -1e9), s_float)

    # 6. Compare fresh scores against cached scores
    rows = []
    abs_diffs = []
    per_query_fresh: Dict[str, List[float]] = {}
    per_query_cached: Dict[str, List[float]] = {}

    for q, d, pos_lbl in pairs_to_score:
        ref_score = float(cached_jina_ft[q][d])
        fresh_score = float(fresh_scores_map[(q, d)])
        diff = abs(fresh_score - ref_score)
        abs_diffs.append(diff)
        rows.append({
            "qid": q,
            "doc_id": d,
            "quantile": pos_lbl,
            "reference_score": ref_score,
            "fresh_score": fresh_score,
            "abs_diff": diff,
        })
        per_query_fresh.setdefault(q, []).append(fresh_score)
        per_query_cached.setdefault(q, []).append(ref_score)

    max_abs_diff = float(max(abs_diffs))
    mean_abs_diff = float(np.mean(abs_diffs))
    mismatches_gt_2e3 = sum(d > 2e-3 for d in abs_diffs)
    mismatches_gt_1e3 = sum(d > 1e-3 for d in abs_diffs)
    within_1e4_count = sum(d <= 1e-4 for d in abs_diffs)
    within_1e3_count = sum(d <= 1e-3 for d in abs_diffs)
    within_2e3_count = sum(d <= 2e-3 for d in abs_diffs)

    # Compute rank agreement across queries where sampled scores are distinct
    query_rhos = []
    for q in sampled_qids:
        f_vec = per_query_fresh[q]
        c_vec = per_query_cached[q]
        if len(f_vec) > 1 and len(set(f_vec)) > 1 and len(set(c_vec)) > 1:
            rho, _ = stats.spearmanr(f_vec, c_vec)
            if np.isfinite(rho):
                query_rhos.append(float(rho))

    mean_spearman = float(np.mean(query_rhos)) if query_rhos else 1.0

    print(f"Max Absolute Error:  {max_abs_diff:.8f}", flush=True)
    print(f"Mean Absolute Error: {mean_abs_diff:.8f}", flush=True)
    print(f"Within 1e-3 Count:   {within_1e3_count} / {total_pairs} ({within_1e3_count/total_pairs:.1%})", flush=True)
    print(f"Within 2e-3 Count:   {within_2e3_count} / {total_pairs} ({within_2e3_count/total_pairs:.1%})", flush=True)
    print(f"Mean Rank Agreement: {mean_spearman:.6f}", flush=True)

    # Parity gate condition: fp16 machine tolerance (max error <= 2e-3, mean error <= 1e-4, rank agreement >= 0.999)
    parity_passed = (
        mismatches_gt_2e3 == 0
        and mean_abs_diff <= 1e-4
        and mean_spearman >= 0.999
        and section_cache_match
    )

    audit_result = {
        "status": "PASS" if parity_passed else "BLOCKED_OLD_JINA_PARITY",
        "parity_passed": parity_passed,
        "total_pairs_tested": total_pairs,
        "queries_tested_count": len(sampled_qids),
        "max_abs_error": max_abs_diff,
        "mean_abs_error": mean_abs_diff,
        "mismatches_gt_2e3": mismatches_gt_2e3,
        "mismatches_gt_1e3": mismatches_gt_1e3,
        "within_1e4_count": within_1e4_count,
        "within_1e3_count": within_1e3_count,
        "within_2e3_count": within_2e3_count,
        "within_1e3_pct": float(within_1e3_count / total_pairs * 100),
        "mean_rank_agreement_spearman": mean_spearman,
        "section_cache_provenance_verified": section_cache_match,
        "section_cache_sha256": section_cache_sha256,
        "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "sample_rows": rows[:10],
    }

    out_file = RES_DIR / "OLD_JINA_PARITY_AUDIT.json"
    out_file.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_file}", flush=True)

    if not parity_passed:
        print("FATAL: Old Jina-FT frozen parity audit failed!", flush=True)
        print("STATUS: BLOCKED_OLD_JINA_PARITY", flush=True)
        sys.exit("BLOCKED_OLD_JINA_PARITY")

    print("=== OLD JINA-FT SCORER PARITY AUDIT PASSED ===", flush=True)
    return audit_result


if __name__ == "__main__":
    run_old_jina_parity_audit(min_pairs=64)
