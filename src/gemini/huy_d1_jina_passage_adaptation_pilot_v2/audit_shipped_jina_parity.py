"""Audit shipped Jina initialization and frozen score parity."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats
import torch

from .common import (
    EXPECTED_SHIPPED_SHA256,
    FROZEN_SECTION_CE_CV_PKL,
    REPO_JINA,
    RES_DIR,
    ROOT,
    WEIGHTS_JINA_FT,
    get_git_status,
    load_cal_candidate_pools,
    load_cal_contexts,
    load_cal_questions_label_free,
    load_jina_base_with_shipped_weights,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import parse_document_into_sections, preselect_legal_sections


def run_shipped_jina_parity(sample_size: int = 32) -> Dict[str, Any]:
    print("=== AUDIT: SHIPPED JINA INITIALIZATION & PARITY ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Verify weights SHA256
    chk_sha = sha256_file(WEIGHTS_JINA_FT)
    print(f"[PARITY] Shipped weights SHA256: {chk_sha}", flush=True)
    if chk_sha != EXPECTED_SHIPPED_SHA256:
        raise RuntimeError(
            f"BLOCKED_SHIPPED_JINA_PARITY: Shipped weights SHA256 mismatch!\n"
            f"Expected: {EXPECTED_SHIPPED_SHA256}\n"
            f"Actual:   {chk_sha}"
        )

    # 2. Instantiate and load weights
    model, tok = load_jina_base_with_shipped_weights()
    model.eval().to("cuda")

    # 3. Check parity against authoritative legal_section_ce_cv.pkl
    if not FROZEN_SECTION_CE_CV_PKL.exists():
        raise FileNotFoundError(f"Authoritative section CE cache not found: {FROZEN_SECTION_CE_CV_PKL}")

    cached_data = pickle.loads(FROZEN_SECTION_CE_CV_PKL.read_bytes())
    cached_scores: Dict[str, Dict[str, float]] = cached_data["scores"]

    contexts = load_cal_contexts()
    all_ids, queries = load_cal_questions_label_free()
    extended = load_cal_candidate_pools()

    sample_qids = all_ids[:sample_size]
    max_score_diff = 0.0
    spearmans = []
    top5_agreements = []

    print(f"[PARITY] Evaluating {sample_size} CAL queries against authoritative section CE cache...", flush=True)

    for qid in sample_qids:
        q_text = queries[qid]
        cand_docs = extended[qid]

        fresh_scores: Dict[str, float] = {}
        pair_list: List[tuple] = []
        pair_meta: List[str] = []

        for doc_id in cand_docs:
            raw_text = contexts.get(doc_id, "")
            secs = parse_document_into_sections(doc_id, raw_text, max_chunk_words=220, overlap_words=60)
            chosen_secs = preselect_legal_sections(q_text, secs, count=2)
            for s in chosen_secs:
                pair_list.append((q_text, s.text))
                pair_meta.append(doc_id)

        # Compute scores with exact compute_score method
        raw_vals = model.compute_score(pair_list, batch_size=32, max_length=512)
        if isinstance(raw_vals, float):
            raw_vals = [raw_vals]

        for doc_id, s_val in zip(pair_meta, raw_vals):
            s_flt = float(s_val)
            fresh_scores[doc_id] = max(fresh_scores.get(doc_id, -1e9), s_flt)

        # Compare with cached scores
        cached_q = cached_scores[qid]
        f_vec = [fresh_scores[d] for d in cand_docs]
        c_vec = [cached_q[d] for d in cand_docs]

        diff = max(abs(f - c) for f, c in zip(f_vec, c_vec))
        max_score_diff = max(max_score_diff, diff)

        rho, _ = stats.spearmanr(f_vec, c_vec)
        if np.isfinite(rho):
            spearmans.append(float(rho))

        f_top5 = set(sorted(cand_docs, key=lambda d: fresh_scores[d], reverse=True)[:5])
        c_top5 = set(sorted(cand_docs, key=lambda d: cached_q[d], reverse=True)[:5])
        top5_agreements.append(f_top5 == c_top5)

    mean_spearman = float(np.mean(spearmans)) if spearmans else 1.0
    top5_agree_rate = float(np.mean(top5_agreements)) if top5_agreements else 1.0

    print(
        f"[PARITY] Result: max_score_diff={max_score_diff:.8f}, "
        f"mean_spearman={mean_spearman:.6f}, top5_agree={top5_agree_rate:.2%}",
        flush=True,
    )

    passed = (max_score_diff < 0.01) and (top5_agree_rate == 1.0) and (mean_spearman > 0.9999)
    if not passed:
        raise RuntimeError(
            f"BLOCKED_SHIPPED_JINA_PARITY: Frozen parity failed!\n"
            f"max_diff={max_score_diff}, top5_agree={top5_agree_rate}, spearman={mean_spearman}"
        )

    report = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.shipped_jina_parity.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "shipped_weights_path": str(WEIGHTS_JINA_FT.relative_to(ROOT)),
        "shipped_weights_sha256": chk_sha,
        "shipped_weights_expected_sha256": EXPECTED_SHIPPED_SHA256,
        "sha256_match": True,
        "sample_size_queries": sample_size,
        "max_score_difference": max_score_diff,
        "mean_spearman": mean_spearman,
        "top5_agreement_fraction": top5_agree_rate,
        "authoritative_cache_reference": str(FROZEN_SECTION_CE_CV_PKL.relative_to(ROOT)),
    }

    out_path = RES_DIR / "SHIPPED_JINA_INITIALIZATION_PARITY.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PARITY] Saved -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    run_shipped_jina_parity()
