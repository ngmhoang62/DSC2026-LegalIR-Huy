"""Build jina_ft_evidence_union channel: max(old, section) and audit perturbation statistics."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    EVIDENCE_UNION_CACHE_PKL,
    OLD_JINA_CACHE_PKL,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    load_cal_data,
    seed_everything,
    sha256_file,
)


def build_evidence_union_channel() -> dict:
    seed_everything(2026)
    print("=== BUILDING JINA EVIDENCE UNION CHANNEL ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load CAL dataset
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

    # 2. Load Old Jina and Section CE caches
    assert OLD_JINA_CACHE_PKL.exists(), f"Old Jina cache missing: {OLD_JINA_CACHE_PKL}"
    assert SECTION_CE_CACHE_PKL.exists(), f"Section CE cache missing: {SECTION_CE_CACHE_PKL}"

    raw_old = pickle.loads(OLD_JINA_CACHE_PKL.read_bytes())
    if isinstance(raw_old, dict) and "scores" in raw_old:
        old_scores = raw_old["scores"]
    else:
        old_scores = raw_old

    raw_sec = pickle.loads(SECTION_CE_CACHE_PKL.read_bytes())
    if isinstance(raw_sec, dict) and "scores" in raw_sec:
        section_scores = raw_sec["scores"]
    else:
        section_scores = raw_sec

    # 3. Verify 100% coverage on exact D1 candidate pool
    total_pairs = sum(len(extended[q]) for q in all_ids)
    missing_old = []
    missing_sec = []

    for q in all_ids:
        for d in extended[q]:
            if d not in old_scores.get(q, {}):
                missing_old.append((q, d))
            if d not in section_scores.get(q, {}):
                missing_sec.append((q, d))

    assert len(missing_old) == 0, f"FATAL: Missing {len(missing_old)} candidate scores in old jina cache!"
    assert len(missing_sec) == 0, f"FATAL: Missing {len(missing_sec)} candidate scores in section ce cache!"
    print(f"Verified 100% coverage: {len(all_ids)} queries, {total_pairs} candidate pairs in both caches.", flush=True)

    # 4. Construct union channel: max(old, section)
    union_scores: Dict[str, Dict[str, float]] = {}
    old_won_count = 0
    sec_won_count = 0
    tie_count = 0
    diff_vec = []
    within_query_spearmans = []

    for q in all_ids:
        union_scores[q] = {}
        q_old_vec = []
        q_union_vec = []

        for d in extended[q]:
            sc_old = float(old_scores[q][d])
            sc_sec = float(section_scores[q][d])
            sc_union = max(sc_old, sc_sec)
            union_scores[q][d] = sc_union

            diff = sc_union - sc_old
            diff_vec.append(diff)

            if sc_old > sc_sec + 1e-9:
                old_won_count += 1
            elif sc_sec > sc_old + 1e-9:
                sec_won_count += 1
            else:
                tie_count += 1

            q_old_vec.append(sc_old)
            q_union_vec.append(sc_union)

        if len(q_old_vec) > 1 and len(set(q_old_vec)) > 1 and len(set(q_union_vec)) > 1:
            rho, _ = stats.spearmanr(q_old_vec, q_union_vec)
            if np.isfinite(rho):
                within_query_spearmans.append(float(rho))

    # 5. Calculate perturbation statistics
    pct_old_won = float(old_won_count / total_pairs * 100)
    pct_sec_won = float(sec_won_count / total_pairs * 100)
    pct_tie = float(tie_count / total_pairs * 100)

    diff_arr = np.array(diff_vec)
    mean_within_query_spearman = float(np.mean(within_query_spearmans)) if within_query_spearmans else 1.0

    print(f"Total Candidate Pairs: {total_pairs}", flush=True)
    print(f"Old Won:     {old_won_count} ({pct_old_won:.2f}%)", flush=True)
    print(f"Section Won: {sec_won_count} ({pct_sec_won:.2f}%)", flush=True)
    print(f"Ties:        {tie_count} ({pct_tie:.2f}%)", flush=True)
    print(f"Union - Old Distribution: min={np.min(diff_arr):.6f}, 25%={np.percentile(diff_arr, 25):.6f}, median={np.median(diff_arr):.6f}, 75%={np.percentile(diff_arr, 75):.6f}, max={np.max(diff_arr):.6f}, mean={np.mean(diff_arr):.6f}", flush=True)
    print(f"Mean Within-Query Spearman (Old vs Union): {mean_within_query_spearman:.6f}", flush=True)

    # 6. Save union cache
    payload = {
        "scores": union_scores,
        "total_queries": len(all_ids),
        "total_pairs": total_pairs,
        "old_cache_path": str(OLD_JINA_CACHE_PKL).replace("\\", "/"),
        "section_cache_path": str(SECTION_CE_CACHE_PKL).replace("\\", "/"),
    }
    EVIDENCE_UNION_CACHE_PKL.write_bytes(pickle.dumps(payload, protocol=5))
    union_cache_sha256 = sha256_file(EVIDENCE_UNION_CACHE_PKL)
    print(f"Saved {EVIDENCE_UNION_CACHE_PKL} (SHA256: {union_cache_sha256})", flush=True)

    # 7. Write JINA_EVIDENCE_UNION_AUDIT.json
    audit_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_evidence_union_v1.audit.v1",
        "experiment_id": "HUY_D1_JINA_EVIDENCE_UNION_V1",
        "total_queries": len(all_ids),
        "total_candidate_pairs": total_pairs,
        "coverage_complete": True,
        "missing_old_count": 0,
        "missing_section_count": 0,
        "old_cache": {
            "path": str(OLD_JINA_CACHE_PKL).replace("\\", "/"),
            "sha256": sha256_file(OLD_JINA_CACHE_PKL),
        },
        "section_cache": {
            "path": str(SECTION_CE_CACHE_PKL).replace("\\", "/"),
            "sha256": sha256_file(SECTION_CE_CACHE_PKL),
        },
        "union_cache": {
            "path": str(EVIDENCE_UNION_CACHE_PKL).replace("\\", "/"),
            "sha256": union_cache_sha256,
        },
        "score_source_statistics": {
            "old_won_count": old_won_count,
            "old_won_pct": pct_old_won,
            "section_won_count": sec_won_count,
            "section_won_pct": pct_sec_won,
            "tie_count": tie_count,
            "tie_pct": pct_tie,
        },
        "union_minus_old_distribution": {
            "min": float(np.min(diff_arr)),
            "p25": float(np.percentile(diff_arr, 25)),
            "median": float(np.median(diff_arr)),
            "p75": float(np.percentile(diff_arr, 75)),
            "max": float(np.max(diff_arr)),
            "mean": float(np.mean(diff_arr)),
            "std": float(np.std(diff_arr)),
        },
        "mean_within_query_spearman_old_vs_union": mean_within_query_spearman,
    }

    out_audit = RES_DIR / "JINA_EVIDENCE_UNION_AUDIT.json"
    out_audit.write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_audit}", flush=True)

    return audit_doc


if __name__ == "__main__":
    build_evidence_union_channel()
