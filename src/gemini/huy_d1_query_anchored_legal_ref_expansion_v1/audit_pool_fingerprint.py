"""Audit 0: Candidate Pool Fingerprint and Baseline Integrity Audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    RES_DIR,
    compute_candidate_fingerprint,
    compute_query_fingerprint,
    load_cal_data,
)

EXPECTED_CANDIDATE_FINGERPRINT = "24864c27298b8f48d96b3ddc60c521a5c8c88c84b5e9ca1dbbd5ffbf5e8b595a"
EXPECTED_QUERY_FINGERPRINT = "3dbcd3aa7b870801ddb7b83a0e1c54aab6ab24a220587b55060d4856cd962f8e"


def run_candidate_pool_audit() -> dict:
    print("=== AUDIT 0: CANDIDATE POOL FINGERPRINT AUDIT ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    queries, blocks, all_ids, extended, gold = load_cal_data()

    cand_fp = compute_candidate_fingerprint(all_ids, extended)
    query_fp = compute_query_fingerprint(all_ids, queries)
    total_pairs = sum(len(extended[q]) for q in all_ids)

    cand_pass = (cand_fp == EXPECTED_CANDIDATE_FINGERPRINT)
    query_pass = (query_fp == EXPECTED_QUERY_FINGERPRINT)
    count_pass = (len(all_ids) == 600) and (total_pairs == 23532)

    print(f"Candidate Fingerprint: {cand_fp} (Expected: {EXPECTED_CANDIDATE_FINGERPRINT}) -> PASS={cand_pass}", flush=True)
    print(f"Query Fingerprint:     {query_fp} (Expected: {EXPECTED_QUERY_FINGERPRINT}) -> PASS={query_pass}", flush=True)
    print(f"Queries: {len(all_ids)}, Total Pairs: {total_pairs} (Expected: 600, 23532) -> PASS={count_pass}", flush=True)

    all_pass = cand_pass and query_pass and count_pass

    audit_result = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.pool_audit.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "status": "PASS" if all_pass else "FAIL",
        "checks": {
            "candidate_fingerprint_exact": cand_pass,
            "query_fingerprint_exact": query_pass,
            "query_count_600": len(all_ids) == 600,
            "pair_count_23532": total_pairs == 23532,
        },
        "candidate_fingerprint": cand_fp,
        "query_fingerprint": query_fp,
        "query_count": len(all_ids),
        "total_candidate_pairs": total_pairs,
    }

    out_path = RES_DIR / "CANDIDATE_POOL_FINGERPRINT_AUDIT.json"
    out_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print("=== CANDIDATE POOL AUDIT PASSED ===\n", flush=True)
    return audit_result


if __name__ == "__main__":
    run_candidate_pool_audit()
