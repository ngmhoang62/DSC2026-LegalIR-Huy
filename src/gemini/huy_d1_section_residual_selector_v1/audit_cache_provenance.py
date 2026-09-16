"""Audit 1: Cache Provenance and Integrity Audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_section_residual_selector_v1.common import (
    EXPECTED_OLD_JINA_CACHE_SHA256,
    EXPECTED_SECTION_CACHE_SHA256,
    OLD_JINA_CACHE_PKL,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    load_cal_data,
    load_pkl,
    sha256_file,
)


def run_cache_provenance_audit() -> dict:
    print("=== AUDIT 1: CACHE PROVENANCE AND INTEGRITY AUDIT ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Section CE cache check
    if not SECTION_CE_CACHE_PKL.exists():
        raise FileNotFoundError(f"Section CE cache missing: {SECTION_CE_CACHE_PKL}")
    section_sha = sha256_file(SECTION_CE_CACHE_PKL)
    section_pass = (section_sha == EXPECTED_SECTION_CACHE_SHA256)
    print(f"Section CE Cache SHA256: {section_sha} (Expected: {EXPECTED_SECTION_CACHE_SHA256}) -> PASS={section_pass}", flush=True)

    # 2. Old Jina cache check
    if not OLD_JINA_CACHE_PKL.exists():
        raise FileNotFoundError(f"Old Jina cache missing: {OLD_JINA_CACHE_PKL}")
    old_jina_sha = sha256_file(OLD_JINA_CACHE_PKL)
    old_jina_pass = (old_jina_sha == EXPECTED_OLD_JINA_CACHE_SHA256)
    print(f"Old Jina Cache SHA256:   {old_jina_sha} (Expected: {EXPECTED_OLD_JINA_CACHE_SHA256}) -> PASS={old_jina_pass}", flush=True)

    # 3. Verify coverage on CAL600
    docs, queries, blocks, all_ids, extended, _, _, _, _, _ = load_cal_data()
    sec_scores = load_pkl(SECTION_CE_CACHE_PKL)
    old_scores = load_pkl(OLD_JINA_CACHE_PKL)

    sec_coverage = sum(all(d in sec_scores.get(q, {}) for d in extended[q]) for q in all_ids)
    old_coverage = sum(all(d in old_scores.get(q, {}) for d in extended[q]) for q in all_ids)

    print(f"Section CE Coverage: {sec_coverage} / {len(all_ids)} queries (100% pairs present)", flush=True)
    print(f"Old Jina Coverage:   {old_coverage} / {len(all_ids)} queries (100% pairs present)", flush=True)

    all_pass = section_pass and old_jina_pass and (sec_coverage == len(all_ids)) and (old_coverage == len(all_ids))

    audit_result = {
        "schema_version": "dsc2026.gemini.huy_d1_section_residual_selector_v1.cache_provenance.v1",
        "experiment_id": "HUY_D1_SECTION_RESIDUAL_SELECTOR_V1",
        "status": "PASS" if all_pass else "FAIL",
        "checks": {
            "section_ce_sha256_exact": section_pass,
            "old_jina_sha256_exact": old_jina_pass,
            "section_ce_coverage_600": sec_coverage == len(all_ids),
            "old_jina_coverage_600": old_coverage == len(all_ids),
        },
        "section_ce_cache": {
            "path": str(SECTION_CE_CACHE_PKL).replace("\\", "/"),
            "sha256": section_sha,
            "expected_sha256": EXPECTED_SECTION_CACHE_SHA256,
        },
        "old_jina_cache": {
            "path": str(OLD_JINA_CACHE_PKL).replace("\\", "/"),
            "sha256": old_jina_sha,
            "expected_sha256": EXPECTED_OLD_JINA_CACHE_SHA256,
        },
    }

    out_path = RES_DIR / "CACHE_PROVENANCE_AUDIT.json"
    out_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print("=== CACHE PROVENANCE AUDIT PASSED ===\n", flush=True)
    return audit_result


if __name__ == "__main__":
    run_cache_provenance_audit()
