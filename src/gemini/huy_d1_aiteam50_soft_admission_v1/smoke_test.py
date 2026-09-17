"""Fast smoke test for HUY_D1_AITEAM50_SOFT_ADMISSION_V1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import (
    AITEAM_REPORT_PATH,
    AITEAM_TOP50_PATH,
    CAL600_5FOLD_PATH,
    CAL_FROZEN_SECTION_CACHE_PATH,
    EXPECTED_5FOLD_SHA256,
    EXPECTED_AITEAM_REPORT_SHA256,
    EXPECTED_AITEAM_TOP50_SHA256,
    EXPECTED_JINA_FT_CV_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    JINA_FT_CV_PATH,
    WEIGHTS_JINA_FT,
    get_git_status,
    get_source_files_sha256,
    sha256_file,
)


def run_smoke():
    print("=== HUY_D1_AITEAM50_SOFT_ADMISSION_V1 SMOKE TEST ===", flush=True)

    # 1. Check file existence and hashes
    checks = [
        ("AITEAM_TOP50", AITEAM_TOP50_PATH, EXPECTED_AITEAM_TOP50_SHA256),
        ("AITEAM_REPORT", AITEAM_REPORT_PATH, EXPECTED_AITEAM_REPORT_SHA256),
        ("CAL600_5FOLD", CAL600_5FOLD_PATH, EXPECTED_5FOLD_SHA256),
        ("JINA_FT_CV", JINA_FT_CV_PATH, EXPECTED_JINA_FT_CV_SHA256),
        ("SECTION_CE_CV", CAL_FROZEN_SECTION_CACHE_PATH, EXPECTED_SECTION_CE_SHA256),
    ]

    all_ok = True
    for name, path, expected in checks:
        exists = path.exists()
        actual = sha256_file(path) if exists else "MISSING"
        match = (actual == expected)
        print(f"[{'PASS' if match else 'FAIL'}] {name}: exists={exists}, sha_match={match}")
        if not match:
            all_ok = False

    # 2. Check source files
    src_shas = get_source_files_sha256()
    print(f"Source files found: {len(src_shas)}")
    for fname, meta in src_shas.items():
        print(f"  {fname}: {meta['size_bytes']} bytes, SHA256: {meta['sha256'][:16]}...")

    # 3. Check git status
    git_st = get_git_status()
    print(f"Git HEAD: {git_st['head_commit']}")
    print(f"Git Remote: {git_st['origin_main_commit']}")
    print(f"Git Parity: {git_st['parity']}")
    print(f"Git Clean: {git_st['status_clean']}")

    if all_ok:
        print("\nSMOKE TEST RESULT: PASS", flush=True)
    else:
        print("\nSMOKE TEST RESULT: FAIL", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    run_smoke()
