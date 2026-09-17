"""Smoke tests for HUY_D1_AITEAM_NOVEL_CONSENSUS_V1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_aiteam_novel_consensus_v1.common import (
    AITEAM_REPORT_PATH,
    AITEAM_TOP50_PATH,
    CAL_FROZEN_SECTION_CACHE_PATH,
    EXPECTED_AITEAM_REPORT_SHA256,
    EXPECTED_AITEAM_TOP50_SHA256,
    EXPECTED_JINA_FT_CV_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    JINA_FT_CV_PATH,
    REPO_JINA,
    WEIGHTS_JINA_FT,
    get_git_status,
    get_source_files_sha256,
    sha256_file,
)


def test_smoke():
    print("Running smoke tests for HUY_D1_AITEAM_NOVEL_CONSENSUS_V1...", flush=True)

    # 1. Verify critical file hashes
    assert sha256_file(AITEAM_TOP50_PATH) == EXPECTED_AITEAM_TOP50_SHA256, "AITeam Top-50 SHA mismatch"
    assert sha256_file(AITEAM_REPORT_PATH) == EXPECTED_AITEAM_REPORT_SHA256, "AITeam Report SHA mismatch"
    assert sha256_file(CAL_FROZEN_SECTION_CACHE_PATH) == EXPECTED_SECTION_CE_SHA256, "Section CE cache SHA mismatch"
    assert sha256_file(JINA_FT_CV_PATH) == EXPECTED_JINA_FT_CV_SHA256, "Jina FT CV SHA mismatch"

    assert REPO_JINA.exists(), f"Missing {REPO_JINA}"
    assert WEIGHTS_JINA_FT.exists(), f"Missing {WEIGHTS_JINA_FT}"

    # 2. Test Git status helper
    git_st = get_git_status()
    assert "head_commit" in git_st

    # 3. Source files check
    src_hashes = get_source_files_sha256()
    assert len(src_hashes) >= 6, f"Expected at least 6 source files, got {len(src_hashes)}"

    print("All smoke tests passed successfully!", flush=True)


if __name__ == "__main__":
    test_smoke()
