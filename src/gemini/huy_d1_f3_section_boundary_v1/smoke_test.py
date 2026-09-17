"""Smoke tests for HUY_D1_F3_SECTION_BOUNDARY_V1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_f3_section_boundary_v1.common import (
    CAL600_5FOLD_PATH,
    CAL_FROZEN_SECTION_CACHE_PATH,
    EXPECTED_5FOLD_SPLIT_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    F3_FEASIBILITY_PATH,
    F3_MODEL_PATH,
    F3_SCORES_PATH,
    F3_SOURCE_PATH,
    get_git_status,
    get_source_files_sha256,
    sha256_file,
)
from src.gemini.huy_d1_f3_section_boundary_v1.confirmation_gate import (
    load_confirmation_split,
)


def test_smoke():
    print("Running smoke tests for HUY_D1_F3_SECTION_BOUNDARY_V1...", flush=True)

    # 1. Verify frozen expert hashes
    assert F3_MODEL_PATH.exists(), f"Missing {F3_MODEL_PATH}"
    assert F3_SCORES_PATH.exists(), f"Missing {F3_SCORES_PATH}"
    assert F3_SOURCE_PATH.exists(), f"Missing {F3_SOURCE_PATH}"
    assert F3_FEASIBILITY_PATH.exists(), f"Missing {F3_FEASIBILITY_PATH}"

    assert sha256_file(CAL_FROZEN_SECTION_CACHE_PATH) == EXPECTED_SECTION_CE_SHA256
    assert sha256_file(CAL600_5FOLD_PATH) == EXPECTED_5FOLD_SPLIT_SHA256

    # 2. Test split loader
    folds, conf_ids, _ = load_confirmation_split()
    assert len(conf_ids) == 240
    assert "fold_3" in folds and "fold_4" in folds

    # 3. Test git status
    git_st = get_git_status()
    assert "head_commit" in git_st

    src_hashes = get_source_files_sha256()
    assert len(src_hashes) >= 4

    print("All smoke tests passed successfully!", flush=True)


if __name__ == "__main__":
    test_smoke()
