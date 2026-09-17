"""Smoke test for HUY_D1_SELECTIVE_REPAIR_V1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_selective_repair_v1.common import (
    CAL_FROZEN_SECTION_CACHE_PATH,
    EXPECTED_OLD_JINA_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    OLD_JINA_CACHE_PATH,
    canonicalize_ref,
    extract_query_references,
    get_git_status,
    get_source_files_sha256,
    sha256_file,
)
from src.gemini.huy_d1_selective_repair_v1.tier1_citation import (
    apply_tier1_repair,
    resolve_tier1_anchor,
)


def test_smoke():
    print("Running smoke tests...", flush=True)

    # 1. Test canonicalize_ref
    assert canonicalize_ref("31/2022/TT-BTC") == "31/2022/TT-BTC"
    assert canonicalize_ref("49/2021/ND-CP") == "49/2021/NĐ-CP"
    assert canonicalize_ref("68/QD-TTG") == "68/QĐ-TTg"
    assert canonicalize_ref("08/2023/TT-BGDDT") == "08/2023/TT-BGDĐT"

    # 2. Test query reference extraction
    q_sample = "Theo Nghị định 49/2021/NĐ-CP và Quyết định 68/QĐ-TTg thì quy định ra sao?"
    refs = extract_query_references(q_sample)
    assert "49/2021/NĐ-CP" in refs
    assert "68/QĐ-TTg" in refs

    # 3. Test resolve_tier1_anchor logic
    mock_ref_to_docs = {"49/2021/NĐ-CP": ["101"]}
    anchor, status, extracted = resolve_tier1_anchor("Theo 49/2021/NĐ-CP", mock_ref_to_docs)
    assert anchor == "101"
    assert status == "UNIQUE_EXACT_MATCH"

    # Test protected injection
    base_top5 = ["1", "2", "3", "4", "5"]
    new_top5, act, inj, evict = apply_tier1_repair(base_top5, "101")
    assert act == "EXACT_CITATION_INJECTION"
    assert new_top5 == ["1", "2", "3", "4", "101"]
    assert inj == "101"
    assert evict == "5"

    # 4. Test cache files existence and hashes
    assert sha256_file(CAL_FROZEN_SECTION_CACHE_PATH) == EXPECTED_SECTION_CE_SHA256
    assert sha256_file(OLD_JINA_CACHE_PATH) == EXPECTED_OLD_JINA_SHA256

    # 5. Test git status utility
    git_st = get_git_status()
    assert "head_commit" in git_st

    src_hashes = get_source_files_sha256()
    assert len(src_hashes) >= 5

    print("All smoke tests passed successfully!", flush=True)


if __name__ == "__main__":
    test_smoke()
