"""Smoke tests for HUY_D1_EXACT_CITATION_PUBLIC_V1."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_exact_citation_public_v1.common import (
    D1_CHAMPION_JSON_PATH,
    D1_CHAMPION_ZIP_PATH,
    PUBLIC_DATASET_DIR,
    PUBLIC_CONTEXTS_DIR,
    apply_tier1_repair,
    canonicalize_ref,
    extract_doc_own_reference,
    extract_query_references,
    get_git_status,
    get_source_files_sha256,
    resolve_tier1_anchor,
    sha256_file,
)
from src.gemini.huy_d1_exact_citation_public_v1.materialize_candidate import (
    validate_submission_zip,
)


def test_smoke():
    print("Running smoke tests for HUY_D1_EXACT_CITATION_PUBLIC_V1...", flush=True)

    # 1. Test canonicalize_ref
    assert canonicalize_ref("31/2022/TT-BTC") == "31/2022/TT-BTC"
    assert canonicalize_ref("17/2022/UBTVQH15") == "17/2022/UBTVQH15"
    assert canonicalize_ref("16/2022/TT-BGTVT") == "16/2022/TT-BGTVT"
    assert canonicalize_ref("15-CT/TW") == "15-CT/TW"

    # 2. Test query reference extraction
    q_sample = "Theo Thông tư 16/2022/TT-BGTVT và Nghị quyết 17/2022/UBTVQH15 thì sao?"
    refs = extract_query_references(q_sample)
    assert "16/2022/TT-BGTVT" in refs
    assert "17/2022/UBTVQH15" in refs

    # 3. Test resolve_tier1_anchor logic
    mock_ref_to_docs = {"16/2022/TT-BGTVT": ["19285"]}
    anchor, status, extracted = resolve_tier1_anchor(q_sample, mock_ref_to_docs)
    assert anchor == "19285"
    assert status == "UNIQUE_EXACT_MATCH"

    # 4. Test protected injection
    base_top5 = ["1", "2", "3", "4", "5"]
    new_top5, act, inj, evict = apply_tier1_repair(base_top5, "19285")
    assert act == "EXACT_CITATION_INJECTION"
    assert new_top5 == ["1", "2", "3", "4", "19285"]
    assert inj == "19285"
    assert evict == "5"

    # 5. Check champion files
    assert D1_CHAMPION_JSON_PATH.exists(), f"Missing {D1_CHAMPION_JSON_PATH}"
    assert D1_CHAMPION_ZIP_PATH.exists(), f"Missing {D1_CHAMPION_ZIP_PATH}"

    # 6. Test validate_submission_zip with dummy files
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        dummy_json_p = td / "candidate.json"
        dummy_zip_p = td / "candidate.zip"
        dummy_payload = {str(i): {"answer": [f"doc_{j}" for j in range(5)]} for i in range(1000)}
        dummy_json_p.write_text(json.dumps(dummy_payload), encoding="utf-8")
        with zipfile.ZipFile(dummy_zip_p, "w") as zf:
            zf.writestr("submission.json", json.dumps(dummy_payload))

        valid_docs = set(f"doc_{j}" for j in range(5))
        validate_submission_zip(dummy_zip_p, dummy_json_p, valid_docs)

    # 7. Check git status
    git_st = get_git_status()
    assert "head_commit" in git_st

    src_hashes = get_source_files_sha256()
    assert len(src_hashes) >= 4

    print("All smoke tests passed successfully!", flush=True)


if __name__ == "__main__":
    test_smoke()
