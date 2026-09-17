"""Smoke test for HUY_D1_FROZEN_SECTION_PUBLIC_V1."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .common import (
    CAL_QUESTIONS_LABEL_FREE_PATH,
    D1_CHAMPION_JSON_PATH,
    D1_CHAMPION_ZIP_PATH,
    REPO_JINA,
    WEIGHTS_JINA_FT,
    load_cal_data_label_free,
    load_cal_gold_labels,
    load_jina_crossencoder,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import LegalSection, parse_document_into_sections, preselect_legal_sections
from .materialize_public_candidate import validate_submission_zip


def run_smoke_test():
    print("=== RUNNING SMOKE TEST FOR HUY_D1_FROZEN_SECTION_PUBLIC_V1 ===", flush=True)
    seed_everything(2026)

    # 1. Test legal section parser
    short_text = (
        "BỘ TÀI CHÍNH\nSố: 01/2026/TT-BTC\nTHÔNG TƯ\nQuy định về thuế.\n"
        "Điều 1. Phạm vi điều chỉnh\nThông tư này quy định phạm vi áp dụng."
    )
    s_secs = parse_document_into_sections("doc_short", short_text)
    assert len(s_secs) == 1 and s_secs[0].section_type == "FULL_DOC", f"Expected 1 section for short doc, got {len(s_secs)}"

    long_text = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n\nLUẬT DOANH NGHIỆP\n"
        + "\n".join([
            f"Điều {i}. Quy định về điều khoản {i} với nội dung chi tiết quy định quyền và nghĩa vụ của các bên liên quan trong hoạt động đầu tư kinh doanh thương mại và dân sự trên lãnh thổ nước Cộng hòa xã hội chủ nghĩa Việt Nam nhằm thúc đẩy phát triển kinh tế xã hội và bảo đảm an ninh quốc phòng."
            for i in range(1, 15)
        ])
    )
    secs = parse_document_into_sections("test_doc_long", long_text, max_chunk_words=220, overlap_words=60)
    assert len(secs) >= 2, f"Expected >= 2 sections for long doc, got {len(secs)}"
    selected = preselect_legal_sections("quy định quyền và nghĩa vụ thương mại", secs, count=2)
    assert len(selected) == 2, f"Expected 2 selected sections, got {len(selected)}"
    assert selected[0].section_type == "DIEU"
    print("Legal section parser test: PASS", flush=True)

    # 2. Test label-free loader
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, type_rows, cite_rows = load_cal_data_label_free()
    assert len(all_ids) == 600, f"Expected 600 queries, got {len(all_ids)}"
    assert len(extended) == 600
    assert len(full_channels_cv) == 10
    assert all(queries[q][1] is None for q in all_ids), "Label-free queries must not contain gold labels!"
    print("Label-free loader test: PASS", flush=True)

    # 3. Test gold loader
    gold, reveal_time = load_cal_gold_labels(all_ids[:10])
    assert len(gold) == 10
    print("Gold loader test: PASS", flush=True)

    # 4. Check model and weights existence
    assert REPO_JINA.exists(), f"Repo Jina missing: {REPO_JINA}"
    assert WEIGHTS_JINA_FT.exists(), f"Weights Jina missing: {WEIGHTS_JINA_FT}"
    assert D1_CHAMPION_JSON_PATH.exists(), f"D1 champion JSON missing: {D1_CHAMPION_JSON_PATH}"
    assert D1_CHAMPION_ZIP_PATH.exists(), f"D1 champion ZIP missing: {D1_CHAMPION_ZIP_PATH}"
    print("File existence test: PASS", flush=True)

    # 5. Test ZIP validator
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        json_tmp = tmp / "submission.json"
        zip_tmp = tmp / "submission.zip"
        dummy_data = {f"q_{i}": {"answer": ["d1", "d2", "d3", "d4", "d5"]} for i in range(1000)}
        dummy_bytes = json.dumps(dummy_data, indent=2).encode("utf-8")
        json_tmp.write_bytes(dummy_bytes)
        with zipfile.ZipFile(zip_tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", dummy_bytes)
        valid_docs = {"d1", "d2", "d3", "d4", "d5", "d6"}
        assert validate_submission_zip(zip_tmp, json_tmp, valid_docs)
    print("ZIP validator test: PASS", flush=True)

    # 6. Test frozen cross-encoder forward pass
    print("Loading frozen cross-encoder on GPU for forward pass check...", flush=True)
    model, tok, prov = load_jina_crossencoder()
    score = model.compute_score([("Quy chuẩn kỹ thuật phương tiện", "Điều 1 quy định phạm vi kỹ thuật")], batch_size=1)
    if isinstance(score, list):
        score = score[0]
    print(f"Model compute_score test: PASS (score={score:.4f})", flush=True)

    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    run_smoke_test()
