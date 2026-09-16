"""100% synthetic unit test for LegalSectionIndex and parent candidate retriever.

Verifies:
1. Section parsing on synthetic legal documents
2. FTS5 index construction
3. Section retrieval and MAX parent aggregation
4. Metadata recording (best_heading, best_section_type, etc.)
5. Exclusion of documents already in baseline pool
6. Cap enforcement (budget limit)
7. Deterministic tie-breaking by numeric doc_id
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .section_retriever import LegalSectionIndex


def run_smoke_test() -> bool:
    print("[SMOKE] Running synthetic unit test...", flush=True)

    # 1. Create synthetic corpus with multi-section structure
    doc_101_text = (
        "LUẬT BẢO HIỂM TIỀN GỬI VIỆT NAM\n"
        "Số 06/2012/QH13 ngày 18 tháng 06 năm 2012 của Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam.\n"
        "Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam năm 1992 đã được sửa đổi bổ sung;\n"
        "Quốc hội ban hành Luật bảo hiểm tiền gửi.\n\n"
        "Chương I\nQUY ĐỊNH CHUNG\n\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "Luật này quy định về hoạt động bảo hiểm tiền gửi; quyền và nghĩa vụ của tổ chức tham gia bảo hiểm tiền gửi.\n\n"
        "Điều 2. Đối tượng áp dụng\n"
        "Luật này áp dụng đối với tổ chức tham gia bảo hiểm tiền gửi bao gồm ngân hàng thương mại, ngân hàng hợp tác xã.\n\n"
        "Điều 3. Quy định đặc thù về kiểm toán bảo hiểm nông nghiệp\n"
        "Tổ chức tín dụng tham gia nghiệp vụ bảo hiểm nông nghiệp phải thực hiện kiểm toán độc lập định kỳ hàng năm "
        "và công khai báo cáo tài chính theo chuẩn mực kế toán kiểm toán Việt Nam. Việc kiểm toán này nhằm bảo đảm an toàn nguồn vốn."
    )

    doc_102_text = (
        "NGHỊ ĐỊNH VỀ THUẾ GIÁ TRỊ GIA TĂNG\n"
        "Số 209/2013/NĐ-CP của Chính phủ quy định chi tiết thi hành Luật Thuế giá trị gia tăng.\n\n"
        "Điều 1. Thuế giá trị gia tăng\n"
        "Mức thuế suất 10% áp dụng cho hàng hóa dịch vụ thông thường và dịch vụ tài chính.\n\n"
        "Điều 2. Khấu trừ thuế nông nghiệp\n"
        "Doanh nghiệp nông nghiệp được khấu trừ thuế đầu vào theo quy định mới đối với máy móc thiết bị."
    )

    doc_103_text = (
        "QUY CHUẨN KỸ THUẬT QUỐC GIA\n\n"
        "Phụ lục I. Tiêu chuẩn đo lường an toàn thực phẩm\n"
        "Quy chuẩn kiểm nghiệm dư lượng hóa chất trong thực phẩm tươi sống."
    )

    doc_104_text = (
        "Văn bản quy định chung về tài chính kế toán không có tiêu đề điều khoản rõ ràng "
        "nhưng chứa các nội dung hướng dẫn nghiệp vụ thanh toán điện tử liên ngân hàng."
    )

    mock_corpus = {
        "101": {"id": "101", "link": "", "passage": doc_101_text},
        "102": {"id": "102", "link": "", "passage": doc_102_text},
        "103": {"id": "103", "link": "", "passage": doc_103_text},
        "104": {"id": "104", "link": "", "passage": doc_104_text},
    }

    # Add dummy documents 201..215 to test cap enforcement
    for i in range(1, 16):
        mock_corpus[f"{200 + i}"] = {
            "id": f"{200 + i}",
            "link": "",
            "passage": f"Điều 1. Quy tắc mẫu số {i} liên quan đến thanh toán điện tử tiền gửi.",
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "test_sections.db"
        index = None

        try:
            # 2. Build index with small chunk words to test section granularity
            index = LegalSectionIndex.build_index(
                mock_corpus, tmp_db, max_chunk_words=50, overlap_words=10, verbose=False
            )

            stats = index.get_stats()
            assert stats["documents_indexed"] == len(mock_corpus), f"Expected {len(mock_corpus)} docs, got {stats['documents_indexed']}"
            assert stats["sections_indexed"] > len(mock_corpus), f"Expected > {len(mock_corpus)} sections, got {stats['sections_indexed']}"
            print(f"[SMOKE] Index built: {stats['documents_indexed']} docs, {stats['sections_indexed']} sections", flush=True)

            # 3. Test query matching specific section
            # "kiểm toán bảo hiểm nông nghiệp" should match doc 101 Điều 3 strongly
            q1 = "kiểm toán bảo hiểm nông nghiệp"
            res1 = index.retrieve_parent_candidates(q1, existing_pool=set(), section_hit_depth=128, cap=8)
            assert len(res1) > 0, "Expected at least 1 candidate for q1"
            top_cand = res1[0]
            assert top_cand["doc_id"] == "101", f"Expected top doc 101, got {top_cand['doc_id']}"
            assert "Điều 3" in top_cand["best_heading"], f"Expected Điều 3 heading, got {top_cand['best_heading']}"
            assert top_cand["best_section_type"] == "DIEU", f"Expected DIEU, got {top_cand['best_section_type']}"
            print(f"[SMOKE] Test 1 passed: doc {top_cand['doc_id']} matched via {top_cand['best_heading']}", flush=True)

            # 4. Test baseline pool exclusion
            # If 101 is already in pool, it must NOT appear in additions
            res1_excluded = index.retrieve_parent_candidates(q1, existing_pool={"101"}, section_hit_depth=128, cap=8)
            assert all(c["doc_id"] != "101" for c in res1_excluded), "Doc 101 was not excluded from pool"
            print("[SMOKE] Test 2 passed: existing pool exclusion verified", flush=True)

            # 5. Test cap enforcement (budget limit = 8)
            q2 = "thanh toán điện tử"
            res2 = index.retrieve_parent_candidates(q2, existing_pool=set(), section_hit_depth=128, cap=8)
            assert len(res2) <= 8, f"Expected at most 8 additions, got {len(res2)}"
            assert len(res2) == 8, f"Expected exactly 8 additions due to many matches, got {len(res2)}"
            print(f"[SMOKE] Test 3 passed: cap=8 enforced (returned {len(res2)} additions)", flush=True)

            # 6. Test deterministic tie-breaking (equal dummy text -> sorted by numeric doc_id)
            for k in range(len(res2) - 1):
                s1 = res2[k]["score"]
                s2 = res2[k + 1]["score"]
                assert s1 >= s2, f"Scores not non-increasing: {s1} < {s2}"
                if abs(s1 - s2) < 1e-9 and res2[k]["doc_id"].isdigit() and res2[k + 1]["doc_id"].isdigit():
                    assert int(res2[k]["doc_id"]) < int(res2[k + 1]["doc_id"]), "Tie breaking not numeric ascending"
            print("[SMOKE] Test 4 passed: deterministic tie-breaking verified", flush=True)
        finally:
            if index is not None:
                index.close()

    print("[SMOKE] ALL SYNTHETIC UNIT TESTS PASSED SUCCESSFULLY!", flush=True)
    return True


if __name__ == "__main__":
    ok = run_smoke_test()
    if not ok:
        sys.exit(1)
