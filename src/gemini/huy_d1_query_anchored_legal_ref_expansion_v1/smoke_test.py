"""Synthetic Smoke Test for HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1.

Uses 100% synthetic dummy data to verify the reference indexer, relation parser,
and query-anchored generator without touching CAL or real legal texts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
    build_legal_reference_index,
    canonicalize_ref,
    extract_doc_own_reference,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.query_anchored_generator import (
    extract_query_references,
    generate_query_additions,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.relation_graph import (
    build_explicit_relation_graph,
)


def run_synthetic_smoke_test() -> dict:
    print("=== RUNNING SYNTHETIC SMOKE TEST ===", flush=True)

    # 1. Synthetic documents
    synthetic_corpus = {
        "101": {
            "id": "101",
            "link": "https://example.vn/van-ban/Thong-tu-10-2023-TT-TEST-sua-doi-Nghi-dinh-20-2021-ND-TEST.aspx",
            "passage": "BỘ TEST\nSố:\n10/2023/TT-TEST\nTHÔNG TƯ\nSửa đổi, bổ sung một số điều của Nghị định số 20/2021/NĐ-TEST...",
        },
        "202": {
            "id": "202",
            "link": "https://example.vn/van-ban/Nghi-dinh-20-2021-ND-TEST.aspx",
            "passage": "CHÍNH PHỦ\nSố: 20/2021/NĐ-TEST\nNGHỊ ĐỊNH\nVề tổ chức kiểm thử mô hình...",
        },
        "303": {
            "id": "303",
            "link": "https://example.vn/van-ban/Quyet-dinh-30-2020-QD-TTG.aspx",
            "passage": "THỦ TƯỚNG\nSố: 30/2020/QĐ-TTg\nQUYẾT ĐỊNH\nHướng dẫn thi hành Nghị định số 20/2021/NĐ-TEST...",
        },
    }

    # Test own-reference extraction
    ref101 = extract_doc_own_reference(synthetic_corpus["101"]["passage"], synthetic_corpus["101"]["link"])
    assert ref101 == "10/2023/TT-TEST", f"Expected 10/2023/TT-TEST, got {ref101}"

    ref202 = extract_doc_own_reference(synthetic_corpus["202"]["passage"], synthetic_corpus["202"]["link"])
    assert ref202 == "20/2021/NĐ-TEST", f"Expected 20/2021/NĐ-TEST, got {ref202}"

    # Build reference index
    ref_to_docs, doc_to_own_ref, _ = build_legal_reference_index(synthetic_corpus)
    assert "10/2023/TT-TEST" in ref_to_docs
    assert "20/2021/NĐ-TEST" in ref_to_docs

    # Build relation graph
    out_edges, in_edges, _ = build_explicit_relation_graph(synthetic_corpus, ref_to_docs, doc_to_own_ref)
    # Doc 101 should amend Doc 202
    assert any(target == "202" for target, rel, is_h, _ in out_edges.get("101", [])), "Doc 101 should amend Doc 202"
    # Doc 303 should guide Doc 202
    assert any(target == "202" for target, rel, is_h, _ in out_edges.get("303", [])), "Doc 303 should guide Doc 202"

    # Test query generation
    query_text = "Quy định theo Thông tư 10/2023/TT-TEST là gì?"
    refs = extract_query_references(query_text)
    assert refs == ["10/2023/TT-TEST"], f"Expected ['10/2023/TT-TEST'], got {refs}"

    existing_pool = set()
    adds, details, q_refs = generate_query_additions(
        query_text=query_text,
        existing_pool=existing_pool,
        ref_to_docs=ref_to_docs,
        out_edges=out_edges,
        in_edges=in_edges,
        cap=8,
    )

    assert "101" in adds, "Exact doc 101 should be added"
    assert "202" in adds, "Relation neighbor doc 202 should be added"
    assert len(adds) <= 8, "Cap should be respected"

    print("Synthetic smoke test completed successfully. All components operational.\n", flush=True)
    return {
        "status": "PASS",
        "sample_additions": adds,
        "sample_details": details,
    }


if __name__ == "__main__":
    run_synthetic_smoke_test()
