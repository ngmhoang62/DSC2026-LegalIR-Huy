"""Smoke test verifying legal section parser, preselector, and frozen Jina cross-encoder scoring.

STRICT ANTI-CONTAMINATION:
Uses 100% synthetic Vietnamese legal text generated in-memory.
No references to CAL queries, CAL error qids, CAL gold documents, or CAL labels.
Does not access D1_ERROR_FORENSICS.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    load_jina_crossencoder,
    seed_everything,
)
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
    preselect_legal_sections,
)

# Synthetic Vietnamese legal documents for testing
SYNTHETIC_DOC_STRUCTURED = """CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
---------------
Số: 99/2024/NĐ-CP
Hà Nội, ngày 01 tháng 01 năm 2024

NGHỊ ĐỊNH
Quy định về quản lý môi trường và xử phạt vi phạm hành chính trong lĩnh vực bảo vệ nguồn nước

Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;
Căn cứ Luật Bảo vệ môi trường ngày 17 tháng 11 năm 2020;
Theo đề nghị của Bộ trưởng Bộ Tài nguyên và Môi trường;
Chính phủ ban hành Nghị định quy định về quản lý môi trường.

Chương I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về các biện pháp phòng ngừa, kiểm soát ô nhiễm nguồn nước, trách nhiệm của cơ quan, tổ chức, hộ gia đình và cá nhân trong việc bảo vệ môi trường nước mặt và nước dưới đất trên lãnh thổ nước Cộng hòa xã hội chủ nghĩa Việt Nam.

Điều 2. Giải thích từ ngữ
Trong Nghị định này, các từ ngữ dưới đây được hiểu như sau:
1. Nguồn nước bao gồm nước mặt, nước dưới đất, nước mưa và nước biển thuộc chủ quyền của Việt Nam.
2. Ô nhiễm nguồn nước là sự biến đổi tính chất vật lý, hóa học, sinh học của nước vượt quá quy chuẩn kỹ thuật môi trường cho phép.
3. Cơ sở xả thải là doanh nghiệp, nhà máy, xí nghiệp có hoạt động thải nước vào nguồn nước tiếp nhận.
4. Chất thải nguy hại là chất thải chứa các yếu tố độc hại, phóng xạ, dễ cháy, dễ nổ hoặc có tính ăn mòn.

Chương II
BIỆN PHÁP XỬ PHẠT VÀ MỨC PHẠT HÀNH CHÍNH

Điều 3. Mức phạt tiền đối với hành vi xả nước thải vượt quy chuẩn kỹ thuật
1. Hành vi xả nước thải sinh hoạt vượt quy chuẩn kỹ thuật môi trường dưới 1,5 lần bị phạt cảnh cáo hoặc phạt tiền từ 500.000 đồng đến 1.000.000 đồng.
2. Hành vi xả nước thải công nghiệp chứa chất thải nguy hại vào nguồn nước sinh hoạt bị phạt tiền từ 50.000.000 đồng đến 100.000.000 đồng đối với cá nhân, và từ 100.000.000 đồng đến 200.000.000 đồng đối với tổ chức.
3. Biện pháp khắc phục hậu quả: Buộc đình chỉ hoạt động xả thải từ 03 tháng đến 06 tháng và buộc thực hiện các biện pháp khắc phục tình trạng ô nhiễm môi trường theo quy định của pháp luật.
"""

SYNTHETIC_DOC_UNSTRUCTURED = """CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Thông báo về việc tổ chức tập huấn nghiệp vụ phòng cháy và chữa cháy năm 2024.
Ủy ban nhân dân phường thông báo đến toàn thể các cơ sở sản xuất kinh doanh, hộ gia đình về việc mở lớp tập huấn kỹ năng cứu nạn cứu hộ và sử dụng phương tiện chữa cháy ban đầu.
Thời gian tổ chức vào ngày 15 tháng 3 năm 2024 tại Hội trường trung tâm văn hóa.
Yêu cầu các đơn vị cử đại diện tham gia đầy đủ, đúng giờ để bảo đảm an toàn phòng chống cháy nổ trên địa bàn.
"""

SYNTHETIC_QUERY = "Hành vi xả nước thải công nghiệp chứa chất nguy hại vào nguồn nước bị xử phạt bao nhiêu tiền?"


def run_smoke_test() -> dict:
    seed_everything(2026)
    print("=== SMOKE TEST: SYNTHETIC LEGAL SECTION PARSER & FROZEN JINA ===", flush=True)

    # 1. Test parser on synthetic structured document
    sections_struct = parse_document_into_sections("syn_doc_1", SYNTHETIC_DOC_STRUCTURED)
    assert len(sections_struct) >= 4, f"Expected >= 4 sections, got {len(sections_struct)}"
    headings = [s.heading for s in sections_struct]
    print(f"Synthetic Structured Doc sections: {len(sections_struct)}, headings: {headings}", flush=True)

    # Verify article heading and preamble preservation
    has_dieu3 = any("Điều 3" in s.heading for s in sections_struct)
    assert has_dieu3, "Heading 'Điều 3' was not extracted in structured document!"

    # 2. Test parser on synthetic unstructured document (fallback windowing)
    sections_unstruct = parse_document_into_sections("syn_doc_2", SYNTHETIC_DOC_UNSTRUCTURED)
    assert len(sections_unstruct) >= 1, "Fallback windowing produced 0 sections!"
    print(f"Synthetic Unstructured Doc sections: {len(sections_unstruct)}", flush=True)

    # 3. Test preselector on synthetic query
    selected_secs = preselect_legal_sections(SYNTHETIC_QUERY, sections_struct, count=2)
    assert len(selected_secs) == 2, f"Expected 2 selected sections, got {len(selected_secs)}"
    selected_headings = [s.heading for s in selected_secs]
    print(f"Preselected sections for synthetic query: {selected_headings}", flush=True)

    # The relevant article for penalty is Điều 3
    assert any("Điều 3" in s.heading for s in selected_secs), "Preselector failed to pick relevant penalty Article 3!"

    # 4. Test neural inference with frozen Jina
    print("Loading frozen Jina cross-encoder on GPU...", flush=True)
    t0 = time.perf_counter()
    model, tok, prov = load_jina_crossencoder()
    t_load = time.perf_counter() - t0
    print(f"Model ready in {t_load:.2f}s on {prov['device']}. is_frozen={prov['is_frozen']}", flush=True)

    relevant_text = [s.text for s in selected_secs if "Điều 3" in s.heading][0]
    irrelevant_text = sections_unstruct[0].text

    pairs = [
        (SYNTHETIC_QUERY, relevant_text),
        (SYNTHETIC_QUERY, irrelevant_text),
    ]

    torch.cuda.reset_peak_memory_stats()
    raw_scores = model.compute_score(pairs, batch_size=2, max_length=512)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    score_rel = float(raw_scores[0])
    score_irrel = float(raw_scores[1])
    diff = score_rel - score_irrel

    print(f"Relevant synthetic section score:   {score_rel:.4f}", flush=True)
    print(f"Irrelevant synthetic section score: {score_irrel:.4f}", flush=True)
    print(f"Score difference:                   {diff:.4f}", flush=True)
    print(f"Peak VRAM:                          {peak_vram_mb:.2f} MB", flush=True)

    assert score_rel != score_irrel, "Scores must not be identically equal!"
    assert diff > 0, f"Expected relevant section score > irrelevant section score, got {diff:.4f}"

    smoke_results = {
        "status": "PASS",
        "contamination_free": True,
        "model_provenance": prov,
        "score_relevant": score_rel,
        "score_irrelevant": score_irrel,
        "score_difference": diff,
        "peak_vram_mb": peak_vram_mb,
    }
    print("=== SYNTHETIC SMOKE TEST PASSED (NO CAL CONTAMINATION) ===", flush=True)
    return smoke_results


if __name__ == "__main__":
    res = run_smoke_test()
    print(json.dumps(res, indent=2))
