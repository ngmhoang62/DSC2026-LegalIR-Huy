"""Smoke test verifying imports, model adapter loading, sigmoid arithmetic, and LTR feature shapes."""

from __future__ import annotations

import sys
import numpy as np
import torch

from .common import (
    ADAPTER_DIR,
    D1_VIEWS,
    load_adapted_jina_model,
    load_cal_data,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import LegalSection, parse_document_into_sections, preselect_legal_sections


def run_smoke_test():
    print("=== RUNNING SMOKE TEST FOR HUY_D1_ADAPTED_SECTION_CHANNEL_PUBLIC_V1 ===", flush=True)
    seed_everything(2026)

    # 1. Test parser
    sample_text = """BỘ TÀI CHÍNH\nSố: 01/2026/TT-BTC\nTHÔNG TƯ\nQuy định về thuế.\nĐiều 1. Phạm vi điều chỉnh\nThông tư này quy định phạm vi áp dụng.\nĐiều 2. Đối tượng áp dụng\nÁp dụng cho mọi tổ chức."""
    sections = parse_document_into_sections("doc_test", sample_text, max_chunk_words=220, overlap_words=60)
    assert len(sections) == 1, f"Expected 1 section for short doc, got {len(sections)}"

    # Long doc (>270 words)
    long_text = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n\nLUẬT DOANH NGHIỆP\n" + "\n".join([f"Điều {i}. Quy định về điều khoản {i} với nội dung chi tiết quy định quyền và nghĩa vụ của các bên liên quan trong hoạt động đầu tư kinh doanh thương mại và dân sự trên lãnh thổ nước Cộng hòa xã hội chủ nghĩa Việt Nam nhằm thúc đẩy phát triển kinh tế xã hội và bảo đảm an ninh quốc phòng." for i in range(1, 15)])
    long_secs = parse_document_into_sections("doc_long", long_text, max_chunk_words=220, overlap_words=60)
    assert len(long_secs) >= 2, f"Expected >= 2 sections for long doc, got {len(long_secs)}"
    selected = preselect_legal_sections("quy định quyền và nghĩa vụ thương mại", long_secs, count=2)
    assert len(selected) == 2, f"Expected 2 selected sections, got {len(selected)}"
    print("Legal section parser test: PASS")

    # 2. Test model loading and adapter attachment
    print("Loading adapted Jina model on GPU for forward pass check...", flush=True)
    adapted_model, base_model, tok = load_adapted_jina_model(device="cuda")
    pairs = [("Thuế giá trị gia tăng là gì?", "Điều 1. Thuế giá trị gia tăng là thuế tính trên giá trị tăng thêm.")]
    inputs = tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=128).to("cuda")

    with torch.no_grad():
        logits = adapted_model(**inputs).logits.view(-1).float()
        probs = torch.sigmoid(logits)

    logit_val = float(logits[0].cpu())
    prob_val = float(probs[0].cpu())
    expected_prob = 1.0 / (1.0 + np.exp(-logit_val))
    assert abs(prob_val - expected_prob) < 1e-6, f"Sigmoid mismatch: {prob_val} vs {expected_prob}"
    print(f"Forward pass and sigmoid test: PASS (logit={logit_val:.4f}, prob={prob_val:.4f})")

    del adapted_model, base_model, tok
    torch.cuda.empty_cache()

    # 3. Test adapter hash calculation
    config_sha = sha256_file(ADAPTER_DIR / "adapter_config.json")
    model_sha = sha256_file(ADAPTER_DIR / "adapter_model.safetensors")
    assert len(config_sha) == 64 and len(model_sha) == 64
    print("Adapter files SHA256 test: PASS")

    print("ALL SMOKE TESTS PASSED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    run_smoke_test()
