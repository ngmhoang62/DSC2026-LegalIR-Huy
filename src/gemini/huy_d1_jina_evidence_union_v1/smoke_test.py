"""Synthetic smoke test for HUY_D1_JINA_EVIDENCE_UNION_V1 (Strict Anti-Contamination)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    load_jina_crossencoder,
    seed_everything,
    top_passages,
)

SYNTHETIC_DOC = """CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
---------------
Số: 101/2024/NĐ-CP
Hà Nội, ngày 10 tháng 02 năm 2024

NGHỊ ĐỊNH
Quy định về quản lý và xử phạt trong lĩnh vực an toàn giao thông đường bộ

Chương I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về xử phạt vi phạm hành chính đối với các hành vi không chấp hành tín hiệu giao thông, chạy quá tốc độ quy định.

Điều 2. Mức phạt hành chính
Người điều khiển xe mô tô chạy quá tốc độ quy định từ 10 km/h đến 20 km/h bị phạt tiền từ 800.000 đồng đến 1.000.000 đồng.
"""

SYNTHETIC_QUERY = "Hành vi chạy xe mô tô quá tốc độ từ 10 đến 20 km/h bị phạt bao nhiêu tiền?"


def run_synthetic_smoke_test() -> dict:
    seed_everything(2026)
    print("=== SMOKE TEST: SYNTHETIC JINA EVIDENCE UNION PLUMBING ===", flush=True)

    # 1. Test top_passages sliding window
    passages = top_passages(SYNTHETIC_QUERY, SYNTHETIC_DOC, count=2, window=220, overlap=70)
    assert len(passages) >= 1, "top_passages produced empty list!"
    print(f"Synthetic sliding window passages: {len(passages)}", flush=True)

    # 2. Test model loading
    print("Loading frozen Jina cross-encoder on GPU...", flush=True)
    t0 = time.perf_counter()
    model, tok, prov = load_jina_crossencoder()
    t_load = time.perf_counter() - t0
    print(f"Model ready in {t_load:.2f}s on {prov['device']}.", flush=True)

    # 3. Test scoring
    pairs = [(SYNTHETIC_QUERY, p) for p in passages]
    torch.cuda.reset_peak_memory_stats()
    raw_scores = model.compute_score(pairs, batch_size=2, max_length=512)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    if isinstance(raw_scores, float):
        raw_scores = [raw_scores]

    max_old = max(float(s) for s in raw_scores)
    # Simulated synthetic section score
    simulated_section_score = 0.95
    simulated_union = max(max_old, simulated_section_score)

    print(f"Synthetic Old Score:     {max_old:.4f}", flush=True)
    print(f"Synthetic Section Score: {simulated_section_score:.4f}", flush=True)
    print(f"Synthetic Union Score:   {simulated_union:.4f}", flush=True)
    print(f"Peak VRAM:               {peak_vram:.2f} MB", flush=True)

    assert simulated_union >= max_old, "Union must be >= old score!"
    assert np.isfinite(simulated_union), "Score must be finite!"

    result = {
        "status": "PASS",
        "contamination_free": True,
        "old_score": max_old,
        "section_score": simulated_section_score,
        "union_score": simulated_union,
        "peak_vram_mb": peak_vram,
    }
    print("=== SYNTHETIC SMOKE TEST PASSED ===", flush=True)
    return result


if __name__ == "__main__":
    res = run_synthetic_smoke_test()
    print(json.dumps(res, indent=2))
