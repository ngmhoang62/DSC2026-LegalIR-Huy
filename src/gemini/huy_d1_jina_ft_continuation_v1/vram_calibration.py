"""VRAM Auto-calibration for training pair_microbatch."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.gemini.huy_d1_jina_ft_continuation_v1.common as common

OUT_CALIB = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/VRAM_CALIBRATION.json"
OUT_SMOKE = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/VRAM_SMOKE_TEST.json"


def run_calibration() -> dict:
    common.seed_everything(2026)
    print("Building LoRA model for VRAM calibration...", flush=True)
    model, tok = common.build_lora_jina_model(r=16, lora_alpha=32, lora_dropout=0.05)
    model.to("cuda")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)

    # Worst-case representative training group: 2 positives + 7 negatives = 9 pairs
    qtext = "Người có hành vi vi phạm quy định về tham gia giao thông đường bộ và quy định về bồi thường thiệt hại ngoài hợp đồng sẽ bị xử lý như thế nào theo quy định pháp luật hiện hành?"
    p_texts = [
        f"Điều {i+1}. Quy định chi tiết về xử phạt vi phạm hành chính và trách nhiệm bồi thường thiệt hại đối với các hành vi vi phạm pháp luật nghiêm trọng tại các cơ quan nhà nước và tổ chức có liên quan theo quy định của bộ luật hiện hành. " * 30
        for i in range(9)
    ]
    pairs = [(qtext, p) for p in p_texts]

    ladder = [1, 2, 4, 8]
    ladder_results = {}
    chosen_microbatch = None

    for mb in ladder:
        print(f"Testing pair_microbatch={mb}...", flush=True)
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        failed = False
        error_msg = ""
        try:
            logits = []
            for i in range(0, len(pairs), mb):
                chunk = pairs[i : i + mb]
                q_list = [q for q, p in chunk]
                p_list = [p for q, p in chunk]
                batch = tok(
                    q_list,
                    p_list,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                ).to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(**batch)
                    logits.append(out.logits.view(-1))

            all_logits = torch.cat(logits)
            pos_logits = all_logits[:2]
            neg_logits = all_logits[2:]
            diff = -(pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0))
            loss = torch.nn.functional.softplus(diff).mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_res = torch.cuda.max_memory_reserved() / (1024**2)
            curr_alloc = torch.cuda.memory_allocated() / (1024**2)
            curr_res = torch.cuda.memory_reserved() / (1024**2)
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mib = free_mem / (1024**2)
            total_mib = total_mem / (1024**2)

            is_safe = (peak_res <= 5000.0) and (free_mib >= 600.0) and (peak_res <= 5300.0)
            ladder_results[mb] = {
                "pair_microbatch": mb,
                "current_allocated_mib": curr_alloc,
                "current_reserved_mib": curr_res,
                "peak_allocated_mib": peak_alloc,
                "peak_reserved_mib": peak_res,
                "free_mib": free_mib,
                "total_mib": total_mib,
                "step_time_sec": dt,
                "safe": is_safe,
            }
            print(
                f"  mb={mb}: peak_res={peak_res:.1f}MiB, free={free_mib:.1f}MiB, "
                f"time={dt:.3f}s, safe={is_safe}",
                flush=True,
            )
            if is_safe:
                chosen_microbatch = mb
        except Exception as e:
            print(f"  mb={mb} FAILED: {e}", flush=True)
            ladder_results[mb] = {
                "pair_microbatch": mb,
                "safe": False,
                "error": str(e),
            }

    if chosen_microbatch is None:
        status = "LOCAL_VRAM_UNSAFE_KAGGLE_REQUIRED"
    else:
        status = "CALIBRATION_SUCCESS"

    result = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "device_name": torch.cuda.get_device_name(0),
        "total_vram_mib": total_mib,
        "target_safe_region_peak_reserved_limit_mib": 5000.0,
        "hard_stop_peak_reserved_limit_mib": 5300.0,
        "min_free_vram_limit_mib": 600.0,
        "ladder_results": ladder_results,
        "chosen_pair_microbatch": chosen_microbatch,
        "status": status,
    }

    OUT_CALIB.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CALIB, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    smoke = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "test_name": "VRAM_SMOKE_TEST",
        "chosen_pair_microbatch": chosen_microbatch,
        "measured_peak_reserved_mib": ladder_results[chosen_microbatch]["peak_reserved_mib"] if chosen_microbatch else None,
        "measured_free_mib": ladder_results[chosen_microbatch]["free_mib"] if chosen_microbatch else None,
        "safe": chosen_microbatch is not None,
        "status": "PASS" if chosen_microbatch is not None else "FAIL",
    }
    with open(OUT_SMOKE, "w", encoding="utf-8") as f:
        json.dump(smoke, f, indent=2)

    print(f"Wrote {OUT_CALIB} and {OUT_SMOKE}", flush=True)
    return result


if __name__ == "__main__":
    run_calibration()
