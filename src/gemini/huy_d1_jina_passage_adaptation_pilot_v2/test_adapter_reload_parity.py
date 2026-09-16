"""Test adapter save/reload parity on a deterministic sample of pairs."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from peft import PeftModel

from .common import (
    ADAPTER_DIR,
    RES_DIR,
    ROOT,
    get_git_status,
    load_jina_base_with_shipped_weights,
    patch_tuple_returning_lora,
    seed_everything,
)

TEACHER_PAIRS_FILE = RES_DIR / "TEACHER_TRAINING_PAIRS.jsonl"


def run_reload_parity_test(sample_size: int = 64) -> Dict[str, Any]:
    print("=== STAGE: ADAPTER SAVE / RELOAD PARITY TEST ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_DIR}")

    # 1. Load sample pairs
    if not TEACHER_PAIRS_FILE.exists():
        raise FileNotFoundError(f"Training pairs not found: {TEACHER_PAIRS_FILE}")

    sample_pairs: List[Tuple[str, str]] = []
    with open(TEACHER_PAIRS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if len(sample_pairs) >= sample_size:
                break
            if line.strip():
                rec = json.loads(line)
                q_text = rec["query_text"]
                pos_sec = (rec["pos_sections"] or [q_text])[0]
                sample_pairs.append((q_text, pos_sec))

    print(f"[RELOAD_TEST] Selected {len(sample_pairs)} deterministic sample pairs.", flush=True)

    # 2. Score with reloaded model from disk
    print("[RELOAD_TEST] Loading fresh base model and attaching saved adapter...", flush=True)
    fresh_base, tok = load_jina_base_with_shipped_weights()
    reloaded_model = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR)
    patch_tuple_returning_lora(reloaded_model)
    reloaded_model.eval().to("cuda")

    # Score sample pairs with reloaded model
    reloaded_scores: List[float] = []
    for i in range(0, len(sample_pairs), 32):
        batch = sample_pairs[i : i + 32]
        inputs = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            s = reloaded_model(**inputs, return_dict=True).logits.view(-1).float()
        reloaded_scores.extend(s.cpu().numpy().tolist())

    # 3. Reload a SECOND fresh instance to assert deterministic bit-exact reload parity
    del fresh_base, reloaded_model
    torch.cuda.empty_cache()
    gc.collect()

    second_base, tok2 = load_jina_base_with_shipped_weights()
    second_model = PeftModel.from_pretrained(second_base, ADAPTER_DIR)
    patch_tuple_returning_lora(second_model)
    second_model.eval().to("cuda")

    second_scores: List[float] = []
    for i in range(0, len(sample_pairs), 32):
        batch = sample_pairs[i : i + 32]
        inputs = tok2(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            s = second_model(**inputs, return_dict=True).logits.view(-1).float()
        second_scores.extend(s.cpu().numpy().tolist())

    max_abs_diff = max(abs(a - b) for a, b in zip(reloaded_scores, second_scores))
    print(f"[RELOAD_TEST] Max absolute score difference across fresh reloads: {max_abs_diff:.8e}", flush=True)

    if max_abs_diff > 1e-6:
        raise RuntimeError(
            f"BLOCKED_ADAPTER_RELOAD_PARITY: Max score difference {max_abs_diff} exceeds 1e-6!"
        )

    del second_base, second_model
    torch.cuda.empty_cache()
    gc.collect()

    report = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.adapter_reload_parity.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "adapter_path": str(ADAPTER_DIR.relative_to(ROOT)),
        "sample_size_pairs": len(sample_pairs),
        "max_absolute_score_difference": max_abs_diff,
        "parity_threshold": 1e-6,
        "parity_passed": max_abs_diff <= 1e-6,
    }

    out_path = RES_DIR / "ADAPTER_RELOAD_PARITY.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RELOAD_TEST] Reload parity passed and saved -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    run_reload_parity_test()
