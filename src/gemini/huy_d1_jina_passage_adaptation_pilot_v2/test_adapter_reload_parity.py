"""Test true adapter save/reload parity: compare in-memory pre-save scores against fresh-reload scores."""

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

PRE_SAVE_REFERENCE_FILE = RES_DIR / "PRE_SAVE_ADAPTED_SCORE_REFERENCE.json"


def run_reload_parity_test() -> Dict[str, Any]:
    print("=== STAGE: TRUE ADAPTER SAVE / RELOAD PARITY TEST ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_DIR}")
    if not PRE_SAVE_REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Pre-save reference file not found: {PRE_SAVE_REFERENCE_FILE}")

    # 1. Load pre-save score reference
    ref_data = json.loads(PRE_SAVE_REFERENCE_FILE.read_text(encoding="utf-8"))
    records = ref_data["records"]
    print(f"[RELOAD_TEST] Loaded {len(records)} pre-save reference score records.", flush=True)

    # 2. Load fresh base Jina model and attach saved LoRA adapter from disk
    print("[RELOAD_TEST] Loading fresh base model and attaching saved adapter from disk...", flush=True)
    fresh_base, tok = load_jina_base_with_shipped_weights()
    reloaded_model = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR)
    patch_tuple_returning_lora(reloaded_model)
    reloaded_model.eval().to("cuda")

    # 3. Score exact same sample with exact same sections
    reloaded_scores: List[float] = []
    for r in records:
        q_text = r["query_text"]
        secs = r["sections"]
        inputs = tok(
            [(q_text, s) for s in secs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to("cuda")
        with torch.no_grad():
            logits = reloaded_model(**inputs, return_dict=True).logits.view(-1).float()
            doc_score = float(torch.max(logits).cpu().item())
        reloaded_scores.extend([doc_score])

    pre_save_scores = [r["pre_save_adapted_score"] for r in records]
    score_diffs = [abs(a - b) for a, b in zip(pre_save_scores, reloaded_scores)]
    max_abs_diff = max(score_diffs)
    mean_abs_diff = float(np.mean(score_diffs))

    print(
        f"[RELOAD_TEST] Pre-save In-Memory vs Fresh-Reload Parity: "
        f"max_diff = {max_abs_diff:.8e}, mean_diff = {mean_abs_diff:.8e}",
        flush=True,
    )

    if max_abs_diff > 1e-6:
        raise RuntimeError(
            f"BLOCKED_ADAPTER_RELOAD_PARITY: Max pre-save vs reload difference {max_abs_diff} exceeds 1e-6!"
        )

    # 4. Optional auxiliary test: second reload determinism
    del fresh_base, reloaded_model
    torch.cuda.empty_cache()
    gc.collect()

    second_base, tok2 = load_jina_base_with_shipped_weights()
    second_model = PeftModel.from_pretrained(second_base, ADAPTER_DIR)
    patch_tuple_returning_lora(second_model)
    second_model.eval().to("cuda")

    second_scores: List[float] = []
    for r in records:
        q_text = r["query_text"]
        secs = r["sections"]
        inputs = tok2(
            [(q_text, s) for s in secs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to("cuda")
        with torch.no_grad():
            logits = second_model(**inputs, return_dict=True).logits.view(-1).float()
            second_scores.append(float(torch.max(logits).cpu().item()))

    second_reload_max_diff = max(abs(a - b) for a, b in zip(reloaded_scores, second_scores))
    print(f"[RELOAD_TEST] Auxiliary second-reload determinism max diff: {second_reload_max_diff:.8e}", flush=True)

    del second_base, second_model
    torch.cuda.empty_cache()
    gc.collect()

    report = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.adapter_reload_parity.v2",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "adapter_path": str(ADAPTER_DIR.relative_to(ROOT)),
        "sample_size_pairs": len(records),
        "pre_save_reference_file": str(PRE_SAVE_REFERENCE_FILE.relative_to(ROOT)),
        "max_absolute_pre_save_vs_reload_difference": max_abs_diff,
        "mean_absolute_pre_save_vs_reload_difference": mean_abs_diff,
        "second_reload_determinism_max_diff": second_reload_max_diff,
        "parity_threshold": 1e-6,
        "parity_passed": max_abs_diff <= 1e-6,
    }

    out_path = RES_DIR / "ADAPTER_RELOAD_PARITY.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RELOAD_TEST] True reload parity passed and saved -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    run_reload_parity_test()
