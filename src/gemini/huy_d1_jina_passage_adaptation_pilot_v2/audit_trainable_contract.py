"""Audit trainable parameter contract: verify pure LoRA structure and freeze classifier."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

from .common import (
    RES_DIR,
    build_pure_lora_jina_model,
    get_git_status,
)


def run_trainable_parameter_audit() -> Dict[str, Any]:
    print("=== AUDIT: TRAINABLE PARAMETER CONTRACT ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_status()

    # 1. Build model with pure LoRA configuration
    print("[CONTRACT] Instantiating Pure-LoRA Jina model...", flush=True)
    model, tok = build_pure_lora_jina_model(r=8, lora_alpha=16, lora_dropout=0.05)

    # 2. Inspect all parameters
    all_params = list(model.named_parameters())
    total_params_count = sum(p.numel() for _, p in all_params)

    trainable_params = [(n, p) for n, p in all_params if p.requires_grad]
    trainable_params_count = sum(p.numel() for _, p in trainable_params)

    frozen_params = [(n, p) for n, p in all_params if not p.requires_grad]
    frozen_params_count = sum(p.numel() for _, p in frozen_params)

    # 3. Assertions
    illegal_params = []
    for n, p in trainable_params:
        is_lora = ("lora_A" in n or "lora_B" in n)
        is_classifier = ("classifier" in n)
        if not is_lora or is_classifier:
            illegal_params.append(n)

    if illegal_params:
        raise RuntimeError(
            f"BLOCKED_TRAINABLE_PARAMETER_CONTRACT: Found {len(illegal_params)} illegal trainable parameters!\n"
            f"Examples: {illegal_params[:5]}"
        )

    # Classifier verification: must have ZERO trainable parameters
    classifier_params = [(n, p) for n, p in all_params if "classifier" in n]
    classifier_trainable = [n for n, p in classifier_params if p.requires_grad]
    if classifier_trainable:
        raise RuntimeError(
            f"BLOCKED_TRAINABLE_PARAMETER_CONTRACT: Classifier head has trainable parameters!\n"
            f"{classifier_trainable}"
        )

    # Snapshot classifier head state
    classifier_state = model.model.classifier.state_dict()
    h = hashlib.sha256()
    for k in sorted(classifier_state.keys()):
        h.update(f"{k}:".encode("utf-8"))
        h.update(classifier_state[k].cpu().numpy().tobytes())
    classifier_sha256 = h.hexdigest()

    trainable_list = [
        {
            "parameter_name": n,
            "shape": list(p.shape),
            "numel": p.numel(),
            "dtype": str(p.dtype),
        }
        for n, p in trainable_params
    ]

    report = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.trainable_contract.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["mixer.Wqkv", "mixer.out_proj"],
            "task_type": None,
            "modules_to_save": None,
        },
        "model_summary": {
            "total_parameter_count": total_params_count,
            "trainable_parameter_count": trainable_params_count,
            "frozen_parameter_count": frozen_params_count,
            "trainable_tensors_count": len(trainable_params),
            "trainable_percent": (trainable_params_count / total_params_count) * 100,
        },
        "classifier_contract": {
            "classifier_modules_count": len(classifier_params),
            "classifier_trainable_count": len(classifier_trainable),
            "classifier_is_strictly_frozen": len(classifier_trainable) == 0,
            "classifier_initial_sha256": classifier_sha256,
        },
        "trainable_tensors": trainable_list,
    }

    out_path = RES_DIR / "TRAINABLE_PARAMETER_CONTRACT.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[CONTRACT] Saved -> {out_path} "
        f"({len(trainable_params)} trainable tensors, 0 classifier params)",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run_trainable_parameter_audit()
