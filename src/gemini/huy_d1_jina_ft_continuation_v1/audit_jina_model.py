"""Audit shipped Jina fine-tuned checkpoint and model topology."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification

import sys
ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.gemini.huy_d1_jina_ft_continuation_v1.common as common

REPO_JINA = ROOT / "models/jina-reranker-v2-base-multilingual"
WEIGHTS = ROOT / "models/from_drive/jina_finetuned/model.safetensors"
OUT_PATH = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/CURRENT_JINA_FT_AUDIT.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(65536):
            h.update(c)
    return h.hexdigest()


def run_audit() -> dict:
    common.patch_transformers_v5()
    print("Computing checkpoint SHA256...", flush=True)
    chk_sha256 = sha256_file(WEIGHTS)
    chk_size = WEIGHTS.stat().st_size

    state = load_file(WEIGHTS)
    checkpoint_keys = set(state.keys())

    print("Instantiating base repo model...", flush=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        REPO_JINA, trust_remote_code=True, dtype=torch.float16
    )
    model_keys = set(base_model.state_dict().keys())

    missing, unexpected = base_model.load_state_dict(
        {k: v.to(torch.float16) for k, v in state.items()}, strict=True
    )
    matched_keys = checkpoint_keys & model_keys

    classifier_keys = [k for k in checkpoint_keys if "classifier" in k or "score" in k]
    classifier_covered = all(k in model_keys for k in classifier_keys)

    audit = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "base_model_identity": "jinaai/jina-reranker-v2-base-multilingual",
        "base_model_path": str(REPO_JINA).replace("\\", "/"),
        "architecture": base_model.__class__.__name__,
        "current_checkpoint_path": str(WEIGHTS).replace("\\", "/"),
        "checkpoint_sha256": chk_sha256,
        "checkpoint_size_bytes": chk_size,
        "checkpoint_tensor_count": len(checkpoint_keys),
        "model_tensor_count": len(model_keys),
        "matched_key_count": len(matched_keys),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "classifier_keys": sorted(classifier_keys),
        "classifier_key_coverage": classifier_covered,
        "inference_passage_contract": {
            "window": 220,
            "overlap": 70,
            "count": 2,
            "max_length": 512,
            "aggregation": "max",
            "score_activation": "sigmoid",
        },
        "training_passage_contract": {
            "window": 220,
            "overlap": 70,
            "count": 1,
            "max_length": 512,
        },
        "historical_training_provenance": {
            "historical_query_count": 1050,
            "source": "notebooks/fine-tune-code/finetune_jina.py:24 (train.json indices 0-749 and 850-1149)",
        },
        "status": "PASS" if len(matched_keys) == 153 and len(missing) == 0 and len(unexpected) == 0 else "FAIL",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"Wrote {OUT_PATH}", flush=True)
    return audit


if __name__ == "__main__":
    run_audit()
