"""Common constants, loaders, and model definitions for HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src/huy_fasttrack") not in sys.path:
    sys.path.insert(0, str(ROOT / "src/huy_fasttrack"))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

SRC_DIR = ROOT / "src/gemini/huy_d1_jina_passage_adaptation_pilot_v2"
RES_DIR = ROOT / "results/gemini/huy_d1_jina_passage_adaptation_pilot_v2"
ADAPTER_DIR = RES_DIR / "jina_passage_adapted_adapter"

# Model paths
REPO_JINA = ROOT / "models/jina-reranker-v2-base-multilingual"
WEIGHTS_JINA_FT = ROOT / "models/from_drive/jina_finetuned/model.safetensors"
EXPECTED_SHIPPED_SHA256 = "45fe9a9715542dc541ca264de2f26f5cf66a642d98cd51737a25767f6deaab26"

# Canonical Strict-V2 paths
V2_CONTEXTS_JSONL = ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
V2_QUERIES_JSONL = ROOT / "cache/research_v2_e5_confirmation/bundle-v1/V2_TRANSFER_QUERIES.jsonl"
V2_CANDIDATE_POOL_JSONL = ROOT / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"

# CAL Paths & Baselines
CAL_CONTEXTS_DIR = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
CAL_TRAIN_JSON = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
FROZEN_SECTION_CE_CV_PKL = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1/legal_section_ce_cv.pkl"
D1_PREDICTIONS_JSONL = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1/S0_S1_CAL_PREDICTIONS.jsonl"

EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_T0_R5_APPROX = 0.8538888888888888
EXPECTED_ORACLE_T0_APPROX = 0.9683333333333334


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(p: Path) -> str:
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_qid_list_fingerprint(qids: List[str]) -> str:
    h = hashlib.sha256()
    for q in qids:
        h.update(f"{q}\n".encode("utf-8"))
    return h.hexdigest()


def get_git_status() -> Dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        origin = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=str(ROOT), text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        ).strip()
        return {
            "head_commit": head,
            "origin_main_commit": origin,
            "parity": head == origin,
            "status_clean": len(status) == 0,
            "porcelain_output": status,
        }
    except Exception as e:
        return {
            "head_commit": "ERROR",
            "origin_main_commit": "ERROR",
            "parity": False,
            "status_clean": False,
            "error": str(e),
        }


def normalize_text(text: str) -> str:
    """Deterministic question text normalization: NFC, lowercase, whitespace collapsing."""
    t = unicodedata.normalize("NFC", text or "")
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def patch_transformers_v5() -> None:
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module

    if hasattr(module, "create_position_ids_from_input_ids"):
        return

    def helper(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx

    module.create_position_ids_from_input_ids = helper


def patch_tuple_returning_lora(model) -> None:
    """Make PEFT LoRA compatible with Jina-v2's LinearResidual."""
    from types import MethodType

    def tuple_lora_forward(self, x, *args, **kwargs):
        if kwargs.get("adapter_names") is not None:
            raise RuntimeError("mixed-adapter batches are unsupported for LinearResidual")
        kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)
        base_out = self.base_layer(x, *args, **kwargs)
        if isinstance(base_out, tuple):
            result, residual = base_out
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x_in = x.to(lora_A.weight.dtype)
                delta = lora_B(lora_A(dropout(x_in))) * scaling
                result = result + delta.to(result.dtype)
            return result, residual
        else:
            result = base_out
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x_in = x.to(lora_A.weight.dtype)
                delta = lora_B(lora_A(dropout(x_in))) * scaling
                result = result + delta.to(result.dtype)
            return result

    for _, module in model.named_modules():
        if module.__class__.__name__ == "Linear" and hasattr(module, "lora_A"):
            module.forward = MethodType(tuple_lora_forward, module)


def load_jina_base_with_shipped_weights():
    """Load base model and overlay shipped jina_finetuned weights."""
    patch_transformers_v5()
    tok = AutoTokenizer.from_pretrained(
        REPO_JINA, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO_JINA, trust_remote_code=True, dtype=torch.float16
    )
    state = load_file(WEIGHTS_JINA_FT)
    missing, unexpected = model.load_state_dict(
        {k: v.to(torch.float16) for k, v in state.items()}, strict=True
    )
    model._tokenizer = tok
    return model, tok


def build_pure_lora_jina_model(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """Build Pure-LoRA Jina model with strictly frozen classifier and attention projection targets."""
    model, tok = load_jina_base_with_shipped_weights()

    # Explicitly freeze every original parameter before enabling LoRA
    for p in model.parameters():
        p.requires_grad = False

    # Pure LoRA configuration: targeting ONLY transformer attention projections
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["mixer.Wqkv", "mixer.out_proj"],
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    patch_tuple_returning_lora(lora_model)
    lora_model.gradient_checkpointing_enable()
    lora_model.enable_input_require_grads()

    # Cast trainable adapter weights to float32 for training stability
    for p in lora_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    return lora_model, tok


def get_classifier_state_hash(model) -> str:
    """Compute deterministic SHA256 of classifier head parameters."""
    classifier_state = model.model.classifier.state_dict() if hasattr(model, "model") else model.classifier.state_dict()
    h = hashlib.sha256()
    for k in sorted(classifier_state.keys()):
        h.update(f"{k}:".encode("utf-8"))
        h.update(classifier_state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def get_frozen_base_parameters_hash(model) -> str:
    """Compute deterministic SHA256 over all frozen non-LoRA parameters."""
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        if not p.requires_grad:
            h.update(f"{n}:".encode("utf-8"))
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_v2_contexts() -> Dict[str, str]:
    """Load canonical V2 document contexts: doc_id -> passage."""
    contexts: Dict[str, str] = {}
    with open(V2_CONTEXTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                contexts[str(row["doc_id"])] = str(row.get("passage", "") or "")
    return contexts


def load_v2_inputs():
    """Load canonical V2 5-fold partition, candidate pools, questions, and golds."""
    import run_huy_5fold_fasttrack as core
    folds, pools, questions, v2_golds, _, _, _, _ = core.load_inputs()
    return folds, pools, questions, v2_golds


def load_cal_contexts() -> Dict[str, str]:
    """Load CAL600 raw document contexts with title_from_link fallback."""
    from run_burst_expanded_fusion_submission import title_from_link

    contexts: Dict[str, str] = {}
    for f in sorted(CAL_CONTEXTS_DIR.glob("context_*.json")):
        doc_id = f.stem[len("context_") :]
        row = json.loads(f.read_text(encoding="utf-8"))
        passage = row.get("passage")
        if not passage:
            passage = title_from_link(row.get("link")) or ""
        contexts[doc_id] = str(passage)
    return contexts


def load_cal_questions_label_free() -> Tuple[List[str], Dict[str, str]]:
    """Genuinely label-free CAL query loader: extracts ONLY question strings, NEVER touches answer labels."""
    if not D1_PREDICTIONS_JSONL.exists():
        raise FileNotFoundError(f"D1 predictions file not found: {D1_PREDICTIONS_JSONL}")
    cal_qids: List[str] = []
    with open(D1_PREDICTIONS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cal_qids.append(str(json.loads(line)["qid"]))

    raw = json.loads(CAL_TRAIN_JSON.read_text(encoding="utf-8"))
    questions: Dict[str, str] = {}
    for qid in cal_qids:
        # Strictly read ONLY the question text field
        questions[qid] = str(raw[qid]["question"])
    return cal_qids, questions


def load_cal_candidate_pools() -> Dict[str, List[str]]:
    """Load CAL candidate pools label-free from authoritative section CE cache."""
    import pickle
    cached = pickle.loads(FROZEN_SECTION_CE_CV_PKL.read_bytes())
    scores = cached["scores"]
    return {q: list(scores[q].keys()) for q in scores}


def load_cal_gold_labels(cal_qids: List[str]) -> Dict[str, Set[str]]:
    """Load CAL gold labels STRICTLY after Held V2 evaluation has materialized."""
    raw = json.loads(CAL_TRAIN_JSON.read_text(encoding="utf-8"))
    return {q: set(str(d) for d in raw[q]["answer"]) for q in cal_qids}
