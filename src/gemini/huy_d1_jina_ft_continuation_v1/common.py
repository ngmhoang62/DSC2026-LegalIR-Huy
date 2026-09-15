"""Common utilities, model loading, and dataset helpers for HUY_D1_JINA_FT_CONTINUATION_V1."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src/huy_fasttrack") not in sys.path:
    sys.path.insert(0, str(ROOT / "src/huy_fasttrack"))

import run_huy_5fold_fasttrack as core
from benchmark_burst_v4_full_sqlite import tokens
from src.gemini.huy_d1_lal_case_memory_v1.audit_data_isolation import load_duplicate_graph
from tune_corpus_cap32_fusion import build_training_cap

REPO_JINA = ROOT / "models/jina-reranker-v2-base-multilingual"
WEIGHTS_JINA_FT = ROOT / "models/from_drive/jina_finetuned/model.safetensors"
CONTEXTS_PATH = (
    ROOT
    / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
)
H_LOCAL_PATH = (
    ROOT
    / "results/gemini/huy_sparse_resurrection_v1/cache/BURST_V2_RETRIEVAL_RESULTS.jsonl"
)

STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}
SPACE_RE = re.compile(r"\S+", re.UNICODE)


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def top_passages(
    question: str, text: str, count: int = 2, window: int = 220, overlap: int = 70
) -> List[str]:
    """Huy exact lexical preselection; return original-Unicode windows for Jina."""
    words = SPACE_RE.findall(text or "")
    if len(words) <= window + 80:
        return [" ".join(words)]
    query_tokens = tokens(question)
    content = {t for t in query_tokens if len(t) >= 3 and t not in STOPWORDS}
    numbers = {t for t in query_tokens if any(c.isdigit() for c in t)}
    bigrams = {" ".join(query_tokens[i : i + 2]) for i in range(len(query_tokens) - 1)}
    header = " ".join(words[:70])
    scored = []
    step = window - overlap
    for start in range(0, len(words), step):
        end = min(start + window, len(words))
        part_words = words[start:end]
        part = " ".join(part_words)
        normalized = tokens(part)
        token_set = set(normalized)
        norm_text = " ".join(normalized)
        coverage = sum(
            1.0 + 0.20 * min(normalized.count(t), 3)
            for t in content
            if t in token_set
        )
        numeric = 3.0 * sum(t in token_set for t in numbers)
        phrase = 1.8 * sum(p in norm_text for p in bigrams)
        density = (coverage + numeric + phrase) / math.sqrt(max(len(normalized), 1))
        scored.append((density, coverage + numeric + phrase, -start, part))
        if end == len(words):
            break
    scored.sort(reverse=True)
    passages = []
    for _, _, neg_start, part in scored:
        candidate = (
            part if -neg_start < 70 else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
        )
        if candidate not in passages:
            passages.append(candidate)
        if len(passages) >= count:
            break
    return passages


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


def build_lora_jina_model(
    r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05
):
    """Build LoRA model initialized strictly from shipped jina_finetuned weights."""
    model, tok = load_jina_base_with_shipped_weights()
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["Wqkv", "out_proj"],
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    patch_tuple_returning_lora(lora_model)
    lora_model.gradient_checkpointing_enable()
    lora_model.enable_input_require_grads()

    # Cast trainable adapter weights to float32
    for p in lora_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    return lora_model, tok


def load_contexts() -> Dict[str, str]:
    """Load canonical V2 document contexts."""
    contexts: Dict[str, str] = {}
    with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                contexts[str(rec["doc_id"])] = str(rec.get("passage") or "")
    return contexts


def load_h_local() -> Dict[str, List[str]]:
    """Load H_LOCAL retrieval rankings."""
    h_local: Dict[str, List[str]] = {}
    if H_LOCAL_PATH.exists():
        with open(H_LOCAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    h_local[str(rec["qid"])] = [
                        str(x[0]) for x in rec.get("h_local_top100", [])
                    ]
    return h_local
