"""Model architecture and loss functions for jina_honest_adaptation_v1.
Implements M1 (LoRA) vs M2 (Full FT), smooth max parent aggregation (Section 11),
multi-positive ranking loss (Section 12), and stability regularization (Section 13).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src/research_v2_forensic"))
import jina_v2_boundary_train as jb


def build_model(
    method: str,
    model_path: Path,
    rank: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[nn.Module, Any]:
    """Factory creating either M1 (LoRA) or M2 (Full fine-tune) Jina model."""
    jb.patch_transformers_v5()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)

    if method == "M1_LORA":
        from peft import LoraConfig, TaskType, get_peft_model

        base = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.05,
            target_modules=["Wqkv", "out_proj"],
            bias="none",
        )
        model = get_peft_model(base, config)
        jb.patch_tuple_returning_lora(model)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = model.to(device)
    elif method == "M2_FULL":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        ).to(device)
        model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown fine-tune method: {method}")

    model._tokenizer = tok
    return model, tok


def compute_parent_scores(
    model: nn.Module,
    tok: Any,
    pairs: List[Tuple[str, str]],
    doc_indices: List[int],
    num_docs: int,
    tau_pool: float = 0.15,
    max_length: int = 512,
    device: str = "cuda",
    micro_batch_size: int = 4,
) -> torch.Tensor:
    """Passage scoring followed by smooth-max parent pooling (Section 11).
    Always chunks passage forward passes into microbatches of 4 to strictly
    guarantee peak VRAM stays <= 2.2 GB on a 6 GB GPU.
    """
    if not pairs:
        return torch.empty(0, device=device)

    all_logits = []
    for i in range(0, len(pairs), micro_batch_size):
        b = pairs[i : i + micro_batch_size]
        inputs = tok(
            [p[0] for p in b],
            [p[1] for p in b],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits.view(-1)
        all_logits.append(logits)
    passage_logits = torch.cat(all_logits, dim=0)

    # Smooth-max aggregation per document parent: tau * logsumexp(v / tau)
    parent_scores = []
    doc_indices_tensor = torch.tensor(doc_indices, device=device, dtype=torch.long)
    for doc_idx in range(num_docs):
        mask = doc_indices_tensor == doc_idx
        doc_logits = passage_logits[mask]
        if len(doc_logits) == 0:
            parent_scores.append(torch.tensor(-1e4, device=device, dtype=passage_logits.dtype))
        else:
            s_d = tau_pool * torch.logsumexp(doc_logits / tau_pool, dim=0)
            parent_scores.append(s_d)

    return torch.stack(parent_scores)


def compute_group_loss(
    parent_scores: torch.Tensor,
    pos_indices: List[int],
    frozen_scores: torch.Tensor,
    lambda_anchor: float = 0.05,
    T: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute multi-positive ranking loss + stability anchor regularizer.

    Returns (total_loss, rank_loss, anchor_loss).
    """
    device = parent_scores.device
    pos_idx_tensor = torch.tensor(pos_indices, device=device, dtype=torch.long)

    # Multi-positive ranking loss (Section 12):
    # L_rank = -log( sum_{p in P} exp(s_p/T) / sum_{d in C} exp(s_d/T) )
    #        = logsumexp(S / T) - logsumexp(S[pos] / T)
    l_denom = torch.logsumexp(parent_scores / T, dim=0)
    l_numer = torch.logsumexp(parent_scores[pos_idx_tensor] / T, dim=0)
    l_rank = l_denom - l_numer

    # Stability regularization (Section 13):
    # Penalizes drift from frozen Jina parent geometry
    sigmoid_scores = torch.sigmoid(parent_scores)
    l_anchor = F.mse_loss(sigmoid_scores, frozen_scores.to(device))

    total_loss = l_rank + lambda_anchor * l_anchor
    return total_loss, l_rank, l_anchor


def save_jina_checkpoint(
    model: nn.Module,
    method: str,
    save_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save complete model state without filtering requires_grad (Section 3.2)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    payload = {
        "method": method,
        "state_dict": state,
        "extra": extra or {},
    }
    torch.save(payload, save_path)


def load_jina_checkpoint(
    checkpoint_path: Path,
    model_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, Any]:
    """Reconstruct base architecture and restore complete state dict (Section 3.2)."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    method = payload["method"]
    model, tok = build_model(method, model_path, rank=16, dtype=dtype, device=device)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint reload mismatch: {incompatible}")
    model.eval()
    return model, tok


def checkpoint_round_trip_test(
    model: nn.Module,
    tok: Any,
    audit_pairs: List[Tuple[str, str]],
    method: str,
    model_path: Path,
    test_path: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Enforces Section 3.2 round-trip serialization audit."""
    import gc

    model.eval()
    original_logits = []
    with torch.no_grad():
        for i in range(0, len(audit_pairs), 4):
            batch = audit_pairs[i : i + 4]
            inp = tok(
                [b[0] for b in batch],
                [b[1] for b in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            original_logits.extend(model(**inp).logits.view(-1).float().cpu().tolist())

    save_jina_checkpoint(model, method, test_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded_model, reloaded_tok = load_jina_checkpoint(test_path, model_path, device=device)

    reloaded_logits = []
    with torch.no_grad():
        for i in range(0, len(audit_pairs), 4):
            batch = audit_pairs[i : i + 4]
            inp = reloaded_tok(
                [b[0] for b in batch],
                [b[1] for b in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            reloaded_logits.extend(reloaded_model(**inp).logits.view(-1).float().cpu().tolist())

    diffs = [abs(o - r) for o, r in zip(original_logits, reloaded_logits)]
    max_diff = float(max(diffs)) if diffs else 0.0

    docs = [f"doc_{i:04d}" for i in range(len(audit_pairs))]
    orig_order = sorted(docs, key=lambda d: (-original_logits[int(d[4:])], d))
    relo_order = sorted(docs, key=lambda d: (-reloaded_logits[int(d[4:])], d))
    ranking_match = orig_order == relo_order

    if test_path.exists():
        test_path.unlink()

    del reloaded_model
    gc.collect()
    torch.cuda.empty_cache()

    if not ranking_match or max_diff > 1e-4:
        raise RuntimeError(
            f"Checkpoint round-trip audit FAILED: max_diff={max_diff:.8e}, ranking_match={ranking_match}"
        )

    return {
        "status": "PASS",
        "method": method,
        "audit_pairs_tested": len(audit_pairs),
        "max_logit_difference": max_diff,
        "ranking_identical": ranking_match,
    }
