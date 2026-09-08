"""Exact local loader for Huy's Jina-FT cross-encoder checkpoint.

The saved fine-tune directory intentionally contains weights, tokenizer JSON,
and only three of the remote-code modules.  The remaining architecture modules
already exist in the immutable Hugging Face cache.  This loader joins those two
read-only locations into one Python package; it never edits or downloads model
files.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Iterable, Sequence

import torch
from safetensors.torch import load_model
from transformers import PreTrainedTokenizerFast


REPO = Path(__file__).resolve().parents[2]
LOCAL_MODEL = REPO / "fine_tune" / "jina_finetuned"
FLASH_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--jinaai--xlm-roberta-flash-implementation"
    / "snapshots"
    / "845308d0fd72a8406a3e378450e1a09522790419"
)
PACKAGE = "_sol_high_rl_jina_flash"


def _architecture_modules():
    if not FLASH_CACHE.is_dir():
        raise FileNotFoundError(f"Missing cached Jina architecture: {FLASH_CACHE}")
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        # Prefer the source files shipped beside this exact fine-tune. Missing
        # support modules are resolved from the immutable architecture cache.
        package.__path__ = [str(LOCAL_MODEL), str(FLASH_CACHE)]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    config_mod = importlib.import_module(f"{PACKAGE}.configuration_xlm_roberta")
    model_mod = importlib.import_module(f"{PACKAGE}.modeling_xlm_roberta")
    # The cached support Block comes from a later compatible commit and passes
    # adapter_mask=None to Mlp. The checkpoint's older Mlp has no such keyword;
    # accepting and ignoring None restores the old no-adapter call exactly.
    mlp_mod = importlib.import_module(f"{PACKAGE}.mlp")
    if not getattr(mlp_mod.Mlp.forward, "_sol_adapter_compat", False):
        original_forward = mlp_mod.Mlp.forward

        def compatible_forward(self, x, adapter_mask=None):
            if adapter_mask is not None:
                raise RuntimeError("This frozen checkpoint has no MLP adapters")
            return original_forward(self, x)

        compatible_forward._sol_adapter_compat = True
        mlp_mod.Mlp.forward = compatible_forward
    return config_mod, model_mod


def load_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(LOCAL_MODEL / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=512,
    )


def load_model_and_tokenizer(device: str = "cuda", dtype=torch.float16):
    config_mod, model_mod = _architecture_modules()
    config = config_mod.XLMRobertaFlashConfig(
        vocab_size=250002,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        classifier_dropout=None,
        use_flash_attn=False,
        num_labels=1,
    )
    model = model_mod.XLMRobertaForSequenceClassification(config)
    missing, unexpected = load_model(
        model, str(LOCAL_MODEL / "model.safetensors"), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"Non-exact state load: missing={missing}, unexpected={unexpected}")
    model.eval().to(device=device, dtype=dtype)
    return model, load_tokenizer()


@torch.inference_mode()
def score_pairs(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float]:
    device = next(model.parameters()).device
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [x[0] for x in batch],
            [x[1] for x in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits.squeeze(-1)
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return scores


def smoke_test() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Jina-FT smoke test")
    model, tokenizer = load_model_and_tokenizer()
    pairs = [
        ("Hợp đồng lao động chấm dứt khi nào?", "Điều 34. Các trường hợp chấm dứt hợp đồng lao động."),
        ("Hợp đồng lao động chấm dứt khi nào?", "Quy định về thuế xuất khẩu, thuế nhập khẩu."),
    ]
    scores = score_pairs(model, tokenizer, pairs, batch_size=2)
    return {
        "strict_state_load": True,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "scores": scores,
        "max_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }


if __name__ == "__main__":
    print(json.dumps(smoke_test(), ensure_ascii=False, indent=2))
