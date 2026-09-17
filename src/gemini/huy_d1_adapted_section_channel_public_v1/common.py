"""Common constants, data loaders, and model loaders for HUY_D1_ADAPTED_SECTION_CHANNEL_PUBLIC_V1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_DIR = ROOT / "src" / "gemini" / "huy_d1_adapted_section_channel_public_v1"
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_adapted_section_channel_public_v1"

UPSTREAM_COMMIT = "bd52a78b54cc13bd11fb3314587de73f65d49f62"
UPSTREAM_RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_jina_passage_adaptation_pilot_v2"
ADAPTER_DIR = UPSTREAM_RESULTS_DIR / "jina_passage_adapted_adapter"

FROZEN_SECTION_CACHE_PATH = ROOT / "results" / "gemini" / "huy_d1_legal_section_evidence_v1" / "legal_section_ce_cv.pkl"
D1_CHAMPION_ZIP_PATH = ROOT / "results" / "gemini" / "huy_vnlegal_rank_ablation_v1" / "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.zip"
D1_CHAMPION_JSON_PATH = ROOT / "results" / "gemini" / "huy_vnlegal_rank_ablation_v1" / "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"

BASE_MODEL_PATH = ROOT / "models" / "jina-reranker-v2-base-multilingual"
FINETUNED_WEIGHTS_PATH = ROOT / "models" / "from_drive" / "jina_finetuned" / "model.safetensors"

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_BLOCK_RECALLS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}
EXPECTED_FROZEN_SECTION_R5 = 0.9577777777777777

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}
EXTRA_PUB_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_public.pkl",
    "jina_ft": "results/from_drive/jina_ft_public.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_public.pkl",
}


def seed_everything(seed: int = 2026):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_fingerprint(items: List[str]) -> str:
    h = hashlib.sha256()
    for it in sorted(items):
        h.update(f"{it}\n".encode("utf-8"))
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


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


class SafeDocumentStore:
    def __init__(self, paths, cache_size=1024):
        self.paths = {p.stem[len("context_") :]: p for p in paths}
        self.cache: Dict[str, str] = {}
        self.cache_size = cache_size

    def __contains__(self, doc: str) -> bool:
        return doc in self.paths

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, doc: str, default: str = "") -> str:
        if doc not in self.paths:
            return default
        return self[doc]

    def __getitem__(self, doc: str) -> str:
        text = self.cache.get(doc)
        if text is None:
            path = self.paths.get(doc)
            if path is None:
                return ""
            row = json.loads(path.read_text(encoding="utf-8"))
            from run_burst_expanded_fusion_submission import title_from_link
            text = row.get("passage") or title_from_link(row.get("link")) or ""
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[doc] = text
        return text


def get_source_files_sha256() -> Dict[str, Dict[str, Any]]:
    source_files = {}
    for f in sorted(SOURCE_DIR.glob("*.py")):
        source_files[f.name] = {
            "path": str(f.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": f.stat().st_size,
            "sha256": sha256_file(f),
        }
    return source_files


def load_cal_data_label_free():
    """Load CAL data without labels: docs, query text, blocks, qids, candidates, views, scores.
    Strictly zero CAL gold labels are read, materialized, or returned.
    """
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    docs = SafeDocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    raw_queries, raw_blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    blocks = {k.upper(): v for k, v in raw_blocks.items()}
    # Strictly strip all gold answers: queries_label_free only contains question text
    queries_label_free = {q: (raw_queries[q][0], None) for q in all_ids}
    del raw_queries

    def load_aligned(rel_path: str, floor=None):
        raw = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in raw for v in raw[q].values())
        return {q: {d: raw.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned(rel) for name, rel in EXTRA_CV_PATHS.items()}
    full_channels_cv = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries_label_free, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    return (
        docs,
        queries_label_free,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
    )


def load_cal_gold_labels(all_ids: List[str]) -> Tuple[Dict[str, Set[str]], str]:
    """Strictly loads gold labels for evaluation stage AFTER adapted cache seal.
    Returns gold mapping and timestamp ISO string.
    """
    reveal_time_utc = datetime.now(timezone.utc).isoformat()
    path = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "train.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    gold = {
        str(qid): {str(d) for d in raw[str(qid)]["answer"]}
        for qid in all_ids
    }
    return gold, reveal_time_utc


def load_cal_data():
    """Convenience helper combining label-free data and gold labels for backward compatibility."""
    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
    ) = load_cal_data_label_free()
    gold, _ = load_cal_gold_labels(all_ids)
    # Restore tuple format (q_text, gold_set) for compatibility
    queries_with_gold = {q: (queries[q][0], gold[q]) for q in all_ids}
    return (
        docs,
        queries_with_gold,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    )



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


def load_shipped_frozen_jina(device: str = "cuda"):
    from safetensors.torch import load_file

    patch_transformers_v5()
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, dtype=torch.float16
    )
    state = load_file(FINETUNED_WEIGHTS_PATH)
    model.load_state_dict(
        {k: v.to(torch.float16) for k, v in state.items()}, strict=True
    )
    model.eval().to(device)
    model._tokenizer = tok
    return model, tok


def load_adapted_jina_model(device: str = "cuda"):
    base_model, tok = load_shipped_frozen_jina(device=device)
    adapted_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    patch_tuple_returning_lora(adapted_model)
    adapted_model.eval().to(device)
    return adapted_model, base_model, tok
