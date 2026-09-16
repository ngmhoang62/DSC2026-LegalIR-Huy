"""Common utilities, model loading, and dataset loaders for HUY_D1_LEGAL_SECTION_EVIDENCE_V1."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RES_DIR = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1"
SRC_DIR = ROOT / "src/gemini/huy_d1_legal_section_evidence_v1"

REPO_JINA = ROOT / "models/jina-reranker-v2-base-multilingual"
WEIGHTS_JINA_FT = ROOT / "models/from_drive/jina_finetuned/model.safetensors"
SCORE_CACHE_PKL = RES_DIR / "legal_section_ce_cv.pkl"

from run_burst_expanded_fusion_submission import title_from_link
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_BLOCK_RECALLS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SafeDocumentStore:
    """Safe lazy document store that defines __contains__ and avoids infinite loops."""

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
            text = row.get("passage") or title_from_link(row.get("link")) or ""
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[doc] = text
        return text


def patch_transformers_v5() -> None:
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module

    if hasattr(module, "create_position_ids_from_input_ids"):
        return

    def helper(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx

    module.create_position_ids_from_input_ids = helper


def load_jina_crossencoder() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load frozen Jina cross-encoder with shipped fine-tuned weights and log provenance."""
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
    model.eval().to("cuda")

    provenance = {
        "base_model_path": str(REPO_JINA).replace("\\", "/"),
        "base_model_name": "jinaai/jina-reranker-v2-base-multilingual",
        "weights_path": str(WEIGHTS_JINA_FT).replace("\\", "/"),
        "weights_sha256": sha256_file(WEIGHTS_JINA_FT),
        "weights_tensors_count": len(state),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "dtype": "torch.float16",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "max_length": 512,
        "is_frozen": True,
    }
    return model, tok, provenance


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_cal_data():
    docs = SafeDocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, raw_blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    # Normalize block keys to uppercase
    blocks = {k.upper(): v for k, v in raw_blocks.items()}
    gold = {q: set(queries[q][1]) for q in all_ids}

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

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
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    return (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    )
