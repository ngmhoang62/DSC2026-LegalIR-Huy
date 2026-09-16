"""Common utilities, data loaders, and model definitions for HUY_D1_SECTION_RESIDUAL_SELECTOR_V1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RES_DIR = ROOT / "results/gemini/huy_d1_section_residual_selector_v1"
SRC_DIR = ROOT / "src/gemini/huy_d1_section_residual_selector_v1"

OLD_JINA_CACHE_PKL = ROOT / "results/from_drive/jina_ft_cv.pkl"
EXPECTED_OLD_JINA_CACHE_SHA256 = "666296dc0bffdf7366c2b5834aed1612bb34ffc45bbd2696b227338367282d00"

SECTION_CE_CACHE_PKL = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1/legal_section_ce_cv.pkl"
EXPECTED_SECTION_CACHE_SHA256 = "94daae7c1273d6f1615912b8d52b64a194620bf08c6f26d7f220495bfeadb0c7"

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

FEATURE_NAMES_13 = [
    "d1_z",
    "d1_gap_to_top",
    "d1_recip_rank",
    "old_jina_z",
    "old_jina_gap_to_top",
    "old_jina_recip_rank",
    "section_ce_z",
    "section_ce_gap_to_top",
    "section_ce_recip_rank",
    "section_minus_old_z",
    "section_minus_old_recip_rank",
    "is_d1_top5",
    "is_section_top5",
]


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception as e:
        return f"UNKNOWN_{e}"


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


def compute_candidate_fingerprint(
    all_ids: List[str], extended: Dict[str, List[str]]
) -> str:
    h = hashlib.sha256()
    for q in sorted(all_ids):
        h.update(f"{q}:".encode("utf-8"))
        for d in sorted(extended[q]):
            h.update(f"{d},".encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_query_fingerprint(all_ids: List[str], queries: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for q in sorted(all_ids):
        text = queries[q][0]
        h.update(f"{q}:{text}\n".encode("utf-8"))
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


def load_pkl(rel_path: str | Path):
    p = ROOT / rel_path if isinstance(rel_path, str) else rel_path
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


def compute_13_pointwise_features(
    q: str,
    cand_docs: List[str],
    d1_scores_dict: Dict[str, float],
    old_jina_scores_dict: Dict[str, float],
    section_scores_dict: Dict[str, float],
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """Compute exact 13 pointwise features from the query full existing candidate pool.
    
    Streams:
    1. D1: d1_z, d1_gap_to_top, d1_recip_rank
    2. Old Jina: old_jina_z, old_jina_gap_to_top, old_jina_recip_rank
    3. Section CE: section_ce_z, section_ce_gap_to_top, section_ce_recip_rank
    4. Interaction: section_minus_old_z, section_minus_old_recip_rank
    5. Top-5 flags: is_d1_top5, is_section_top5
    """
    # 1. D1 stream
    d1_arr = np.array([d1_scores_dict[d] for d in cand_docs], dtype=np.float64)
    d1_std = float(np.std(d1_arr))
    d1_z = (d1_arr - float(np.mean(d1_arr))) / (d1_std if d1_std > 1e-9 else 1.0)
    d1_gap = d1_arr - float(np.max(d1_arr))
    d1_order = sorted(range(len(cand_docs)), key=lambda i: d1_arr[i], reverse=True)
    d1_ranks = {cand_docs[d1_order[r]]: r + 1 for r in range(len(cand_docs))}
    d1_top5 = [cand_docs[d1_order[r]] for r in range(min(5, len(cand_docs)))]
    d1_top5_set = set(d1_top5)

    # 2. Old Jina stream
    old_arr = np.array([old_jina_scores_dict.get(d, -999.0) for d in cand_docs], dtype=np.float64)
    old_std = float(np.std(old_arr))
    old_z = (old_arr - float(np.mean(old_arr))) / (old_std if old_std > 1e-9 else 1.0)
    old_gap = old_arr - float(np.max(old_arr))
    old_order = sorted(range(len(cand_docs)), key=lambda i: old_arr[i], reverse=True)
    old_ranks = {cand_docs[old_order[r]]: r + 1 for r in range(len(cand_docs))}

    # 3. Section CE stream
    sec_arr = np.array([section_scores_dict.get(d, -999.0) for d in cand_docs], dtype=np.float64)
    sec_std = float(np.std(sec_arr))
    sec_z = (sec_arr - float(np.mean(sec_arr))) / (sec_std if sec_std > 1e-9 else 1.0)
    sec_gap = sec_arr - float(np.max(sec_arr))
    sec_order = sorted(range(len(cand_docs)), key=lambda i: sec_arr[i], reverse=True)
    sec_ranks = {cand_docs[sec_order[r]]: r + 1 for r in range(len(cand_docs))}
    sec_top5 = [cand_docs[sec_order[r]] for r in range(min(5, len(cand_docs)))]
    sec_top5_set = set(sec_top5)

    feats: Dict[str, np.ndarray] = {}
    for i, d in enumerate(cand_docs):
        r_d1 = d1_ranks[d]
        r_old = old_ranks[d]
        r_sec = sec_ranks[d]

        recip_d1 = 1.0 / (10.0 + r_d1)
        recip_old = 1.0 / (10.0 + r_old)
        recip_sec = 1.0 / (10.0 + r_sec)

        f13 = np.array([
            d1_z[i],
            d1_gap[i],
            recip_d1,
            old_z[i],
            old_gap[i],
            recip_old,
            sec_z[i],
            sec_gap[i],
            recip_sec,
            sec_z[i] - old_z[i],
            recip_sec - recip_old,
            1.0 if d in d1_top5_set else 0.0,
            1.0 if d in sec_top5_set else 0.0,
        ], dtype=np.float64)
        feats[d] = f13

    return feats, d1_top5, sec_top5
