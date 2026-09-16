"""Common utilities, data loaders, and path definitions for HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RES_DIR = ROOT / "results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1"
SRC_DIR = ROOT / "src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1"

# CAL Paths & Constants
CAL_CONTEXTS_DIR = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
EXPECTED_CAL_DOCS_COUNT = 8532
EXPECTED_CAL_QUERIES_COUNT = 600
EXPECTED_CAL_CANDIDATE_PAIRS = 23532

# Canonical V2 Paths
CANONICAL_V2_CONTEXTS_JSONL = ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
BOUNDARY_V2_CONTEXTS_JSONL = ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v2/V2_CONTEXTS.jsonl"
CANONICAL_V2_QUERIES_JSONL = ROOT / "cache/research_v2_e5_confirmation/bundle-v1/V2_TRANSFER_QUERIES.jsonl"
CANONICAL_V2_CANDIDATE_POOL_JSONL = ROOT / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"
MIRROR_V2_CANDIDATE_POOL_JSONL = ROOT / "cache/research_v2_e5_confirmation/bundle-v1/V2_CANDIDATE_POOL.jsonl"
EXPECTED_V2_DOCS_COUNT = 8507
EXPECTED_V2_QUERIES_COUNT = 6991

from tune_corpus_cap32_fusion import build_training_cap


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
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


def compute_query_fingerprint(all_ids: List[str], query_texts: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for q in sorted(all_ids):
        val = query_texts[q]
        text = val[0] if isinstance(val, (list, tuple)) else val
        h.update(f"{q}:{text}\n".encode("utf-8"))
    return h.hexdigest()


def compute_corpus_fingerprint(corpus: Dict[str, Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for doc_id in sorted(corpus.keys(), key=lambda x: int(x) if x.isdigit() else x):
        passage = corpus[doc_id].get("passage", "")
        p_hash = hashlib.sha256(passage.encode("utf-8")).hexdigest()
        h.update(f"{doc_id}:{p_hash}\n".encode("utf-8"))
    return h.hexdigest()


def load_cal_corpus() -> Dict[str, Dict[str, Any]]:
    """Load all 8,532 raw context documents from selected-contexts."""
    files = sorted(CAL_CONTEXTS_DIR.glob("context_*.json"))
    corpus: Dict[str, Dict[str, Any]] = {}
    for f in files:
        doc_id = f.stem[len("context_") :]
        row = json.loads(f.read_text(encoding="utf-8"))
        corpus[doc_id] = {
            "id": doc_id,
            "link": row.get("link", ""),
            "passage": row.get("passage", ""),
        }
    return corpus


def load_cal_generation_inputs() -> Tuple[Dict[str, str], Dict[str, List[str]], List[str], Dict[str, List[str]]]:
    """Load CAL generation inputs strictly WITHOUT exposing gold labels.
    Returns:
        query_texts: Dict[qid, question_text]
        blocks: Dict[block_name, list_of_qids]
        all_ids: List[qid]
        extended: Dict[qid, list_of_baseline_candidate_doc_ids]
    """
    raw_queries, raw_blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    blocks = {k.upper(): v for k, v in raw_blocks.items()}
    # Extract ONLY query text (index 0). Gold labels (index 1) are strictly excluded!
    query_texts = {q: raw_queries[q][0] for q in all_ids}
    return query_texts, blocks, all_ids, extended


def load_cal_gold_labels(all_ids: List[str]) -> Dict[str, Set[str]]:
    """Load CAL gold labels strictly for post-generation evaluation."""
    raw_queries, _, _, _, _, _ = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    return {q: set(raw_queries[q][1]) for q in all_ids}


def load_v2_generation_inputs() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], List[str], Dict[str, List[str]]]:
    """Load canonical V2 corpus, query texts, and candidate pools strictly WITHOUT exposing gold labels."""
    corpus: Dict[str, Dict[str, Any]] = {}
    with open(CANONICAL_V2_CONTEXTS_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                corpus[str(row["doc_id"])] = {
                    "id": str(row["doc_id"]),
                    "link": "",
                    "passage": row.get("passage", ""),
                }

    queries: Dict[str, str] = {}
    with open(CANONICAL_V2_QUERIES_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                queries[str(row["qid"])] = row.get("question", "")

    candidate_pools: Dict[str, List[str]] = {}
    with open(CANONICAL_V2_CANDIDATE_POOL_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                candidate_pools[str(row["qid"])] = [str(d) for d in row.get("doc_ids", [])]

    v2_qids = sorted(queries.keys(), key=lambda x: int(x) if x.isdigit() else x)
    return corpus, queries, v2_qids, candidate_pools


def load_v2_gold_labels(qids: List[str]) -> Dict[str, Set[str]]:
    """Load canonical V2 gold labels strictly for post-generation evaluation."""
    gold: Dict[str, Set[str]] = {}
    with open(CANONICAL_V2_QUERIES_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in qids:
                    gold[qid] = set(str(d) for d in row.get("gold", []))
    return gold
