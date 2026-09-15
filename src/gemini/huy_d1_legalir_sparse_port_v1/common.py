"""Common definitions and utilities for HUY_D1_LEGALIR_SPARSE_PORT_V1."""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "HUY_D1_LEGALIR_SPARSE_PORT_V1"
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_legalir_sparse_port_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LEGALIR_DIR = Path("D:/Study/DSC2026/LegalIR")
SOURCES_DB = LEGALIR_DIR / "cache" / "exp112_task_adaptive_retrieval" / "sources.sqlite"
EXP021_DB = LEGALIR_DIR / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "bm25_v3.sqlite"
TRIGRAM_DB = LEGALIR_DIR / "cache" / "exp111_multiview_sparse" / "v2_v3" / "index.sqlite"

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SPARSE_RANK_VIEWS = ["legalir_bm25", "legalir_trigram"]
S1_VIEWS = D1_VIEWS + SPARSE_RANK_VIEWS
SPARSE_SCORE_CHANNELS = ["legalir_bm25", "legalir_trigram"]

EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_DIM = 48
EXPECTED_S1_DIM = 56

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

DUPLICATE_MAP = {
    "121575": "84226",
    "158189": "206810",
    "184972": "206810",
    "254937": "280171",
    "35337": "277743",
}

EMPTY_PASSAGE_DOCS = {
    "10533", "131890", "149317", "177151", "181693", "187338", "191261",
    "196918", "208668", "210808", "232489", "255762", "263763", "288457",
    "34810", "55497", "56098", "57978", "67660", "71014"
}


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_aligned(rel_path: str, candidate_pool: Dict[str, List[str]], all_ids: List[str], floor=None):
    obj = load_pkl(rel_path)
    fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
    return {q: {d: obj.get(q, {}).get(d, fl) for d in candidate_pool[q]} for q in all_ids}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_sparse_channels_and_views(candidate_pool: Dict[str, List[str]], qids: List[str]) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, List[str]]]]:
    con = sqlite3.connect(f"file:{SOURCES_DB.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    scores = {"legalir_bm25": {}, "legalir_trigram": {}}
    ranks = {"legalir_bm25": {}, "legalir_trigram": {}}

    for source_key, channel in [("legalir_bm25", "bm25"), ("legalir_trigram", "trigram")]:
        cur = con.cursor()
        for qid in qids:
            docs = candidate_pool[qid]
            wanted = set(docs)
            row = cur.execute("SELECT payload FROM sources WHERE q=? AND source=?", (qid, channel)).fetchone()
            score_map = {}
            native_rank = {}
            if row:
                values = json.loads(row[0])
                for r in values:
                    d = str(r["doc_id"])
                    if d in wanted:
                        score_map[d] = float(r["score"])
                        native_rank[d] = int(r["rank"])
                for d in docs:
                    if d not in score_map and d in DUPLICATE_MAP:
                        twin = DUPLICATE_MAP[d]
                        for r in values:
                            if str(r["doc_id"]) == twin:
                                score_map[d] = float(r["score"])
                                native_rank[d] = int(r["rank"])
                                break
            order = sorted(docs, key=lambda d: (native_rank.get(d, 10**9), d))
            scores[source_key][qid] = score_map
            ranks[source_key][qid] = order
    con.close()
    return scores, ranks
