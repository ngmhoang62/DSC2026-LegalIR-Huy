"""Common definitions, utilities, exact D1 baseline, and parity verification.

Module for HUY_D1_SPARSE_MUTUAL_TOP5_V1.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

EXPERIMENT_ID = "HUY_D1_SPARSE_MUTUAL_TOP5_V1"
SRC_DIR = ROOT / "src/gemini/huy_d1_sparse_mutual_top5_v1"
RESULTS_DIR = ROOT / "results/gemini/huy_d1_sparse_mutual_top5_v1"

# Exact D1 Specification
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_DIM = 48
EXPECTED_BLOCK_RECALLS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}
SEED = 2026

# CAL data paths
CAL_QUESTIONS_LABEL_FREE_PATH = ROOT / "src/gemini/huy_d1_frozen_section_public_v1/CAL_QUESTIONS_LABEL_FREE.json"
CAL_CONTEXTS_DIR = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
CAL_GOLD_PATH = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}

from run_burst_expanded_fusion_submission import title_from_link
from tune_citation_graph import build_citation_table, citation_features
from tune_doctype_features import build_type_table, type_features


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
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


def get_source_files_sha256() -> Dict[str, Dict[str, Any]]:
    files = {}
    for p in sorted(list(SRC_DIR.glob("*.py"))):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        files[p.name] = {
            "path": rel,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    return files


class SafeDocumentStore:
    """Safe lazy document store that defines __contains__ and avoids infinite loops."""

    def __init__(self, paths, cache_size=2048):
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


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_retrieval_cache(root: Path, tag: str) -> Dict[str, Any]:
    large = root / "results" / "burst_large_ltr"
    if tag == "validation_a":
        return pickle.loads(
            (large / "retrieval_train1000_tune50_val100.pkl").read_bytes()
        )["cache"]
    return pickle.loads((large / f"{tag}_retrieval.pkl").read_bytes())["cache"]


def raw_union_local(cache_row, depth: int = 50, rrf_k: int = 20) -> List[str]:
    scores = {}
    for branch in cache_row:
        for rank, item in enumerate(branch[:depth], 1):
            doc = str(item[0])
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


def weighted_rrf_local(
    rankings: List[Dict[str, List[str]]], weights: Tuple[float, float], k: int
) -> Dict[str, List[str]]:
    output = {}
    for q in rankings[0]:
        rankmaps = [{d: i + 1 for i, d in enumerate(branch[q])} for branch in rankings]
        docs = set().union(*(r.keys() for r in rankmaps))
        output[q] = sorted(
            docs,
            key=lambda d: (
                -sum(w / (k + ranks.get(d, 100000)) for w, ranks in zip(weights, rankmaps)),
                d,
            ),
        )[:100]
    return output


def load_cal_data_label_free():
    """Load CAL data without labels: docs, query text, blocks, qids, candidates, views, scores.
    Strictly zero CAL gold labels are read, materialized, or returned.
    """
    docs = SafeDocumentStore(sorted(CAL_CONTEXTS_DIR.glob("context_*.json")))

    if not CAL_QUESTIONS_LABEL_FREE_PATH.exists():
        raise FileNotFoundError(
            f"Missing label-free questions artifact: {CAL_QUESTIONS_LABEL_FREE_PATH}"
        )

    raw_q = json.loads(CAL_QUESTIONS_LABEL_FREE_PATH.read_text(encoding="utf-8"))
    questions_map = {
        str(k): (v["question"] if isinstance(v, dict) else str(v))
        for k, v in raw_q.items()
    }
    all_ids = list(questions_map.keys())
    if len(all_ids) != 600:
        raise ValueError(
            f"Expected exactly 600 queries in label-free questions, got {len(all_ids)}"
        )

    blocks = {
        "A": all_ids[:100],
        "B": all_ids[100:200],
        "C": all_ids[200:300],
        "D": all_ids[300:600],
    }
    TAGS = {
        "A": "validation_a",
        "B": "fresh_1251_1350",
        "C": "fresh_1351_1450",
        "D": "fresh_1451_1750",
    }

    queries_label_free = {q: (questions_map[q], None) for q in all_ids}

    load = lambda p: pickle.loads((ROOT / p).read_bytes())

    raw_cache = {}
    for n, b in blocks.items():
        c = load_retrieval_cache(ROOT, TAGS[n])
        raw_cache.update({q: raw_union_local(c[q]) for q in b})

    dense_expansion_scores = load("results/dense_expansion/union50_scores.pkl")["scores"]
    dense_ranks = {
        q: sorted(raw_cache[q], key=lambda d: (-dense_expansion_scores[q][d], d))
        for q in all_ids
    }
    expanded_ranks = weighted_rrf_local([raw_cache, dense_ranks], (0.55, 0.45), 10)

    old_jina = load("results/jina_reranker/holdout_scores_finetuned.pkl")["scores"]
    old_dense = load("results/aiteamvn_dense/holdout_scores_512.pkl")["scores"]
    old_e5 = load("results/e5_dense/holdout_scores.pkl")["scores"]
    vi_rerank = load("results/vietnamese_reranker/holdout_scores_512_finetuned.pkl")["scores"]
    expanded_scores = load("results/expanded_rerank/scores.pkl")

    def rank_by(candidates_dict, scores_dict):
        return {
            q: sorted(
                candidates_dict[q], key=lambda d: (-scores_dict.get(q, {}).get(d, -1e9), d)
            )
            for q in candidates_dict
        }

    base_cands = {q: list(old_jina[q]) for q in all_ids}
    views_cands = {
        q: list(dict.fromkeys(base_cands[q] + expanded_ranks[q][:20])) for q in all_ids
    }

    views_rebuilt = {
        "base": base_cands,
        "expanded": {q: expanded_ranks[q][:20] for q in all_ids},
        "raw": {q: raw_cache[q][:20] for q in all_ids},
        "jina": rank_by(views_cands, expanded_scores["jina"]),
        "dense": rank_by(views_cands, expanded_scores["dense"]),
        "vi": rank_by(base_cands, vi_rerank),
        "e5": rank_by(base_cands, old_e5),
        "old_jina": rank_by(base_cands, old_jina),
        "old_dense": rank_by(base_cands, old_dense),
    }

    dense_saved = load("results/corpus_index/holdout_dense_rank_cap32.pkl")
    corpus_rank, corpus_score = dense_saved["ranking"], dense_saved["scores"]
    extended_scores_cap = load("results/corpus_index/holdout_extended_scores_cap32.pkl")

    extended = {
        q: list(dict.fromkeys(list(views_cands[q]) + corpus_rank[q][:20])) for q in all_ids
    }
    local_views = dict(views_rebuilt)
    for name, table in (
        ("jina", extended_scores_cap["jina"]),
        ("dense", extended_scores_cap["dense"]),
    ):
        local_views[name] = {
            q: sorted(extended[q], key=lambda d: (-table[q].get(d, -1e9), d))
            for q in all_ids
        }
    local_views["corpus"] = {
        q: [d for d in corpus_rank[q] if d in set(extended[q])] for q in all_ids
    }

    base_scores = {
        "jina": extended_scores_cap["jina"],
        "dense": extended_scores_cap["dense"],
        "expansion": dense_expansion_scores,
        "e5": old_e5,
        "corpus": {
            q: {d: corpus_score[q].get(d, -1.0) for d in extended[q]} for q in all_ids
        },
    }

    def load_aligned(rel_path: str, floor=None):
        raw = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in raw for v in raw[q].values())
        return {q: {d: raw.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    vnlegal_cv = (
        load_pkl("results/embedding_finetunc/vnlegal_lal_cv_scores.pkl")
        if (ROOT / "results/embedding_finetunc/vnlegal_lal_cv_scores.pkl").exists()
        else load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    )
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
    """Strictly loads gold labels for evaluation stage AFTER scoring and parity verification."""
    reveal_time_utc = datetime.now(timezone.utc).isoformat()
    raw = json.loads(CAL_GOLD_PATH.read_text(encoding="utf-8"))
    gold = {str(qid): {str(d) for d in raw[str(qid)]["answer"]} for qid in all_ids}
    return gold, reveal_time_utc


def compute_d1_lobo(
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    extended: Dict[str, List[str]],
    local_views: Dict[str, Dict[str, List[str]]],
    full_channels_cv: Dict[str, Dict[str, Dict[str, float]]],
    type_rows: Dict[str, np.ndarray],
    cite_rows: Dict[str, np.ndarray],
    gold: Dict[str, Set[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray], int]:
    """Compute exact D1 LOBO rankings, decision scores, and feature dimension across 4 blocks."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from tune_expanded_fusion_selection import ltr_features

    d1_rankings: Dict[str, List[str]] = {}
    d1_scores: Dict[str, np.ndarray] = {}
    dim = 0

    for held in sorted(blocks.keys()):
        train_ids = sum((blocks[n] for n in blocks if n != held), [])
        test_ids = blocks[held]
        eval_ids = train_ids + test_ids

        eval_rows, eval_groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, full_channels_cv
        )
        for q in eval_rows:
            eval_rows[q] = np.hstack([eval_rows[q], type_rows[q], cite_rows[q]])

        dim = eval_rows[eval_ids[0]].shape[1]

        X_train = np.vstack([eval_rows[q] for q in train_ids])
        y_train = np.array(
            [
                1 if eval_groups[q][i] in gold[q] else 0
                for q in train_ids
                for i in range(len(eval_groups[q]))
            ]
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED,
        )
        model.fit(scaler.transform(X_train), y_train)

        for q in test_ids:
            X_test = scaler.transform(eval_rows[q])
            dec_scores = model.decision_function(X_test)
            order = sorted(
                range(len(dec_scores)), key=lambda i: dec_scores[i], reverse=True
            )
            ranking = [eval_groups[q][i] for i in order]
            d1_rankings[q] = ranking
            d1_scores[q] = dec_scores

    return d1_rankings, d1_scores, dim


def verify_d1_parity(
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    d1_rankings: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
) -> Dict[str, Any]:
    recalls = [
        len(set(d1_rankings[q][:5]) & gold[q]) / len(gold[q]) for q in all_ids
    ]
    mean_r5 = float(np.mean(recalls))

    block_recalls = {}
    for b_name, b_qids in blocks.items():
        b_r5 = float(
            np.mean(
                [
                    len(set(d1_rankings[q][:5]) & gold[q]) / len(gold[q])
                    for q in b_qids
                ]
            )
        )
        block_recalls[b_name] = b_r5

    overall_parity = abs(mean_r5 - EXPECTED_D1_R5) < 1e-9
    block_parity = all(
        abs(block_recalls[b] - EXPECTED_BLOCK_RECALLS[b]) < 1e-9 for b in blocks
    )
    parity_pass = overall_parity and block_parity

    parity_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_sparse_mutual_top5_v1.d1_parity.v1",
        "experiment_id": EXPERIMENT_ID,
        "mean_recall_at_5": mean_r5,
        "expected_recall_at_5": EXPECTED_D1_R5,
        "block_recalls": block_recalls,
        "expected_block_recalls": EXPECTED_BLOCK_RECALLS,
        "overall_parity": overall_parity,
        "block_parity": block_parity,
        "parity_pass": parity_pass,
        "views": D1_VIEWS,
        "dimension": EXPECTED_D1_DIM,
    }
    return parity_doc


def write_json_artifact(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
