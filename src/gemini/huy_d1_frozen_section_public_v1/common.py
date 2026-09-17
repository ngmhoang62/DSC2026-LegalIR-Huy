"""Common utilities, model loading, and dataset loaders for HUY_D1_FROZEN_SECTION_PUBLIC_V1."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RESULTS_DIR = ROOT / "results/gemini/huy_d1_frozen_section_public_v1"
SRC_DIR = ROOT / "src/gemini/huy_d1_frozen_section_public_v1"

REPO_JINA = ROOT / "models/jina-reranker-v2-base-multilingual"
WEIGHTS_JINA_FT = ROOT / "models/from_drive/jina_finetuned/model.safetensors"

CAL_FROZEN_SECTION_CACHE_PATH = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1/legal_section_ce_cv.pkl"
PUB_FROZEN_SECTION_CACHE_PATH = RESULTS_DIR / "frozen_section_ce_public.pkl"
PUB_FROZEN_SECTION_MANIFEST_PATH = RESULTS_DIR / "FROZEN_SECTION_PUBLIC_MANIFEST.json"

D1_CHAMPION_JSON_PATH = ROOT / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
D1_CHAMPION_ZIP_PATH = ROOT / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.zip"
CAL_QUESTIONS_LABEL_FREE_PATH = SRC_DIR / "CAL_QUESTIONS_LABEL_FREE.json"

from run_burst_expanded_fusion_submission import title_from_link
from tune_citation_graph import build_citation_table, citation_features
from tune_doctype_features import build_type_table, type_features

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_FROZEN_SECTION_R5 = 0.9577777777777777
EXPECTED_BLOCK_RECALLS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}
EXPECTED_FROZEN_BLOCK_RECALLS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9355555555555555,
}

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


def compute_fingerprint(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_candidate_fingerprint(all_ids: List[str], extended: Dict[str, List[str]]) -> str:
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
        item = queries[q]
        text = item[0] if isinstance(item, (list, tuple)) else str(item)
        h.update(f"{q}:{text}\n".encode("utf-8"))
    return h.hexdigest()


def get_source_files_sha256() -> Dict[str, Dict[str, Any]]:
    files = {}
    for p in sorted(list(SRC_DIR.glob("*.py")) + list(SRC_DIR.glob("*.json"))):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        files[p.name] = {
            "path": rel,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    return files


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


def weighted_rrf_local(rankings: List[Dict[str, List[str]]], weights: Tuple[float, float], k: int) -> Dict[str, List[str]]:
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
    Uses strictly CAL_QUESTIONS_LABEL_FREE.json and frozen result caches.
    Strictly zero CAL gold labels are read, materialized, or returned.
    """
    docs = SafeDocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )

    if not CAL_QUESTIONS_LABEL_FREE_PATH.exists():
        raise FileNotFoundError(f"Missing label-free questions artifact: {CAL_QUESTIONS_LABEL_FREE_PATH}")

    raw_q = json.loads(CAL_QUESTIONS_LABEL_FREE_PATH.read_text(encoding="utf-8"))
    questions_map = {str(k): (v["question"] if isinstance(v, dict) else str(v)) for k, v in raw_q.items()}
    all_ids = list(questions_map.keys())
    if len(all_ids) != 600:
        raise ValueError(f"Expected exactly 600 queries in label-free questions, got {len(all_ids)}")

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

    # queries_label_free strictly contains only question text, and gold set is None
    queries_label_free = {q: (questions_map[q], None) for q in all_ids}

    # Self-contained reconstruction of views and candidates from frozen caches
    load = lambda p: pickle.loads((ROOT / p).read_bytes())

    raw_cache = {}
    for n, b in blocks.items():
        c = load_retrieval_cache(ROOT, TAGS[n])
        raw_cache.update({q: raw_union_local(c[q]) for q in b})

    dense_expansion_scores = load("results/dense_expansion/union50_scores.pkl")["scores"]
    dense_ranks = {q: sorted(raw_cache[q], key=lambda d: (-dense_expansion_scores[q][d], d)) for q in all_ids}
    expanded_ranks = weighted_rrf_local([raw_cache, dense_ranks], (0.55, 0.45), 10)

    old_jina = load("results/jina_reranker/holdout_scores_finetuned.pkl")["scores"]
    old_dense = load("results/aiteamvn_dense/holdout_scores_512.pkl")["scores"]
    old_e5 = load("results/e5_dense/holdout_scores.pkl")["scores"]
    vi_rerank = load("results/vietnamese_reranker/holdout_scores_512_finetuned.pkl")["scores"]
    expanded_scores = load("results/expanded_rerank/scores.pkl")

    def rank_by(candidates_dict, scores_dict):
        return {
            q: sorted(candidates_dict[q], key=lambda d: (-scores_dict.get(q, {}).get(d, -1e9), d))
            for q in candidates_dict
        }

    base_cands = {q: list(old_jina[q]) for q in all_ids}
    views_cands = {q: list(dict.fromkeys(base_cands[q] + expanded_ranks[q][:20])) for q in all_ids}

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

    extended = {q: list(dict.fromkeys(list(views_cands[q]) + corpus_rank[q][:20])) for q in all_ids}
    local_views = dict(views_rebuilt)
    for name, table in (("jina", extended_scores_cap["jina"]), ("dense", extended_scores_cap["dense"])):
        local_views[name] = {
            q: sorted(extended[q], key=lambda d: (-table[q].get(d, -1e9), d))
            for q in all_ids
        }
    local_views["corpus"] = {
        q: [d for d in corpus_rank[q] if d in set(extended[q])]
        for q in all_ids
    }

    base_scores = {
        "jina": extended_scores_cap["jina"],
        "dense": extended_scores_cap["dense"],
        "expansion": dense_expansion_scores,
        "e5": old_e5,
        "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in extended[q]} for q in all_ids},
    }

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
    """Strictly loads gold labels for evaluation stage AFTER scoring and parity verification."""
    reveal_time_utc = datetime.now(timezone.utc).isoformat()
    path = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "train.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    gold = {
        str(qid): {str(d) for d in raw[str(qid)]["answer"]}
        for qid in all_ids
    }
    return gold, reveal_time_utc
