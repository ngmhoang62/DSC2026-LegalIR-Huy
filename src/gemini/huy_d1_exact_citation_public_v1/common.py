"""Common paths, utilities, provenance, and frozen citation rules for HUY_D1_EXACT_CITATION_PUBLIC_V1."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import random
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
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

SRC_DIR = ROOT / "src/gemini/huy_d1_exact_citation_public_v1"
RESULTS_DIR = ROOT / "results/gemini/huy_d1_exact_citation_public_v1"

D1_CHAMPION_JSON_PATH = ROOT / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
D1_CHAMPION_ZIP_PATH = ROOT / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.zip"

PUBLIC_DATASET_DIR = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
PUBLIC_CONTEXTS_DIR = PUBLIC_DATASET_DIR / "selected-contexts"
CAL_QUESTIONS_LABEL_FREE_PATH = ROOT / "src/gemini/huy_d1_frozen_section_public_v1/CAL_QUESTIONS_LABEL_FREE.json"
CAL_GOLD_PATH = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"

FROZEN_PRIOR_SOURCE_SHA = "08637fb6438e18ff0e3ef2da5b7f6af506d4afb1"
FROZEN_CAL_EVIDENCE = {
    "unique_exact_anchors": 3,
    "anchor_gold": 3,
    "anchor_precision": 1.0,
    "interventions": 1,
    "beneficial": 1,
    "harmful": 0,
}
FROZEN_STRICTV2_EVIDENCE = {
    "unique_exact_anchors": 26,
    "gold_anchors": 24,
    "anchor_precision": 0.9230769230769231,
}

from run_burst_expanded_fusion_submission import title_from_link
from tune_citation_graph import build_citation_table, citation_features
from tune_doctype_features import build_type_table, type_features

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_PUBLIC_QUERIES = 1000
SEED = 2026

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
    Strictly zero CAL gold labels are read, materialized, or returned.
    """
    docs = SafeDocumentStore(sorted(PUBLIC_CONTEXTS_DIR.glob("context_*.json")))

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

    queries_label_free = {q: (questions_map[q], None) for q in all_ids}

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
    """Loads CAL gold labels strictly for training the production D1 learner on full CAL600."""
    reveal_time_utc = datetime.now(timezone.utc).isoformat()
    raw = json.loads(CAL_GOLD_PATH.read_text(encoding="utf-8"))
    gold = {
        str(qid): {str(d) for d in raw[str(qid)]["answer"]}
        for qid in all_ids
    }
    return gold, reveal_time_utc


# --- Frozen Legal Reference Canonicalization & Extraction ---
# Semantics frozen from HUY_D1_SELECTIVE_REPAIR_V1 (commit 08637fb6438e18ff0e3ef2da5b7f6af506d4afb1)

REF_REGEX = re.compile(
    r'(?:'
    r'\b\d+/(?:\d{4}/)?(?:[A-ZĐa-zđ\d]+[-/])*[A-ZĐa-zđ\d]+\b'
    r'|\b\d+-(?:CT|NQ|QĐ|QD|TT)/[A-ZĐa-zđ\d]+\b'
    r'|\b(?:QCVN|TCVN)\s*[\d\.\-]+(?::\d{4}|/\d{4})?(?:/[A-ZĐa-zđ\d]+)?\b'
    r')',
    re.UNICODE,
)

SO_PAT = re.compile(
    r'S[ốoỐ]\s*[:\.]\s*([0-9]+[0-9a-zA-ZĐđ\.\-_/]+(?:/[0-9a-zA-ZĐđ\.\-_/]+)*)',
    re.UNICODE | re.IGNORECASE,
)

LINK_PAT = re.compile(
    r'/(?:Thong-tu|Nghi-dinh|Quyet-dinh|Luat|Nghi-quyet|Chi-thi|Cong-van|Thong-bao)-([0-9]+(?:-[0-9]+)?-[0-9a-zA-ZĐđ\-]+)-',
    re.UNICODE | re.IGNORECASE,
)


def canonicalize_ref(ref_str: str) -> str:
    """Deterministic conservative canonicalization of legal references."""
    s = unicodedata.normalize("NFC", (ref_str or "").strip()).upper()
    s = re.sub(r"\s*([/\-:])\s*", r"\1", s)
    s = s.replace("ND-CP", "NĐ-CP")
    s = s.replace("QD-TTG", "QĐ-TTg").replace("QĐ-TTG", "QĐ-TTg")
    s = s.replace("BGDDT", "BGDĐT")
    s = s.rstrip(".,;:()")
    return s


def extract_doc_own_reference(passage: str, link: str = "") -> Optional[str]:
    """Extract own canonical legal reference from document passage header and link."""
    # 1. Primary: Passage header 'Số: ...'
    m_so = SO_PAT.search((passage or "")[:800])
    if m_so:
        raw_val = m_so.group(1).split()[0]
        ref = canonicalize_ref(raw_val)
        if len(ref) >= 3 and any(c.isdigit() for c in ref):
            return ref

    # 2. Secondary: URL slug in link
    if link:
        m_link = LINK_PAT.search(link)
        if m_link:
            raw_slug = m_link.group(1)
            slug_norm = re.sub(r"^(\d+)-(\d{4})-", r"\1/\2/", raw_slug)
            slug_norm = re.sub(r"^(\d+)-([A-ZĐa-zđ]+)-", r"\1/\2-", slug_norm)
            ref = canonicalize_ref(slug_norm)
            if len(ref) >= 3 and any(c.isdigit() for c in ref):
                return ref

    return None


def extract_query_references(query_text: str) -> List[str]:
    """Extract canonical legal references from raw query text."""
    raw_matches = REF_REGEX.findall(query_text or "")
    refs = []
    for m in raw_matches:
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", m):
            continue
        c_ref = canonicalize_ref(m)
        if len(c_ref) >= 3 and any(c.isdigit() for c in c_ref):
            if c_ref not in refs:
                refs.append(c_ref)
    return refs


def resolve_tier1_anchor(
    query_text: str,
    ref_to_docs: Dict[str, List[str]],
) -> Tuple[Optional[str], str, List[str]]:
    """Resolve query text to at most ONE exact anchor document."""
    refs = extract_query_references(query_text)
    if not refs:
        return None, "NO_REFERENCES", []

    matched_docs_per_ref = {}
    has_ambiguous_doc_match = False

    for r in refs:
        matching_docs = ref_to_docs.get(r, [])
        if len(matching_docs) == 1:
            matched_docs_per_ref[r] = matching_docs[0]
        elif len(matching_docs) > 1:
            has_ambiguous_doc_match = True

    if has_ambiguous_doc_match:
        return None, "AMBIGUOUS_DOC_MATCH", refs

    distinct_matched_docs = set(matched_docs_per_ref.values())
    if len(distinct_matched_docs) == 0:
        return None, "UNMATCHED_REFERENCES", refs
    elif len(distinct_matched_docs) > 1:
        return None, "MULTIPLE_ELIGIBLE_DOCS", refs
    else:
        anchor_doc = next(iter(distinct_matched_docs))
        return anchor_doc, "UNIQUE_EXACT_MATCH", refs


def apply_tier1_repair(
    d1_top5: List[str],
    anchor_doc: Optional[str],
) -> Tuple[List[str], str, Optional[str], Optional[str]]:
    """Apply protected injection of unique exact citation anchor into D1 Top-5."""
    top5 = list(d1_top5[:5])
    if anchor_doc is None:
        return top5, "ABSTAIN", None, None

    if anchor_doc in top5:
        return top5, "KEEP_ALREADY_IN_TOP5", None, None

    evicted_doc = top5[4]
    new_top5 = top5[:4] + [anchor_doc]
    return new_top5, "EXACT_CITATION_INJECTION", anchor_doc, evicted_doc
