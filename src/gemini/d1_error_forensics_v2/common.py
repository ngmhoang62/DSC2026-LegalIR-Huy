"""Common utilities, constants, and data loaders for D1 Error Forensics V2.

Strictly observational and raw data derived. Zero LLM interpretations.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC_DIR = ROOT / "src/gemini/d1_error_forensics_v2"
RESULTS_DIR = ROOT / "results/gemini/d1_error_forensics_v2"

# Authoritative input paths
TRAIN_JSON_PATH = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
CONTEXTS_DIR = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
FORENSICS_JSON_PATH = ROOT / "results/gemini/d1_error_forensics/D1_ERROR_FORENSICS.json"
INTEGRITY_JSON_PATH = ROOT / "results/gemini/d1_error_forensics/ERROR_FORENSICS_INTEGRITY.json"
ORACLE_AUDIT_PATH = ROOT / "results/gemini/huy_d1_section_residual_selector_v1/ONE_SWAP_ORACLE_AUDIT.json"
SWAP_DIAGNOSTICS_PATH = ROOT / "results/gemini/huy_d1_section_residual_selector_v1/SECTION_SELECTOR_SWAP_DIAGNOSTICS.json"
LEGAL_SECTION_PKL_PATH = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1/legal_section_ce_cv.pkl"
RECOVERY_CASES_PATH = ROOT / "results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/OUTSIDE_POOL_GOLD_RECOVERY_CASES.json"
V2_SHADOW_PATH = ROOT / "results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/V2_SHADOW_EXPANSION_RESULTS.json"

# Old Jina exact cache from HUY_D1_SECTION_RESIDUAL_SELECTOR_V1
OLD_JINA_CACHE_PKL = ROOT / "results/from_drive/jina_ft_cv.pkl"
EXPECTED_OLD_JINA_CACHE_SHA256 = "666296dc0bffdf7366c2b5834aed1612bb34ffc45bbd2696b227338367282d00"

# D1 Candidate pool source
HOLDOUT_EXTENDED_PKL_PATH = ROOT / "results/corpus_index/holdout_extended_scores_cap32.pkl"

# D1 Champion specification
EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_BLOCK_RECALLS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}
EXPECTED_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_FEATURE_DIM = 48

from run_burst_expanded_fusion_submission import title_from_link
from tune_doctype_features import doc_type
from tune_citation_graph import own_number
from tune_corpus_cap32_fusion import build_training_cap

# In-memory document cache to avoid re-reading JSON files
_DOC_CACHE: Dict[str, Dict[str, Any]] = {}


def get_doc_data(doc_id: str) -> Dict[str, Any]:
    """Retrieve raw document data, title, doctype, and number in O(1) time."""
    if doc_id in _DOC_CACHE:
        return _DOC_CACHE[doc_id]

    path = CONTEXTS_DIR / f"context_{doc_id}.json"
    if not path.exists():
        res = {
            "doc_id": doc_id,
            "title": "",
            "doctype": None,
            "doc_number": None,
            "passage": "",
            "exists_in_corpus": False,
        }
        _DOC_CACHE[doc_id] = res
        return res

    row = json.loads(path.read_text(encoding="utf-8"))
    link = row.get("link") or ""
    title = title_from_link(link) or ""
    passage = row.get("passage") or ""
    d_type = doc_type(passage) if passage else None
    d_num = own_number(passage) if passage else None

    res = {
        "doc_id": doc_id,
        "title": title,
        "doctype": d_type,
        "doc_number": d_num,
        "passage": passage,
        "exists_in_corpus": True,
    }
    _DOC_CACHE[doc_id] = res
    return res


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_source_provenance() -> Tuple[bool, Dict[str, Any]]:
    """Hard-gate source provenance before reading/generating authoritative artifacts."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        origin = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=str(ROOT), text=True
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        ).strip()
        parity = head == origin
        status_clean = len(porcelain) == 0
        src_files_sha256 = {
            "common.py": sha256_file(SRC_DIR / "common.py"),
            "generate_raw_dossier.py": sha256_file(SRC_DIR / "generate_raw_dossier.py"),
        }
        provenance_info = {
            "head_commit": head,
            "origin_main_commit": origin,
            "parity": parity,
            "status_clean": status_clean,
            "porcelain_output": porcelain,
            "source_files_sha256": src_files_sha256,
        }
        passed = parity and status_clean
        return passed, provenance_info
    except Exception as e:
        return False, {
            "head_commit": "ERROR",
            "origin_main_commit": "ERROR",
            "parity": False,
            "status_clean": False,
            "error": str(e),
        }


def compute_pairwise_preferences(
    q_forensics: Dict[str, Any],
    qid: str,
    challenger_id: str,
    defender_id: str,
    sec_scores: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """Compute exact pairwise preference for 11 signals without speculation."""
    ee = q_forensics.get("expert_evidence", {})
    c_ee = ee.get(challenger_id, {})
    d_ee = ee.get(defender_id, {})

    prefs: Dict[str, str] = {}

    # 1. 5 Rank views (lower within_query_rank is better)
    rank_view_keys = ["base", "expanded", "jina", "dense", "corpus"]
    for k in rank_view_keys:
        c_r = c_ee.get("rank_views", {}).get(k, {}).get("within_query_rank")
        d_r = d_ee.get("rank_views", {}).get(k, {}).get("within_query_rank")
        if c_r is not None and d_r is not None:
            if c_r < d_r:
                prefs[k] = "CHALLENGER > DEFENDER"
            elif d_r < c_r:
                prefs[k] = "DEFENDER > CHALLENGER"
            else:
                prefs[k] = "TIE"
        else:
            prefs[k] = "UNAVAILABLE"

    # 2. 5 Score channels (higher raw_score is better)
    score_chan_keys = ["vnlegal_lal", "crossenc", "aiteamvn_ft", "jina_ft", "title_embed"]
    for k in score_chan_keys:
        c_s = c_ee.get("score_channels", {}).get(k, {}).get("raw_score")
        d_s = d_ee.get("score_channels", {}).get(k, {}).get("raw_score")
        if c_s is not None and d_s is not None:
            if c_s > d_s:
                prefs[k] = "CHALLENGER > DEFENDER"
            elif d_s > c_s:
                prefs[k] = "DEFENDER > CHALLENGER"
            else:
                prefs[k] = "TIE"
        else:
            prefs[k] = "UNAVAILABLE"

    # 3. Frozen Section CE (higher is better)
    sec_c = sec_scores.get(qid, {}).get(challenger_id)
    sec_d = sec_scores.get(qid, {}).get(defender_id)
    if sec_c is not None and sec_d is not None:
        if sec_c > sec_d:
            prefs["frozen_section_ce"] = "CHALLENGER > DEFENDER"
        elif sec_d > sec_c:
            prefs["frozen_section_ce"] = "DEFENDER > CHALLENGER"
        else:
            prefs["frozen_section_ce"] = "TIE"
    else:
        prefs["frozen_section_ce"] = "UNAVAILABLE"

    return prefs
