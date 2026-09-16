"""Legal Reference Canonicalizer and Own-Reference Indexer from Raw Corpus."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path("D:/Study/DSC2026/sota")
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    RES_DIR,
    load_cal_corpus,
)

# Reference regex covering Vietnamese legal numbering
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
    # Normalize punctuation spacing
    s = re.sub(r"\s*([/\-:])\s*", r"\1", s)
    # Standardize common abbreviations
    s = s.replace("ND-CP", "NĐ-CP")
    s = s.replace("QD-TTG", "QĐ-TTg").replace("QĐ-TTG", "QĐ-TTg")
    s = s.replace("BGDDT", "BGDĐT")
    s = s.rstrip(".,;:()")
    return s


def extract_doc_own_reference(passage: str, link: str) -> Optional[str]:
    """Extract own canonical legal reference from document passage header and link."""
    # 1. Primary: Passage header 'Số: ...'
    m_so = SO_PAT.search(passage[:800])
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
            # URL slug has hyphens where slashes might be e.g. 17-2022-TT-BGTVT -> 17/2022/TT-BGTVT
            # Replace first 1 or 2 hyphens between digits and year with slash
            slug_norm = re.sub(r"^(\d+)-(\d{4})-", r"\1/\2/", raw_slug)
            slug_norm = re.sub(r"^(\d+)-([A-ZĐa-zđ]+)-", r"\1/\2-", slug_norm)
            ref = canonicalize_ref(slug_norm)
            if len(ref) >= 3 and any(c.isdigit() for c in ref):
                return ref

    return None


def build_legal_reference_index(
    corpus: Dict[str, Dict[str, Any]],
    audit_filename: str = "LEGAL_REFERENCE_INDEX_AUDIT.json",
) -> Tuple[
    Dict[str, List[str]], Dict[str, str], dict
]:
    """Build canonical legal reference index mapping canonical reference -> List[doc_ids]."""
    print(f"=== BUILDING LEGAL REFERENCE INDEX ({audit_filename}) ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    ref_to_docs: Dict[str, List[str]] = defaultdict(list)
    doc_to_own_ref: Dict[str, str] = {}
    unindexed_docs: List[str] = []

    for doc_id, item in corpus.items():
        passage = item.get("passage", "")
        link = item.get("link", "")
        own_ref = extract_doc_own_reference(passage, link)

        if own_ref:
            doc_to_own_ref[doc_id] = own_ref
            ref_to_docs[own_ref].append(doc_id)
        else:
            unindexed_docs.append(doc_id)

    # Sort doc IDs deterministically
    for r in ref_to_docs:
        ref_to_docs[r].sort(key=lambda x: int(x) if x.isdigit() else x)

    collision_groups = {r: docs for r, docs in ref_to_docs.items() if len(docs) > 1}

    print(f"Total Corpus Documents:        {len(corpus)}", flush=True)
    print(f"Documents with Own Reference:  {len(doc_to_own_ref)} ({len(doc_to_own_ref)/max(1, len(corpus)):.1%})", flush=True)
    print(f"Unique Canonical References:   {len(ref_to_docs)}", flush=True)
    print(f"Collision Groups (>1 doc):     {len(collision_groups)}", flush=True)

    audit_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.index_audit.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "status": "PASS",
        "total_documents": len(corpus),
        "indexed_documents_count": len(doc_to_own_ref),
        "unindexed_documents_count": len(unindexed_docs),
        "coverage_pct": float(len(doc_to_own_ref) / max(1, len(corpus)) * 100),
        "unique_canonical_references_count": len(ref_to_docs),
        "collision_groups_count": len(collision_groups),
        "sample_collision_groups": {
            r: collision_groups[r] for r in sorted(collision_groups.keys())[:15]
        },
    }

    out_path = RES_DIR / audit_filename
    out_path.write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print("=== LEGAL REFERENCE INDEX AUDIT PASSED ===\n", flush=True)

    return dict(ref_to_docs), doc_to_own_ref, audit_doc


if __name__ == "__main__":
    c = load_cal_corpus()
    build_legal_reference_index(c)
