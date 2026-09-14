"""Build Label-Free Legal Document Relation Graph from Canonical Corpus.

Strictly label-free: Processes only canonical 8,507-parent documents from
V2_CONTEXTS.jsonl and selected-contexts metadata. Imports/reads NO gold labels.
Extracts normalized document identities, detects explicit relations (REPLACES,
REPEALS, AMENDS, GUIDES, CITES) with strict direction tracking, and outputs
RELATION_GRAPH.jsonl and RELATION_GRAPH_AUDIT.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Resolve repository root
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
RESULTS_DIR = REPO_ROOT / "results/gemini/rabr_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CONTEXTS = REPO_ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
SELECTED_CONTEXTS_DIR = REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_ref_key(ref: str) -> str:
    if not ref:
        return ""
    ref = strip_accents(ref).upper()
    ref = re.sub(r"[\s]+", " ", ref).strip()
    return ref


DOCTYPE_MAP = [
    ("THONG_TU_LIEN_TICH", re.compile(r"\bTHÔNG TƯ LIÊN TỊCH\b", re.I)),
    ("THONG_TU", re.compile(r"\bTHÔNG TƯ\b", re.I)),
    ("NGHI_DINH", re.compile(r"\bNGHỊ ĐỊNH\b", re.I)),
    ("QUYET_DINH", re.compile(r"\bQUYẾT ĐỊNH\b", re.I)),
    ("NGHI_QUYET", re.compile(r"\bNGHỊ QUYẾT\b", re.I)),
    ("BO_LUAT", re.compile(r"\bBỘ LUẬT\b", re.I)),
    ("LUAT", re.compile(r"\bLUẬT\b", re.I)),
    ("PHAP_LENH", re.compile(r"\bPHÁP LỆNH\b", re.I)),
    ("CHI_THI", re.compile(r"\bCHỈ THỊ\b", re.I)),
    ("QCVN", re.compile(r"\bQCVN\b", re.I)),
    ("TCVN", re.compile(r"\bTCVN\b", re.I)),
]

NUM_RE = re.compile(r"Số:?\s*([0-9]+[0-9\.\/]*\/(?:[0-9]{4}\/)?[A-ZĐƯƠa-zđươ0-9\-]+)", re.I)
NUM_HYPHEN_RE = re.compile(r"Số:?\s*([0-9]+(?:\-[0-9A-ZĐƯƠa-zđươ]+)+\/(?:[0-9]{4}\/)?[A-ZĐƯƠa-zđươ0-9\-]+)", re.I)
STD_RE = re.compile(r"\b(QCVN|TCVN)\s*([0-9]+(?:\-[0-9]+)?(?:\:[0-9]{4})?(?:\/[A-ZĐƯƠa-zđươ0-9\-]+)?)", re.I)


def extract_document_identity(doc_id: str, passage: str, contexts_dir: Path) -> Dict[str, Any]:
    norm_passage = re.sub(r"\s+", " ", unicodedata.normalize("NFC", passage))
    header = norm_passage[:500]

    # Document Type
    dtype = None
    for dt_name, pat in DOCTYPE_MAP:
        if pat.search(header[:300]):
            dtype = dt_name
            break

    # Number
    m_num = NUM_RE.search(header) or NUM_HYPHEN_RE.search(header)
    m_std = STD_RE.search(header)
    ref = None
    if m_num:
        ref = m_num.group(1).strip()
    elif m_std:
        ref = f"{m_std.group(1)} {m_std.group(2)}".strip()
    else:
        # Fallback to context json
        cp = contexts_dir / f"context_{doc_id}.json"
        if cp.exists():
            try:
                crow = json.loads(cp.read_text(encoding="utf-8"))
                cname = crow.get("name") or ""
                m_c = re.search(r"(\d+[\-_]\d{4}[\-_][A-Z0-9\-]+)", cname)
                if m_c:
                    ref = m_c.group(1).replace("_", "/").replace("-", "/")
                elif "TCVN" in cname.upper():
                    m_tcvn = re.search(r"(TCVN[\-_0-9]+(?:\-[0-9]+)?)", cname, re.I)
                    if m_tcvn:
                        ref = m_tcvn.group(1).replace("-", " ")
            except Exception:
                pass

    # Year
    year = None
    if ref:
        m_yr = re.search(r"\b(19\d{2}|20\d{2})\b", ref)
        if m_yr:
            year = int(m_yr.group(1))
    if not year:
        m_dyr = re.search(r"ngày\s+\d+\s+tháng\s+\d+\s+năm\s+(\d{4})", header, re.I)
        if m_dyr:
            year = int(m_dyr.group(1))

    # Issuing suffix
    suffix = None
    if ref and "/" in ref:
        parts = ref.split("/")
        suffix = parts[-1]

    # Normalized primary reference key
    norm_key = normalize_ref_key(ref) if ref else None

    # Title / preamble
    title_line = header[:200]

    return {
        "doc_id": doc_id,
        "doc_type": dtype,
        "official_number": ref,
        "year": year,
        "issuing_suffix": suffix,
        "primary_ref_key": norm_key,
        "title": title_line,
    }


# Relation Regexes
CITED_REF_CORE = r"([0-9]+[0-9\.\/]*\/(?:[0-9]{4}\/)?[A-ZĐƯƠa-zđươ0-9\-]+)"
STD_CITED_CORE = r"((?:QCVN|TCVN)\s*[0-9]+(?:\-[0-9]+)?(?:\:[0-9]{4})?(?:\/[A-ZĐƯƠa-zđươ0-9\-]+)?)"

# 1. REPLACE:
# A: Current replaces cited
REPLACE_FWD_PATTERNS = [
    re.compile(rf"(?:thay thế|thay cho)\s+(?:toàn bộ\s+)?(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị|văn bản)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
    re.compile(rf"(?:thay thế|thay cho)\s+(?:toàn bộ\s+)?{STD_CITED_CORE}", re.I),
    re.compile(rf"về việc thay thế\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
]
# B: Cited replaces current
REPLACE_REV_PATTERNS = [
    re.compile(rf"(?:được|bị)\s+thay thế\s+(?:bởi|bằng)\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
    re.compile(rf"(?:được|bị)\s+thay thế\s+(?:bởi|bằng)\s+{STD_CITED_CORE}", re.I),
]

# 2. REPEAL:
# A: Current repeals cited
REPEAL_FWD_PATTERNS = [
    re.compile(rf"bãi bỏ\s+(?:toàn bộ\s+)?(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị|văn bản)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
    re.compile(rf"bãi bỏ\s+(?:toàn bộ\s+)?{STD_CITED_CORE}", re.I),
    re.compile(rf"(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị)\s*(?:số\s+)?{CITED_REF_CORE}\s+(?:hết|chấm dứt)\s+hiệu lực", re.I),
    re.compile(rf"{STD_CITED_CORE}\s+(?:hết|chấm dứt)\s+hiệu lực", re.I),
    re.compile(rf"(?:chấm dứt|hết)\s+hiệu lực\s+(?:thi hành\s+)?(?:đối với\s+)?(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
]
# B: Cited repeals current
REPEAL_REV_PATTERNS = [
    re.compile(rf"(?:được|bị)\s+bãi bỏ\s+(?:bởi|theo|tại)\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
]

# 3. AMEND:
# A: Current amends cited
AMEND_FWD_PATTERNS = [
    re.compile(rf"(?:sửa đổi(?:,\s*bổ sung)?|bổ sung)(?:\s+(?:một số|các)\s+điều của|\s+khoản.*?của)?\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị)\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
    re.compile(rf"(?:sửa đổi(?:,\s*bổ sung)?|bổ sung)\s+(?:một số|các)?\s+{STD_CITED_CORE}", re.I),
]
# B: Cited amends current
AMEND_REV_PATTERNS = [
    re.compile(rf"(?:được|bị)\s+sửa đổi(?:,\s*bổ sung)?\s+(?:bởi|theo|tại)\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Chỉ thị)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
]

# 4. GUIDE / IMPLEMENT:
# A: Current guides cited
GUIDE_FWD_PATTERNS = [
    re.compile(rf"(?:hướng dẫn thi hành|quy định chi tiết|hướng dẫn thực hiện)(?:\s+(?:một số|các)\s+điều của)?\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh)\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
]
# B: Cited guides current
GUIDE_REV_PATTERNS = [
    re.compile(rf"(?:được|bị)\s+(?:hướng dẫn|quy định chi tiết)\s+(?:bởi|tại|theo)\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
]

# 5. CITATION:
CITE_FWD_PATTERNS = [
    re.compile(rf"Căn cứ\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh|Hiến pháp)?\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
    re.compile(rf"Căn cứ\s+{STD_CITED_CORE}", re.I),
    re.compile(rf"(?:theo quy định tại|áp dụng)\s+(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|Pháp lệnh)\s*(?:số\s+)?{CITED_REF_CORE}", re.I),
]


def extract_relations_for_doc(
    doc_id: str,
    passage: str,
    unambiguous_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    edges = []
    norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", passage))
    
    # Divide into sections: header/title (first 500 chars), preamble (first 3000 chars), closing (last 3000 chars)
    preamble_end = re.search(r"Điều\s+1\b", norm[:3000], re.I)
    preamble = norm[:preamble_end.start()] if preamble_end else norm[:2000]
    closing = norm[-3000:] if len(norm) > 3000 else norm

    def match_and_add(patterns, text, rel_type, is_forward, confidence, loc):
        for p in patterns:
            for m in p.finditer(text):
                raw_ref = m.group(1)
                norm_k = normalize_ref_key(raw_ref)
                if norm_k in unambiguous_lookup:
                    target_doc = unambiguous_lookup[norm_k]
                    if target_doc != doc_id:
                        src = doc_id if is_forward else target_doc
                        dst = target_doc if is_forward else doc_id
                        snippet_start = max(0, m.start() - 30)
                        snippet_end = min(len(text), m.end() + 30)
                        edges.append({
                            "src_doc": src,
                            "dst_doc": dst,
                            "relation_type": rel_type,
                            "confidence": confidence,
                            "source_text": text[snippet_start:snippet_end].strip(),
                            "source_location": loc,
                            "normalized_cited_key": norm_k,
                        })

    # REPLACE (scan full text, focus on closing and title)
    match_and_add(REPLACE_FWD_PATTERNS, closing, "REPLACES", is_forward=True, confidence=1.0, loc="closing")
    match_and_add(REPLACE_FWD_PATTERNS, norm[:1000], "REPLACES", is_forward=True, confidence=1.0, loc="title")
    match_and_add(REPLACE_REV_PATTERNS, norm[:3000], "REPLACES", is_forward=False, confidence=1.0, loc="preamble")

    # REPEAL (scan full text, focus on closing and body)
    match_and_add(REPEAL_FWD_PATTERNS, closing, "REPEALS", is_forward=True, confidence=1.0, loc="closing")
    match_and_add(REPEAL_FWD_PATTERNS, norm[:1000], "REPEALS", is_forward=True, confidence=1.0, loc="title")
    match_and_add(REPEAL_REV_PATTERNS, norm[:3000], "REPEALS", is_forward=False, confidence=1.0, loc="preamble")

    # AMEND (scan title and preamble)
    match_and_add(AMEND_FWD_PATTERNS, norm[:1500], "AMENDS", is_forward=True, confidence=0.95, loc="header")
    match_and_add(AMEND_REV_PATTERNS, norm[:3000], "AMENDS", is_forward=False, confidence=0.95, loc="preamble")

    # GUIDE (scan preamble and title)
    match_and_add(GUIDE_FWD_PATTERNS, preamble, "GUIDES", is_forward=True, confidence=0.90, loc="preamble")
    match_and_add(GUIDE_REV_PATTERNS, preamble, "GUIDES", is_forward=False, confidence=0.90, loc="preamble")

    # CITES (scan preamble for citations)
    match_and_add(CITE_FWD_PATTERNS, preamble, "CITES", is_forward=True, confidence=0.80, loc="preamble")

    return edges


def main():
    started = time.perf_counter()
    print("=== Step 4: Building Label-Free Legal Document Relation Graph ===", flush=True)

    # 1. Parse all canonical 8,507 parent documents
    docs_text = {}
    doc_meta = {}
    ref_to_docs = defaultdict(set)

    with CANONICAL_CONTEXTS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            passage = row["passage"]
            docs_text[doc_id] = passage
            meta = extract_document_identity(doc_id, passage, SELECTED_CONTEXTS_DIR)
            doc_meta[doc_id] = meta
            if meta["primary_ref_key"]:
                ref_to_docs[meta["primary_ref_key"]].add(doc_id)

    total_canonical = len(docs_text)
    recognized_docs = sum(1 for m in doc_meta.values() if m["primary_ref_key"])
    print(f"Loaded {total_canonical} canonical documents. Recognized primary ref for {recognized_docs} docs.", flush=True)

    # Disambiguation: Keep only keys mapping to exactly ONE canonical parent doc
    unambiguous_lookup = {}
    ambiguous_keys = {}
    for k, docs in ref_to_docs.items():
        if len(docs) == 1:
            unambiguous_lookup[k] = next(iter(docs))
        else:
            ambiguous_keys[k] = sorted(list(docs))

    print(f"Unambiguous reference keys: {len(unambiguous_lookup)}, Ambiguous keys: {len(ambiguous_keys)} (skipped)", flush=True)

    # 2. Extract relation edges
    raw_edges = []
    for doc_id, passage in docs_text.items():
        doc_edges = extract_relations_for_doc(doc_id, passage, unambiguous_lookup)
        raw_edges.extend(doc_edges)

    print(f"Extracted {len(raw_edges)} raw relation instances.", flush=True)

    # Deduplicate edges: When multiple relations exist between src and dst, prioritize:
    # REPLACES (1) > REPEALS (2) > AMENDS (3) > GUIDES (4) > CITES (5)
    PRIORITY = {
        "REPLACES": 1,
        "REPEALS": 2,
        "AMENDS": 3,
        "GUIDES": 4,
        "CITES": 5,
    }

    unique_edge_map = {}
    for edge in raw_edges:
        key = (edge["src_doc"], edge["dst_doc"])
        rel = edge["relation_type"]
        if key not in unique_edge_map:
            unique_edge_map[key] = edge
        else:
            existing_rel = unique_edge_map[key]["relation_type"]
            if PRIORITY.get(rel, 99) < PRIORITY.get(existing_rel, 99):
                unique_edge_map[key] = edge

    deduped_edges = sorted(unique_edge_map.values(), key=lambda e: (e["relation_type"], e["src_doc"], e["dst_doc"]))
    print(f"Deduplicated to {len(deduped_edges)} unique directed relation edges.", flush=True)

    # Edge counts by type
    counts_by_type = Counter(e["relation_type"] for e in deduped_edges)
    unique_nodes = set()
    for e in deduped_edges:
        unique_nodes.add(e["src_doc"])
        unique_nodes.add(e["dst_doc"])

    print(f"Unique nodes touched: {len(unique_nodes)}")
    print("Counts by edge type:", json.dumps(counts_by_type, indent=2))

    # Write RELATION_GRAPH.jsonl
    graph_path = RESULTS_DIR / "RELATION_GRAPH.jsonl"
    with graph_path.open("w", encoding="utf-8") as f:
        for e in deduped_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote {graph_path}", flush=True)

    # Collect 20 deterministic samples per major relation type
    samples_by_type = defaultdict(list)
    for e in deduped_edges:
        rel = e["relation_type"]
        if len(samples_by_type[rel]) < 20:
            samples_by_type[rel].append({
                "src_doc": e["src_doc"],
                "src_ref": doc_meta[e["src_doc"]]["official_number"],
                "dst_doc": e["dst_doc"],
                "dst_ref": doc_meta[e["dst_doc"]]["official_number"],
                "relation_type": e["relation_type"],
                "confidence": e["confidence"],
                "source_text": e["source_text"],
                "source_location": e["source_location"],
            })

    # Save document metadata cache for fast retrieval by RABR models
    meta_cache_path = RESULTS_DIR / "cache/DOCUMENT_METADATA.json"
    meta_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_cache_path.open("w", encoding="utf-8") as f:
        json.dump(doc_meta, f, ensure_ascii=False, indent=2)

    # Write RELATION_GRAPH_AUDIT.json
    audit_report = {
        "schema_version": "dsc2026.gemini.rabr_v1.relation_graph_audit.v1",
        "canonical_document_count": total_canonical,
        "count_documents_with_recognized_own_reference": recognized_docs,
        "number_of_unique_reference_keys": len(unambiguous_lookup),
        "ambiguous_keys_count": len(ambiguous_keys),
        "ambiguous_keys_sample": {k: ambiguous_keys[k] for k in list(ambiguous_keys.keys())[:10]},
        "total_edges": len(deduped_edges),
        "counts_by_edge_type": dict(counts_by_type),
        "unique_nodes_touched": len(unique_nodes),
        "sample_edges_per_major_relation_type": {rel: samples_by_type[rel] for rel in sorted(samples_by_type)},
        "zero_manual_or_gold_derived_mappings": True,
        "runtime_seconds": time.perf_counter() - started,
    }

    audit_path = RESULTS_DIR / "RELATION_GRAPH_AUDIT.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {audit_path}", flush=True)


if __name__ == "__main__":
    main()
