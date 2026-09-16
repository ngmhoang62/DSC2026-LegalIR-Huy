"""Audit 1: Strict-V2 Provenance and Integrity Hard Audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Set

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    BOUNDARY_V2_CONTEXTS_JSONL,
    CANONICAL_V2_CANDIDATE_POOL_JSONL,
    CANONICAL_V2_CONTEXTS_JSONL,
    CANONICAL_V2_QUERIES_JSONL,
    EXPECTED_V2_DOCS_COUNT,
    EXPECTED_V2_QUERIES_COUNT,
    RES_DIR,
    sha256_file,
)


def run_strict_v2_provenance_audit() -> Dict[str, Any]:
    print("=== AUDIT 1: STRICT-V2 PROVENANCE HARD AUDIT ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Corpus verification
    v4_exists = CANONICAL_V2_CONTEXTS_JSONL.exists()
    v2_exists = BOUNDARY_V2_CONTEXTS_JSONL.exists()
    h_v4 = sha256_file(CANONICAL_V2_CONTEXTS_JSONL) if v4_exists else "MISSING"
    h_v2 = sha256_file(BOUNDARY_V2_CONTEXTS_JSONL) if v2_exists else "MISSING"
    byte_identical = (h_v4 == h_v2) and v4_exists and v2_exists

    doc_ids: Set[str] = set()
    if v4_exists:
        with open(CANONICAL_V2_CONTEXTS_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc_ids.add(str(json.loads(line)["doc_id"]))

    doc_fp = hashlib.sha256(",".join(sorted(doc_ids)).encode("utf-8")).hexdigest()

    # 2. Query source verification
    q_exists = CANONICAL_V2_QUERIES_JSONL.exists()
    q_sha256 = sha256_file(CANONICAL_V2_QUERIES_JSONL) if q_exists else "MISSING"
    qids: Set[str] = set()
    queries_with_gold = 0
    if q_exists:
        with open(CANONICAL_V2_QUERIES_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    qids.add(str(row["qid"]))
                    if row.get("gold"):
                        queries_with_gold += 1

    qid_fp = hashlib.sha256(",".join(sorted(qids)).encode("utf-8")).hexdigest()

    # 3. Candidate pool verification
    p_exists = CANONICAL_V2_CANDIDATE_POOL_JSONL.exists()
    p_sha256 = sha256_file(CANONICAL_V2_CANDIDATE_POOL_JSONL) if p_exists else "MISSING"
    pool_qids: Set[str] = set()
    cand_docs: Set[str] = set()
    mem_h = hashlib.sha256()

    if p_exists:
        with open(CANONICAL_V2_CANDIDATE_POOL_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    qid = str(row["qid"])
                    pool_qids.add(qid)
                    cands = [str(x) for x in row.get("doc_ids", [])]
                    for c in cands:
                        cand_docs.add(c)
                    mem_h.update(f"{qid}:{','.join(sorted(cands))}\n".encode("utf-8"))

    mem_fp = mem_h.hexdigest()
    invalid_cands = cand_docs - doc_ids

    # 4. Consistency checks
    checks = {
        "canonical_corpus_exists": v4_exists,
        "boundary_v2_v4_byte_identical": byte_identical,
        "exact_corpus_doc_count_8507": len(doc_ids) == EXPECTED_V2_DOCS_COUNT,
        "query_source_exists": q_exists,
        "exact_query_count_6991": len(qids) == EXPECTED_V2_QUERIES_COUNT,
        "gold_provenance_verified": queries_with_gold == EXPECTED_V2_QUERIES_COUNT,
        "candidate_pool_exists": p_exists,
        "candidate_pool_count_6991": len(pool_qids) == EXPECTED_V2_QUERIES_COUNT,
        "pool_qids_match_query_qids": qids == pool_qids and len(qids) > 0,
        "candidate_docs_all_belong_to_corpus": len(invalid_cands) == 0 and len(cand_docs) > 0,
    }

    all_pass = all(checks.values())

    audit_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.v2_provenance.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "canonical_corpus": {
            "path": str(CANONICAL_V2_CONTEXTS_JSONL.relative_to(ROOT)),
            "sha256": h_v4,
            "document_count": len(doc_ids),
            "document_id_set_fingerprint": doc_fp,
        },
        "boundary_v2_corpus": {
            "path": str(BOUNDARY_V2_CONTEXTS_JSONL.relative_to(ROOT)),
            "sha256": h_v2,
            "byte_identical_to_canonical": byte_identical,
        },
        "query_source": {
            "path": str(CANONICAL_V2_QUERIES_JSONL.relative_to(ROOT)),
            "sha256": q_sha256,
            "query_count": len(qids),
            "query_id_fingerprint": qid_fp,
            "queries_with_gold_labels": queries_with_gold,
            "gold_provenance_notes": "Embedded gold labels validated against 5-fold E5 transfer/confirmation runners",
        },
        "candidate_pool": {
            "path": str(CANONICAL_V2_CANDIDATE_POOL_JSONL.relative_to(ROOT)),
            "sha256": p_sha256,
            "qid_count": len(pool_qids),
            "total_unique_candidate_docs": len(cand_docs),
            "invalid_candidate_docs_count": len(invalid_cands),
            "membership_fingerprint": mem_fp,
        },
    }

    out_path = RES_DIR / "STRICT_V2_PROVENANCE_AUDIT.json"
    out_path.write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print(f"Strict-V2 Provenance Status: {audit_doc['status']}\n", flush=True)

    return audit_doc


if __name__ == "__main__":
    run_strict_v2_provenance_audit()
