"""Audit: Legal Section Parser Parity against Audited Evidence V1 Parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections as original_parse,
)
from src.gemini.huy_d1_legal_section_retrieval_expansion_v1.common import (
    CAL_CONTEXTS_DIR,
    RES_DIR,
    load_cal_corpus,
    sha256_file,
)
from src.gemini.huy_d1_legal_section_retrieval_expansion_v1.legal_section_parser import (
    parse_document_into_sections as current_parse,
)

ORIGINAL_PARSER_PATH = ROOT / "src/gemini/huy_d1_legal_section_evidence_v1/legal_section_parser.py"
CURRENT_PARSER_PATH = ROOT / "src/gemini/huy_d1_legal_section_retrieval_expansion_v1/legal_section_parser.py"


def run_parser_parity_audit(sample_size: int = 50) -> Dict[str, Any]:
    print("=== AUDIT: LEGAL SECTION PARSER PARITY ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    orig_sha = sha256_file(ORIGINAL_PARSER_PATH)
    curr_sha = sha256_file(CURRENT_PARSER_PATH)
    code_exact_match = (orig_sha == curr_sha)

    corpus = load_cal_corpus()
    sample_doc_ids = sorted(corpus.keys(), key=lambda x: int(x) if x.isdigit() else x)[:sample_size]

    mismatches = []
    total_sections_evaluated = 0

    for doc_id in sample_doc_ids:
        raw_text = corpus[doc_id].get("passage", "")
        orig_secs = original_parse(doc_id, raw_text)
        curr_secs = current_parse(doc_id, raw_text)

        total_sections_evaluated += len(curr_secs)

        if len(orig_secs) != len(curr_secs):
            mismatches.append({
                "doc_id": doc_id,
                "error": f"Section count mismatch: orig={len(orig_secs)}, curr={len(curr_secs)}"
            })
            continue

        for i, (os, cs) in enumerate(zip(orig_secs, curr_secs)):
            diffs = []
            if os.section_index != cs.section_index:
                diffs.append("section_index")
            if os.section_type != cs.section_type:
                diffs.append("section_type")
            if os.heading != cs.heading:
                diffs.append("heading")
            if os.text != cs.text:
                diffs.append("text")
            if os.word_count != cs.word_count:
                diffs.append("word_count")

            if diffs:
                mismatches.append({
                    "doc_id": doc_id,
                    "section_index": i,
                    "differing_fields": diffs,
                })

    status_pass = (len(mismatches) == 0) and code_exact_match

    audit_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_retrieval_expansion_v1.parser_parity.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1",
        "status": "PASS" if status_pass else "FAIL",
        "checks": {
            "source_code_sha256_match": code_exact_match,
            "deterministic_sample_zero_mismatches": len(mismatches) == 0,
            "sample_documents_count": len(sample_doc_ids),
            "total_sections_evaluated": total_sections_evaluated,
        },
        "original_parser": {
            "path": str(ORIGINAL_PARSER_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": orig_sha,
        },
        "current_parser": {
            "path": str(CURRENT_PARSER_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": curr_sha,
        },
        "mismatches_count": len(mismatches),
        "mismatches": mismatches[:10],
    }

    out_path = RES_DIR / "PARSER_PARITY_AUDIT.json"
    out_path.write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print(f"Parser Parity Status: {audit_doc['status']} (Evaluated {total_sections_evaluated} sections across {len(sample_doc_ids)} sample docs)\n", flush=True)

    return audit_doc


if __name__ == "__main__":
    run_parser_parity_audit()
