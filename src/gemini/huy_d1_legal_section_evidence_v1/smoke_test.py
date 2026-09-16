"""Smoke test verifying legal section parser, preselector, and frozen Jina cross-encoder scoring."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    SafeDocumentStore,
    load_jina_crossencoder,
    seed_everything,
)
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
    preselect_legal_sections,
)


def run_smoke_test() -> dict:
    seed_everything(2026)
    print("=== SMOKE TEST: LEGAL SECTION PARSER & FROZEN JINA ===", flush=True)

    # 1. Test DocumentStore and parser
    contexts_dir = (
        ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    docs = SafeDocumentStore(sorted(contexts_dir.glob("context_*.json")))
    print(f"Loaded SafeDocumentStore with {len(docs)} documents.", flush=True)

    test_doc_ids = ["100125", "100062", "100109", "100453", "151846"]
    parser_results = {}
    for did in test_doc_ids:
        raw_text = docs[did]
        sections = parse_document_into_sections(did, raw_text)
        assert len(sections) > 0, f"Doc {did} produced 0 sections!"
        # Check that heading is retained
        assert any(
            s.heading for s in sections
        ), f"Doc {did} sections have no headings!"
        parser_results[did] = {
            "total_sections": len(sections),
            "sample_headings": [s.heading for s in sections[:4]],
            "sample_word_counts": [s.word_count for s in sections[:4]],
        }
        print(
            f"Doc {did}: {len(sections)} sections, headings: {[s.heading[:30] for s in sections[:3]]}",
            flush=True,
        )

    # 2. Test preselector on sample query
    sample_query = (
        "Ngân hàng phá sản thì người gửi tiền sẽ được nhận tiền đền bù là bao nhiêu?"
    )
    doc_151846_sections = parse_document_into_sections("151846", docs["151846"])
    selected_secs = preselect_legal_sections(
        sample_query, doc_151846_sections, count=2
    )
    assert (
        len(selected_secs) == 2
    ), f"Expected 2 selected sections, got {len(selected_secs)}"
    print(
        f"Selected sections for query: {[s.heading for s in selected_secs]}",
        flush=True,
    )

    # 3. Test frozen Jina cross-encoder inference
    print("Loading frozen Jina cross-encoder...", flush=True)
    t0 = time.perf_counter()
    model, tok, prov = load_jina_crossencoder()
    t_load = time.perf_counter() - t0
    print(
        f"Loaded model in {t_load:.2f}s on {prov['device']}. Model is frozen: {prov['is_frozen']}",
        flush=True,
    )

    # Score good section vs bad section
    good_text = selected_secs[0].text
    bad_text = (
        doc_151846_sections[0].text
    )  # preamble or general scope, not compensation limit
    pairs = [
        (sample_query, good_text),
        (sample_query, bad_text),
    ]

    torch.cuda.reset_peak_memory_stats()
    raw_scores = model.compute_score(pairs, batch_size=2, max_length=512)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    score_good = float(raw_scores[0])
    score_bad = float(raw_scores[1])
    score_diff = score_good - score_bad
    print(f"Good section score: {score_good:.4f}", flush=True)
    print(f"Bad section score:  {score_bad:.4f}", flush=True)
    print(f"Score difference:   {score_diff:.4f}", flush=True)
    print(f"Peak VRAM used:     {peak_vram_mb:.2f} MB", flush=True)

    assert score_good != score_bad, "Scores are identical!"

    smoke_results = {
        "status": "PASS",
        "provenance": prov,
        "parser_test": parser_results,
        "score_good": score_good,
        "score_bad": score_bad,
        "score_difference": score_diff,
        "peak_vram_mb": peak_vram_mb,
    }
    print("=== SMOKE TEST PASSED ===", flush=True)
    return smoke_results


if __name__ == "__main__":
    res = run_smoke_test()
    print(json.dumps(res, indent=2))
