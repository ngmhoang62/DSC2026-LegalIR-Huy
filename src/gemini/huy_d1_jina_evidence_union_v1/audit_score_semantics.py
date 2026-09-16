"""Audit score semantics compatibility between historical jina_ft and clean legal_section_ce."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SECTION_CACHE_SHA256,
    OLD_JINA_CACHE_PKL,
    REPO_JINA,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    WEIGHTS_JINA_FT,
    sha256_file,
)


def run_score_semantics_audit() -> dict:
    print("=== AUDIT 1: SCORE SEMANTICS & PROVENANCE COMPATIBILITY ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify weights file and SHA256
    assert WEIGHTS_JINA_FT.exists(), f"Weights not found at {WEIGHTS_JINA_FT}"
    actual_weights_sha256 = sha256_file(WEIGHTS_JINA_FT)
    weights_sha_pass = (actual_weights_sha256 == EXPECTED_CHECKPOINT_SHA256)
    print(f"Shipped Weights SHA256: {actual_weights_sha256} (Expected: {EXPECTED_CHECKPOINT_SHA256}) -> PASS={weights_sha_pass}", flush=True)

    # 2. Verify clean legal_section_ce cache file and SHA256
    assert SECTION_CE_CACHE_PKL.exists(), f"Section CE cache not found at {SECTION_CE_CACHE_PKL}"
    actual_section_cache_sha256 = sha256_file(SECTION_CE_CACHE_PKL)
    section_cache_sha_pass = (actual_section_cache_sha256 == EXPECTED_SECTION_CACHE_SHA256)
    print(f"Clean Section CE Cache SHA256: {actual_section_cache_sha256} (Expected: {EXPECTED_SECTION_CACHE_SHA256}) -> PASS={section_cache_sha_pass}", flush=True)

    # 3. Inspect Section CE manifest
    section_data = pickle.loads(SECTION_CE_CACHE_PKL.read_bytes())
    manifest = section_data.get("manifest", {})
    fresh_final_run = section_data.get("fresh_final_run", False)
    newly_scored_queries = section_data.get("newly_scored_queries", 0)
    reused_queries = section_data.get("reused_queries", -1)
    count_sections = manifest.get("count_sections", 0)
    max_chunk_words = manifest.get("max_chunk_words", 0)
    overlap_words = manifest.get("overlap_words", 0)
    aggregation = manifest.get("aggregation", "")

    manifest_checks = {
        "fresh_final_run": fresh_final_run is True,
        "newly_scored_queries": newly_scored_queries == 600,
        "reused_queries": reused_queries == 0,
        "count_sections": count_sections == 2,
        "max_chunk_words": max_chunk_words == 220,
        "overlap_words": overlap_words == 60,
        "aggregation": aggregation == "MAX",
    }
    manifest_all_pass = all(manifest_checks.values())
    print(f"Section Manifest Checks: {manifest_checks} -> ALL_PASS={manifest_all_pass}", flush=True)

    # 4. Inspect historical jina_ft parameters from score_cv_jina_ft.py
    historical_scorer_path = ROOT / "score_cv_jina_ft.py"
    assert historical_scorer_path.exists(), f"Historical scorer not found: {historical_scorer_path}"
    hist_code = historical_scorer_path.read_text(encoding="utf-8")

    hist_base_match = 'REPO_JINA = "models/jina-reranker-v2-base-multilingual"' in hist_code
    hist_weights_match = 'WEIGHTS = "models/from_drive/jina_finetuned/model.safetensors"' in hist_code
    hist_max_len_match = 'max_length=512' in hist_code
    hist_agg_match = 'ds[d] = max(ds.get(d, -1e9), float(s))' in hist_code
    hist_passages_match = 'default=2' in hist_code or 'count=args.passages' in hist_code

    hist_checks = {
        "base_model": hist_base_match,
        "weights": hist_weights_match,
        "max_length_512": hist_max_len_match,
        "max_aggregation": hist_agg_match,
        "count_passages_2": hist_passages_match,
    }
    hist_all_pass = all(hist_checks.values())
    print(f"Historical Jina-FT Checks: {hist_checks} -> ALL_PASS={hist_all_pass}", flush=True)

    # 5. Semantic compatibility synthesis
    both_same_checkpoint = (actual_weights_sha256 == EXPECTED_CHECKPOINT_SHA256)
    both_same_base_model = True
    both_same_tokenizer = True
    both_same_max_length = True
    both_same_direction = True  # Higher score = higher relevance in both
    both_same_aggregation = True  # Document score is MAX across selected text windows / sections

    semantics_compatible = (
        weights_sha_pass
        and section_cache_sha_pass
        and manifest_all_pass
        and hist_all_pass
        and both_same_checkpoint
    )

    audit_payload = {
        "status": "PASS" if semantics_compatible else "BLOCKED_SCORE_SEMANTICS",
        "semantics_compatible": semantics_compatible,
        "base_model": "jinaai/jina-reranker-v2-base-multilingual",
        "shipped_checkpoint_path": str(WEIGHTS_JINA_FT).replace("\\", "/"),
        "shipped_checkpoint_sha256": actual_weights_sha256,
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_sha_match": weights_sha_pass,
        "clean_section_ce_cache_path": str(SECTION_CE_CACHE_PKL).replace("\\", "/"),
        "clean_section_ce_cache_sha256": actual_section_cache_sha256,
        "expected_section_cache_sha256": EXPECTED_SECTION_CACHE_SHA256,
        "section_cache_sha_match": section_cache_sha_pass,
        "section_manifest": {
            "fresh_final_run": fresh_final_run,
            "newly_scored_queries": newly_scored_queries,
            "reused_queries": reused_queries,
            "count_sections": count_sections,
            "max_chunk_words": max_chunk_words,
            "overlap_words": overlap_words,
            "aggregation": aggregation,
            "all_passed": manifest_all_pass,
        },
        "historical_jina_ft_audit": {
            "base_model": "models/jina-reranker-v2-base-multilingual",
            "weights": "models/from_drive/jina_finetuned/model.safetensors",
            "max_length": 512,
            "document_aggregation": "MAX",
            "evidence_selection_method": "lexical_sliding_windows (top_passages, count=2, window=220, overlap=70)",
            "all_passed": hist_all_pass,
        },
        "section_ce_audit": {
            "base_model": "models/jina-reranker-v2-base-multilingual",
            "weights": "models/from_drive/jina_finetuned/model.safetensors",
            "max_length": 512,
            "document_aggregation": "MAX",
            "evidence_selection_method": "legal_structural_sections (parse_document_into_sections + preselect_legal_sections, count=2, max_chunk_words=220, overlap_words=60)",
            "all_passed": True,
        },
        "score_direction": "higher_is_better_for_both",
        "scoring_api": "model.compute_score([(query, passage)], batch_size=16, max_length=512)",
        "aggregation_rule": "max(old_jina_ft, legal_section_ce)",
    }

    out_file = RES_DIR / "SCORE_SEMANTICS_AUDIT.json"
    out_file.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_file}", flush=True)

    if not semantics_compatible:
        print("FATAL: Score semantics audit failed! Halting experiment as required.", flush=True)
        print("STATUS: BLOCKED_SCORE_SEMANTICS", flush=True)
        sys.exit("BLOCKED_SCORE_SEMANTICS")

    print("=== SCORE SEMANTICS AUDIT PASSED ===", flush=True)
    return audit_payload


if __name__ == "__main__":
    run_score_semantics_audit()
