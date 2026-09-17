"""Novel candidate proposals from frozen AITeamVN-FT full-corpus Top-20."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.gemini.huy_d1_aiteam_novel_consensus_v1.common import (
    AITEAM_REPORT_PATH,
    AITEAM_TOP50_PATH,
    EXPECTED_AITEAM_REPORT_SHA256,
    EXPECTED_AITEAM_TOP50_SHA256,
    RESULTS_DIR,
    ROOT,
    sha256_file,
)


def load_and_verify_aiteam_source() -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Load and strictly verify AITeam Full-Corpus Top-50 and Report provenance."""
    top50_sha = sha256_file(AITEAM_TOP50_PATH)
    if top50_sha != EXPECTED_AITEAM_TOP50_SHA256:
        raise RuntimeError(
            f"AITeam Top-50 SHA mismatch! Expected {EXPECTED_AITEAM_TOP50_SHA256}, got {top50_sha}"
        )

    report_sha = sha256_file(AITEAM_REPORT_PATH)
    if report_sha != EXPECTED_AITEAM_REPORT_SHA256:
        raise RuntimeError(
            f"AITeam Report SHA mismatch! Expected {EXPECTED_AITEAM_REPORT_SHA256}, got {report_sha}"
        )

    top50_raw = json.loads(AITEAM_TOP50_PATH.read_text(encoding="utf-8"))
    report_raw = json.loads(AITEAM_REPORT_PATH.read_text(encoding="utf-8"))

    rankings_raw = top50_raw.get("rankings", top50_raw)
    top20_rankings: Dict[str, List[str]] = {}

    for q, rlist in rankings_raw.items():
        # Strictly slice ranks 1..20
        top20 = [str(x) if isinstance(x, (int, str)) else str(x[0]) for x in rlist[:20]]
        top20_rankings[str(q)] = top20

    prov_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.aiteam_provenance.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
        "aiteam_top50_path": str(AITEAM_TOP50_PATH.relative_to(ROOT)).replace("\\", "/"),
        "aiteam_top50_sha256": top50_sha,
        "aiteam_report_path": str(AITEAM_REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "aiteam_report_sha256": report_sha,
        "source_contract": report_raw.get("source_contract", {}),
        "slice_depth": 20,
        "queries_count": len(top20_rankings),
    }

    out_prov = RESULTS_DIR / "AITEAM_SOURCE_PROVENANCE.json"
    out_prov.write_text(json.dumps(prov_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_prov}", flush=True)

    return top20_rankings, prov_doc


def extract_novel_proposals(
    all_ids: List[str],
    extended: Dict[str, List[str]],
    aiteam20_rankings: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Extract out-of-pool novel candidates from AITeam Top-20 for each query.
    
    STRICT LEAKAGE RULE: Zero gold labels are read, materialized, or evaluated here.
    """
    records = []
    novel_map: Dict[str, List[str]] = {}
    novel_counts = []
    zero_novel_count = 0
    ge_1_novel_count = 0

    for q in all_ids:
        d1_pool = set(extended[q])
        aiteam20 = aiteam20_rankings.get(q, [])

        novel_docs = [d for d in aiteam20 if d not in d1_pool]
        novel_ranks = [aiteam20.index(d) + 1 for d in novel_docs]
        novel_map[q] = novel_docs
        n_cnt = len(novel_docs)
        novel_counts.append(n_cnt)

        if n_cnt == 0:
            zero_novel_count += 1
        else:
            ge_1_novel_count += 1

        records.append({
            "qid": q,
            "aiteam_top20_doc_ids": aiteam20,
            "exact_d1_candidate_pool_size": len(d1_pool),
            "novel_doc_ids": novel_docs,
            "novel_aiteam_ranks": novel_ranks,
            "novel_count": n_cnt,
        })

    summary = {
        "total_queries": len(all_ids),
        "total_novel_candidates": int(sum(novel_counts)),
        "mean_novel_per_query": float(np.mean(novel_counts)),
        "median_novel_per_query": float(np.median(novel_counts)),
        "max_novel_per_query": int(max(novel_counts)),
        "queries_with_zero_novel": zero_novel_count,
        "queries_with_ge_1_novel": ge_1_novel_count,
    }

    proposals_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.novel_proposals.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
        "summary": summary,
        "proposals": records,
    }

    out_path = RESULTS_DIR / "AITEAM20_NOVEL_PROPOSALS_LABEL_FREE.json"
    out_path.write_text(json.dumps(proposals_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    return novel_map, proposals_doc
