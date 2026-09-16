"""Audit baseline candidate recall and pool parity for CAL600 and Strict-V2 benchmarks.

Outputs:
- CAL_BASELINE_PARITY.json
- V2_BASELINE_PARITY.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from .common import (
    EXPECTED_CAL_BASE_CANDIDATE_RECALL,
    EXPECTED_CAL_CANDIDATE_PAIRS,
    EXPECTED_CAL_DOCS_COUNT,
    EXPECTED_CAL_QUERIES_COUNT,
    EXPECTED_V2_BASE_CANDIDATE_RECALL,
    EXPECTED_V2_DOCS_COUNT,
    EXPECTED_V2_QUERIES_COUNT,
    RES_DIR,
    compute_candidate_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    load_cal_generation_inputs,
    load_cal_gold_labels,
    load_v2_generation_inputs,
    load_v2_gold_labels,
)

EXPECTED_CAL_CANDIDATE_FINGERPRINT = "24864c27298b8f48d96b3ddc60c521a5c8c88c84b5e9ca1dbbd5ffbf5e8b595a"
EXPECTED_CAL_QUERY_FINGERPRINT = "3dbcd3aa7b870801ddb7b83a0e1c54aab6ab24a220587b55060d4856cd962f8e"


def audit_cal_baseline() -> Dict[str, Any]:
    print("[PARITY] Auditing CAL600 baseline candidate pool...", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    query_texts, blocks, all_ids, extended = load_cal_generation_inputs()
    gold_labels = load_cal_gold_labels(all_ids)

    cand_fp = compute_candidate_fingerprint(all_ids, extended)
    query_fp = compute_query_fingerprint(all_ids, query_texts)
    total_pairs = sum(len(extended[q]) for q in all_ids)

    # Compute candidate recall
    recalls = []
    gold_misses = []
    for q in all_ids:
        g = gold_labels[q]
        c = set(extended[q])
        rec = len(g & c) / len(g) if g else 1.0
        recalls.append(rec)
        missing = g - c
        if missing:
            gold_misses.append({
                "qid": q,
                "gold_total": len(g),
                "gold_found": len(g & c),
                "gold_missing": sorted(list(missing)),
                "pool_size": len(c),
            })

    macro_recall = sum(recalls) / len(recalls)

    # Block breakdown
    block_recalls: Dict[str, float] = {}
    for block_name, qids in blocks.items():
        b_recalls = [
            len(gold_labels[q] & set(extended[q])) / len(gold_labels[q]) if gold_labels[q] else 1.0
            for q in qids
        ]
        block_recalls[block_name] = sum(b_recalls) / len(b_recalls) if b_recalls else 0.0

    checks = {
        "candidate_fingerprint_exact": cand_fp == EXPECTED_CAL_CANDIDATE_FINGERPRINT,
        "query_fingerprint_exact": query_fp == EXPECTED_CAL_QUERY_FINGERPRINT,
        "query_count_600": len(all_ids) == EXPECTED_CAL_QUERIES_COUNT,
        "pair_count_23532": total_pairs == EXPECTED_CAL_CANDIDATE_PAIRS,
        "recall_exact_match": abs(macro_recall - EXPECTED_CAL_BASE_CANDIDATE_RECALL) < 1e-6,
    }
    all_pass = all(checks.values())

    git_info = get_git_status()
    out_record: Dict[str, Any] = {
        "benchmark": "CAL600",
        "status": "PASS" if all_pass else "FAIL",
        "macro_candidate_recall": macro_recall,
        "expected_macro_candidate_recall": EXPECTED_CAL_BASE_CANDIDATE_RECALL,
        "query_count": len(all_ids),
        "total_candidate_pairs": total_pairs,
        "average_candidates_per_query": total_pairs / len(all_ids),
        "missing_gold_query_count": len(gold_misses),
        "block_recalls": block_recalls,
        "candidate_fingerprint": cand_fp,
        "query_fingerprint": query_fp,
        "checks": checks,
        "git_commit": git_info["head_commit"],
    }

    out_file = RES_DIR / "CAL_BASELINE_PARITY.json"
    out_file.write_text(json.dumps(out_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PARITY] CAL600 baseline parity: {out_record['status']} (Recall={macro_recall:.6f}) -> {out_file}", flush=True)
    return out_record


def audit_v2_baseline() -> Dict[str, Any]:
    print("[PARITY] Auditing Strict-V2 baseline candidate pool...", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    corpus, queries, v2_qids, candidate_pools = load_v2_generation_inputs()
    gold_labels = load_v2_gold_labels(v2_qids)

    total_pairs = sum(len(candidate_pools[q]) for q in v2_qids)

    recalls = []
    missing_queries = 0
    for q in v2_qids:
        g = gold_labels.get(q, set())
        c = set(candidate_pools[q])
        rec = len(g & c) / len(g) if g else 1.0
        recalls.append(rec)
        if len(g - c) > 0:
            missing_queries += 1

    macro_recall = sum(recalls) / len(recalls)

    checks = {
        "corpus_doc_count_8507": len(corpus) == EXPECTED_V2_DOCS_COUNT,
        "query_count_6991": len(v2_qids) == EXPECTED_V2_QUERIES_COUNT,
        "recall_exact_match": abs(macro_recall - EXPECTED_V2_BASE_CANDIDATE_RECALL) < 1e-6,
    }
    all_pass = all(checks.values())

    git_info = get_git_status()
    out_record: Dict[str, Any] = {
        "benchmark": "Strict-V2",
        "status": "PASS" if all_pass else "FAIL",
        "macro_candidate_recall": macro_recall,
        "expected_macro_candidate_recall": EXPECTED_V2_BASE_CANDIDATE_RECALL,
        "query_count": len(v2_qids),
        "total_candidate_pairs": total_pairs,
        "average_candidates_per_query": total_pairs / len(v2_qids),
        "missing_gold_query_count": missing_queries,
        "checks": checks,
        "git_commit": git_info["head_commit"],
    }

    out_file = RES_DIR / "V2_BASELINE_PARITY.json"
    out_file.write_text(json.dumps(out_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PARITY] V2 baseline parity: {out_record['status']} (Recall={macro_recall:.6f}) -> {out_file}", flush=True)
    return out_record


def run_baseline_parity_audits() -> bool:
    cal_res = audit_cal_baseline()
    v2_res = audit_v2_baseline()
    return (cal_res["status"] == "PASS") and (v2_res["status"] == "PASS")


if __name__ == "__main__":
    ok = run_baseline_parity_audits()
    if not ok:
        sys.exit(1)
