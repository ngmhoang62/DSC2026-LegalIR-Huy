"""Audit candidate-level diagnostic complementarity between:
1. Baseline Candidate Pool
2. Query-Anchored Legal Reference Additions (HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1)
3. Legal Section Retrieval Additions (HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from .common import (
    CAL_LEGAL_REF_ADDITIONS_JSONL,
    RES_DIR,
    ROOT,
    get_git_status,
    load_cal_generation_inputs,
    load_cal_gold_labels,
    sha256_file,
)


def run_complementarity_audit() -> Dict[str, Any]:
    print("[COMPLEMENTARITY] Evaluating generator diagnostic complementarity...", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    query_texts, blocks, all_ids, extended = load_cal_generation_inputs()
    gold_labels = load_cal_gold_labels(all_ids)

    # 1. Load Section Retrieval additions
    sec_file = RES_DIR / "CAL_SECTION_RETRIEVAL_ADDITIONS.jsonl"
    if not sec_file.exists():
        raise FileNotFoundError(f"Section retrieval additions not found: {sec_file}")

    sec_additions: Dict[str, Set[str]] = {}
    with open(sec_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sec_additions[row["qid"]] = set(c["doc_id"] for c in row.get("additions", []))

    # 2. Load Legal Reference additions
    ref_exists = CAL_LEGAL_REF_ADDITIONS_JSONL.exists()
    ref_additions: Dict[str, Set[str]] = {}
    if ref_exists:
        with open(CAL_LEGAL_REF_ADDITIONS_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    ref_additions[row["qid"]] = set(row.get("newly_added_doc_ids", []))
    else:
        print(f"[COMPLEMENTARITY] Warning: Legal ref additions file not found at {CAL_LEGAL_REF_ADDITIONS_JSONL}", flush=True)

    # 3. Evaluate 4 configurations
    recalls_base = []
    recalls_ref = []
    recalls_sec = []
    recalls_joint = []

    recovered_by_ref: Set[str] = set()
    recovered_by_sec: Set[str] = set()
    recovered_by_joint: Set[str] = set()

    for qid in all_ids:
        g = gold_labels[qid]
        b = set(extended[qid])
        r = ref_additions.get(qid, set())
        s = sec_additions.get(qid, set())
        j = b | r | s

        r_base = len(g & b) / len(g) if g else 1.0
        r_ref = len(g & (b | r)) / len(g) if g else 1.0
        r_sec = len(g & (b | s)) / len(g) if g else 1.0
        r_joint = len(g & j) / len(g) if g else 1.0

        recalls_base.append(r_base)
        recalls_ref.append(r_ref)
        recalls_sec.append(r_sec)
        recalls_joint.append(r_joint)

        if r_ref - r_base > 1e-9:
            recovered_by_ref.add(qid)
        if r_sec - r_base > 1e-9:
            recovered_by_sec.add(qid)
        if r_joint - r_base > 1e-9:
            recovered_by_joint.add(qid)

    m_base = sum(recalls_base) / len(recalls_base)
    m_ref = sum(recalls_ref) / len(recalls_ref)
    m_sec = sum(recalls_sec) / len(recalls_sec)
    m_joint = sum(recalls_joint) / len(recalls_joint)

    ref_only = recovered_by_ref - recovered_by_sec
    sec_only = recovered_by_sec - recovered_by_ref
    both = recovered_by_ref & recovered_by_sec

    total_ref_adds = sum(len(v) for v in ref_additions.values())
    total_sec_adds = sum(len(v) for v in sec_additions.values())

    git_info = get_git_status()
    result: Dict[str, Any] = {
        "benchmark": "CAL600",
        "reference_generator_available": ref_exists,
        "reference_generator_artifact": str(CAL_LEGAL_REF_ADDITIONS_JSONL.relative_to(ROOT)) if ref_exists else "N/A",
        "reference_generator_sha256": sha256_file(CAL_LEGAL_REF_ADDITIONS_JSONL) if ref_exists else "N/A",
        "metrics": {
            "baseline": {
                "macro_candidate_recall": m_base,
                "delta_vs_baseline": 0.0,
                "recovered_queries_count": 0,
            },
            "legal_ref_expansion_only": {
                "macro_candidate_recall": m_ref,
                "delta_vs_baseline": m_ref - m_base,
                "recovered_queries_count": len(recovered_by_ref),
                "total_additions": total_ref_adds,
            },
            "section_retrieval_expansion_only": {
                "macro_candidate_recall": m_sec,
                "delta_vs_baseline": m_sec - m_base,
                "recovered_queries_count": len(recovered_by_sec),
                "total_additions": total_sec_adds,
            },
            "joint_diagnostic_union": {
                "macro_candidate_recall": m_joint,
                "delta_vs_baseline": m_joint - m_base,
                "delta_vs_ref_alone": m_joint - m_ref,
                "delta_vs_sec_alone": m_joint - m_sec,
                "recovered_queries_count": len(recovered_by_joint),
            },
        },
        "query_overlap_analysis": {
            "recovered_only_by_legal_ref": sorted(list(ref_only)),
            "recovered_only_by_section_retrieval": sorted(list(sec_only)),
            "recovered_by_both": sorted(list(both)),
            "total_unique_recovered_queries": len(recovered_by_joint),
        },
        "git_commit": git_info["head_commit"],
    }

    out_file = RES_DIR / "GENERATOR_COMPLEMENTARITY_AUDIT.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[COMPLEMENTARITY] Saved -> {out_file}", flush=True)
    print(
        f"[COMPLEMENTARITY] Base={m_base:.6f} | Ref={m_ref:.6f} (+{m_ref - m_base:.6f}) | "
        f"Sec={m_sec:.6f} (+{m_sec - m_base:.6f}) | Joint={m_joint:.6f} (+{m_joint - m_base:.6f})\n"
        f"  Ref-only recoveries: {len(ref_only)}, Sec-only recoveries: {len(sec_only)}, Both: {len(both)}"
    )
    return result


if __name__ == "__main__":
    run_complementarity_audit()
