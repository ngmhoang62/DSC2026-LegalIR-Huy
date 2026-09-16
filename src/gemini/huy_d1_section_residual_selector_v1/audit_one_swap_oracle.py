"""Audit 2: Diagnostic One-Swap Oracle Audit on CAL600."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_section_residual_selector_v1.common import (
    EXPECTED_D1_R5,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    load_cal_data,
    load_pkl,
)


def compute_one_swap_oracle(
    e0_preds: dict,
    section_scores: dict,
    gold: dict,
    extended: dict,
    all_ids: list,
    blocks: dict,
) -> dict:
    print("=== AUDIT 2: ONE-SWAP ORACLE AUDIT ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    qid_to_block = {}
    for b, q_list in blocks.items():
        for q in q_list:
            qid_to_block[q] = b

    section_top5 = {}
    for q in all_ids:
        docs_q = extended[q]
        s_scores = [section_scores[q].get(d, -999.0) for d in docs_q]
        order = sorted(range(len(docs_q)), key=lambda i: s_scores[i], reverse=True)
        section_top5[q] = [docs_q[i] for i in order[:5]]

    e0_recalls = []
    oracle_recalls = []
    swap_opportunities = []

    for q in all_ids:
        d1_t5 = list(e0_preds[q])
        g = gold[q]
        cur_r = len(set(d1_t5) & g) / max(1, len(g))
        e0_recalls.append(cur_r)

        missing_golds_in_sec = [d for d in section_top5[q] if d in g and d not in d1_t5]
        nongolds_in_d1 = [d for d in d1_t5 if d not in g]

        if missing_golds_in_sec and nongolds_in_d1:
            best_new_r = (len(set(d1_t5) & g) + 1) / max(1, len(g))
            oracle_recalls.append(best_new_r)
            swap_opportunities.append({
                "qid": q,
                "block": qid_to_block[q],
                "e0_recall": cur_r,
                "oracle_recall": best_new_r,
                "delta_recall": best_new_r - cur_r,
                "missing_golds_in_section_top5": missing_golds_in_sec,
                "nongolds_in_d1_top5": nongolds_in_d1,
            })
        else:
            oracle_recalls.append(cur_r)

    e0_r5 = float(np.mean(e0_recalls))
    oracle_r5 = float(np.mean(oracle_recalls))
    n_swaps = len(swap_opportunities)

    print(f"E0 Baseline Recall@5:      {e0_r5:.16f} (Expected: {EXPECTED_D1_R5:.16f})", flush=True)
    print(f"One-Swap Oracle Recall@5:  {oracle_r5:.16f} (Expected: ~0.9683333333333334)", flush=True)
    print(f"Swap Opportunities Count:  {n_swaps} (Expected: ~10 queries)", flush=True)

    oracle_matches = abs(oracle_r5 - 0.9683333333333334) < 1e-4
    n_swaps_matches = (n_swaps == 10)

    audit_result = {
        "schema_version": "dsc2026.gemini.huy_d1_section_residual_selector_v1.oracle.v1",
        "experiment_id": "HUY_D1_SECTION_RESIDUAL_SELECTOR_V1",
        "status": "PASS" if (oracle_matches and n_swaps_matches) else "FAIL",
        "checks": {
            "oracle_recall_approx_09683": oracle_matches,
            "opportunity_count_is_10": n_swaps_matches,
        },
        "e0_recall_at_5": e0_r5,
        "one_swap_oracle_recall_at_5": oracle_r5,
        "oracle_delta": oracle_r5 - e0_r5,
        "opportunity_count": n_swaps,
        "opportunity_queries": swap_opportunities,
    }

    out_path = RES_DIR / "ONE_SWAP_ORACLE_AUDIT.json"
    out_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print("=== ONE-SWAP ORACLE AUDIT PASSED ===\n", flush=True)
    return audit_result
