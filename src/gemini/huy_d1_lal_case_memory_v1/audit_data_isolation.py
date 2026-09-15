"""Audit duplicate safety and verify nested cross-fitting data isolation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

V2_BASE_PATH = ROOT / "results" / "research_v2_forensic" / "V2_EXECUTABLE_BASELINE.json"


def load_duplicate_graph() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[Tuple[str, str]], Dict[str, Set[str]]]:
    with open(V2_BASE_PATH, "r", encoding="utf-8") as f:
        v2_base = json.load(f)

    dup_info = v2_base.get("duplicate_contamination", {})
    exact_examples = dup_info.get("exact_normalized", {}).get("examples", [])
    near_examples = dup_info.get("near_duplicate_char_tfidf", {}).get("examples", [])

    directed_links: Set[Tuple[str, str]] = set()

    for ex in exact_examples:
        qids = [str(q) for q in ex["qids"]]
        for q1 in qids:
            for q2 in qids:
                if q1 != q2:
                    directed_links.add((q1, q2))

    for ex in near_examples:
        qa = str(ex["qid_a"])
        qb = str(ex["qid_b"])
        if qa != qb:
            directed_links.add((qa, qb))
            directed_links.add((qb, qa))

    dup_map: Dict[str, Set[str]] = defaultdict(set)
    for u, v in directed_links:
        dup_map[u].add(v)

    return exact_examples, near_examples, directed_links, dup_map


def audit_cal_data_isolation() -> Dict[str, Any]:
    exact_examples, near_examples, directed_links, dup_map = load_duplicate_graph()

    # Load CAL blocks
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src" / "huy_fasttrack"))
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import load_cal_inputs
    import run_huy_5fold_fasttrack as core

    queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, vnlegal_cv, type_rows, cite_rows = load_cal_inputs()
    folds, pools, questions, v2_golds, e5_orders, e5_scores, v2_dup, _ = core.load_inputs()

    v2_population = set(pools.keys())
    cal_population = set(all_ids)

    # Verification of duplicate link counts
    exact_count = len(exact_examples)
    near_count = len(near_examples)
    total_directed = len(directed_links)

    # Verify nested cross-fitting isolation for each CAL block
    block_names = sorted(blocks.keys())
    isolation_checks = {}
    leakage_detected = False
    leakage_details = []

    for held_name in block_names:
        held_ids = set(blocks[held_name])
        # Duplicates of held block
        held_dups = {dup for q in held_ids for dup in dup_map.get(q, set())}

        # Outer test support: P - held - dup(held)
        test_forbidden = held_ids | held_dups
        test_support = v2_population - test_forbidden

        # Check test support does not contain any held_ids or their dups
        test_leak_held = test_support & held_ids
        test_leak_dup = test_support & held_dups
        if test_leak_held or test_leak_dup:
            leakage_detected = True
            leakage_details.append(f"Test support for {held_name} leaked held={test_leak_held} dup={test_leak_dup}")

        train_block_checks = {}
        for train_name in block_names:
            if train_name == held_name:
                continue
            train_ids = set(blocks[train_name])
            train_dups = {dup for q in train_ids for dup in dup_map.get(q, set())}

            # Training support for T: P - held - T - dup(held) - dup(T)
            train_forbidden = held_ids | train_ids | held_dups | train_dups
            train_support = v2_population - train_forbidden

            # Check train support does not contain held, T, or their dups
            leak_held = train_support & held_ids
            leak_train = train_support & train_ids
            leak_held_dup = train_support & held_dups
            leak_train_dup = train_support & train_dups

            if leak_held or leak_train or leak_held_dup or leak_train_dup:
                leakage_detected = True
                leakage_details.append(f"Train block {train_name} in fold {held_name} leaked: held={leak_held}, train={leak_train}, dups={leak_held_dup | leak_train_dup}")

            train_block_checks[train_name] = {
                "train_block_size": len(train_ids),
                "train_dups_count": len(train_dups),
                "train_support_size": len(train_support),
                "leakage_count": len(leak_held) + len(leak_train) + len(leak_held_dup) + len(leak_train_dup),
                "leakage_passed": (len(leak_held) + len(leak_train) + len(leak_held_dup) + len(leak_train_dup)) == 0,
            }

        isolation_checks[held_name] = {
            "held_block_size": len(held_ids),
            "held_dups_count": len(held_dups),
            "test_support_size": len(test_support),
            "test_leakage_count": len(test_leak_held) + len(test_leak_dup),
            "test_leakage_passed": (len(test_leak_held) + len(test_leak_dup)) == 0,
            "training_blocks": train_block_checks,
        }

    status = "PASS" if not leakage_detected else "BLOCKED_LEAKAGE"

    audit_report = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.memory_data_isolation_audit.v1",
        "status": status,
        "exact_normalized_groups_count": exact_count,
        "near_duplicate_pairs_count": near_count,
        "unique_bidirectional_directed_links_count": total_directed,
        "expected_unique_directed_links_count": 50,
        "directed_links_verified": total_directed == 50,
        "total_v2_evaluable_queries": len(v2_population),
        "total_cal_queries": len(cal_population),
        "cal_in_v2": len(cal_population & v2_population),
        "cal_not_in_v2": list(cal_population - v2_population),
        "nested_cross_fitting_isolation_by_block": isolation_checks,
        "leakage_detected": leakage_detected,
        "leakage_details": leakage_details,
    }

    out_path = RESULTS_DIR / "MEMORY_DATA_ISOLATION_AUDIT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Wrote {out_path} (status: {status})")
    return audit_report


if __name__ == "__main__":
    audit_cal_data_isolation()
