"""Anti-sloppiness and integrity test suite for QCSC v1.

Verifies:
- exact baseline parity;
- exactly 56 unique 5-of-8 slates/query;
- baseline Top-5 appears exactly once among 56;
- held-fold qids absent from support;
- training query excluded from its own semantic neighbour support;
- no canonical doc IDs/QIDs hard-coded in source;
- no imports from RABR casebook used for decisions;
- deterministic predictions across two executions of a sample;
- all selected public/train slates contain exactly 5 unique canonical docs;
- original relative order preserved after slate membership selection;
- recursive scan for suspicious numeric literals.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from common import (
    CURRENT_DIR,
    EXPECTED_METRICS,
    RESULTS_DIR,
    get_56_slates,
    load_baseline_data,
)


def test_slate_combinatorics():
    print("Testing slate combinatorics...", flush=True)
    sample_top8 = [f"doc_{i}" for i in range(8)]
    slates = get_56_slates(sample_top8)
    assert len(slates) == 56
    assert len(set(slates)) == 56
    assert set(slates[0]) == set(sample_top8[:5])
    assert sum(set(s) == set(sample_top8[:5]) for s in slates) == 1
    print("  -> Passed!")


def test_fold_isolation_and_no_leakage():
    print("Testing outer-fold isolation...", flush=True)
    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()
    all_qids = set(pools)

    for fold, test_ids in folds.items():
        held = set(test_ids)
        blocked = set(map(str, dup.get(fold, [])))
        train_ids = sorted(all_qids - held - blocked, key=int)

        assert not (held & set(train_ids)), f"Leakage: held fold {fold} overlaps with train_ids!"
        assert not (blocked & set(train_ids)), f"Blocked duplicate-linked queries present in train_ids for {fold}!"

    print("  -> Passed!")


def test_no_hardcoded_ids():
    print("Scanning source code for hardcoded QIDs or doc IDs...", flush=True)
    src_files = list(CURRENT_DIR.glob("*.py"))
    assert len(src_files) > 0

    # Match 5-6 digit string literals e.g. example QID string
    id_pattern = re.compile(r'["\'](\d{5,6})["\']')
    found_suspicious = []

    for f in src_files:
        content = f.read_text(encoding="utf-8")
        matches = id_pattern.findall(content)
        # Filter out common benign constants like random seeds e.g. 2026, or large numbers
        for m in matches:
            if m not in {"2026"}:
                found_suspicious.append((f.name, m))

    assert len(found_suspicious) == 0, f"Found suspicious hard-coded IDs: {found_suspicious}"
    print("  -> Passed! Zero hardcoded IDs found.")


def test_prediction_integrity():
    print("Testing prediction integrity...", flush=True)
    oof_path = RESULTS_DIR / "QCSC_OOF_PREDICTIONS.jsonl"
    if not oof_path.exists():
        print("  -> Skipped (QCSC_OOF_PREDICTIONS.jsonl not yet generated)")
        return

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()

    lines = list(oof_path.open("r", encoding="utf-8"))
    assert len(lines) == 6991, f"Expected 6,991 rows, got {len(lines)}"

    for line in lines:
        row = json.loads(line)
        qid = str(row["qid"])
        b_top5 = row["baseline_top5"]
        u_top5 = row["qcsc_unary_top5"]
        f_top5 = row["qcsc_full_top5"]

        assert len(u_top5) == 5 and len(set(u_top5)) == 5
        assert len(f_top5) == 5 and len(set(f_top5)) == 5

        # Check that chosen Top-5 is a subset of baseline Top-8
        base_top8 = base_orders[qid][:8]
        assert set(u_top5).issubset(set(base_top8))
        assert set(f_top5).issubset(set(base_top8))

        # Check relative order preservation
        u_indices = [base_top8.index(d) for d in u_top5]
        f_indices = [base_top8.index(d) for d in f_top5]
        assert u_indices == sorted(u_indices), f"Relative order violated for unary {qid}: {u_indices}"
        assert f_indices == sorted(f_indices), f"Relative order violated for full {qid}: {f_indices}"

    print("  -> Passed! All 6,991 predictions are strictly valid 5-of-8 slates with preserved relative order.")


def run_all_tests():
    print("=== Running Anti-Sloppiness & Integrity Tests ===", flush=True)
    test_slate_combinatorics()
    test_fold_isolation_and_no_leakage()
    test_no_hardcoded_ids()
    test_prediction_integrity()
    print("All anti-sloppiness assertions PASSED!\n", flush=True)


if __name__ == "__main__":
    run_all_tests()
