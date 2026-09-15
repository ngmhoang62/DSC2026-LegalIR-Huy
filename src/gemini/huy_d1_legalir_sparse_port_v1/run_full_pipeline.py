"""Master runner for HUY_D1_LEGALIR_SPARSE_PORT_V1."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .audit_corpus_coverage import audit_corpus_coverage
from .audit_sparse_source import audit_sparse_source
from .audit_strict_prior import audit_prior_evidence
from .build_decision_and_trace import build_decision_and_trace
from .evaluate_sparse_cal import evaluate_sparse_cal
from .materialize_public_packages import materialize_public_packages
from .run_paired_bootstrap import run_paired_bootstrap
from .verify_sparse_repro_parity import verify_sparse_repro_parity


def run_full_pipeline() -> None:
    t_start = time.perf_counter()
    print("================================================================================")
    print("STARTING PIPELINE: HUY_D1_LEGALIR_SPARSE_PORT_V1")
    print("================================================================================")

    # Step 1: Strict-V2 Prior Audit
    print("\n>>> [1/7] Auditing Strict-V2 Prior Evidence...")
    audit_prior_evidence()

    # Step 2: Exact Sparse Source Audit
    print("\n>>> [2/7] Auditing Exact Sparse Source Implementations...")
    audit_sparse_source()

    # Step 3: Corpus Coverage Audit (8,507 vs 8,532)
    print("\n>>> [3/7] Auditing Corpus Coverage and Duplicate Twin Mappings...")
    audit_corpus_coverage()

    # Step 4: Strict-V2 Sparse Repro Parity (>=64 queries)
    print("\n>>> [4/7] Verifying Fresh Sparse Repro Parity on 64 queries...")
    verify_sparse_repro_parity(num_samples=64)

    # Step 5: CAL600 LOBO Evaluation (S0 vs S1)
    print("\n>>> [5/7] Evaluating CAL600 LOBO for S0 and S1...")
    evaluate_sparse_cal()

    # Step 6: Paired Bootstrap (10,000 resamples)
    print("\n>>> [6/7] Running 10,000 Paired Bootstrap Resamples...")
    run_paired_bootstrap(n_resamples=10000, seed=2026)

    # Step 7: Public Packages & Churn
    print("\n>>> [7/7] Materializing Public Test Packages & Auditing Churn...")
    materialize_public_packages()

    # Finalize Decision & Provenance
    print("\n>>> Finalizing Decision and Audit Reports...")
    res = build_decision_and_trace()

    elapsed = time.perf_counter() - t_start
    print("================================================================================")
    print(f"PIPELINE COMPLETE: Verdict = {res['verdict']} (runtime: {elapsed:.2f}s)")
    print("================================================================================")


if __name__ == "__main__":
    run_full_pipeline()
