"""End-to-end execution pipeline for HUY_FULLTRAIN_PROFILE_PORT_V1."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

STAGES = [
    ("STAGE_1_AUDIT_STRICT_EVIDENCE", "audit_strict_profile_evidence.py"),
    ("STAGE_2_PROFILE_DATA_ISOLATION", "profile_data_isolation.py"),
    ("STAGE_3_SCORE_PROFILE_BM25", "score_profile_bm25.py"),
    ("STAGE_4_EVALUATE_DUAL_CAL", "evaluate_profile_dual_cal.py"),
    ("STAGE_5_MATERIALIZE_AND_AUDIT_PUBLIC", "materialize_and_audit_public.py"),
    ("STAGE_6_BUILD_DECISION_AND_PROVENANCE", "build_decision_and_provenance.py"),
]


def main():
    total_start = time.perf_counter()
    print("================================================================================")
    print("STARTING PIPELINE: HUY_FULLTRAIN_PROFILE_PORT_V1")
    print("================================================================================")

    for stage_name, script_name in STAGES:
        script_path = DIR / script_name
        print(f"\n>>> Running {stage_name} ({script_name})...", flush=True)
        t0 = time.perf_counter()
        res = subprocess.run([PYTHON, str(script_path)], capture_output=False)
        if res.returncode != 0:
            print(f"FAILED: {stage_name} exited with code {res.returncode}", flush=True)
            sys.exit(res.returncode)
        print(f">>> Completed {stage_name} in {time.perf_counter() - t0:.2f}s", flush=True)

    print("\n================================================================================")
    print(f"PIPELINE COMPLETE in {time.perf_counter() - total_start:.2f}s")
    print("================================================================================")


if __name__ == "__main__":
    main()
