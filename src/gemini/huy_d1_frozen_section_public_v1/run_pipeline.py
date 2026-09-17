"""Master Pipeline for HUY_D1_FROZEN_SECTION_PUBLIC_V1.

Executes:
1. Source and Frozen Model Provenance Audit
2. Local Parity Gate on CAL600 (P0 = 0.956944, P1 = 0.957778, 2W / 1L / 597T)
3. Public D1 Control Parity Audit (1000/1000 exact ordered matches vs current D1 champion)
4. Public Frozen Legal-Section CE Scoring (39,233 candidate pairs, GPU, fresh cache & seal)
5. Production Training (50D P1), Public Prediction Churn Diagnostics, Submission Packaging, and Manifest
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .audit_provenance import audit_source_and_model_provenance
from .common import RESULTS_DIR, seed_everything
from .local_parity import run_local_parity
from .materialize_public_candidate import materialize_public_candidate
from .public_d1_control_parity import run_public_d1_control_parity
from .score_public_frozen_section_ce import score_public_frozen_section_ce


def main():
    t_start = time.perf_counter()
    seed_everything(2026)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("================================================================================", flush=True)
    print("STARTING PIPELINE: HUY_D1_FROZEN_SECTION_PUBLIC_V1", flush=True)
    print("================================================================================\n", flush=True)

    # Stage 1: Audit Provenance
    print("--- STAGE 1: AUDIT PROVENANCE ---", flush=True)
    prov = audit_source_and_model_provenance()
    print("Stage 1 PASSED.\n", flush=True)

    # Stage 2: Local Parity Gate (CAL600)
    print("--- STAGE 2: LOCAL PARITY GATE (CAL600) ---", flush=True)
    local_parity_report = run_local_parity()
    print("Stage 2 PASSED.\n", flush=True)

    # Stage 3: Public D1 Control Parity Gate
    print("--- STAGE 3: PUBLIC D1 CONTROL PARITY GATE ---", flush=True)
    public_control_report, public_bundle = run_public_d1_control_parity()
    print("Stage 3 PASSED.\n", flush=True)

    # Stage 4: Public Frozen Section Scoring
    print("--- STAGE 4: SCORE PUBLIC CANDIDATES WITH FROZEN JINA CE ---", flush=True)
    public_frozen_scores, pub_manifest, cache_sha, seal_time = score_public_frozen_section_ce(
        public_bundle=public_bundle, batch_size=64, force_fresh=True
    )
    print(f"Stage 4 PASSED: Sealed cache {cache_sha} at {seal_time}\n", flush=True)

    # Stages 5, 6, 7: Production Training, Churn Diagnostics, Packaging
    print("--- STAGES 5, 6, 7: MATERIALIZE CANDIDATE & PACKAGING ---", flush=True)
    submission_manifest = materialize_public_candidate(
        public_bundle=public_bundle,
        public_frozen_scores=public_frozen_scores,
        local_parity_report=local_parity_report,
        public_control_report=public_control_report,
    )
    print("Stages 5, 6, 7 PASSED.\n", flush=True)

    t_total = time.perf_counter() - t_start
    print("================================================================================", flush=True)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {t_total:.1f}s ({t_total/60:.2f} min)", flush=True)
    print(f"Final Verdict: {submission_manifest.get('final_verdict')}", flush=True)
    print(f"Candidate ZIP: {submission_manifest.get('candidate_zip_path')}", flush=True)
    print(f"Candidate ZIP SHA256: {submission_manifest.get('candidate_zip_sha256')}", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    main()
