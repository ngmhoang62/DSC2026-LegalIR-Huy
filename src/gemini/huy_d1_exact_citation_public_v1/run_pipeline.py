"""Master pipeline orchestrator for HUY_D1_EXACT_CITATION_PUBLIC_V1."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_exact_citation_public_v1.common import (
    PUBLIC_CONTEXTS_DIR,
    RESULTS_DIR,
    get_git_status,
    get_source_files_sha256,
    seed_everything,
)
from src.gemini.huy_d1_exact_citation_public_v1.materialize_candidate import (
    process_candidate_and_churn,
)
from src.gemini.huy_d1_exact_citation_public_v1.public_citation_audit import (
    build_public_citation_index,
    run_public_citation_audit,
)
from src.gemini.huy_d1_exact_citation_public_v1.public_d1_control_parity import (
    run_public_d1_control_parity,
)


def run_pipeline() -> Dict[str, Any]:
    print("==================================================================", flush=True)
    print("STARTING EXPERIMENT: HUY_D1_EXACT_CITATION_PUBLIC_V1", flush=True)
    print("==================================================================", flush=True)

    start_time_utc = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)

    # 1. HARD PROVENANCE GATE
    print("=== STAGE 1: HARD PROVENANCE GATE ===", flush=True)
    git_info = get_git_status()
    source_files = get_source_files_sha256()

    print(f"Git HEAD Commit:         {git_info.get('head_commit')}", flush=True)
    print(f"Git Origin/Main Commit:  {git_info.get('origin_main_commit')}", flush=True)
    print(f"Remote Parity:           {git_info.get('parity')}", flush=True)
    print(f"Working Tree Clean:      {git_info.get('status_clean')}", flush=True)

    provenance_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.source_provenance.v1",
        "experiment_id": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
        "timestamp_utc": start_time_utc,
        "git": git_info,
        "source_files": source_files,
    }
    prov_path = RESULTS_DIR / "SOURCE_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {prov_path}", flush=True)

    if not git_info.get("parity") or not git_info.get("status_clean"):
        print("\nBLOCKED_SOURCE_PROVENANCE: HEAD != origin/main or working tree is not clean! STOPPING.", flush=True)
        raise RuntimeError("BLOCKED_SOURCE_PROVENANCE")

    # 2. PUBLIC D1 CONTROL PARITY AUDIT
    parity_doc, preds_d1, scores_d1, public_candidates, docs_store, public_meta, public_ids = run_public_d1_control_parity()

    if not parity_doc.get("control_parity_passed"):
        print(f"\nBLOCKED_PUBLIC_D1_PARITY: Parity matches {parity_doc.get('exact_matches_count')}/1000. STOPPING.", flush=True)
        raise RuntimeError("BLOCKED_PUBLIC_D1_PARITY")

    # 3. PUBLIC CITATION INDEXING
    ref_to_docs, doc_to_own_ref, index_manifest = build_public_citation_index(PUBLIC_CONTEXTS_DIR)

    # 4. LABEL-FREE PUBLIC CITATION AUDIT
    audit_summary, actions_doc, actions_dict = run_public_citation_audit(
        public_ids=public_ids,
        public_meta=public_meta,
        d1_preds=preds_d1,
        public_candidates=public_candidates,
        ref_to_docs=ref_to_docs,
        doc_to_own_ref=doc_to_own_ref,
        docs_store=docs_store,
    )

    # 5. PUBLIC ACTION SAFETY GATE & CANDIDATE MATERIALIZATION
    churn_report, final_verdict = process_candidate_and_churn(
        public_ids=public_ids,
        d1_preds=preds_d1,
        actions_dict=actions_dict,
        audit_summary=audit_summary,
        parity_doc=parity_doc,
        index_manifest=index_manifest,
        git_info=git_info,
    )

    print("\n==================================================================", flush=True)
    print(f"EXPERIMENT COMPLETE. FINAL VERDICT: {final_verdict}", flush=True)
    print("==================================================================", flush=True)

    return {
        "final_verdict": final_verdict,
        "parity_doc": parity_doc,
        "audit_summary": audit_summary,
        "churn_report": churn_report,
    }


if __name__ == "__main__":
    run_pipeline()
