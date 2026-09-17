"""Candidate Materialization, Submission Packaging, and Churn Audit for HUY_D1_EXACT_CITATION_PUBLIC_V1."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.gemini.huy_d1_exact_citation_public_v1.common import (
    D1_CHAMPION_JSON_PATH,
    D1_CHAMPION_ZIP_PATH,
    FROZEN_CAL_EVIDENCE,
    FROZEN_PRIOR_SOURCE_SHA,
    FROZEN_STRICTV2_EVIDENCE,
    PUBLIC_DATASET_DIR,
    PUBLIC_CONTEXTS_DIR,
    RESULTS_DIR,
    ROOT,
    SRC_DIR,
    get_git_status,
    sha256_file,
)


def validate_submission_zip(zip_path: Path, json_path: Path, valid_doc_ids: Set[str]) -> None:
    """Validate that candidate submission ZIP strictly conforms to format requirements."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing submission ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names != ["submission.json"]:
            raise ValueError(f"ZIP must contain ONLY submission.json, got: {names}")
        content = zf.read("submission.json").decode("utf-8")
        parsed_zip = json.loads(content)

    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))

    if parsed_zip != parsed_json:
        raise ValueError("submission.json inside ZIP does not logically match candidate JSON file!")

    if len(parsed_zip) != 1000:
        raise ValueError(f"Expected 1000 queries in submission.json, got {len(parsed_zip)}")

    for q, item in parsed_zip.items():
        if "answer" not in item:
            raise ValueError(f"Query {q} missing 'answer' key")
        ans = item["answer"]
        if len(ans) != 5:
            raise ValueError(f"Query {q} does not have exactly 5 predictions: {len(ans)}")
        if len(set(ans)) != 5:
            raise ValueError(f"Query {q} contains duplicates: {ans}")
        for doc_id in ans:
            if doc_id not in valid_doc_ids:
                raise ValueError(f"Query {q} contains invalid document ID: {doc_id}")


def process_candidate_and_churn(
    public_ids: List[str],
    d1_preds: Dict[str, List[str]],
    actions_dict: Dict[str, Any],
    audit_summary: Dict[str, Any],
    parity_doc: Dict[str, Any],
    index_manifest: Dict[str, Any],
    git_info: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Execute action safety gate, materialize candidate (if 1 <= N <= 10), and audit churn."""
    print("=== EXECUTING PUBLIC ACTION SAFETY GATE ===", flush=True)

    N = len(actions_dict)
    print(f"Number of exact citation injection actions (N): {N}", flush=True)

    if N == 0:
        final_verdict = "NO_PUBLIC_ACTIONS"
        print("Verdict: NO_PUBLIC_ACTIONS (Candidate identical to D1). No packaging.", flush=True)
        churn_report = {
            "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.prediction_churn.v1",
            "experiment_id": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
            "status": "NO_ACTIONS",
            "actions_count": 0,
            "verdict": final_verdict,
        }
        (RESULTS_DIR / "PUBLIC_PREDICTION_CHURN.json").write_text(
            json.dumps(churn_report, indent=2), encoding="utf-8"
        )
        return churn_report, final_verdict

    elif N > 10:
        final_verdict = "BLOCKED_PUBLIC_INTERVENTION_SHIFT"
        print(f"Verdict: BLOCKED_PUBLIC_INTERVENTION_SHIFT (N={N} > 10). Safety cap exceeded.", flush=True)
        churn_report = {
            "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.prediction_churn.v1",
            "experiment_id": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
            "status": "BLOCKED_INTERVENTION_SHIFT",
            "actions_count": N,
            "verdict": final_verdict,
        }
        (RESULTS_DIR / "PUBLIC_PREDICTION_CHURN.json").write_text(
            json.dumps(churn_report, indent=2), encoding="utf-8"
        )
        return churn_report, final_verdict

    # 1 <= N <= 10: Proceed to candidate packaging
    final_verdict = "READY_FOR_EXTERNAL_AUDIT"
    print(f"Verdict: READY_FOR_EXTERNAL_AUDIT (1 <= N={N} <= 10). Packaging candidate...", flush=True)

    # 1. Build Candidate Predictions
    candidate_preds: Dict[str, List[str]] = {}
    for q in public_ids:
        if q in actions_dict:
            candidate_preds[q] = list(actions_dict[q]["d1_top5_after"])
        else:
            candidate_preds[q] = list(d1_preds[q])

    # 2. Churn Audit vs Exact Public D1
    print("Computing prediction churn vs public D1...", flush=True)
    ordered_changes = sum(1 for q in public_ids if candidate_preds[q] != d1_preds[q])
    set_changes = sum(1 for q in public_ids if set(candidate_preds[q]) != set(d1_preds[q]))

    overlap_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    single_doc_changes = 0
    multi_doc_changes = 0
    boundary_diagnostics = []

    for q in public_ids:
        d1_set = set(d1_preds[q])
        cand_set = set(candidate_preds[q])
        overlap = len(d1_set & cand_set)
        overlap_hist[overlap] += 1

        entering = list(cand_set - d1_set)
        leaving = list(d1_set - cand_set)

        if len(entering) == 1:
            single_doc_changes += 1
        elif len(entering) > 1:
            multi_doc_changes += 1

        if entering or leaving:
            # Check ranks 1-4 preservation
            ranks_1_to_4_preserved = (candidate_preds[q][:4] == d1_preds[q][:4])
            boundary_diagnostics.append({
                "qid": q,
                "anchor_doc_entered": entering[0] if entering else None,
                "defender_doc_left": leaving[0] if leaving else None,
                "d1_top5": d1_preds[q],
                "candidate_top5": candidate_preds[q],
                "ranks_1_to_4_preserved": ranks_1_to_4_preserved,
            })

    # Integrity assertion
    if set_changes != N:
        raise RuntimeError(f"BLOCKED_CANDIDATE_INTEGRITY: set_changes ({set_changes}) != N ({N})")
    if single_doc_changes != N or multi_doc_changes != 0:
        raise RuntimeError("BLOCKED_CANDIDATE_INTEGRITY: not all changed queries have exactly 1 change!")
    if not all(b["ranks_1_to_4_preserved"] for b in boundary_diagnostics):
        raise RuntimeError("BLOCKED_CANDIDATE_INTEGRITY: ranks 1-4 were not strictly preserved on all changed queries!")

    churn_report = {
        "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.prediction_churn.v1",
        "experiment_id": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
        "verdict": final_verdict,
        "total_queries": len(public_ids),
        "ordered_top5_changes": ordered_changes,
        "set_top5_changes": set_changes,
        "overlap_histogram": overlap_hist,
        "queries_changed_1_doc": single_doc_changes,
        "queries_changed_gt1_doc": multi_doc_changes,
        "integrity_verified": True,
        "boundary_diagnostics": boundary_diagnostics,
    }

    churn_path = RESULTS_DIR / "PUBLIC_PREDICTION_CHURN.json"
    churn_path.write_text(json.dumps(churn_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {churn_path}", flush=True)

    # 3. Materialize Submission JSON and ZIP
    candidate_json_path = RESULTS_DIR / "CANDIDATE_D1_EXACT_CITATION.json"
    candidate_zip_path = RESULTS_DIR / "CANDIDATE_D1_EXACT_CITATION.zip"

    submission_payload = {q: {"answer": candidate_preds[q]} for q in public_ids}
    submission_bytes = json.dumps(submission_payload, indent=2).encode("utf-8")
    candidate_json_path.write_bytes(submission_bytes)
    print(f"Wrote {candidate_json_path}", flush=True)

    with zipfile.ZipFile(candidate_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", submission_bytes)
    print(f"Wrote {candidate_zip_path}", flush=True)

    # Validate ZIP
    valid_docs = set(p.stem[len("context_") :] for p in PUBLIC_CONTEXTS_DIR.glob("context_*.json"))
    validate_submission_zip(candidate_zip_path, candidate_json_path, valid_docs)
    cand_zip_sha = sha256_file(candidate_zip_path)
    cand_json_sha = sha256_file(candidate_json_path)
    print(f"Validated ZIP successfully! SHA256: {cand_zip_sha}", flush=True)

    # 4. Build SUBMISSION_MANIFEST.json
    manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.submission_manifest.v1",
        "experiment_name": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_commit": git_info.get("head_commit"),
        "source_origin_commit": git_info.get("origin_main_commit"),
        "baseline_d1_champion": {
            "source_provenance": "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json",
            "json_path": str(D1_CHAMPION_JSON_PATH.relative_to(ROOT)).replace("\\", "/"),
            "json_sha256": sha256_file(D1_CHAMPION_JSON_PATH),
            "zip_path": str(D1_CHAMPION_ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "zip_sha256": sha256_file(D1_CHAMPION_ZIP_PATH),
            "public_control_parity_exact_matches": parity_doc["exact_matches_count"],
        },
        "citation_rule_provenance": {
            "frozen_prior_experiment": "HUY_D1_SELECTIVE_REPAIR_V1",
            "frozen_source_commit": FROZEN_PRIOR_SOURCE_SHA,
            "cal_evidence": FROZEN_CAL_EVIDENCE,
            "strict_v2_evidence": FROZEN_STRICTV2_EVIDENCE,
        },
        "public_fingerprints": {
            "public_query_fingerprint": parity_doc["public_query_fingerprint"],
            "public_candidate_pool_fingerprint": parity_doc["public_candidate_pool_fingerprint"],
        },
        "candidate_artifacts": {
            "candidate_json_path": str(candidate_json_path.relative_to(ROOT)).replace("\\", "/"),
            "candidate_json_sha256": cand_json_sha,
            "candidate_zip_path": str(candidate_zip_path.relative_to(ROOT)).replace("\\", "/"),
            "candidate_zip_sha256": cand_zip_sha,
        },
        "public_citation_statistics": {
            "parsed_references_queries_count": audit_summary["queries_with_parsed_legal_references"],
            "unique_exact_matches_count": audit_summary["unique_exact_matches"],
            "already_in_d1_top5_count": audit_summary["already_in_d1_top5_anchors"],
            "interventions_count": N,
            "ambiguous_abstentions_count": audit_summary["ambiguous_document_matches"],
            "multiple_eligible_abstentions_count": audit_summary["multiple_eligible_document_abstentions"],
            "unmatched_references_count": audit_summary["unmatched_references"],
        },
        "interventions_detail": [
            {
                "qid": q,
                "query_text": actions_dict[q]["query_text"],
                "anchor_doc_id": actions_dict[q]["anchor_doc_id"],
                "anchor_title": actions_dict[q]["anchor_title"],
                "defender_doc_id": actions_dict[q]["defender_doc_id"],
                "defender_title": actions_dict[q]["defender_title"],
            }
            for q in sorted(actions_dict.keys())
        ],
        "zip_validation": {
            "validated": True,
            "total_queries": 1000,
            "predictions_per_query": 5,
            "all_doc_ids_valid": True,
            "no_duplicates": True,
            "byte_match_submission_json": True,
        },
        "final_verdict": final_verdict,
        "submission_instruction": "DO NOT SUBMIT TO CODABENCH. Candidate materialized for external audit only.",
    }

    manifest_path = RESULTS_DIR / "SUBMISSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)

    return churn_report, final_verdict
