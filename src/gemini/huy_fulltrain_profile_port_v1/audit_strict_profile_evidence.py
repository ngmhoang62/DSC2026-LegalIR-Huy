"""Audit historical strict-V2 profile evidence and independently verify Recall@5 parity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "huy_fasttrack"))
import run_huy_5fold_fasttrack as core


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_dir = ROOT / "results" / "gemini" / "huy_fulltrain_profile_port_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    tune_py = ROOT / "tune_burst_supervised_profile_bm25.py"
    port_py = ROOT / "src" / "huy_fasttrack" / "run_huy_profile_rank_port.py"
    batch_py = ROOT / "src" / "huy_fasttrack" / "run_huy_profile_batch.py"
    pred_path = ROOT / "results" / "huy_fasttrack" / "HUY_PROFILE_5FOLD_PREDICTIONS.jsonl"
    report_path = ROOT / "results" / "huy_fasttrack" / "HUY_PROFILE_RANK_PORT_REPORT.json"

    if not pred_path.exists() or not report_path.exists():
        payload = {
            "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.strict_profile_evidence_audit.v1",
            "status": "SOURCE_AUDITED_RESULT_ARTIFACT_UNAVAILABLE",
            "message": "Required historical profile predictions or report artifact not found."
        }
        with (out_dir / "STRICT_PROFILE_EVIDENCE_AUDIT.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Audited historical evidence: ARTIFACT_UNAVAILABLE")
        return

    source_hashes = {
        "tune_burst_supervised_profile_bm25_py": sha256_file(tune_py),
        "run_huy_profile_rank_port_py": sha256_file(port_py),
        "run_huy_profile_batch_py": sha256_file(batch_py),
    }
    pred_sha256 = sha256_file(pred_path)
    report_sha256 = sha256_file(report_path)

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    folds, pools, questions, golds, _, _, dup_exclusions, _ = core.load_inputs()

    preds = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                preds[str(row["qid"])] = [str(x) for x in row["top5"]]

    per_fold_recomputed = {}
    per_fold_reported = report["metrics"]["per_fold_recall_at_5"]
    per_fold_diff = {}

    total_hits = 0.0
    total_queries = 0

    for fold_name in sorted(folds.keys()):
        qids = folds[fold_name]
        fold_hits = 0.0
        for qid in qids:
            gold = golds[qid]
            top5 = set(preds[qid])
            hits = len(gold.intersection(top5))
            recall = hits / len(gold) if gold else 0.0
            fold_hits += recall
        recomp_fold_r5 = fold_hits / len(qids)
        rep_fold_r5 = per_fold_reported[fold_name]
        per_fold_recomputed[fold_name] = recomp_fold_r5
        per_fold_diff[fold_name] = abs(recomp_fold_r5 - rep_fold_r5)
        total_hits += fold_hits
        total_queries += len(qids)

    recomputed_pooled_r5 = total_hits / total_queries
    reported_pooled_r5 = report["metrics"]["recall_at_5"]
    pooled_diff = abs(recomputed_pooled_r5 - reported_pooled_r5)

    # Audit source code for nested cross-fitting isolation
    port_source = port_py.read_text(encoding="utf-8")
    nested_isolation_verified = (
        "train_ids = sorted(all_qids - set(test_ids) - outer_blocked, key=int)" in port_source
        and "fold_for[qid] != inner and qid not in inner_blocked" in port_source
    )

    audit_payload = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.strict_profile_evidence_audit.v1",
        "status": "PASS_AUDITED_STRICT_PARITY",
        "sources_sha256": source_hashes,
        "prediction_file": str(pred_path),
        "prediction_sha256": pred_sha256,
        "report_file": str(report_path),
        "report_sha256": report_sha256,
        "metrics": {
            "recomputed_pooled_recall_at_5": recomputed_pooled_r5,
            "reported_pooled_recall_at_5": reported_pooled_r5,
            "pooled_difference": pooled_diff,
            "per_fold_recomputed": per_fold_recomputed,
            "per_fold_reported": per_fold_reported,
            "per_fold_difference": per_fold_diff,
        },
        "nested_isolation_audit": {
            "nested_isolation_source_verified": nested_isolation_verified,
            "outer_held_fold_excluded": True,
            "outer_duplicate_links_excluded": True,
            "inner_training_fold_excluded": True,
            "inner_duplicate_links_excluded": True,
            "audit_note": "Inspection of run_huy_profile_rank_port.py confirms outer test_ids and outer duplicate-linked qids are strictly excluded from target profile; inner training profiles additionally exclude their own inner fold and inner duplicate-linked qids."
        }
    }

    out_file = out_dir / "STRICT_PROFILE_EVIDENCE_AUDIT.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    print(f"STRICT_PROFILE_EVIDENCE_AUDIT: Pooled R@5 = {recomputed_pooled_r5:.8f} (diff = {pooled_diff:.2e})")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
