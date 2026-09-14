"""Independent structural/hash audit for the final Huy-fasttrack ZIP."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/huy_fasttrack/submission_candidate_with_huy_jina"
PUBLIC = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json"
CANONICAL = ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
JINA_DB = ROOT / "cache/huy_fasttrack/public_frozen_jina_scores.sqlite"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(path):
    path = Path(path)
    h = hashlib.sha256()
    for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha(child)))
    return h.hexdigest()


def main():
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    with CANONICAL.open("r", encoding="utf-8") as handle:
        canonical = {str(json.loads(line)["doc_id"]) for line in handle if line.strip()}
    submission_path = OUT / "submission.json"
    zip_path = OUT / "submission.zip"
    payload = json.loads(submission_path.read_text(encoding="utf-8"))
    structural = (
        len(canonical) == 8507
        and set(payload) == set(public)
        and len(payload) == 1000
        and all(
            isinstance(row, dict)
            and len(row.get("answer", [])) == 5
            and len(set(map(str, row["answer"]))) == 5
            and set(map(str, row["answer"])) <= canonical
            for row in payload.values()
        )
    )
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        zip_exact = names == ["submission.json"] and archive.read("submission.json") == submission_path.read_bytes()
        zip_test = archive.testzip()
    db = sqlite3.connect(f"file:{JINA_DB.as_posix()}?mode=ro", uri=True)
    jina_integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    jina_scores = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid) FROM scores").fetchone()
    jina_progress = db.execute("SELECT COUNT(*),SUM(pairs),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    db.close()
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest_failures = []
    for group in ("inputs", "outputs"):
        for path, expected in manifest[group].items():
            if path.endswith("/**"):
                root = Path(path[:-3])
                actual = tree_sha(root) if root.exists() else None
            else:
                actual = sha(path) if Path(path).exists() else None
            if actual != expected:
                manifest_failures.append(path)
    report = json.loads((OUT / "REPORT.json").read_text(encoding="utf-8"))
    audit = {
        "schema_version": "dsc2026.huy_fasttrack.public_candidate_audit.v1",
        "status": "PASS" if structural and zip_exact and zip_test is None and jina_integrity == "ok" and jina_scores == (50000, 1000) and not manifest_failures else "FAIL",
        "submission": {
            "queries": len(payload), "canonical_parents": len(canonical),
            "answers_per_query": 5, "all_unique_and_canonical": structural,
            "zip_exact": zip_exact, "zip_test": zip_test,
            "submission_json_sha256": sha(submission_path),
            "submission_zip_sha256": sha(zip_path),
        },
        "frozen_jina": {
            "integrity": jina_integrity, "scores_and_qids": list(jina_scores),
            "progress_queries_pairs_seconds_peak_mib": list(jina_progress),
            "database_sha256": sha(JINA_DB),
        },
        "strict_oof_support": report["strict_oof_support"],
        "parameter_limit": report["system"],
        "manifest_failures": manifest_failures,
    }
    (OUT / "FINAL_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
