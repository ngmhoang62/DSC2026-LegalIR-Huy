"""Stage an all-query miner-source bundle for strict E5 confirmation.

The Fold-0 Kaggle bundle intentionally exported miner sources only for its
5,586 training qids.  Confirmation folds need the same four EXP-112 source
lists for qids that were held out by Fold-0.  This script exports those rows
from the same read-only EXP-112 SQLite database and hard-links every other
sealed input without recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable


SOURCE_KEYS = ("e5", "lal", "bm25", "jina")


def records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ) + "\n")
    os.replace(temporary, path)


def link_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256(source) != sha256(target):
            raise RuntimeError(f"existing target differs: {target}")
        return
    os.link(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--target-bundle", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_bundle.resolve()
    target = args.target_bundle.resolve()
    target.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(source / "E5_TRANSFER_INPUT_MANIFEST.json")
    if sha256(source / "E5_TRANSFER_INPUT_MANIFEST.json") != (
        "44957ab7c836135fafd949b281140b89e8e01dae6dc3daa95f2e9b5b1e8f26d9"
    ):
        raise RuntimeError("source bundle manifest is not the sealed Fold-0 input")

    linked = [
        "V2_TRANSFER_QUERIES.jsonl",
        "V2_CANDIDATE_POOL.jsonl",
        "V2_FOLDS.json",
        "V2_BOUNDARY_GROUPS_MANIFEST.json",
        "V2_FOLD0_FROZEN_REFERENCE.jsonl",
        "chunk_ids.jsonl",
        "embeddings.f16.npy",
        "train_query_ids.json",
        "train_queries.f32.npy",
    ]
    linked.extend(
        str(path.relative_to(source)).replace("\\", "/")
        for path in sorted((source / "vietlegal-e5").rglob("*")) if path.is_file()
    )
    for relative in linked:
        link_verified(source / relative, target / relative)

    query_rows = list(records(source / "V2_TRANSFER_QUERIES.jsonl"))
    ordered_qids = [str(row["qid"]) for row in query_rows]
    if len(ordered_qids) != 6991 or len(set(ordered_qids)) != 6991:
        raise RuntimeError("unexpected evaluable query population")
    old_rows = {
        str(row["qid"]): {key: [str(value) for value in row[key]] for key in SOURCE_KEYS}
        for row in records(source / "V2_EXP112_MINER_SOURCES.jsonl")
    }
    if len(old_rows) != 5586:
        raise RuntimeError("Fold-0 miner-source snapshot row count drifted")

    connection = sqlite3.connect(f"file:{args.source_db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("EXP-112 source database integrity check failed")
        exported: list[dict[str, Any]] = []
        for index, qid in enumerate(ordered_qids, 1):
            found: dict[str, list[str]] = {}
            for source_name, payload in connection.execute(
                "SELECT source,payload FROM sources WHERE q=?", (qid,)
            ):
                source_name = str(source_name)
                if source_name in SOURCE_KEYS:
                    found[source_name] = [str(item["doc_id"]) for item in json.loads(payload)]
            if set(found) != set(SOURCE_KEYS):
                raise RuntimeError(f"missing source for {qid}: {sorted(found)}")
            if any(len(values) != len(set(values)) for values in found.values()):
                raise RuntimeError(f"duplicate source doc ids for {qid}")
            if qid in old_rows and found != old_rows[qid]:
                raise RuntimeError(f"existing Fold-0 source row changed: {qid}")
            exported.append({"qid": qid, **found})
            if index % 500 == 0:
                print(json.dumps({"stage": "export_sources", "completed": index, "total": 6991}))
    finally:
        connection.close()

    source_path = target / "V2_EXP112_MINER_SOURCES.jsonl"
    write_jsonl(source_path, exported)
    file_hashes = {relative: sha256(target / relative) for relative in linked}
    file_hashes["V2_EXP112_MINER_SOURCES.jsonl"] = sha256(source_path)
    boundary = read_json(target / "V2_BOUNDARY_GROUPS_MANIFEST.json")
    manifest = {
        "schema_version": "dsc2026.research_v2.exp112_e5_confirmation_input.v1",
        "status": "SEALED_FOLD0_ONLY",
        "status_compatibility_note": "Legacy status token required by the imported sealed core; actual scope is strict held-fold confirmation.",
        "scope": "FOLDS_1_4_STRICT_CONFIRMATION_ONE_FOLD_PER_INVOCATION",
        "scientific_contract": source_manifest["scientific_contract"],
        "model_id": source_manifest["model_id"],
        "query_population": 6991,
        "fold0_queries": 1398,
        "training_queries_after_duplicate_exclusion": source_manifest[
            "training_queries_after_duplicate_exclusion"
        ],
        "fold0_duplicate_exclusions": source_manifest["fold0_duplicate_exclusions"],
        "held_fold_duplicate_exclusions": boundary["held_fold_duplicate_exclusions"],
        "candidate_membership": source_manifest["candidate_membership"],
        "candidate_pool_sha256": source_manifest["candidate_pool_sha256"],
        "v2_folds_sha256": source_manifest["v2_folds_sha256"],
        "miner_source_rows": 6991,
        "miner_source_origin": str(args.source_db.resolve()),
        "miner_source_origin_integrity": "ok",
        "original_5586_rows_exact": True,
        "source_fold0_manifest_sha256": sha256(source / "E5_TRANSFER_INPUT_MANIFEST.json"),
        "files_sha256": file_hashes,
    }
    manifest_path = target / "E5_TRANSFER_INPUT_MANIFEST.json"
    write_json(manifest_path, manifest)
    report = {
        "schema_version": "dsc2026.research_v2.e5_confirmation_bundle_seal.v1",
        "status": "PASS",
        "bundle": str(target),
        "manifest_sha256": sha256(manifest_path),
        "files": len(file_hashes),
        "miner_source_rows": len(exported),
        "miner_source_sha256": file_hashes["V2_EXP112_MINER_SOURCES.jsonl"],
        "original_5586_rows_exact": True,
        "source_db_integrity": "ok",
        "hard_linked_files": len(linked),
    }
    write_json(target / "CONFIRMATION_BUNDLE_SEAL.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
