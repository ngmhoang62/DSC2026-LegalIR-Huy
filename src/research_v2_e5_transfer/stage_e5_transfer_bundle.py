"""Stage and seal the Fold-0-only EXP-112 -> Research V2 Kaggle bundle.

This module only materializes immutable inputs and integrity receipts.  It does
not train, score, or modify any historical/production namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
LEGALIR = WORKSPACE / "LegalIR"
RESULTS = ROOT / "results" / "research_v2_e5_transfer"
AUDIT_RESULTS = ROOT / "results" / "research_v2_kaggle_audit"
DEFAULT_BUNDLE = (
    ROOT / "cache" / "research_v2_e5_transfer" / "kaggle_input"
    / "research-v2-e5-transfer-fold0-v1"
)
EXPECTED_FOLDS_SHA = "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19"
EXPECTED_POOL_SHA = "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277"
FOLD0_DUPLICATE_EXCLUSIONS = [
    "114846", "117908", "139536", "156640", "61406", "63562", "77610",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl_checked(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    if path.exists():
        if sha256(path) != sha256(temporary):
            temporary.unlink()
            raise RuntimeError(f"refusing to overwrite non-identical staged artifact: {path}")
        temporary.unlink()
    else:
        os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == source.stat().st_size and sha256(destination) == sha256(source):
            return
        raise RuntimeError(f"refusing to overwrite staged path: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def locate_model_snapshot() -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--mainguyen9--vietlegal-e5" / "snapshots"
    candidates = sorted(path for path in hub.glob("*") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one local VietLegal-E5 snapshot, got {candidates}")
    return candidates[0]


def canonical_labels() -> tuple[dict[str, set[str]], dict[str, Any]]:
    import sys
    sys.path.insert(0, str(LEGALIR / "src"))
    try:
        from exp109b_encoder_complementarity import canonical_labels as load  # type: ignore
        return load()
    finally:
        sys.path.pop(0)


def prepare(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    folds_path = ROOT / "results" / "research_v2_forensic" / "V2_FOLDS.json"
    pool_path = ROOT / "results" / "research_v2_forensic" / "V2_CANDIDATE_POOL.jsonl"
    boundary_path = ROOT / "results" / "research_v2_forensic" / "V2_BOUNDARY_GROUPS_MANIFEST.json"
    if sha256(folds_path) != EXPECTED_FOLDS_SHA or sha256(pool_path) != EXPECTED_POOL_SHA:
        raise RuntimeError("immutable V2 split or candidate-pool hash mismatch")
    folds_raw = read_json(folds_path)["folds"]
    fold_for = {str(qid): fold for fold, qids in folds_raw.items() for qid in qids}
    if len(fold_for) != 7000:
        raise RuntimeError("V2 fold partition is not exactly 7,000 queries")
    boundary = read_json(boundary_path)
    observed_exclusions = sorted(str(qid) for qid in boundary["held_fold_duplicate_exclusions"]["fold_0"])
    if observed_exclusions != sorted(FOLD0_DUPLICATE_EXCLUSIONS):
        raise RuntimeError("Fold-0 duplicate-linked exclusions drifted")

    labels, label_audit = canonical_labels()
    pool_rows = list(records(pool_path))
    pool = {str(row["qid"]): row for row in pool_rows}
    if len(pool) != 6991 or set(pool) != {qid for qid, gold in labels.items() if gold}:
        raise RuntimeError("candidate pool does not equal the canonical evaluable population")
    ordered_qids = [
        str(qid) for fold in (f"fold_{index}" for index in range(5))
        for qid in folds_raw[fold] if str(qid) in pool
    ]
    if len(ordered_qids) != 6991 or len(set(ordered_qids)) != 6991:
        raise RuntimeError("ordered evaluable V2 population mismatch")
    write_jsonl_checked(bundle / "V2_TRANSFER_QUERIES.jsonl", (
        {
            "qid": qid,
            "fold": fold_for[qid],
            "question": str(pool[qid]["query"]),
            "gold": sorted(labels[qid]),
        }
        for qid in ordered_qids
    ))
    link_or_copy(pool_path, bundle / "V2_CANDIDATE_POOL.jsonl")
    link_or_copy(folds_path, bundle / "V2_FOLDS.json")
    link_or_copy(boundary_path, bundle / "V2_BOUNDARY_GROUPS_MANIFEST.json")

    training_qids = [qid for qid in ordered_qids if fold_for[qid] != "fold_0" and qid not in FOLD0_DUPLICATE_EXCLUSIONS]
    if len(training_qids) != 5586:
        raise RuntimeError(f"unexpected Fold-0 training population: {len(training_qids)}")
    source_db = LEGALIR / "cache" / "exp112_task_adaptive_retrieval" / "sources.sqlite"
    connection = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("EXP-112 source database integrity check failed")
        source_rows: list[dict[str, Any]] = []
        for index, qid in enumerate(training_qids, 1):
            found: dict[str, list[str]] = {}
            for source, payload in connection.execute(
                "SELECT source,payload FROM sources WHERE q=?", (qid,)
            ):
                source = str(source)
                if source in {"e5", "lal", "bm25", "jina"}:
                    found[source] = [str(item["doc_id"]) for item in json.loads(payload)]
            if set(found) != {"e5", "lal", "bm25", "jina"}:
                raise RuntimeError(f"missing EXP-112 mining source for {qid}: {sorted(found)}")
            if any(len(values) != len(set(values)) for values in found.values()):
                raise RuntimeError(f"duplicate doc ids in EXP-112 source for {qid}")
            source_rows.append({"qid": qid, **found})
            if index % 500 == 0:
                print(json.dumps({"stage": "export_sources", "completed": index, "total": len(training_qids)}), flush=True)
    finally:
        connection.close()
    write_jsonl_checked(bundle / "V2_EXP112_MINER_SOURCES.jsonl", source_rows)

    fold0 = {qid for qid in ordered_qids if fold_for[qid] == "fold_0"}
    references: dict[str, dict[str, Any]] = {}
    query_id_path = LEGALIR / "cache" / "exp021_e5_dense_candidates" / "query_embeddings" / "train_query_ids.json"
    query_vector_path = LEGALIR / "cache" / "exp021_e5_dense_candidates" / "query_embeddings" / "train_queries.f32.npy"
    query_ids = [str(value) for value in read_json(query_id_path)]
    query_row = {qid: index for index, qid in enumerate(query_ids)}
    query_vectors = np.load(query_vector_path, mmap_mode="r")
    chunk_path = LEGALIR / "cache" / "e5_final_v1" / "chunk_ids.jsonl"
    positions: dict[str, list[int]] = {}
    for index, row in enumerate(records(chunk_path)):
        positions.setdefault(str(row["doc_id"]), []).append(index)
    chunk_vectors = np.load(LEGALIR / "cache" / "e5_final_v1" / "embeddings.f16.npy", mmap_mode="r")
    computed_parent_scores = 0
    for qid in ordered_qids:
        if qid not in fold0:
            continue
        docs = [str(value) for value in pool[qid]["doc_ids"]]
        query = np.array(query_vectors[query_row[qid]], dtype=np.float32, copy=True)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        available: dict[str, float] = {}
        for doc_id in docs:
            indices = positions.get(doc_id)
            if not indices:
                raise RuntimeError(f"V2 pool parent absent from frozen chunk bank: {qid}/{doc_id}")
            chunks = np.array(chunk_vectors[indices], dtype=np.float32, copy=True)
            chunks /= np.maximum(np.linalg.norm(chunks, axis=1, keepdims=True), 1e-12)
            scores_for_parent = chunks @ query
            count = min(2, len(scores_for_parent))
            available[doc_id] = float(np.partition(scores_for_parent, -count)[-count:].mean())
            computed_parent_scores += 1
        scores = [available[doc_id] for doc_id in docs]
        order = [docs[index] for index in sorted(range(len(docs)), key=lambda index: (-scores[index], docs[index]))]
        references[qid] = {"qid": qid, "doc_ids": docs, "scores": scores, "ranking": order}
    if set(references) != fold0 or len(references) != 1398:
        raise RuntimeError("Fold-0 frozen E5 reference population mismatch")
    write_jsonl_checked(bundle / "V2_FOLD0_FROZEN_REFERENCE.jsonl", (references[qid] for qid in ordered_qids if qid in fold0))

    fixed_files = {
        LEGALIR / "cache" / "e5_final_v1" / "embeddings.f16.npy": bundle / "embeddings.f16.npy",
        LEGALIR / "cache" / "e5_final_v1" / "chunk_ids.jsonl": bundle / "chunk_ids.jsonl",
        LEGALIR / "cache" / "exp021_e5_dense_candidates" / "query_embeddings" / "train_queries.f32.npy": bundle / "train_queries.f32.npy",
        LEGALIR / "cache" / "exp021_e5_dense_candidates" / "query_embeddings" / "train_query_ids.json": bundle / "train_query_ids.json",
        LEGALIR / "results" / "exp112_task_adaptive_retrieval" / "EXP112_SOURCE_SNAPSHOT_20260905.zip": bundle / "EXP112_SOURCE_SNAPSHOT_20260905.zip",
        ROOT / "src" / "research_v2_e5_transfer" / "e5_transfer_runner.py": bundle / "e5_transfer_runner.py",
        AUDIT_RESULTS / "EXP112_V2_E5_TRANSFER_AUDIT.md": bundle / "EXP112_V2_E5_TRANSFER_AUDIT.md",
        AUDIT_RESULTS / "NEXT_HYPOTHESIS_PREREGISTRATION.json": bundle / "NEXT_HYPOTHESIS_PREREGISTRATION.json",
    }
    notebook = AUDIT_RESULTS / "RESEARCH_V2_EXP112_E5_TRANSFER_FOLD0_KAGGLE.ipynb"
    requirements = AUDIT_RESULTS / "E5_TRANSFER_KAGGLE_REQUIREMENTS.txt"
    for optional in (notebook, requirements):
        if not optional.is_file():
            raise RuntimeError(f"missing staged Kaggle control file: {optional}")
        fixed_files[optional] = bundle / optional.name
    for source, destination in fixed_files.items():
        link_or_copy(source, destination)

    model_source = locate_model_snapshot()
    model_files = [path for path in model_source.rglob("*") if path.is_file()]
    if not model_files:
        raise RuntimeError("local VietLegal-E5 snapshot is empty")
    for source in model_files:
        link_or_copy(source, bundle / "vietlegal-e5" / source.relative_to(model_source))

    included = sorted(path for path in bundle.rglob("*") if path.is_file() and path.name != "E5_TRANSFER_INPUT_MANIFEST.json")
    files_sha = {path.relative_to(bundle).as_posix(): sha256(path) for path in included}
    manifest = {
        "schema_version": "dsc2026.research_v2.exp112_e5_transfer_input.v1",
        "status": "STAGED_FOR_LOCAL_PARITY",
        "scope": "FOLD_0_ONLY_NO_AUTOMATIC_FIVE_FOLD",
        "v2_folds_sha256": EXPECTED_FOLDS_SHA,
        "candidate_pool_sha256": EXPECTED_POOL_SHA,
        "query_population": 6991,
        "fold0_queries": 1398,
        "training_queries_after_duplicate_exclusion": 5586,
        "fold0_duplicate_exclusions": sorted(FOLD0_DUPLICATE_EXCLUSIONS),
        "label_policy": label_audit,
        "model_snapshot_source": str(model_source),
        "model_id": "mainguyen9/vietlegal-e5",
        "candidate_membership": "immutable E5@50 set-union novel BM25@10",
        "scientific_contract": "exact_exp112_query_only_qv_lora_top2mean_epoch2_to_v2_fixed_pool_direct_top5",
        "frozen_reference_parent_scores_computed_from_all_locked_chunks": computed_parent_scores,
        "frozen_reference_note": "Independent adapter-disabled reference; not EXP-021 depth-limited evidence aggregation.",
        "files_sha256": files_sha,
    }
    write_json(bundle / "E5_TRANSFER_INPUT_MANIFEST.json", manifest)
    return manifest


def finalize(bundle: Path, parity_dir: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest_path = bundle / "E5_TRANSFER_INPUT_MANIFEST.json"
    manifest = read_json(manifest_path)
    if manifest["status"] not in {"STAGED_FOR_LOCAL_PARITY", "SEALED_FOLD0_ONLY"}:
        raise RuntimeError("bundle is not eligible for parity finalization")
    report_paths = {
        "frozen_baseline": parity_dir / "FROZEN_BASELINE_PARITY.json",
        "mining_policy": parity_dir / "MINING_POLICY_PARITY.json",
        "smoke_resume": parity_dir / "local_smoke" / "TRAINING_SMOKE_RESUME_PARITY.json",
    }
    reports = {name: read_json(path) for name, path in report_paths.items()}
    if any(report["status"] != "PASS" for report in reports.values()):
        raise RuntimeError(f"cannot seal: a pre-GPU parity gate failed: {reports}")
    for name, source in report_paths.items():
        destination = bundle / "pre_gpu_parity" / source.name
        link_or_copy(source, destination)
    aggregate = {
        "schema_version": "dsc2026.research_v2.exp112_e5_transfer_pre_gpu_gate.v1",
        "status": "PASS_ALL_THREE_GATES",
        "gates": {
            name: {"status": report["status"], "report_sha256": sha256(report_paths[name])}
            for name, report in reports.items()
        },
        "authorizes": "one Fold-0 Kaggle pilot only",
        "does_not_authorize": ["five-fold training", "contract changes", "hyperparameter search"],
    }
    write_json(bundle / "PRE_GPU_PARITY_GATE.json", aggregate)
    manifest["status"] = "SEALED_FOLD0_ONLY"
    manifest["pre_gpu_parity"] = aggregate
    included = sorted(path for path in bundle.rglob("*") if path.is_file() and path.name != manifest_path.name)
    manifest["files_sha256"] = {path.relative_to(bundle).as_posix(): sha256(path) for path in included}
    write_json(manifest_path, manifest)
    receipt = {
        "schema_version": "dsc2026.research_v2.exp112_e5_transfer_bundle_seal.v1",
        "status": "SEALED_FOLD0_ONLY",
        "bundle": str(bundle),
        "manifest_sha256": sha256(manifest_path),
        "files": len(manifest["files_sha256"]),
        "total_bytes_excluding_manifest": sum((bundle / relative).stat().st_size for relative in manifest["files_sha256"]),
        "full_five_fold_launcher_present": False,
    }
    write_json(RESULTS / "KAGGLE_BUNDLE_SEAL.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "finalize"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--parity-dir", type=Path, default=RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(args.bundle) if args.command == "prepare" else finalize(args.bundle, args.parity_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
