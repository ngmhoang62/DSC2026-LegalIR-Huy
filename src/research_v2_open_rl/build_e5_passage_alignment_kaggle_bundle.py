"""Build the compact, hash-locked Kaggle input Dataset for 2xT4 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "research_v2_open_rl"))
import legal_mlm_query_likelihood_fold0 as common  # noqa: E402

ORIGINAL_PREREG_SHA = "10e30921bce67a6276355be5fc43cb65c0d98a84d477341a83618111c8b7c192"
RUNTIME_PREREG_SHA = "38df8087e5701b38c0b1a974928b3a51df638fe0d09e72acad3033bd82349364"
EXPECTED_QUERIES = 1398
EXPECTED_PAIRS = 73128
EXPECTED_SEQUENCES = 146249


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_locked(source: Path, target: Path, expected: str | None = None) -> None:
    if not source.is_file(): raise FileNotFoundError(source)
    observed = sha256(source)
    if expected is not None and observed != expected:
        raise RuntimeError(f"source hash mismatch: {source} {observed} != {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == source.stat().st_size and sha256(target) == observed:
            return
        raise RuntimeError(f"refusing mismatched existing bundle file: {target}")
    shutil.copy2(source, target)
    if sha256(target) != observed: raise RuntimeError(f"copy hash mismatch: {target}")


def jsonl_write(path: Path, rows) -> None:
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def source_args() -> SimpleNamespace:
    return SimpleNamespace(
        root=ROOT,
        folds=ROOT / "results/research_v2_forensic/V2_FOLDS.json",
        pool=ROOT / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl",
        contexts=ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl",
        e5_predictions=ROOT / "results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl",
        jina_predictions=ROOT / "results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl",
        sources_db=ROOT.parent / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite",
    )


def render_evaluation(target: Path) -> dict:
    args = source_args(); rows = common.fold0_rows(args); documents = common.load_documents(args, rows)
    counters = {"queries": 0, "pairs": 0, "sequences": 0}; started = time.perf_counter()
    def output_rows():
        for row in rows:
            _, owners, passages = common.evidence_for_row(args, row, documents)
            if len(owners) != len(passages): raise RuntimeError("owner/passage mismatch")
            payload = {
                "qid": str(row["qid"]), "query": str(row["query"]),
                "doc_ids": list(map(str, row["doc_ids"])),
                "passages": [{"doc_id": str(doc), "passage_index": int(index), "text": str(text)}
                             for (doc, index), text in zip(owners, passages)],
            }
            counters["queries"] += 1; counters["pairs"] += len(payload["doc_ids"]); counters["sequences"] += len(payload["passages"])
            if counters["queries"] % 50 == 0:
                print(f"rendered={counters['queries']}/{EXPECTED_QUERIES}", flush=True)
            yield payload
    jsonl_write(target, output_rows())
    # Recount existing or newly generated data rather than trusting generator state.
    recount = {"queries": 0, "pairs": 0, "sequences": 0}
    for row in common.read_jsonl(target):
        recount["queries"] += 1; recount["pairs"] += len(row["doc_ids"]); recount["sequences"] += len(row["passages"])
    if recount != {"queries": EXPECTED_QUERIES, "pairs": EXPECTED_PAIRS, "sequences": EXPECTED_SEQUENCES}:
        raise RuntimeError(f"rendered evaluation cardinality: {recount}")
    return {**recount, "sha256": sha256(target), "bytes": target.stat().st_size,
            "runtime_seconds_this_builder": time.perf_counter() - started}


def render_clean_union(target: Path) -> dict:
    args = source_args(); rows = common.fold0_rows(args)
    pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    clean = common.load_clean_sets(args, pool)
    jsonl_write(target, ({"qid": qid, "top5_union": sorted(clean[qid])}
                         for qid in sorted(clean, key=int)))
    loaded = {str(row["qid"]): set(map(str, row["top5_union"])) for row in common.read_jsonl(target)}
    if set(loaded) != set(pool) or any(not values <= pool[qid] for qid, values in loaded.items()):
        raise RuntimeError("clean union compact artifact mismatch")
    return {"queries": len(loaded), "sha256": sha256(target), "bytes": target.stat().st_size,
            "metric_read": False}


def build(args) -> None:
    out = args.output.resolve()
    if (out / "BUNDLE_MANIFEST.json").exists():
        raise RuntimeError("sealed bundle already exists; refusing overwrite")
    files = {
        ROOT / "results/research_v2_open_rl/E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREREGISTRATION.json": out / "preregistration/E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREREGISTRATION.json",
        ROOT / "results/research_v2_open_rl/E5_PASSAGE_ALIGNMENT_2XT4_RUNTIME_PREREGISTRATION.json": out / "preregistration/E5_PASSAGE_ALIGNMENT_2XT4_RUNTIME_PREREGISTRATION.json",
        ROOT / "results/research_v2_forensic/V2_FOLDS.json": out / "data/V2_FOLDS.json",
        ROOT / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl": out / "data/V2_CANDIDATE_POOL.jsonl",
        ROOT / "results/research_v2_forensic/V2_BOUNDARY_GROUPS.jsonl": out / "data/V2_BOUNDARY_GROUPS.jsonl",
        ROOT / "results/research_v2_forensic/V2_BOUNDARY_GROUPS_MANIFEST.json": out / "data/V2_BOUNDARY_GROUPS_MANIFEST.json",
        ROOT / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl": out / "data/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl",
        ROOT / "cache/research_v2_open_rl/e5_passage_alignment_top2mean_fold0/adapted_query_vectors.f32.npy": out / "cache/adapted_query_vectors.f32.npy",
        ROOT / "cache/research_v2_open_rl/e5_passage_alignment_top2mean_fold0/query_ids.json": out / "cache/query_ids.json",
        ROOT / "results/research_v2_open_rl/E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_QUERY_CACHE_MANIFEST.json": out / "cache/QUERY_CACHE_MANIFEST.json",
        ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt": out / "model/fold0_query_adapter_epoch2.pt",
        ROOT / "src/research_v2_open_rl/kaggle_e5_passage_alignment_2xt4.py": out / "code/kaggle_e5_passage_alignment_2xt4.py",
        ROOT / "src/research_v2_open_rl/kaggle_e5_passage_alignment_requirements.txt": out / "code/requirements.txt",
        ROOT / "results/research_v2_open_rl/E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_CLOSURE_AUDIT.json": out / "audit/LOCAL_COST_CLOSURE.json",
        ROOT / "results/research_v2_open_rl/E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_INPUT_MANIFEST.json": out / "audit/ORIGINAL_INPUT_MANIFEST.json",
        ROOT / "results/research_v2_open_rl/E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_SMOKE.json": out / "audit/ORIGINAL_SMOKE.json",
    }
    for source, target in files.items():
        expected = ORIGINAL_PREREG_SHA if source.name == "E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREREGISTRATION.json" else RUNTIME_PREREG_SHA if source.name == "E5_PASSAGE_ALIGNMENT_2XT4_RUNTIME_PREREGISTRATION.json" else None
        copy_locked(source, target, expected)
        print(f"copied {target.relative_to(out)}", flush=True)
    model_source = ROOT / "cache/research_v2_e5_confirmation/bundle-v1/vietlegal-e5"
    for source in sorted(item for item in model_source.rglob("*") if item.is_file()):
        copy_locked(source, out / "model/vietlegal-e5" / source.relative_to(model_source))
    rendered = render_evaluation(out / "data/FOLD0_EVAL_RENDERED_TOP2.jsonl")
    clean = render_clean_union(out / "data/FOLD0_CLEAN_EXPERT_UNION.jsonl")
    manifest_files = {}
    for path in sorted(item for item in out.rglob("*") if item.is_file() and item.name != "BUNDLE_MANIFEST.json"):
        relative = path.relative_to(out).as_posix()
        manifest_files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": "dsc2026.research_v2.e5_passage_alignment_2xt4_kaggle_input_bundle.v1",
        "status": "SEALED_KAGGLE_INPUT_BUNDLE", "created_date": "2026-09-13",
        "files": manifest_files, "file_count": len(manifest_files),
        "total_bytes": sum(item["bytes"] for item in manifest_files.values()),
        "rendered_evaluation": rendered, "clean_union": clean,
        "source_database_included": False,
        "source_database_exclusion_reason": "exact required clean union was compacted; database is not used for training/scoring",
        "upload_instruction": "Upload this directory as one private Kaggle Dataset and attach it to the supplied notebook.",
        "held_fold_metric_read_during_build": False,
    }
    write_json(out / "BUNDLE_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "path": str(out),
                      "files": manifest["file_count"], "total_bytes": manifest["total_bytes"],
                      "manifest_sha256": sha256(out / "BUNDLE_MANIFEST.json")}, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "results/research_v2_open_rl/kaggle_bundle_e5_passage_alignment_2xt4_v1")
    return p


if __name__ == "__main__":
    build(parser().parse_args())
