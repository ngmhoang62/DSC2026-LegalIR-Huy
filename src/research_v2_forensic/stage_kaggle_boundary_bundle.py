"""Stage a hash-locked private Kaggle Dataset without duplicating large files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) == sha256(source):
            return
        raise RuntimeError(f"refuse overwrite nonmatching staged file: {target}")
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    small = [
        args.results / "V2_BOUNDARY_GROUPS.jsonl",
        args.results / "V2_BOUNDARY_GROUPS_MANIFEST.json",
        args.results / "V2_CANDIDATE_POOL.jsonl",
        args.results / "V2_FOLDS.json",
        args.results / "EVIDENCE_RENDERER_LOCK.json",
        args.results / "EVIDENCE_CONTRACT_MATCHED_AB.json",
        args.results / "RESEARCH_V2_JINA_BOUNDARY_KAGGLE.ipynb",
        args.results / "KAGGLE_REQUIREMENTS.txt",
        Path(__file__).with_name("jina_v2_boundary_train.py"),
        args.cache / "evidence_ab_scores.sqlite",
    ]
    for source in small:
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source.resolve(), args.output / source.name)
    context_pack = args.output / "V2_CONTEXTS.jsonl"
    if not context_pack.exists():
        with context_pack.open("w", encoding="utf-8", newline="\n") as sink:
            for source in sorted(args.contexts.glob("context_*.json")):
                row = json.loads(source.read_text(encoding="utf-8"))
                passage = str(row.get("passage") or "")
                if not passage:
                    raise RuntimeError(f"empty retained canonical parent: {source}")
                sink.write(json.dumps({"doc_id": str(row["id"]), "passage": passage},
                                      ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")) + "\n")
    if sum(1 for _ in context_pack.open("r", encoding="utf-8")) != 8507:
        raise RuntimeError("canonical context pack must contain exactly 8,507 parents")
    model_target = args.output / "jina-reranker-v2-base-multilingual"
    for source in sorted(args.model.rglob("*")):
        if source.is_file():
            link_or_copy(source.resolve(), model_target / source.relative_to(args.model))
    files = {str(path.relative_to(args.output)).replace("\\", "/"): sha256(path)
             for path in sorted(args.output.rglob("*"))
             if path.is_file() and path.name != "KAGGLE_INPUT_MANIFEST.json"}
    lock = json.loads((args.results / "EVIDENCE_RENDERER_LOCK.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "dsc2026.research_v2.kaggle_input.v1",
        "status": "SEALED", "private_dataset_name": "research-v2-jina-boundary",
        "renderer": lock["winner"],
        "v2_folds_sha256": sha256(args.results / "V2_FOLDS.json"),
        "files_sha256": files,
        "note": "Upload this directory as one private Kaggle Dataset; hard links are local staging only.",
    }
    (args.output / "KAGGLE_INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": len(files),
                      "bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
                      "renderer": lock["winner"]}, indent=2))


if __name__ == "__main__":
    main()
