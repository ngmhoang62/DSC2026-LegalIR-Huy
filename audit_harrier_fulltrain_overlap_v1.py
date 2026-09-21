#!/usr/bin/env python
"""
HUY DEADLINE — HAR(R)IER FULL-TRAIN PROVENANCE / SPLIT OVERLAP AUDIT V1
======================================================================

CPU-only.  No model inference.

Checks whether the freshly fine-tuned Harrier result files contain query IDs
from the CAL600 population or public-test population.  If CAL overlaps the
fine-tuning train result, CAL performance from this checkpoint is NOT an honest
generalization estimate and must be treated as training-set diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_named(base: Path, name: str):
    direct = base / name
    if direct.is_file():
        return direct
    hits = list(base.rglob(name))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Defaults to <repo>/models/vietlegal_finetuned_results_HNSW",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    model_root = (
        args.model_root.resolve()
        if args.model_root
        else root / "models/vietlegal_finetuned_results_HNSW"
    )
    sys.path.insert(0, str(root))

    from src.gemini.huy_d1_legal_section_evidence_v1.common import load_cal_data

    print("Model root:", model_root)

    train_path = find_named(model_root, "train_finetuned_best.json")
    warm_path = find_named(model_root, "warmup_finetuned_best.json")
    log_path = find_named(model_root, "finetuning_log.json")
    cfg_path = find_named(model_root, "config.json")

    print("train result :", train_path)
    print("warmup result:", warm_path)
    print("log          :", log_path)

    (
        _docs, _queries, _blocks, cal_ids, _extended, _views, _channels,
        _gold, _type_rows, _cite_rows,
    ) = load_cal_data()
    cal_set = set(map(str, cal_ids))

    public_path = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json"
    )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    pub_set = set(map(str, public))

    def keys(path):
        if path is None:
            return set()
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise RuntimeError(f"Expected dict JSON: {path}")
        return set(map(str, obj))

    train_ids = keys(train_path)
    warm_ids = keys(warm_path)

    report = {
        "model_root": str(model_root),
        "train_result": str(train_path) if train_path else None,
        "warmup_result": str(warm_path) if warm_path else None,
        "train_queries": len(train_ids),
        "warmup_queries": len(warm_ids),
        "cal600": len(cal_set),
        "public": len(pub_set),
        "train_cal_overlap": len(train_ids & cal_set),
        "warmup_cal_overlap": len(warm_ids & cal_set),
        "train_public_overlap": len(train_ids & pub_set),
        "warmup_public_overlap": len(warm_ids & pub_set),
        "train_warmup_overlap": len(train_ids & warm_ids),
        "sample_train_cal": sorted(train_ids & cal_set)[:20],
        "sample_warmup_cal": sorted(warm_ids & cal_set)[:20],
    }

    if log_path:
        try:
            report["finetuning_log"] = json.loads(
                log_path.read_text(encoding="utf-8")
            )
        except Exception as e:
            report["finetuning_log_error"] = repr(e)

    out = root / "results/manual/huy_harrier_fulltrain_overlap_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    op = out / "REPORT.json"
    op.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(
        f"train∩CAL={report['train_cal_overlap']}/{len(cal_set)} | "
        f"warmup∩CAL={report['warmup_cal_overlap']}/{len(cal_set)}"
    )
    print(
        f"train∩PUBLIC={report['train_public_overlap']}/{len(pub_set)} | "
        f"warmup∩PUBLIC={report['warmup_public_overlap']}/{len(pub_set)}"
    )
    if report["train_cal_overlap"]:
        print("CAL STATUS: CONTAMINATED_FOR_THIS_CHECKPOINT (diagnostic only)")
    else:
        print("CAL STATUS: NO_QID_OVERLAP_WITH_SAVED_TRAIN_RESULT")
    if report["train_public_overlap"] or report["warmup_public_overlap"]:
        print("WARNING: PUBLIC QID OVERLAP DETECTED — INVESTIGATE BEFORE SUBMISSION")
    else:
        print("PUBLIC QID STATUS: UNSEEN_BY_SAVED_TRAIN/WARMUP POPULATIONS")
    print("Report:", op)
    print("=" * 100)


if __name__ == "__main__":
    main()
