"""Strict folds 1-4 confirmation for the frozen EXP-112 E5 transfer mechanism.

The learning implementation is imported from the sealed Fold-0 runner.  This
module only parameterizes the held fold, its duplicate-linked exclusions, and
held-fold reporting.  One invocation handles exactly one fold.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from research_v2_e5_transfer import e5_transfer_runner as core


ALLOWED_FOLDS = {f"fold_{index}" for index in range(5)}


class ConfirmationData(core.TransferData):
    def __init__(self, bundle: Path, held_fold: str):
        if held_fold not in ALLOWED_FOLDS:
            raise ValueError(f"Unsupported held fold: {held_fold}")
        super().__init__(bundle)
        boundary = core.read_json(self.bundle / "V2_BOUNDARY_GROUPS_MANIFEST.json")
        if boundary["status"] != "SEALED":
            raise RuntimeError("boundary-group manifest is not sealed")
        exclusions = boundary["held_fold_duplicate_exclusions"].get(held_fold)
        if exclusions is None:
            raise RuntimeError(f"missing duplicate exclusions for {held_fold}")
        self.held_fold = held_fold
        self.duplicate_exclusions = {str(value) for value in exclusions}

    def training_qids(self) -> list[str]:
        held = set(self.fold_qids(self.held_fold))
        qids = [qid for qid, fold in self.fold_for.items() if fold != self.held_fold]
        qids = [qid for qid in qids if qid not in self.duplicate_exclusions]
        if set(qids) & held or set(qids) & self.duplicate_exclusions:
            raise RuntimeError(f"{self.held_fold} or duplicate-linked qid leaked into training")
        expected = len(self.questions) - len(held) - len(self.duplicate_exclusions)
        if len(qids) != expected:
            raise RuntimeError(
                f"unexpected {self.held_fold} training population: {len(qids)} != {expected}"
            )
        return qids


def train_one_fold(data: ConfirmationData, output: Path, microbatch: int = 4) -> dict[str, Any]:
    qids = data.training_qids()
    # Scientific learning code is the sealed Fold-0 implementation.  Explicit
    # qids are its only fold-dependent input.
    result = core.train_adapter(data, output, microbatch=microbatch, qids=qids)
    if result["status"] != "COMPLETE_EPOCH2" or int(result["epochs"]) != 2:
        raise RuntimeError(f"incomplete training: {result['status']}")
    if set(map(str, result["scientific_contract"]["qids"])) != set(qids):
        raise RuntimeError("checkpoint training qids differ from held-fold contract")
    checkpoint = output / "epoch-2.pt"
    manifest = {
        "schema_version": "dsc2026.research_v2.e5_confirmation_training.v1",
        "status": "PASS",
        "held_fold": data.held_fold,
        "training_queries": len(qids),
        "held_queries": len(data.fold_qids(data.held_fold)),
        "duplicate_exclusions": sorted(data.duplicate_exclusions, key=int),
        "training_qids_sha256": core.digest(sorted(qids, key=int)),
        "folds_sha256": data.manifest["v2_folds_sha256"],
        "candidate_pool_sha256": data.manifest["candidate_pool_sha256"],
        "core_runner_sha256": core.sha256(Path(core.__file__)),
        "confirmation_runner_sha256": core.sha256(Path(__file__)),
        "checkpoint_sha256": core.sha256(checkpoint),
        "checkpoint_contract_hash": result["contract_hash"],
        "checkpoint_updates": result["updates"],
        "checkpoint_epochs": result["epochs"],
        "legacy_core_result_held_fold_field": result["held_fold"],
        "legacy_field_note": "The reused core serializes fold_0 in _SUCCESS metadata; qids, exclusions and this confirmation manifest are authoritative for the parameterized held fold.",
        "runtime_seconds": result["runtime_seconds_this_process"],
        "peak_allocated_mib": result["peak_allocated_mib"],
        "peak_reserved_mib": result["peak_reserved_mib"],
    }
    core.write_json(output / "CONFIRMATION_TRAINING_MANIFEST.json", manifest)
    return manifest


def query_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    recalls = [float(row[f"{key}_recall_at_5"]) for row in rows]
    precisions = [float(row[f"{key}_precision_at_5"]) for row in rows]
    return {"recall_at_5": mean(recalls), "precision_at_5": mean(precisions)}


def score_one_fold(
    data: ConfirmationData,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    qids = data.fold_qids(data.held_fold)
    model = core.QueryEncoder(data.bundle / "vietlegal-e5", checkpoint_path=checkpoint_path)
    model.eval()
    bank = core.ParentBank(data.vectors, data.parent)
    rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for index, qid in enumerate(qids, 1):
            with torch.no_grad():
                adapted_vector = model([data.questions[qid]])[0]
                with model.adapter_disabled():
                    frozen_vector = model([data.questions[qid]])[0]
            docs = data.pool[qid]
            adapted_scores = bank.score_pool(adapted_vector, docs, data)
            frozen_scores = bank.score_pool(frozen_vector, docs, data)
            adapted_order = core.ranking(docs, adapted_scores)
            frozen_order = core.ranking(docs, frozen_scores)
            if set(adapted_order) != set(docs) or set(frozen_order) != set(docs):
                raise RuntimeError(f"candidate membership changed: {qid}")
            gold = data.gold[qid]
            adapted_ranks = {
                doc_id: adapted_order.index(doc_id) + 1 if doc_id in adapted_order else None
                for doc_id in gold
            }
            frozen_ranks = {
                doc_id: frozen_order.index(doc_id) + 1 if doc_id in frozen_order else None
                for doc_id in gold
            }
            for doc_id in gold:
                transitions[
                    f"{core.rank_bucket(frozen_ranks[doc_id])}->{core.rank_bucket(adapted_ranks[doc_id])}"
                ] += 1
            base_hits = len(set(frozen_order[:5]) & gold)
            ft_hits = len(set(adapted_order[:5]) & gold)
            rows.append({
                "qid": qid,
                "fold": data.held_fold,
                "gold": sorted(gold),
                "base_order": frozen_order,
                "base_scores": [frozen_scores[docs.index(doc_id)] for doc_id in frozen_order],
                "ft_order": adapted_order,
                "ft_scores": [adapted_scores[docs.index(doc_id)] for doc_id in adapted_order],
                "base_recall_at_5": base_hits / len(gold),
                "ft_recall_at_5": ft_hits / len(gold),
                "base_precision_at_5": base_hits / 5,
                "ft_precision_at_5": ft_hits / 5,
                "frozen_query_cosine": float(
                    adapted_vector.detach().cpu().numpy() @ data.query_vector(qid)
                ),
            })
            if index % 25 == 0 or index == len(qids):
                print(json.dumps({
                    "stage": "score", "held_fold": data.held_fold,
                    "completed": index, "total": len(qids),
                }), flush=True)
    finally:
        del model, bank
        gc.collect()
        torch.cuda.empty_cache()

    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / f"E5_CONFIRMATION_{data.held_fold.upper()}_PREDICTIONS.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n")

    base = query_metrics(rows, "base")
    adapted = query_metrics(rows, "ft")
    wins = sum(row["ft_recall_at_5"] > row["base_recall_at_5"] for row in rows)
    losses = sum(row["ft_recall_at_5"] < row["base_recall_at_5"] for row in rows)
    single = [row for row in rows if len(row["gold"]) == 1]
    multi = [row for row in rows if len(row["gold"]) > 1]
    into = sum(
        count for movement, count in transitions.items()
        if not movement.startswith("1-5->") and movement.endswith("->1-5")
    )
    out = sum(
        count for movement, count in transitions.items()
        if movement.startswith("1-5->") and not movement.endswith("->1-5")
    )
    report = {
        "schema_version": "dsc2026.research_v2.e5_confirmation_fold.v1",
        "status": "COMPLETE",
        "held_fold": data.held_fold,
        "queries": len(rows),
        "base_recall_at_5": base["recall_at_5"],
        "ft_recall_at_5": adapted["recall_at_5"],
        "delta_recall_at_5": adapted["recall_at_5"] - base["recall_at_5"],
        "base_precision_at_5": base["precision_at_5"],
        "ft_precision_at_5": adapted["precision_at_5"],
        "delta_precision_at_5": adapted["precision_at_5"] - base["precision_at_5"],
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "changed_top5_sets": sum(
            set(row["base_order"][:5]) != set(row["ft_order"][:5]) for row in rows
        ),
        "changed_top5_order": sum(
            row["base_order"][:5] != row["ft_order"][:5] for row in rows
        ),
        "single_gold": {
            "queries": len(single),
            "base": mean(row["base_recall_at_5"] for row in single),
            "ft": mean(row["ft_recall_at_5"] for row in single),
            "wins": sum(row["ft_recall_at_5"] > row["base_recall_at_5"] for row in single),
            "losses": sum(row["ft_recall_at_5"] < row["base_recall_at_5"] for row in single),
        },
        "multi_gold": {
            "queries": len(multi),
            "base": mean(row["base_recall_at_5"] for row in multi),
            "ft": mean(row["ft_recall_at_5"] for row in multi),
            "delta": mean(
                row["ft_recall_at_5"] - row["base_recall_at_5"] for row in multi
            ),
            "wins": sum(row["ft_recall_at_5"] > row["base_recall_at_5"] for row in multi),
            "losses": sum(row["ft_recall_at_5"] < row["base_recall_at_5"] for row in multi),
        },
        "gold_crossings_into_top5": into,
        "gold_crossings_out_of_top5": out,
        "gold_rank_bucket_movements": dict(sorted(transitions.items())),
        "query_drift_cosine_mean": mean(row["frozen_query_cosine"] for row in rows),
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20),
        "checkpoint_sha256": core.sha256(checkpoint_path),
        "predictions_sha256": core.sha256(prediction_path),
        "candidate_pool_sha256": data.manifest["candidate_pool_sha256"],
        "folds_sha256": data.manifest["v2_folds_sha256"],
        "core_runner_sha256": core.sha256(Path(core.__file__)),
        "confirmation_runner_sha256": core.sha256(Path(__file__)),
        "scientific_contract": "exact_exp112_query_only_v1_to_immutable_v2_pool_direct_top5",
    }
    report_path = output / f"E5_CONFIRMATION_{data.held_fold.upper()}_REPORT.json"
    core.write_json(report_path, report)
    return report


def audit_data(data: ConfirmationData) -> dict[str, Any]:
    training = data.training_qids()
    held = data.fold_qids(data.held_fold)
    result = {
        "schema_version": "dsc2026.research_v2.e5_confirmation_data_audit.v1",
        "status": "PASS",
        "held_fold": data.held_fold,
        "held_queries": len(held),
        "training_queries": len(training),
        "duplicate_exclusions": sorted(data.duplicate_exclusions, key=int),
        "held_training_intersection": sorted(set(held) & set(training), key=int),
        "excluded_training_intersection": sorted(
            data.duplicate_exclusions & set(training), key=int
        ),
        "training_qids_sha256": core.digest(sorted(training, key=int)),
        "folds_sha256": data.manifest["v2_folds_sha256"],
        "candidate_pool_sha256": data.manifest["candidate_pool_sha256"],
        "core_runner_sha256": core.sha256(Path(core.__file__)),
        "confirmation_runner_sha256": core.sha256(Path(__file__)),
    }
    if result["held_training_intersection"] or result["excluded_training_intersection"]:
        raise RuntimeError(f"isolation failure: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-data", "train", "score"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--held-fold", choices=sorted(ALLOWED_FOLDS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--microbatch", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = ConfirmationData(args.bundle, args.held_fold)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "audit-data":
        result = audit_data(data)
        core.write_json(args.output / f"{args.held_fold.upper()}_DATA_AUDIT.json", result)
    elif args.command == "train":
        result = train_one_fold(data, args.output, microbatch=args.microbatch)
    else:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise ValueError("score requires an existing --checkpoint")
        result = score_one_fold(data, args.checkpoint, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
