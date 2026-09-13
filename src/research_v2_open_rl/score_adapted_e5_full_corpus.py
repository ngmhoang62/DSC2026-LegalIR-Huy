"""Strict-V2 full-corpus scoring with already-confirmed E5 query adapters.

This is an inference-only transfer diagnostic.  It imports the sealed EXP-112
V2 implementation so query encoding, frozen bank normalization and parent
top2_mean are not reimplemented.  No labels affect scoring or configuration.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import torch

from research_v2_e5_confirmation.e5_confirmation_runner import ConfirmationData
from research_v2_e5_transfer import e5_transfer_runner as core


FOLDS = tuple(f"fold_{index}" for index in range(5))
DEPTHS = (5, 20, 50, 100, 150)
EXPECTED_FOLDS_SHA = "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19"
EXPECTED_POOL_SHA = "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277"
EXPECTED_BANK_SHA = "54c80d8da4b26806179b186e4cfb3995c5d9dae87e786360ed3f8b721260cc24"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def lexical_order(doc_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Score descending, then lexical doc-id, matching core.ranking."""
    return np.lexsort((doc_ids, -scores))


def checkpoint_for(root: Path, fold: str) -> Path:
    index = int(fold.rsplit("_", 1)[1])
    if index == 0:
        return root / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt"
    return root / f"results/research_v2_e5_confirmation/fold_{index}/training/epoch-2.pt"


def existing_predictions_for(root: Path, fold: str) -> Path:
    index = int(fold.rsplit("_", 1)[1])
    if index == 0:
        return root / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/score/E5_TRANSFER_FOLD0_PREDICTIONS.jsonl"
    return root / f"results/research_v2_e5_confirmation/fold_{index}/score/E5_CONFIRMATION_FOLD_{index}_PREDICTIONS.jsonl"


def verify_contract(data: ConfirmationData, root: Path) -> dict[str, Any]:
    manifest = data.manifest
    checks = {
        "folds_sha256": manifest["v2_folds_sha256"] == EXPECTED_FOLDS_SHA,
        "pool_sha256": manifest["candidate_pool_sha256"] == EXPECTED_POOL_SHA,
        "bank_manifest_sha256": manifest["files_sha256"]["embeddings.f16.npy"] == EXPECTED_BANK_SHA,
        "queries": len(data.questions) == 6991,
        "parents": len(data.doc_ids) == 8507,
        "chunks": len(data.chunk_ids) == 343347,
        "all_checkpoints_exist": all(checkpoint_for(root, fold).is_file() for fold in FOLDS),
        "all_fixed_pool_predictions_exist": all(existing_predictions_for(root, fold).is_file() for fold in FOLDS),
    }
    if not all(checks.values()):
        raise RuntimeError(f"contract verification failed: {checks}")
    return checks


@torch.inference_mode()
def score_fold(
    *, root: Path, bundle: Path, output: Path, fold: str, batch_size: int,
) -> dict[str, Any]:
    fold_dir = output / fold
    prediction_path = fold_dir / "FULL_CORPUS_PREDICTIONS.jsonl"
    report_path = fold_dir / "FULL_CORPUS_REPORT.json"
    if prediction_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE" or report.get("predictions_sha256") != sha256(prediction_path):
            raise RuntimeError(f"invalid completed fold artifact: {fold}")
        return report

    data = ConfirmationData(bundle, fold)
    checks = verify_contract(data, root)
    checkpoint = checkpoint_for(root, fold)
    existing = {str(row["qid"]): row for row in read_jsonl(existing_predictions_for(root, fold))}
    qids = data.fold_qids(fold)
    if set(existing) != set(qids):
        raise RuntimeError(f"existing prediction population mismatch: {fold}")

    model = core.QueryEncoder(bundle / "vietlegal-e5", checkpoint_path=checkpoint)
    model.eval()
    bank = core.ParentBank(data.vectors, data.parent)
    doc_ids = np.asarray(data.doc_ids, dtype=str)
    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    try:
        for start in range(0, len(qids), batch_size):
            local_qids = qids[start:start + batch_size]
            adapted_vectors = model([data.questions[qid] for qid in local_qids])
            with model.adapter_disabled():
                frozen_vectors = model([data.questions[qid] for qid in local_qids])
            adapted_matrix, _ = bank.mine(adapted_vectors)
            frozen_matrix, _ = bank.mine(frozen_vectors)
            adapted_np = adapted_matrix.detach().cpu().numpy()
            frozen_np = frozen_matrix.detach().cpu().numpy()
            for offset, qid in enumerate(local_qids):
                adapted_order_index = lexical_order(doc_ids, adapted_np[offset])
                frozen_order_index = lexical_order(doc_ids, frozen_np[offset])
                adapted_order = doc_ids[adapted_order_index[:150]].tolist()
                frozen_order = doc_ids[frozen_order_index[:150]].tolist()
                adapted_scores = adapted_np[offset, adapted_order_index[:150]].astype(float).tolist()
                frozen_scores = frozen_np[offset, frozen_order_index[:150]].astype(float).tolist()
                gold = data.gold[qid]
                rows.append({
                    "qid": qid,
                    "fold": fold,
                    "gold": sorted(gold),
                    "adapted_order_top150": adapted_order,
                    "adapted_scores_top150": adapted_scores,
                    "frozen_order_top150": frozen_order,
                    "frozen_scores_top150": frozen_scores,
                })

                # Fixed deterministic sample, checked against the already sealed
                # fixed-pool scorer.  It validates the exact query/checkpoint/bank
                # computation without reading labels or changing configuration.
                if len(parity_rows) < 5:
                    docs = data.pool[qid]
                    adapted_pool_scores = bank.score_pool(adapted_vectors[offset], docs, data)
                    frozen_pool_scores = bank.score_pool(frozen_vectors[offset], docs, data)
                    adapted_pool_order = core.ranking(docs, adapted_pool_scores)
                    frozen_pool_order = core.ranking(docs, frozen_pool_scores)
                    reference = existing[qid]
                    ref_adapted = {doc: float(score) for doc, score in zip(reference["ft_order"], reference["ft_scores"])}
                    ref_frozen = {doc: float(score) for doc, score in zip(reference["base_order"], reference["base_scores"])}
                    parity_rows.append({
                        "qid": qid,
                        "adapted_max_abs_error": max(abs(float(score) - ref_adapted[doc]) for doc, score in zip(docs, adapted_pool_scores)),
                        "frozen_max_abs_error": max(abs(float(score) - ref_frozen[doc]) for doc, score in zip(docs, frozen_pool_scores)),
                        "adapted_top5_exact": adapted_pool_order[:5] == reference["ft_order"][:5],
                        "frozen_top5_exact": frozen_pool_order[:5] == reference["base_order"][:5],
                    })
            completed = min(start + len(local_qids), len(qids))
            print(json.dumps({"fold": fold, "completed": completed, "total": len(qids)}), flush=True)
    finally:
        peak_allocated = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        del model, bank
        gc.collect()
        torch.cuda.empty_cache()

    max_error = max(max(row["adapted_max_abs_error"], row["frozen_max_abs_error"]) for row in parity_rows)
    parity_pass = max_error <= 2e-6 and all(row["adapted_top5_exact"] and row["frozen_top5_exact"] for row in parity_rows)
    if not parity_pass:
        raise RuntimeError(f"fixed-pool parity failed for {fold}: {parity_rows}")
    write_jsonl(prediction_path, rows)
    report = {
        "schema_version": "dsc2026.research_v2.adapted_e5_full_corpus_fold.v1",
        "status": "COMPLETE",
        "fold": fold,
        "queries": len(rows),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "contract_checks": checks,
        "parity": {"status": "PASS", "max_abs_error": max_error, "rows": parity_rows},
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "predictions_sha256": sha256(prediction_path),
    }
    write_json(report_path, report)
    return report


def recall(order: list[str], gold: set[str], depth: int) -> float:
    return len(set(order[:depth]) & gold) / len(gold)


def aggregate(*, root: Path, bundle: Path, output: Path) -> dict[str, Any]:
    data = ConfirmationData(bundle, "fold_0")
    verify_contract(data, root)
    rows: list[dict[str, Any]] = []
    fold_reports: dict[str, Any] = {}
    for fold in FOLDS:
        report_path = output / fold / "FULL_CORPUS_REPORT.json"
        prediction_path = output / fold / "FULL_CORPUS_PREDICTIONS.jsonl"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "COMPLETE" or report["predictions_sha256"] != sha256(prediction_path):
            raise RuntimeError(f"incomplete or corrupt fold: {fold}")
        fold_reports[fold] = report
        rows.extend(read_jsonl(prediction_path))
    if len(rows) != 6991 or len({row["qid"] for row in rows}) != 6991:
        raise RuntimeError("OOF population is not exactly 6991 unique qids")

    fixed = {}
    for fold in FOLDS:
        fixed.update({str(row["qid"]): row for row in read_jsonl(existing_predictions_for(root, fold))})
    metrics: dict[str, Any] = {}
    for system, field in (("frozen_full_corpus", "frozen_order_top150"), ("adapted_full_corpus", "adapted_order_top150")):
        metrics[system] = {
            f"recall_at_{depth}": mean(recall(row[field], set(row["gold"]), depth) for row in rows)
            for depth in DEPTHS
        }

    fixed_recall = mean(recall(fixed[row["qid"]]["ft_order"], set(row["gold"]), 5) for row in rows)
    query_delta = []
    transitions: Counter[str] = Counter()
    unique_recovered: list[dict[str, Any]] = []
    current_pool_recall = []
    expanded_pool_recall = []
    for row in rows:
        qid = row["qid"]
        gold = set(row["gold"])
        fixed_value = recall(fixed[qid]["ft_order"], gold, 5)
        adapted_value = recall(row["adapted_order_top150"], gold, 5)
        query_delta.append(adapted_value - fixed_value)
        pool = set(data.pool[qid])
        expanded = pool | set(row["adapted_order_top150"][:50])
        current_pool_recall.append(len(pool & gold) / len(gold))
        expanded_pool_recall.append(len(expanded & gold) / len(gold))
        for doc in gold:
            current_rank = fixed[qid]["ft_order"].index(doc) + 1 if doc in fixed[qid]["ft_order"] else None
            full_rank = row["adapted_order_top150"].index(doc) + 1 if doc in row["adapted_order_top150"] else None
            transitions[f"{core.rank_bucket(current_rank)}->{core.rank_bucket(full_rank)}"] += 1
            if doc not in pool and full_rank is not None and full_rank <= 50:
                unique_recovered.append({"qid": qid, "fold": row["fold"], "doc_id": doc, "adapted_rank": full_rank})

    pool_ceiling = mean(current_pool_recall)
    expanded_ceiling = mean(expanded_pool_recall)
    wins = sum(value > 0 for value in query_delta)
    losses = sum(value < 0 for value in query_delta)
    recovered_top20 = sum(row["adapted_rank"] <= 20 for row in unique_recovered)
    recovered_fraction_top20 = recovered_top20 / len(unique_recovered) if unique_recovered else 0.0
    standalone_delta = metrics["adapted_full_corpus"]["recall_at_5"] - fixed_recall
    integrity = all(report["parity"]["status"] == "PASS" for report in fold_reports.values())
    promote = (
        integrity and expanded_ceiling - pool_ceiling >= 0.003 and
        recovered_fraction_top20 >= 0.20 and standalone_delta >= -0.002 and wins >= losses
    )
    kill = (
        expanded_ceiling - pool_ceiling < 0.002 or recovered_fraction_top20 < 0.10 or
        standalone_delta < -0.005
    )
    verdict = "PROMOTE_FULL_CORPUS_SOURCE" if promote else "REJECT_FULL_CORPUS_SOURCE" if kill else "INCONCLUSIVE_CANDIDATE_ONLY"

    per_fold: dict[str, Any] = {}
    for fold in FOLDS:
        subset = [row for row in rows if row["fold"] == fold]
        deltas = [
            recall(row["adapted_order_top150"], set(row["gold"]), 5) -
            recall(fixed[row["qid"]]["ft_order"], set(row["gold"]), 5)
            for row in subset
        ]
        per_fold[fold] = {
            "queries": len(subset),
            "adapted_full_corpus_recall_at_5": mean(recall(row["adapted_order_top150"], set(row["gold"]), 5) for row in subset),
            "adapted_fixed_pool_recall_at_5": mean(recall(fixed[row["qid"]]["ft_order"], set(row["gold"]), 5) for row in subset),
            "delta": mean(deltas),
            "wins": sum(value > 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
        }

    result = {
        "schema_version": "dsc2026.research_v2.adapted_e5_full_corpus_oof.v1",
        "status": "COMPLETE_STRICT_V2_OOF",
        "verdict": verdict,
        "classification": "PARTIAL_DUPLICATE_WITH_POSITIVE_PRIOR",
        "queries": len(rows),
        "metrics": metrics,
        "adapted_fixed_pool_recall_at_5": fixed_recall,
        "adapted_full_corpus_delta_vs_fixed_pool": standalone_delta,
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "current_pool_ceiling": pool_ceiling,
        "current_plus_adapted_top50_ceiling": expanded_ceiling,
        "oracle_delta": expanded_ceiling - pool_ceiling,
        "unique_out_of_pool_gold_recovered_at_50": len(unique_recovered),
        "unique_out_of_pool_gold_recovered_at_20": recovered_top20,
        "unique_recovered_fraction_at_20": recovered_fraction_top20,
        "rank_transitions_fixed_pool_to_full_corpus": dict(sorted(transitions.items())),
        "per_fold": per_fold,
        "fold_runtime": {fold: {key: fold_reports[fold][key] for key in ("runtime_seconds", "peak_allocated_mib", "peak_reserved_mib")} for fold in FOLDS},
        "artifact_hashes": {
            "preregistration": sha256(output / "ADAPTED_E5_FULL_CORPUS_PREREGISTRATION.json"),
            **{f"{fold}_predictions": fold_reports[fold]["predictions_sha256"] for fold in FOLDS},
        },
        "unique_recovered_gold": unique_recovered,
        "gate": {
            "integrity_parity_pass": integrity,
            "oracle_delta_gte_0_003": expanded_ceiling - pool_ceiling >= 0.003,
            "recovered_fraction_top20_gte_0_20": recovered_fraction_top20 >= 0.20,
            "standalone_delta_gte_minus_0_002": standalone_delta >= -0.002,
            "wins_gte_losses": wins >= losses,
        },
    }
    write_json(output / "ADAPTED_E5_FULL_CORPUS_OOF_REPORT.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score", "aggregate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.bundle = (args.bundle or args.root / "cache/research_v2_e5_confirmation/bundle-v1").resolve()
    args.output = (args.output or args.root / "results/research_v2_open_rl").resolve()
    if args.command == "score" and args.fold is None:
        parser.error("score requires --fold")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "score":
        print(json.dumps(score_fold(root=args.root, bundle=args.bundle, output=args.output, fold=args.fold, batch_size=args.batch_size), indent=2), flush=True)
    else:
        print(json.dumps(aggregate(root=args.root, bundle=args.bundle, output=args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
