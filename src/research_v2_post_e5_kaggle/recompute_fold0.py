"""Read-only recomputation audit for the Research V2 EXP-112 E5 Fold-0 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51+"


def paired_bootstrap(deltas: list[float], draws: int = 20_000, seed: int = 112) -> dict[str, float | int]:
    rng = random.Random(seed)
    values = []
    size = len(deltas)
    for _ in range(draws):
        values.append(sum(deltas[rng.randrange(size)] for _ in range(size)) / size)
    values.sort()
    return {
        "draws": draws,
        "seed": seed,
        "ci95_low": values[int(0.025 * draws)],
        "ci95_high": values[int(0.975 * draws)],
        "probability_positive": sum(value > 0 for value in values) / draws,
        "probability_delta_gte_0_005": sum(value >= 0.005 for value in values) / draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    evidence = args.evidence.resolve()
    bundle = args.bundle.resolve()
    run = evidence / "research_v2_e5_transfer_fold0"
    prediction_path = run / "score" / "E5_TRANSFER_FOLD0_PREDICTIONS.jsonl"
    report_path = run / "score" / "E5_TRANSFER_FOLD0_PILOT_REPORT.json"
    receipt_path = run / "KAGGLE_RUN_RECEIPT.json"
    runner_path = run / "e5_transfer_runner_kaggle_patch.py"
    checkpoint_path = run / "training" / "epoch-2.pt"
    training_path = run / "training" / "_SUCCESS.json"

    manifest_path = bundle / "E5_TRANSFER_INPUT_MANIFEST.json"
    manifest = read_json(manifest_path)
    manifest_issues: list[str] = []
    for relative, expected in manifest["files_sha256"].items():
        path = bundle / relative
        if not path.is_file():
            manifest_issues.append(f"missing:{relative}")
        elif sha256(path) != expected:
            manifest_issues.append(f"hash:{relative}")

    queries = {str(row["qid"]): row for row in records(bundle / "V2_TRANSFER_QUERIES.jsonl")}
    pools = {str(row["qid"]): [str(value) for value in row["doc_ids"]]
             for row in records(bundle / "V2_CANDIDATE_POOL.jsonl")}
    references = {str(row["qid"]): row
                  for row in records(bundle / "V2_FOLD0_FROZEN_REFERENCE.jsonl")}
    rows = list(records(prediction_path))
    by_qid = {str(row["qid"]): row for row in rows}
    fold0 = {qid for qid, row in queries.items() if row["fold"] == "fold_0"}

    row_issues: list[str] = []
    if len(by_qid) != len(rows):
        row_issues.append("duplicate_prediction_qid")
    if set(by_qid) != fold0:
        row_issues.append("prediction_population_not_exact_fold0")

    base_recalls: list[float] = []
    ft_recalls: list[float] = []
    base_precisions: list[float] = []
    ft_precisions: list[float] = []
    single_base: list[float] = []
    single_ft: list[float] = []
    multi_base: list[float] = []
    multi_ft: list[float] = []
    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    changed_sets: list[str] = []
    changed_top5_order: list[str] = []
    transitions: Counter[str] = Counter()
    crossings_in: list[dict[str, Any]] = []
    crossings_out: list[dict[str, Any]] = []
    base_reference_top5_exact = 0
    base_reference_full_rank_exact = 0
    base_reference_max_abs_error = 0.0
    base_reference_abs_errors: list[float] = []

    for qid in sorted(fold0, key=lambda value: int(value)):
        row = by_qid[qid]
        gold = {str(value) for value in queries[qid]["gold"]}
        if set(map(str, row["gold"])) != gold:
            row_issues.append(f"gold:{qid}")
        base_order = [str(value) for value in row["base_order"]]
        ft_order = [str(value) for value in row["ft_order"]]
        pool = pools[qid]
        if len(base_order) != len(set(base_order)) or set(base_order) != set(pool):
            row_issues.append(f"base_membership:{qid}")
        if len(ft_order) != len(set(ft_order)) or set(ft_order) != set(pool):
            row_issues.append(f"ft_membership:{qid}")
        if len(row["base_scores"]) != len(base_order) or len(row["ft_scores"]) != len(ft_order):
            row_issues.append(f"score_length:{qid}")

        base_hits = len(set(base_order[:5]) & gold)
        ft_hits = len(set(ft_order[:5]) & gold)
        base_recall = base_hits / len(gold)
        ft_recall = ft_hits / len(gold)
        base_precision = base_hits / 5
        ft_precision = ft_hits / 5
        base_recalls.append(base_recall)
        ft_recalls.append(ft_recall)
        base_precisions.append(base_precision)
        ft_precisions.append(ft_precision)
        (single_base if len(gold) == 1 else multi_base).append(base_recall)
        (single_ft if len(gold) == 1 else multi_ft).append(ft_recall)
        if ft_recall > base_recall:
            wins.append(qid)
        elif ft_recall < base_recall:
            losses.append(qid)
        else:
            ties.append(qid)
        if set(base_order[:5]) != set(ft_order[:5]):
            changed_sets.append(qid)
        if base_order[:5] != ft_order[:5]:
            changed_top5_order.append(qid)

        reference = references[qid]
        reference_order = [str(value) for value in reference["ranking"]]
        if base_order[:5] == reference_order[:5]:
            base_reference_top5_exact += 1
        if base_order == reference_order:
            base_reference_full_rank_exact += 1
        # Frozen-reference scores are stored in candidate ``doc_ids`` order;
        # ``ranking`` is a separate derived field.
        reference_score = {
            str(doc): float(score)
            for doc, score in zip(reference["doc_ids"], reference["scores"])
        }
        base_score = {str(doc): float(score) for doc, score in zip(base_order, row["base_scores"])}
        for doc in base_order:
            error = abs(base_score[doc] - reference_score[doc])
            base_reference_abs_errors.append(error)
            base_reference_max_abs_error = max(base_reference_max_abs_error, error)

        base_score_by_doc = {doc: float(score) for doc, score in zip(base_order, row["base_scores"])}
        ft_score_by_doc = {doc: float(score) for doc, score in zip(ft_order, row["ft_scores"])}
        for doc in sorted(gold):
            base_rank = base_order.index(doc) + 1 if doc in base_order else None
            ft_rank = ft_order.index(doc) + 1 if doc in ft_order else None
            transitions[f"{rank_bucket(base_rank)}->{rank_bucket(ft_rank)}"] += 1
            detail = {
                "qid": qid,
                "doc_id": doc,
                "base_rank": base_rank,
                "ft_rank": ft_rank,
                "base_boundary_margin": (
                    base_score_by_doc[doc] - float(row["base_scores"][4]) if doc in base_score_by_doc else None
                ),
                "ft_boundary_margin": (
                    ft_score_by_doc[doc] - float(row["ft_scores"][4]) if doc in ft_score_by_doc else None
                ),
            }
            if (base_rank is None or base_rank > 5) and ft_rank is not None and ft_rank <= 5:
                crossings_in.append(detail)
            elif base_rank is not None and base_rank <= 5 and (ft_rank is None or ft_rank > 5):
                crossings_out.append(detail)

    training = read_json(training_path)
    training_qids = {str(value) for value in training["scientific_contract"]["qids"]}
    exclusions = {str(value) for value in manifest["fold0_duplicate_exclusions"]}
    expected_training = {qid for qid, row in queries.items() if row["fold"] != "fold_0"} - exclusions
    isolation = {
        "training_qids": len(training_qids),
        "expected_training_qids": len(expected_training),
        "training_set_exact": training_qids == expected_training,
        "fold0_intersection": sorted(training_qids & fold0),
        "duplicate_exclusion_intersection": sorted(training_qids & exclusions),
        "duplicate_exclusions_exact": sorted(exclusions, key=int) == sorted(training["duplicate_exclusions"], key=int),
    }

    report = read_json(report_path)
    receipt = read_json(receipt_path)
    metrics = {
        "queries": len(rows),
        "base_recall_at_5": mean(base_recalls),
        "ft_recall_at_5": mean(ft_recalls),
        "delta_recall_at_5": mean(ft_recalls) - mean(base_recalls),
        "base_precision_at_5": mean(base_precisions),
        "ft_precision_at_5": mean(ft_precisions),
        "delta_precision_at_5": mean(ft_precisions) - mean(base_precisions),
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(ties),
        "paired_bootstrap": paired_bootstrap([
            ft - base for base, ft in zip(base_recalls, ft_recalls)
        ]),
        "changed_top5_sets": len(changed_sets),
        "changed_top5_order": len(changed_top5_order),
        "single_gold": {
            "queries": len(single_base),
            "base": mean(single_base),
            "ft": mean(single_ft),
            "delta": mean(single_ft) - mean(single_base),
            "wins": sum(ft > base for base, ft in zip(single_base, single_ft)),
            "losses": sum(ft < base for base, ft in zip(single_base, single_ft)),
            "ties": sum(ft == base for base, ft in zip(single_base, single_ft)),
        },
        "multi_gold": {
            "queries": len(multi_base),
            "base": mean(multi_base),
            "ft": mean(multi_ft),
            "delta": mean(multi_ft) - mean(multi_base),
            "wins": sum(ft > base for base, ft in zip(multi_base, multi_ft)),
            "losses": sum(ft < base for base, ft in zip(multi_base, multi_ft)),
            "ties": sum(ft == base for base, ft in zip(multi_base, multi_ft)),
        },
        "gold_crossings_into_top5": len(crossings_in),
        "gold_crossings_out_of_top5": len(crossings_out),
        "gold_rank_bucket_movements": dict(sorted(transitions.items())),
        "rank1_5_gold_retention": transitions["1-5->1-5"] / (
            transitions["1-5->1-5"] + len(crossings_out)
        ),
        "crossing_margin_summary": {
            "into_base_margin_mean": mean([row["base_boundary_margin"] for row in crossings_in]),
            "into_base_margin_median": median([row["base_boundary_margin"] for row in crossings_in]),
            "into_ft_margin_mean": mean([row["ft_boundary_margin"] for row in crossings_in]),
            "into_ft_margin_median": median([row["ft_boundary_margin"] for row in crossings_in]),
            "out_base_margin_mean": mean([row["base_boundary_margin"] for row in crossings_out]),
            "out_base_margin_median": median([row["base_boundary_margin"] for row in crossings_out]),
            "out_ft_margin_mean": mean([row["ft_boundary_margin"] for row in crossings_out]),
            "out_ft_margin_median": median([row["ft_boundary_margin"] for row in crossings_out]),
        },
    }
    report_metric_checks = {
        "base_recall": math.isclose(metrics["base_recall_at_5"], report["base_recall_at_5"], abs_tol=1e-15),
        "ft_recall": math.isclose(metrics["ft_recall_at_5"], report["ft_recall_at_5"], abs_tol=1e-15),
        "delta_recall": math.isclose(metrics["delta_recall_at_5"], report["delta_recall_at_5"], abs_tol=1e-15),
        "wins_losses_ties": [metrics["wins"], metrics["losses"], metrics["ties"]] == [report["wins"], report["losses"], report["ties"]],
        "changed_sets": metrics["changed_top5_sets"] == report["changed_top5_sets"],
        "crossings": [metrics["gold_crossings_into_top5"], metrics["gold_crossings_out_of_top5"]] == [report["gold_crossings_into_top5"], report["gold_crossings_out_of_top5"]],
        "rank_buckets": metrics["gold_rank_bucket_movements"] == report["gold_rank_bucket_movements"],
    }
    hashes = {
        "manifest": sha256(manifest_path),
        "runner": sha256(runner_path),
        "checkpoint": sha256(checkpoint_path),
        "predictions": sha256(prediction_path),
        "pilot_report": sha256(report_path),
    }
    receipt_hash_checks = {
        key: hashes[key] == receipt[f"{key}_sha256"]
        for key in ("manifest", "runner", "checkpoint", "predictions", "pilot_report")
    }
    frozen_reference = {
        "top5_exact_queries": base_reference_top5_exact,
        "top5_agreement": base_reference_top5_exact / len(rows),
        "full_rank_exact_queries": base_reference_full_rank_exact,
        "full_rank_agreement": base_reference_full_rank_exact / len(rows),
        "score_max_abs_error": base_reference_max_abs_error,
        "score_mean_abs_error": mean(base_reference_abs_errors),
    }
    output = {
        "schema_version": "dsc2026.research_v2.post_e5_kaggle_recompute.v1",
        "metrics_recomputed_from_predictions": metrics,
        "metric_report_checks": report_metric_checks,
        "prediction_details": {
            "win_qids": wins,
            "loss_qids": losses,
            "changed_top5_set_qids": changed_sets,
            "crossings_into_top5": crossings_in,
            "crossings_out_of_top5": crossings_out,
        },
        "candidate_and_fold_integrity": {
            "row_issues": row_issues,
            "prediction_population_exact_fold0": not row_issues,
            "fold0_queries": len(fold0),
            **isolation,
        },
        "frozen_reference_parity_recomputed": frozen_reference,
        "bundle_manifest": {
            "files_declared": len(manifest["files_sha256"]),
            "issues": manifest_issues,
            "all_hashes_pass": not manifest_issues,
        },
        "artifact_hashes": hashes,
        "receipt_hash_checks": receipt_hash_checks,
        "all_recomputed_checks_pass": (
            not row_issues
            and not manifest_issues
            and all(isolation[key] in (True, []) for key in (
                "training_set_exact", "fold0_intersection",
                "duplicate_exclusion_intersection", "duplicate_exclusions_exact",
            ))
            and all(report_metric_checks.values())
            and all(receipt_hash_checks.values())
            and frozen_reference["top5_agreement"] == 1.0
        ),
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
