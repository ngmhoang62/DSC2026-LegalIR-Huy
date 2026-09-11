"""Fail-closed audit for the preregistered EXP-112 E5 strict confirmation.

Folds 1--4 form the untouched confirmation stream.  Fold 0 is appended only
after that gate is evaluated to describe the complete development OOF result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "cache/research_v2_e5_confirmation/bundle-v1"
DEFAULT_RESULTS = ROOT / "results/research_v2_e5_confirmation"
DEFAULT_FOLD0 = (
    ROOT
    / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def bucket(rank: int | None) -> str:
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


def rank_of(order: list[str], doc_id: str) -> int | None:
    try:
        return order.index(doc_id) + 1
    except ValueError:
        return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_values: list[float] = []
    ft_values: list[float] = []
    base_precision: list[float] = []
    ft_precision: list[float] = []
    transitions: Counter[str] = Counter()
    into = out = wins = losses = changed_sets = changed_order = 0
    single: list[tuple[float, float]] = []
    multi: list[tuple[float, float]] = []
    drifts: list[float] = []

    for row in rows:
        gold = set(map(str, row["gold"]))
        base_order = list(map(str, row["base_order"]))
        ft_order = list(map(str, row["ft_order"]))
        base = len(gold & set(base_order[:5])) / len(gold)
        ft = len(gold & set(ft_order[:5])) / len(gold)
        base_values.append(base)
        ft_values.append(ft)
        base_precision.append(len(gold & set(base_order[:5])) / 5)
        ft_precision.append(len(gold & set(ft_order[:5])) / 5)
        wins += ft > base
        losses += ft < base
        changed_sets += set(base_order[:5]) != set(ft_order[:5])
        changed_order += base_order[:5] != ft_order[:5]
        (single if len(gold) == 1 else multi).append((base, ft))
        drifts.append(float(row["frozen_query_cosine"]))
        for doc_id in gold:
            before = bucket(rank_of(base_order, doc_id))
            after = bucket(rank_of(ft_order, doc_id))
            transitions[f"{before}->{after}"] += 1
            into += before != "1-5" and after == "1-5"
            out += before == "1-5" and after != "1-5"

    def sliced(values: list[tuple[float, float]]) -> dict[str, Any]:
        return {
            "queries": len(values),
            "base": mean(x[0] for x in values),
            "ft": mean(x[1] for x in values),
            "delta": mean(x[1] - x[0] for x in values),
            "wins": sum(x[1] > x[0] for x in values),
            "losses": sum(x[1] < x[0] for x in values),
        }

    return {
        "queries": len(rows),
        "base_recall_at_5": mean(base_values),
        "ft_recall_at_5": mean(ft_values),
        "delta_recall_at_5": mean(f - b for b, f in zip(base_values, ft_values)),
        "base_precision_at_5": mean(base_precision),
        "ft_precision_at_5": mean(ft_precision),
        "delta_precision_at_5": mean(f - b for b, f in zip(base_precision, ft_precision)),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "changed_top5_sets": changed_sets,
        "changed_top5_order": changed_order,
        "single_gold": sliced(single),
        "multi_gold": sliced(multi),
        "gold_crossings_into_top5": into,
        "gold_crossings_out_of_top5": out,
        "gold_rank_bucket_movements": dict(sorted(transitions.items())),
        "query_drift_cosine_mean": mean(drifts),
    }


def bootstrap(rows: list[dict[str, Any]], seed: int = 112, samples: int = 20_000) -> dict[str, Any]:
    delta = np.asarray(
        [float(row["ft_recall_at_5"]) - float(row["base_recall_at_5"]) for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        draws[start : start + count] = delta[indices].mean(axis=1)
    return {
        "seed": seed,
        "samples": samples,
        "observed_delta": float(delta.mean()),
        "mean": float(draws.mean()),
        "median": float(np.median(draws)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_delta_gt_0": float((draws > 0).mean()),
        "probability_delta_gte_0_005": float((draws >= 0.005).mean()),
    }


def audit_prediction_rows(
    rows: list[dict[str, Any]], expected_qids: set[str], pools: dict[str, list[str]],
    fold: str,
) -> list[str]:
    failures: list[str] = []
    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        failures.append("duplicate_prediction_qids")
    if set(qids) != expected_qids:
        failures.append("held_fold_prediction_population_mismatch")
    for row in rows:
        qid = str(row["qid"])
        if qid not in pools:
            failures.append(f"missing_pool:{qid}")
            continue
        expected = set(pools[qid])
        for key in ("base_order", "ft_order"):
            order = list(map(str, row[key]))
            if len(order) != len(set(order)) or set(order) != expected:
                failures.append(f"candidate_membership:{qid}:{key}")
        if "fold" in row and row["fold"] != fold:
            failures.append(f"fold_tag:{qid}")
        gold = list(map(str, row["gold"]))
        if not gold or len(gold) != len(set(gold)):
            failures.append(f"gold_contract:{qid}")
    return failures


def file_manifest(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(paths):
        result.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fold0", type=Path, default=DEFAULT_FOLD0)
    args = parser.parse_args()

    folds_data = read_json(args.bundle / "V2_FOLDS.json")
    non_evaluable = set(map(str, folds_data["population"]["non_evaluable_qids"]))
    fold_qids = {
        fold: set(map(str, qids)) - non_evaluable
        for fold, qids in folds_data["folds"].items()
    }
    pools = {
        str(row["qid"]): list(map(str, row["doc_ids"]))
        for row in read_jsonl(args.bundle / "V2_CANDIDATE_POOL.jsonl")
    }
    prereg = read_json(args.results / "E5_STRICT_CONFIRMATION_PREREGISTRATION.json")
    seal = read_json(args.results / "CONFIRMATION_RUNNER_SEAL.json")
    expected_runner = seal["confirmation_runner_sha256"]
    expected_core = seal["imported_core_runner_sha256"]

    fold_rows: dict[str, list[dict[str, Any]]] = {}
    fold_metrics: dict[str, dict[str, Any]] = {}
    fold_integrity: dict[str, dict[str, Any]] = {}

    fold0_prediction = args.fold0 / "score/E5_TRANSFER_FOLD0_PREDICTIONS.jsonl"
    fold0_report = args.fold0 / "score/E5_TRANSFER_FOLD0_PILOT_REPORT.json"
    rows0 = read_jsonl(fold0_prediction)
    report0 = read_json(fold0_report)
    failures0 = audit_prediction_rows(rows0, fold_qids["fold_0"], pools, "fold_0")
    if sha256(fold0_prediction) != prereg["development_evidence"]["predictions_sha256"]:
        failures0.append("fold0_prediction_hash")
    if report0["checkpoint_sha256"] != prereg["development_evidence"]["checkpoint_sha256"]:
        failures0.append("fold0_checkpoint_hash")
    fold_rows["fold_0"] = rows0
    fold_metrics["fold_0"] = summarize(rows0)
    fold_integrity["fold_0"] = {
        "status": "PASS" if not failures0 else "FAIL",
        "failures": failures0,
        "role": "immutable development evidence; not part of primary confirmation gate",
        "predictions_sha256": sha256(fold0_prediction),
        "checkpoint_sha256": report0["checkpoint_sha256"],
    }

    for index in range(1, 5):
        fold = f"fold_{index}"
        directory = args.results / f"fold_{index}"
        prediction = directory / "score" / f"E5_CONFIRMATION_FOLD_{index}_PREDICTIONS.jsonl"
        score_report_path = directory / "score" / f"E5_CONFIRMATION_FOLD_{index}_REPORT.json"
        training_manifest_path = directory / "training/CONFIRMATION_TRAINING_MANIFEST.json"
        data_audit_path = (
            args.results
            / "confirmation_bundle_preflight"
            / f"fold_{index}"
            / f"FOLD_{index}_DATA_AUDIT.json"
        )
        checkpoint = directory / "training/epoch-2.pt"
        success_path = directory / "training/_SUCCESS.json"
        epoch_hash_path = directory / "training/epoch-2.sha.json"
        resume_path = directory / "training/resume.pt"
        resume_hash_path = directory / "training/resume.sha.json"
        required = [
            prediction, score_report_path, training_manifest_path, data_audit_path,
            checkpoint, success_path, epoch_hash_path, resume_path, resume_hash_path,
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{fold} incomplete: {missing}")
        rows = read_jsonl(prediction)
        score_report = read_json(score_report_path)
        training_manifest = read_json(training_manifest_path)
        data_audit = read_json(data_audit_path)
        success = read_json(success_path)
        epoch_hash = read_json(epoch_hash_path)
        resume_hash = read_json(resume_hash_path)
        failures = audit_prediction_rows(rows, fold_qids[fold], pools, fold)
        recomputed = summarize(rows)
        checks = {
            "prediction_report_hash": sha256(prediction) == score_report["predictions_sha256"],
            "checkpoint_score_hash": sha256(checkpoint) == score_report["checkpoint_sha256"],
            "checkpoint_training_hash": sha256(checkpoint) == training_manifest["checkpoint_sha256"],
            "held_fold_tags": score_report["held_fold"] == fold == training_manifest["held_fold"],
            "data_audit_status": data_audit["status"] == "PASS",
            "data_no_held_overlap": not data_audit["held_training_intersection"],
            "data_no_excluded_overlap": not data_audit["excluded_training_intersection"],
            "training_qids_hash": training_manifest["training_qids_sha256"] == data_audit["training_qids_sha256"],
            "training_population_counts": (
                training_manifest["training_queries"] == data_audit["training_queries"]
                and training_manifest["held_queries"] == data_audit["held_queries"]
            ),
            "success_training_qids": set(map(str, success["scientific_contract"]["qids"])) == (set(pools) - fold_qids[fold] - set(map(str, training_manifest["duplicate_exclusions"]))),
            "success_duplicate_exclusions": set(map(str, success["duplicate_exclusions"])) == set(map(str, training_manifest["duplicate_exclusions"])),
            "epoch2_sidecar_hash": epoch_hash["sha256"] == sha256(checkpoint),
            "resume_sidecar_hash": resume_hash["sha256"] == sha256(resume_path),
            "success_epoch2_hash": success["checkpoint_sha256"]["epoch-2.pt"] == sha256(checkpoint),
            "success_two_epochs": success["epochs"] == 2,
            "runner_hash": score_report["confirmation_runner_sha256"] == expected_runner,
            "core_hash": score_report["core_runner_sha256"] == expected_core,
            "folds_hash": score_report["folds_sha256"] == prereg["frozen_scientific_contract"]["evaluation"]["folds_sha256"],
            "candidate_pool_hash": score_report["candidate_pool_sha256"] == prereg["frozen_scientific_contract"]["evaluation"]["candidate_pool_sha256"],
            "two_epochs": training_manifest["checkpoint_epochs"] == 2,
            "scientific_contract": score_report["scientific_contract"] == "exact_exp112_query_only_v1_to_immutable_v2_pool_direct_top5",
            "direct_metric_recompute": all(
                abs(float(score_report[key]) - float(recomputed[key])) <= 1e-12
                for key in (
                    "base_recall_at_5", "ft_recall_at_5", "delta_recall_at_5",
                    "base_precision_at_5", "ft_precision_at_5", "delta_precision_at_5",
                )
            ),
            "direct_count_recompute": all(
                score_report[key] == recomputed[key]
                for key in (
                    "wins", "losses", "ties", "changed_top5_sets",
                    "changed_top5_order", "gold_crossings_into_top5",
                    "gold_crossings_out_of_top5", "gold_rank_bucket_movements",
                )
            ),
        }
        failures.extend(key for key, passed in checks.items() if not passed)
        fold_rows[fold] = rows
        fold_metrics[fold] = recomputed
        fold_integrity[fold] = {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "checks": checks,
            "predictions_sha256": sha256(prediction),
            "checkpoint_sha256": sha256(checkpoint),
            "training_queries": training_manifest["training_queries"],
            "held_queries": training_manifest["held_queries"],
            "duplicate_exclusions": training_manifest["duplicate_exclusions"],
            "runtime_seconds_training": training_manifest["runtime_seconds"],
            "runtime_seconds_scoring": score_report["runtime_seconds"],
            "peak_allocated_mib_training": training_manifest["peak_allocated_mib"],
            "peak_allocated_mib_scoring": score_report["peak_allocated_mib"],
        }

    confirmation_rows = sum((fold_rows[f"fold_{i}"] for i in range(1, 5)), [])
    confirmation = summarize(confirmation_rows)
    confirmation["paired_bootstrap"] = bootstrap(confirmation_rows)
    positive_folds = sum(fold_metrics[f"fold_{i}"]["delta_recall_at_5"] > 0 for i in range(1, 5))
    worst_fold = min(fold_metrics[f"fold_{i}"]["delta_recall_at_5"] for i in range(1, 5))
    checks = {
        "pooled_delta_recall_at_5_gte_0_005": confirmation["delta_recall_at_5"] >= 0.005,
        "at_least_3_of_4_folds_positive": positive_folds >= 3,
        "pooled_wins_exceed_losses": confirmation["wins"] > confirmation["losses"],
        "pooled_crossings_into_exceed_out": confirmation["gold_crossings_into_top5"] > confirmation["gold_crossings_out_of_top5"],
        "pooled_multi_gold_delta_gte_minus_0_005": confirmation["multi_gold"]["delta"] >= -0.005,
        "no_fold_regression_below_minus_0_005": worst_fold >= -0.005,
        "all_isolation_hash_checkpoint_checks_pass": all(
            fold_integrity[f"fold_{i}"]["status"] == "PASS" for i in range(1, 5)
        ),
    }
    confirmation["directionally_positive_folds"] = positive_folds
    confirmation["worst_fold_delta_recall_at_5"] = worst_fold
    confirmation["gate_checks"] = checks
    verdict = "CONFIRMED" if all(checks.values()) else "NOT_CONFIRMED"

    full_rows = sum((fold_rows[f"fold_{i}"] for i in range(5)), [])
    full = summarize(full_rows)
    full["paired_bootstrap"] = bootstrap(full_rows)
    full["role"] = "complete development OOF; Fold 0 promoted the family and is not independent confirmation"

    artifact_paths: list[Path] = [
        args.results / "E5_STRICT_CONFIRMATION_PREREGISTRATION.json",
        args.results / "CONFIRMATION_RUNNER_SEAL.json",
        ROOT / "src/research_v2_e5_confirmation/e5_confirmation_runner.py",
        ROOT / "src/research_v2_e5_confirmation/audit_strict_confirmation.py",
        ROOT / "src/research_v2_e5_confirmation/stage_confirmation_bundle.py",
        ROOT / "src/research_v2_e5_confirmation/run_confirmation_folds.ps1",
        ROOT / "src/research_v2_e5_transfer/e5_transfer_runner.py",
        args.bundle / "CONFIRMATION_BUNDLE_SEAL.json",
        args.results / "STRICT_CONFIRMATION_FOLDS_1_4_TRANSCRIPT.log",
        fold0_prediction,
        fold0_report,
    ]
    for index in range(1, 5):
        directory = args.results / f"fold_{index}"
        artifact_paths.extend(path for path in directory.rglob("*") if path.is_file())
        artifact_paths.append(
            args.results
            / "confirmation_bundle_preflight"
            / f"fold_{index}"
            / f"FOLD_{index}_DATA_AUDIT.json"
        )

    report = {
        "schema_version": "dsc2026.research_v2.e5_strict_confirmation_report.v1",
        "verdict": verdict,
        "family": prereg["family"],
        "frozen_scientific_contract": prereg["frozen_scientific_contract"],
        "fold_0_development": fold_metrics["fold_0"],
        "folds_1_4_primary_confirmation": confirmation,
        "complete_five_fold_development_oof": full,
        "per_fold": fold_metrics,
        "integrity": fold_integrity,
        "artifact_manifest": file_manifest(artifact_paths, ROOT),
        "anti_rescue_policy": prereg["anti_rescue_policy"],
        "next_stage": (
            "PROPOSE_CACHE_ONLY_MATCHED_COMPLEMENTARITY_AUDIT_NO_FUSION_OR_SUBMISSION"
            if verdict == "CONFIRMED"
            else "CLOSE_TRANSFER_FAMILY_WITHOUT_HYPERPARAMETER_RESCUE"
        ),
    }
    output = args.results / "E5_STRICT_CONFIRMATION_REPORT.json"
    write_json(output, report)

    for index in range(5):
        fold = f"fold_{index}"
        manifest = {
            "schema_version": "dsc2026.research_v2.e5_confirmation_prediction_manifest.v1",
            "fold": fold,
            "role": "development" if index == 0 else "strict_confirmation",
            "metrics": fold_metrics[fold],
            "integrity": fold_integrity[fold],
        }
        write_json(args.results / f"FOLD_{index}_PREDICTION_MANIFEST.json", manifest)

    final_paths = list(artifact_paths)
    final_paths.append(output)
    markdown_report = args.results / "E5_STRICT_CONFIRMATION_REPORT.md"
    if markdown_report.is_file():
        final_paths.append(markdown_report)
    final_paths.extend(args.results / f"FOLD_{index}_PREDICTION_MANIFEST.json" for index in range(5))
    final_seal = {
        "schema_version": "dsc2026.research_v2.e5_strict_confirmation_final_seal.v1",
        "status": "SEALED",
        "verdict": verdict,
        "files": file_manifest(final_paths, ROOT),
    }
    write_json(args.results / "E5_STRICT_CONFIRMATION_FINAL_SEAL.json", final_seal)

    print(json.dumps({
        "verdict": verdict,
        "confirmation": confirmation,
        "full_oof": full,
        "report": str(output),
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
