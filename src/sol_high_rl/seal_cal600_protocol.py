"""Seal canonical CAL600 stratified folds and reproduce Huy's baseline OOF."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from forensic_world_model import fit_rankings, metrics, prepare_contract


OUT = ROOT / "results/sol_high_rl"
SPLIT_PATH = OUT / "CAL600_STRATIFIED_5FOLD_SEED42.json"
SPLIT_HASH_PATH = OUT / "CAL600_STRATIFIED_5FOLD_SEED42.sha256"
PRED_PATH = OUT / "CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json"
REPORT_PATH = OUT / "CAL600_CANONICAL_BASELINE_REPORT.json"
SEED = 42
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_SAMPLES = 10000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def seal_split(queries, old_blocks, all_ids):
    labels = [len(queries[q][1]) for q in all_ids]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = {}
    for fold_index, (_, valid) in enumerate(splitter.split(np.asarray(all_ids), labels)):
        folds[f"fold_{fold_index}"] = [all_ids[int(i)] for i in valid]
    block_of = {q: block for block, ids in old_blocks.items() for q in ids}
    payload = {
        "schema_version": "cal600_stratified_5fold_v1",
        "population_name": "CAL600",
        "population_semantics": "clean calibration/validation holdout for frozen lower-level scorers; labels allowed only inside outer-train meta-ranking folds",
        "seed": SEED,
        "n_splits": 5,
        "shuffle": True,
        "stratification": "exact number of gold answers, matching LegalIR src/split_cv.py methodology",
        "ordered_qids_sha256": sha256_bytes("\n".join(all_ids).encode("utf-8")),
        "qid_gold_sets_sha256": sha256_bytes(
            "\n".join(f"{q}\t{','.join(sorted(queries[q][1]))}" for q in all_ids).encode("utf-8")
        ),
        "population_size": len(all_ids),
        "gold_count_distribution": dict(sorted(Counter(labels).items())),
        "folds": folds,
        "fold_diagnostics": {
            name: {
                "queries": len(ids),
                "gold_count_distribution": dict(sorted(Counter(len(queries[q][1]) for q in ids).items())),
                "old_block_distribution": dict(sorted(Counter(block_of[q] for q in ids).items())),
            }
            for name, ids in folds.items()
        },
    }
    encoded = canonical_bytes(payload)
    if SPLIT_PATH.exists() and SPLIT_PATH.read_bytes() != encoded:
        raise RuntimeError(f"Immutable CAL600 split mismatch: {SPLIT_PATH}")
    SPLIT_PATH.write_bytes(encoded)
    checksum = sha256_bytes(encoded)
    expected_hash_file = f"{checksum}  {SPLIT_PATH.name}\n"
    if SPLIT_HASH_PATH.exists() and SPLIT_HASH_PATH.read_text(encoding="ascii") != expected_hash_file:
        raise RuntimeError("Immutable split checksum sidecar mismatch")
    SPLIT_HASH_PATH.write_text(expected_hash_file, encoding="ascii")
    return folds, checksum, payload


def paired_bootstrap(candidate_values, baseline_values):
    delta = np.asarray(candidate_values, dtype=np.float64) - np.asarray(baseline_values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    n = len(delta)
    for begin in range(0, BOOTSTRAP_SAMPLES, 1000):
        size = min(1000, BOOTSTRAP_SAMPLES - begin)
        indices = rng.integers(0, n, size=(size, n))
        samples[begin : begin + size] = delta[indices].mean(axis=1)
    return {
        "observed_delta": float(delta.mean()),
        "ci95_percentile": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bootstrap_probability_delta_gt_0": float(np.mean(samples > 0)),
        "bootstrap_probability_delta_ge_0": float(np.mean(samples >= 0)),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def main():
    started = time.perf_counter()
    (queries, old_blocks, all_ids, candidates, views, scores, names, _, extras, _) = prepare_contract()
    folds, split_checksum, split_payload = seal_split(queries, old_blocks, all_ids)
    canonical, apparent = fit_rankings(queries, folds, all_ids, candidates, views, scores, names, extras)
    canonical_metrics, canonical_per_q = metrics(canonical, queries, all_ids)
    apparent_metrics, _ = metrics(apparent, queries, all_ids)
    old_predictions = json.loads((OUT / "BASELINE_LOBO_PREDICTIONS.json").read_text(encoding="utf-8"))
    old_metrics, old_per_q = metrics(old_predictions, queries, all_ids)
    comparison = {
        "canonical_vs_old_4block_lobo": paired_bootstrap(
            [canonical_per_q[q] for q in all_ids], [old_per_q[q] for q in all_ids]
        ),
        "wins": sum(canonical_per_q[q] > old_per_q[q] for q in all_ids),
        "losses": sum(canonical_per_q[q] < old_per_q[q] for q in all_ids),
        "ties": sum(canonical_per_q[q] == old_per_q[q] for q in all_ids),
        "warning": "different outer splits; this comparison diagnoses split sensitivity and is not a method gain",
    }
    fold_metrics = {name: metrics(canonical, queries, ids)[0] for name, ids in folds.items()}
    old_block_stress = {name: metrics(canonical, queries, ids)[0] for name, ids in old_blocks.items()}
    candidate_ceiling = {
        "pooled": float(np.mean([len(set(candidates[q]) & queries[q][1]) / len(queries[q][1]) for q in all_ids])),
        "folds": {
            name: float(np.mean([len(set(candidates[q]) & queries[q][1]) / len(queries[q][1]) for q in ids]))
            for name, ids in folds.items()
        },
        "old_block_stress": {
            name: float(np.mean([len(set(candidates[q]) & queries[q][1]) / len(queries[q][1]) for q in ids]))
            for name, ids in old_blocks.items()
        },
    }
    pred_bytes = canonical_bytes(canonical)
    PRED_PATH.write_bytes(pred_bytes)
    report = {
        "status": "SEALED",
        "protocol": {
            "name": "CAL600 canonical stratified 5-fold OOF",
            "split_path": str(SPLIT_PATH.relative_to(ROOT)),
            "split_sha256": split_checksum,
            "seed": SEED,
            "primary_metric": "pooled query-macro Recall@5",
            "meta_label_policy": "strict outer-fold isolation for every CAL600-supervised fit",
            "frozen_scorer_policy": "frozen scorers trained outside CAL600 may score all CAL600 once",
            "old_blocks": "secondary stress slices only",
        },
        "candidate_contract_sha256": sha256_bytes(
            canonical_bytes({q: candidates[q] for q in all_ids})
        ),
        "predictions_sha256": sha256_bytes(pred_bytes),
        "baseline": canonical_metrics,
        "folds": fold_metrics,
        "old_block_stress": old_block_stress,
        "candidate_ceiling": candidate_ceiling,
        "old_4block_lobo_anchor": old_metrics,
        "canonical_vs_old": comparison,
        "apparent_fit_all_600_diagnostic_only": apparent_metrics,
        "split_diagnostics": split_payload["fold_diagnostics"],
        "runtime_seconds": time.perf_counter() - started,
    }
    report_bytes = canonical_bytes(report)
    REPORT_PATH.write_bytes(report_bytes)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"report_sha256={sha256_bytes(report_bytes)}", flush=True)


if __name__ == "__main__":
    main()
