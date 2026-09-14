"""Run and verify baseline parity for profile_memory_plus_sparse_rank_scores.

Reproduces exact strict 5-fold OOF predictions, metrics, and saves per-query
predictions and pre-sort LR decision values.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_SNAPSHOT_DIR = Path(__file__).resolve().parent / "baseline_snapshot"
sys.path.insert(0, str(BASE_SNAPSHOT_DIR))

import run_huy_5fold_fasttrack as core

sys.path.insert(0, str(core.ROOT))
sys.path.insert(0, str(core.WORKSPACE / "LegalIR/scripts"))
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank
from exp_final_memory_ltr_probe import MEMORY_NAMES, memory_features, support_index


OUT_DIR = core.ROOT / "results/gemini/rabr_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAL_QUERIES = core.WORKSPACE / "LegalIR/cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz"

EXPECTED_METRICS = {
    "recall_at_5": 0.9488556715777428,
    "precision_at_5": 0.20314690316120732,
    "single_gold_recall_at_5": 0.9641304347826087,
    "multi_gold_recall_at_5": 0.7703266787658803,
    "per_fold_recall_at_5": {
        "fold_0": 0.9524320457796852,
        "fold_1": 0.9451597520267048,
        "fold_2": 0.9511555873242793,
        "fold_3": 0.9420243204577969,
        "fold_4": 0.9535050071530758,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def main():
    started = time.perf_counter()
    folds, pools, questions, golds, e5_orders, e5_scores, dup, _ = core.load_inputs()
    fold_for = {qid: fold for fold, qids in folds.items() for qid in qids}
    queries = {qid: (questions[qid], golds[qid]) for qid in pools}

    jina_order, jina_scores, _ = core.load_jina(pools)
    lal_order, lal_scores, _ = core.load_source_channel("lal", pools)
    legalir_jina_order, _, _ = core.load_source_channel("jina", pools)
    bm25_order, bm25_scores, _ = core.load_source_channel("bm25", pools)
    trigram_order, trigram_scores, _ = core.load_source_channel("trigram", pools)
    heads = core.document_heads(pools)
    doctype, citation = core.metadata_arrays(pools, questions, heads)

    with np.load(LAL_QUERIES, allow_pickle=False) as payload:
        embedding_ids = list(map(str, payload["query_ids"].tolist()))
        embedding_values = normalize(payload["vectors"])
    source_row = {qid: i for i, qid in enumerate(embedding_ids)}
    ordered_qids = sorted(pools, key=int)
    vectors = embedding_values[[source_row[qid] for qid in ordered_qids]]
    row = {qid: i for i, qid in enumerate(ordered_qids)}
    similarities = np.asarray(vectors @ vectors.T, dtype=np.float32)

    base_orders = {
        "jina_ce": jina_order,
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
        "legalir_jina": legalir_jina_order,
        "legalir_bm25": bm25_order,
        "legalir_trigram": trigram_order,
    }
    base_scores = {
        "jina_ce": jina_scores,
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
        "legalir_bm25": bm25_scores,
        "legalir_trigram": trigram_scores,
    }
    frozen_rank = {name: core.rank_columns(value, pools) for name, value in base_orders.items()}
    frozen_score = {name: core.score_columns(value, pools) for name, value in base_scores.items()}
    frozen_meta = {"doctype": doctype, "citation": citation}

    base_rank = ["jina_ce", "adapted_e5", "lal_native", "legalir_jina", "huy_profile"]
    base_score = ["jina_ce", "adapted_e5", "lal_native"]
    config = dict(
        rank_views=base_rank + ["legalir_bm25", "legalir_trigram"],
        score_channels=base_score + ["legalir_bm25", "legalir_trigram"],
        metadata=["citation", "lal_memory"],
    )

    predictions = {}
    decision_scores = {}
    saved_rows = {}
    all_qids = set(pools)

    for outer, test_ids in folds.items():
        fold_started = time.perf_counter()
        blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)
        profile_orders = {}
        memory_rows = {}

        test_profile_model = build_profiles(queries, train_ids)
        test_by_doc, test_frequency = support_index(golds, train_ids)
        train_index = [row[qid] for qid in train_ids]
        for qid in test_ids:
            profile_orders[qid] = profile_rank(questions[qid], test_profile_model, 2, 1.2, .75, .3)
            memory_rows[qid] = memory_features(
                similarities[row[qid], train_index], pools[qid], train_ids,
                golds, test_by_doc, test_frequency,
            )

        for inner in folds:
            if inner == outer:
                continue
            inner_ids = [qid for qid in train_ids if fold_for[qid] == inner]
            inner_blocked = set(map(str, dup.get(inner, [])))
            memory_ids = [
                qid for qid in train_ids
                if fold_for[qid] != inner and qid not in inner_blocked
            ]
            profile_model = build_profiles(queries, memory_ids)
            by_doc, frequency = support_index(golds, memory_ids)
            memory_index = [row[qid] for qid in memory_ids]
            for qid in inner_ids:
                profile_orders[qid] = profile_rank(questions[qid], profile_model, 2, 1.2, .75, .3)
                memory_rows[qid] = memory_features(
                    similarities[row[qid], memory_index], pools[qid], memory_ids,
                    golds, by_doc, frequency,
                )

        local_ids = train_ids + list(test_ids)
        local_pools = {qid: pools[qid] for qid in local_ids}
        rank_features = {name: {qid: frozen_rank[name][qid] for qid in local_ids} for name in frozen_rank}
        rank_features["huy_profile"] = core.rank_columns(profile_orders, local_pools)
        score_features = {name: {qid: frozen_score[name][qid] for qid in local_ids} for name in frozen_score}
        metadata = {name: {qid: value[qid] for qid in local_ids} for name, value in frozen_meta.items()}
        metadata["lal_memory"] = memory_rows

        rows = core.make_rows(config, local_pools, rank_features, score_features, metadata)
        x = np.vstack([rows[qid] for qid in train_ids])
        y = np.concatenate([
            np.asarray([doc in golds[qid] for doc in local_pools[qid]], dtype=np.int8)
            for qid in train_ids
        ])
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=2026,
        ).fit(scaler.transform(x), y)

        for qid in test_ids:
            values = model.decision_function(scaler.transform(rows[qid]))
            index = np.lexsort((np.asarray(local_pools[qid]), -values))
            predictions[qid] = [local_pools[qid][i] for i in index]
            decision_scores[qid] = {local_pools[qid][i]: float(values[i]) for i in range(len(values))}
            saved_rows[qid] = rows[qid]

        elapsed = time.perf_counter() - fold_started
        print(f"[{outer}] completed in {elapsed:.1f}s", flush=True)

    actual_metrics = core.metrics(predictions, golds, folds)
    print("Baseline Metrics:", json.dumps(actual_metrics, indent=2), flush=True)

    # Check metric parity
    differences = {}
    tolerance = 1e-9
    parity_pass = True

    for k in ["recall_at_5", "precision_at_5", "single_gold_recall_at_5", "multi_gold_recall_at_5"]:
        diff = abs(actual_metrics[k] - EXPECTED_METRICS[k])
        differences[k] = diff
        if diff > tolerance:
            parity_pass = False

    for fold_k, exp_v in EXPECTED_METRICS["per_fold_recall_at_5"].items():
        act_v = actual_metrics["per_fold_recall_at_5"][fold_k]
        diff = abs(act_v - exp_v)
        differences[f"per_fold_{fold_k}"] = diff
        if diff > tolerance:
            parity_pass = False

    # Snapshot file hashes
    snapshot_files = [
        BASE_SNAPSHOT_DIR / "run_huy_5fold_fasttrack.py",
        BASE_SNAPSHOT_DIR / "run_huy_memory_port.py",
        BASE_SNAPSHOT_DIR / "materialize_public_candidate.py",
        BASE_SNAPSHOT_DIR / "verify_public_candidate.py",
    ]
    file_hashes = {p.name: sha256(p) for p in snapshot_files}

    pool_hash = sha256(core.POOL_PATH)
    folds_hash = sha256(core.FOLDS_PATH)

    parity_report = {
        "schema_version": "dsc2026.gemini.rabr_v1.baseline_parity.v1",
        "parity_pass": parity_pass,
        "endpoint": "profile_memory_plus_sparse_rank_scores",
        "qid_count": len(predictions),
        "candidate_pool_hash": pool_hash,
        "fold_hash": folds_hash,
        "prediction_source": "gemini_namespace_reproduction",
        "snapshot_file_hashes": file_hashes,
        "expected_metrics": EXPECTED_METRICS,
        "actual_metrics": actual_metrics,
        "absolute_differences": differences,
        "tolerance": tolerance,
        "runtime_seconds": time.perf_counter() - started,
    }

    parity_json_path = OUT_DIR / "BASELINE_PARITY.json"
    with parity_json_path.open("w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {parity_json_path}", flush=True)

    # Save cached predictions and decision scores for downstream RABR
    cache_path = CACHE_DIR / "BASELINE_PREDICTIONS_AND_SCORES.jsonl"
    with cache_path.open("w", encoding="utf-8") as f:
        for qid in sorted(predictions, key=int):
            record = {
                "qid": qid,
                "order": predictions[qid],
                "scores": decision_scores[qid],
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Cached baseline predictions and scores to {cache_path}", flush=True)

    feat_cache_path = CACHE_DIR / "BASELINE_FEATURE_ROWS.npz"
    np.savez_compressed(
        feat_cache_path,
        **{f"row_{qid}": saved_rows[qid] for qid in saved_rows}
    )
    print(f"Cached feature rows to {feat_cache_path}", flush=True)

    if not parity_pass:
        print("ERROR: Baseline parity FAILED! Differences:", json.dumps(differences, indent=2), file=sys.stderr)
        sys.exit(1)
    else:
        print("SUCCESS: Baseline parity PASSED perfectly!", flush=True)


if __name__ == "__main__":
    main()
