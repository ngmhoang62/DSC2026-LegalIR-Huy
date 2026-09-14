"""Port the proven low-capacity LegalIR LambdaRank objective into Huy features."""

from __future__ import annotations

import json
import time

import numpy as np
from lightgbm import LGBMRanker

import run_huy_5fold_fasttrack as core


def main():
    started = time.perf_counter()
    folds, pools, questions, golds, e5_orders, e5_scores, dup, _ = core.load_inputs()
    jina_order, jina_scores, _ = core.load_jina(pools)
    lal_order, lal_scores, _ = core.load_source_channel("lal", pools)
    legalir_jina_order, legalir_jina_scores, _ = core.load_source_channel("jina", pools)
    heads = core.document_heads(pools)
    doctype, citation = core.metadata_arrays(pools, questions, heads)
    orders = {
        "jina_ce": jina_order,
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
        "legalir_jina": legalir_jina_order,
    }
    score_maps = {
        "jina_ce": jina_scores,
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
        "legalir_jina": legalir_jina_scores,
    }
    rank_features = {name: core.rank_columns(values, pools) for name, values in orders.items()}
    score_features = {name: core.score_columns(values, pools) for name, values in score_maps.items()}
    metadata = {"doctype": doctype, "citation": citation}
    config = {
        "rank_views": ["jina_ce", "adapted_e5", "lal_native", "legalir_jina"],
        "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
        "metadata": ["doctype", "citation"],
    }
    rows = core.make_rows(config, pools, rank_features, score_features, metadata)
    all_qids = set(pools)
    predictions = {}
    fold_runtime = {}
    for fold, test_ids in folds.items():
        fold_started = time.perf_counter()
        train_ids = sorted(
            all_qids - set(test_ids) - set(map(str, dup.get(fold, []))), key=int
        )
        x = np.vstack([rows[q] for q in train_ids])
        y = np.concatenate([
            np.asarray([doc in golds[q] for doc in pools[q]], dtype=np.int8)
            for q in train_ids
        ])
        groups = [len(pools[q]) for q in train_ids]
        model = LGBMRanker(
            objective="lambdarank",
            learning_rate=.05,
            n_estimators=300,
            num_leaves=7,
            min_child_samples=50,
            lambdarank_truncation_level=30,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            deterministic=True,
            force_col_wise=True,
            n_jobs=8,
            random_state=6112,
            verbosity=-1,
        )
        model.fit(x, y, group=groups, eval_at=[5])
        for qid in test_ids:
            values = model.predict(rows[qid])
            predictions[qid] = [
                pools[qid][i] for i in np.lexsort((np.asarray(pools[qid]), -values))
            ]
        fold_runtime[fold] = time.perf_counter() - fold_started

    reference = {
        str(row["qid"]): list(map(str, row["top5"]))
        for row in core.read_jsonl(core.OUT / "BEST_5FOLD_PREDICTIONS.jsonl")
    }
    baseline_payload = json.loads((core.OUT / "HUY_5FOLD_BASELINE.json").read_text(encoding="utf-8"))
    lr_name = baseline_payload["best_screened_variant"]
    report = {
        "schema_version": "dsc2026.huy_fasttrack.lambdarank_screen.v1",
        "status": "COMPLETE_STRICT_5FOLD_OOF",
        "donor_evidence": "LegalIR meta-LTR l7_t30 fixed config; objective only, not donor systems/features",
        "config": config,
        "model": {
            "family": "LightGBM LambdaRank",
            "num_leaves": 7,
            "min_child_samples": 50,
            "lambdarank_truncation_level": 30,
            "n_estimators": 300,
            "learning_rate": .05,
        },
        "metrics": core.metrics(predictions, golds, folds),
        "lr_reference_name": lr_name,
        "lr_reference_metrics": baseline_payload["available_component_screen"][lr_name]["metrics"],
        "paired_vs_lr_best": core.compare(predictions, reference, golds, folds),
        "fold_runtime_seconds": fold_runtime,
        "runtime_seconds": time.perf_counter() - started,
    }
    report["verdict"] = "KEEP" if report["paired_vs_lr_best"]["delta_recall_at_5"] >= .001 else "DROP"
    core.write_json(core.OUT / "HUY_LAMBDARANK_FUSION_SCREEN.json", report)
    with (core.OUT / "LAMBDARANK_5FOLD_PREDICTIONS.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(pools, key=int):
            f.write(json.dumps({"qid": qid, "top5": predictions[qid][:5]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
