"""One bounded LightGBM screen on the current Huy-first best feature set."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

import run_huy_5fold_fasttrack as core


def load_reference(path: Path):
    return {
        str(row["qid"]): list(map(str, row["top5"]))
        for row in core.read_jsonl(path)
    }


def main():
    started = time.perf_counter()
    core.OUT.mkdir(parents=True, exist_ok=True)
    folds, pools, questions, golds, e5_orders, e5_scores, dup, _ = core.load_inputs()
    jina_order, jina_scores, _ = core.load_jina(pools)
    lal_order, lal_scores, _ = core.load_source_channel("lal", pools)
    heads = core.document_heads(pools)
    doctype, citation = core.metadata_arrays(pools, questions, heads)
    orders = {
        "jina_ce": jina_order,
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
    }
    score_maps = {
        "jina_ce": jina_scores,
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
    }
    rank_features = {name: core.rank_columns(values, pools) for name, values in orders.items()}
    score_features = {name: core.score_columns(values, pools) for name, values in score_maps.items()}
    metadata = {"doctype": doctype, "citation": citation}
    config = {
        "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
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
        model = LGBMClassifier(
            objective="binary",
            n_estimators=240,
            learning_rate=.04,
            num_leaves=15,
            max_depth=5,
            min_child_samples=100,
            subsample=.9,
            colsample_bytree=.9,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=2026,
            deterministic=True,
            force_col_wise=True,
            n_jobs=8,
            verbosity=-1,
        )
        model.fit(x, y)
        for qid in test_ids:
            values = model.predict_proba(rows[qid])[:, 1]
            predictions[qid] = [
                pools[qid][i] for i in np.lexsort((np.asarray(pools[qid]), -values))
            ]
        fold_runtime[fold] = time.perf_counter() - fold_started

    reference_path = core.OUT / "BEST_5FOLD_PREDICTIONS.jsonl"
    reference = load_reference(reference_path)
    report = {
        "schema_version": "dsc2026.huy_fasttrack.fusion_screen.v1",
        "status": "COMPLETE_5FOLD_OOF",
        "model": {
            "family": "LightGBM binary classifier",
            "parameters": model.get_params(),
            "feature_contract": config,
            "feature_count": next(iter(rows.values())).shape[1],
        },
        "metrics": core.metrics(predictions, golds, folds),
        "paired_vs_lr_best": core.compare(predictions, reference, golds, folds),
        "lr_reference": json.loads((core.OUT / "HUY_5FOLD_BASELINE.json").read_text(encoding="utf-8"))[
            "available_component_screen"
        ]["huy_plus_adapted_e5_lal"]["metrics"],
        "runtime_seconds": time.perf_counter() - started,
        "fold_runtime_seconds": fold_runtime,
        "decision": "KEEP_IF_DELTA_GTE_0_001_ELSE_DROP",
    }
    delta = report["paired_vs_lr_best"]["delta_recall_at_5"]
    report["verdict"] = "KEEP" if delta >= .001 else "DROP"
    core.write_json(core.OUT / "HUY_FUSION_SCREEN.json", report)
    out = core.OUT / "LGBM_5FOLD_PREDICTIONS.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(pools, key=int):
            f.write(json.dumps({"qid": qid, "top5": predictions[qid][:5]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
