"""Cache-only fixed-prior fusion of strict-V2 learner prediction locks."""

from __future__ import annotations

import json

import run_huy_5fold_fasttrack as core


LOCK = core.OUT / "learner_prediction_locks"


def load(name):
    return {
        str(row["qid"]): list(map(str, row["order"]))
        for row in core.read_jsonl(LOCK / f"{name}.jsonl")
    }


def fuse(systems, weights, depths, offset):
    qids = next(iter(systems.values())).keys()
    result = {}
    for qid in qids:
        scores = {}
        for name, weight, depth in zip(systems, weights, depths):
            for rank, doc in enumerate(systems[name][qid][:depth], 1):
                scores[doc] = scores.get(doc, 0.0) + weight / (offset + rank)
        result[qid] = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return result


def main():
    folds, pools, questions, golds, *_ = core.load_inputs()
    names = [
        "memory_winner_no_doctype",
        "xgb_d4_profile_memory",
        "xgb_d5_profile_memory",
        "lgbm_l7_t30_profile_memory",
        "lal_memory_no_profile",
    ]
    systems = {name: load(name) for name in names}
    baseline = systems["memory_winner_no_doctype"]
    configs = {
        "equal4_rrf0": (["memory_winner_no_doctype", "xgb_d4_profile_memory", "xgb_d5_profile_memory", "lgbm_l7_t30_profile_memory"], [1, 1, 1, 1], [20, 20, 20, 20], 0),
        "equal4_rrf32": (["memory_winner_no_doctype", "xgb_d4_profile_memory", "xgb_d5_profile_memory", "lgbm_l7_t30_profile_memory"], [1, 1, 1, 1], [20, 20, 20, 20], 32),
        "huy_stack_prior": (["memory_winner_no_doctype", "xgb_d4_profile_memory", "lgbm_l7_t30_profile_memory", "lal_memory_no_profile"], [.40, .30, .15, .15], [20, 20, 20, 20], 0),
        "legalir_h66_proxy": (["xgb_d5_profile_memory", "lgbm_l7_t30_profile_memory", "xgb_d4_profile_memory", "memory_winner_no_doctype"], [.41, .29, .12, .18], [18, 10, 15, 15], 0),
        "lr_heavy_prior": (["memory_winner_no_doctype", "xgb_d4_profile_memory", "lgbm_l7_t30_profile_memory", "xgb_d5_profile_memory"], [.50, .25, .15, .10], [15, 18, 10, 18], 0),
        "lr_xgb_pair": (["memory_winner_no_doctype", "xgb_d4_profile_memory"], [.60, .40], [20, 20], 0),
    }
    rows = []
    prediction = {}
    for name, (members, weights, depths, offset) in configs.items():
        pred = fuse({member: systems[member] for member in members}, weights, depths, offset)
        prediction[name] = pred
        rows.append({
            "name": name,
            "members": members,
            "weights": weights,
            "depths": depths,
            "rrf_offset": offset,
            "metrics": core.metrics(pred, golds, folds),
            "paired_vs_lr": core.compare(pred, baseline, golds, folds),
        })
    rows.sort(key=lambda item: item["metrics"]["recall_at_5"], reverse=True)
    report = {
        "status": "COMPLETE_FIXED_PRIOR_CACHE_ONLY_SCREEN",
        "selection_note": "All formulas frozen from historical Huy/LegalIR patterns; no V2 grid tuning.",
        "baseline_metrics": core.metrics(baseline, golds, folds),
        "best": rows[0],
        "results": rows,
    }
    core.write_json(core.OUT / "HUY_LEARNER_FUSION_SCREEN.json", report)
    with (core.OUT / "BEST_LEARNER_FUSION_PREDICTIONS.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(pools, key=int):
            f.write(json.dumps({"qid": qid, "top5": prediction[rows[0]["name"]][qid][:5]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
