"""Merge late profile/memory/sparse ablations into the required Pareto report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/huy_fasttrack"
PARAMETERS = {
    "jina_ce": 278_043_648,
    "adapted_e5": 559_890_432,
    "frozen_e5": 559_890_432,
    "lal_native": 596_049_920,
    "lal_memory": 596_049_920,
    "legalir_jina": 571_754_368,
}


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def complexity(config):
    rank = config.get("rank_views", [])
    score = config.get("score_channels", [])
    meta = config.get("metadata", [])
    keys = set(rank) | set(score) | set(meta)
    models = []
    if "jina_ce" in keys:
        models.append("jinaai/jina-reranker-v2-base-multilingual")
    if "adapted_e5" in keys or "frozen_e5" in keys:
        models.append("mainguyen9/vietlegal-e5")
    if "lal_native" in keys or "lal_memory" in keys:
        models.append("darklethelong/vnlegal-lal")
    if "legalir_jina" in keys:
        models.append("jinaai/jina-embeddings-v3")
    count = sum(PARAMETERS[k] for k in (
        "jina_ce", "adapted_e5", "lal_native", "legalir_jina"
    ) if k in keys)
    return {
        "model_inference_count": len(models),
        "models": models,
        "approx_original_parameters": count,
        "rank_views": len(rank),
        "score_channels": len(score),
        "ltr": True,
    }


def main():
    base = read("HUY_ABLATION_PARETO.json")
    memory = read("HUY_LAL_MEMORY_PORT_REPORT.json")
    checkpoints = list(base["all_checkpoints"])
    for row in memory["results"]:
        config = row["config"]
        if "base" in config:
            continue
        item = {
            "name": row["name"],
            "recall_at_5": row["metrics"]["recall_at_5"],
            "precision_at_5": row["metrics"]["precision_at_5"],
            "single_gold_recall_at_5": row["metrics"]["single_gold_recall_at_5"],
            "multi_gold_recall_at_5": row["metrics"]["multi_gold_recall_at_5"],
            "per_fold_recall_at_5": row["metrics"]["per_fold_recall_at_5"],
            "feature_count": row["feature_count"],
            "config": config,
            "complexity": complexity(config),
            "paired_vs_huy_profile": row["paired_vs_profile_reference"],
            "shared_batch_runtime_seconds": memory["runtime_seconds"],
        }
        checkpoints.append(item)
    checkpoints.sort(key=lambda x: x["recall_at_5"], reverse=True)
    best = checkpoints[0]

    def smallest(delta):
        eligible = [x for x in checkpoints if x["recall_at_5"] >= best["recall_at_5"] - delta]
        return min(eligible, key=lambda x: (
            x.get("complexity", {}).get("model_inference_count", x.get("model_inference_count", 99)),
            x.get("feature_count", 999),
            x.get("complexity", {}).get("score_channels", x.get("score_channels", 99)),
            -x["recall_at_5"],
        ))

    # Keep only non-dominated model-count/feature-count/Recall checkpoints.
    pareto = []
    for row in checkpoints:
        models = row.get("complexity", {}).get("model_inference_count", row.get("model_inference_count", 99))
        features = row.get("feature_count", 999)
        if not any(
            other["recall_at_5"] >= row["recall_at_5"]
            and other.get("complexity", {}).get("model_inference_count", other.get("model_inference_count", 99)) <= models
            and other.get("feature_count", 999) <= features
            and (other["recall_at_5"] > row["recall_at_5"] or other.get("feature_count", 999) < features)
            for other in checkpoints
        ):
            pareto.append(row)

    base.update({
        "schema_version": "dsc2026.huy_fasttrack.ablation_pareto.v2",
        "status": "COMPLETE_CACHE_FIRST_STRICT_5FOLD_OOF",
        "full": best,
        "best_absolute_score": best,
        "minimal_within_minus_0_001": smallest(.001),
        "minimal_within_minus_0_003": smallest(.003),
        "minimal_within_minus_0_005": smallest(.005),
        "pareto_frontier": pareto,
        "all_checkpoints": checkpoints,
        "late_stage_notes": {
            "profile_memory_sparse_batch_runtime_seconds": memory["runtime_seconds"],
            "candidate_ceiling": 0.9819315310160681,
            "best_rank_recall_depth": {"5": 0.9488556715777428, "6": 0.9551375578129023, "7": 0.959178468030329},
        },
    })
    (OUT / "HUY_ABLATION_PARETO.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    state = {
        "schema_version": "dsc2026.huy_fasttrack.final_state.v1",
        "state": "NEW_STRICT_HUY_FIRST_ANCHOR",
        "best": {"name": best["name"], "recall_at_5": best["recall_at_5"], "precision_at_5": best["precision_at_5"]},
        "minimal_within_0_001": {"name": smallest(.001)["name"], "recall_at_5": smallest(.001)["recall_at_5"]},
        "target_0_96_reached": best["recall_at_5"] >= .96,
        "candidate_ceiling": 0.9819315310160681,
        "submission_candidate": str(OUT / "submission_candidate_with_huy_jina/submission.zip"),
    }
    (OUT / "FINAL_STATE.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
