"""Merge immutable long-horizon observations into WORLD_MODEL without rewriting history."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "results/sol_high_rl/WORLD_MODEL.json"
world = json.loads(path.read_text(encoding="utf-8"))
anatomy = json.loads((ROOT / "results/sol_high_rl/RANKING_FAILURE_ANATOMY_V2.json").read_text(encoding="utf-8"))
world["status"] = "ACTIVE_LONG_HORIZON_RESEARCH"
world["ranking_failure_anatomy_v2"] = {
    "artifact": "results/sol_high_rl/RANKING_FAILURE_ANATOMY_V2.json",
    "protocol_lock": anatomy["protocol_lock"],
    "candidate_oracle": anatomy["candidate_oracle"],
    "headroom": anatomy["headroom"],
    "miss_occurrence_bins": anatomy["miss_decomposition"]["occurrence_counts"],
    "exclusive_failure_class_recall_mass": anatomy["miss_decomposition"]["exclusive_failure_class_recall_mass"],
    "multi_gold": {k: anatomy["multi_gold"][k] for k in (
        "queries", "candidate_oracle_recall", "baseline_recall", "expert_top5_union_recall",
        "average_reachable_gold_count", "queries_with_reachable_missed_gold")},
}
world["anti_adaptive_protocol"] = {
    "development_folds": ["fold_0", "fold_1", "fold_2"],
    "confirmation_folds": ["fold_3", "fold_4"],
    "confirmation_family_budget": 2,
    "confirmation_families_used": 0,
    "policy": "do not inspect confirmation-fold gold outcomes for micro-variants; preregister and promote one mechanism first",
}
world["current_beliefs"]["candidate_vs_ranking"] = (
    "existing expert Top-5 union oracle 0.978056 is +0.031111 over canonical baseline and captures 60.2% "
    "of missed mass; fusion/capacity dominates, while consensus-blind representation mass is only 0.001667"
)
world["current_beliefs"]["next_single_hypothesis"] = "F1 fixed historical pairwise/LambdaRank capacity audit on DEV folds 0-2"
encoded = (json.dumps(world, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
tmp = path.with_suffix(".json.tmp")
tmp.write_bytes(encoded)
tmp.replace(path)
print("updated", path)
