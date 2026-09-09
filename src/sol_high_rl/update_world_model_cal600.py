"""Update the campaign world model with the sealed CAL600 protocol and Round 8."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "results/sol_high_rl/WORLD_MODEL.json"


def main():
    world = json.loads(PATH.read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "results/sol_high_rl/CAL600_CANONICAL_BASELINE_REPORT.json").read_text(encoding="utf-8"))
    parity = json.loads((ROOT / "results/sol_high_rl/AITEAM_FT_INFERENCE_PARITY.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_REPORT.json").read_text(encoding="utf-8"))
    fallback_path = ROOT / "results/sol_high_rl/FULLBODY_RANKING_SIGNAL_REPORT.json"
    fallback = json.loads(fallback_path.read_text(encoding="utf-8")) if fallback_path.exists() else None
    world["status"] = "CONVERGED_NO_SUBMISSION_CANDIDATE"
    world["protocol_revision"] = {
        "population": "CAL600",
        "semantics": "clean calibration/validation holdout for lower-level fine-tuned scorers; supervised meta-ranking is outer-fold isolated",
        "canonical_primary": baseline["protocol"],
        "old_four_blocks": "secondary stress slices only; prior four-block LOBO retained as a historical anchor",
        "split_sensitivity": baseline["canonical_vs_old"],
    }
    world["canonical_cal600"] = {
        "baseline_metrics": baseline["baseline"],
        "folds": baseline["folds"],
        "old_block_stress": baseline["old_block_stress"],
        "candidate_ceiling": baseline["candidate_ceiling"],
        "ranking_headroom_to_current_pool_ceiling": baseline["candidate_ceiling"]["pooled"] - baseline["baseline"]["recall_at_5"],
        "unreachable_candidate_mass": 1.0 - baseline["candidate_ceiling"]["pooled"],
        "candidate_contract_sha256": baseline["candidate_contract_sha256"],
        "predictions_sha256": baseline["predictions_sha256"],
    }
    world["aiteam_ft_full_corpus"] = {
        "status": "REJECT_CANDIDATE_SOURCE",
        "contract_parity": {"status": parity["status"], "aggregate": parity["aggregate"]},
        "index_integrity": source["index_integrity"],
        "top20": source["depths"]["20"],
        "top50": source["depths"]["50"],
        "candidate_gate": source["candidate_gate"],
        "belief": "top20 adds only three out-of-pool golds, all old block d; top50 is too diffuse. Do not append candidates or explore cap/depth variants.",
    }
    world["current_beliefs"] = {
        "candidate_vs_ranking": "canonical ranking headroom is 0.037778 inside the existing pool versus 0.015278 missing-pool mass; prioritize one orthogonal ranking representation",
        "closed": [
            "full-corpus title candidate source",
            "existing-source broad/bounded depth expansion",
            "within-parent BM25 evidence",
            "within-parent dense evidence pilot (weak/deprioritized negative)",
            "direct-utility Ridge",
            "AITeamVN-FT full-corpus candidate append",
        ],
        "next_single_hypothesis": None,
        "convergence_reason": "AITeam candidate branch failed its preregistered stability gate; the one permitted orthogonal ranking fallback was positive but noise-level and below promotion threshold",
    }
    if fallback is not None:
        world["fullbody_ranking_fallback"] = {
            "status": fallback["status"],
            "baseline_recall": fallback["baseline"]["recall_at_5"],
            "experiment_recall": fallback["experiment"]["recall_at_5"],
            "recall_delta": fallback["recall_delta"],
            "paired": fallback["paired"],
            "fold_deltas": {name: row["recall_delta"] for name, row in fallback["folds"].items()},
            "old_block_deltas": {name: row["recall_delta"] for name, row in fallback["old_block_stress"].items()},
            "multi_gold": fallback["slices"]["multi"],
            "belief": "substantively different representation but only +0.001667 on CAL600, CI crosses zero; reject without tuning",
        }
    temp = PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(world, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(PATH)


if __name__ == "__main__":
    main()
