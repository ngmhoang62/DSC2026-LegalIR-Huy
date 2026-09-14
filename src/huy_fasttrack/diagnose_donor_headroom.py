"""Unsafe-lineage donor predictions used only to choose the next mechanism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_huy_5fold_fasttrack as core


DONOR = core.WORKSPACE / "LegalIR/results"
FILES = {
    "gemini_best_ensemble": DONOR / "gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json",
    "gemini_h66_xgb145d": DONOR / "gemini/exp_xgb_145d_d5/H66_OOF_PREDICTIONS.json",
    "gemini_action_utility": DONOR / "gemini/exp_action_utility_top25/ACTION_UTILITY_TOP25_PREDICTIONS.json",
    "legalir_profile_l15": DONOR / "exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json",
    "legalir_nested_slate": DONOR / "exp_final_retrieval/nested_slate_probe/OOF_PREDICTIONS.json",
    "legalir_nested_content_slate": DONOR / "exp_final_retrieval/nested_content_slate_probe/OOF_PREDICTIONS.json",
}


def load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for qid, value in payload.items():
        if isinstance(value, dict):
            value = value.get("order") or value.get("top5") or value.get("answer")
        result[str(qid)] = list(map(str, value))
    return result


def recall(pred, golds):
    return float(np.mean([
        len(set(pred[qid][:5]) & golds[qid]) / len(golds[qid])
        for qid in golds
    ]))


def union_oracle(left, right, golds, depth):
    values = []
    for qid, gold in golds.items():
        candidates = set(left[qid][:depth]) | set(right[qid][:depth])
        values.append(min(5, len(candidates & gold)) / len(gold))
    return float(np.mean(values))


def main():
    folds, pools, questions, golds, *_ = core.load_inputs()
    current = {
        str(row["qid"]): list(map(str, row["top5"]))
        for row in core.read_jsonl(core.OUT / "BEST_LAL_MEMORY_PREDICTIONS.jsonl")
    }
    current_score = recall(current, golds)
    rows = []
    for name, path in FILES.items():
        donor = load(path)
        if not set(golds) <= set(donor):
            continue
        rows.append({
            "name": name,
            "path": str(path),
            "unsafe_old_fold_recall_at_5": recall(donor, golds),
            "top5_union_choice_oracle_with_current": union_oracle(current, donor, golds, 5),
            "top20_union_candidate_oracle_with_current": union_oracle(current, donor, golds, 20),
            "paired_vs_current": core.compare(donor, current, golds, folds),
        })
    rows.sort(key=lambda item: item["top5_union_choice_oracle_with_current"], reverse=True)
    report = {
        "status": "DIAGNOSTIC_ONLY_UNSAFE_FOR_SELECTION",
        "warning": "Donor OOF fold lineage differs from V2; scores select a mechanism to reimplement, never a candidate.",
        "current_strict_v2_recall_at_5": current_score,
        "donors": rows,
    }
    core.write_json(core.OUT / "DONOR_HEADROOM_DIAGNOSTIC.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
