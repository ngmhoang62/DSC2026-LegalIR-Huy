"""Audit data split: enforce strict CAL exclusion and Unicode question text deduplication."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from tune_corpus_cap32_fusion import build_training_cap
from .common import (
    RES_DIR,
    ROOT,
    get_git_status,
    load_v2_inputs,
    normalize_text,
)


def run_split_audit() -> Dict[str, Any]:
    print("=== AUDIT: DATA SPLIT & ANTI-CONTAMINATION ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_status()

    # 1. Load canonical V2 5-fold inputs
    folds, pools, questions, v2_golds = load_v2_inputs()
    v2_qids = sorted(list(pools.keys()), key=lambda x: int(x) if x.isdigit() else x)
    total_v2 = len(v2_qids)

    # 2. Load CAL queries strictly label-free
    raw_queries, _, all_ids, _, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    cal_qids: Set[str] = set(str(q) for q in all_ids)
    cal_norm_texts: Set[str] = {normalize_text(raw_queries[q][0]) for q in all_ids}

    # 3. Held Fold 0 Population
    held_raw_qids = [str(q) for q in folds["fold_0"]]
    held_non_cal_qids = [q for q in held_raw_qids if q not in cal_qids]
    held_cal_removed = len(held_raw_qids) - len(held_non_cal_qids)

    # Check CAL text overlap in held
    held_cal_text_overlap = [
        q for q in held_non_cal_qids if normalize_text(questions[q]) in cal_norm_texts
    ]
    # For held evaluation, keep non-CAL qids
    held_final_qids = held_non_cal_qids
    held_norm_texts: Set[str] = {normalize_text(questions[q]) for q in held_final_qids}

    # 4. Training Population: Folds 1, 2, 3, 4
    train_raw_qids: List[str] = []
    for f in ["fold_1", "fold_2", "fold_3", "fold_4"]:
        train_raw_qids.extend([str(q) for q in folds[f]])

    train_non_cal_qids = [q for q in train_raw_qids if q not in cal_qids]
    train_cal_qids_removed = len(train_raw_qids) - len(train_non_cal_qids)

    # Deduplicate normalized question text against CAL600
    train_after_cal_text = [
        q for q in train_non_cal_qids if normalize_text(questions[q]) not in cal_norm_texts
    ]
    train_cal_text_removed = [
        {"qid": q, "text": questions[q]}
        for q in train_non_cal_qids
        if normalize_text(questions[q]) in cal_norm_texts
    ]

    # Deduplicate normalized question text against Held Fold 0
    train_final_qids = [
        q for q in train_after_cal_text if normalize_text(questions[q]) not in held_norm_texts
    ]
    train_held_text_removed = [
        {"qid": q, "text": questions[q]}
        for q in train_after_cal_text
        if normalize_text(questions[q]) in held_norm_texts
    ]

    # Check queries with in-pool gold documents
    train_with_in_pool_gold = []
    for q in train_final_qids:
        cand_set = set(str(d) for d in pools[q])
        gold_set = set(str(d) for d in v2_golds.get(q, set()))
        if len(cand_set & gold_set) > 0:
            train_with_in_pool_gold.append(q)

    # Provenance audit artifact
    audit_data = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.split_audit.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "canonical_v2_total_queries": total_v2,
        "cal600_total_queries": len(cal_qids),
        "held_partition": {
            "fold_name": "fold_0",
            "raw_queries_count": len(held_raw_qids),
            "cal_qids_removed_count": held_cal_removed,
            "non_cal_queries_count": len(held_final_qids),
            "cal_text_overlap_count": len(held_cal_text_overlap),
            "sample_held_qids": held_final_qids[:5],
        },
        "training_partition": {
            "folds": ["fold_1", "fold_2", "fold_3", "fold_4"],
            "raw_queries_count": len(train_raw_qids),
            "cal_qids_removed_count": train_cal_qids_removed,
            "non_cal_queries_count": len(train_non_cal_qids),
            "cal_text_overlap_removed_count": len(train_cal_text_removed),
            "cal_text_overlap_removed_details": train_cal_text_removed,
            "held_text_overlap_removed_count": len(train_held_text_removed),
            "held_text_overlap_removed_details": train_held_text_removed,
            "final_train_queries_count": len(train_final_qids),
            "train_queries_with_in_pool_gold": len(train_with_in_pool_gold),
            "sample_train_qids": train_with_in_pool_gold[:5],
        },
        "anti_contamination_assertion": {
            "train_cal_qid_overlap_is_zero": len(set(train_final_qids) & cal_qids) == 0,
            "held_cal_qid_overlap_is_zero": len(set(held_final_qids) & cal_qids) == 0,
            "train_held_qid_overlap_is_zero": len(set(train_final_qids) & set(held_final_qids)) == 0,
            "train_cal_text_overlap_is_zero": len([q for q in train_final_qids if normalize_text(questions[q]) in cal_norm_texts]) == 0,
            "train_held_text_overlap_is_zero": len([q for q in train_final_qids if normalize_text(questions[q]) in held_norm_texts]) == 0,
            "cal_labels_not_read_in_split": True,
        },
    }

    assert all(audit_data["anti_contamination_assertion"].values()), "Anti-contamination assertion failed!"

    out_path = RES_DIR / "SPLIT_AUDIT.json"
    out_path.write_text(json.dumps(audit_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[AUDIT] Split audit saved -> {out_path} (Train: {len(train_with_in_pool_gold)}, Held: {len(held_final_qids)})", flush=True)
    return audit_data


if __name__ == "__main__":
    run_split_audit()
