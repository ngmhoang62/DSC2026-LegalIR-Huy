"""Audit training population and negative sampling safety for Jina continuation."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src/huy_fasttrack") not in sys.path:
    sys.path.insert(0, str(ROOT / "src/huy_fasttrack"))

import run_huy_5fold_fasttrack as core
from src.gemini.huy_d1_lal_case_memory_v1.audit_data_isolation import load_duplicate_graph
from tune_corpus_cap32_fusion import build_training_cap
import src.gemini.huy_d1_jina_ft_continuation_v1.common as common

OUT_PATH = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/JINA_TRAIN_DATA_AUDIT.json"


def sample_query_negatives(
    qid: str,
    pool: List[str],
    gold: Set[str],
    h_local: Dict[str, List[str]],
    v2_doc_ids: Set[str],
    seed: int = 2026,
) -> List[str]:
    """Sample deterministic negatives per query following Section 5 recipe."""
    rng = random.Random(seed + int(qid))

    # up to 4 non-gold from ranks 1-12
    r1_12 = [d for d in pool[:12] if d not in gold]
    hard = rng.sample(r1_12, min(4, len(r1_12)))
    selected = list(hard)

    # up to 2 non-gold from ranks 13-30
    r13_30 = [d for d in pool[12:30] if d not in gold and d not in selected]
    semi = rng.sample(r13_30, min(2, len(r13_30)))
    selected.extend(semi)

    # up to 1 diversity negative from H_LOCAL
    if str(qid) in h_local:
        for d in h_local[str(qid)]:
            if d in v2_doc_ids and d not in gold and d not in selected:
                selected.append(d)
                break

    return selected


def run_audit() -> dict:
    print("Loading V2 inputs...", flush=True)
    folds, pools, questions, v2_golds, _, _, _, _ = core.load_inputs()
    v2_qids = set(str(q) for q in pools.keys())

    print("Loading CAL600 ids...", flush=True)
    _, _, all_ids, _, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    cal_qids = set(str(q) for q in all_ids)

    print("Loading duplicate graph...", flush=True)
    _, _, _, dup_map = load_duplicate_graph()
    cal_dups = {str(dup) for q in cal_qids for dup in dup_map.get(q, set())}
    forbidden = cal_qids | cal_dups
    eligible_v2 = sorted(list(v2_qids - forbidden), key=int)

    contexts = common.load_contexts()
    v2_doc_ids = set(contexts.keys())
    h_local = common.load_h_local()

    print("Sampling training candidate groups and auditing safety...", flush=True)
    training_queries = []
    skipped_queries = []
    single_gold = 0
    multi_gold = 0
    total_pos_parents = 0
    total_negs_sampled = 0

    for q in eligible_v2:
        pool = [str(d) for d in pools[q]]
        gold = {str(d) for d in v2_golds.get(q, set())}
        positives = [d for d in pool if d in gold]

        if not positives:
            skipped_queries.append({"qid": q, "reason": "no_usable_positive_in_pool"})
            continue

        if len(positives) == 1:
            single_gold += 1
        else:
            multi_gold += 1

        total_pos_parents += len(positives)
        negs = sample_query_negatives(q, pool, gold, h_local, v2_doc_ids, seed=2026)

        assert len(set(positives) & set(negs)) == 0, f"Overlap in query {q}"
        assert len(negs) <= 7, f"Excess negatives in query {q}"
        total_negs_sampled += len(negs)
        training_queries.append(q)

    # Assert zero overlap with CAL and duplicates
    train_set = set(training_queries)
    assert len(train_set & cal_qids) == 0, "FATAL: overlap with CAL600"
    assert len(train_set & cal_dups) == 0, "FATAL: overlap with CAL duplicates"

    audit = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "total_v2_evaluable_queries": len(v2_qids),
        "cal600_in_v2": len(cal_qids & v2_qids),
        "cal_duplicate_linked_in_v2": len(cal_dups & v2_qids),
        "total_forbidden_queries": len(forbidden & v2_qids),
        "eligible_v2_queries": len(eligible_v2),
        "training_population_count": len(training_queries),
        "skipped_queries_count": len(skipped_queries),
        "skipped_queries": skipped_queries,
        "single_gold_count": single_gold,
        "multi_gold_count": multi_gold,
        "total_positive_parents": total_pos_parents,
        "total_negatives_sampled": total_negs_sampled,
        "mean_negatives_per_query": total_negs_sampled / len(training_queries),
        "cal_overlap_count": len(train_set & cal_qids),
        "cal_duplicate_overlap_count": len(train_set & cal_dups),
        "gold_negative_overlap_count": 0,
        "status": "PASS",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"Wrote {OUT_PATH}", flush=True)
    return audit


if __name__ == "__main__":
    run_audit()
