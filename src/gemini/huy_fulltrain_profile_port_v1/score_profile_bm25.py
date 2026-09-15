"""Score supervised BM25 document profiles with nested isolation on CAL and Public candidate pools."""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tune_burst_supervised_profile_bm25 import build_profiles, features, profile_rank

sys.path.insert(0, str(ROOT / "src" / "huy_fasttrack"))
import run_huy_5fold_fasttrack as core

from profile_data_isolation import (
    get_dup_linked,
    load_linked_duplicates,
    load_populations,
)


def load_cal_questions_and_golds(blocks, cal_ids):
    train_path = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "train.json"
    raw = json.loads(train_path.read_text(encoding="utf-8"))
    questions = {q: raw[q]["question"] for q in cal_ids}
    golds = {q: set(map(str, raw[q].get("answer", []))) for q in cal_ids}
    return questions, golds


def load_public_questions_and_candidates():
    test_path = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "test.json"
    raw_test = json.loads(test_path.read_text(encoding="utf-8"))
    questions = {str(q): row["question"] for q, row in raw_test.items()}
    public_ids = sorted(questions.keys(), key=int)

    # Load candidate pool from production submission cache
    # Candidate pool is identical to public_candidates in audit_baseline_contracts
    base_pub = pickle.loads((ROOT / "results/burst_gpu_threeview/cpu_top20.pkl").read_bytes())["rankings"]
    rerank_path = ROOT / "results/expanded_rerank/scores.pkl"
    # Or load public candidates from cache or reconstruct
    # Let's inspect how public_candidates is constructed in audit_baseline_contracts
    return questions, public_ids


def compute_lexical_support(text: str, postings: dict, max_ngram: int = 2) -> int:
    feats = features(text, max_ngram)
    return sum(1 for feat in feats if feat in postings)


def main():
    started = time.perf_counter()
    out_dir = ROOT / "results" / "gemini" / "huy_fulltrain_profile_port_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        v2_qids,
        v2_set,
        blocks,
        cal_ids,
        cal_set,
        cal_in_v2,
        missing_cal,
        non_cal_in_v2,
    ) = load_populations()
    links = load_linked_duplicates()

    # Load canonical queries dict for build_profiles: {qid: (question, golds)}
    _, _, v2_questions, v2_golds, _, _, _, _ = core.load_inputs()
    v2_queries = {qid: (v2_questions[qid], v2_golds[qid]) for qid in v2_qids}

    # For CAL questions and golds
    cal_questions, cal_golds = load_cal_questions_and_golds(blocks, cal_ids)

    print(f"Loaded {len(v2_queries)} V2 queries and {len(cal_questions)} CAL queries.")

    # 1. Build Nested LOBO Profile Rankings for CAL evaluation
    # For each held block H in [a, b, c, d]:
    #   rankings for queries in H come from model(M(H))
    #   rankings for queries in T != H come from model(M(H, T))
    nested_cal_rankings = {}
    cal_lexical_support = {}
    cal_oof_rankings = {}

    print("Building nested CAL profile rankings...")
    for held_name, held_ids in blocks.items():
        held_set = set(held_ids)
        held_dup = get_dup_linked(held_ids, links)
        mem_held = sorted(v2_set - held_set - held_dup, key=int)

        # Model for held block
        model_held = build_profiles(v2_queries, mem_held)

        held_ranks = {}
        for q in held_ids:
            q_text = cal_questions[q]
            ranked = profile_rank(q_text, model_held, 2, 1.2, 0.75, 0.3)
            held_ranks[q] = ranked
            cal_lexical_support[q] = compute_lexical_support(q_text, model_held[0], 2)
            cal_oof_rankings[q] = ranked

        # Models for training blocks T != H
        train_ranks = {}
        for train_name, train_ids in blocks.items():
            if train_name == held_name:
                continue
            train_set = set(train_ids)
            train_dup = get_dup_linked(train_ids, links)
            mem_train = sorted(
                v2_set - held_set - train_set - held_dup - train_dup, key=int
            )
            model_train = build_profiles(v2_queries, mem_train)
            for q in train_ids:
                q_text = cal_questions[q]
                train_ranks[q] = profile_rank(q_text, model_train, 2, 1.2, 0.75, 0.3)

        nested_cal_rankings[held_name] = {
            "held_ranks": held_ranks,
            "train_ranks": train_ranks,
        }

    # 2. Build Full 6991 Profile Model for Public test queries
    print("Building full 6991 profile model for Public test...")
    full_mem = sorted(v2_set, key=int)
    full_model = build_profiles(v2_queries, full_mem)

    test_path = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "public-official.json"
    raw_test = json.loads(test_path.read_text(encoding="utf-8"))
    public_questions = {str(q): row["question"] for q, row in raw_test.items()}
    public_ids = sorted(public_questions.keys(), key=int)

    public_rankings = {}
    public_lexical_support = {}
    for q in public_ids:
        q_text = public_questions[q]
        public_rankings[q] = profile_rank(q_text, full_model, 2, 1.2, 0.75, 0.3)
        public_lexical_support[q] = compute_lexical_support(q_text, full_model[0], 2)

    # 3. Standalone Retrieval Diagnostic on CAL600 using OOF profiles
    # Also load H0 predictions to compute oracle union
    h0_pred_path = ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2" / "H0_REBUILT.json"
    # Recompute or load H0 CAL predictions
    # Let's compute standalone metrics for profile alone
    cal_recalls_at_k = {1: 0.0, 5: 0.0, 8: 0.0, 10: 0.0}
    single_gold_hits_5 = 0.0
    single_gold_total = 0
    multi_gold_hits_5 = 0.0
    multi_gold_total = 0

    for q in cal_ids:
        gold = cal_golds[q]
        if not gold:
            continue
        ranked = cal_oof_rankings[q]
        for k in cal_recalls_at_k:
            top_k = set(ranked[:k])
            cal_recalls_at_k[k] += len(gold & top_k) / len(gold)
        if len(gold) == 1:
            single_gold_hits_5 += len(gold & set(ranked[:5])) / len(gold)
            single_gold_total += 1
        else:
            multi_gold_hits_5 += len(gold & set(ranked[:5])) / len(gold)
            multi_gold_total += 1

    n_cal = len(cal_ids)
    standalone_metrics = {
        "recall_at_1": cal_recalls_at_k[1] / n_cal,
        "recall_at_5": cal_recalls_at_k[5] / n_cal,
        "recall_at_8": cal_recalls_at_k[8] / n_cal,
        "recall_at_10": cal_recalls_at_k[10] / n_cal,
        "single_gold_recall_at_5": single_gold_hits_5 / max(single_gold_total, 1),
        "multi_gold_recall_at_5": multi_gold_hits_5 / max(multi_gold_total, 1),
        "queries_evaluated": n_cal,
    }

    # Save cached profile rankings
    cache_payload = {
        "nested_cal_rankings": nested_cal_rankings,
        "cal_oof_rankings": cal_oof_rankings,
        "cal_lexical_support": cal_lexical_support,
        "public_rankings": public_rankings,
        "public_lexical_support": public_lexical_support,
    }
    cache_file = out_dir / "PROFILE_BM25_RANKINGS.pkl"
    with cache_file.open("wb") as f:
        pickle.dump(cache_payload, f)

    # Standalone diagnostic report
    report = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.profile_standalone_report.v1",
        "component": "Huy supervised label-profile BM25 (full 6991 memory)",
        "recipe": {
            "max_ngram": 2,
            "k1": 1.2,
            "b": 0.75,
            "prior_power": 0.3,
        },
        "metrics": standalone_metrics,
        "oracle_union_h0": {
            "note": "Oracle union evaluated in evaluate_profile_dual_cal.py alongside baseline predictions."
        },
        "runtime_seconds": time.perf_counter() - started,
    }

    report_file = out_dir / "PROFILE_STANDALONE_REPORT.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"PROFILE_STANDALONE_REPORT:")
    print(f"  Recall@1:  {standalone_metrics['recall_at_1']:.4f}")
    print(f"  Recall@5:  {standalone_metrics['recall_at_5']:.4f}")
    print(f"  Recall@8:  {standalone_metrics['recall_at_8']:.4f}")
    print(f"  Recall@10: {standalone_metrics['recall_at_10']:.4f}")
    print(f"  Single R@5: {standalone_metrics['single_gold_recall_at_5']:.4f}")
    print(f"  Multi R@5:  {standalone_metrics['multi_gold_recall_at_5']:.4f}")
    print(f"Wrote {report_file} and {cache_file} in {report['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
