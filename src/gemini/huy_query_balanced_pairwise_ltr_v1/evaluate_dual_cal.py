"""Dual CAL protocol evaluation for Q0_POINTWISE, Q1_PAIRWISE, and Q2_PAIRWISE_PLUS_PROFILE."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_query_balanced_pairwise_ltr_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "huy_fasttrack"))
from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

from pairwise_ranker import QueryBalancedPairwiseRanker, audit_pairwise_data

HISTORICAL_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
DEPLOYMENT_VIEWS = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]

EXPECTED_HISTORICAL_R5 = 0.9569444444444444
EXPECTED_DEPLOYMENT_R5 = 0.9511111111111110

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_base_inputs():
    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: queries[q][1] for q in all_ids}

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned(rel) for name, rel in EXTRA_CV_PATHS.items()}

    full_channels_cv = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    return (
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        vnlegal_cv,
        type_rows,
        cite_rows,
    )


def run_lobo_eval(
    protocol_name: str,
    arm_name: str,
    base_names: List[str],
    extended: Dict[str, List[str]],
    local_views: Dict[str, Any],
    full_channels_cv: Dict[str, Any],
    type_rows: Dict[str, Any],
    cite_rows: Dict[str, Any],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    gold: Dict[str, Set[str]],
    profile_rankings_nested: Dict[str, Any] | None = None,
    training_audits: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    is_profile = profile_rankings_nested is not None
    names = list(base_names)
    if is_profile:
        names.append("fulltrain_huy_profile")

    preds = {}
    feature_dim = 0

    for held in sorted(blocks.keys()):
        train = sum((blocks[n] for n in blocks if n != held), [])

        fold_views = dict(local_views)
        if is_profile:
            held_ranks = profile_rankings_nested[held]["held_ranks"]
            train_ranks = profile_rankings_nested[held]["train_ranks"]
            fold_profile_view = {}
            for q in blocks[held]:
                fold_profile_view[q] = held_ranks[q]
            for q in train:
                fold_profile_view[q] = train_ranks[q]
            fold_views["fulltrain_huy_profile"] = fold_profile_view

        eval_ids = train + blocks[held]
        eval_rows, eval_groups = ltr_features(
            fold_views, names, extended, eval_ids, full_channels_cv
        )
        for q in eval_rows:
            eval_rows[q] = np.concatenate(
                [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        feature_dim = eval_rows[all_ids[0]].shape[1]

        y_by_qid = {
            q: np.asarray([d in gold[q] for d in eval_groups[q]], dtype=np.int8)
            for q in eval_ids
        }

        if arm_name == "Q0_POINTWISE":
            X_train = np.vstack([eval_rows[q] for q in train])
            y_train = np.concatenate([y_by_qid[q] for q in train])

            scaler = StandardScaler().fit(X_train)
            model = LogisticRegression(
                C=0.15,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=2026,
            )
            model.fit(scaler.transform(X_train), y_train)

            for q in blocks[held]:
                score = model.decision_function(scaler.transform(eval_rows[q]))
                preds[q] = [eval_groups[q][i] for i in np.argsort(-score)[:5]]

        elif arm_name in ("Q1_PAIRWISE", "Q2_PAIRWISE_PLUS_PROFILE"):
            # Query weight audit for outer training run
            if training_audits is not None:
                audit_record = audit_pairwise_data(
                    eval_rows,
                    y_by_qid,
                    train,
                    f"{protocol_name}_{arm_name}_held_{held}",
                )
                training_audits.append(audit_record)

            ranker = QueryBalancedPairwiseRanker(C=0.15, max_iter=3000, random_state=2026)
            ranker.fit(eval_rows, y_by_qid, train)

            for q in blocks[held]:
                utility = ranker.predict_utility(eval_rows[q])
                preds[q] = [eval_groups[q][i] for i in np.argsort(-utility)[:5]]
        else:
            raise ValueError(f"Unknown arm name: {arm_name}")

    pooled_r5 = float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids]))
    pooled_p5 = float(np.mean([len(gold[q] & set(preds[q])) / 5.0 for q in all_ids]))
    block_r5 = {
        b: float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in blocks[b]]))
        for b in sorted(blocks.keys())
    }
    single_r5 = float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) == 1]))
    multi_r5 = float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) > 1]))

    return {
        "protocol": protocol_name,
        "arm": arm_name,
        "feature_dim": feature_dim,
        "rank_views": names,
        "metrics": {
            "pooled_recall_at_5": pooled_r5,
            "pooled_precision_at_5": pooled_p5,
            "single_gold_recall_at_5": single_r5,
            "multi_gold_recall_at_5": multi_r5,
            "blocks": block_r5,
            "distance_to_0_96": float(0.96 - pooled_r5),
        },
        "predictions": preds,
    }


def compare_arms(
    cand_preds: Dict[str, List[str]],
    ref_preds: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    blocks: Dict[str, List[str]],
) -> Dict[str, Any]:
    all_ids = sorted(ref_preds.keys())
    wins = 0
    losses = 0
    ties = 0
    crossings_in = 0
    crossings_out = 0
    churn_queries = 0

    block_comp = {b: {"wins": 0, "losses": 0, "ties": 0} for b in blocks}

    for q in all_ids:
        g = gold[q]
        s_ref = set(ref_preds[q][:5])
        s_cand = set(cand_preds[q][:5])
        r_ref = len(g & s_ref) / len(g)
        r_cand = len(g & s_cand) / len(g)

        b_q = next(b for b in blocks if q in blocks[b])

        if r_cand > r_ref:
            wins += 1
            crossings_in += 1
            block_comp[b_q]["wins"] += 1
        elif r_cand < r_ref:
            losses += 1
            crossings_out += 1
            block_comp[b_q]["losses"] += 1
        else:
            ties += 1
            block_comp[b_q]["ties"] += 1

        if s_ref != s_cand:
            churn_queries += 1

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "gold_crossings_into_top5": crossings_in,
        "gold_crossings_out_of_top5": crossings_out,
        "top5_churn_queries": churn_queries,
        "top5_churn_pct": (churn_queries / len(all_ids)) * 100.0,
        "blocks": block_comp,
    }


def main() -> Dict[str, Any]:
    started_all = time.perf_counter()
    print("=== Step 2: Dual CAL LOBO Evaluation (Q0 vs Q1 vs Q2) ===", flush=True)

    print("Loading base inputs and features...", flush=True)
    (
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        vnlegal_cv,
        type_rows,
        cite_rows,
    ) = load_base_inputs()

    # Load cached profile rankings
    profile_cache_path = ROOT / "results" / "gemini" / "huy_fulltrain_profile_port_v1" / "PROFILE_BM25_RANKINGS.pkl"
    profile_cache = pickle.loads(profile_cache_path.read_bytes())
    nested_cal_rankings = profile_cache["nested_cal_rankings"]

    training_audits: List[Dict[str, Any]] = []

    # 1. HISTORICAL_CAL LOBO
    print("\n--- Running HISTORICAL_CAL Protocol ---", flush=True)
    hist_q0 = run_lobo_eval(
        "HISTORICAL_CAL",
        "Q0_POINTWISE",
        HISTORICAL_VIEWS,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )
    print(f"HISTORICAL_CAL Q0 R@5: {hist_q0['metrics']['pooled_recall_at_5']:.8f} (Expected: {EXPECTED_HISTORICAL_R5:.8f})")
    diff_hist_q0 = abs(hist_q0["metrics"]["pooled_recall_at_5"] - EXPECTED_HISTORICAL_R5)
    assert diff_hist_q0 < 1e-9, f"HISTORICAL_CAL Q0 parity failed: {hist_q0['metrics']['pooled_recall_at_5']} vs {EXPECTED_HISTORICAL_R5}"

    hist_q1 = run_lobo_eval(
        "HISTORICAL_CAL",
        "Q1_PAIRWISE",
        HISTORICAL_VIEWS,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
        training_audits=training_audits,
    )
    print(f"HISTORICAL_CAL Q1 R@5: {hist_q1['metrics']['pooled_recall_at_5']:.8f} (Delta vs Q0: {hist_q1['metrics']['pooled_recall_at_5'] - hist_q0['metrics']['pooled_recall_at_5']:+.8f})")

    hist_q2 = run_lobo_eval(
        "HISTORICAL_CAL",
        "Q2_PAIRWISE_PLUS_PROFILE",
        HISTORICAL_VIEWS,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
        profile_rankings_nested=nested_cal_rankings,
        training_audits=training_audits,
    )
    print(f"HISTORICAL_CAL Q2 R@5: {hist_q2['metrics']['pooled_recall_at_5']:.8f} (Delta vs Q0: {hist_q2['metrics']['pooled_recall_at_5'] - hist_q0['metrics']['pooled_recall_at_5']:+.8f})")

    # 2. DEPLOYMENT_CAL LOBO
    print("\n--- Running DEPLOYMENT_CAL Protocol ---", flush=True)
    dep_q0 = run_lobo_eval(
        "DEPLOYMENT_CAL",
        "Q0_POINTWISE",
        DEPLOYMENT_VIEWS,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )
    print(f"DEPLOYMENT_CAL Q0 R@5: {dep_q0['metrics']['pooled_recall_at_5']:.8f} (Expected: {EXPECTED_DEPLOYMENT_R5:.8f})")
    diff_dep_q0 = abs(dep_q0["metrics"]["pooled_recall_at_5"] - EXPECTED_DEPLOYMENT_R5)
    assert diff_dep_q0 < 1e-9, f"DEPLOYMENT_CAL Q0 parity failed: {dep_q0['metrics']['pooled_recall_at_5']} vs {EXPECTED_DEPLOYMENT_R5}"

    dep_q1 = run_lobo_eval(
        "DEPLOYMENT_CAL",
        "Q1_PAIRWISE",
        DEPLOYMENT_VIEWS,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
        training_audits=training_audits,
    )
    print(f"DEPLOYMENT_CAL Q1 R@5: {dep_q1['metrics']['pooled_recall_at_5']:.8f} (Delta vs Q0: {dep_q1['metrics']['pooled_recall_at_5'] - dep_q0['metrics']['pooled_recall_at_5']:+.8f})")

    dep_q2 = run_lobo_eval(
        "DEPLOYMENT_CAL",
        "Q2_PAIRWISE_PLUS_PROFILE",
        DEPLOYMENT_VIEWS,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
        profile_rankings_nested=nested_cal_rankings,
        training_audits=training_audits,
    )
    print(f"DEPLOYMENT_CAL Q2 R@5: {dep_q2['metrics']['pooled_recall_at_5']:.8f} (Delta vs Q0: {dep_q2['metrics']['pooled_recall_at_5'] - dep_q0['metrics']['pooled_recall_at_5']:+.8f})")

    # Comparisons vs Q0
    comp_hist_q1 = compare_arms(hist_q1["predictions"], hist_q0["predictions"], gold, blocks)
    comp_hist_q2 = compare_arms(hist_q2["predictions"], hist_q0["predictions"], gold, blocks)
    comp_dep_q1 = compare_arms(dep_q1["predictions"], dep_q0["predictions"], gold, blocks)
    comp_dep_q2 = compare_arms(dep_q2["predictions"], dep_q0["predictions"], gold, blocks)

    # Parity Report
    parity_report = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.baseline_parity.v1",
        "historical_cal": {
            "expected_pooled_r5": EXPECTED_HISTORICAL_R5,
            "computed_pooled_r5": hist_q0["metrics"]["pooled_recall_at_5"],
            "difference": diff_hist_q0,
            "feature_dim": hist_q0["feature_dim"],
            "blocks": hist_q0["metrics"]["blocks"],
            "parity_passed": diff_hist_q0 < 1e-9,
        },
        "deployment_cal": {
            "expected_pooled_r5": EXPECTED_DEPLOYMENT_R5,
            "computed_pooled_r5": dep_q0["metrics"]["pooled_recall_at_5"],
            "difference": diff_dep_q0,
            "feature_dim": dep_q0["feature_dim"],
            "blocks": dep_q0["metrics"]["blocks"],
            "parity_passed": diff_dep_q0 < 1e-9,
        },
        "all_parity_passed": diff_hist_q0 < 1e-9 and diff_dep_q0 < 1e-9,
        "status": "PASS" if (diff_hist_q0 < 1e-9 and diff_dep_q0 < 1e-9) else "FAIL",
    }
    with (RESULTS_DIR / "BASELINE_PARITY.json").open("w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)

    # Training Audit Report
    training_audit_report = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.pairwise_training_audit.v1",
        "training_runs_count": len(training_audits),
        "all_runs_query_balanced": all(a["query_sample_weights"]["query_balanced_assert_passed"] for a in training_audits),
        "training_runs": training_audits,
        "status": "PASS",
    }
    with (RESULTS_DIR / "PAIRWISE_TRAINING_AUDIT.json").open("w", encoding="utf-8") as f:
        json.dump(training_audit_report, f, indent=2)

    # Dual CAL Full Report
    def make_arm_summary(cand_res, ref_res, comp):
        r_cand = cand_res["metrics"]["pooled_recall_at_5"]
        r_ref = ref_res["metrics"]["pooled_recall_at_5"]
        return {
            "feature_dim": cand_res["feature_dim"],
            "rank_views": cand_res["rank_views"],
            "metrics": cand_res["metrics"],
            "delta_vs_q0": {
                "delta_pooled_recall_at_5": r_cand - r_ref,
                "delta_pooled_precision_at_5": cand_res["metrics"]["pooled_precision_at_5"] - ref_res["metrics"]["pooled_precision_at_5"],
                "delta_single_gold": cand_res["metrics"]["single_gold_recall_at_5"] - ref_res["metrics"]["single_gold_recall_at_5"],
                "delta_multi_gold": cand_res["metrics"]["multi_gold_recall_at_5"] - ref_res["metrics"]["multi_gold_recall_at_5"],
                "block_deltas": {
                    b: cand_res["metrics"]["blocks"][b] - ref_res["metrics"]["blocks"][b]
                    for b in sorted(cand_res["metrics"]["blocks"].keys())
                },
            },
            "comparison_vs_q0": comp,
        }

    dual_report = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.dual_cal_report.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": time.perf_counter() - started_all,
        "historical_cal": {
            "q0_pointwise": {
                "feature_dim": hist_q0["feature_dim"],
                "rank_views": hist_q0["rank_views"],
                "metrics": hist_q0["metrics"],
            },
            "q1_pairwise": make_arm_summary(hist_q1, hist_q0, comp_hist_q1),
            "q2_pairwise_plus_profile": make_arm_summary(hist_q2, hist_q0, comp_hist_q2),
        },
        "deployment_cal": {
            "q0_pointwise": {
                "feature_dim": dep_q0["feature_dim"],
                "rank_views": dep_q0["rank_views"],
                "metrics": dep_q0["metrics"],
            },
            "q1_pairwise": make_arm_summary(dep_q1, dep_q0, comp_dep_q1),
            "q2_pairwise_plus_profile": make_arm_summary(dep_q2, dep_q0, comp_dep_q2),
        },
    }

    with (RESULTS_DIR / "HUY_PAIRWISE_DUAL_CAL_REPORT.json").open("w", encoding="utf-8") as f:
        json.dump(dual_report, f, indent=2)

    print(f"\nSaved BASELINE_PARITY.json, PAIRWISE_TRAINING_AUDIT.json, HUY_PAIRWISE_DUAL_CAL_REPORT.json", flush=True)
    return dual_report


if __name__ == "__main__":
    main()
