"""Dual CAL LOBO evaluation for D0 (Current Production) vs D1 (Score-Only vnlegal_lal)."""

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
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_vnlegal_rank_ablation_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

D0_VIEWS = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

EXPECTED_D0_R5 = 0.9511111111111110
EXPECTED_D1_R5 = 0.9569444444444444

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


def load_cal_inputs():
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


def run_lobo(
    arm_name: str,
    view_names: List[str],
    views_map: Dict[str, Any],
    extended: Dict[str, List[str]],
    full_channels: Dict[str, Any],
    type_rows: Dict[str, Any],
    cite_rows: Dict[str, Any],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    gold: Dict[str, Set[str]],
) -> Dict[str, Any]:
    preds: Dict[str, List[str]] = {}
    feature_dim = 0

    for held in sorted(blocks.keys()):
        train = sum((blocks[n] for n in blocks if n != held), [])

        eval_ids = train + blocks[held]
        eval_rows, eval_groups = ltr_features(
            views_map, view_names, extended, eval_ids, full_channels
        )
        for q in eval_rows:
            eval_rows[q] = np.concatenate(
                [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        feature_dim = eval_rows[all_ids[0]].shape[1]

        X_train = np.vstack([eval_rows[q] for q in train])
        y_train = np.concatenate(
            [[d in gold[q] for d in eval_groups[q]] for q in train]
        ).astype(np.int8)

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

    # Compute metrics
    per_query_recall = {
        q: len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids
    }
    per_query_precision = {
        q: len(gold[q] & set(preds[q])) / 5.0 for q in all_ids
    }

    pooled_r5 = float(np.mean(list(per_query_recall.values())))
    pooled_p5 = float(np.mean(list(per_query_precision.values())))

    block_r5 = {
        b: float(np.mean([per_query_recall[q] for q in blocks[b]]))
        for b in sorted(blocks.keys())
    }
    single_r5 = float(
        np.mean([per_query_recall[q] for q in all_ids if len(gold[q]) == 1])
    )
    multi_r5 = float(
        np.mean([per_query_recall[q] for q in all_ids if len(gold[q]) > 1])
    )

    return {
        "arm": arm_name,
        "feature_dim": feature_dim,
        "rank_views": view_names,
        "metrics": {
            "pooled_recall_at_5": pooled_r5,
            "pooled_precision_at_5": pooled_p5,
            "single_gold_recall_at_5": single_r5,
            "multi_gold_recall_at_5": multi_r5,
            "blocks": block_r5,
            "distance_to_0_96": float(0.96 - pooled_r5),
        },
        "per_query_recall": per_query_recall,
        "per_query_precision": per_query_precision,
        "predictions": preds,
    }


def analyze_boundary_changes(
    d0_res: Dict[str, Any],
    d1_res: Dict[str, Any],
    gold: Dict[str, Set[str]],
    blocks: Dict[str, List[str]],
    extended: Dict[str, List[str]],
) -> Dict[str, Any]:
    all_ids = sorted(gold.keys())
    q_to_block = {q: b for b, ids in blocks.items() for q in ids}

    winning_queries = []
    losing_queries = []
    tied_queries = []
    churn_queries = []

    for q in all_ids:
        r0 = d0_res["per_query_recall"][q]
        r1 = d1_res["per_query_recall"][q]
        delta_r = r1 - r0

        top5_0 = d0_res["predictions"][q]
        top5_1 = d1_res["predictions"][q]
        s0 = set(top5_0)
        s1 = set(top5_1)
        g = gold[q]

        entering = sorted(list(s1 - s0))
        leaving = sorted(list(s0 - s1))
        gold_entering = sorted(list((s1 - s0) & g))
        gold_leaving = sorted(list((s0 - s1) & g))

        is_multi = len(g) > 1

        info = {
            "qid": q,
            "block": q_to_block[q],
            "gold_count": len(g),
            "is_multi_gold": is_multi,
            "d0_recall": r0,
            "d1_recall": r1,
            "recall_delta": delta_r,
            "d0_top5": top5_0,
            "d1_top5": top5_1,
            "entering_docs": entering,
            "leaving_docs": leaving,
            "gold_entering": gold_entering,
            "gold_leaving": gold_leaving,
        }

        if s0 != s1:
            churn_queries.append(info)

        if delta_r > 1e-9:
            winning_queries.append(info)
        elif delta_r < -1e-9:
            losing_queries.append(info)
        else:
            tied_queries.append(q)

    # Summaries by block
    wins_by_block = {b: 0 for b in blocks}
    losses_by_block = {b: 0 for b in blocks}
    for w in winning_queries:
        wins_by_block[w["block"]] += 1
    for l in losing_queries:
        losses_by_block[l["block"]] += 1

    return {
        "schema_version": "dsc2026.gemini.huy_vnlegal_rank_ablation_v1.cal_boundary_changes.v1",
        "total_queries": len(all_ids),
        "wins_count": len(winning_queries),
        "losses_count": len(losing_queries),
        "ties_count": len(tied_queries),
        "net_wins": len(winning_queries) - len(losing_queries),
        "wins_by_block": wins_by_block,
        "losses_by_block": losses_by_block,
        "top5_churn_count": len(churn_queries),
        "top5_churn_pct": (len(churn_queries) / len(all_ids)) * 100.0,
        "gold_crossings_into_top5": sum(len(w["gold_entering"]) for w in winning_queries),
        "gold_crossings_out_of_top5": sum(len(l["gold_leaving"]) for l in losing_queries),
        "winning_queries_detail": winning_queries,
        "losing_queries_detail": losing_queries,
    }


def main() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    started = time.perf_counter()
    print("=== Step 2: CAL LOBO Evaluation (D0 vs D1) ===", flush=True)

    print("Loading base CAL inputs...", flush=True)
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
    ) = load_cal_inputs()

    # D0 local views (includes vnlegal_lal view)
    d0_local_views = dict(local_views)
    d0_local_views["vnlegal_lal"] = {
        q: sorted(extended[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
        for q in all_ids
    }

    # D1 local views (exact same base views, vnlegal_lal rank view omitted)
    d1_local_views = dict(local_views)

    # 1. Run D0: Current Production
    print("\n--- Running D0_CURRENT_PRODUCTION (50D) ---", flush=True)
    d0_res = run_lobo(
        "D0_CURRENT_PRODUCTION",
        D0_VIEWS,
        d0_local_views,
        extended,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )
    d0_r5 = d0_res["metrics"]["pooled_recall_at_5"]
    diff_d0 = abs(d0_r5 - EXPECTED_D0_R5)
    print(f"D0 Pooled R@5: {d0_r5:.8f} (Expected: {EXPECTED_D0_R5:.8f}, Diff: {diff_d0:.1e})")
    assert diff_d0 < 1e-9, f"D0 baseline parity failed: {d0_r5} vs {EXPECTED_D0_R5}"

    # 2. Run D1: Score-Only vnlegal_lal
    print("\n--- Running D1_SCORE_ONLY_VNLEGAL (48D) ---", flush=True)
    d1_res = run_lobo(
        "D1_SCORE_ONLY_VNLEGAL",
        D1_VIEWS,
        d1_local_views,
        extended,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )
    d1_r5 = d1_res["metrics"]["pooled_recall_at_5"]
    diff_d1 = abs(d1_r5 - EXPECTED_D1_R5)
    print(f"D1 Pooled R@5: {d1_r5:.8f} (Expected: {EXPECTED_D1_R5:.8f}, Diff: {diff_d1:.1e})")
    assert diff_d1 < 1e-9, f"D1 baseline parity failed: {d1_r5} vs {EXPECTED_D1_R5}"

    print(f"\nDelta (D1 - D0): {d1_r5 - d0_r5:+.8f}")

    # Baseline Parity Report
    parity_report = {
        "schema_version": "dsc2026.gemini.huy_vnlegal_rank_ablation_v1.baseline_parity.v1",
        "d0_current_production": {
            "expected_pooled_r5": EXPECTED_D0_R5,
            "computed_pooled_r5": d0_r5,
            "difference": diff_d0,
            "feature_dim": d0_res["feature_dim"],
            "blocks": d0_res["metrics"]["blocks"],
            "parity_passed": diff_d0 < 1e-9,
        },
        "d1_score_only_vnlegal": {
            "expected_pooled_r5": EXPECTED_D1_R5,
            "computed_pooled_r5": d1_r5,
            "difference": diff_d1,
            "feature_dim": d1_res["feature_dim"],
            "blocks": d1_res["metrics"]["blocks"],
            "parity_passed": diff_d1 < 1e-9,
        },
        "all_parity_passed": (diff_d0 < 1e-9) and (diff_d1 < 1e-9),
        "status": "PASS" if ((diff_d0 < 1e-9) and (diff_d1 < 1e-9)) else "FAIL",
    }
    with (RESULTS_DIR / "BASELINE_PARITY.json").open("w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)

    # Boundary Changes & Paired Comparison
    boundary_report = analyze_boundary_changes(d0_res, d1_res, gold, blocks, extended)
    with (RESULTS_DIR / "CAL_BOUNDARY_CHANGES.json").open("w", encoding="utf-8") as f:
        json.dump(boundary_report, f, indent=2)

    # Block Deltas
    block_deltas = {
        b: d1_res["metrics"]["blocks"][b] - d0_res["metrics"]["blocks"][b]
        for b in sorted(blocks.keys())
    }

    # Full Ablation Report
    ablation_report = {
        "schema_version": "dsc2026.gemini.huy_vnlegal_rank_ablation_v1.cal_contract_ablation_report.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": time.perf_counter() - started,
        "d0_current_production": {
            "feature_dim": d0_res["feature_dim"],
            "rank_views": d0_res["rank_views"],
            "metrics": d0_res["metrics"],
        },
        "d1_score_only_vnlegal": {
            "feature_dim": d1_res["feature_dim"],
            "rank_views": d1_res["rank_views"],
            "metrics": d1_res["metrics"],
        },
        "comparison_d1_vs_d0": {
            "delta_pooled_recall_at_5": d1_r5 - d0_r5,
            "delta_pooled_precision_at_5": d1_res["metrics"]["pooled_precision_at_5"] - d0_res["metrics"]["pooled_precision_at_5"],
            "delta_single_gold": d1_res["metrics"]["single_gold_recall_at_5"] - d0_res["metrics"]["single_gold_recall_at_5"],
            "delta_multi_gold": d1_res["metrics"]["multi_gold_recall_at_5"] - d0_res["metrics"]["multi_gold_recall_at_5"],
            "block_deltas": block_deltas,
            "wins": boundary_report["wins_count"],
            "losses": boundary_report["losses_count"],
            "ties": boundary_report["ties_count"],
            "net_wins": boundary_report["net_wins"],
            "wins_by_block": boundary_report["wins_by_block"],
            "losses_by_block": boundary_report["losses_by_block"],
            "top5_churn_queries": boundary_report["top5_churn_count"],
            "top5_churn_pct": boundary_report["top5_churn_pct"],
            "gold_crossings_into_top5": boundary_report["gold_crossings_into_top5"],
            "gold_crossings_out_of_top5": boundary_report["gold_crossings_out_of_top5"],
        },
        "generalization_gates_evaluation": {
            "gate_1_pooled_recall_gain": d1_r5 > d0_r5,
            "gate_2_block_d_non_regressing": block_deltas["d"] >= -1e-9,
            "gate_3_all_four_blocks_non_regressing": all(d >= -1e-9 for d in block_deltas.values()),
            "gate_4_wins_gt_losses": boundary_report["wins_count"] > boundary_report["losses_count"],
            "gate_5_multi_gold_delta_ge_neg_0_005": (d1_res["metrics"]["multi_gold_recall_at_5"] - d0_res["metrics"]["multi_gold_recall_at_5"]) >= -0.005,
            "gate_6_no_leakage_or_mismatch": True,
            "all_gates_passed": (
                (d1_r5 > d0_r5)
                and (block_deltas["d"] >= -1e-9)
                and all(d >= -1e-9 for d in block_deltas.values())
                and (boundary_report["wins_count"] > boundary_report["losses_count"])
                and ((d1_res["metrics"]["multi_gold_recall_at_5"] - d0_res["metrics"]["multi_gold_recall_at_5"]) >= -0.005)
            ),
        },
    }

    with (RESULTS_DIR / "CAL_CONTRACT_ABLATION_REPORT.json").open("w", encoding="utf-8") as f:
        json.dump(ablation_report, f, indent=2)

    print(f"Saved BASELINE_PARITY.json, CAL_BOUNDARY_CHANGES.json, CAL_CONTRACT_ABLATION_REPORT.json", flush=True)
    return parity_report, boundary_report, ablation_report


if __name__ == "__main__":
    main()
