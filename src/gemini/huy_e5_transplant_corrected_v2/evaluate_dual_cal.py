import sys
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure stdout is utf-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path("d:/Study/DSC2026/sota").resolve()
sys.path.insert(0, str(REPO_ROOT))

from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

RESULTS_DIR = REPO_ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}

def load_pkl(rel_path: str):
    p = REPO_ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj

def evaluate_lobo_arm(
    views: Dict[str, Dict[str, List[str]]],
    view_names: List[str],
    channels: Dict[str, Dict[str, Dict[str, float]]],
    meta_features: Dict[str, np.ndarray],
    candidates: Dict[str, List[str]],
    all_ids: List[str],
    blocks: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    baseline_preds: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    # Compute LTR features
    rows_base, groups = ltr_features(views, view_names, candidates, all_ids, channels)
    full_rows = {q: np.concatenate([rows_base[q], meta_features[q]], axis=1) for q in all_ids}
    feat_dim = full_rows[all_ids[0]].shape[1]

    # LOBO evaluation
    preds = {}
    for held in blocks:
        train = sum((blocks[n] for n in blocks if n != held), [])
        X_train = np.vstack([full_rows[q] for q in train])
        y_train = np.concatenate([[d in gold[q] for d in groups[q]] for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
        model.fit(scaler.transform(X_train), y_train)

        for q in blocks[held]:
            scores = model.decision_function(scaler.transform(full_rows[q]))
            preds[q] = [groups[q][i] for i in np.argsort(-scores)[:5]]

    # Metrics
    pooled_r5 = float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids]))
    pooled_p5 = float(np.mean([len(gold[q] & set(preds[q])) / 5.0 for q in all_ids]))
    block_r5 = {
        b: float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in blocks[b]]))
        for b in blocks
    }
    single_r5 = float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) == 1]))
    multi_r5 = float(np.mean([len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) > 1]))

    comp = {}
    if baseline_preds is not None:
        wins = 0
        losses = 0
        ties = 0
        top5_churn = 0
        gold_crossings_in = 0
        gold_crossings_out = 0

        for q in all_ids:
            p_arm = set(preds[q])
            p_base = set(baseline_preds[q])
            g = gold[q]

            rec_arm = len(g & p_arm) / len(g)
            rec_base = len(g & p_base) / len(g)

            if rec_arm > rec_base:
                wins += 1
            elif rec_arm < rec_base:
                losses += 1
            else:
                ties += 1

            if p_arm != p_base:
                top5_churn += 1

            # Gold crossings
            in_cross = len((p_arm - p_base) & g)
            out_cross = len((p_base - p_arm) & g)
            gold_crossings_in += in_cross
            gold_crossings_out += out_cross

        comp = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_wins": wins - losses,
            "top5_churn_queries": top5_churn,
            "gold_crossings_into_top5": gold_crossings_in,
            "gold_crossings_out_of_top5": gold_crossings_out,
            "delta_pooled_recall_at_5": pooled_r5 - float(np.mean([len(gold[q] & set(baseline_preds[q])) / len(gold[q]) for q in all_ids])),
            "delta_multi_gold_recall_at_5": multi_r5 - float(np.mean([len(gold[q] & set(baseline_preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) > 1])),
            "block_deltas": {
                b: block_r5[b] - float(np.mean([len(gold[q] & set(baseline_preds[q])) / len(gold[q]) for q in blocks[b]]))
                for b in blocks
            }
        }

    return {
        "feature_count": feat_dim,
        "view_names": view_names,
        "channel_names": sorted(list(channels.keys())),
        "metrics": {
            "pooled_recall_at_5": pooled_r5,
            "pooled_precision_at_5": pooled_p5,
            "single_gold_recall_at_5": single_r5,
            "multi_gold_recall_at_5": multi_r5,
            "blocks": block_r5,
        },
        "comparison_vs_e0": comp,
        "predictions": preds,
    }

def main():
    print("=== DUAL CAL PROTOCOL EVALUATION FOR E0, E1, E2, E3 ===", flush=True)

    # 1. Load data
    docs = DocumentStore(sorted(
        (REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")
    ))
    queries_cal, blocks_cal, all_ids_cal, extended_cal, local_cal, base_scores_cal = build_training_cap(
        REPO_ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    gold_cal = {q: queries_cal[q][1] for q in all_ids_cal}

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended_cal[q]} for q in all_ids_cal}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned(rel) for name, rel in EXTRA_CV_PATHS.items()}

    # Base score channels (E0 channels)
    channels_e0 = {
        **base_scores_cal,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    # Load corrected adapted E5 scores
    e5_scores_pkl = RESULTS_DIR / "CAL600_CORRECTED_VIETLEGAL_E5_SCORES.pkl"
    with open(e5_scores_pkl, "rb") as f:
        e5_data = pickle.load(f)
    adapted_e5_scores = e5_data["adapted_scores"]
    adapted_e5_orders = e5_data["adapted_orders"]

    # Build Metadata features
    type_table = build_type_table(REPO_ROOT, docs, all_ids_cal, extended_cal)
    type_rows = type_features(extended_cal, type_table, queries_cal, all_ids_cal)
    own, cited = build_citation_table(docs, all_ids_cal, extended_cal)
    cite_rows = citation_features(extended_cal, own, cited, all_ids_cal)
    meta_features = {q: np.concatenate([type_rows[q], cite_rows[q]], axis=1) for q in all_ids_cal}

    # Channel configurations
    # E1: REPLACE 'e5' with 'adapted_vietlegal_e5'
    channels_e1 = {k: v for k, v in channels_e0.items() if k != "e5"}
    channels_e1["adapted_vietlegal_e5"] = adapted_e5_scores

    # E2: AUGMENT SCORE ONLY (keep 'e5' + add 'adapted_vietlegal_e5')
    channels_e2 = dict(channels_e0)
    channels_e2["adapted_vietlegal_e5"] = adapted_e5_scores

    # E3: AUGMENT SCORE PLUS RANK (keep 'e5' + add 'adapted_vietlegal_e5' score + add 'adapted_vietlegal_e5' rank view)
    channels_e3 = dict(channels_e0)
    channels_e3["adapted_vietlegal_e5"] = adapted_e5_scores

    protocols = ["HISTORICAL_CAL", "DEPLOYMENT_CAL"]
    dual_results = {}

    for proto in protocols:
        print(f"\n==================================================")
        print(f"EVALUATING PROTOCOL: {proto}")
        print(f"==================================================")

        if proto == "HISTORICAL_CAL":
            # 5 rank views
            base_views = local_cal
            base_view_names = ["base", "expanded", "jina", "dense", "corpus"]
        else:
            # 6 rank views (adds vnlegal_lal)
            base_views = dict(local_cal)
            base_views["vnlegal_lal"] = {
                q: sorted(extended_cal[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
                for q in all_ids_cal
            }
            base_view_names = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]

        # View setup for E3 (adds adapted_vietlegal_e5 view)
        views_e3 = dict(base_views)
        views_e3["adapted_vietlegal_e5"] = adapted_e5_orders
        view_names_e3 = base_view_names + ["adapted_vietlegal_e5"]

        # 1. Evaluate E0 (Baseline)
        print(f"Running E0 (Baseline)...", flush=True)
        res_e0 = evaluate_lobo_arm(
            base_views, base_view_names, channels_e0, meta_features,
            extended_cal, all_ids_cal, blocks_cal, gold_cal, baseline_preds=None
        )
        print(f"  E0: R@5={res_e0['metrics']['pooled_recall_at_5']:.6f}, dims={res_e0['feature_count']}, blocks: {res_e0['metrics']['blocks']}")

        e0_preds = res_e0["predictions"]

        # 2. Evaluate E1 (Replace Generic E5)
        print(f"Running E1 (Replace generic E5 score)...", flush=True)
        res_e1 = evaluate_lobo_arm(
            base_views, base_view_names, channels_e1, meta_features,
            extended_cal, all_ids_cal, blocks_cal, gold_cal, baseline_preds=e0_preds
        )
        print(f"  E1: R@5={res_e1['metrics']['pooled_recall_at_5']:.6f} (delta={res_e1['comparison_vs_e0']['delta_pooled_recall_at_5']:+.6f}), dims={res_e1['feature_count']}")
        print(f"      Blocks: {res_e1['metrics']['blocks']}")
        print(f"      W/L/T: {res_e1['comparison_vs_e0']['wins']}/{res_e1['comparison_vs_e0']['losses']}/{res_e1['comparison_vs_e0']['ties']}")

        # 3. Evaluate E2 (Augment Score Only)
        print(f"Running E2 (Augment adapted score only)...", flush=True)
        res_e2 = evaluate_lobo_arm(
            base_views, base_view_names, channels_e2, meta_features,
            extended_cal, all_ids_cal, blocks_cal, gold_cal, baseline_preds=e0_preds
        )
        print(f"  E2: R@5={res_e2['metrics']['pooled_recall_at_5']:.6f} (delta={res_e2['comparison_vs_e0']['delta_pooled_recall_at_5']:+.6f}), dims={res_e2['feature_count']}")
        print(f"      Blocks: {res_e2['metrics']['blocks']}")
        print(f"      W/L/T: {res_e2['comparison_vs_e0']['wins']}/{res_e2['comparison_vs_e0']['losses']}/{res_e2['comparison_vs_e0']['ties']}")

        # 4. Evaluate E3 (Augment Score + Rank View)
        print(f"Running E3 (Augment score + rank view)...", flush=True)
        res_e3 = evaluate_lobo_arm(
            views_e3, view_names_e3, channels_e3, meta_features,
            extended_cal, all_ids_cal, blocks_cal, gold_cal, baseline_preds=e0_preds
        )
        print(f"  E3: R@5={res_e3['metrics']['pooled_recall_at_5']:.6f} (delta={res_e3['comparison_vs_e0']['delta_pooled_recall_at_5']:+.6f}), dims={res_e3['feature_count']}")
        print(f"      Blocks: {res_e3['metrics']['blocks']}")
        print(f"      W/L/T: {res_e3['comparison_vs_e0']['wins']}/{res_e3['comparison_vs_e0']['losses']}/{res_e3['comparison_vs_e0']['ties']}")

        # Store results (without huge prediction dicts in json)
        clean_res = {}
        for arm_name, arm_obj in [("E0_BASELINE", res_e0), ("E1_REPLACE_E5", res_e1), ("E2_AUGMENT_SCORE", res_e2), ("E3_AUGMENT_SCORE_RANK", res_e3)]:
            clean_res[arm_name] = {
                "feature_count": arm_obj["feature_count"],
                "view_names": arm_obj["view_names"],
                "channel_names": arm_obj["channel_names"],
                "metrics": arm_obj["metrics"],
                "comparison_vs_e0": arm_obj["comparison_vs_e0"],
            }
        dual_results[proto] = clean_res

    # 5. Evaluate Generalization Gate (Section 11)
    # A candidate is eligible for promotion only if:
    # 1. pooled Recall@5 improves over E0 on BOTH HISTORICAL_CAL and DEPLOYMENT_CAL
    # 2. Block D does NOT regress on either protocol
    # 3. wins > losses on both protocols
    # 4. no catastrophic multi-gold regression (delta >= -0.005)
    # 5. uses no public labels
    print("\n==================================================")
    print("GENERALIZATION GATE EVALUATION")
    print("==================================================")

    gate_verdicts = {}
    for arm in ["E1_REPLACE_E5", "E2_AUGMENT_SCORE", "E3_AUGMENT_SCORE_RANK"]:
        h_comp = dual_results["HISTORICAL_CAL"][arm]["comparison_vs_e0"]
        d_comp = dual_results["DEPLOYMENT_CAL"][arm]["comparison_vs_e0"]

        c1_gain = (h_comp["delta_pooled_recall_at_5"] > 0) and (d_comp["delta_pooled_recall_at_5"] > 0)
        c2_block_d = (h_comp["block_deltas"]["d"] >= -1e-9) and (d_comp["block_deltas"]["d"] >= -1e-9)
        c3_wins = (h_comp["wins"] > h_comp["losses"]) and (d_comp["wins"] > d_comp["losses"])
        c4_multi = (h_comp["delta_multi_gold_recall_at_5"] >= -0.005) and (d_comp["delta_multi_gold_recall_at_5"] >= -0.005)

        passed = c1_gain and c2_block_d and c3_wins and c4_multi

        gate_verdicts[arm] = {
            "passes_all_gates": passed,
            "criterion_1_pooled_gain_both": c1_gain,
            "criterion_2_no_block_d_regression_both": c2_block_d,
            "criterion_3_wins_greater_losses_both": c3_wins,
            "criterion_4_no_catastrophic_multi_regression": c4_multi,
            "historical_cal_delta": h_comp["delta_pooled_recall_at_5"],
            "deployment_cal_delta": d_comp["delta_pooled_recall_at_5"],
            "historical_block_d_delta": h_comp["block_deltas"]["d"],
            "deployment_block_d_delta": d_comp["block_deltas"]["d"],
            "historical_wins_losses": f"{h_comp['wins']}/{h_comp['losses']}",
            "deployment_wins_losses": f"{d_comp['wins']}/{d_comp['losses']}",
            "historical_multi_delta": h_comp["delta_multi_gold_recall_at_5"],
            "deployment_multi_delta": d_comp["delta_multi_gold_recall_at_5"],
        }
        print(f"{arm:28s} Gate Passed: {passed}")
        print(f"   Pooled Delta: Hist={h_comp['delta_pooled_recall_at_5']:+.6f}, Dep={d_comp['delta_pooled_recall_at_5']:+.6f}")
        print(f"   Block D Delta: Hist={h_comp['block_deltas']['d']:+.6f}, Dep={d_comp['block_deltas']['d']:+.6f}")
        print(f"   W/L:          Hist={h_comp['wins']}/{h_comp['losses']}, Dep={d_comp['wins']}/{d_comp['losses']}")

    # Final verdict selection
    passing_arms = [arm for arm, v in gate_verdicts.items() if v["passes_all_gates"]]
    if passing_arms:
        # Tie-break: highest HISTORICAL_CAL pooled recall, then highest DEPLOYMENT_CAL, then fewer features (prefer E1)
        def tie_breaker(arm_name):
            hist_rec = dual_results["HISTORICAL_CAL"][arm_name]["metrics"]["pooled_recall_at_5"]
            dep_rec = dual_results["DEPLOYMENT_CAL"][arm_name]["metrics"]["pooled_recall_at_5"]
            feat_count = dual_results["DEPLOYMENT_CAL"][arm_name]["feature_count"]
            return (hist_rec, dep_rec, -feat_count)

        passing_arms.sort(key=tie_breaker, reverse=True)
        selected_arm = passing_arms[0]
        final_verdict = f"PROMOTE_{selected_arm.split('_')[0]}"
    else:
        selected_arm = None
        final_verdict = "KILL"

    print(f"\nFINAL PROMOTION VERDICT: {final_verdict}")
    if selected_arm:
        print(f"Selected Candidate: {selected_arm}")

    report_data = {
        "schema_version": "dsc2026.gemini.huy_e5_transplant_corrected_v2.dual_cal_evaluation_report.v1",
        "final_verdict": final_verdict,
        "selected_promoted_arm": selected_arm,
        "gate_summary": gate_verdicts,
        "protocols": dual_results
    }

    report_path = RESULTS_DIR / "DUAL_CAL_EVALUATION_REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote dual CAL evaluation report to {report_path}", flush=True)

if __name__ == "__main__":
    main()
