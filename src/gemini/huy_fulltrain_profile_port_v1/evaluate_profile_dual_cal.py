"""Dual CAL protocol LOBO evaluation for baseline P0 vs fulltrain profile port P1."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "huy_fasttrack"))
import run_huy_5fold_fasttrack as core
from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

HISTORICAL_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
DEPLOYMENT_VIEWS = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]

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

    # Prepare CV score channels
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
    base_names: list[str],
    extended: dict[str, list[str]],
    local_views: dict[str, Any],
    full_channels_cv: dict[str, Any],
    type_rows: dict[str, Any],
    cite_rows: dict[str, Any],
    blocks: dict[str, list[str]],
    all_ids: list[str],
    gold: dict[str, set[str]],
    profile_rankings_nested: dict[str, Any] | None = None,
):
    is_p1 = profile_names_active = profile_rankings_nested is not None
    names = list(base_names)
    if is_p1:
        names.append("fulltrain_huy_profile")

    preds = {}
    feature_dim = 0

    for held in blocks:
        train = sum((blocks[n] for n in blocks if n != held), [])

        # Build view mapping for this outer held fold
        fold_views = dict(local_views)
        if is_p1:
            held_ranks = profile_rankings_nested[held]["held_ranks"]
            train_ranks = profile_rankings_nested[held]["train_ranks"]
            fold_profile_view = {}
            for q in blocks[held]:
                fold_profile_view[q] = held_ranks[q]
            for q in train:
                fold_profile_view[q] = train_ranks[q]
            fold_views["fulltrain_huy_profile"] = fold_profile_view

        # Generate feature rows for train + test queries
        eval_ids = train + blocks[held]
        eval_rows, eval_groups = ltr_features(
            fold_views, names, extended, eval_ids, full_channels_cv
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
    pooled_r5 = float(
        np.mean(
            [len(gold[q] & set(preds[q])) / len(gold[q]) for q in all_ids]
        )
    )
    pooled_p5 = float(
        np.mean([len(gold[q] & set(preds[q])) / 5.0 for q in all_ids])
    )
    block_r5 = {
        b: float(
            np.mean(
                [
                    len(gold[q] & set(preds[q])) / len(gold[q])
                    for q in blocks[b]
                ]
            )
        )
        for b in blocks
    }
    single_r5 = float(
        np.mean(
            [
                len(gold[q] & set(preds[q])) / len(gold[q])
                for q in all_ids
                if len(gold[q]) == 1
            ]
        )
    )
    multi_r5 = float(
        np.mean(
            [
                len(gold[q] & set(preds[q])) / len(gold[q])
                for q in all_ids
                if len(gold[q]) > 1
            ]
        )
    )

    return {
        "protocol": protocol_name,
        "arm": "P1" if is_p1 else "P0",
        "feature_dim": feature_dim,
        "rank_views": names,
        "metrics": {
            "pooled_recall_at_5": pooled_r5,
            "pooled_precision_at_5": pooled_p5,
            "single_gold_recall_at_5": single_r5,
            "multi_gold_recall_at_5": multi_r5,
            "blocks": block_r5,
        },
        "predictions": preds,
    }


def compare_arms(p0_preds: dict[str, list[str]], p1_preds: dict[str, list[str]], gold: dict[str, set[str]], blocks: dict[str, list[str]]):
    all_ids = sorted(p0_preds.keys())
    wins = 0
    losses = 0
    ties = 0
    crossings_in = 0
    crossings_out = 0
    churn_queries = 0

    query_deltas = {}
    for q in all_ids:
        g = gold[q]
        s0 = set(p0_preds[q])
        s1 = set(p1_preds[q])
        r0 = len(g & s0) / len(g)
        r1 = len(g & s1) / len(g)
        delta = r1 - r0
        query_deltas[q] = delta

        if r1 > r0:
            wins += 1
            crossings_in += 1
        elif r1 < r0:
            losses += 1
            crossings_out += 1
        else:
            ties += 1

        if s0 != s1:
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
        "query_deltas": query_deltas,
    }


def main():
    started = time.perf_counter()
    out_dir = ROOT / "results" / "gemini" / "huy_fulltrain_profile_port_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base inputs and features...")
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
    cache_path = out_dir / "PROFILE_BM25_RANKINGS.pkl"
    if not cache_path.exists():
        raise RuntimeError(f"Missing cached profile rankings: {cache_path}")
    profile_cache = pickle.loads(cache_path.read_bytes())
    nested_cal_rankings = profile_cache["nested_cal_rankings"]
    cal_oof_rankings = profile_cache["cal_oof_rankings"]
    cal_lexical_support = profile_cache["cal_lexical_support"]

    # --- 1. HISTORICAL_CAL LOBO ---
    print("\n--- Running HISTORICAL_CAL Protocol ---")
    hist_p0 = run_lobo_eval(
        "HISTORICAL_CAL",
        HISTORICAL_VIEWS,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
        profile_rankings_nested=None,
    )

    hist_p1 = run_lobo_eval(
        "HISTORICAL_CAL",
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
    )

    hist_comp = compare_arms(hist_p0["predictions"], hist_p1["predictions"], gold, blocks)
    hist_delta = hist_p1["metrics"]["pooled_recall_at_5"] - hist_p0["metrics"]["pooled_recall_at_5"]
    hist_block_deltas = {
        b: hist_p1["metrics"]["blocks"][b] - hist_p0["metrics"]["blocks"][b]
        for b in blocks
    }
    hist_multi_delta = hist_p1["metrics"]["multi_gold_recall_at_5"] - hist_p0["metrics"]["multi_gold_recall_at_5"]

    print(f"HISTORICAL_CAL P0 R@5: {hist_p0['metrics']['pooled_recall_at_5']:.6f} ({hist_p0['feature_dim']} dims)")
    print(f"HISTORICAL_CAL P1 R@5: {hist_p1['metrics']['pooled_recall_at_5']:.6f} ({hist_p1['feature_dim']} dims) -> Delta: {hist_delta:+.6f}")
    print(f"  Block D Delta: {hist_block_deltas['d']:+.6f} | Wins/Losses: {hist_comp['wins']}/{hist_comp['losses']}")

    # --- 2. DEPLOYMENT_CAL LOBO ---
    print("\n--- Running DEPLOYMENT_CAL Protocol ---")
    dep_local_views = dict(local_views)
    dep_local_views["vnlegal_lal"] = {
        q: sorted(extended[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
        for q in all_ids
    }

    dep_p0 = run_lobo_eval(
        "DEPLOYMENT_CAL",
        DEPLOYMENT_VIEWS,
        extended,
        dep_local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
        profile_rankings_nested=None,
    )

    dep_p1 = run_lobo_eval(
        "DEPLOYMENT_CAL",
        DEPLOYMENT_VIEWS,
        extended,
        dep_local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
        profile_rankings_nested=nested_cal_rankings,
    )

    dep_comp = compare_arms(dep_p0["predictions"], dep_p1["predictions"], gold, blocks)
    dep_delta = dep_p1["metrics"]["pooled_recall_at_5"] - dep_p0["metrics"]["pooled_recall_at_5"]
    dep_block_deltas = {
        b: dep_p1["metrics"]["blocks"][b] - dep_p0["metrics"]["blocks"][b]
        for b in blocks
    }
    dep_multi_delta = dep_p1["metrics"]["multi_gold_recall_at_5"] - dep_p0["metrics"]["multi_gold_recall_at_5"]

    print(f"DEPLOYMENT_CAL P0 R@5: {dep_p0['metrics']['pooled_recall_at_5']:.6f} ({dep_p0['feature_dim']} dims)")
    print(f"DEPLOYMENT_CAL P1 R@5: {dep_p1['metrics']['pooled_recall_at_5']:.6f} ({dep_p1['feature_dim']} dims) -> Delta: {dep_delta:+.6f}")
    print(f"  Block D Delta: {dep_block_deltas['d']:+.6f} | Wins/Losses: {dep_comp['wins']}/{dep_comp['losses']}")

    # --- 3. BASELINE PARITY VERIFICATION ---
    expected_hist_r5 = 0.9569444444444444
    expected_dep_r5 = 0.9511111111111110
    hist_diff = abs(hist_p0["metrics"]["pooled_recall_at_5"] - expected_hist_r5)
    dep_diff = abs(dep_p0["metrics"]["pooled_recall_at_5"] - expected_dep_r5)
    parity_passed = hist_diff < 1e-6 and dep_diff < 1e-6

    parity_report = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.baseline_parity.v1",
        "parity_gate_passed": parity_passed,
        "historical_cal": {
            "expected_pooled_r5": expected_hist_r5,
            "computed_pooled_r5": hist_p0["metrics"]["pooled_recall_at_5"],
            "difference": hist_diff,
            "feature_dim": hist_p0["feature_dim"],
            "blocks": hist_p0["metrics"]["blocks"],
        },
        "deployment_cal": {
            "expected_pooled_r5": expected_dep_r5,
            "computed_pooled_r5": dep_p0["metrics"]["pooled_recall_at_5"],
            "difference": dep_diff,
            "feature_dim": dep_p0["feature_dim"],
            "blocks": dep_p0["metrics"]["blocks"],
        },
        "status": "PASS" if parity_passed else "FAIL",
    }
    with (out_dir / "BASELINE_PARITY.json").open("w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)
    print(f"BASELINE_PARITY: status={parity_report['status']} (Hist diff: {hist_diff:.2e}, Dep diff: {dep_diff:.2e})")

    # --- 4. STANDALONE ORACLE UNION ---
    oracle_union_hits = 0.0
    for q in all_ids:
        g = gold[q]
        top5_p0 = set(hist_p0["predictions"][q])
        top5_prof = set(cal_oof_rankings[q][:5])
        union_top = top5_p0 | top5_prof
        oracle_union_hits += len(g & union_top) / len(g)
    oracle_union_r5 = oracle_union_hits / len(all_ids)
    print(f"Oracle Union (H0 Top-5 | Profile Top-5) Recall: {oracle_union_r5:.6f} (+{oracle_union_r5 - expected_hist_r5:+.6f} over H0)")

    # Update PROFILE_STANDALONE_REPORT.json with oracle union
    standalone_rep_path = out_dir / "PROFILE_STANDALONE_REPORT.json"
    if standalone_rep_path.exists():
        standalone_rep = json.loads(standalone_rep_path.read_text(encoding="utf-8"))
        standalone_rep["oracle_union_h0"] = {
            "oracle_union_recall": oracle_union_r5,
            "h0_baseline_recall": expected_hist_r5,
            "oracle_headroom": oracle_union_r5 - expected_hist_r5,
            "description": "Recall of union between H0 Top-5 and Profile Top-5 (up to 10 docs)",
        }
        with standalone_rep_path.open("w", encoding="utf-8") as f:
            json.dump(standalone_rep, f, indent=2)

    # --- 5. GENERALIZATION AUDIT (Section 14) ---
    # Load profile model memory to inspect label familiarity and frequency
    from profile_data_isolation import load_populations, load_linked_duplicates, get_dup_linked
    (
        v2_qids, v2_set, _, _, _, _, _, _
    ) = load_populations()
    links = load_linked_duplicates()

    # Load canonical queries dict
    _, _, v2_questions, v2_golds, _, _, _, _ = core.load_inputs()
    v2_queries = {qid: (v2_questions[qid], v2_golds[qid]) for qid in v2_qids}

    # Precompute label frequencies in M(H) for each held block H
    mem_label_frequencies = {}
    for held_name, held_ids in blocks.items():
        held_set = set(held_ids)
        held_dup = get_dup_linked(held_ids, links)
        mem_held = v2_set - held_set - held_dup
        freqs = {}
        for q in mem_held:
            for doc in v2_queries[q][1]:
                freqs[doc] = freqs.get(doc, 0) + 1
        mem_label_frequencies[held_name] = freqs

    # Categorize queries into slices
    slice_all_seen = []
    slice_unseen_any = []
    slice_freq_0 = []
    slice_freq_1 = []
    slice_freq_2_3 = []
    slice_freq_4plus = []
    slice_single_gold = []
    slice_multi_gold = []

    for q in all_ids:
        # Determine block of q
        held_name = next(b for b in blocks if q in blocks[b])
        freqs = mem_label_frequencies[held_name]
        g = gold[q]

        gold_freqs = [freqs.get(d, 0) for d in g]
        min_f = min(gold_freqs) if gold_freqs else 0

        if all(f > 0 for f in gold_freqs):
            slice_all_seen.append(q)
        else:
            slice_unseen_any.append(q)

        if min_f == 0:
            slice_freq_0.append(q)
        elif min_f == 1:
            slice_freq_1.append(q)
        elif 2 <= min_f <= 3:
            slice_freq_2_3.append(q)
        else:
            slice_freq_4plus.append(q)

        if len(g) == 1:
            slice_single_gold.append(q)
        else:
            slice_multi_gold.append(q)

    # Lexical support quantiles
    support_vals = [cal_lexical_support[q] for q in all_ids]
    q25, q50, q75 = np.percentile(support_vals, [25, 50, 75])
    slice_supp_q1 = [q for q in all_ids if cal_lexical_support[q] <= q25]
    slice_supp_q2 = [q for q in all_ids if q25 < cal_lexical_support[q] <= q50]
    slice_supp_q3 = [q for q in all_ids if q50 < cal_lexical_support[q] <= q75]
    slice_supp_q4 = [q for q in all_ids if cal_lexical_support[q] > q75]

    def evaluate_slice(slice_ids: list[str], proto_p0, proto_p1):
        if not slice_ids:
            return {"queries": 0, "p0_r5": 0.0, "p1_r5": 0.0, "delta": 0.0, "wins": 0, "losses": 0, "ties": 0}
        r0 = float(np.mean([len(gold[q] & set(proto_p0["predictions"][q])) / len(gold[q]) for q in slice_ids]))
        r1 = float(np.mean([len(gold[q] & set(proto_p1["predictions"][q])) / len(gold[q]) for q in slice_ids]))
        w = sum(proto_p1["predictions"][q] != proto_p0["predictions"][q] and (len(gold[q] & set(proto_p1["predictions"][q])) > len(gold[q] & set(proto_p0["predictions"][q]))) for q in slice_ids)
        l = sum(proto_p1["predictions"][q] != proto_p0["predictions"][q] and (len(gold[q] & set(proto_p1["predictions"][q])) < len(gold[q] & set(proto_p0["predictions"][q]))) for q in slice_ids)
        t = len(slice_ids) - w - l
        return {
            "queries": len(slice_ids),
            "p0_r5": r0,
            "p1_r5": r1,
            "delta": r1 - r0,
            "wins": w,
            "losses": l,
            "ties": t,
        }

    gen_audit = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.profile_generalization_audit.v1",
        "historical_cal_slices": {
            "label_familiarity": {
                "all_gold_seen_in_memory": evaluate_slice(slice_all_seen, hist_p0, hist_p1),
                "at_least_one_gold_unseen": evaluate_slice(slice_unseen_any, hist_p0, hist_p1),
            },
            "gold_label_frequency": {
                "freq_0": evaluate_slice(slice_freq_0, hist_p0, hist_p1),
                "freq_1": evaluate_slice(slice_freq_1, hist_p0, hist_p1),
                "freq_2_to_3": evaluate_slice(slice_freq_2_3, hist_p0, hist_p1),
                "freq_4_plus": evaluate_slice(slice_freq_4plus, hist_p0, hist_p1),
            },
            "cardinality": {
                "single_gold": evaluate_slice(slice_single_gold, hist_p0, hist_p1),
                "multi_gold": evaluate_slice(slice_multi_gold, hist_p0, hist_p1),
            },
            "lexical_support_quantiles": {
                "thresholds": {"q25": float(q25), "q50": float(q50), "q75": float(q75)},
                "q1_low": evaluate_slice(slice_supp_q1, hist_p0, hist_p1),
                "q2_mid_low": evaluate_slice(slice_supp_q2, hist_p0, hist_p1),
                "q3_mid_high": evaluate_slice(slice_supp_q3, hist_p0, hist_p1),
                "q4_high": evaluate_slice(slice_supp_q4, hist_p0, hist_p1),
            },
        },
        "deployment_cal_slices": {
            "label_familiarity": {
                "all_gold_seen_in_memory": evaluate_slice(slice_all_seen, dep_p0, dep_p1),
                "at_least_one_gold_unseen": evaluate_slice(slice_unseen_any, dep_p0, dep_p1),
            },
            "cardinality": {
                "single_gold": evaluate_slice(slice_single_gold, dep_p0, dep_p1),
                "multi_gold": evaluate_slice(slice_multi_gold, dep_p0, dep_p1),
            },
        },
    }

    with (out_dir / "PROFILE_GENERALIZATION_AUDIT.json").open("w", encoding="utf-8") as f:
        json.dump(gen_audit, f, indent=2)
    print("Wrote PROFILE_GENERALIZATION_AUDIT.json")

    # --- 6. PROMOTION GATE VERIFICATION (Section 15) ---
    c1 = hist_delta > 0.0
    c2 = dep_delta >= -1e-6
    c3 = hist_block_deltas["d"] >= -1e-6
    c4 = hist_comp["wins"] > hist_comp["losses"]
    c5 = hist_multi_delta >= -0.005
    unseen_delta = gen_audit["historical_cal_slices"]["label_familiarity"]["at_least_one_gold_unseen"]["delta"]
    c6 = unseen_delta >= -0.02  # no catastrophic collapse on unseen labels
    c7 = parity_passed

    all_criteria_passed = c1 and c2 and c3 and c4 and c5 and c6 and c7
    verdict = "PROMOTE_PROFILE" if all_criteria_passed else "KILL_PROFILE"

    gate_summary = {
        "final_verdict": verdict,
        "criteria": {
            "criterion_1_historical_gain": {"passed": c1, "delta": hist_delta},
            "criterion_2_deployment_non_regression": {"passed": c2, "delta": dep_delta},
            "criterion_3_historical_block_d_safe": {"passed": c3, "delta": hist_block_deltas["d"]},
            "criterion_4_wins_greater_losses": {"passed": c4, "wins": hist_comp["wins"], "losses": hist_comp["losses"]},
            "criterion_5_multi_gold_safe": {"passed": c5, "delta": hist_multi_delta},
            "criterion_6_unseen_labels_safe": {"passed": c6, "unseen_delta": unseen_delta},
            "criterion_7_baseline_parity_passed": {"passed": c7},
        },
        "rejection_reasons": [] if all_criteria_passed else [
            name for name, d in {
                "criterion_1_historical_gain": c1,
                "criterion_2_deployment_non_regression": c2,
                "criterion_3_historical_block_d_safe": c3,
                "criterion_4_wins_greater_losses": c4,
                "criterion_5_multi_gold_safe": c5,
                "criterion_6_unseen_labels_safe": c6,
                "criterion_7_baseline_parity_passed": c7,
            }.items() if not d
        ]
    }

    dual_cal_report = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.profile_dual_cal_report.v1",
        "final_verdict": verdict,
        "gate_summary": gate_summary,
        "protocols": {
            "HISTORICAL_CAL": {
                "P0_BASELINE": {
                    "feature_dim": hist_p0["feature_dim"],
                    "rank_views": hist_p0["rank_views"],
                    "metrics": hist_p0["metrics"],
                },
                "P1_PROFILE_PORT": {
                    "feature_dim": hist_p1["feature_dim"],
                    "rank_views": hist_p1["rank_views"],
                    "metrics": hist_p1["metrics"],
                    "comparison_vs_p0": {
                        "delta_pooled_recall_at_5": hist_delta,
                        "delta_multi_gold_recall_at_5": hist_multi_delta,
                        "block_deltas": hist_block_deltas,
                        "wins": hist_comp["wins"],
                        "losses": hist_comp["losses"],
                        "ties": hist_comp["ties"],
                        "net_wins": hist_comp["net_wins"],
                        "gold_crossings_into_top5": hist_comp["gold_crossings_into_top5"],
                        "gold_crossings_out_of_top5": hist_comp["gold_crossings_out_of_top5"],
                        "top5_churn_queries": hist_comp["top5_churn_queries"],
                    }
                }
            },
            "DEPLOYMENT_CAL": {
                "P0_BASELINE": {
                    "feature_dim": dep_p0["feature_dim"],
                    "rank_views": dep_p0["rank_views"],
                    "metrics": dep_p0["metrics"],
                },
                "P1_PROFILE_PORT": {
                    "feature_dim": dep_p1["feature_dim"],
                    "rank_views": dep_p1["rank_views"],
                    "metrics": dep_p1["metrics"],
                    "comparison_vs_p0": {
                        "delta_pooled_recall_at_5": dep_delta,
                        "delta_multi_gold_recall_at_5": dep_multi_delta,
                        "block_deltas": dep_block_deltas,
                        "wins": dep_comp["wins"],
                        "losses": dep_comp["losses"],
                        "ties": dep_comp["ties"],
                        "net_wins": dep_comp["net_wins"],
                        "gold_crossings_into_top5": dep_comp["gold_crossings_into_top5"],
                        "gold_crossings_out_of_top5": dep_comp["gold_crossings_out_of_top5"],
                        "top5_churn_queries": dep_comp["top5_churn_queries"],
                    }
                }
            }
        },
        "runtime_seconds": time.perf_counter() - started,
    }

    with (out_dir / "PROFILE_DUAL_CAL_REPORT.json").open("w", encoding="utf-8") as f:
        json.dump(dual_cal_report, f, indent=2)

    print(f"\n=======================================================")
    print(f"DUAL CAL EVALUATION COMPLETE -> VERDICT: {verdict}")
    print(f"=======================================================")
    print(f"HISTORICAL_CAL: P0={hist_p0['metrics']['pooled_recall_at_5']:.6f} -> P1={hist_p1['metrics']['pooled_recall_at_5']:.6f} (Delta: {hist_delta:+.6f})")
    print(f"DEPLOYMENT_CAL: P0={dep_p0['metrics']['pooled_recall_at_5']:.6f} -> P1={dep_p1['metrics']['pooled_recall_at_5']:.6f} (Delta: {dep_delta:+.6f})")
    print(f"Wrote PROFILE_DUAL_CAL_REPORT.json in {dual_cal_report['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
