"""Evaluate Arms H0, H1, H2, H3 under strict 4-block Leave-One-Block-Out on CAL600.

Arms:
- H0: Current Huy public-semantics baseline (50 features)
- H1: H0 + frozen VietLegal-E5 view & score channel (54 features)
- H2: H0 + task-adapted VietLegal-E5 view & score channel (54 features)
- H3: H2 + 6 low-capacity adaptation delta features (60 features)

Produces:
- results/gemini/huy_e5_adaptation_delta_v1/CAL600_EVALUATION_REPORT.json
"""

from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results/gemini/huy_e5_adaptation_delta_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_standardized_scores(raw_scores: Dict[str, float], docs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute z-score and gap-to-top for candidate docs."""
    values = np.asarray([raw_scores.get(d, np.nan) for d in docs], dtype=np.float64)
    present = values[~np.isnan(values)]
    if present.size:
        mean = float(present.mean())
        std = float(present.std()) or 1.0
        top = float(present.max())
    else:
        mean, std, top = 0.0, 1.0, 0.0
    filled = np.where(np.isnan(values), mean - 2 * std, values)
    z = (filled - mean) / std
    gap = (filled - top) / std
    return z, gap


def build_base_h0_components():
    """Build exact public-semantics H0 features on CAL600."""
    import sys
    sys.path.insert(0, str(ROOT))
    from run_burst_expanded_fusion_submission import DocumentStore
    from run_burst_multistage_submission import load_metadata
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_dense_fusion import build_training
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    data_dir = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    paths, doc_ids, train, public = load_metadata(data_dir)
    documents = DocumentStore(paths)

    queries, blocks, holdout_ids, holdout_candidates, holdout_views, training_scores = build_training(
        ROOT, depth=10, cap=32, extended_scores_path="results/corpus_index/holdout_extended_scores_cap32.pkl"
    )

    # 1. vnlegal_lal
    holdout_vnlegal_scores = pickle.loads(
        (ROOT / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl").read_bytes()
    )
    names = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]
    training_scores = dict(training_scores)
    training_scores["vnlegal_lal"] = holdout_vnlegal_scores
    holdout_views = dict(holdout_views)
    holdout_views["vnlegal_lal"] = {
        q: sorted(holdout_candidates[q], key=lambda d: (-holdout_vnlegal_scores.get(q, {}).get(d, -1e9), d))
        for q in holdout_ids
    }

    # 2. crossenc
    cv_ce = pickle.loads((ROOT / "results/crossenc_fullpool/cv_scores.pkl").read_bytes())["scores"]
    floor_ce = min(min(v.values()) for v in cv_ce.values() if v)
    training_scores["crossenc"] = {
        q: {d: cv_ce.get(q, {}).get(d, floor_ce) for d in holdout_candidates[q]}
        for q in holdout_ids
    }

    # 3. extra channels: aiteamvn_ft, jina_ft, title_embed
    extra_specs = [
        ("aiteamvn_ft", "results/from_drive/aiteamvn_ft_cv.pkl"),
        ("jina_ft", "results/from_drive/jina_ft_cv.pkl"),
        ("title_embed", "results/burst_fresh_block/title_embed_scores.pkl"),
    ]
    for name, rel_path in extra_specs:
        raw = pickle.loads((ROOT / rel_path).read_bytes())
        if isinstance(raw, dict) and isinstance(raw.get("scores"), dict):
            raw = raw["scores"]
        fl = min(v for q in raw for v in raw[q].values())
        training_scores[name] = {
            q: {d: raw.get(q, {}).get(d, fl) for d in holdout_candidates[q]}
            for q in holdout_ids
        }

    # Base LTR features (14 rank + 20 score = 34)
    rows_base, groups = ltr_features(holdout_views, names, holdout_candidates, holdout_ids, training_scores)

    # Doctype & Citation
    holdout_types = build_type_table(ROOT, documents, holdout_ids, holdout_candidates)
    t_rows = type_features(holdout_candidates, holdout_types, queries, holdout_ids)
    holdout_own, holdout_cited = build_citation_table(documents, holdout_ids, holdout_candidates)
    c_rows = citation_features(holdout_candidates, holdout_own, holdout_cited, holdout_ids)

    # H0 combined rows (34 + 12 + 4 = 50)
    rows_h0 = {}
    for q in holdout_ids:
        rows_h0[q] = np.concatenate([rows_base[q], t_rows[q], c_rows[q]], axis=1)

    return (
        queries,
        blocks,
        holdout_ids,
        holdout_candidates,
        holdout_views,
        names,
        training_scores,
        t_rows,
        c_rows,
        rows_h0,
        groups,
    )


def evaluate_lobo(
    feature_rows: Dict[str, np.ndarray],
    groups: Dict[str, List[str]],
    queries: Dict[str, Tuple[str, Set[str]]],
    blocks: Dict[str, List[str]],
    holdout_ids: List[str],
    ltr_c: float = 0.15,
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Execute 4-block Leave-One-Block-Out evaluation."""
    ranked_predictions: Dict[str, List[str]] = {}

    block_metrics = {}
    for block_name in sorted(blocks.keys()):
        test_ids = [q for q in blocks[block_name] if q in feature_rows]
        train_ids = [q for q in holdout_ids if q not in test_ids]

        x_train = np.vstack([feature_rows[q] for q in train_ids])
        y_train = np.concatenate([[d in queries[q][1] for d in groups[q]] for q in train_ids]).astype(np.int8)

        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=ltr_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(x_train), y_train)

        block_hits = []
        block_prec = []
        for q in test_ids:
            x_test = scaler.transform(feature_rows[q])
            proba = model.predict_proba(x_test)[:, 1]
            order = np.argsort(-proba)
            pred_order = [groups[q][i] for i in order]
            ranked_predictions[q] = pred_order

            top5 = pred_order[:5]
            gold = queries[q][1]
            hits = len(set(top5) & gold)
            block_hits.append(hits / len(gold))
            block_prec.append(hits / 5.0)

        block_metrics[block_name] = {
            "queries": len(test_ids),
            "recall_at_5": float(np.mean(block_hits)),
            "precision_at_5": float(np.mean(block_prec)),
        }

    # Pooled metrics
    all_recalls = []
    all_precisions = []
    single_recalls = []
    multi_recalls = []

    for q in holdout_ids:
        gold = queries[q][1]
        top5 = ranked_predictions[q][:5]
        hits = len(set(top5) & gold)
        rec = hits / len(gold)
        all_recalls.append(rec)
        all_precisions.append(hits / 5.0)
        if len(gold) == 1:
            single_recalls.append(rec)
        else:
            multi_recalls.append(rec)

    metrics = {
        "pooled_recall_at_5": float(np.mean(all_recalls)),
        "pooled_precision_at_5": float(np.mean(all_precisions)),
        "single_gold_recall_at_5": float(np.mean(single_recalls)),
        "multi_gold_recall_at_5": float(np.mean(multi_recalls)),
        "blocks": block_metrics,
    }

    return metrics, ranked_predictions


def compare_against_h0(
    cand_preds: Dict[str, List[str]],
    h0_preds: Dict[str, List[str]],
    queries: Dict[str, Tuple[str, Set[str]]],
    holdout_ids: List[str],
) -> Dict[str, Any]:
    wins = 0
    losses = 0
    ties = 0
    churn_queries = 0
    gold_crossings_in = 0
    gold_crossings_out = 0

    for q in holdout_ids:
        gold = queries[q][1]
        top5_cand = set(cand_preds[q][:5])
        top5_h0 = set(h0_preds[q][:5])

        rec_cand = len(top5_cand & gold) / len(gold)
        rec_h0 = len(top5_h0 & gold) / len(gold)

        if rec_cand > rec_h0:
            wins += 1
        elif rec_cand < rec_h0:
            losses += 1
        else:
            ties += 1

        if top5_cand != top5_h0:
            churn_queries += 1

        # Gold crossings
        gold_crossings_in += len((top5_cand - top5_h0) & gold)
        gold_crossings_out += len((top5_h0 - top5_cand) & gold)

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_loss_diff": wins - losses,
        "top5_churn_queries": churn_queries,
        "gold_crossings_into_top5": gold_crossings_in,
        "gold_crossings_out_of_top5": gold_crossings_out,
    }


def main():
    start_time = time.perf_counter()
    print("=== Step 2: Evaluating CAL600 Arms H0, H1, H2, H3 ===", flush=True)

    # 1. Load base H0 environment
    print("Building base H0 feature components...", flush=True)
    (
        queries,
        blocks,
        holdout_ids,
        holdout_candidates,
        holdout_views,
        names,
        training_scores,
        t_rows,
        c_rows,
        rows_h0,
        groups,
    ) = build_base_h0_components()

    # 2. Load CAL600 VietLegal-E5 scores
    e5_scores_file = RESULTS_DIR / "CAL600_VIETLEGAL_E5_SCORES.pkl"
    if not e5_scores_file.exists():
        raise FileNotFoundError(f"Missing {e5_scores_file}")
    with e5_scores_file.open("rb") as f:
        e5_data = pickle.load(f)

    frozen_scores = e5_data["frozen_scores"]
    adapted_scores = e5_data["adapted_scores"]
    frozen_orders = e5_data["frozen_orders"]
    adapted_orders = e5_data["adapted_orders"]

    # 3. Evaluate H0 (Current Public-Semantics Baseline)
    print("\n--- Arm H0: Current Public-Semantics Baseline ---", flush=True)
    dim_h0 = rows_h0[holdout_ids[0]].shape[1]
    print(f"H0 feature dimension: {dim_h0}", flush=True)
    metrics_h0, preds_h0 = evaluate_lobo(rows_h0, groups, queries, blocks, holdout_ids)
    print(f"H0 Pooled Recall@5: {metrics_h0['pooled_recall_at_5']:.6f}")
    for b in ("a", "b", "c", "d"):
        print(f"  Block {b}: {metrics_h0['blocks'][b]['recall_at_5']:.6f}")

    # 4. Build and Evaluate H1 (Frozen VietLegal-E5)
    print("\n--- Arm H1: H0 + Frozen VietLegal-E5 ---", flush=True)
    from tune_expanded_fusion_selection import ltr_features

    names_h1 = names + ["frozen_e5"]
    views_h1 = dict(holdout_views)
    views_h1["frozen_e5"] = frozen_orders
    scores_h1 = dict(training_scores)
    scores_h1["frozen_e5"] = frozen_scores

    rows_base_h1, _ = ltr_features(views_h1, names_h1, holdout_candidates, holdout_ids, scores_h1)
    rows_h1 = {}
    for q in holdout_ids:
        rows_h1[q] = np.concatenate([rows_base_h1[q], t_rows[q], c_rows[q]], axis=1)

    dim_h1 = rows_h1[holdout_ids[0]].shape[1]
    print(f"H1 feature dimension: {dim_h1} (H0 {dim_h0} + 4)", flush=True)
    metrics_h1, preds_h1 = evaluate_lobo(rows_h1, groups, queries, blocks, holdout_ids)
    comp_h1 = compare_against_h0(preds_h1, preds_h0, queries, holdout_ids)
    print(f"H1 Pooled Recall@5: {metrics_h1['pooled_recall_at_5']:.6f} (delta: {metrics_h1['pooled_recall_at_5'] - metrics_h0['pooled_recall_at_5']:+.6f})")
    for b in ("a", "b", "c", "d"):
        d = metrics_h1['blocks'][b]['recall_at_5'] - metrics_h0['blocks'][b]['recall_at_5']
        print(f"  Block {b}: {metrics_h1['blocks'][b]['recall_at_5']:.6f} (delta: {d:+.6f})")
    print(f"  Wins: {comp_h1['wins']}, Losses: {comp_h1['losses']}, Ties: {comp_h1['ties']}")

    # 5. Build and Evaluate H2 (Task-Adapted VietLegal-E5)
    print("\n--- Arm H2: H0 + Task-Adapted VietLegal-E5 ---", flush=True)
    names_h2 = names + ["adapted_e5"]
    views_h2 = dict(holdout_views)
    views_h2["adapted_e5"] = adapted_orders
    scores_h2 = dict(training_scores)
    scores_h2["adapted_e5"] = adapted_scores

    rows_base_h2, _ = ltr_features(views_h2, names_h2, holdout_candidates, holdout_ids, scores_h2)
    rows_h2 = {}
    for q in holdout_ids:
        rows_h2[q] = np.concatenate([rows_base_h2[q], t_rows[q], c_rows[q]], axis=1)

    dim_h2 = rows_h2[holdout_ids[0]].shape[1]
    print(f"H2 feature dimension: {dim_h2} (H0 {dim_h0} + 4)", flush=True)
    metrics_h2, preds_h2 = evaluate_lobo(rows_h2, groups, queries, blocks, holdout_ids)
    comp_h2 = compare_against_h0(preds_h2, preds_h0, queries, holdout_ids)
    print(f"H2 Pooled Recall@5: {metrics_h2['pooled_recall_at_5']:.6f} (delta: {metrics_h2['pooled_recall_at_5'] - metrics_h0['pooled_recall_at_5']:+.6f})")
    for b in ("a", "b", "c", "d"):
        d = metrics_h2['blocks'][b]['recall_at_5'] - metrics_h0['blocks'][b]['recall_at_5']
        print(f"  Block {b}: {metrics_h2['blocks'][b]['recall_at_5']:.6f} (delta: {d:+.6f})")
    print(f"  Wins: {comp_h2['wins']}, Losses: {comp_h2['losses']}, Ties: {comp_h2['ties']}")

    # 6. Build and Evaluate H3 (Adaptation Delta)
    print("\n--- Arm H3: H2 + 6 Adaptation Delta Features ---", flush=True)
    rows_h3 = {}
    for q in holdout_ids:
        docs = holdout_candidates[q]
        n_cands = len(docs)

        # Standardized scores for adapted and frozen
        ad_raw = adapted_scores[q]
        fr_raw = frozen_scores[q]

        ad_z, ad_gap = compute_standardized_scores(ad_raw, docs)
        fr_z, fr_gap = compute_standardized_scores(fr_raw, docs)

        ranks_ad = {d: i + 1 for i, d in enumerate(adapted_orders[q])}
        ranks_fr = {d: i + 1 for i, d in enumerate(frozen_orders[q])}

        delta_rows = []
        for i, d in enumerate(docs):
            has_ad = d in ad_raw and not (math.isnan(ad_raw[d]) or np.isnan(ad_raw[d]))
            has_fr = d in fr_raw and not (math.isnan(fr_raw[d]) or np.isnan(fr_raw[d]))

            if has_ad and has_fr:
                d_z = float(ad_z[i] - fr_z[i])
                d_gap = float(ad_gap[i] - fr_gap[i])
                r_ad = ranks_ad.get(d, 60)
                r_fr = ranks_fr.get(d, 60)
                r_gain = float(r_fr - r_ad) / float(n_cands)
                rr_gain = 1.0 / (10.0 + r_ad) - 1.0 / (10.0 + r_fr)
                prom_top5 = 1.0 if (r_ad <= 5 and r_fr > 5) else 0.0
                dem_top5 = 1.0 if (r_fr <= 5 and r_ad > 5) else 0.0
            else:
                # Explicit neutral missing treatment
                d_z = 0.0
                d_gap = 0.0
                r_gain = 0.0
                rr_gain = 0.0
                prom_top5 = 0.0
                dem_top5 = 0.0

            delta_rows.append([d_z, d_gap, r_gain, rr_gain, prom_top5, dem_top5])

        delta_block = np.asarray(delta_rows, dtype=np.float32)
        rows_h3[q] = np.concatenate([rows_h2[q], delta_block], axis=1)

    dim_h3 = rows_h3[holdout_ids[0]].shape[1]
    print(f"H3 feature dimension: {dim_h3} (H2 {dim_h2} + 6)", flush=True)
    metrics_h3, preds_h3 = evaluate_lobo(rows_h3, groups, queries, blocks, holdout_ids)
    comp_h3 = compare_against_h0(preds_h3, preds_h0, queries, holdout_ids)
    print(f"H3 Pooled Recall@5: {metrics_h3['pooled_recall_at_5']:.6f} (delta: {metrics_h3['pooled_recall_at_5'] - metrics_h0['pooled_recall_at_5']:+.6f})")
    for b in ("a", "b", "c", "d"):
        d = metrics_h3['blocks'][b]['recall_at_5'] - metrics_h0['blocks'][b]['recall_at_5']
        print(f"  Block {b}: {metrics_h3['blocks'][b]['recall_at_5']:.6f} (delta: {d:+.6f})")
    print(f"  Wins: {comp_h3['wins']}, Losses: {comp_h3['losses']}, Ties: {comp_h3['ties']}")

    # 7. Robustness Gate Check
    def check_gates(arm_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        delta_pooled = metrics["pooled_recall_at_5"] - metrics_h0["pooled_recall_at_5"]
        pooled_gain = delta_pooled > 1e-9
        block_regressions = {}
        for b in ("a", "b", "c", "d"):
            d_b = metrics["blocks"][b]["recall_at_5"] - metrics_h0["blocks"][b]["recall_at_5"]
            if d_b < -1e-9:
                block_regressions[b] = d_b
        zero_regressions = len(block_regressions) == 0
        passes = pooled_gain and zero_regressions
        return {
            "pooled_gain": pooled_gain,
            "delta_pooled": delta_pooled,
            "zero_block_regressions": zero_regressions,
            "regressing_blocks": block_regressions,
            "passes_promotion_gate": passes,
        }

    gates_h1 = check_gates("H1", metrics_h1)
    gates_h2 = check_gates("H2", metrics_h2)
    gates_h3 = check_gates("H3", metrics_h3)

    print("\n=== Robustness Gate Summary ===", flush=True)
    print(f"H1: pooled_gain={gates_h1['pooled_gain']} ({gates_h1['delta_pooled']:+.6f}), zero_regressions={gates_h1['zero_block_regressions']} -> PASS={gates_h1['passes_promotion_gate']}")
    print(f"H2: pooled_gain={gates_h2['pooled_gain']} ({gates_h2['delta_pooled']:+.6f}), zero_regressions={gates_h2['zero_block_regressions']} -> PASS={gates_h2['passes_promotion_gate']}")
    print(f"H3: pooled_gain={gates_h3['pooled_gain']} ({gates_h3['delta_pooled']:+.6f}), zero_regressions={gates_h3['zero_block_regressions']} -> PASS={gates_h3['passes_promotion_gate']}")

    elapsed = time.perf_counter() - start_time

    # 8. Save Evaluation Report
    report = {
        "schema_version": "dsc2026.gemini.huy_e5_adaptation_delta_v1.cal600_evaluation_report.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": elapsed,
        "feature_dimensions": {
            "H0": dim_h0,
            "H1": dim_h1,
            "H2": dim_h2,
            "H3": dim_h3,
        },
        "arms": {
            "H0_BASELINE": {
                "name": "Current Huy Public-Semantics Baseline",
                "feature_count": dim_h0,
                "metrics": metrics_h0,
            },
            "H1_FROZEN_E5": {
                "name": "H0 + Frozen VietLegal-E5",
                "feature_count": dim_h1,
                "metrics": metrics_h1,
                "delta_vs_h0": metrics_h1["pooled_recall_at_5"] - metrics_h0["pooled_recall_at_5"],
                "comparison": comp_h1,
                "gate": gates_h1,
            },
            "H2_ADAPTED_E5": {
                "name": "H0 + Task-Adapted VietLegal-E5",
                "feature_count": dim_h2,
                "metrics": metrics_h2,
                "delta_vs_h0": metrics_h2["pooled_recall_at_5"] - metrics_h0["pooled_recall_at_5"],
                "comparison": comp_h2,
                "gate": gates_h2,
            },
            "H3_ADAPTATION_DELTA": {
                "name": "H2 + 6 Adaptation Delta Features",
                "feature_count": dim_h3,
                "metrics": metrics_h3,
                "delta_vs_h0": metrics_h3["pooled_recall_at_5"] - metrics_h0["pooled_recall_at_5"],
                "comparison": comp_h3,
                "gate": gates_h3,
            },
        },
    }

    out_file = RESULTS_DIR / "CAL600_EVALUATION_REPORT.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {out_file}", flush=True)


if __name__ == "__main__":
    main()
