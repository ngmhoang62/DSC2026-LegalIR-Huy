"""Learned Relation-Aware Boundary Resolver (RABR).

Implements:
- 7.1 Recover candidate features from final Huy-fasttrack endpoint
- 7.2 Small, principled relation feature family
- 7.3 Pairwise training on boundary pairs with equal query weights
- 7.4 StandardScaler + LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear")
- 7.5 Strict outer-fold inference
- 7.6 Nested inner-validation threshold selection in [0.55, 0.65, 0.75, 0.85, 0.90]
- 8.0 Evaluation of 3 endpoints: RABR_SAFE, RABR_PAIRWISE, RABR_SAFE_THEN_PAIRWISE
- 9.0 Complete audit, metrics, and report generation
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
BASE_SNAPSHOT_DIR = CURRENT_DIR / "baseline_snapshot"
sys.path.insert(0, str(BASE_SNAPSHOT_DIR))

import run_huy_5fold_fasttrack as core

RESULTS_DIR = REPO_ROOT / "results/gemini/rabr_v1"
CACHE_DIR = RESULTS_DIR / "cache"
GRAPH_FILE = RESULTS_DIR / "RELATION_GRAPH.jsonl"
PREDICTIONS_FILE = CACHE_DIR / "BASELINE_PREDICTIONS_AND_SCORES.jsonl"
FEAT_FILE = CACHE_DIR / "BASELINE_FEATURE_ROWS.npz"
META_FILE = CACHE_DIR / "DOCUMENT_METADATA.json"


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text.replace("đ", "d").replace("Đ", "D")


def load_graph():
    forward = defaultdict(list)
    backward = defaultdict(list)
    undirected = defaultdict(set)

    with GRAPH_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src, dst, rel = str(row["src_doc"]), str(row["dst_doc"]), row["relation_type"]
            forward[src].append((dst, rel))
            backward[dst].append((src, rel))
            undirected[src].add(dst)
            undirected[dst].add(src)

    # Connected components
    visited = set()
    components = {}
    comp_id = 0
    for node in list(undirected.keys()):
        if node not in visited:
            queue = [node]
            visited.add(node)
            comp_nodes = {node}
            while queue:
                curr = queue.pop()
                for neighbor in undirected[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        comp_nodes.add(neighbor)
                        queue.append(neighbor)
            for c_node in comp_nodes:
                components[c_node] = comp_id
            comp_id += 1

    return forward, backward, undirected, components


def extract_candidate_features(
    doc: str,
    qid: str,
    rank: int,
    score: float,
    base_row: np.ndarray,
    top4: List[str],
    top8: List[str],
    query: str,
    meta: Dict[str, Any],
    forward: Dict[str, List[Tuple[str, str]]],
    backward: Dict[str, List[Tuple[str, str]]],
    components: Dict[str, int],
) -> np.ndarray:
    top4_set = set(top4)
    top8_set = set(top8)

    # 1. Superseded / Supersedes
    superseded_by_top8 = 0.0
    supersedes_top8 = 0.0
    repealed_by_top8 = 0.0
    repeals_top8 = 0.0
    amends_top4 = 0.0
    amended_by_top4 = 0.0
    guides_top4 = 0.0
    guided_by_top4 = 0.0
    cites_top4_count = 0.0
    cited_by_top4_count = 0.0

    for dst, rel in forward[doc]:
        if dst in top8_set:
            if rel == "REPLACES": supersedes_top8 = 1.0
            elif rel == "REPEALS": repeals_top8 = 1.0
        if dst in top4_set:
            if rel == "AMENDS": amends_top4 = 1.0
            elif rel == "GUIDES": guides_top4 = 1.0
            elif rel == "CITES": cites_top4_count += 1.0

    for src, rel in backward[doc]:
        if src in top8_set:
            if rel == "REPLACES": superseded_by_top8 = 1.0
            elif rel == "REPEALS": repealed_by_top8 = 1.0
        if src in top4_set:
            if rel == "AMENDS": amended_by_top4 = 1.0
            elif rel == "GUIDES": guided_by_top4 = 1.0
            elif rel == "CITES": cited_by_top4_count += 1.0

    # Degrees
    total_rel_top8 = float(len([dst for dst, _ in forward[doc] if dst in top8_set] + [src for src, _ in backward[doc] if src in top8_set])) / 8.0
    total_rel_top4 = float(len([dst for dst, _ in forward[doc] if dst in top4_set] + [src for src, _ in backward[doc] if src in top4_set])) / 4.0

    # Components
    doc_comp = components.get(doc)
    top1_comp = components.get(top4[0]) if top4 else None
    same_comp_top1 = 1.0 if (doc_comp is not None and doc_comp == top1_comp) else 0.0
    same_comp_top4 = 1.0 if (doc_comp is not None and any(doc_comp == components.get(t) for t in top4 if components.get(t) is not None)) else 0.0

    # Document metadata
    doc_m = meta.get(doc, {})
    year = doc_m.get("year")
    has_year = 1.0 if year else 0.0
    norm_year = ((year - 2000.0) / 20.0) if year else 0.0

    # Query textual matches
    q_norm = strip_accents(query).upper()
    q_norm_clean = re.sub(r"[\s\-_]+", " ", q_norm)
    ref = doc_m.get("official_number")
    query_mentions_ref = 0.0
    if ref:
        ref_clean = re.sub(r"[\s\-_]+", " ", strip_accents(ref).upper())
        if ref_clean in q_norm_clean:
            query_mentions_ref = 1.0
        else:
            m_ny = re.search(r"(\d+\/\d{4})", ref)
            if m_ny and m_ny.group(1) in q_norm:
                query_mentions_ref = 1.0

    query_mentions_year = 1.0 if (year and re.search(rf"\b{year}\b", q_norm)) else 0.0

    # Query cues
    q_has_amend = 1.0 if any(w in q_norm for w in ["SUA DOI", "BO SUNG"]) else 0.0
    q_has_replace = 1.0 if any(w in q_norm for w in ["THAY THE", "BAI BO", "HET HIEU LUC"]) else 0.0
    q_has_temporal = 1.0 if any(w in q_norm for w in ["HIEN HANH", "MOI NHAT", "TU NGAY", "NAM "]) else 0.0

    relation_feats = np.array([
        superseded_by_top8,
        supersedes_top8,
        repealed_by_top8,
        repeals_top8,
        amends_top4,
        amended_by_top4,
        guides_top4,
        guided_by_top4,
        cites_top4_count / 4.0,
        cited_by_top4_count / 4.0,
        total_rel_top8,
        total_rel_top4,
        same_comp_top1,
        same_comp_top4,
        has_year,
        norm_year,
        query_mentions_ref,
        query_mentions_year,
        q_has_amend,
        q_has_replace,
        q_has_temporal,
        score,
        rank / 10.0,
    ], dtype=np.float32)

    return np.concatenate([base_row, relation_feats])


def build_pair_features(
    challenger_doc: str,
    defender_doc: str,
    c_feats: np.ndarray,
    d_feats: np.ndarray,
    c_score: float,
    d_score: float,
    c_rank: int,
    d_rank: int,
    forward: Dict[str, List[Tuple[str, str]]],
    backward: Dict[str, List[Tuple[str, str]]],
) -> np.ndarray:
    diff_feats = c_feats - d_feats

    # Relation-specific pair indicators
    c_replaces_d = 0.0
    d_replaces_c = 0.0
    c_repeals_d = 0.0
    d_repeals_c = 0.0
    c_amends_d = 0.0
    d_amends_c = 0.0
    c_guides_d = 0.0
    d_guides_c = 0.0
    c_cites_d = 0.0
    d_cites_c = 0.0

    for dst, rel in forward[challenger_doc]:
        if dst == defender_doc:
            if rel == "REPLACES": c_replaces_d = 1.0
            elif rel == "REPEALS": c_repeals_d = 1.0
            elif rel == "AMENDS": c_amends_d = 1.0
            elif rel == "GUIDES": c_guides_d = 1.0
            elif rel == "CITES": c_cites_d = 1.0

    for dst, rel in forward[defender_doc]:
        if dst == challenger_doc:
            if rel == "REPLACES": d_replaces_c = 1.0
            elif rel == "REPEALS": d_repeals_c = 1.0
            elif rel == "AMENDS": d_amends_c = 1.0
            elif rel == "GUIDES": d_guides_c = 1.0
            elif rel == "CITES": d_cites_c = 1.0

    score_margin = c_score - d_score
    rank_distance = float(c_rank - d_rank) / 3.0

    pair_indicators = np.array([
        c_replaces_d,
        d_replaces_c,
        c_repeals_d,
        d_repeals_c,
        c_amends_d,
        d_amends_c,
        c_guides_d,
        d_guides_c,
        c_cites_d,
        d_cites_c,
        score_margin,
        rank_distance,
    ], dtype=np.float32)

    return np.concatenate([diff_feats, pair_indicators])


def main():
    started = time.perf_counter()
    print("=== Step 7: Learned Relation-Aware Boundary Resolver (RABR) ===", flush=True)

    folds, pools, questions, golds, _, _, dup, _ = core.load_inputs()
    fold_for = {qid: fold for fold, qids in folds.items() for qid in qids}
    forward, backward, undirected, components = load_graph()

    with META_FILE.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load baseline predictions and decision scores
    predictions = {}
    scores = {}
    with PREDICTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            predictions[qid] = row["order"]
            scores[qid] = row["scores"]

    # Load raw feature rows
    feat_data = np.load(FEAT_FILE)

    print("Extracting candidate and pair features for all queries...", flush=True)
    t0 = time.perf_counter()

    # Pre-extract features for ranks 5-8
    cand_feats = {}
    doc_pos_cache = {qid: {doc: i for i, doc in enumerate(pools[qid])} for qid in pools}

    for qid in pools:
        order = predictions[qid]
        top4 = order[:4]
        top8 = order[:8]
        q_text = questions[qid]
        q_rows = feat_data[f"row_{qid}"]
        q_scores = scores[qid]
        pos_map = doc_pos_cache[qid]

        for rank_idx in range(4, 8):
            doc = order[rank_idx]
            cand_feats[(qid, doc)] = extract_candidate_features(
                doc=doc,
                qid=qid,
                rank=rank_idx + 1,
                score=q_scores[doc],
                base_row=q_rows[pos_map[doc]],
                top4=top4,
                top8=top8,
                query=q_text,
                meta=meta,
                forward=forward,
                backward=backward,
                components=components,
            )

    print(f"Pre-extracted candidate features in {time.perf_counter() - t0:.1f}s. Feature dim: {cand_feats[next(iter(cand_feats))].shape[0]}", flush=True)

    # Build all pairs: (qid, challenger, defender)
    query_pairs = defaultdict(list)
    query_pair_features = {}

    for qid in pools:
        order = predictions[qid]
        q_scores = scores[qid]
        defender = order[4]  # Rank 5
        d_feats = cand_feats[(qid, defender)]
        d_score = q_scores[defender]

        for c_rank_idx in range(5, 8):
            challenger = order[c_rank_idx]
            c_feats = cand_feats[(qid, challenger)]
            c_score = q_scores[challenger]

            pair_vec = build_pair_features(
                challenger_doc=challenger,
                defender_doc=defender,
                c_feats=c_feats,
                d_feats=d_feats,
                c_score=c_score,
                d_score=d_score,
                c_rank=c_rank_idx + 1,
                d_rank=5,
                forward=forward,
                backward=backward,
            )

            query_pairs[qid].append((challenger, defender, c_rank_idx + 1, pair_vec))

    # Construct training datasets
    THRESHOLD_CANDIDATES = [0.55, 0.65, 0.75, 0.85, 0.90]

    def get_training_data(train_qids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_list, y_list, w_list = [], [], []
        for qid in train_qids:
            gold = golds[qid]
            defender = predictions[qid][4]
            d_is_gold = defender in gold

            usable = []
            for challenger, _, c_rank, pair_vec in query_pairs[qid]:
                c_is_gold = challenger in gold
                if c_is_gold and not d_is_gold:
                    usable.append((pair_vec, 1.0))
                elif d_is_gold and not c_is_gold:
                    usable.append((pair_vec, 0.0))

            if usable:
                weight_per_pair = 1.0 / len(usable)
                for p_vec, target in usable:
                    x_list.append(p_vec)
                    y_list.append(target)
                    w_list.append(weight_per_pair)

        return (
            np.asarray(x_list, dtype=np.float32),
            np.asarray(y_list, dtype=np.float32),
            np.asarray(w_list, dtype=np.float32),
        )

    # 7.5 & 7.6: Strict outer-fold inference with nested threshold selection
    selected_thresholds = {}
    pairwise_preds = {}
    pairwise_action_records = []
    all_qids = set(pools)

    for outer, test_ids in folds.items():
        outer_started = time.perf_counter()
        blocked = set(map(str, dup.get(outer, [])))
        outer_train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)

        # Inner cross-validation across the 4 training folds to choose threshold
        inner_threshold_recalls = {t: [] for t in THRESHOLD_CANDIDATES}

        for inner in folds:
            if inner == outer:
                continue
            inner_test_ids = [qid for qid in outer_train_ids if fold_for[qid] == inner]
            inner_blocked = set(map(str, dup.get(inner, [])))
            inner_train_ids = [qid for qid in outer_train_ids if fold_for[qid] != inner and qid not in inner_blocked]

            x_inner, y_inner, w_inner = get_training_data(inner_train_ids)
            scaler_inner = StandardScaler().fit(x_inner)
            model_inner = LogisticRegression(
                C=0.15, class_weight="balanced", solver="liblinear",
                max_iter=3000, random_state=2026,
            ).fit(scaler_inner.transform(x_inner), y_inner, sample_weight=w_inner)

            # Predict on inner_test_ids and evaluate each threshold
            for t_val in THRESHOLD_CANDIDATES:
                inner_hits = 0.0
                for qid in inner_test_ids:
                    gold = golds[qid]
                    order = list(predictions[qid])
                    top4 = order[:4]
                    defender = order[4]

                    # Predict probability for each challenger
                    best_challenger = None
                    best_prob = -1.0
                    best_orig_rank = 999

                    for challenger, _, c_rank, pair_vec in query_pairs[qid]:
                        prob = float(model_inner.predict_proba(scaler_inner.transform(pair_vec.reshape(1, -1)))[0, 1])
                        if prob >= t_val:
                            # Higher prob, tie-break by lower original rank, then doc ID
                            if (prob > best_prob) or (abs(prob - best_prob) < 1e-7 and c_rank < best_orig_rank):
                                best_prob = prob
                                best_challenger = challenger
                                best_orig_rank = c_rank

                    slot5 = best_challenger if best_challenger else defender
                    hits = len((set(top4) | {slot5}) & gold) / len(gold)
                    inner_hits += hits

                inner_threshold_recalls[t_val].append(inner_hits / len(inner_test_ids))

        # Select threshold with best average Recall@5, ties choose HIGHER threshold
        avg_recalls = {t_val: float(np.mean(inner_threshold_recalls[t_val])) for t_val in THRESHOLD_CANDIDATES}
        best_t = max(THRESHOLD_CANDIDATES, key=lambda t: (avg_recalls[t], t))
        selected_thresholds[outer] = best_t
        print(f"[{outer}] Inner CV Recall@5 by threshold: {avg_recalls} -> Selected threshold: {best_t}", flush=True)

        # Refit on all outer_train_ids
        x_outer, y_outer, w_outer = get_training_data(outer_train_ids)
        scaler_outer = StandardScaler().fit(x_outer)
        model_outer = LogisticRegression(
            C=0.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=2026,
        ).fit(scaler_outer.transform(x_outer), y_outer, sample_weight=w_outer)

        # Apply to test_ids
        for qid in test_ids:
            order = list(predictions[qid])
            top4 = order[:4]
            defender = order[4]

            best_challenger = None
            best_prob = -1.0
            best_orig_rank = 999

            for challenger, _, c_rank, pair_vec in query_pairs[qid]:
                prob = float(model_outer.predict_proba(scaler_outer.transform(pair_vec.reshape(1, -1)))[0, 1])
                if prob >= best_t:
                    if (prob > best_prob) or (abs(prob - best_prob) < 1e-7 and c_rank < best_orig_rank):
                        best_prob = prob
                        best_challenger = challenger
                        best_orig_rank = c_rank

            if best_challenger and best_challenger != defender:
                new_order = list(top4) + [best_challenger] + [doc for doc in order[4:] if doc != best_challenger]
                pairwise_preds[qid] = new_order
                pairwise_action_records.append({
                    "qid": qid,
                    "outer_fold": outer,
                    "defender": defender,
                    "challenger": best_challenger,
                    "challenger_orig_rank": best_orig_rank,
                    "predicted_prob": best_prob,
                    "threshold": best_t,
                    "defender_is_gold": defender in golds[qid],
                    "challenger_is_gold": best_challenger in golds[qid],
                })
            else:
                pairwise_preds[qid] = order

        elapsed = time.perf_counter() - outer_started
        print(f"[{outer}] Outer fold complete in {elapsed:.1f}s", flush=True)

    # 8. Construct 3 Endpoints:
    # 1. RABR_SAFE
    # 2. RABR_PAIRWISE
    # 3. RABR_SAFE_THEN_PAIRWISE
    with (RESULTS_DIR / "RABR_SAFE_REPORT.json").open("r", encoding="utf-8") as f:
        safe_report = json.load(f)
    safe_actions_by_qid = {str(a["qid"]): a for a in safe_report["actions"]}

    # Load safe preds
    safe_preds = {}
    with (RESULTS_DIR / "RABR_SAFE_PREDICTIONS.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                safe_preds[str(row["qid"])] = row["order"]

    # SAFE_THEN_PAIRWISE
    safe_then_pairwise_preds = {}
    safe_then_pairwise_actions = []

    for qid in sorted(predictions, key=int):
        if qid in safe_actions_by_qid:
            # Safe already modified this query -> keep safe prediction
            safe_then_pairwise_preds[qid] = safe_preds[qid]
            safe_then_pairwise_actions.append({
                "qid": qid,
                "source": "RABR_SAFE",
                "action": safe_actions_by_qid[qid],
            })
        else:
            # Safe did not modify -> check if pairwise modified
            if pairwise_preds[qid] != predictions[qid]:
                safe_then_pairwise_preds[qid] = pairwise_preds[qid]
                safe_then_pairwise_actions.append({
                    "qid": qid,
                    "source": "RABR_PAIRWISE",
                    "action": next(a for a in pairwise_action_records if a["qid"] == qid),
                })
            else:
                safe_then_pairwise_preds[qid] = predictions[qid]

    # Evaluate all endpoints
    base_metrics = core.metrics(predictions, golds, folds)
    safe_metrics = core.metrics(safe_preds, golds, folds)
    pairwise_metrics = core.metrics(pairwise_preds, golds, folds)
    stp_metrics = core.metrics(safe_then_pairwise_preds, golds, folds)

    def analyze_actions(variant_preds, actions_list):
        wins, losses, ties = 0, 0, 0
        gold_crossings_in = 0
        gold_crossings_out = 0
        rank_distribution = Counter()
        baseline_ranks_recovered = []

        for qid in predictions:
            gold = golds[qid]
            b_hit = len(set(predictions[qid][:5]) & gold) / len(gold)
            v_hit = len(set(variant_preds[qid][:5]) & gold) / len(gold)
            if v_hit > b_hit + 1e-9:
                wins += 1
                gold_crossings_in += 1
                # Find which doc was promoted and its baseline rank
                promoted = variant_preds[qid][4]
                baseline_ranks_recovered.append(predictions[qid].index(promoted) + 1)
            elif v_hit < b_hit - 1e-9:
                losses += 1
                gold_crossings_out += 1
            else:
                ties += 1

        for a in actions_list:
            r = a.get("challenger_orig_rank") or a.get("action", {}).get("challenger_orig_rank") or a.get("replacement_orig_rank")
            if r:
                rank_distribution[r] += 1

        changed_queries = sum(1 for qid in predictions if variant_preds[qid][:5] != predictions[qid][:5])
        swap_precision = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0

        return {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "changed_queries": changed_queries,
            "number_of_swaps": changed_queries,
            "swap_precision": swap_precision,
            "gold_crossings_into_top5": gold_crossings_in,
            "gold_crossings_out_of_top5": gold_crossings_out,
            "distribution_promoted_ranks": dict(rank_distribution),
            "baseline_ranks_of_recovered_gold": baseline_ranks_recovered,
        }

    pairwise_analysis = analyze_actions(pairwise_preds, pairwise_action_records)
    stp_analysis = analyze_actions(safe_then_pairwise_preds, safe_then_pairwise_actions)
    safe_analysis = analyze_actions(safe_preds, safe_report["actions"])

    # Compute deltas
    def compute_deltas(metrics):
        per_fold = {}
        for f in folds:
            per_fold[f] = metrics["per_fold_recall_at_5"][f] - base_metrics["per_fold_recall_at_5"][f]
        return {
            "delta_recall_at_5": metrics["recall_at_5"] - base_metrics["recall_at_5"],
            "delta_precision_at_5": metrics["precision_at_5"] - base_metrics["precision_at_5"],
            "delta_single_gold_recall_at_5": metrics["single_gold_recall_at_5"] - base_metrics["single_gold_recall_at_5"],
            "delta_multi_gold_recall_at_5": metrics["multi_gold_recall_at_5"] - base_metrics["multi_gold_recall_at_5"],
            "per_fold_deltas": per_fold,
            "worst_fold_delta": min(per_fold.values()),
        }

    final_report = {
        "schema_version": "dsc2026.gemini.rabr_v1.final_report.v1",
        "baseline": {
            "endpoint": "profile_memory_plus_sparse_rank_scores",
            "metrics": base_metrics,
        },
        "variants": {
            "RABR_SAFE": {
                "metrics": safe_metrics,
                "deltas": compute_deltas(safe_metrics),
                "analysis": safe_analysis,
            },
            "RABR_PAIRWISE": {
                "metrics": pairwise_metrics,
                "deltas": compute_deltas(pairwise_metrics),
                "analysis": pairwise_analysis,
                "selected_thresholds": selected_thresholds,
            },
            "RABR_SAFE_THEN_PAIRWISE": {
                "metrics": stp_metrics,
                "deltas": compute_deltas(stp_metrics),
                "analysis": stp_analysis,
                "selected_thresholds": selected_thresholds,
            },
        },
        "runtime_seconds": time.perf_counter() - started,
    }

    report_path = RESULTS_DIR / "RABR_FINAL_REPORT.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {report_path}", flush=True)

    # Pick best variant
    best_variant_name = max(
        ["RABR_SAFE", "RABR_PAIRWISE", "RABR_SAFE_THEN_PAIRWISE"],
        key=lambda name: final_report["variants"][name]["metrics"]["recall_at_5"]
    )
    best_variant = final_report["variants"][best_variant_name]
    best_delta = best_variant["deltas"]["delta_recall_at_5"]
    best_worst_fold = best_variant["deltas"]["worst_fold_delta"]
    best_wins = best_variant["analysis"]["wins"]
    best_losses = best_variant["analysis"]["losses"]

    print("=== FINAL COMPARISON ===")
    print(f"Baseline Recall@5: {base_metrics['recall_at_5']:.9f}")
    for vname in ["RABR_SAFE", "RABR_PAIRWISE", "RABR_SAFE_THEN_PAIRWISE"]:
        v = final_report["variants"][vname]
        print(f"{vname}: R@5={v['metrics']['recall_at_5']:.9f} (delta={v['deltas']['delta_recall_at_5']:+.9f}, wins={v['analysis']['wins']}, losses={v['analysis']['losses']}, worst_fold={v['deltas']['worst_fold_delta']:+.9f})")

    # Output RABR_OOF_PREDICTIONS.jsonl with best variant full rankings
    best_preds_map = {
        "RABR_SAFE": safe_preds,
        "RABR_PAIRWISE": pairwise_preds,
        "RABR_SAFE_THEN_PAIRWISE": safe_then_pairwise_preds,
    }[best_variant_name]

    oof_pred_path = RESULTS_DIR / "RABR_OOF_PREDICTIONS.jsonl"
    with oof_pred_path.open("w", encoding="utf-8") as f:
        for qid in sorted(best_preds_map, key=int):
            f.write(json.dumps({"qid": qid, "order": best_preds_map[qid][:20]}, separators=(",", ":")) + "\n")
    print(f"Wrote {oof_pred_path}", flush=True)

    # Decision criteria
    decision = ""
    decision_reason = ""
    if best_delta >= 0.0010 and best_wins > best_losses and best_worst_fold >= -0.0015:
        decision = "PROMOTE"
        decision_reason = f"Best variant {best_variant_name} achieved delta R@5 = {best_delta:+.6f} >= +0.0010, wins ({best_wins}) > losses ({best_losses}), and worst fold delta = {best_worst_fold:+.6f} >= -0.0015."
    elif 0 < best_delta < 0.0010:
        decision = "MARGINAL"
        decision_reason = f"Best variant {best_variant_name} achieved positive delta R@5 = {best_delta:+.6f}, but under the +0.0010 promotion threshold."
    else:
        decision = "KILL"
        decision_reason = f"Delta R@5 = {best_delta:+.6f} <= 0, or losses ({best_losses}) >= wins ({best_wins})."

    decision_md = f"""# DECISION: {decision}

**Reason**: {decision_reason}

## Summary of Results

| Endpoint | Recall@5 | Delta R@5 | Precision@5 | Single-Gold R@5 | Multi-Gold R@5 | Worst-Fold Delta | Wins | Losses | Swaps | Swap Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | `{base_metrics['recall_at_5']:.9f}` | `+0.000000` | `{base_metrics['precision_at_5']:.9f}` | `{base_metrics['single_gold_recall_at_5']:.9f}` | `{base_metrics['multi_gold_recall_at_5']:.9f}` | `+0.000000` | - | - | - | - |
| **RABR_SAFE** | `{safe_metrics['recall_at_5']:.9f}` | `{final_report['variants']['RABR_SAFE']['deltas']['delta_recall_at_5']:+.9f}` | `{safe_metrics['precision_at_5']:.9f}` | `{safe_metrics['single_gold_recall_at_5']:.9f}` | `{safe_metrics['multi_gold_recall_at_5']:.9f}` | `{final_report['variants']['RABR_SAFE']['deltas']['worst_fold_delta']:+.9f}` | {safe_analysis['wins']} | {safe_analysis['losses']} | {safe_analysis['number_of_swaps']} | {safe_analysis['swap_precision']:.2%} |
| **RABR_PAIRWISE** | `{pairwise_metrics['recall_at_5']:.9f}` | `{final_report['variants']['RABR_PAIRWISE']['deltas']['delta_recall_at_5']:+.9f}` | `{pairwise_metrics['precision_at_5']:.9f}` | `{pairwise_metrics['single_gold_recall_at_5']:.9f}` | `{pairwise_metrics['multi_gold_recall_at_5']:.9f}` | `{final_report['variants']['RABR_PAIRWISE']['deltas']['worst_fold_delta']:+.9f}` | {pairwise_analysis['wins']} | {pairwise_analysis['losses']} | {pairwise_analysis['number_of_swaps']} | {pairwise_analysis['swap_precision']:.2%} |
| **RABR_SAFE_THEN_PAIRWISE** | `{stp_metrics['recall_at_5']:.9f}` | `{final_report['variants']['RABR_SAFE_THEN_PAIRWISE']['deltas']['delta_recall_at_5']:+.9f}` | `{stp_metrics['precision_at_5']:.9f}` | `{stp_metrics['single_gold_recall_at_5']:.9f}` | `{stp_metrics['multi_gold_recall_at_5']:.9f}` | `{final_report['variants']['RABR_SAFE_THEN_PAIRWISE']['deltas']['worst_fold_delta']:+.9f}` | {stp_analysis['wins']} | {stp_analysis['losses']} | {stp_analysis['number_of_swaps']} | {stp_analysis['swap_precision']:.2%} |

## Per-Fold Recall@5
- Baseline: `{base_metrics['per_fold_recall_at_5']}`
- RABR_SAFE: `{safe_metrics['per_fold_recall_at_5']}`
- RABR_PAIRWISE: `{pairwise_metrics['per_fold_recall_at_5']}`
- RABR_SAFE_THEN_PAIRWISE: `{stp_metrics['per_fold_recall_at_5']}`

## Selected Thresholds (Outer Folds)
`{selected_thresholds}`
"""
    decision_path = RESULTS_DIR / "DECISION.md"
    decision_path.write_text(decision_md, encoding="utf-8")
    print(f"Wrote {decision_path}", flush=True)
    print(f"DECISION: {decision}", flush=True)


if __name__ == "__main__":
    main()
