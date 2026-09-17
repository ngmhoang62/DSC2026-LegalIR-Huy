"""Tier 2: Bootstrap-Consensus Rank-6 Boundary Repair."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.gemini.huy_d1_selective_repair_v1.common import (
    BOOTSTRAP_REPLICAS,
    BOOTSTRAP_THRESHOLD,
    CAL_FROZEN_SECTION_CACHE_PATH,
    OLD_JINA_CACHE_PATH,
    ROOT,
)


def load_tier2_score_caches() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Load frozen Section-CE CV scores and historical old-Jina CV scores."""
    sec_obj = pickle.loads(CAL_FROZEN_SECTION_CACHE_PATH.read_bytes())
    sec_scores = sec_obj.get("scores", sec_obj)

    jina_obj = pickle.loads(OLD_JINA_CACHE_PATH.read_bytes())
    jina_scores = jina_obj.get("scores", jina_obj)

    return sec_scores, jina_scores


def find_rank6_proposals(
    all_ids: List[str],
    blocks: Dict[str, List[str]],
    d1_rankings: Dict[str, List[str]],
    d1_scores: Dict[str, np.ndarray],
    eval_groups: Dict[str, List[str]],
    sec_scores: Dict[str, Dict[str, float]],
    jina_scores: Dict[str, Dict[str, float]],
) -> Dict[str, List[dict]]:
    """Identify queries where rank-6 challenger satisfies the proposal gate."""
    proposals_by_block: Dict[str, List[dict]] = defaultdict(list)
    block_names = sorted(blocks.keys())

    for q in all_ids:
        q_block = next(b for b in block_names if q in blocks[b])
        ranking = d1_rankings[q]
        if len(ranking) < 6:
            continue

        defender = ranking[4]    # rank 5 (0-indexed 4)
        challenger = ranking[5]  # rank 6 (0-indexed 5)

        q_sec = sec_scores.get(q, {})
        q_jina = jina_scores.get(q, {})

        if not q_sec or not q_jina:
            continue
        if challenger not in q_sec or defender not in q_sec:
            continue
        if challenger not in q_jina or defender not in q_jina:
            continue

        sec_top5 = sorted(q_sec.keys(), key=lambda d: (-q_sec[d], d))[:5]
        cond1 = challenger in sec_top5
        cond2 = q_sec[challenger] > q_sec[defender]
        cond3 = q_jina[challenger] > q_jina[defender]

        if cond1 and cond2 and cond3:
            # Baseline D1 margin = score(defender) - score(challenger) >= 0
            q_groups = eval_groups[q]
            def_idx = q_groups.index(defender)
            chal_idx = q_groups.index(challenger)
            d1_margin = float(d1_scores[q][def_idx] - d1_scores[q][chal_idx])

            proposals_by_block[q_block].append({
                "qid": q,
                "block": q_block,
                "defender": defender,
                "challenger": challenger,
                "d1_margin": d1_margin,
                "sec_score_defender": float(q_sec[defender]),
                "sec_score_challenger": float(q_sec[challenger]),
                "sec_delta": float(q_sec[challenger] - q_sec[defender]),
                "jina_score_defender": float(q_jina[defender]),
                "jina_score_challenger": float(q_jina[challenger]),
                "jina_delta": float(q_jina[challenger] - q_jina[defender]),
            })

    return dict(proposals_by_block)


def evaluate_bootstrap_committee(
    blocks: Dict[str, List[str]],
    proposals_by_block: Dict[str, List[dict]],
    eval_rows: Dict[str, np.ndarray],
    eval_groups: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    replicas: int = BOOTSTRAP_REPLICAS,
    threshold: int = BOOTSTRAP_THRESHOLD,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Run deterministic B=51 query-bootstrap committee for proposals in each outer fold."""
    block_names = sorted(blocks.keys())
    all_evaluated_proposals: List[dict] = []

    for fold_idx, held in enumerate(block_names):
        train_ids = sum((blocks[n] for n in blocks if n != held), [])
        props = proposals_by_block.get(held, [])
        if not props:
            continue

        prop_feature_rows = {}
        for p in props:
            q = p["qid"]
            cands = eval_groups[q]
            def_idx = cands.index(p["defender"])
            chal_idx = cands.index(p["challenger"])
            prop_feature_rows[q] = {
                "def_row": eval_rows[q][def_idx : def_idx + 1],
                "chal_row": eval_rows[q][chal_idx : chal_idx + 1],
            }

        votes = {p["qid"]: 0 for p in props}
        train_matrices = {q: eval_rows[q] for q in train_ids}
        train_labels = {
            q: np.array([d in gold[q] for d in eval_groups[q]], dtype=np.int8)
            for q in train_ids
        }

        for b in range(replicas):
            seed = 2026000 + fold_idx * 1000 + b
            rng = np.random.RandomState(seed)
            sampled_qids = rng.choice(train_ids, size=len(train_ids), replace=True)

            X_b = np.vstack([train_matrices[q] for q in sampled_qids])
            y_b = np.concatenate([train_labels[q] for q in sampled_qids])

            scaler_b = StandardScaler().fit(X_b)
            model_b = LogisticRegression(
                C=0.15,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=2026,
            )
            model_b.fit(scaler_b.transform(X_b), y_b)

            for p in props:
                q = p["qid"]
                x_def = scaler_b.transform(prop_feature_rows[q]["def_row"])
                x_chal = scaler_b.transform(prop_feature_rows[q]["chal_row"])
                s_def = float(model_b.decision_function(x_def)[0])
                s_chal = float(model_b.decision_function(x_chal)[0])
                if s_chal > s_def:
                    votes[q] += 1

        for p in props:
            q = p["qid"]
            p_res = dict(p)
            v = votes[q]
            p_res["bootstrap_votes_for_challenger"] = v
            p_res["bootstrap_vote_rate"] = float(v / replicas)
            p_res["passes_action_threshold"] = v >= threshold
            p_res["defender_is_gold"] = p["defender"] in gold[q]
            p_res["challenger_is_gold"] = p["challenger"] in gold[q]
            all_evaluated_proposals.append(p_res)

    # Compile diagnostics
    total_proposals = len(all_evaluated_proposals)
    passing_proposals = [p for p in all_evaluated_proposals if p["passes_action_threshold"]]

    vote_counts = [p["bootstrap_votes_for_challenger"] for p in all_evaluated_proposals]
    margins = [p["d1_margin"] for p in all_evaluated_proposals]
    sec_deltas = [p["sec_delta"] for p in all_evaluated_proposals]
    jina_deltas = [p["jina_delta"] for p in all_evaluated_proposals]

    hist_counts, bin_edges = np.histogram(vote_counts, bins=range(0, replicas + 2, 5)) if vote_counts else ([], [])

    quantiles = {}
    if vote_counts:
        for q_pct in [0, 25, 50, 75, 90, 95, 100]:
            quantiles[f"p{q_pct}"] = float(np.percentile(vote_counts, q_pct))

    diagnostics_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.bootstrap_boundary_diagnostics.v1",
        "experiment_id": "HUY_D1_SELECTIVE_REPAIR_V1",
        "bootstrap_parameters": {
            "replicas": replicas,
            "threshold_votes": threshold,
            "threshold_rate": float(threshold / replicas),
            "sampling_unit": "whole_query_with_replacement",
            "seed_policy": "2026000 + outer_fold_index * 1000 + b",
        },
        "summary": {
            "total_rank6_proposals_before_bootstrap_gate": total_proposals,
            "proposals_passing_threshold": len(passing_proposals),
            "action_fraction": float(len(passing_proposals) / max(1, total_proposals)),
        },
        "bootstrap_vote_distribution": {
            "histogram_bins": [int(x) for x in bin_edges],
            "histogram_counts": [int(x) for x in hist_counts],
            "quantiles": quantiles,
            "mean_votes": float(np.mean(vote_counts)) if vote_counts else 0.0,
            "std_votes": float(np.std(vote_counts)) if vote_counts else 0.0,
        },
        "baseline_d1_margin_distribution": {
            "mean": float(np.mean(margins)) if margins else 0.0,
            "median": float(np.median(margins)) if margins else 0.0,
            "min": float(np.min(margins)) if margins else 0.0,
            "max": float(np.max(margins)) if margins else 0.0,
        },
        "section_delta_distribution": {
            "mean": float(np.mean(sec_deltas)) if sec_deltas else 0.0,
            "median": float(np.median(sec_deltas)) if sec_deltas else 0.0,
            "min": float(np.min(sec_deltas)) if sec_deltas else 0.0,
            "max": float(np.max(sec_deltas)) if sec_deltas else 0.0,
        },
        "old_jina_delta_distribution": {
            "mean": float(np.mean(jina_deltas)) if jina_deltas else 0.0,
            "median": float(np.median(jina_deltas)) if jina_deltas else 0.0,
            "min": float(np.min(jina_deltas)) if jina_deltas else 0.0,
            "max": float(np.max(jina_deltas)) if jina_deltas else 0.0,
        },
        "actions": passing_proposals,
        "all_proposals": all_evaluated_proposals,
    }

    return all_evaluated_proposals, diagnostics_doc
