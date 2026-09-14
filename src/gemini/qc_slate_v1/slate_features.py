"""Query-conditioned slate feature extraction for QCSC v1.

Extracts candidate slate feature vectors (56 slates per query) across 4 families:
1. Unary baseline evidence (7 features)
2. Query-conditioned pair completion (6 features)
3. Neighbour answer-set coverage (5 features)
4. Cardinality-conditioned interactions (4 features)
Total feature dimension D = 22.

Also generates PAIR_SUPPORT_AUDIT.json (comparing K=32 vs K=16).
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from common import (
    RESULTS_DIR,
    get_56_slates,
    load_baseline_data,
    load_lal_query_vectors,
)


SLATE_INDICES = list(combinations(range(8), 5))  # 56 tuples of 5 candidate indices
SLATE_PAIRS_INDICES = [list(combinations(s, 2)) for s in SLATE_INDICES]  # 56 lists of 10 pairs


def precompute_support_neighbors(
    queries: List[str],
    support_queries: List[str],
    lal_vecs: np.ndarray,
    qrow: Dict[str, int],
    K: int,
    is_training_self: bool = False,
) -> Tuple[np.ndarray, List[List[str]]]:
    """Find K nearest support neighbours for queries.
    
    If is_training_self is True, queries == support_queries, and self-similarity is masked to -inf.
    Returns:
    - sims: (N, K) float array
    - neighbor_qids: list of N lists of K query IDs
    """
    q_mat = lal_vecs[np.array([qrow[q] for q in queries], dtype=np.int64)]
    supp_mat = lal_vecs[np.array([qrow[q] for q in support_queries], dtype=np.int64)]

    sim_matrix = np.dot(q_mat, supp_mat.T)
    if is_training_self:
        np.fill_diagonal(sim_matrix, -np.inf)

    topk_idx = np.argpartition(-sim_matrix, K, axis=1)[:, :K]

    sims_list = []
    neighbor_qids = []
    for row_i in range(len(queries)):
        part_idx = topk_idx[row_i]
        sorted_part = sorted(part_idx, key=lambda j: (-sim_matrix[row_i, j], support_queries[j]))
        sims_list.append(sim_matrix[row_i, sorted_part])
        neighbor_qids.append([support_queries[j] for j in sorted_part])

    return np.array(sims_list, dtype=np.float32), neighbor_qids


def extract_slate_feature_tensor_for_query(
    qid: str,
    top8: List[str],
    scores: Dict[str, float],
    sims: np.ndarray,
    neighbor_qids: List[str],
    golds: Dict[str, Set[str]],
    p_multi: float,
) -> np.ndarray:
    """Extract (56, 22) feature matrix for one query."""
    K = len(neighbor_qids)
    weights = np.array([math.exp(20.0 * (float(s) - 1.0)) for s in sims], dtype=np.float64)
    sum_w = float(np.sum(weights)) + 1e-12

    # Pre-compute presence of Top-8 candidates in the K neighbours: (8, K) bool array
    cand_in_n = np.zeros((8, K), dtype=bool)
    for c_idx, doc_id in enumerate(top8):
        for k_idx, nq in enumerate(neighbor_qids):
            if doc_id in golds[nq]:
                cand_in_n[c_idx, k_idx] = True

    # Pre-compute pair statistics for all 28 pairs among Top-8
    pair_mass: Dict[Tuple[int, int], float] = {}
    pair_seen: Dict[Tuple[int, int], float] = {}
    pair_cond: Dict[Tuple[int, int], float] = {}
    pair_max_sim: Dict[Tuple[int, int], float] = {}

    for i in range(8):
        w_a = float(np.sum(weights[cand_in_n[i]]))
        for j in range(i + 1, 8):
            both = cand_in_n[i] & cand_in_n[j]
            c_both = int(np.sum(both))
            if c_both > 0:
                w_both = float(np.sum(weights[both]))
                w_b = float(np.sum(weights[cand_in_n[j]]))
                pair_mass[(i, j)] = w_both
                pair_seen[(i, j)] = 1.0
                pair_cond[(i, j)] = (w_both / (w_a + 1e-4)) + (w_both / (w_b + 1e-4))
                pair_max_sim[(i, j)] = float(np.max(sims[both]))
            else:
                pair_mass[(i, j)] = 0.0
                pair_seen[(i, j)] = 0.0
                pair_cond[(i, j)] = 0.0
                pair_max_sim[(i, j)] = 0.0

    # Neighbor gold set sizes: (K,)
    neighbor_gold_sizes = np.array([max(1, len(golds[nq])) for nq in neighbor_qids], dtype=np.float32)
    is_multi_neighbor = np.array([len(golds[nq]) > 1 for nq in neighbor_qids], dtype=bool)
    top3_w = weights[:3]
    sum_top3_w = float(np.sum(top3_w)) + 1e-12

    # Allocate (56, 22) array
    slate_feats = np.zeros((56, 22), dtype=np.float32)
    s_scores_cache = [scores[d] for d in top8]

    for s_idx, slate in enumerate(SLATE_INDICES):
        # 1. Unary baseline features
        s_scores = [s_scores_cache[i] for i in slate]
        s_sum = sum(s_scores)
        slate_feats[s_idx, 0] = s_sum
        slate_feats[s_idx, 1] = s_sum / 5.0
        slate_feats[s_idx, 2] = min(s_scores)
        slate_feats[s_idx, 3] = max(s_scores)
        slate_feats[s_idx, 4] = sum(1.0 / (i + 1) for i in slate)
        slate_feats[s_idx, 5] = sum(1.0 for i in slate if i < 5)
        slate_feats[s_idx, 6] = 1.0 if s_idx == 0 else 0.0

        # 2. Query-conditioned pair completion (10 pairs)
        pairs_in_s = SLATE_PAIRS_INDICES[s_idx]
        m_ab = [pair_mass[p] for p in pairs_in_s]
        slate_feats[s_idx, 7] = sum(m_ab)
        slate_feats[s_idx, 8] = max(m_ab)
        slate_feats[s_idx, 9] = sum(m_ab) / 10.0
        slate_feats[s_idx, 10] = sum(pair_seen[p] for p in pairs_in_s)
        slate_feats[s_idx, 11] = sum(pair_cond[p] for p in pairs_in_s)
        slate_feats[s_idx, 12] = max(pair_max_sim[p] for p in pairs_in_s)

        # 3. Neighbour answer-set coverage
        cov_k = np.sum(cand_in_n[list(slate)], axis=0) / neighbor_gold_sizes  # (K,)
        slate_feats[s_idx, 13] = float(np.sum(weights * cov_k) / sum_w)
        slate_feats[s_idx, 14] = float(np.max(weights * cov_k))
        slate_feats[s_idx, 15] = float(np.sum(top3_w * cov_k[:3]) / sum_top3_w)

        full_covered = (cov_k >= 1.0 - 1e-6)
        slate_feats[s_idx, 16] = float(np.sum(weights[full_covered]))
        slate_feats[s_idx, 17] = float(np.sum(weights[full_covered & is_multi_neighbor]))

        # 4. Cardinality-conditioned interactions
        slate_feats[s_idx, 18] = p_multi
        slate_feats[s_idx, 19] = p_multi * slate_feats[s_idx, 7]
        slate_feats[s_idx, 20] = p_multi * slate_feats[s_idx, 13]
        slate_feats[s_idx, 21] = p_multi * slate_feats[s_idx, 17]

    return slate_feats
