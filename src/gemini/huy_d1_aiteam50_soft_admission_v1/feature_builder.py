"""Feature builder for candidate universe S(q) and label-free feature manifest sealing.

Module for HUY_D1_AITEAM50_SOFT_ADMISSION_V1.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import (
    RESULTS_DIR,
    ROOT,
    SafeDocumentStore,
    sha256_file,
)
from tune_citation_graph import build_citation_table, citation_features
from tune_doctype_features import build_type_table, type_features

FEATURE_NAMES = [
    # AITeam features (4)
    "aiteam_raw_score",
    "aiteam_query_zscore",
    "aiteam_rank_within_S",
    "aiteam_rrf_10",
    # Jina features (4)
    "jina_raw_score",
    "jina_query_zscore",
    "jina_rank_within_S",
    "jina_rrf_10",
    # Section features (4)
    "section_raw_score",
    "section_query_zscore",
    "section_rank_within_S",
    "section_rrf_10",
    # Doctype features (12)
    "type_luat",
    "type_nghidinh",
    "type_thongtu",
    "type_quyetdinh",
    "type_congvan",
    "type_khac",
    "type_pool_share",
    "hint_luat",
    "hint_nghidinh",
    "hint_thongtu",
    "hint_quyetdinh",
    "hint_congvan",
    # Citation features (4)
    "cite_cites_another",
    "cite_n_cited_in_pool",
    "cite_is_cited_by_another",
    "cite_n_citing_it",
]


def build_s_universe_and_novel_maps(
    all_ids: List[str],
    frozen_top50: Dict[str, List[str]],
    d1_top5: Dict[str, List[str]],
    d1_candidate_pool: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, List[str]]]:
    """Defines S(q) = deduplicated A50(q) UNION {DEFENDER(q)},
    DEFENDER(q) = D1 rank 5,
    NOVEL(q) = [d for d in A50(q) if d not in D1 candidate pool].
    """
    s_universe: Dict[str, List[str]] = {}
    defenders: Dict[str, str] = {}
    novel_map: Dict[str, List[str]] = {}

    for q in all_ids:
        a50 = frozen_top50[q]
        def_doc = d1_top5[q][4]  # 0-indexed rank 4 is D1 rank 5
        defenders[q] = def_doc

        # Deduplicated S(q) maintaining A50 order, defender appended if absent
        s_docs = list(dict.fromkeys(a50 + [def_doc]))
        s_universe[q] = s_docs

        pool_set = set(d1_candidate_pool[q])
        novel_docs = [d for d in a50 if d not in pool_set]
        novel_map[q] = novel_docs

    return s_universe, defenders, novel_map


def build_label_free_features(
    all_ids: List[str],
    s_universe: Dict[str, List[str]],
    defenders: Dict[str, str],
    novel_map: Dict[str, List[str]],
    aiteam_accessor: Any,
    jina_scores: Dict[str, Dict[str, float]],
    sec_scores: Dict[str, Dict[str, float]],
    docs: SafeDocumentStore,
    queries_label_free: Dict[str, Tuple[str, Any]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Constructs fixed 28D label-free features for every (q, d) in S(q),
    writes AITEAM50_ADMISSION_FEATURE_MANIFEST.json, and seals it before model fitting.
    """
    print("=== BUILDING LABEL-FREE 28D FEATURE MATRIX OVER S(q) ===", flush=True)
    t0 = time.perf_counter()

    # Structural features over S(q)
    print("Computing doctype and citation features over S(q)...", flush=True)
    type_table = build_type_table(ROOT, docs, all_ids, s_universe)
    type_rows = type_features(s_universe, type_table, queries_label_free, all_ids)
    own, cited = build_citation_table(docs, all_ids, s_universe)
    cite_rows = citation_features(s_universe, own, cited, all_ids)

    features_per_query: Dict[str, np.ndarray] = {}
    manifest_queries: Dict[str, Any] = {}

    for q in all_ids:
        docs_q = s_universe[q]
        n_docs = len(docs_q)

        # 1. AITeam channel scores
        a_scores = np.array([aiteam_accessor.get_score(q, d) for d in docs_q], dtype=np.float32)
        a_mu = float(np.mean(a_scores))
        a_std = float(np.std(a_scores))
        a_z = (a_scores - a_mu) / (a_std + 1e-9)
        # Ranks: higher score = rank 1, tie-break doc_id ascending
        a_order = sorted(range(n_docs), key=lambda i: (-a_scores[i], docs_q[i]))
        a_ranks = np.zeros(n_docs, dtype=np.float32)
        for rank_1based, idx in enumerate(a_order, 1):
            a_ranks[idx] = rank_1based
        a_rrf = 1.0 / (10.0 + a_ranks)

        # 2. Jina-FT channel scores
        j_scores = np.array([jina_scores[q].get(d, -1e9) for d in docs_q], dtype=np.float32)
        j_mu = float(np.mean(j_scores))
        j_std = float(np.std(j_scores))
        j_z = (j_scores - j_mu) / (j_std + 1e-9)
        j_order = sorted(range(n_docs), key=lambda i: (-j_scores[i], docs_q[i]))
        j_ranks = np.zeros(n_docs, dtype=np.float32)
        for rank_1based, idx in enumerate(j_order, 1):
            j_ranks[idx] = rank_1based
        j_rrf = 1.0 / (10.0 + j_ranks)

        # 3. Section CE channel scores
        s_scores = np.array([sec_scores[q].get(d, -1e9) for d in docs_q], dtype=np.float32)
        s_mu = float(np.mean(s_scores))
        s_std = float(np.std(s_scores))
        s_z = (s_scores - s_mu) / (s_std + 1e-9)
        s_order = sorted(range(n_docs), key=lambda i: (-s_scores[i], docs_q[i]))
        s_ranks = np.zeros(n_docs, dtype=np.float32)
        for rank_1based, idx in enumerate(s_order, 1):
            s_ranks[idx] = rank_1based
        s_rrf = 1.0 / (10.0 + s_ranks)

        # 4. Neural block: (n_docs, 12)
        neural_block = np.column_stack([
            a_scores, a_z, a_ranks, a_rrf,
            j_scores, j_z, j_ranks, j_rrf,
            s_scores, s_z, s_ranks, s_rrf,
        ])

        # 5. Full feature vector: (n_docs, 28)
        feat_q = np.concatenate([neural_block, type_rows[q], cite_rows[q]], axis=1).astype(np.float32)
        if feat_q.shape[1] != 28:
            raise ValueError(f"Feature dimension mismatch: expected 28, got {feat_q.shape[1]}")

        features_per_query[q] = feat_q

        # Manifest query record (NO gold labels)
        manifest_queries[q] = {
            "s_universe": docs_q,
            "defender": defenders[q],
            "novel_docs": novel_map[q],
            "aiteam_scores": {d: float(s) for d, s in zip(docs_q, a_scores)},
            "jina_scores": {d: float(s) for d, s in zip(docs_q, j_scores)},
            "sec_scores": {d: float(s) for d, s in zip(docs_q, s_scores)},
            "features_sha256": hashlib.sha256(feat_q.tobytes()).hexdigest(),
        }

    # Save compact NPZ
    npz_path = RESULTS_DIR / "AITEAM50_ADMISSION_FEATURES_LABEL_FREE.npz"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **features_per_query)
    npz_sha = sha256_file(npz_path)

    manifest_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.feature_manifest.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "feature_dim": 28,
        "feature_names": FEATURE_NAMES,
        "query_count": len(all_ids),
        "total_universe_candidates": sum(len(s_universe[q]) for q in all_ids),
        "total_novel_candidates": sum(len(novel_map[q]) for q in all_ids),
        "queries_with_novel_candidates": sum(1 for q in all_ids if len(novel_map[q]) > 0),
        "npz_file": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
        "npz_sha256": npz_sha,
        "label_free_assertions": {
            "no_gold_labels_used": True,
            "no_qid_features": True,
            "no_block_features": True,
            "no_gold_count_features": True,
            "no_is_novel_features": True,
            "no_known_rescue_features": True,
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "queries": manifest_queries,
    }

    manifest_path = RESULTS_DIR / "AITEAM50_ADMISSION_FEATURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    print(f"Sealed feature manifest: {manifest_path} (SHA: {manifest_sha})", flush=True)

    return features_per_query, manifest_doc
