#!/usr/bin/env python
"""
HUY_D1_NONCAL_SPARSE_ADMISSION_V1
================================

Goal
----
Train a conservative swap-utility model ONLY on labelled NON-CAL queries,
nested-calibrate its admission threshold there, freeze it, then apply zero-shot
to exact D1 CAL600.

Crucially, raw LegalIR BM25/trigram Top-20 candidates are NOT filtered to the
D1 pool, so this experiment can repair candidate-membership failures.

No CAL label is used to fit the utility model or choose the threshold.

Run (Git Bash, from sota repo):
    python ../run_noncal_sparse_admission_v1.py --repo-root /d/Study/DSC2026/sota

Optional:
    --sources-db /d/Study/DSC2026/LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SEED = 2026
EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_DIM = 48
EXPECTED_BLOCKS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
RRF_WEIGHTS = [0.063, 0.357, 0.28, 0.30]
RRF_K = 5
SPARSE_DEPTH = 20
RANK_MISSING = 1000

# Threshold calibration safety contract.
CALIBRATION_MAX_ACTION_RATE = 0.03
CALIBRATION_MIN_BENEFICIAL = 2
CALIBRATION_MIN_PRECISION = 0.50

# Non-CAL end-to-end gate before CAL actions are even evaluated.
NONCAL_GATE_MIN_BENEFICIAL = 5
NONCAL_GATE_MAX_HARMFUL = 1
NONCAL_GATE_MIN_ACTION_PRECISION = 0.40
NONCAL_GATE_MIN_POSITIVE_PAIRS = 15

# CAL deployment safety.
CAL_MAX_ACTIONS = 30

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}

DUPLICATE_MAP = {
    "121575": "84226",
    "158189": "206810",
    "184972": "206810",
    "254937": "280171",
    "35337": "277743",
}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_info(root: Path) -> Dict[str, Any]:
    def cmd(*args: str) -> str:
        try:
            return subprocess.check_output(
                list(args), cwd=str(root), text=True, stderr=subprocess.STDOUT
            ).strip()
        except Exception as e:
            return f"ERROR: {e}"

    return {
        "head": cmd("git", "rev-parse", "HEAD"),
        "origin_main": cmd("git", "rev-parse", "origin/main"),
        "porcelain": cmd("git", "status", "--porcelain"),
    }


def stable_fold(qid: str, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{SEED}|{qid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def load_pkl(root: Path, rel_path: str):
    p = root / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_aligned(
    root: Path,
    rel_path: str,
    candidate_pool: Dict[str, List[str]],
    all_ids: List[str],
    floor=None,
):
    obj = load_pkl(root, rel_path)
    if floor is None:
        vals = [v for q in obj.values() for v in q.values()]
        if not vals:
            raise RuntimeError(f"No score values in {rel_path}")
        floor = min(vals)
    return {
        q: {d: obj.get(q, {}).get(d, floor) for d in candidate_pool[q]}
        for q in all_ids
    }


def metrics(
    pred: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    all_ids: List[str],
    blocks: Dict[str, List[str]],
) -> Dict[str, Any]:
    per_r = {
        q: len(set(pred[q]) & gold[q]) / len(gold[q])
        for q in all_ids
    }
    per_p = {
        q: len(set(pred[q]) & gold[q]) / 5.0
        for q in all_ids
    }
    singles = [per_r[q] for q in all_ids if len(gold[q]) == 1]
    multis = [per_r[q] for q in all_ids if len(gold[q]) > 1]
    return {
        "recall_at_5": float(np.mean(list(per_r.values()))),
        "precision_at_5": float(np.mean(list(per_p.values()))),
        "single_gold_recall_at_5": float(np.mean(singles)) if singles else None,
        "multi_gold_recall_at_5": float(np.mean(multis)) if multis else None,
        "block_recalls": {
            str(b).upper(): float(np.mean([per_r[q] for q in ids]))
            for b, ids in blocks.items()
        },
        "per_query_recall": per_r,
    }


# ---------------------------------------------------------------------------
# Exact D1 reconstruction + CAL raw retrieval branches
# ---------------------------------------------------------------------------

def reconstruct_d1(root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import DocumentStore
    from benchmark_jina_reranker_holdouts import load_cache
    from tune_burst_phrases import multi_rrf
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_robust import TAGS
    from tune_expanded_fusion_selection import ltr_features

    ctx_dir = (
        root
        / "DSC2026-LegalIR-main"
        / "v4_run"
        / "public_test_dataset"
        / "selected-contexts"
    )
    docs = DocumentStore(sorted(ctx_dir.glob("context_*.json")))

    (
        queries,
        blocks_raw,
        all_ids,
        extended,
        local_views,
        base_scores,
    ) = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )

    blocks = {str(k).upper(): list(v) for k, v in blocks_raw.items()}
    if set(blocks) != {"A", "B", "C", "D"}:
        raise RuntimeError(
            f"Unexpected CAL block keys: raw={list(blocks_raw)}, normalized={list(blocks)}"
        )
    gold = {q: set(queries[q][1]) for q in all_ids}

    vnlegal_cv = load_pkl(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned(
        root, "results/crossenc_fullpool/cv_scores.pkl", extended, all_ids, -11.5
    )
    extra_cv = {
        name: load_aligned(root, rel, extended, all_ids)
        for name, rel in EXTRA_CV_PATHS.items()
    }
    d1_channels = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(root, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    preds_top5: Dict[str, List[str]] = {}
    full_rankings: Dict[str, List[str]] = {}
    feature_dim: Optional[int] = None

    for held in sorted(blocks):
        train_ids = sum((blocks[b] for b in sorted(blocks) if b != held), [])
        eval_ids = train_ids + blocks[held]

        rows, groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, d1_channels
        )
        for q in rows:
            rows[q] = np.concatenate(
                [rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        feature_dim = int(rows[all_ids[0]].shape[1])
        X_train = np.vstack([rows[q] for q in train_ids])
        y_train = np.concatenate(
            [[d in gold[q] for d in groups[q]] for q in train_ids]
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
            score = model.decision_function(scaler.transform(rows[q]))
            order_idx = np.argsort(-score)
            ranked = [groups[q][i] for i in order_idx]
            full_rankings[q] = ranked
            preds_top5[q] = ranked[:5]

    d1_metrics = metrics(preds_top5, gold, all_ids, blocks)

    # Recover the exact four raw retrieval branch orders for portable
    # non-neural features. Keep original block keys for TAGS lookup.
    branch_orders: Dict[str, Dict[str, List[str]]] = {
        f"branch_{i}": {} for i in range(4)
    }
    for raw_block_name, ids in blocks_raw.items():
        tag = TAGS[raw_block_name]
        cache = load_cache(root, tag)
        for q in ids:
            row = cache[q]
            if len(row) != 4:
                raise RuntimeError(f"Expected 4 raw branches for CAL qid={q}, got {len(row)}")
            for i, source in enumerate(row):
                branch_orders[f"branch_{i}"][q] = [str(d) for d, *_ in source]

    rrf_orders = {}
    for q in all_ids:
        sources = [branch_orders[f"branch_{i}"][q] for i in range(4)]
        rrf_orders[q] = list(multi_rrf(sources, RRF_WEIGHTS, RRF_K))

    return {
        "queries": queries,
        "blocks": blocks,
        "all_ids": all_ids,
        "extended": extended,
        "gold": gold,
        "preds_top5": preds_top5,
        "full_rankings": full_rankings,
        "feature_dim": feature_dim,
        "metrics": d1_metrics,
        "branch_orders": branch_orders,
        "rrf_orders": rrf_orders,
    }


def assert_d1_parity(d1: Dict[str, Any]) -> None:
    errors = []
    m = d1["metrics"]
    if d1["feature_dim"] != EXPECTED_D1_DIM:
        errors.append(f"feature_dim={d1['feature_dim']} expected={EXPECTED_D1_DIM}")
    if abs(m["recall_at_5"] - EXPECTED_D1_R5) > 1e-12:
        errors.append(f"Recall@5={m['recall_at_5']} expected={EXPECTED_D1_R5}")
    for b, exp in EXPECTED_BLOCKS.items():
        got = m["block_recalls"][b]
        if abs(got - exp) > 1e-12:
            errors.append(f"Block {b}={got} expected={exp}")
    if errors:
        raise RuntimeError("BLOCKED_D1_PARITY:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Raw sparse source loading (FULL source, no D1-pool filtering)
# ---------------------------------------------------------------------------

def load_sparse_raw(
    db_path: Path,
    qids: Sequence[str],
    depth: int,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing sparse source DB: {db_path}")

    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    orders = {"bm25": {}, "trigram": {}}
    missing = {"bm25": [], "trigram": []}
    row_lengths = {"bm25": [], "trigram": []}

    try:
        cur = con.cursor()
        for q in qids:
            for channel in ("bm25", "trigram"):
                row = cur.execute(
                    "SELECT payload FROM sources WHERE q=? AND source=?",
                    (q, channel),
                ).fetchone()
                if row is None:
                    orders[channel][q] = []
                    missing[channel].append(q)
                    row_lengths[channel].append(0)
                    continue
                values = json.loads(row[0])
                # Reconstruct native order from the explicit rank field rather
                # than trusting JSON payload order.
                ranked_values = sorted(
                    values,
                    key=lambda r: (int(r.get("rank", RANK_MISSING)), str(r["doc_id"])),
                )
                docs = [str(r["doc_id"]) for r in ranked_values]
                orders[channel][q] = docs[:depth]
                row_lengths[channel].append(len(docs))
    finally:
        con.close()

    n = max(len(qids), 1)
    audit = {
        "queries": len(qids),
        "depth_requested": depth,
        "coverage": {
            ch: 1.0 - len(missing[ch]) / n for ch in ("bm25", "trigram")
        },
        "missing_counts": {ch: len(missing[ch]) for ch in ("bm25", "trigram")},
        "missing_sample": {ch: missing[ch][:20] for ch in ("bm25", "trigram")},
        "payload_length_min": {
            ch: min(row_lengths[ch]) if row_lengths[ch] else 0
            for ch in ("bm25", "trigram")
        },
        "payload_length_median": {
            ch: float(np.median(row_lengths[ch])) if row_lengths[ch] else 0.0
            for ch in ("bm25", "trigram")
        },
    }
    return orders, audit


# ---------------------------------------------------------------------------
# NON-CAL retrieval dataset
# ---------------------------------------------------------------------------

def load_noncal_dataset(root: Path, cal_qids: Set[str]) -> Dict[str, Any]:
    sys.path.insert(0, str(root))
    from tune_burst_phrases import multi_rrf

    train_path = (
        root
        / "DSC2026-LegalIR-main"
        / "v4_run"
        / "public_test_dataset"
        / "train.json"
    )
    raw = json.loads(train_path.read_text(encoding="utf-8"))
    queries = {
        str(q): (
            str(row.get("question", "")),
            {str(d) for d in row.get("answer", [])},
        )
        for q, row in raw.items()
        if row.get("answer")
    }

    cache_path = root / "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl"
    saved = pickle.loads(cache_path.read_bytes())
    cache_raw = saved["cache"]
    cache = {str(k): v for k, v in cache_raw.items()}
    saved_qids = [str(q) for q in saved["qids"]]

    eligible = [
        q for q in saved_qids
        if q in queries and q not in cal_qids and q in cache
    ]

    branch_orders: Dict[str, Dict[str, List[str]]] = {
        f"branch_{i}": {} for i in range(4)
    }
    rrf_orders: Dict[str, List[str]] = {}
    malformed = []

    for q in eligible:
        row = cache[q]
        if len(row) != 4:
            malformed.append(q)
            continue
        for i, source in enumerate(row):
            branch_orders[f"branch_{i}"][q] = [str(item[0]) for item in source]
        rrf_orders[q] = list(
            multi_rrf(
                [branch_orders[f"branch_{i}"][q] for i in range(4)],
                RRF_WEIGHTS,
                RRF_K,
            )
        )

    eligible = [q for q in eligible if q in rrf_orders and len(rrf_orders[q]) >= 5]

    return {
        "queries": queries,
        "qids": eligible,
        "branch_orders": branch_orders,
        "rrf_orders": rrf_orders,
        "cache_path": cache_path,
        "malformed_qids": malformed,
    }


# ---------------------------------------------------------------------------
# Portable pairwise feature engineering
# ---------------------------------------------------------------------------

SOURCE_NAMES = [
    "branch_0",
    "branch_1",
    "branch_2",
    "branch_3",
    "rrf",
    "bm25",
    "trigram",
]


def canonical_lookup_doc(doc: str, rank_map: Dict[str, int]) -> str:
    if doc in rank_map:
        return doc
    twin = DUPLICATE_MAP.get(doc)
    if twin is not None and twin in rank_map:
        return twin
    return doc


def rank_maps_for_query(
    q: str,
    branch_orders: Dict[str, Dict[str, List[str]]],
    rrf_orders: Dict[str, List[str]],
    sparse_orders: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, int]]:
    maps = {}
    for i in range(4):
        name = f"branch_{i}"
        maps[name] = {
            d: idx + 1
            for idx, d in enumerate(branch_orders[name].get(q, []))
        }
    maps["rrf"] = {d: idx + 1 for idx, d in enumerate(rrf_orders.get(q, []))}
    maps["bm25"] = {
        d: idx + 1 for idx, d in enumerate(sparse_orders["bm25"].get(q, []))
    }
    maps["trigram"] = {
        d: idx + 1 for idx, d in enumerate(sparse_orders["trigram"].get(q, []))
    }
    return maps


def doc_rank(doc: str, rank_map: Dict[str, int]) -> int:
    key = canonical_lookup_doc(doc, rank_map)
    return int(rank_map.get(key, RANK_MISSING))


def rr(rank: int) -> float:
    return 1.0 / (10.0 + min(rank, RANK_MISSING))


def nrank(rank: int) -> float:
    return min(rank, 100.0) / 100.0


def pair_features(
    challenger: str,
    defender: str,
    maps: Dict[str, Dict[str, int]],
) -> Tuple[np.ndarray, List[str]]:
    vals: List[float] = []
    names: List[str] = []

    cranks = {s: doc_rank(challenger, maps[s]) for s in SOURCE_NAMES}
    dranks = {s: doc_rank(defender, maps[s]) for s in SOURCE_NAMES}

    # Per-source absolute and delta features.
    for s in SOURCE_NAMES:
        cr, dr = cranks[s], dranks[s]
        crr, drr = rr(cr), rr(dr)
        for name, value in (
            (f"{s}_cand_rr", crr),
            (f"{s}_def_rr", drr),
            (f"{s}_delta_rr", crr - drr),
            (f"{s}_cand_nrank", nrank(cr)),
            (f"{s}_def_nrank", nrank(dr)),
            (f"{s}_delta_nrank", nrank(cr) - nrank(dr)),
        ):
            names.append(name)
            vals.append(float(value))

    # Consensus at fixed, non-tuned cutoffs.
    for cutoff in (5, 10, 20):
        ccount = sum(cranks[s] <= cutoff for s in SOURCE_NAMES)
        dcount = sum(dranks[s] <= cutoff for s in SOURCE_NAMES)
        names.extend([
            f"cand_top{cutoff}_count",
            f"def_top{cutoff}_count",
            f"delta_top{cutoff}_count",
        ])
        vals.extend([float(ccount), float(dcount), float(ccount - dcount)])

    # How many independent sources prefer challenger over defender.
    prefer = sum(cranks[s] < dranks[s] for s in SOURCE_NAMES)
    names.append("sources_preferring_challenger")
    vals.append(float(prefer))

    # Sparse-specific discrete agreement features.
    for cutoff in (5, 10, 20):
        c_b = cranks["bm25"] <= cutoff
        c_t = cranks["trigram"] <= cutoff
        d_b = dranks["bm25"] <= cutoff
        d_t = dranks["trigram"] <= cutoff
        features = (
            (f"cand_sparse_mutual_top{cutoff}", c_b and c_t),
            (f"def_sparse_mutual_top{cutoff}", d_b and d_t),
            (f"cand_sparse_any_top{cutoff}", c_b or c_t),
            (f"def_sparse_any_top{cutoff}", d_b or d_t),
        )
        for name, value in features:
            names.append(name)
            vals.append(float(value))

    # Aggregate reciprocal-rank geometry.
    c_rrs = np.asarray([rr(cranks[s]) for s in SOURCE_NAMES], dtype=np.float64)
    d_rrs = np.asarray([rr(dranks[s]) for s in SOURCE_NAMES], dtype=np.float64)
    for label, arr in (("cand", c_rrs), ("def", d_rrs)):
        for stat_name, value in (
            ("mean_rr", np.mean(arr)),
            ("max_rr", np.max(arr)),
            ("std_rr", np.std(arr)),
        ):
            names.append(f"{label}_{stat_name}")
            vals.append(float(value))
    delta = c_rrs - d_rrs
    for stat_name, value in (
        ("mean_rr", np.mean(delta)),
        ("max_rr", np.max(delta)),
        ("min_rr", np.min(delta)),
        ("std_rr", np.std(delta)),
    ):
        names.append(f"delta_{stat_name}")
        vals.append(float(value))

    return np.asarray(vals, dtype=np.float64), names


def sparse_union_candidates(
    q: str,
    sparse_orders: Dict[str, Dict[str, List[str]]],
    incumbent_top5: Sequence[str],
    depth: int = SPARSE_DEPTH,
) -> List[str]:
    incumbent = set(incumbent_top5)
    bm = sparse_orders["bm25"].get(q, [])[:depth]
    tri = sparse_orders["trigram"].get(q, [])[:depth]
    # Deterministic candidate order: best min sparse rank, then rank sum, then id.
    bm_rank = {d: i + 1 for i, d in enumerate(bm)}
    tri_rank = {d: i + 1 for i, d in enumerate(tri)}
    docs = (set(bm) | set(tri)) - incumbent
    return sorted(
        docs,
        key=lambda d: (
            min(bm_rank.get(d, RANK_MISSING), tri_rank.get(d, RANK_MISSING)),
            bm_rank.get(d, RANK_MISSING) + tri_rank.get(d, RANK_MISSING),
            d,
        ),
    )


def build_pair_dataset(
    qids: Sequence[str],
    gold_map: Dict[str, Set[str]],
    incumbent_orders: Dict[str, List[str]],
    branch_orders: Dict[str, Dict[str, List[str]]],
    rrf_orders: Dict[str, List[str]],
    sparse_orders: Dict[str, Dict[str, List[str]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], Dict[str, Any]]:
    rows: Dict[str, List[Dict[str, Any]]] = {}
    feature_names: Optional[List[str]] = None
    positive_pairs = harmful_pairs = neutral_pairs = 0
    queries_with_candidates = 0

    for q in qids:
        top5 = incumbent_orders[q][:5]
        if len(top5) < 5:
            continue
        defender = top5[4]
        candidates = sparse_union_candidates(q, sparse_orders, top5)
        if candidates:
            queries_with_candidates += 1

        maps = rank_maps_for_query(q, branch_orders, rrf_orders, sparse_orders)
        gold = gold_map[q]
        base_recall = len(set(top5) & gold) / len(gold)
        qrows = []

        for cand in candidates:
            x, names = pair_features(cand, defender, maps)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("Feature-name drift detected")

            new_top5 = list(top5[:4]) + [cand]
            new_recall = len(set(new_top5) & gold) / len(gold)
            utility = float(new_recall - base_recall)
            label = 1 if utility > 0 else 0
            if utility > 0:
                positive_pairs += 1
            elif utility < 0:
                harmful_pairs += 1
            else:
                neutral_pairs += 1

            qrows.append({
                "qid": q,
                "challenger": cand,
                "defender": defender,
                "x": x,
                "label": label,
                "utility": utility,
            })
        rows[q] = qrows

    audit = {
        "queries_requested": len(qids),
        "queries_with_sparse_candidates": queries_with_candidates,
        "positive_pairs": positive_pairs,
        "harmful_pairs": harmful_pairs,
        "neutral_pairs": neutral_pairs,
        "feature_dim": len(feature_names or []),
    }
    return rows, (feature_names or []), audit


# ---------------------------------------------------------------------------
# Model + nested threshold calibration
# ---------------------------------------------------------------------------

def fit_model(
    pair_rows: Dict[str, List[Dict[str, Any]]],
    train_qids: Sequence[str],
) -> Tuple[StandardScaler, LogisticRegression]:
    Xs = []
    ys = []
    ws = []
    for q in train_qids:
        rows = pair_rows.get(q, [])
        if not rows:
            continue
        q_weight = 1.0 / len(rows)
        for row in rows:
            Xs.append(row["x"])
            ys.append(row["label"])
            ws.append(q_weight)

    if not Xs:
        raise RuntimeError("No pair rows available for model fit")
    X = np.vstack(Xs)
    y = np.asarray(ys, dtype=np.int8)
    w = np.asarray(ws, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Training labels have only one class: counts={np.bincount(y)}"
        )

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    model.fit(scaler.transform(X), y, sample_weight=w)
    return scaler, model


def best_candidate_predictions(
    pair_rows: Dict[str, List[Dict[str, Any]]],
    qids: Sequence[str],
    scaler: StandardScaler,
    model: LogisticRegression,
) -> Dict[str, Dict[str, Any]]:
    out = {}
    for q in qids:
        rows = pair_rows.get(q, [])
        if not rows:
            continue
        X = np.vstack([r["x"] for r in rows])
        scores = model.decision_function(scaler.transform(X))
        # deterministic tie break by challenger doc id
        indices = sorted(
            range(len(rows)),
            key=lambda i: (-float(scores[i]), str(rows[i]["challenger"])),
        )
        i = indices[0]
        row = rows[i]
        out[q] = {
            "qid": q,
            "challenger": row["challenger"],
            "defender": row["defender"],
            "score": float(scores[i]),
            "utility": float(row["utility"]),
            "beneficial_label": int(row["label"]),
        }
    return out


def summarize_threshold(
    best: Dict[str, Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    selected = [r for r in best.values() if r["score"] >= threshold]
    beneficial = sum(r["utility"] > 0 for r in selected)
    harmful = sum(r["utility"] < 0 for r in selected)
    neutral = sum(r["utility"] == 0 for r in selected)
    actions = len(selected)
    precision = beneficial / actions if actions else 0.0
    macro_delta_sum = float(sum(r["utility"] for r in selected))
    return {
        "threshold": float(threshold),
        "actions": actions,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "action_precision": precision,
        "macro_recall_delta_sum": macro_delta_sum,
    }


def choose_threshold(
    best: Dict[str, Dict[str, Any]],
    n_queries: int,
) -> Tuple[float, Dict[str, Any]]:
    if not best:
        return float("inf"), summarize_threshold(best, float("inf"))

    max_actions = max(5, int(math.ceil(CALIBRATION_MAX_ACTION_RATE * n_queries)))
    unique_scores = sorted({float(r["score"]) for r in best.values()}, reverse=True)

    feasible = []
    for t in unique_scores:
        s = summarize_threshold(best, t)
        if (
            s["actions"] <= max_actions
            and s["harmful"] == 0
            and s["beneficial"] >= CALIBRATION_MIN_BENEFICIAL
            and s["action_precision"] >= CALIBRATION_MIN_PRECISION
        ):
            feasible.append(s)

    if not feasible:
        return float("inf"), {
            **summarize_threshold(best, float("inf")),
            "max_actions_allowed": max_actions,
            "status": "NO_FEASIBLE_THRESHOLD",
        }

    # Primary: rescue as many beneficial queries as possible.
    # Secondary: higher action precision, fewer total actions, higher threshold.
    feasible.sort(
        key=lambda s: (
            s["beneficial"],
            s["action_precision"],
            -s["actions"],
            s["threshold"],
        ),
        reverse=True,
    )
    chosen = dict(feasible[0])
    chosen["max_actions_allowed"] = max_actions
    chosen["status"] = "FEASIBLE"
    return float(chosen["threshold"]), chosen


def make_folds(qids: Sequence[str]) -> Dict[int, List[str]]:
    folds = {i: [] for i in range(5)}
    for q in qids:
        folds[stable_fold(q)].append(q)
    for i in folds:
        folds[i].sort()
    return folds


def nested_noncal_validation(
    pair_rows: Dict[str, List[Dict[str, Any]]],
    qids: Sequence[str],
    incumbent_orders: Dict[str, List[str]],
    gold_map: Dict[str, Set[str]],
) -> Dict[str, Any]:
    folds = make_folds(qids)
    all_outer_actions = {}
    fold_reports = {}

    for outer in range(5):
        outer_qids = folds[outer]
        train_folds = [i for i in range(5) if i != outer]
        train_qids = [q for i in train_folds for q in folds[i]]

        # Inner OOF predictions on the outer-training population.
        inner_best = {}
        for inner in train_folds:
            inner_test = folds[inner]
            inner_train = [
                q for i in train_folds if i != inner for q in folds[i]
            ]
            scaler_i, model_i = fit_model(pair_rows, inner_train)
            inner_best.update(
                best_candidate_predictions(
                    pair_rows, inner_test, scaler_i, model_i
                )
            )

        threshold, calibration = choose_threshold(inner_best, len(train_qids))

        # Final outer model fit on all outer-training folds.
        scaler, model = fit_model(pair_rows, train_qids)
        outer_best = best_candidate_predictions(
            pair_rows, outer_qids, scaler, model
        )

        selected = {
            q: r for q, r in outer_best.items()
            if r["score"] >= threshold
        }

        beneficial = sum(r["utility"] > 0 for r in selected.values())
        harmful = sum(r["utility"] < 0 for r in selected.values())
        neutral = sum(r["utility"] == 0 for r in selected.values())
        actions = len(selected)
        delta_sum = float(sum(r["utility"] for r in selected.values()))
        action_precision = beneficial / actions if actions else 0.0

        for q, r in selected.items():
            all_outer_actions[q] = {**r, "outer_fold": outer}

        fold_reports[f"fold_{outer}"] = {
            "queries": len(outer_qids),
            "train_queries": len(train_qids),
            "threshold": threshold,
            "calibration": calibration,
            "actions": actions,
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
            "action_precision": action_precision,
            "macro_recall_delta_sum": delta_sum,
            "macro_recall_delta": delta_sum / max(len(outer_qids), 1),
        }

    total_actions = len(all_outer_actions)
    beneficial = sum(r["utility"] > 0 for r in all_outer_actions.values())
    harmful = sum(r["utility"] < 0 for r in all_outer_actions.values())
    neutral = sum(r["utility"] == 0 for r in all_outer_actions.values())
    delta_sum = float(sum(r["utility"] for r in all_outer_actions.values()))
    action_precision = beneficial / total_actions if total_actions else 0.0

    folds_nonnegative = sum(
        r["macro_recall_delta"] >= -1e-12
        for r in fold_reports.values()
    )

    gate = {
        "beneficial_ge_min": beneficial >= NONCAL_GATE_MIN_BENEFICIAL,
        "harmful_le_max": harmful <= NONCAL_GATE_MAX_HARMFUL,
        "action_precision_ge_min": action_precision >= NONCAL_GATE_MIN_ACTION_PRECISION,
        "macro_delta_positive": delta_sum > 0,
        "at_least_4_of_5_folds_nonnegative": folds_nonnegative >= 4,
    }

    return {
        "folds": fold_reports,
        "aggregate": {
            "queries": len(qids),
            "actions": total_actions,
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
            "action_precision": action_precision,
            "macro_recall_delta_sum": delta_sum,
            "macro_recall_delta": delta_sum / max(len(qids), 1),
            "folds_nonnegative": folds_nonnegative,
        },
        "gate": gate,
        "pass": all(gate.values()),
        "actions": all_outer_actions,
    }


def full_oof_threshold(
    pair_rows: Dict[str, List[Dict[str, Any]]],
    qids: Sequence[str],
) -> Dict[str, Any]:
    folds = make_folds(qids)
    oof_best = {}
    for held in range(5):
        test_qids = folds[held]
        train_qids = [q for i in range(5) if i != held for q in folds[i]]
        scaler, model = fit_model(pair_rows, train_qids)
        oof_best.update(
            best_candidate_predictions(pair_rows, test_qids, scaler, model)
        )
    threshold, calibration = choose_threshold(oof_best, len(qids))
    return {
        "threshold": threshold,
        "calibration": calibration,
        "oof_best": oof_best,
    }


# ---------------------------------------------------------------------------
# CAL application using frozen NON-CAL model + threshold
# ---------------------------------------------------------------------------

def build_cal_pair_rows_label_free(
    d1: Dict[str, Any],
    sparse_cal: Dict[str, Dict[str, List[str]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], Dict[str, Any]]:
    rows: Dict[str, List[Dict[str, Any]]] = {}
    feature_names: Optional[List[str]] = None
    queries_with_candidates = 0
    total_candidates = 0
    outside_pool_candidates = 0

    for q in d1["all_ids"]:
        top5 = d1["preds_top5"][q]
        defender = top5[4]
        candidates = sparse_union_candidates(q, sparse_cal, top5)
        if candidates:
            queries_with_candidates += 1
        total_candidates += len(candidates)
        pool = set(d1["extended"][q])
        outside_pool_candidates += sum(c not in pool for c in candidates)

        maps = rank_maps_for_query(
            q, d1["branch_orders"], d1["rrf_orders"], sparse_cal
        )
        qrows = []
        for cand in candidates:
            x, names = pair_features(cand, defender, maps)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("CAL feature-name drift detected")
            qrows.append({
                "qid": q,
                "challenger": cand,
                "defender": defender,
                "x": x,
            })
        rows[q] = qrows

    audit = {
        "queries": len(d1["all_ids"]),
        "queries_with_candidates": queries_with_candidates,
        "total_candidates": total_candidates,
        "outside_d1_pool_candidates": outside_pool_candidates,
        "feature_dim": len(feature_names or []),
    }
    return rows, (feature_names or []), audit


def score_cal_pairs(
    cal_rows: Dict[str, List[Dict[str, Any]]],
    qids: Sequence[str],
    scaler: StandardScaler,
    model: LogisticRegression,
) -> Dict[str, Dict[str, Any]]:
    out = {}
    for q in qids:
        rows = cal_rows.get(q, [])
        if not rows:
            continue
        X = np.vstack([r["x"] for r in rows])
        scores = model.decision_function(scaler.transform(X))
        indices = sorted(
            range(len(rows)),
            key=lambda i: (-float(scores[i]), str(rows[i]["challenger"])),
        )
        i = indices[0]
        row = rows[i]
        out[q] = {
            "qid": q,
            "challenger": row["challenger"],
            "defender": row["defender"],
            "score": float(scores[i]),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    ap.add_argument(
        "--sources-db",
        type=Path,
        default=Path(
            "D:/Study/DSC2026/LegalIR/cache/"
            "exp112_task_adaptive_retrieval/sources.sqlite"
        ),
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sources_db = args.sources_db.resolve()

    if not (root / "tune_corpus_cap32_fusion.py").exists():
        raise RuntimeError(
            f"{root} does not look like the sota repo root. "
            "Pass --repo-root /d/Study/DSC2026/sota in Git Bash."
        )

    out = root / "results/manual/huy_d1_noncal_sparse_admission_v1"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[1/8] Reconstructing exact D1 and portable CAL retrieval features...", flush=True)
    d1 = reconstruct_d1(root)
    assert_d1_parity(d1)
    print(
        f"  PASS D1: R@5={d1['metrics']['recall_at_5']:.12f}, "
        f"dim={d1['feature_dim']}"
    )

    cal_qids = set(d1["all_ids"])

    print("[2/8] Loading NON-CAL retrieval dataset...", flush=True)
    noncal = load_noncal_dataset(root, cal_qids)
    print(
        f"  eligible non-CAL queries: {len(noncal['qids'])}; "
        f"malformed={len(noncal['malformed_qids'])}"
    )
    if len(noncal["qids"]) < 800:
        raise RuntimeError(
            f"BLOCKED_NONCAL_POPULATION: only {len(noncal['qids'])} usable queries"
        )

    print("[3/8] Loading RAW full-source BM25/trigram rankings...", flush=True)
    sparse_noncal, sparse_noncal_audit = load_sparse_raw(
        sources_db, noncal["qids"], depth=SPARSE_DEPTH
    )
    sparse_cal, sparse_cal_audit = load_sparse_raw(
        sources_db, d1["all_ids"], depth=SPARSE_DEPTH
    )
    print(
        "  sparse coverage nonCAL: "
        f"BM25={sparse_noncal_audit['coverage']['bm25']:.3f}, "
        f"TRI={sparse_noncal_audit['coverage']['trigram']:.3f}"
    )
    print(
        "  sparse coverage CAL   : "
        f"BM25={sparse_cal_audit['coverage']['bm25']:.3f}, "
        f"TRI={sparse_cal_audit['coverage']['trigram']:.3f}"
    )

    min_noncal_cov = min(sparse_noncal_audit["coverage"].values())
    min_cal_cov = min(sparse_cal_audit["coverage"].values())
    sparse_depth_ok = (
        sparse_noncal_audit["payload_length_median"]["bm25"] >= SPARSE_DEPTH
        and sparse_noncal_audit["payload_length_median"]["trigram"] >= SPARSE_DEPTH
        and sparse_cal_audit["payload_length_median"]["bm25"] >= SPARSE_DEPTH
        and sparse_cal_audit["payload_length_median"]["trigram"] >= SPARSE_DEPTH
    )
    if min_noncal_cov < 0.90 or min_cal_cov < 0.98 or not sparse_depth_ok:
        report = {
            "status": "BLOCKED_SPARSE_SOURCE_COVERAGE",
            "noncal": sparse_noncal_audit,
            "cal": sparse_cal_audit,
            "median_depth_ok": sparse_depth_ok,
        }
        json_dump(out / "SPARSE_SOURCE_COVERAGE.json", report)
        raise RuntimeError(
            "BLOCKED_SPARSE_SOURCE_COVERAGE: raw source rows are insufficient"
        )

    print("[4/8] Building NON-CAL pairwise swap-utility dataset...", flush=True)
    noncal_gold = {q: noncal["queries"][q][1] for q in noncal["qids"]}
    noncal_rows, feature_names, pair_audit = build_pair_dataset(
        noncal["qids"],
        noncal_gold,
        noncal["rrf_orders"],
        noncal["branch_orders"],
        noncal["rrf_orders"],
        sparse_noncal,
    )
    print(
        f"  features={pair_audit['feature_dim']} "
        f"positive_pairs={pair_audit['positive_pairs']} "
        f"harmful_pairs={pair_audit['harmful_pairs']} "
        f"neutral_pairs={pair_audit['neutral_pairs']}"
    )
    if pair_audit["positive_pairs"] < NONCAL_GATE_MIN_POSITIVE_PAIRS:
        raise RuntimeError(
            "BLOCKED_NONCAL_POSITIVE_PAIRS: "
            f"only {pair_audit['positive_pairs']} beneficial pairs"
        )

    source_manifest = {
        "status": "PASS",
        "git": git_info(root),
        "sources_db": str(sources_db),
        "sources_db_sha256": sha256_file(sources_db),
        "noncal_retrieval_cache": str(noncal["cache_path"]),
        "noncal_retrieval_cache_sha256": sha256_file(noncal["cache_path"]),
        "noncal_queries": len(noncal["qids"]),
        "cal_queries": len(d1["all_ids"]),
        "sparse_noncal_audit": sparse_noncal_audit,
        "sparse_cal_audit": sparse_cal_audit,
        "pair_dataset_audit": pair_audit,
        "feature_names": feature_names,
        "model_contract": {
            "model": "StandardScaler + LogisticRegression",
            "C": 0.15,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 3000,
            "random_state": SEED,
            "query_balanced_sample_weights": True,
        },
        "threshold_contract": {
            "nested_noncal_only": True,
            "max_action_rate": CALIBRATION_MAX_ACTION_RATE,
            "min_beneficial": CALIBRATION_MIN_BENEFICIAL,
            "min_action_precision": CALIBRATION_MIN_PRECISION,
            "harmful_required": 0,
        },
    }
    json_dump(out / "SOURCE_AND_FEATURE_MANIFEST.json", source_manifest)

    print("[5/8] Nested NON-CAL validation + threshold calibration...", flush=True)
    nested = nested_noncal_validation(
        noncal_rows,
        noncal["qids"],
        noncal["rrf_orders"],
        noncal_gold,
    )
    json_dump(
        out / "NONCAL_NESTED_VALIDATION.json",
        {k: v for k, v in nested.items() if k != "actions"},
    )
    agg = nested["aggregate"]
    print(
        f"  nonCAL actions={agg['actions']} "
        f"beneficial={agg['beneficial']} harmful={agg['harmful']} "
        f"neutral={agg['neutral']} precision={agg['action_precision']:.3f} "
        f"delta={agg['macro_recall_delta']:+.6f}"
    )
    print(f"  nonCAL gate: {'PASS' if nested['pass'] else 'FAIL'}")

    if not nested["pass"]:
        json_dump(
            out / "FINAL_REPORT.json",
            {
                "status": "KILL_NONCAL_GENERALIZATION_GATE",
                "d1_parity": "PASS",
                "pair_audit": pair_audit,
                "noncal_nested": {
                    k: v for k, v in nested.items() if k != "actions"
                },
                "runtime_seconds": time.perf_counter() - started,
            },
        )
        print("[STOP] KILL_NONCAL_GENERALIZATION_GATE")
        return

    print("[6/8] Fit frozen final NON-CAL model and threshold...", flush=True)
    final_calib = full_oof_threshold(noncal_rows, noncal["qids"])
    threshold = float(final_calib["threshold"])
    print(
        f"  frozen threshold={threshold:.6f} "
        f"calib={final_calib['calibration']}"
    )
    final_scaler, final_model = fit_model(noncal_rows, noncal["qids"])

    print("[7/8] Building and sealing LABEL-FREE CAL actions...", flush=True)
    cal_rows, cal_feature_names, cal_feature_audit = build_cal_pair_rows_label_free(
        d1, sparse_cal
    )
    if cal_feature_names != feature_names:
        raise RuntimeError("BLOCKED_FEATURE_PARITY: NON-CAL vs CAL feature names differ")

    cal_best = score_cal_pairs(
        cal_rows, d1["all_ids"], final_scaler, final_model
    )
    selected = {
        q: r for q, r in cal_best.items()
        if r["score"] >= threshold
    }

    actions = {}
    outside_pool_actions = 0
    for q in d1["all_ids"]:
        base = list(d1["preds_top5"][q])
        row = selected.get(q)
        if row is None:
            actions[q] = {
                "qid": q,
                "action": False,
                "d1_top5": base,
                "candidate_top5": base,
            }
            continue
        challenger = row["challenger"]
        is_outside = challenger not in set(d1["extended"][q])
        outside_pool_actions += int(is_outside)
        actions[q] = {
            "qid": q,
            "action": True,
            "score": row["score"],
            "threshold": threshold,
            "defender": row["defender"],
            "challenger": challenger,
            "challenger_outside_d1_pool": is_outside,
            "d1_top5": base,
            "candidate_top5": base[:4] + [challenger],
        }

    action_payload = {
        "schema": "manual.noncal_sparse_admission.cal_actions.label_free.v1",
        "training_population": "NON-CAL only",
        "threshold_source": "nested/OOF NON-CAL only",
        "threshold": threshold,
        "feature_dim": len(feature_names),
        "cal_feature_audit": cal_feature_audit,
        "actions": len(selected),
        "outside_pool_actions": outside_pool_actions,
        "rows": actions,
    }
    action_path = out / "CAL_ACTIONS_LABEL_FREE.json"
    json_dump(action_path, action_payload)
    action_sha = sha256_file(action_path)
    print(
        f"  sealed CAL actions={len(selected)} "
        f"outside_pool={outside_pool_actions} sha={action_sha[:12]}..."
    )

    if len(selected) > CAL_MAX_ACTIONS:
        json_dump(
            out / "FINAL_REPORT.json",
            {
                "status": "BLOCKED_CAL_ACTION_SURFACE",
                "actions": len(selected),
                "max_allowed": CAL_MAX_ACTIONS,
                "action_seal_sha256": action_sha,
                "noncal_nested": {
                    k: v for k, v in nested.items() if k != "actions"
                },
                "runtime_seconds": time.perf_counter() - started,
            },
        )
        print(
            f"[STOP] BLOCKED_CAL_ACTION_SURFACE: "
            f"{len(selected)} > {CAL_MAX_ACTIONS}"
        )
        return

    # ONLY NOW use CAL gold for utility evaluation.
    print("[8/8] Evaluating sealed CAL actions...", flush=True)
    candidate_preds = {}
    for q in d1["all_ids"]:
        candidate_preds[q] = actions[q]["candidate_top5"]

    base_m = d1["metrics"]
    cand_m = metrics(
        candidate_preds, d1["gold"], d1["all_ids"], d1["blocks"]
    )

    beneficial = harmful = neutral = wins = losses = ties = 0
    gold_in = gold_out = 0
    outside_pool_beneficial = 0
    action_details = []

    for q in d1["all_ids"]:
        r0 = base_m["per_query_recall"][q]
        r1 = cand_m["per_query_recall"][q]
        if r1 > r0:
            wins += 1
        elif r1 < r0:
            losses += 1
        else:
            ties += 1

        if actions[q]["action"]:
            if r1 > r0:
                beneficial += 1
                if actions[q]["challenger_outside_d1_pool"]:
                    outside_pool_beneficial += 1
                utility = "BENEFICIAL"
            elif r1 < r0:
                harmful += 1
                utility = "HARMFUL"
            else:
                neutral += 1
                utility = "NEUTRAL"

            bset = set(actions[q]["d1_top5"])
            cset = set(actions[q]["candidate_top5"])
            g = d1["gold"][q]
            gold_in += len((cset - bset) & g)
            gold_out += len((bset - cset) & g)
            action_details.append({
                "qid": q,
                "score": actions[q]["score"],
                "defender": actions[q]["defender"],
                "challenger": actions[q]["challenger"],
                "challenger_outside_d1_pool": actions[q]["challenger_outside_d1_pool"],
                "utility": utility,
                "recall_before": r0,
                "recall_after": r1,
            })

    block_deltas = {
        b: cand_m["block_recalls"][b] - base_m["block_recalls"][b]
        for b in EXPECTED_BLOCKS
    }
    delta_recall = cand_m["recall_at_5"] - base_m["recall_at_5"]
    delta_precision = cand_m["precision_at_5"] - base_m["precision_at_5"]
    delta_single = cand_m["single_gold_recall_at_5"] - base_m["single_gold_recall_at_5"]
    delta_multi = cand_m["multi_gold_recall_at_5"] - base_m["multi_gold_recall_at_5"]

    promotion = {
        "recall_improves": delta_recall > 0,
        "precision_no_decrease": delta_precision >= -1e-12,
        "harmful_eq_0": harmful == 0,
        "wins_gt_losses": wins > losses,
        "no_block_decrease": all(v >= -1e-12 for v in block_deltas.values()),
        "single_no_decrease": delta_single >= -1e-12,
        "multi_no_decrease": delta_multi >= -1e-12,
        "action_surface_le_30": len(selected) <= CAL_MAX_ACTIONS,
    }

    report = {
        "status": (
            "LOCAL_PROMOTE_NONCAL_SPARSE_ADMISSION_V1"
            if all(promotion.values())
            else "KILL_NONCAL_SPARSE_ADMISSION_V1"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "action_seal_sha256": action_sha,
        "frozen_noncal_threshold": threshold,
        "noncal_nested": {
            k: v for k, v in nested.items() if k != "actions"
        },
        "cal_feature_audit": cal_feature_audit,
        "d1": {k: v for k, v in base_m.items() if k != "per_query_recall"},
        "candidate": {k: v for k, v in cand_m.items() if k != "per_query_recall"},
        "delta": {
            "recall_at_5": delta_recall,
            "precision_at_5": delta_precision,
            "single_gold_recall_at_5": delta_single,
            "multi_gold_recall_at_5": delta_multi,
            "blocks": block_deltas,
        },
        "actions": {
            "total": len(selected),
            "outside_pool": outside_pool_actions,
            "outside_pool_beneficial": outside_pool_beneficial,
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
        },
        "promotion_gates": promotion,
        "action_details": action_details,
    }
    json_dump(out / "CAL_EVALUATION_REPORT.json", report)
    json_dump(out / "FINAL_REPORT.json", report)

    print("=" * 78)
    print(
        f"D1       R@5={base_m['recall_at_5']:.10f} "
        f"P@5={base_m['precision_at_5']:.10f}"
    )
    print(
        f"Candidate R@5={cand_m['recall_at_5']:.10f} "
        f"P@5={cand_m['precision_at_5']:.10f}"
    )
    print(
        f"Delta     R={delta_recall:+.10f} "
        f"P={delta_precision:+.10f}"
    )
    print(
        f"Actions   total={len(selected)} outside_pool={outside_pool_actions} "
        f"beneficial={beneficial} harmful={harmful} neutral={neutral}"
    )
    print(f"W/L/T     {wins}/{losses}/{ties}")
    print(
        f"NonCAL    actions={agg['actions']} beneficial={agg['beneficial']} "
        f"harmful={agg['harmful']} precision={agg['action_precision']:.3f}"
    )
    print(f"Threshold {threshold:.6f}")
    print(f"Verdict   {report['status']}")
    print(f"Report    {out / 'FINAL_REPORT.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
