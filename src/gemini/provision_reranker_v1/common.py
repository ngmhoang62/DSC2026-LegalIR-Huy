"""Common utilities for provision_reranker_v1.

Provides constants, git metadata, paths, data loading, metrics calculation,
and baseline verification.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# Ensure stdout uses UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

BASE_SNAPSHOT_DIR = REPO_ROOT / "src/gemini/rabr_v1/baseline_snapshot"
sys.path.insert(0, str(BASE_SNAPSHOT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "LegalIR/scripts"))

import run_huy_5fold_fasttrack as core

RESULTS_DIR = REPO_ROOT / "results/gemini/provision_reranker_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RABR_CACHE_DIR = REPO_ROOT / "results/gemini/rabr_v1/cache"
BASELINE_PRED_FILE = RABR_CACHE_DIR / "BASELINE_PREDICTIONS_AND_SCORES.jsonl"
BASELINE_FEAT_FILE = RABR_CACHE_DIR / "BASELINE_FEATURE_ROWS.npz"

GIT_COMMIT_SHA = "aed1c69dd8d0b72c466ef9b83b123e0ccfa6c31f"

EXPECTED_METRICS = {
    "recall_at_5": 0.9488556715777428,
    "precision_at_5": 0.20314690316120732,
    "single_gold_recall_at_5": 0.9641304347826087,
    "multi_gold_recall_at_5": 0.7703266787658803,
    "per_fold_recall_at_5": {
        "fold_0": 0.9524320457796852,
        "fold_1": 0.9451597520267048,
        "fold_2": 0.9511555873242793,
        "fold_3": 0.9420243204577969,
        "fold_4": 0.9535050071530758,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_baseline_data():
    """Load core fasttrack inputs, folds, golds, pools, duplicate exclusions, and predictions."""
    folds, pools, questions, golds, e5_orders, e5_scores, dup, pred_hashes = core.load_inputs()
    fold_for = {qid: fold for fold, qids in folds.items() for qid in qids}
    
    base_orders: Dict[str, List[str]] = {}
    base_scores: Dict[str, Dict[str, float]] = {}
    with BASELINE_PRED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            base_orders[qid] = [str(x) for x in row["order"]]
            base_scores[qid] = {str(k): float(v) for k, v in row["scores"].items()}
            
    return folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores


def load_cached_feature_rows() -> Dict[str, np.ndarray]:
    """Load cached (51, 44) baseline feature matrices for all queries."""
    with np.load(BASELINE_FEAT_FILE) as z:
        rows = {f.replace("row_", ""): z[f].astype(np.float32) for f in z.files}
    return rows


def transform_rank_features(row: np.ndarray, k: float) -> np.ndarray:
    """Transform reciprocal rank columns for 7 rank views using offset k."""
    new_row = row.copy()
    for i in range(7):
        # column 2*i+1 is raw_rank / 60.0
        raw_rank = np.round(row[:, 2 * i + 1] * 60.0)
        new_row[:, 2 * i] = (1.0 / (k + raw_rank)).astype(np.float32)
    return new_row


def evaluate_predictions(predictions: Dict[str, List[str]], golds: Dict[str, Set[str]], folds: Dict[str, List[str]]) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    values, precisions = [], []
    single_values, multi_values = [], []
    by_fold = {}
    
    for fold, qids in folds.items():
        fold_values = []
        for qid in qids:
            gold = golds[qid]
            top5 = predictions[qid][:5]
            hits = len(set(top5) & gold)
            val = hits / len(gold)
            values.append(val)
            precisions.append(hits / 5.0)
            fold_values.append(val)
            if len(gold) == 1:
                single_values.append(val)
            else:
                multi_values.append(val)
        by_fold[fold] = float(np.mean(fold_values))
        
    return {
        "queries": len(values),
        "recall_at_5": float(np.mean(values)),
        "precision_at_5": float(np.mean(precisions)),
        "single_gold_recall_at_5": float(np.mean(single_values)),
        "multi_gold_recall_at_5": float(np.mean(multi_values)),
        "per_fold_recall_at_5": by_fold,
    }


def compare_endpoints(candidate: Dict[str, List[str]], reference: Dict[str, List[str]], golds: Dict[str, Set[str]], folds: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compare candidate predictions against reference predictions."""
    wins = losses = ties = churn = crossings_in = crossings_out = 0
    fold_delta = {}
    
    for fold, qids in folds.items():
        deltas = []
        for qid in qids:
            gold = golds[qid]
            ref_top5 = set(reference[qid][:5])
            cand_top5 = set(candidate[qid][:5])
            
            ra = len(ref_top5 & gold) / len(gold)
            rb = len(cand_top5 & gold) / len(gold)
            deltas.append(rb - ra)
            
            if rb > ra + 1e-9:
                wins += 1
            elif rb < ra - 1e-9:
                losses += 1
            else:
                ties += 1
                
            churn += (ref_top5 != cand_top5)
            crossings_in += len((cand_top5 & gold) - ref_top5)
            crossings_out += len((ref_top5 & gold) - cand_top5)
            
        fold_delta[fold] = float(np.mean(deltas))
        
    return {
        "delta_recall_at_5": float(np.mean([
            len(set(candidate[q][:5]) & golds[q]) / len(golds[q])
            - len(set(reference[q][:5]) & golds[q]) / len(golds[q])
            for q in golds
        ])),
        "worst_fold_delta": min(fold_delta.values()),
        "per_fold_delta": fold_delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "top5_churn": churn,
        "gold_crossings_into_top5": crossings_in,
        "gold_crossings_out_of_top5": crossings_out,
    }
