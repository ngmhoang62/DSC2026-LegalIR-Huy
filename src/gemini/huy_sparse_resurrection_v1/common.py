"""
Common utilities, runtime proof, and evaluation infrastructure for huy_sparse_resurrection_v1.
Enforces dynamic Git HEAD resolution, SHA256 hashing, and execution tracing.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Root directories
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FASTTRACK_DIR = REPO_ROOT / "src/huy_fasttrack"
BASE_SNAPSHOT_DIR = REPO_ROOT / "src/gemini/rabr_v1/baseline_snapshot"
sys.path.insert(0, str(FASTTRACK_DIR))
sys.path.insert(0, str(BASE_SNAPSHOT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "LegalIR/scripts"))

import run_huy_5fold_fasttrack as core

RESULTS_DIR = REPO_ROOT / "results/gemini/huy_sparse_resurrection_v1"
CACHE_DIR = RESULTS_DIR / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RABR_CACHE_DIR = REPO_ROOT / "results/gemini/rabr_v1/cache"
BASELINE_PRED_FILE = RABR_CACHE_DIR / "BASELINE_PREDICTIONS_AND_SCORES.jsonl"
BASELINE_FEAT_FILE = RABR_CACHE_DIR / "BASELINE_FEATURE_ROWS.npz"

TRACE_FILE = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
PROOF_FILE = RESULTS_DIR / "EXECUTION_PROOF.json"

EXPECTED_METRICS = {
    "total_queries": 6991,
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
    "top8_oracle": 0.9631359366804939,
    "candidate_pool_ceiling": 0.981931531,
}


def get_git_info() -> Dict[str, str]:
    """Dynamically resolve runtime Git HEAD commit SHA and working tree status."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True
        ).strip()
    except Exception as e:
        sha = f"ERROR: {e}"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            text=True
        ).strip()
    except Exception as e:
        status = f"ERROR: {e}"

    return {
        "git_commit_sha": sha,
        "git_status_porcelain": status,
        "is_dirty": len(status) > 0,
    }


def sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    if not path.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def log_trace(
    stage: str,
    status: str,
    script_path: Optional[Path] = None,
    input_paths: Optional[List[Path]] = None,
    output_path: Optional[Path] = None,
    records_processed: int = 0,
    wall_clock_sec: float = 0.0,
    extra_info: Optional[Dict[str, Any]] = None,
):
    """Log structured execution record into EXECUTION_TRACE.jsonl."""
    git_info = get_git_info()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "git_commit_sha": git_info["git_commit_sha"],
        "is_dirty": git_info["is_dirty"],
        "script_path": str(script_path.relative_to(REPO_ROOT)) if script_path and script_path.is_relative_to(REPO_ROOT) else str(script_path),
        "script_sha256": sha256(script_path) if script_path and script_path.exists() else None,
        "input_hashes": {str(p): sha256(p) for p in (input_paths or [])},
        "output_path": str(output_path.relative_to(REPO_ROOT)) if output_path and output_path.is_relative_to(REPO_ROOT) else str(output_path),
        "output_sha256": sha256(output_path) if output_path and output_path.exists() else None,
        "records_processed": records_processed,
        "wall_clock_seconds": round(wall_clock_sec, 3),
        "extra_info": extra_info or {},
    }
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_execution_proof(stage: str, stage_data: Dict[str, Any]):
    """Update EXECUTION_PROOF.json with structured audit metadata."""
    git_info = get_git_info()
    proof = {}
    if PROOF_FILE.exists():
        try:
            with PROOF_FILE.open("r", encoding="utf-8") as f:
                proof = json.load(f)
        except Exception:
            proof = {}

    proof["schema_version"] = "dsc2026.gemini.huy_sparse_resurrection_v1.execution_proof.v1"
    proof["last_updated"] = datetime.now(timezone.utc).isoformat()
    proof["runtime_git"] = git_info
    stages = proof.setdefault("stages", {})
    stages[stage] = stage_data

    with PROOF_FILE.open("w", encoding="utf-8") as f:
        json.dump(proof, f, indent=2, ensure_ascii=False)


def load_baseline_data() -> Tuple[Any, ...]:
    """Load canonical baseline fasttrack data structures."""
    folds, pools, questions, golds, e5_orders, e5_scores, dup, pred_hashes = core.load_inputs()
    fold_for = {qid: fold for fold, qids in folds.items() for qid in qids}

    base_orders: Dict[str, List[str]] = {}
    base_scores: Dict[str, Dict[str, float]] = {}
    if BASELINE_PRED_FILE.exists():
        with BASELINE_PRED_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = str(rec["qid"])
                base_orders[qid] = [str(x) for x in rec["order"]]
                base_scores[qid] = {str(d): float(s) for d, s in rec["scores"].items()}

    return folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores


def load_cached_feature_rows() -> Dict[str, np.ndarray]:
    """Load baseline 44D feature matrices for each query."""
    if not BASELINE_FEAT_FILE.exists():
        raise FileNotFoundError(f"Missing {BASELINE_FEAT_FILE}")
    with np.load(BASELINE_FEAT_FILE) as z:
        rows = {f.replace("row_", ""): z[f].astype(np.float32) for f in z.files}
    return rows


def evaluate_orders(orders: Dict[str, List[str]], golds: Dict[str, List[str]], folds: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute exact Recall@k, Precision@k, single/multi gold, and per-fold metrics."""
    qids = sorted(orders.keys(), key=int)
    n = len(qids)
    if n == 0:
        return {}

    recalls_at_k = {1: [], 5: [], 6: [], 7: [], 8: [], 10: [], 20: [], 50: [], 100: []}
    precisions_at_5 = []
    single_recalls_at_5 = []
    multi_recalls_at_5 = []

    fold_for = {}
    for f, f_qids in folds.items():
        for q in f_qids:
            fold_for[str(q)] = f

    fold_r5 = {f: [] for f in folds}

    for qid in qids:
        g = set(golds.get(qid, []))
        if not g:
            continue
        ord_q = orders[qid]
        f = fold_for.get(qid)

        # Hits at k
        for k in recalls_at_k:
            topk = set(ord_q[:k])
            hit_cnt = len(g & topk)
            rec = hit_cnt / len(g)
            recalls_at_k[k].append(rec)

        # P@5
        p5 = len(g & set(ord_q[:5])) / 5.0
        precisions_at_5.append(p5)

        # Single vs Multi
        r5 = len(g & set(ord_q[:5])) / len(g)
        if len(g) == 1:
            single_recalls_at_5.append(r5)
        else:
            multi_recalls_at_5.append(r5)

        if f in fold_r5:
            fold_r5[f].append(r5)

    return {
        "num_queries": n,
        "recall_at_1": float(np.mean(recalls_at_k[1])),
        "recall_at_5": float(np.mean(recalls_at_k[5])),
        "recall_at_6": float(np.mean(recalls_at_k[6])),
        "recall_at_7": float(np.mean(recalls_at_k[7])),
        "recall_at_8": float(np.mean(recalls_at_k[8])),
        "recall_at_10": float(np.mean(recalls_at_k[10])),
        "recall_at_20": float(np.mean(recalls_at_k[20])),
        "recall_at_50": float(np.mean(recalls_at_k[50])),
        "recall_at_100": float(np.mean(recalls_at_k[100])),
        "precision_at_5": float(np.mean(precisions_at_5)),
        "single_gold_recall_at_5": float(np.mean(single_recalls_at_5)),
        "multi_gold_recall_at_5": float(np.mean(multi_recalls_at_5)),
        "per_fold_recall_at_5": {f: float(np.mean(scores)) for f, scores in fold_r5.items()},
    }
