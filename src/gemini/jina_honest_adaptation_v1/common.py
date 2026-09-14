"""Shared utilities for jina_honest_adaptation_v1 experiment.
Provides dynamic Git status, baseline data loading, metric computation,
hashing, and audit trace logging.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(r"d:\Study\DSC2026\sota").resolve()
EXP_SRC = REPO_ROOT / "src/gemini/jina_honest_adaptation_v1"
EXP_RESULTS = REPO_ROOT / "results/gemini/jina_honest_adaptation_v1"
EXP_CACHE = EXP_RESULTS / "cache"
EXP_CHECKPOINTS = EXP_RESULTS / "checkpoints"

EXP_RESULTS.mkdir(parents=True, exist_ok=True)
EXP_CACHE.mkdir(parents=True, exist_ok=True)
EXP_CHECKPOINTS.mkdir(parents=True, exist_ok=True)


def get_git_info() -> Dict[str, Any]:
    """Dynamically query Git HEAD and working-tree porcelain status."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return {
            "head": head,
            "branch": branch,
            "porcelain_clean": len(status) == 0,
            "porcelain_status": status.splitlines() if status else [],
        }
    except Exception as exc:
        return {"head": "UNKNOWN", "error": str(exc), "porcelain_clean": False}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_os_gpu_memory() -> Dict[str, Any]:
    """Query OS-level Dedicated and Shared GPU memory via nvidia-smi and Windows performance counters."""
    res = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        tot, used, free = [float(x.strip()) for x in out.split(",")]
        res["nvidia_smi"] = {"total_mb": tot, "used_mb": used, "free_mb": free}
    except Exception as e:
        res["nvidia_smi"] = {"error": str(e)}

    try:
        cmd = "Get-Counter '\\GPU Non Local Adapter Memory(*)\\Non Local Usage' | Select-Object -ExpandProperty CounterSamples | Select-Object InstanceName, CookedValue | ConvertTo-Json"
        out = subprocess.check_output(["powershell", "-Command", cmd], text=True)
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        shared_mb = sum(
            item["CookedValue"] / (1024**2)
            for item in data
            if "phys_0" in item.get("InstanceName", "")
        )
        if shared_mb == 0 and data:
            shared_mb = data[0]["CookedValue"] / (1024**2)
        res["windows_shared_gpu_memory_mb"] = round(shared_mb, 2)
    except Exception as e:
        res["windows_shared_gpu_memory_mb"] = None
        res["windows_shared_error"] = str(e)
    return res


def init_execution_trace(reuse_parity_sha: Optional[str] = None) -> None:
    """Reinitialize EXECUTION_TRACE.jsonl for the current experiment."""
    trace_path = EXP_RESULTS / "EXECUTION_TRACE.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        if reuse_parity_sha:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": "REUSE_FROZEN_PARITY",
                "status": "PASS",
                "reused_artifact": "FROZEN_JINA_PARITY.json",
                "reused_artifact_sha256": reuse_parity_sha,
                "note": "Reusing verified pre-training parity results under identical contracts",
                "git": get_git_info(),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_execution_trace(
    stage_name: str,
    command: str,
    code_hash: str,
    model_checkpoint_hash: str,
    train_query_count: int,
    val_query_count: int,
    optimizer_steps: int,
    gpu_info: str,
    runtime_sec: float,
    output_hashes: Dict[str, str],
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append structured execution trace entry to EXECUTION_TRACE.jsonl."""
    trace_path = EXP_RESULTS / "EXECUTION_TRACE.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage_name,
        "command": command,
        "code_hash": code_hash,
        "model_checkpoint_hash": model_checkpoint_hash,
        "train_query_count": train_query_count,
        "val_query_count": val_query_count,
        "optimizer_steps": optimizer_steps,
        "gpu": gpu_info,
        "runtime_sec": round(runtime_sec, 3),
        "output_hashes": output_hashes,
        "status": status,
        "git": get_git_info(),
    }
    if extra:
        entry["extra"] = extra
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_metrics(
    predictions: Dict[str, List[str]],
    golds: Dict[str, Set[str]],
    fold_for: Dict[str, str],
    k: int = 5,
) -> Dict[str, Any]:
    """Compute overall Recall@k, Precision@k, single-gold, multi-gold, and per-fold metrics."""
    qids = [q for q in predictions if q in golds and golds[q]]
    if not qids:
        return {}

    recalls = []
    precisions = []
    single_recalls = []
    multi_recalls = []
    fold_recalls = {f: [] for f in sorted(set(fold_for.values()))}

    for q in qids:
        gold = golds[q]
        top_k = predictions[q][:k]
        hits = len(set(top_k) & gold)
        r = hits / len(gold)
        p = hits / float(k)
        recalls.append(r)
        precisions.append(p)
        if len(gold) == 1:
            single_recalls.append(r)
        else:
            multi_recalls.append(r)
        f = fold_for.get(q)
        if f in fold_recalls:
            fold_recalls[f].append(r)

    return {
        f"recall_at_{k}": float(sum(recalls) / len(recalls)),
        f"precision_at_{k}": float(sum(precisions) / len(precisions)),
        f"single_gold_recall_at_{k}": float(sum(single_recalls) / len(single_recalls)) if single_recalls else 0.0,
        f"multi_gold_recall_at_{k}": float(sum(multi_recalls) / len(multi_recalls)) if multi_recalls else 0.0,
        "queries_count": len(qids),
        "single_gold_count": len(single_recalls),
        "multi_gold_count": len(multi_recalls),
        "per_fold_recall": {f: float(sum(v) / len(v)) if v else 0.0 for f, v in fold_recalls.items()},
    }
