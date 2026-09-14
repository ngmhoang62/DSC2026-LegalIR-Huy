"""Strict 5-fold OOF training, training-size scaling, and seed ensemble execution.
Implements Sections 16, 17, 18, and 19.
Produces:
- results/gemini/jina_honest_adaptation_v1/cache/ADAPTED_JINA_FOLD_{i}_PREDICTIONS.jsonl
- results/gemini/jina_honest_adaptation_v1/TRAIN_SIZE_SCALING_REPORT.json
- results/gemini/jina_honest_adaptation_v1/JINA_OOF_REPORT.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from common import (
    EXP_CACHE,
    EXP_CHECKPOINTS,
    EXP_RESULTS,
    REPO_ROOT,
    get_git_info,
    log_execution_trace,
    sha256_file,
)
from dataset import LegalRetrievalDataset
from model import build_model, compute_group_loss, compute_parent_scores

sys.path.insert(0, str(REPO_ROOT))
from benchmark_jina_reranker_holdouts import top_passages


def train_single_jina_model(
    train_qids: List[str],
    dataset: LegalRetrievalDataset,
    method: str,
    lr: float,
    lambda_anchor: float,
    epochs: int = 2,
    seed: int = 2026,
    model_path: Optional[Path] = None,
    save_checkpoint: Optional[Path] = None,
    device: str = "cuda",
) -> Tuple[nn.Module, Any]:
    """Train Jina model on train_qids with curriculum and stability regularizer."""
    if model_path is None:
        model_path = REPO_ROOT / "cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"

    model, tok = build_model(method, model_path, rank=16, dtype=torch.bfloat16, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        rng = random.Random(seed + epoch * 1000)
        shuffled = list(train_qids)
        rng.shuffle(shuffled)

        for step, qid in enumerate(shuffled):
            sample = dataset.sample_query_candidates(
                qid, epoch=epoch, seed_offset=step
            )
            if not sample["pos_indices"]:
                continue

            optimizer.zero_grad()
            parent_scores = compute_parent_scores(
                model,
                tok,
                sample["pairs"],
                sample["doc_indices_for_passages"],
                sample["num_docs"],
                tau_pool=0.15,
                device=device,
            )

            frozen = torch.tensor(
                sample["frozen_doc_scores"], device=device, dtype=parent_scores.dtype
            )
            tot_loss, _, _ = compute_group_loss(
                parent_scores, sample["pos_indices"], frozen, lambda_anchor=lambda_anchor
            )
            tot_loss.backward()
            optimizer.step()

    if save_checkpoint is not None:
        save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainable = {
            k: v.cpu()
            for k, v in model.state_dict().items()
            if v.requires_grad or "classifier" in k or "lora" in k
        }
        torch.save({"method": method, "state_dict": trainable}, save_checkpoint)

    return model, tok


def score_candidate_pool(
    model: nn.Module,
    tok: Any,
    dataset: LegalRetrievalDataset,
    target_qids: List[str],
    device: str = "cuda",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """Score candidate pool for target queries. Returns (score_map, order_map)."""
    model.eval()
    score_map: Dict[str, Dict[str, float]] = {}
    order_map: Dict[str, List[str]] = {}

    with torch.no_grad():
        for qid in target_qids:
            qtext = dataset.questions[qid]
            pool = dataset.pools[qid]

            pairs = []
            doc_indices = []
            for doc_idx, d in enumerate(pool):
                dtext = dataset.contexts.get(d, "")
                passages = top_passages(qtext, dtext, count=2, window=220, overlap=70)
                for p in passages:
                    pairs.append((qtext, p))
                    doc_indices.append(doc_idx)

            parent_scores = compute_parent_scores(
                model, tok, pairs, doc_indices, len(pool), tau_pool=0.15, device=device
            )

            # Map to sigmoid probabilities for consistent score scale
            sigmoid_scores = torch.sigmoid(parent_scores).cpu().tolist()
            q_scores = {d: float(s) for d, s in zip(pool, sigmoid_scores)}
            q_order = [d for _, d in sorted(zip(sigmoid_scores, pool), reverse=True)]

            score_map[qid] = q_scores
            order_map[qid] = q_order

    return score_map, order_map


def evaluate_standalone_metrics(
    order_map: Dict[str, List[str]],
    golds: Dict[str, Set[str]],
    folds: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute R@1, R@5, R@8, R@10 standalone metrics."""
    res = {}
    for k in (1, 5, 8, 10):
        recalls = []
        single, multi = [], []
        fold_recalls = {f: [] for f in folds}
        for f, qids in folds.items():
            for q in qids:
                gold = golds[q]
                hits = len(set(order_map[q][:k]) & gold)
                r = hits / len(gold)
                recalls.append(r)
                if len(gold) == 1:
                    single.append(r)
                else:
                    multi.append(r)
                fold_recalls[f].append(r)

        res[f"recall_at_{k}"] = float(np.mean(recalls))
        if k == 5:
            res["single_gold_recall_at_5"] = float(np.mean(single))
            res["multi_gold_recall_at_5"] = float(np.mean(multi))
            res["per_fold_recall_at_5"] = {f: float(np.mean(v)) for f, v in fold_recalls.items()}
    return res
