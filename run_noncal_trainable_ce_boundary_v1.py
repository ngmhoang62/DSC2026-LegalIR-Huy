#!/usr/bin/env python
"""
NONCAL TRAINABLE CE BOUNDARY V1
===============================

Purpose
-------
Final high-EV rescue experiment for DSC2026 LegalIR.

Key change versus the killed frozen-verifier family:
  * semantic verifier is TRAINED for relevance/boundary discrimination;
  * model family and optimization are inherited from the previously successful
    selective CE pilot in sibling LegalIR:
        BAAI/bge-reranker-v2-m3
        LoRA r16 alpha32 dropout.05
        LR 5e-5, AdamW, one epoch, query-balanced BCE
  * hard negatives explicitly include the new full-corpus frontier.

Leakage contract
----------------
CAL600 qids are obtained LABEL-FREE first and are excluded from:
  - every training fold
  - every OOF gate metric
  - final training

Only nonCAL Strict-V2 qids are used for fitting / validation.
CAL actions are generated and sealed before CAL gold labels are loaded.

V2 evaluation frontier:
    overlap novel = Adapted-E5@50 ∩ LAL@50 - current_pool
    base = sealed Adapted-E5+LAL RRF32 Top5
    ranks 1-4 immutable
    exactly one CE crossover of rank5 -> swap; otherwise abstain

Training negatives per query are deterministic and include:
    base Top5 non-golds
    overlap-novel non-golds
    E5/LAL union-novel non-golds
    source Top20 non-golds
No threshold tuning, no policy grid, no CAL fitting.

If strict nonCAL OOF gate passes, a final CE is fit on ALL nonCAL qids and
applied label-free to:
    CAL overlap novel = Adapted-E5@50 ∩ AITeamVN-FT@50 - D1 pool.

Run
---
python ../run_noncal_trainable_ce_boundary_v1.py \
  --repo-root /d/Study/DSC2026/sota \
  --pair-microbatch 2

This is compute-heavy and resumable per fold/query.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch.nn import functional as F


DEPTH = 50
NEGATIVE_CAP = 12
ACCUMULATION_QUERIES = 8
LR = 5e-5
WEIGHT_DECAY = 0.01
SEED = 112

EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_P5 = 0.20566666666666666

OOF_MIN_DELTA = 0.001
OOF_MIN_NONNEG_FOLDS = 4
OOF_WORST_FOLD = -0.005
OOF_MIN_INTERVENTION_PRECISION = 0.55


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall(top5: Sequence[str], gold: Set[str]) -> float:
    return len(set(top5[:5]) & gold) / len(gold)


def precision(top5: Sequence[str], gold: Set[str]) -> float:
    return len(set(top5[:5]) & gold) / 5.0


def metrics(pred, gold, ids):
    return {
        "recall_at_5": float(
            np.mean([recall(pred[q], gold[q]) for q in ids])
        ),
        "precision_at_5": float(
            np.mean([precision(pred[q], gold[q]) for q in ids])
        ),
    }


def overlap_novel(a, b, pool):
    pool_set = set(map(str, pool))
    rb = {str(d): i + 1 for i, d in enumerate(b[:DEPTH])}
    ra = {str(d): i + 1 for i, d in enumerate(a[:DEPTH])}
    c = [
        str(d)
        for d in a[:DEPTH]
        if str(d) in rb and str(d) not in pool_set
    ]
    return sorted(
        set(c),
        key=lambda d: (ra[d] + rb[d], max(ra[d], rb[d]), d),
    )


def source_union_novel(a, b, pool):
    ps = set(map(str, pool))
    merged = list(
        dict.fromkeys(
            [str(d) for d in a[:DEPTH]]
            + [str(d) for d in b[:DEPTH]]
        )
    )
    return [d for d in merged if d not in ps]


def action_outcomes(ids, base, modified, gold):
    beneficial = harmful = neutral = 0
    details = []
    for q in ids:
        if base[q][:5] == modified[q][:5]:
            continue
        before = recall(base[q], gold[q])
        after = recall(modified[q], gold[q])
        delta = after - before
        if delta > 0:
            beneficial += 1
            kind = "beneficial"
        elif delta < 0:
            harmful += 1
            kind = "harmful"
        else:
            neutral += 1
            kind = "neutral"
        details.append(
            {
                "qid": q,
                "before": before,
                "after": after,
                "delta": delta,
                "outcome": kind,
                "base_top5": base[q][:5],
                "modified_top5": modified[q][:5],
            }
        )
    return {
        "actions": len(details),
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "intervention_precision_excluding_neutral": (
            beneficial / max(beneficial + harmful, 1)
        ),
        "details": details,
    }


def get_cal_ids_label_free(root: Path) -> Tuple[List[str], Dict[str, str]]:
    from src.gemini.huy_d1_frozen_section_public_v1.common import (
        load_cal_data_label_free,
    )

    (
        _docs,
        queries,
        _blocks,
        ids,
        _pool,
        _views,
        _scores,
        _types,
        _cite,
    ) = load_cal_data_label_free()

    ids = [str(q) for q in ids]
    questions = {q: str(queries[q][0]) for q in ids}
    if len(ids) != 600:
        raise RuntimeError(f"Expected CAL600, got {len(ids)}")
    return ids, questions


class RenderData:
    """
    Minimal label-free adapter required by sibling LegalIR's Evidence class.
    It exposes only corpus/query representation; no gold labels.
    """

    def __init__(
        self,
        *,
        bundle: Path,
        questions: Dict[str, str],
        sibling_evidence_db: Path,
    ):
        self.questions = dict(questions)

        # Reuse the sibling Evidence inventory signature exactly.
        con = sqlite3.connect(sibling_evidence_db)
        row = con.execute(
            "SELECT signature FROM complete WHERE kind='inventory'"
        ).fetchone()
        con.close()
        if row is None:
            raise RuntimeError(
                "Sibling exp_final Evidence inventory is not prepared. "
                "Run the old selective CE pilot preflight/benchmark first."
            )
        self.fingerprint = str(row[0])

        chunk_rows = read_jsonl(bundle / "chunk_ids.jsonl")
        self.chunk_ids = [str(r["chunk_id"]) for r in chunk_rows]
        parents = [str(r["doc_id"]) for r in chunk_rows]
        self.doc_ids = sorted(set(parents))
        self.doc_row = {d: i for i, d in enumerate(self.doc_ids)}
        parent_idx = np.asarray(
            [self.doc_row[d] for d in parents], dtype=np.int64
        )
        positions = [[] for _ in self.doc_ids]
        for i, p in enumerate(parent_idx):
            positions[int(p)].append(i)
        self.positions = [
            np.asarray(v, dtype=np.int64) for v in positions
        ]

        self._matrix = np.load(
            bundle / "embeddings.f16.npy", mmap_mode="r"
        )

        qids = [
            str(x)
            for x in json.loads(
                (bundle / "train_query_ids.json").read_text(
                    encoding="utf-8"
                )
            )
        ]
        self._qvec = np.load(
            bundle / "train_queries.f32.npy", mmap_mode="r"
        )
        self._qrow = {q: i for i, q in enumerate(qids)}
        missing = set(self.questions) - set(self._qrow)
        if missing:
            raise RuntimeError(
                f"Query-vector cache missing {len(missing)} ids; "
                f"sample={sorted(missing)[:10]}"
            )

    def matrix(self, source: str):
        if source != "e5":
            raise ValueError(source)
        return self._matrix

    def query_vector(self, qid: str, source: str):
        if source != "e5":
            raise ValueError(source)
        v = np.asarray(
            self._qvec[self._qrow[qid]], dtype=np.float32
        )
        return v / max(float(np.linalg.norm(v)), 1e-12)


def load_noncal_world(
    root: Path,
    sibling: Path,
    cal_ids: Set[str],
):
    bundle = root / "cache/research_v2_e5_confirmation/bundle-v1"

    # Sanitize labels: CAL qids receive no gold entry at all.
    questions: Dict[str, str] = {}
    folds: Dict[str, str] = {}
    gold: Dict[str, Set[str]] = {}
    transfer_path = bundle / "V2_TRANSFER_QUERIES.jsonl"
    for row in read_jsonl(transfer_path):
        q = str(row["qid"])
        questions[q] = str(row["question"])
        folds[q] = str(row["fold"])
        if q not in cal_ids:
            gold[q] = {str(d) for d in row["gold"]}

    all_ids = sorted(questions, key=int)
    noncal = [q for q in all_ids if q not in cal_ids]

    if len(all_ids) != 6991:
        raise RuntimeError(f"V2 population mismatch: {len(all_ids)}")
    if len(noncal) != len(all_ids) - 600:
        raise RuntimeError(
            f"Expected 600 CAL overlap, got nonCAL={len(noncal)}"
        )
    if set(gold) != set(noncal):
        raise RuntimeError("Gold map is not exactly nonCAL population")

    pool = {
        str(r["qid"]): [str(d) for d in r["doc_ids"]]
        for r in read_jsonl(bundle / "V2_CANDIDATE_POOL.jsonl")
    }

    anchor_path = (
        root
        / "results/research_v2_post_e5/"
        "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
    )
    anchor = {
        str(r["qid"]): r for r in read_jsonl(anchor_path)
    }
    base = {
        q: [str(d) for d in anchor[q]["fused_top5"]]
        for q in all_ids
    }

    e5 = {}
    for i in range(5):
        p = (
            root
            / f"results/research_v2_open_rl/fold_{i}/"
            "FULL_CORPUS_PREDICTIONS.jsonl"
        )
        if not p.is_file():
            raise FileNotFoundError(p)
        for r in read_jsonl(p):
            e5[str(r["qid"])] = [
                str(d) for d in r["adapted_order_top150"]
            ]

    source_db = (
        sibling
        / "cache/exp112_task_adaptive_retrieval/sources.sqlite"
    )
    con = sqlite3.connect(
        f"file:{source_db.resolve().as_posix()}?mode=ro&immutable=1",
        uri=True,
    )
    if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        con.close()
        raise RuntimeError("sources.sqlite integrity failure")

    lal = {}
    for q in all_ids:
        row = con.execute(
            "SELECT payload FROM sources WHERE q=? AND source='lal'",
            (q,),
        ).fetchone()
        if row is None:
            con.close()
            raise RuntimeError(f"Missing LAL source qid={q}")
        lal[q] = [
            str(item["doc_id"]) for item in json.loads(row[0])
        ]
    con.close()

    expected = set(all_ids)
    if not (
        set(pool) == set(base) == set(e5) == set(lal) == expected
    ):
        raise RuntimeError("World population mismatch")

    sibling_evidence = (
        sibling / "cache/exp_final_retrieval/evidence.sqlite"
    )
    if not sibling_evidence.is_file():
        raise FileNotFoundError(
            "Missing sibling exp_final Evidence DB: "
            f"{sibling_evidence}"
        )

    render = RenderData(
        bundle=bundle,
        questions=questions,
        sibling_evidence_db=sibling_evidence,
    )

    if set(render.doc_ids) != set(
        str(d) for d in render.doc_ids
    ):
        raise RuntimeError("Document-id normalization failure")

    # Strong chunk-bank alignment check against sibling Evidence DB.
    con = sqlite3.connect(
        f"file:{sibling_evidence.resolve().as_posix()}?mode=ro&immutable=1",
        uri=True,
    )
    probe = [
        0,
        len(render.chunk_ids) // 3,
        2 * len(render.chunk_ids) // 3,
        len(render.chunk_ids) - 1,
    ]
    for idx in probe:
        row = con.execute(
            "SELECT payload FROM chunks WHERE idx=?", (idx,)
        ).fetchone()
        if row is None:
            con.close()
            raise RuntimeError(f"Evidence DB missing chunk idx={idx}")
        payload = json.loads(row[0])
        if str(payload["chunk_id"]) != str(render.chunk_ids[idx]):
            con.close()
            raise RuntimeError(
                f"Chunk bank mismatch idx={idx}: "
                f"{payload['chunk_id']} != {render.chunk_ids[idx]}"
            )
    con.close()

    return {
        "bundle": bundle,
        "questions": questions,
        "folds": folds,
        "gold": gold,
        "pool": pool,
        "base": base,
        "e5": e5,
        "lal": lal,
        "noncal": noncal,
        "render": render,
        "source_db": source_db,
        "anchor_path": anchor_path,
        "transfer_path": transfer_path,
    }


def deterministic_negatives(
    q: str,
    gold: Set[str],
    base: Dict[str, List[str]],
    e5: Dict[str, List[str]],
    other: Dict[str, List[str]],
    pool: Dict[str, List[str]],
) -> List[str]:
    selected: List[str] = []
    blocked = set(gold)

    def add(values: Iterable[str], cap: int | None = None):
        added = 0
        for raw in values:
            d = str(raw)
            if d in blocked:
                continue
            blocked.add(d)
            selected.append(d)
            added += 1
            if cap is not None and added >= cap:
                break

    # Existing Top5 false positives: strongest eviction protection examples.
    add(base[q][:5])

    # New frontier hard negatives.
    ov = overlap_novel(e5[q], other[q], pool[q])
    add(ov, 4)

    union = source_union_novel(e5[q], other[q], pool[q])
    add(union, 4)

    # Backfill with high-ranked dense confusers.
    add(e5[q][:20])
    add(other[q][:20])

    return selected[:NEGATIVE_CAP]


def train_model(
    *,
    sibling: Path,
    render: RenderData,
    world: dict,
    train_ids: List[str],
    directory: Path,
    pair_microbatch: int,
):
    sys.path.insert(0, str(sibling))
    sys.path.insert(0, str(sibling / "src"))

    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence
    from exp_final.learning import checkpoint, set_rng, rng_state
    from transformers import get_cosine_schedule_with_warmup

    directory.mkdir(parents=True, exist_ok=True)
    signature = digest(
        {
            "experiment": "noncal_trainable_ce_boundary_v1",
            "train_qids_sha": digest(sorted(train_ids, key=int)),
            "queries": len(train_ids),
            "negative_cap": NEGATIVE_CAP,
            "accumulation_queries": ACCUMULATION_QUERIES,
            "pair_microbatch": pair_microbatch,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "model": "BAAI/bge-reranker-v2-m3 LoRA r16 alpha32",
        }
    )

    success = directory / "_SUCCESS.json"
    model_path = directory / "model.pt"
    if success.is_file() and model_path.is_file():
        s = json.loads(success.read_text(encoding="utf-8"))
        if (
            s.get("signature") == signature
            and s.get("model_sha256") == sha256_file(model_path)
        ):
            print(
                f"  training cache valid: {directory.name}",
                flush=True,
            )
            return s
        raise RuntimeError(
            f"Completed training cache contract mismatch: {directory}"
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = CrossEncoder()
    model.train()
    evidence = Evidence(render, model.tokenizer)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=LR, weight_decay=WEIGHT_DECAY
    )

    order = list(train_ids)
    random.Random(SEED).shuffle(order)
    updates = math.ceil(len(order) / ACCUMULATION_QUERIES)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(0.1 * updates)),
        updates,
    )

    resume = directory / "resume.pt"
    position = 0
    if resume.is_file():
        receipt = resume.with_suffix(".sha.json")
        if (
            not receipt.is_file()
            or json.loads(receipt.read_text(encoding="utf-8"))["sha256"]
            != sha256_file(resume)
        ):
            raise RuntimeError("Unverified CE resume checkpoint")
        state = torch.load(
            resume, map_location="cpu", weights_only=False
        )
        if state["signature"] != signature:
            raise RuntimeError("CE resume signature mismatch")
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        set_rng(state["rng"])
        position = int(state["position"])
        print(
            f"  resume {directory.name}: {position}/{len(order)}",
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    pairs_seen = 0
    losses = []

    try:
        for index in range(position, len(order)):
            q = order[index]
            positives = [
                d
                for d in sorted(world["gold"][q])
                if d in render.doc_row
            ]
            negatives = deterministic_negatives(
                q,
                world["gold"][q],
                world["base"],
                world["e5"],
                world["lal"],
                world["pool"],
            )
            if not positives or not negatives:
                raise RuntimeError(
                    f"Invalid training group {q}: "
                    f"pos={len(positives)} neg={len(negatives)}"
                )

            docs = positives + negatives
            weights = np.asarray(
                [0.5 / len(positives)] * len(positives)
                + [0.5 / len(negatives)] * len(negatives),
                dtype=np.float32,
            )
            targets = np.asarray(
                [1.0] * len(positives)
                + [0.0] * len(negatives),
                dtype=np.float32,
            )

            query_loss = 0.0
            for start in range(0, len(docs), pair_microbatch):
                local_docs = docs[start:start + pair_microbatch]
                pairs = [
                    evidence.package(q, d) for d in local_docs
                ]
                logits = model(pairs)
                target = torch.as_tensor(
                    targets[start:start + pair_microbatch],
                    dtype=torch.float32,
                    device="cuda",
                )
                weight = torch.as_tensor(
                    weights[start:start + pair_microbatch],
                    dtype=torch.float32,
                    device="cuda",
                )
                loss = (
                    F.binary_cross_entropy_with_logits(
                        logits,
                        target,
                        reduction="none",
                    )
                    * weight
                ).sum() / ACCUMULATION_QUERIES

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss q={q} index={index}"
                    )
                loss.backward()
                query_loss += (
                    float(loss.detach())
                    * ACCUMULATION_QUERIES
                )
                pairs_seen += len(local_docs)

            losses.append(query_loss)

            boundary = (
                (index + 1) % ACCUMULATION_QUERIES == 0
                or index + 1 == len(order)
            )
            if boundary:
                torch.nn.utils.clip_grad_norm_(
                    params, 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if boundary and (
                (index + 1) % 256 == 0
                or index + 1 == len(order)
            ):
                checkpoint(
                    resume,
                    model,
                    optimizer,
                    scheduler,
                    signature=signature,
                    position=index + 1,
                )

            if (index + 1) % 32 == 0 or index + 1 == len(order):
                elapsed = time.perf_counter() - started
                done = index + 1 - position
                eta = (
                    elapsed / max(done, 1)
                    * (len(order) - index - 1)
                )
                print(
                    f"    train {directory.name}: "
                    f"{index+1}/{len(order)} "
                    f"loss32={np.mean(losses[-32:]):.5f} "
                    f"qps={done/max(elapsed,1e-9):.3f} "
                    f"eta_min={eta/60:.1f} "
                    f"vram={torch.cuda.max_memory_reserved()/2**30:.2f}GiB",
                    flush=True,
                )

        checkpoint(
            model_path,
            model,
            optimizer,
            scheduler,
            signature=signature,
            position=len(order),
        )
        result = {
            "status": "COMPLETE",
            "signature": signature,
            "queries": len(order),
            "pairs": pairs_seen,
            "seconds_this_process": time.perf_counter() - started,
            "model_sha256": sha256_file(model_path),
            "loss_last32": float(np.mean(losses[-32:])),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30
            ),
        }
        write_json(success, result)
        return result
    finally:
        evidence.db.close()
        del evidence, model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def score_frontier(
    *,
    sibling: Path,
    render: RenderData,
    ids: List[str],
    base: Dict[str, List[str]],
    e5: Dict[str, List[str]],
    other: Dict[str, List[str]],
    pool: Dict[str, List[str]],
    model_path: Path,
    score_dir: Path,
    pair_microbatch: int,
):
    sys.path.insert(0, str(sibling))
    sys.path.insert(0, str(sibling / "src"))

    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence

    model = CrossEncoder(model_path)
    model.eval()
    evidence = Evidence(render, model.tokenizer)
    score_dir.mkdir(parents=True, exist_ok=True)

    modified = {q: list(base[q][:5]) for q in ids}
    actions = {}
    started = time.perf_counter()

    try:
        for index, q in enumerate(ids, 1):
            novels = overlap_novel(
                e5[q], other[q], pool[q]
            )
            universe = list(
                dict.fromkeys(base[q][:5] + novels)
            )
            path = score_dir / f"{q}.json"
            signature = digest(
                [
                    sha256_file(model_path),
                    q,
                    universe,
                    "overlap-ce-crossover-v1",
                ]
            )

            if path.is_file():
                row = json.loads(
                    path.read_text(encoding="utf-8")
                )
                if row["signature"] != signature:
                    raise RuntimeError(
                        f"Score cache mismatch q={q}"
                    )
                scores = {
                    str(d): float(s)
                    for d, s in row["scores"].items()
                }
            else:
                scores_list = []
                for start in range(
                    0, len(universe), pair_microbatch
                ):
                    docs = universe[
                        start:start + pair_microbatch
                    ]
                    pairs = [
                        evidence.package(q, d) for d in docs
                    ]
                    scores_list.extend(
                        model(pairs).detach().cpu().tolist()
                    )
                scores = {
                    d: float(s)
                    for d, s in zip(universe, scores_list)
                }
                write_json(
                    path,
                    {
                        "signature": signature,
                        "universe": universe,
                        "scores": scores,
                    },
                )

            if novels:
                defender = base[q][4]
                ordered = sorted(
                    universe,
                    key=lambda d: (-scores[d], str(d)),
                )
                ranks = {
                    d: i + 1 for i, d in enumerate(ordered)
                }
                eligible = [
                    c
                    for c in novels
                    if scores[c] > scores[defender]
                    and ranks[c] <= 5
                    and ranks[defender] > 5
                ]

                if len(eligible) == 1:
                    c = eligible[0]
                    modified[q] = list(base[q][:4]) + [c]
                    actions[q] = {
                        "qid": q,
                        "defender": defender,
                        "challenger": c,
                        "defender_score": scores[defender],
                        "challenger_score": scores[c],
                        "defender_rank": ranks[defender],
                        "challenger_rank": ranks[c],
                        "overlap_novel_count": len(novels),
                        "new_top5": modified[q],
                    }

            if index % 50 == 0 or index == len(ids):
                elapsed = time.perf_counter() - started
                print(
                    f"    score {score_dir.name}: "
                    f"{index}/{len(ids)} "
                    f"qps={index/max(elapsed,1e-9):.3f}",
                    flush=True,
                )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect()
        torch.cuda.empty_cache()

    return modified, actions


def run_oof(
    *,
    root: Path,
    sibling: Path,
    world: dict,
    out: Path,
    pair_microbatch: int,
):
    noncal = world["noncal"]
    folds = world["folds"]
    render = world["render"]

    all_pred = {}
    all_actions = {}
    training_reports = {}
    fold_reports = {}

    fold_names = [f"fold_{i}" for i in range(5)]

    for outer in fold_names:
        train_ids = [
            q for q in noncal if folds[q] != outer
        ]
        test_ids = [
            q for q in noncal if folds[q] == outer
        ]

        train_dir = out / "oof" / outer / "training"
        print(
            f"[OOF {outer}] train={len(train_ids)} "
            f"test={len(test_ids)}",
            flush=True,
        )
        training = train_model(
            sibling=sibling,
            render=render,
            world=world,
            train_ids=train_ids,
            directory=train_dir,
            pair_microbatch=pair_microbatch,
        )
        training_reports[outer] = training

        modified, actions = score_frontier(
            sibling=sibling,
            render=render,
            ids=test_ids,
            base=world["base"],
            e5=world["e5"],
            other=world["lal"],
            pool=world["pool"],
            model_path=train_dir / "model.pt",
            score_dir=out / "oof" / outer / "scores",
            pair_microbatch=pair_microbatch,
        )
        all_pred.update(modified)
        all_actions.update(actions)

        bm = metrics(
            world["base"], world["gold"], test_ids
        )
        mm = metrics(
            modified, world["gold"], test_ids
        )
        po = action_outcomes(
            test_ids,
            world["base"],
            modified,
            world["gold"],
        )
        po.pop("details")

        fold_reports[outer] = {
            "queries": len(test_ids),
            "baseline": bm,
            "modified": mm,
            "recall_delta": (
                mm["recall_at_5"] - bm["recall_at_5"]
            ),
            "paired_actions": po,
        }
        print(
            f"  {outer}: "
            f"{bm['recall_at_5']:.10f} -> "
            f"{mm['recall_at_5']:.10f} "
            f"({fold_reports[outer]['recall_delta']:+.10f}) "
            f"W/L/N={po['beneficial']}/"
            f"{po['harmful']}/{po['neutral']}",
            flush=True,
        )

    if set(all_pred) != set(noncal):
        raise RuntimeError("OOF prediction population incomplete")

    bm = metrics(
        world["base"], world["gold"], noncal
    )
    mm = metrics(
        all_pred, world["gold"], noncal
    )
    po = action_outcomes(
        noncal,
        world["base"],
        all_pred,
        world["gold"],
    )
    deltas = [
        fold_reports[f]["recall_delta"] for f in fold_names
    ]
    dr = mm["recall_at_5"] - bm["recall_at_5"]

    checks = {
        "delta_gte_0_001": dr >= OOF_MIN_DELTA,
        "wins_gt_losses": (
            po["beneficial"] > po["harmful"]
        ),
        "at_least_4_of_5_folds_nonnegative": (
            sum(d >= 0 for d in deltas)
            >= OOF_MIN_NONNEG_FOLDS
        ),
        "worst_fold_gte_minus_0_005": (
            min(deltas) >= OOF_WORST_FOLD
        ),
        "intervention_precision_gte_0_55": (
            po[
                "intervention_precision_excluding_neutral"
            ]
            >= OOF_MIN_INTERVENTION_PRECISION
        ),
    }

    report = {
        "schema": "manual.noncal_trainable_ce_boundary_v1.oof",
        "status": (
            "PASS_NONCAL_OOF_GATE"
            if all(checks.values())
            else "KILL_AT_NONCAL_OOF_GATE"
        ),
        "population": {
            "all_v2": 6991,
            "cal_excluded": 600,
            "noncal": len(noncal),
        },
        "mechanism": {
            "model": "BAAI/bge-reranker-v2-m3 LoRA r16 alpha32 dropout.05",
            "epochs": 1,
            "lr": LR,
            "negative_cap": NEGATIVE_CAP,
            "hard_negative_sources": [
                "base_top5",
                "E5∩LAL novel",
                "E5∪LAL novel",
                "E5 top20",
                "LAL top20",
            ],
            "candidate_prior": "E5@50 intersection LAL@50 novel",
            "action": "exactly one CE crossover of rank5; else abstain",
            "threshold_tuning": False,
        },
        "baseline": bm,
        "modified": mm,
        "delta_recall_at_5": dr,
        "paired_actions": po,
        "folds": fold_reports,
        "training": training_reports,
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
    }
    write_json(out / "NONCAL_OOF_REPORT.json", report)
    write_json(out / "NONCAL_OOF_PREDICTIONS.json", all_pred)
    write_json(out / "NONCAL_OOF_ACTIONS.json", all_actions)
    return report


def load_cal_frontier_label_free(root: Path):
    from src.gemini.huy_d1_frozen_section_public_v1.common import (
        load_cal_data_label_free,
    )

    (
        _docs,
        queries,
        blocks,
        ids,
        pool,
        _views,
        _scores,
        _types,
        _cite,
    ) = load_cal_data_label_free()

    ids = [str(q) for q in ids]
    questions = {q: str(queries[q][0]) for q in ids}
    pool = {
        str(q): [str(d) for d in pool[q]]
        for q in ids
    }
    blocks = {
        str(k): [str(q) for q in v]
        for k, v in blocks.items()
    }

    base = {
        str(q): [str(d) for d in r]
        for q, r in json.loads(
            (
                root
                / "results/sol_high_rl/"
                "BASELINE_LOBO_PREDICTIONS.json"
            ).read_text(encoding="utf-8")
        ).items()
    }
    e5 = {
        str(q): [str(d) for d in r]
        for q, r in json.loads(
            (
                root
                / "results/manual/"
                "huy_cal600_adapted_e5_full_corpus_v1/"
                "CAL600_ADAPTED_E5_FULL_CORPUS_TOP150.json"
            ).read_text(encoding="utf-8")
        ).items()
    }
    ai = {
        str(q): [str(d) for d in r]
        for q, r in json.loads(
            (
                root
                / "results/sol_high_rl/"
                "AITEAM_FT_FULL_CORPUS_TOP50.json"
            ).read_text(encoding="utf-8")
        ).items()
    }

    if not (
        set(base)
        == set(e5)
        == set(ai)
        == set(pool)
        == set(ids)
    ):
        raise RuntimeError("CAL frontier population mismatch")

    return ids, questions, blocks, pool, base, e5, ai


def final_fit_and_cal(
    *,
    root: Path,
    sibling: Path,
    world: dict,
    out: Path,
    pair_microbatch: int,
):
    final_dir = out / "final_noncal_training"
    print(
        f"[FINAL TRAIN] nonCAL queries={len(world['noncal'])}",
        flush=True,
    )
    train_model(
        sibling=sibling,
        render=world["render"],
        world=world,
        train_ids=world["noncal"],
        directory=final_dir,
        pair_microbatch=pair_microbatch,
    )

    (
        ids,
        cal_questions,
        blocks,
        pool,
        base,
        e5,
        ai,
    ) = load_cal_frontier_label_free(root)

    # Query texts must agree with the label-free render cache.
    for q in ids:
        if world["questions"][q] != cal_questions[q]:
            raise RuntimeError(
                f"CAL question mismatch qid={q}"
            )

    modified, actions = score_frontier(
        sibling=sibling,
        render=world["render"],
        ids=ids,
        base=base,
        e5=e5,
        other=ai,
        pool=pool,
        model_path=final_dir / "model.pt",
        score_dir=out / "cal_scores_label_free",
        pair_microbatch=pair_microbatch,
    )

    action_doc = {
        "schema": "manual.noncal_trainable_ce_boundary_v1.cal_actions",
        "status": "SEALED_BEFORE_CAL_GOLD",
        "training_population": (
            f"{len(world['noncal'])} Strict-V2 nonCAL qids"
        ),
        "cal_training_overlap": 0,
        "candidate_prior": (
            "Adapted-E5@50 ∩ AITeamVN-FT@50 - D1 pool"
        ),
        "action": (
            "exactly one trained-CE crossover of D1 rank5; "
            "ranks1-4 immutable"
        ),
        "actions_count": len(actions),
        "actions": actions,
        "predictions": modified,
        "final_model_sha256": sha256_file(
            final_dir / "model.pt"
        ),
    }
    action_path = out / "CAL_ACTIONS_LABEL_FREE.json"
    write_json(action_path, action_doc)
    action_sha = sha256_file(action_path)

    # Gold reveal only after the label-free action artifact exists.
    from src.gemini.huy_d1_frozen_section_public_v1.common import (
        load_cal_gold_labels,
    )

    gold, reveal_time = load_cal_gold_labels(ids)

    bm = metrics(base, gold, ids)
    mm = metrics(modified, gold, ids)
    if abs(bm["recall_at_5"] - EXPECTED_D1_R5) > 1e-12:
        raise RuntimeError(
            f"D1 recall parity failed: {bm}"
        )
    if abs(bm["precision_at_5"] - EXPECTED_D1_P5) > 5e-10:
        raise RuntimeError(
            f"D1 precision parity failed: {bm}"
        )

    po = action_outcomes(ids, base, modified, gold)
    dr = mm["recall_at_5"] - bm["recall_at_5"]
    dp = mm["precision_at_5"] - bm["precision_at_5"]

    single = [q for q in ids if len(gold[q]) == 1]
    multi = [q for q in ids if len(gold[q]) > 1]

    block_delta = {}
    for b, qids in blocks.items():
        block_delta[b] = (
            metrics(modified, gold, qids)["recall_at_5"]
            - metrics(base, gold, qids)["recall_at_5"]
        )

    gates = {
        "recall_positive": dr > 0,
        "precision_no_decrease": dp >= -1e-12,
        "wins_gt_losses": (
            po["beneficial"] > po["harmful"]
        ),
        "no_block_decrease": all(
            x >= -1e-12 for x in block_delta.values()
        ),
    }

    if all(gates.values()) and mm["recall_at_5"] >= 0.96:
        verdict = "STRONG_PROMOTE_NONCAL_TRAINABLE_CE_BOUNDARY_V1"
    elif (
        dr > 0
        and dp >= -1e-12
        and po["beneficial"] > po["harmful"]
    ):
        verdict = "PROMISING_NONCAL_TRAINABLE_CE_BOUNDARY_V1"
    else:
        verdict = "KILL_NONCAL_TRAINABLE_CE_BOUNDARY_V1"

    report = {
        "schema": "manual.noncal_trainable_ce_boundary_v1.cal",
        "gold_reveal_time": reveal_time,
        "action_artifact_sha256": action_sha,
        "baseline": bm,
        "modified": mm,
        "delta": {
            "recall_at_5": dr,
            "precision_at_5": dp,
            "single_gold": (
                metrics(modified, gold, single)["recall_at_5"]
                - metrics(base, gold, single)["recall_at_5"]
            ),
            "multi_gold": (
                metrics(modified, gold, multi)["recall_at_5"]
                - metrics(base, gold, multi)["recall_at_5"]
            ),
            "blocks": block_delta,
        },
        "paired_actions": po,
        "promotion_gates": gates,
        "verdict": verdict,
    }
    write_json(out / "CAL_REPORT.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--pair-microbatch", type=int, default=2
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    if str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))
    if str(sibling / "src") not in sys.path:
        sys.path.insert(0, str(sibling / "src"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not sibling.is_dir():
        raise FileNotFoundError(
            f"Sibling LegalIR repo not found: {sibling}"
        )

    out = (
        root
        / "results/manual/"
        "huy_noncal_trainable_ce_boundary_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading CAL ids LABEL-FREE...", flush=True)
    cal_ids_list, cal_questions = get_cal_ids_label_free(root)
    cal_ids = set(cal_ids_list)

    print(
        "[2/6] Building sanitized Strict-V2 nonCAL world...",
        flush=True,
    )
    world = load_noncal_world(
        root, sibling, cal_ids
    )
    print(
        f"  all V2=6991 | CAL excluded={len(cal_ids)} | "
        f"nonCAL={len(world['noncal'])}",
        flush=True,
    )

    # Strong question-identity assertion; still no CAL labels.
    for q in cal_ids_list:
        if world["questions"][q] != cal_questions[q]:
            raise RuntimeError(
                f"CAL question identity mismatch qid={q}"
            )

    print(
        "[3/6] 5-fold OOF trainable CE on NONCAL only...",
        flush=True,
    )
    oof = run_oof(
        root=root,
        sibling=sibling,
        world=world,
        out=out,
        pair_microbatch=args.pair_microbatch,
    )

    p = oof["paired_actions"]
    print("=" * 96)
    print(
        f"NONCAL OOF {oof['baseline']['recall_at_5']:.10f} "
        f"-> {oof['modified']['recall_at_5']:.10f} "
        f"({oof['delta_recall_at_5']:+.10f})"
    )
    print(
        f"Actions {p['actions']} | "
        f"beneficial={p['beneficial']} "
        f"harmful={p['harmful']} "
        f"neutral={p['neutral']} | "
        f"precision={p['intervention_precision_excluding_neutral']:.3f}"
    )
    print(
        f"OOF gate={oof['gate_pass']} "
        f"{oof['gate_checks']}"
    )
    print("=" * 96)

    if not oof["gate_pass"]:
        print(
            "[4/6] STOP: nonCAL OOF gate failed. "
            "No final model; CAL gold NOT loaded."
        )
        print(
            "Verdict : KILL_AT_NONCAL_OOF_GATE"
        )
        print(
            f"Report  : {out / 'NONCAL_OOF_REPORT.json'}"
        )
        return

    print(
        "[4/6] OOF PASS. Training final CE on all NONCAL...",
        flush=True,
    )
    cal = final_fit_and_cal(
        root=root,
        sibling=sibling,
        world=world,
        out=out,
        pair_microbatch=args.pair_microbatch,
    )

    print(
        "[5/6] CAL actions were sealed before gold reveal.",
        flush=True,
    )
    print("[6/6] DONE")
    print("=" * 96)
    print(
        f"D1       R@5={cal['baseline']['recall_at_5']:.10f} "
        f"P@5={cal['baseline']['precision_at_5']:.10f}"
    )
    print(
        f"TrainCE  R@5={cal['modified']['recall_at_5']:.10f} "
        f"P@5={cal['modified']['precision_at_5']:.10f}"
    )
    print(
        f"Delta    R={cal['delta']['recall_at_5']:+.10f} "
        f"P={cal['delta']['precision_at_5']:+.10f}"
    )
    print(
        f"Single Δ {cal['delta']['single_gold']:+.10f} | "
        f"Multi Δ {cal['delta']['multi_gold']:+.10f}"
    )
    print(
        "Blocks   "
        + " ".join(
            f"{b}:{d:+.6f}"
            for b, d in sorted(
                cal["delta"]["blocks"].items()
            )
        )
    )
    p = cal["paired_actions"]
    print(
        f"Actions  {p['actions']} | "
        f"beneficial={p['beneficial']} "
        f"harmful={p['harmful']} "
        f"neutral={p['neutral']}"
    )
    print(f"Verdict  {cal['verdict']}")
    print(f"Report   {out / 'CAL_REPORT.json'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
