#!/usr/bin/env python
"""
EXPORT EXACT PRIVATE D1 CANDIDATE WORKLOAD FOR PRISM
====================================================

CPU-only and cache-only. NEVER runs neural inference.

Reconstructs the exact private D1 candidate pool from completed v14 artifacts:
- private retrieval cache
- historical CPU Top20
- Stage-5 expansion scores
- Stage-5 corpus rank cap32
- Stage-5 rerank scores (coverage guard only)

Writes:
  results/manual/huy_private_prism_v1/PRISM_PRIVATE_WORKLOAD.jsonl

Each row:
  qid, question, candidate_doc_ids

Required external Prism score output:
  pickle dict {qid: {doc_id: float_score}}
covering every qid/doc pair in the workload.
"""

from __future__ import annotations
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pickle(path: Path):
    return pickle.loads(path.read_bytes())


def load_questions(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(q) for q in raw]
    questions = {
        str(q): str(v["question"] if isinstance(v, dict) else v)
        for q, v in raw.items()
    }
    return ids, questions


def unwrap_scores(obj):
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def reconstruct_candidates(root: Path, ids, retrieval, base, stage5: Path):
    from benchmark_dense_expansion_holdouts import raw_union
    from tune_burst_multistage_posterior import weighted_rrf
    from run_burst_expanded_fusion_submission import (
        EXPANSION_CONFIG, RERANK_CONFIG, CORPUS_CAP, CORPUS_DEPTH,
    )

    exp_path = stage5 / "expansion_scores.pkl"
    corpus_path = stage5 / f"corpus_rank_cap{CORPUS_CAP}.pkl"
    rr_path = stage5 / "rerank_scores.pkl"

    for p in (exp_path, corpus_path, rr_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Required Stage-5 cache missing: {p}. Refusing recompute."
            )

    expansion_scores = unwrap_scores(load_pickle(exp_path))
    corpus_obj = load_pickle(corpus_path)
    rr = load_pickle(rr_path)
    corpus_rank = corpus_obj.get("ranking")
    corpus_score = corpus_obj.get("scores")
    if not isinstance(corpus_rank, dict) or not isinstance(corpus_score, dict):
        raise RuntimeError("Corpus cache missing ranking/scores")

    raw = {
        q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"])
        for q in ids
    }

    missing_exp = [
        q for q in ids
        if q not in expansion_scores
        or any(d not in expansion_scores[q] for d in raw[q])
    ]
    missing_corpus = [
        q for q in ids
        if q not in corpus_rank or q not in corpus_score
    ]
    if missing_exp or missing_corpus:
        raise RuntimeError(
            "Stage-5 incomplete; refusing GPU fallback. "
            f"missing_exp={len(missing_exp)} missing_corpus={len(missing_corpus)}"
        )

    dense_rank = {
        q: sorted(raw[q], key=lambda d: (-expansion_scores[q][d], d))
        for q in ids
    }
    expanded_all = weighted_rrf(
        [raw, dense_rank],
        RERANK_CONFIG["expansion_weights"],
        RERANK_CONFIG["expansion_rrf_k"],
    )
    expanded = {
        q: expanded_all[q][:RERANK_CONFIG["expanded_depth"]]
        for q in ids
    }

    candidates = {
        q: list(dict.fromkeys(
            list(base[q])
            + expanded[q]
            + list(corpus_rank[q][:CORPUS_DEPTH])
        ))
        for q in ids
    }

    for family in ("jina", "dense"):
        if family not in rr:
            raise RuntimeError(f"rerank cache missing family={family}")
        bad = [
            q for q in ids
            if q not in rr[family]
            or any(d not in rr[family][q] for d in candidates[q])
        ]
        if bad:
            raise RuntimeError(
                f"rerank cache incomplete family={family}: {len(bad)} queries"
            )

    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--private-file", default="private-official.json")
    ap.add_argument("--stage5-cache", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    private_path = data / args.private_file
    ids, questions = load_questions(private_path)
    if len(ids) != 2080:
        raise RuntimeError(f"Expected 2080 private queries, got {len(ids)}")

    exact_root = root / "results/manual/huy_private_d1_rel_l0_exact_v1"
    cache = exact_root / "cache"
    retrieval_path = cache / "private_retrieval.pkl"
    base_path = cache / "private_base_top20_historical_exact.pkl"
    d1_path = exact_root / "D1_PRIVATE_V14_FAST.json"

    stage5 = (
        args.stage5_cache.resolve()
        if args.stage5_cache
        else (
            root
            / "results/manual/huy_private_d1_rel_l0_approx_v1/"
            "cache/candidate_generation"
        ).resolve()
    )

    for p in (retrieval_path, base_path, d1_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    if not stage5.is_dir():
        raise FileNotFoundError(stage5)

    retrieval_obj = load_pickle(retrieval_path)
    retrieval = retrieval_obj.get("cache", retrieval_obj)
    base_obj = load_pickle(base_path)
    base = base_obj.get("rankings", base_obj)

    if any(q not in retrieval for q in ids):
        raise RuntimeError("Private retrieval cache incomplete")
    if any(q not in base or len(base[q]) < 20 for q in ids):
        raise RuntimeError("Private CPU Top20 cache incomplete")

    print("[1/3] Reconstructing exact private candidate pool from caches only...")
    candidates = reconstruct_candidates(root, ids, retrieval, base, stage5)
    sizes = np.asarray([len(candidates[q]) for q in ids])

    print(
        f"  q={len(ids)} min={sizes.min()} mean={sizes.mean():.2f} "
        f"p50={np.percentile(sizes,50):.0f} "
        f"p95={np.percentile(sizes,95):.0f} max={sizes.max()}"
    )

    print("[2/3] Checking v14 Top5 containment...")
    d1 = json.loads(d1_path.read_text(encoding="utf-8"))
    bad = []
    for q in ids:
        top5 = [str(d) for d in d1[q]["answer"]]
        if not set(top5) <= set(candidates[q]):
            bad.append(q)
    if bad:
        raise RuntimeError(
            f"Candidate pool misses v14 Top5 for {len(bad)} queries"
        )

    out = root / "results/manual/huy_private_prism_v1"
    out.mkdir(parents=True, exist_ok=True)
    workload = out / "PRISM_PRIVATE_WORKLOAD.jsonl"

    print("[3/3] Writing Prism workload...")
    with workload.open("w", encoding="utf-8") as f:
        for q in ids:
            f.write(json.dumps({
                "qid": q,
                "question": questions[q],
                "candidate_doc_ids": [str(d) for d in candidates[q]],
            }, ensure_ascii=False) + "\n")

    meta = {
        "schema": "manual.prism_private_workload.v1",
        "private_file": str(private_path),
        "private_file_sha256": sha256(private_path),
        "queries": len(ids),
        "candidate_stats": {
            "min": int(sizes.min()),
            "mean": float(sizes.mean()),
            "p50": float(np.percentile(sizes, 50)),
            "p90": float(np.percentile(sizes, 90)),
            "p95": float(np.percentile(sizes, 95)),
            "p99": float(np.percentile(sizes, 99)),
            "max": int(sizes.max()),
            "pairs": int(sizes.sum()),
        },
        "source_artifacts": {
            "retrieval": str(retrieval_path),
            "retrieval_sha256": sha256(retrieval_path),
            "base_top20": str(base_path),
            "base_top20_sha256": sha256(base_path),
            "stage5": str(stage5),
            "v14_submission": str(d1_path),
            "v14_submission_sha256": sha256(d1_path),
        },
        "workload": str(workload),
        "workload_sha256": sha256(workload),
        "required_output_contract": (
            "pickle dict {qid: {doc_id: float_score}} covering every "
            "candidate_doc_id in the workload"
        ),
    }
    meta_path = out / "PRISM_PRIVATE_WORKLOAD_META.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("WORKLOAD:", workload)
    print("PAIRS:", int(sizes.sum()))
    print("META:", meta_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
