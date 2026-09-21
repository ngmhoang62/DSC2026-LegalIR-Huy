#!/usr/bin/env python
"""
HUY_D1_TITLE_DPP_ONE_SWAP_V1
============================

Set-level diversification experiment for exact D1.

Motivation
----------
D1 is a pointwise ranker: every document is scored independently. Recall@5,
however, is a SET metric, and the weakest region is multi-gold queries. The
hypothesis is that rank-5 sometimes spends a slot on a document whose legal
topic is redundant with ranks 1-4, while rank 6-20 contains a relevant but
topically distinct document.

This experiment uses a k-DPP-inspired, parameter-free local objective.

For a 5-document set S:
    objective(S) = sum_{d in S} D1_decision_score(d)
                   + log det(G_S + eps I)

where G_S is the cosine Gram matrix of L2-normalized Vietnamese title
embeddings. This follows a DPP kernel L = diag(q) G diag(q) with
q_d = exp(D1_score(d)/2). Since all compared sets have size 5, subtracting a
constant from D1 scores does not affect the comparison.

Protocol
--------
* Exact D1 ranks 1-4 are immutable.
* Defender = exact D1 rank 5.
* Challengers = exact D1 ranks 6-20.
* For each challenger, compare objective(top4 + challenger) to
  objective(exact top5).
* If no challenger strictly improves the objective -> KEEP.
* Otherwise choose the challenger with largest objective delta.
* Maximum one swap.
* No learned parameter, no tuned lambda, no QID rule, no query-type rule.
* Actions are written + hashed BEFORE held-query gold utility is evaluated.

Document representation
-----------------------
Use the already-present AITeamVN Vietnamese Embedding model, CLS+L2, on the
document's extracted legal title. If title extraction fails, use the first 80
whitespace tokens of the document as a deterministic header fallback.

Dependency
----------
Keep this script beside:
    run_d1_query_bootstrap_bagging_v1.py

Run from Git Bash:
    python ../run_d1_title_dpp_one_swap_v1.py \
      --repo-root /d/Study/DSC2026/sota

Uses CUDA when available; otherwise falls back to CPU for one-time title embedding. Cached embeddings are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np


TOP_DEPTH = 20
SET_SIZE = 5
NUMERICAL_EPS = 1e-6
EXPECTED_D1_R5 = 0.9569444444444444


def load_base(script_dir: Path):
    path = script_dir / "run_d1_query_bootstrap_bagging_v1.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dependency: {path}\n"
            "Keep run_d1_query_bootstrap_bagging_v1.py beside this script."
        )
    spec = importlib.util.spec_from_file_location("d1_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_rank(groups: List[str], scores: np.ndarray) -> List[str]:
    order = np.argsort(-np.asarray(scores))
    return [groups[i] for i in order]


def normalize_vectors(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return x / norms


def dpp_logdet(vectors: np.ndarray) -> float:
    """
    Numerical log-det of cosine Gram matrix with tiny jitter.
    The jitter is solely for numerical stability and is fixed globally.
    """
    V = normalize_vectors(vectors)
    gram = V @ V.T
    gram = (gram + gram.T) * 0.5
    gram = gram + NUMERICAL_EPS * np.eye(len(V), dtype=np.float64)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0 or not np.isfinite(logdet):
        # Should be exceptionally rare for a Gram+jitter matrix.
        # Use eigenvalue clipping as deterministic numerical fallback.
        eig = np.linalg.eigvalsh(gram)
        eig = np.maximum(eig, NUMERICAL_EPS)
        logdet = float(np.log(eig).sum())
    return float(logdet)


def set_objective(
    docs: List[str],
    score_map: Dict[str, float],
    vector_map: Dict[str, np.ndarray],
) -> Tuple[float, float, float]:
    relevance = float(sum(score_map[d] for d in docs))
    V = np.vstack([vector_map[d] for d in docs])
    diversity = dpp_logdet(V)
    return relevance + diversity, relevance, diversity


def max_pair_similarity(
    docs: List[str],
    vector_map: Dict[str, np.ndarray],
) -> float:
    if len(docs) < 2:
        return 0.0
    V = normalize_vectors(np.vstack([vector_map[d] for d in docs]))
    sim = V @ V.T
    mask = ~np.eye(len(docs), dtype=bool)
    return float(sim[mask].max())


def title_embedding_cache(
    root: Path,
    needed_docs: Set[str],
    out_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Build/reuse title embeddings only for docs needed by D1 top-20.
    Cache semantics include the exact sorted doc-id list fingerprint.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_title_features import extract_title

    doc_ids = sorted(needed_docs)
    fingerprint = hashlib.sha256(
        ("\n".join(doc_ids) + "\n").encode("utf-8")
    ).hexdigest()

    cache_npz = out_dir / "TITLE_VECTORS_TOP20.npz"
    cache_meta = out_dir / "TITLE_VECTORS_TOP20_META.json"

    if cache_npz.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if (
            meta.get("doc_id_fingerprint") == fingerprint
            and meta.get("document_count") == len(doc_ids)
        ):
            saved = np.load(cache_npz, allow_pickle=False)
            ids = [str(x) for x in saved["doc_ids"].tolist()]
            vecs = np.asarray(saved["vectors"], dtype=np.float32)
            if ids == doc_ids and vecs.shape[0] == len(doc_ids):
                print(
                    f"  Loaded cached title embeddings: "
                    f"{len(ids)} docs x {vecs.shape[1]}D",
                    flush=True,
                )
                return (
                    {d: vecs[i] for i, d in enumerate(ids)},
                    {**meta, "cache_reused": True},
                )

    # Device is automatic. CUDA is preferred, but CPU is a valid and
    # scientifically identical fallback for deterministic embedding inference.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    ctx = (
        root
        / "DSC2026-LegalIR-main"
        / "v4_run"
        / "public_test_dataset"
        / "selected-contexts"
    )
    docs = DocumentStore(sorted(ctx.glob("context_*.json")))

    texts: List[str] = []
    extracted_count = 0
    fallback_count = 0

    for d in doc_ids:
        full = docs[d]
        title = extract_title(full)
        if title:
            extracted_count += 1
            text = title
        else:
            fallback_count += 1
            # Deterministic legal-header fallback, not query dependent.
            text = " ".join((full or "").split()[:80]).strip()
            if not text:
                text = "văn bản pháp luật"
        texts.append(text)

    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing title encoder: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = (
        AutoModel.from_pretrained(model_path, dtype=dtype)
        .eval()
        .to(device)
    )

    # Do not reuse benchmark_aiteamvn_holdouts.encode_cls here: that historical
    # helper hard-codes CUDA. This local encoder preserves the same CLS+L2
    # semantics on either CUDA or CPU.
    @torch.inference_mode()
    def encode_texts_portable(
        items: List[str],
        batch_size: int,
        max_length: int = 128,
    ) -> np.ndarray:
        vectors = []
        for start in range(0, len(items), batch_size):
            batch_text = items[start : start + batch_size]
            encoded = tokenizer(
                batch_text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {
                k: v.to(device, non_blocking=(device == "cuda"))
                for k, v in encoded.items()
            }
            cls = model(**encoded).last_hidden_state[:, 0].float()
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            vectors.append(cls.cpu().numpy())

            done = min(start + batch_size, len(items))
            if done % max(batch_size * 10, 1) == 0 or done == len(items):
                print(
                    f"    embedded {done}/{len(items)}",
                    flush=True,
                )
        return np.vstack(vectors)

    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        batch_size = 64
    else:
        device_name = "CPU"
        # Conservative CPU batch to avoid RAM spikes with the ~568M encoder.
        batch_size = 8

    print(
        f"  Encoding {len(doc_ids)} legal titles/headers on "
        f"{device_name} ({dtype})...",
        flush=True,
    )
    started = time.perf_counter()
    vecs = encode_texts_portable(
        texts,
        batch_size=batch_size,
        max_length=128,
    ).astype(np.float32)

    # Local encoder already L2-normalizes; re-normalize defensively.
    vecs = normalize_vectors(vecs).astype(np.float32)

    elapsed = time.perf_counter() - started
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    np.savez_compressed(
        cache_npz,
        doc_ids=np.asarray(doc_ids, dtype="U32"),
        vectors=vecs,
    )
    meta = {
        "schema": "manual.d1_title_dpp_v1.title_vectors",
        "model_path": str(model_path),
        "model_name": "AITeamVN_Vietnamese_Embedding",
        "pooling": "CLS+L2",
        "device": device_name,
        "dtype": str(dtype),
        "batch_size": batch_size,
        "max_length": 128,
        "document_count": len(doc_ids),
        "embedding_dim": int(vecs.shape[1]),
        "title_extracted_count": extracted_count,
        "header_fallback_count": fallback_count,
        "doc_id_fingerprint": fingerprint,
        "cache_npz": str(cache_npz),
        "elapsed_seconds": elapsed,
    }
    json_dump(cache_meta, meta)

    print(
        f"  Embedded {len(doc_ids)} docs in {elapsed:.1f}s "
        f"(title={extracted_count}, fallback={fallback_count})",
        flush=True,
    )
    return {d: vecs[i] for i, d in enumerate(doc_ids)}, {
        **meta,
        "cache_reused": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base = load_base(script_dir)

    root = args.repo_root.resolve()
    if not (root / "tune_corpus_cap32_fusion.py").exists():
        raise RuntimeError(
            f"{root} does not look like the sota repo root. "
            "Use --repo-root /d/Study/DSC2026/sota in Git Bash."
        )

    out = root / "results/manual/huy_d1_title_dpp_one_swap_v1"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Reconstructing exact D1 48D state...", flush=True)
    state = base.reconstruct_feature_state(root)
    blocks = state["blocks"]
    all_ids = state["all_ids"]
    rows = state["rows"]
    groups = state["groups"]
    gold = state["gold"]

    baseline_ranked: Dict[str, List[str]] = {}
    score_maps: Dict[str, Dict[str, float]] = {}

    print("[2/6] Fitting exact D1 LOBO + collecting top-20...", flush=True)
    needed_docs: Set[str] = set()

    for held in sorted(blocks):
        train_ids = [
            q
            for b in sorted(blocks)
            if b != held
            for q in blocks[b]
        ]
        scaler, model = base.fit_exact_d1(state, train_ids)

        for q in blocks[held]:
            score = np.asarray(
                model.decision_function(scaler.transform(rows[q])),
                dtype=np.float64,
            )
            ranking = exact_rank(groups[q], score)
            baseline_ranked[q] = ranking
            score_maps[q] = {
                d: float(score[i])
                for i, d in enumerate(groups[q])
            }
            needed_docs.update(ranking[:TOP_DEPTH])

    base_metrics = base.metrics(
        baseline_ranked,
        gold,
        all_ids,
        blocks,
    )

    parity_errors = []
    if abs(base_metrics["recall_at_5"] - EXPECTED_D1_R5) > 1e-12:
        parity_errors.append(
            f"R@5={base_metrics['recall_at_5']} expected={EXPECTED_D1_R5}"
        )
    for b, expected in base.EXPECTED_BLOCKS.items():
        got = base_metrics["block_recalls"][b]
        if abs(got - expected) > 1e-12:
            parity_errors.append(
                f"Block {b}={got} expected={expected}"
            )
    if parity_errors:
        raise RuntimeError(
            "BLOCKED_D1_PARITY:\n  - " + "\n  - ".join(parity_errors)
        )

    print(
        f"  PASS D1 R@5={base_metrics['recall_at_5']:.12f}; "
        f"unique top20 docs={len(needed_docs)}",
        flush=True,
    )

    print("[3/6] Building/reusing title embeddings...", flush=True)
    vector_map, vector_meta = title_embedding_cache(
        root,
        needed_docs,
        out,
    )

    missing_vectors = needed_docs - set(vector_map)
    if missing_vectors:
        raise RuntimeError(
            f"Missing {len(missing_vectors)} title vectors: "
            f"{sorted(missing_vectors)[:10]}"
        )

    print("[4/6] Building label-free DPP one-swap actions...", flush=True)
    candidate_top5: Dict[str, List[str]] = {}
    action_rows: Dict[str, Any] = {}
    action_count = 0

    for q in all_ids:
        ranking = baseline_ranked[q]
        score_map = score_maps[q]
        top5 = ranking[:SET_SIZE]
        top4 = ranking[:4]
        defender = ranking[4]

        base_obj, base_rel, base_div = set_objective(
            top5,
            score_map,
            vector_map,
        )
        base_max_sim = max_pair_similarity(top5, vector_map)

        trials = []
        for d in ranking[5:TOP_DEPTH]:
            trial_set = top4 + [d]
            obj, rel, div = set_objective(
                trial_set,
                score_map,
                vector_map,
            )
            trials.append({
                "doc_id": d,
                "d1_rank": ranking.index(d) + 1,
                "d1_score": score_map[d],
                "objective": obj,
                "objective_delta": obj - base_obj,
                "relevance_sum": rel,
                "relevance_delta": rel - base_rel,
                "diversity_logdet": div,
                "diversity_delta": div - base_div,
                "max_pair_similarity": max_pair_similarity(
                    trial_set,
                    vector_map,
                ),
            })

        # Strictly objective-improving only.
        improving = [x for x in trials if x["objective_delta"] > 0.0]
        improving.sort(
            key=lambda x: (
                -x["objective_delta"],
                x["d1_rank"],
                x["doc_id"],
            )
        )

        selected = improving[0] if improving else None
        if selected is not None:
            action_count += 1
            new_top5 = top4 + [selected["doc_id"]]
        else:
            new_top5 = list(top5)

        candidate_top5[q] = new_top5
        action_rows[q] = {
            "qid": q,
            "d1_top5": top5,
            "defender": defender,
            "baseline_objective": base_obj,
            "baseline_relevance_sum": base_rel,
            "baseline_diversity_logdet": base_div,
            "baseline_max_pair_similarity": base_max_sim,
            "action": selected is not None,
            "selected": selected,
            "candidate_top5": new_top5,
            "challenger_trials": trials,
        }

    payload = {
        "schema": "manual.d1_title_dpp_one_swap_v1.label_free",
        "rule": {
            "top_depth": TOP_DEPTH,
            "set_size": SET_SIZE,
            "ranks_1_to_4_immutable": True,
            "defender": "exact D1 rank5",
            "challengers": "exact D1 ranks6-20",
            "objective": (
                "sum exact D1 decision scores + "
                "logdet(cosine title Gram + 1e-6 I)"
            ),
            "quality_interpretation": "q_d = exp(D1_score/2)",
            "selection": "largest strictly-positive objective delta",
            "max_swaps_per_query": 1,
            "learned_new_parameters": 0,
            "tuned_lambda": None,
            "gold_used_for_actions": False,
        },
        "title_vector_meta": vector_meta,
        "summary": {
            "queries": len(all_ids),
            "actions": action_count,
        },
        "rows": action_rows,
    }

    actions_path = out / "DPP_ACTIONS_LABEL_FREE.json"
    json_dump(actions_path, payload)
    action_sha = sha256_file(actions_path)
    json_dump(
        out / "LABEL_FREE_SEAL.json",
        {
            "action_sha256": action_sha,
            "actions": action_count,
        },
    )

    print(
        f"  sealed actions={action_count}, "
        f"sha={action_sha[:12]}...",
        flush=True,
    )

    print("[5/6] Evaluating sealed actions against CAL gold...", flush=True)
    candidate_ranked = {
        q: candidate_top5[q]
        for q in all_ids
    }
    cand_metrics = base.metrics(
        candidate_ranked,
        gold,
        all_ids,
        blocks,
    )

    wins = losses = ties = 0
    beneficial = harmful = neutral = 0
    gold_in = gold_out = 0
    changed = []

    single_actions = multi_actions = 0
    single_beneficial = single_harmful = 0
    multi_beneficial = multi_harmful = 0

    for q in all_ids:
        r0 = base_metrics["per_query_recall"][q]
        r1 = cand_metrics["per_query_recall"][q]

        if r1 > r0:
            wins += 1
        elif r1 < r0:
            losses += 1
        else:
            ties += 1

        row = action_rows[q]
        if not row["action"]:
            continue

        is_multi = len(gold[q]) > 1
        if is_multi:
            multi_actions += 1
        else:
            single_actions += 1

        if r1 > r0:
            beneficial += 1
            effect = "BENEFICIAL"
            if is_multi:
                multi_beneficial += 1
            else:
                single_beneficial += 1
        elif r1 < r0:
            harmful += 1
            effect = "HARMFUL"
            if is_multi:
                multi_harmful += 1
            else:
                single_harmful += 1
        else:
            neutral += 1
            effect = "NEUTRAL"

        before = set(row["d1_top5"])
        after = set(row["candidate_top5"])
        gold_in += len((after - before) & gold[q])
        gold_out += len((before - after) & gold[q])

        changed.append({
            "qid": q,
            "gold_count": len(gold[q]),
            "effect": effect,
            "recall_before": r0,
            "recall_after": r1,
            "defender": row["defender"],
            "challenger": row["selected"]["doc_id"],
            "challenger_d1_rank": row["selected"]["d1_rank"],
            "objective_delta": row["selected"]["objective_delta"],
            "relevance_delta": row["selected"]["relevance_delta"],
            "diversity_delta": row["selected"]["diversity_delta"],
            "baseline_max_pair_similarity": row[
                "baseline_max_pair_similarity"
            ],
            "candidate_max_pair_similarity": row["selected"][
                "max_pair_similarity"
            ],
        })

    delta_r = (
        cand_metrics["recall_at_5"]
        - base_metrics["recall_at_5"]
    )
    delta_p = (
        cand_metrics["precision_at_5"]
        - base_metrics["precision_at_5"]
    )
    delta_single = (
        cand_metrics["single_gold_recall_at_5"]
        - base_metrics["single_gold_recall_at_5"]
    )
    delta_multi = (
        cand_metrics["multi_gold_recall_at_5"]
        - base_metrics["multi_gold_recall_at_5"]
    )
    block_delta = {
        b: (
            cand_metrics["block_recalls"][b]
            - base_metrics["block_recalls"][b]
        )
        for b in blocks
    }

    gates = {
        "recall_positive": delta_r > 0,
        "precision_no_decrease": delta_p >= -1e-12,
        "wins_gt_losses": wins > losses,
        "no_block_decrease": all(
            x >= -1e-12 for x in block_delta.values()
        ),
        "multi_recall_no_decrease": delta_multi >= -1e-12,
    }

    if all(gates.values()):
        verdict = (
            "STRONG_PROMOTE_D1_TITLE_DPP_ONE_SWAP_V1"
            if cand_metrics["recall_at_5"] >= 0.96
            else "PROMISING_D1_TITLE_DPP_ONE_SWAP_V1"
        )
    else:
        verdict = "KILL_D1_TITLE_DPP_ONE_SWAP_V1"

    report = {
        "schema": "manual.d1_title_dpp_one_swap_v1.report",
        "label_free_action_sha256": action_sha,
        "baseline": {
            k: v
            for k, v in base_metrics.items()
            if k != "per_query_recall"
        },
        "candidate": {
            k: v
            for k, v in cand_metrics.items()
            if k != "per_query_recall"
        },
        "delta": {
            "recall_at_5": delta_r,
            "precision_at_5": delta_p,
            "single_gold_recall_at_5": delta_single,
            "multi_gold_recall_at_5": delta_multi,
            "blocks": block_delta,
        },
        "actions": {
            "total": action_count,
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
            "single_gold_actions": single_actions,
            "multi_gold_actions": multi_actions,
            "single_gold_beneficial": single_beneficial,
            "single_gold_harmful": single_harmful,
            "multi_gold_beneficial": multi_beneficial,
            "multi_gold_harmful": multi_harmful,
        },
        "promotion_gates": gates,
        "verdict": verdict,
        "changed_actions": changed,
        "title_vector_meta": vector_meta,
    }
    json_dump(out / "FINAL_REPORT.json", report)

    print("[6/6] DONE")
    print("=" * 80)
    print(
        f"D1       R@5={base_metrics['recall_at_5']:.10f} "
        f"P@5={base_metrics['precision_at_5']:.10f}"
    )
    print(
        f"TitleDPP R@5={cand_metrics['recall_at_5']:.10f} "
        f"P@5={cand_metrics['precision_at_5']:.10f}"
    )
    print(
        f"Delta    R={delta_r:+.10f} "
        f"P={delta_p:+.10f}"
    )
    print(
        f"Single   {base_metrics['single_gold_recall_at_5']:.10f} "
        f"-> {cand_metrics['single_gold_recall_at_5']:.10f} "
        f"({delta_single:+.10f})"
    )
    print(
        f"Multi    {base_metrics['multi_gold_recall_at_5']:.10f} "
        f"-> {cand_metrics['multi_gold_recall_at_5']:.10f} "
        f"({delta_multi:+.10f})"
    )
    print(
        "Blocks   "
        + " ".join(
            f"{b}:{block_delta[b]:+.6f}"
            for b in sorted(block_delta)
        )
    )
    print(
        f"Actions  {action_count} | "
        f"beneficial={beneficial} harmful={harmful} neutral={neutral}"
    )
    print(
        f"ByGold   single actions={single_actions} "
        f"(+{single_beneficial}/-{single_harmful}) | "
        f"multi actions={multi_actions} "
        f"(+{multi_beneficial}/-{multi_harmful})"
    )
    print(
        f"W/L/T    {wins}/{losses}/{ties} | "
        f"gold in/out={gold_in}/{gold_out}"
    )
    print(f"Verdict  {verdict}")
    print(f"Report   {out / 'FINAL_REPORT.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
