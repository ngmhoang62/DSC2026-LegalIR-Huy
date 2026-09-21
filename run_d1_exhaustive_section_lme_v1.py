#!/usr/bin/env python
"""
HUY_D1_EXHAUSTIVE_SECTION_LME_V1
================================

Representation-level experiment for DSC2026 Vietnamese Legal IR.

Hypothesis
----------
The current D1 `jina_ft` channel scores only the top-2 lexically preselected
passages and aggregates them by MAX. The existing structured-section branch
also scores only the top-2 preselected legal sections and aggregates by MAX.

Therefore a relevant "needle" Điều/Mục can be invisible to the cross-encoder,
while a single false-positive section can dominate a whole document.

This experiment changes ONLY the evidence representation of the existing
`jina_ft` score channel:

  1. Parse each exact-D1 candidate document into ALL deterministic legal
     sections/chunks using the existing legal-section parser.
  2. Score EVERY section with the exact frozen Jina-FT checkpoint.
  3. Aggregate ALL section scores with normalized log-mean-exp (LME):
         LME(x_1...x_n) = log(mean(exp(x_i)))
     computed stably.
     If the checkpoint channel is probability-valued [0,1], scores are first
     mapped to logit space, LME is computed there, then mapped back by sigmoid.
     This is inferred from the immutable historical `jina_ft` cache, not labels.
  4. Replace the old `jina_ft` channel with this exhaustive channel.
  5. Re-run the exact same 48D D1 LOBO:
         StandardScaler
         LogisticRegression(C=.15, class_weight="balanced",
                            solver="liblinear", random_state=2026)

No new feature dimensions.
No candidate expansion.
No query/QID rule.
No tuned threshold or lambda.
No selector after D1.
No gold is used by the scorer/aggregation.

The script is resumable. Cross-encoder outputs are cached per query.

Run (Git Bash):
    python ../run_d1_exhaustive_section_lme_v1.py \
      --repo-root /d/Study/DSC2026/sota

Requires CUDA for Jina cross-encoder scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_DIM = 48
EXPECTED_BLOCKS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
AGGREGATION = "ALL_SECTIONS_NORMALIZED_LOGMEANEXP"
MODEL_BATCH_SIZE = 128
OUTER_PAIR_CHUNK = 2048
PROB_EPS = 1e-6


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


def stable_logmeanexp(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        raise ValueError("stable_logmeanexp requires at least one value")
    m = float(np.max(a))
    return float(m + np.log(np.mean(np.exp(a - m))))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def aggregate_section_scores(
    raw_scores: Sequence[float],
    score_semantics: str,
) -> float:
    if not raw_scores:
        raise ValueError("No section scores to aggregate")
    arr = np.asarray(raw_scores, dtype=np.float64)
    if score_semantics == "probability":
        ev = logit(arr)
        return float(sigmoid(stable_logmeanexp(ev)))
    if score_semantics == "raw":
        return stable_logmeanexp(arr)
    raise ValueError(f"Unknown score_semantics={score_semantics}")


def metrics(
    pred: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    all_ids: List[str],
    blocks: Dict[str, List[str]],
) -> Dict[str, Any]:
    per_r = {
        q: len(set(pred[q][:5]) & gold[q]) / len(gold[q])
        for q in all_ids
    }
    per_p = {
        q: len(set(pred[q][:5]) & gold[q]) / 5.0
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


def infer_score_semantics(old_jina: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    vals = np.asarray(
        [float(v) for row in old_jina.values() for v in row.values()],
        dtype=np.float64,
    )
    if vals.size == 0:
        raise RuntimeError("Historical jina_ft channel is empty")
    mn = float(vals.min())
    mx = float(vals.max())
    semantics = "probability" if mn >= 0.0 and mx <= 1.0 else "raw"
    return {
        "semantics": semantics,
        "min": mn,
        "max": mx,
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "count": int(vals.size),
    }


def fingerprint_candidates(
    all_ids: Sequence[str],
    extended: Dict[str, List[str]],
) -> str:
    h = hashlib.sha256()
    for q in all_ids:
        h.update(q.encode("utf-8"))
        h.update(b"\0")
        for d in extended[q]:
            h.update(str(d).encode("utf-8"))
            h.update(b",")
        h.update(b"\n")
    return h.hexdigest()


def build_score_contract(
    root: Path,
    all_ids: Sequence[str],
    extended: Dict[str, List[str]],
    semantics_audit: Dict[str, Any],
) -> Dict[str, Any]:
    weights = root / "models/from_drive/jina_finetuned/model.safetensors"
    parser = (
        root
        / "src/gemini/huy_d1_legal_section_evidence_v1/legal_section_parser.py"
    )
    return {
        "schema": "manual.exhaustive_section_lme_v1.contract",
        "aggregation": AGGREGATION,
        "evidence_selection": "ALL parsed legal sections; no preselection",
        "max_chunk_words": 220,
        "overlap_words": 60,
        "crossencoder_max_length": 512,
        "model_batch_size": MODEL_BATCH_SIZE,
        "outer_pair_chunk": OUTER_PAIR_CHUNK,
        "score_semantics": semantics_audit,
        "candidate_fingerprint": fingerprint_candidates(all_ids, extended),
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights) if weights.exists() else None,
        "parser_path": str(parser),
        "parser_sha256": sha256_file(parser) if parser.exists() else None,
    }


def contract_matches(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = [
        "aggregation",
        "evidence_selection",
        "max_chunk_words",
        "overlap_words",
        "crossencoder_max_length",
        "score_semantics",
        "candidate_fingerprint",
        "weights_sha256",
        "parser_sha256",
    ]
    return all(a.get(k) == b.get(k) for k in keys)


def load_cal_state(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.gemini.huy_d1_legal_section_evidence_v1.common import load_cal_data

    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    ) = load_cal_data()

    blocks = {str(k).upper(): list(v) for k, v in blocks.items()}
    if set(blocks) != {"A", "B", "C", "D"}:
        raise RuntimeError(f"Unexpected CAL block keys: {list(blocks)}")

    return {
        "docs": docs,
        "queries": queries,
        "blocks": blocks,
        "all_ids": list(all_ids),
        "extended": extended,
        "local_views": local_views,
        "channels": full_channels_cv,
        "gold": gold,
        "type_rows": type_rows,
        "cite_rows": cite_rows,
    }


def run_lobo(
    root: Path,
    state: Dict[str, Any],
    channels: Dict[str, Dict[str, Dict[str, float]]],
) -> Tuple[Dict[str, List[str]], Dict[str, Any], int]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tune_expanded_fusion_selection import ltr_features

    rows, groups = ltr_features(
        state["local_views"],
        D1_VIEWS,
        state["extended"],
        state["all_ids"],
        channels,
    )
    for q in state["all_ids"]:
        rows[q] = np.concatenate(
            [rows[q], state["type_rows"][q], state["cite_rows"][q]],
            axis=1,
        )

    dim = int(rows[state["all_ids"][0]].shape[1])
    ranked: Dict[str, List[str]] = {}

    for held in sorted(state["blocks"]):
        train_ids = [
            q
            for b in sorted(state["blocks"])
            if b != held
            for q in state["blocks"][b]
        ]

        X = np.vstack([rows[q] for q in train_ids])
        y = np.concatenate(
            [
                [d in state["gold"][q] for d in groups[q]]
                for q in train_ids
            ]
        ).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X), y)

        for q in state["blocks"][held]:
            score = model.decision_function(scaler.transform(rows[q]))
            order = np.argsort(-np.asarray(score))
            ranked[q] = [groups[q][i] for i in order]

    m = metrics(
        ranked,
        state["gold"],
        state["all_ids"],
        state["blocks"],
    )
    return ranked, m, dim


def assert_baseline_parity(metrics_obj: Dict[str, Any], dim: int) -> None:
    errors = []
    if dim != EXPECTED_D1_DIM:
        errors.append(f"feature_dim={dim}, expected={EXPECTED_D1_DIM}")
    if abs(metrics_obj["recall_at_5"] - EXPECTED_D1_R5) > 1e-12:
        errors.append(
            f"R@5={metrics_obj['recall_at_5']}, expected={EXPECTED_D1_R5}"
        )
    for b, expected in EXPECTED_BLOCKS.items():
        got = metrics_obj["block_recalls"][b]
        if abs(got - expected) > 1e-12:
            errors.append(f"Block {b}={got}, expected={expected}")
    if errors:
        raise RuntimeError(
            "BLOCKED_D1_PARITY:\n  - " + "\n  - ".join(errors)
        )


def score_exhaustive_sections(
    root: Path,
    state: Dict[str, Any],
    out_dir: Path,
    semantics_audit: Dict[str, Any],
    fresh: bool,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for exhaustive Jina cross-encoder scoring. "
            "Verify the CUDA PyTorch wheel inside dsc_env_huy."
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.gemini.huy_d1_legal_section_evidence_v1.common import (
        load_jina_crossencoder,
    )
    from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
        parse_document_into_sections,
    )

    cache_path = out_dir / "EXHAUSTIVE_SECTION_LME_CV.pkl"
    manifest_path = out_dir / "EXHAUSTIVE_SECTION_LME_MANIFEST.json"
    contract = build_score_contract(
        root,
        state["all_ids"],
        state["extended"],
        semantics_audit,
    )

    scores: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}

    if cache_path.exists() and not fresh:
        saved = pickle.loads(cache_path.read_bytes())
        if contract_matches(saved.get("contract", {}), contract):
            scores = saved.get("scores", {})
            diagnostics = saved.get("diagnostics", {})
            print(
                f"  Reusing valid exhaustive cache: {len(scores)}/"
                f"{len(state['all_ids'])} queries",
                flush=True,
            )
        else:
            print("  Existing exhaustive cache contract mismatch; starting fresh.")
    elif fresh and cache_path.exists():
        print("  --fresh requested; ignoring existing exhaustive cache.")

    todo = [
        q
        for q in state["all_ids"]
        if any(d not in scores.get(q, {}) for d in state["extended"][q])
    ]
    if not todo:
        return scores, {
            **contract,
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "queries_scored": len(scores),
            "reused_complete_cache": True,
        }

    model, tokenizer, provenance = load_jina_crossencoder()
    del tokenizer  # compute_score owns/uses model._tokenizer in this implementation.
    print(
        f"  Jina-FT ready on {provenance.get('device', 'cuda')}; "
        f"scoring {len(todo)} queries exhaustively",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    doc_sections_cache: Dict[str, Any] = {}
    score_semantics = semantics_audit["semantics"]

    total_pairs = 0
    total_doc_pairs = 0
    total_sections_seen = 0
    section_count_values: List[int] = []

    for qi, q in enumerate(todo, 1):
        qtext = state["queries"][q][0]
        q_scores = dict(scores.get(q, {}))
        q_diag = dict(diagnostics.get(q, {}))

        owners: List[str] = []
        pair_texts: List[Tuple[str, str]] = []
        section_counts: Dict[str, int] = {}

        for d in state["extended"][q]:
            if d in q_scores:
                continue
            if d not in doc_sections_cache:
                doc_sections_cache[d] = parse_document_into_sections(
                    d,
                    state["docs"][d],
                    max_chunk_words=220,
                    overlap_words=60,
                )
            secs = doc_sections_cache[d]

            if not secs:
                # Defensive fallback for pathological empty text.
                text = state["docs"][d]
                if text:
                    from types import SimpleNamespace
                    secs = [
                        SimpleNamespace(
                            text=text,
                            section_index=0,
                            section_type="FALLBACK_FULL",
                            heading="",
                        )
                    ]
                else:
                    raise RuntimeError(
                        f"Document {d} produced no sections and no fallback text"
                    )

            section_counts[d] = len(secs)
            for sec in secs:
                owners.append(d)
                pair_texts.append((qtext, sec.text))

        raw_all: List[float] = []
        for start in range(0, len(pair_texts), OUTER_PAIR_CHUNK):
            chunk = pair_texts[start : start + OUTER_PAIR_CHUNK]
            raw = model.compute_score(
                chunk,
                batch_size=MODEL_BATCH_SIZE,
                max_length=512,
            )
            if isinstance(raw, (float, int)):
                raw = [float(raw)]
            raw_all.extend(float(x) for x in raw)

        if len(raw_all) != len(owners):
            raise RuntimeError(
                f"Score cardinality mismatch q={q}: "
                f"scores={len(raw_all)} owners={len(owners)}"
            )

        grouped: Dict[str, List[float]] = {}
        for d, s in zip(owners, raw_all):
            grouped.setdefault(d, []).append(s)

        for d, vals in grouped.items():
            agg = aggregate_section_scores(vals, score_semantics)
            q_scores[d] = agg
            arr = np.asarray(vals, dtype=np.float64)
            q_diag[d] = {
                "section_count": len(vals),
                "aggregate_lme": agg,
                "raw_max": float(arr.max()),
                "raw_mean": float(arr.mean()),
                "raw_std": float(arr.std()),
            }
            section_count_values.append(len(vals))

        scores[q] = q_scores
        diagnostics[q] = q_diag

        total_pairs += len(pair_texts)
        total_doc_pairs += len(grouped)
        total_sections_seen += sum(section_counts.values())

        if qi % 5 == 0 or qi == len(todo):
            payload = {
                "contract": contract,
                "scores": scores,
                "diagnostics": diagnostics,
                "progress": {
                    "completed_queries": len(scores),
                    "total_queries": len(state["all_ids"]),
                },
            }
            cache_path.write_bytes(pickle.dumps(payload, protocol=5))

            elapsed = time.perf_counter() - started
            rate = qi / max(elapsed, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(
                f"    [{qi}/{len(todo)}] q={q} "
                f"pairs_this_q={len(pair_texts)} "
                f"rate={rate:.3f} q/s "
                f"peakVRAM={peak:.0f} MiB",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    payload = {
        "contract": contract,
        "scores": scores,
        "diagnostics": diagnostics,
        "progress": {
            "completed_queries": len(scores),
            "total_queries": len(state["all_ids"]),
        },
    }
    cache_path.write_bytes(pickle.dumps(payload, protocol=5))

    # Complete coverage check.
    missing = [
        (q, d)
        for q in state["all_ids"]
        for d in state["extended"][q]
        if d not in scores.get(q, {})
    ]
    if missing:
        raise RuntimeError(
            f"Exhaustive cache missing {len(missing)} query-doc pairs; "
            f"sample={missing[:10]}"
        )

    manifest = {
        **contract,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "queries_scored": len(scores),
        "elapsed_seconds_this_run": elapsed,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "new_query_doc_pairs": total_doc_pairs,
        "new_section_pairs": total_pairs,
        "section_count_summary_this_run": {
            "min": int(min(section_count_values)) if section_count_values else None,
            "median": (
                float(np.median(section_count_values))
                if section_count_values else None
            ),
            "p90": (
                float(np.percentile(section_count_values, 90))
                if section_count_values else None
            ),
            "max": int(max(section_count_values)) if section_count_values else None,
            "mean": (
                float(np.mean(section_count_values))
                if section_count_values else None
            ),
        },
        "model_provenance": provenance,
    }
    json_dump(manifest_path, manifest)
    return scores, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any resumable exhaustive-section cache.",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    if not (root / "tune_corpus_cap32_fusion.py").exists():
        raise RuntimeError(
            f"{root} does not look like the sota repo root. "
            "Use --repo-root /d/Study/DSC2026/sota"
        )

    out = root / "results/manual/huy_d1_exhaustive_section_lme_v1"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading exact D1 CAL state...", flush=True)
    state = load_cal_state(root)

    if "jina_ft" not in state["channels"]:
        raise RuntimeError(
            f"D1 channels do not include jina_ft: {sorted(state['channels'])}"
        )

    semantics = infer_score_semantics(state["channels"]["jina_ft"])
    print(
        "  historical jina_ft score semantics: "
        f"{semantics['semantics']} "
        f"range=[{semantics['min']:.6f}, {semantics['max']:.6f}]",
        flush=True,
    )

    print("[2/6] Reconstructing authoritative D1 baseline...", flush=True)
    base_ranked, base_metrics, base_dim = run_lobo(
        root,
        state,
        state["channels"],
    )
    assert_baseline_parity(base_metrics, base_dim)
    print(
        f"  PASS D1 parity: R@5={base_metrics['recall_at_5']:.12f}, "
        f"dim={base_dim}",
        flush=True,
    )

    print("[3/6] Scoring ALL legal sections with frozen Jina-FT...", flush=True)
    exhaustive_scores, score_manifest = score_exhaustive_sections(
        root,
        state,
        out,
        semantics,
        args.fresh,
    )
    print(
        f"  exhaustive cache ready: "
        f"{len(exhaustive_scores)}/{len(state['all_ids'])} queries",
        flush=True,
    )

    # Seal scorer output BEFORE candidate utility evaluation.
    cache_path = Path(score_manifest["cache_path"])
    seal = {
        "exhaustive_cache_sha256": sha256_file(cache_path),
        "aggregation": AGGREGATION,
        "score_semantics": semantics,
        "candidate_fingerprint": fingerprint_candidates(
            state["all_ids"],
            state["extended"],
        ),
    }
    json_dump(out / "EXHAUSTIVE_SCORE_SEAL.json", seal)
    print(
        f"[4/6] Sealed exhaustive score channel: "
        f"{seal['exhaustive_cache_sha256'][:12]}...",
        flush=True,
    )

    print("[5/6] Re-running exact 48D D1 with jina_ft channel replaced...", flush=True)
    candidate_channels = dict(state["channels"])
    candidate_channels["jina_ft"] = exhaustive_scores

    cand_ranked, cand_metrics, cand_dim = run_lobo(
        root,
        state,
        candidate_channels,
    )
    if cand_dim != EXPECTED_D1_DIM:
        raise RuntimeError(
            f"Feature dimension drift: candidate={cand_dim}, expected={EXPECTED_D1_DIM}"
        )

    # Gold utility summary.
    wins = losses = ties = 0
    gold_in = gold_out = 0
    top5_set_churn = 0
    details = []

    for q in state["all_ids"]:
        r0 = base_metrics["per_query_recall"][q]
        r1 = cand_metrics["per_query_recall"][q]
        if r1 > r0:
            wins += 1
            effect = "WIN"
        elif r1 < r0:
            losses += 1
            effect = "LOSS"
        else:
            ties += 1
            effect = "TIE"

        b0 = base_ranked[q][:5]
        b1 = cand_ranked[q][:5]
        if set(b0) != set(b1):
            top5_set_churn += 1
            gold_in += len((set(b1) - set(b0)) & state["gold"][q])
            gold_out += len((set(b0) - set(b1)) & state["gold"][q])
            details.append(
                {
                    "qid": q,
                    "gold_count": len(state["gold"][q]),
                    "effect": effect,
                    "recall_before": r0,
                    "recall_after": r1,
                    "d1_top5": b0,
                    "exhaustive_top5": b1,
                }
            )

    delta_r = cand_metrics["recall_at_5"] - base_metrics["recall_at_5"]
    delta_p = cand_metrics["precision_at_5"] - base_metrics["precision_at_5"]
    delta_single = (
        cand_metrics["single_gold_recall_at_5"]
        - base_metrics["single_gold_recall_at_5"]
    )
    delta_multi = (
        cand_metrics["multi_gold_recall_at_5"]
        - base_metrics["multi_gold_recall_at_5"]
    )
    block_delta = {
        b: cand_metrics["block_recalls"][b] - base_metrics["block_recalls"][b]
        for b in EXPECTED_BLOCKS
    }

    gates = {
        "recall_positive": delta_r > 0,
        "precision_no_decrease": delta_p >= -1e-12,
        "wins_gt_losses": wins > losses,
        "no_block_decrease": all(v >= -1e-12 for v in block_delta.values()),
        "single_no_decrease": delta_single >= -1e-12,
        "multi_no_decrease": delta_multi >= -1e-12,
    }
    if all(gates.values()):
        verdict = (
            "STRONG_PROMOTE_D1_EXHAUSTIVE_SECTION_LME_V1"
            if cand_metrics["recall_at_5"] >= 0.96
            else "PROMISING_D1_EXHAUSTIVE_SECTION_LME_V1"
        )
    else:
        verdict = "KILL_D1_EXHAUSTIVE_SECTION_LME_V1"

    report = {
        "schema": "manual.d1_exhaustive_section_lme_v1.report",
        "score_seal": seal,
        "score_manifest": score_manifest,
        "baseline": {
            k: v for k, v in base_metrics.items()
            if k != "per_query_recall"
        },
        "candidate": {
            k: v for k, v in cand_metrics.items()
            if k != "per_query_recall"
        },
        "delta": {
            "recall_at_5": delta_r,
            "precision_at_5": delta_p,
            "single_gold_recall_at_5": delta_single,
            "multi_gold_recall_at_5": delta_multi,
            "blocks": block_delta,
        },
        "paired": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "top5_set_churn": top5_set_churn,
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
        },
        "promotion_gates": gates,
        "verdict": verdict,
        "changed_top5_details": details,
    }
    json_dump(out / "FINAL_REPORT.json", report)

    print("[6/6] DONE")
    print("=" * 82)
    print(
        f"D1         R@5={base_metrics['recall_at_5']:.10f} "
        f"P@5={base_metrics['precision_at_5']:.10f}"
    )
    print(
        f"Exhaustive R@5={cand_metrics['recall_at_5']:.10f} "
        f"P@5={cand_metrics['precision_at_5']:.10f}"
    )
    print(
        f"Delta      R={delta_r:+.10f} "
        f"P={delta_p:+.10f}"
    )
    print(
        f"Single     {base_metrics['single_gold_recall_at_5']:.10f} "
        f"-> {cand_metrics['single_gold_recall_at_5']:.10f} "
        f"({delta_single:+.10f})"
    )
    print(
        f"Multi      {base_metrics['multi_gold_recall_at_5']:.10f} "
        f"-> {cand_metrics['multi_gold_recall_at_5']:.10f} "
        f"({delta_multi:+.10f})"
    )
    print(
        "Blocks     "
        + " ".join(
            f"{b}:{block_delta[b]:+.6f}"
            for b in sorted(block_delta)
        )
    )
    print(
        f"W/L/T      {wins}/{losses}/{ties} | "
        f"set churn={top5_set_churn} | "
        f"gold in/out={gold_in}/{gold_out}"
    )
    print(f"Verdict    {verdict}")
    print(f"Report     {out / 'FINAL_REPORT.json'}")
    print("=" * 82)


if __name__ == "__main__":
    main()
