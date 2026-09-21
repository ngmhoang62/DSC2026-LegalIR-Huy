#!/usr/bin/env python
"""
AUDIT HISTORICAL burst_jf_harrier_maxrecall REPRODUCIBILITY V1
===============================================================

READ-ONLY / CPU-only unless the user later chooses to score private Harrier.

Goals
-----
1) Discover whether the historical Harrier caches/checkpoint still exist locally.
2) Reconstruct the likely historical CAL600 recipe from cached scores.
3) Use feature dimension + Recall@5 fingerprint to identify the correct recipe.
4) Decide whether private reproduction is:
     A. CACHE_ONLY_READY
     B. CHECKPOINT_READY_NEEDS_PRIVATE_SCORING
     C. BLOCKED_MISSING_CHECKPOINT

Historical clue supplied by the project owner
----------------------------------------------
burst_jf_harrier_maxrecall:
  CAL Recall@5 ~ 0.9594
  auxiliary channels: jina_ft + harrier
  feature dimension: 48
  same base: rank+score+doctype+citation+vnlegal_lal+crossenc

The 48D fingerprint strongly suggests:
  - jina_ft is score-only (+2D)
  - harrier is score + one rank view (+4D total vs no Harrier)
because a score-only jina_ft+harrier recipe is only 46D under the current
historical feature builder.

This script tests BOTH interpretations:
  A) JINA_HARRIER_SCORE_ONLY
  B) JINA_HARRIER_SCORE_RANK

No files are modified except the audit report.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_TARGET = 0.9594
NAMES5 = ["base", "expanded", "jina", "dense", "corpus"]


def unwrap(obj):
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_scores(path: Path):
    obj = unwrap(pickle.loads(path.read_bytes()))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported score artifact {path}: {type(obj)}")
    return {
        str(q): {str(d): float(s) for d, s in row.items()}
        for q, row in obj.items()
    }


def align(raw, ids, candidates):
    vals = [v for row in raw.values() for v in row.values()]
    if not vals:
        raise RuntimeError("Empty score artifact")
    floor = float(min(vals))
    out = {
        q: {d: float(raw.get(q, {}).get(d, floor)) for d in candidates[q]}
        for q in ids
    }
    present = sum(
        d in raw.get(q, {})
        for q in ids
        for d in candidates[q]
    )
    total = sum(len(candidates[q]) for q in ids)
    return out, floor, present / max(total, 1)


def pinfo(p: Path):
    if not p.exists():
        return {"exists": False, "path": str(p)}
    if p.is_dir():
        files = [x for x in p.rglob("*") if x.is_file()]
        return {
            "exists": True,
            "path": str(p),
            "type": "dir",
            "files": len(files),
            "size_mb": sum(x.stat().st_size for x in files) / 2**20,
        }
    return {
        "exists": True,
        "path": str(p),
        "type": "file",
        "size_mb": p.stat().st_size / 2**20,
    }


def discover(root: Path):
    exact = {
        "harrier_cv": root / "results/from_drive/harrier_ft_cv.pkl",
        "harrier_public": root / "results/from_drive/harrier_ft_public.pkl",
        "jina_cv": root / "results/from_drive/jina_ft_cv.pkl",
        "jina_public": root / "results/from_drive/jina_ft_public.pkl",
        "base_user_path": root / "models/vietlegal-harrier-0.6b",
        "adapter_user_path": (
            root / "models/from_drive/vietlegal_finetuned_results_HNSW/"
            "vietlegal_finetuned/best_adapter"
        ),
        "adapter_alt1": (
            root / "models/from_drive/vietlegal_finetuned_results_HNSW/"
            "best_adapter"
        ),
        "adapter_current_results1": (
            root / "results/from_drive/vietlegal/"
            "vietlegal_finetuned/best_adapter"
        ),
        "adapter_current_results2": (
            root / "results/from_drive/vietlegal/best_adapter"
        ),
        "private_harrier_guess": root / "results/from_drive/harrier_ft_private.pkl",
    }

    found_names = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if "harrier" in low or "vietlegal_finetuned" in low:
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                rel = str(p)
            found_names.append(rel)

    return {
        "exact": {k: pinfo(v) for k, v in exact.items()},
        "harrier_named_files": sorted(found_names)[:500],
    }


def block_metrics(pred, gold, blocks):
    out = {}
    for b, qs in blocks.items():
        out[b] = float(np.mean([
            len(set(pred[q][:5]) & gold[q]) / max(1, len(gold[q]))
            for q in qs
        ]))
    return out


def evaluate_recipe(
    *,
    name,
    all_ids,
    blocks,
    candidates,
    local_views,
    base_channels,
    jina,
    harrier,
    type_rows,
    cite_rows,
    gold,
    harrier_rank_view,
):
    from tune_expanded_fusion_selection import ltr_features

    channels = dict(base_channels)
    channels["jina_ft"] = jina
    channels["harrier_ft"] = harrier

    views = dict(local_views)
    view_names = list(NAMES5)
    if harrier_rank_view:
        views["harrier_ft"] = {
            q: sorted(
                candidates[q],
                key=lambda d: (-harrier[q][d], d),
            )
            for q in all_ids
        }
        view_names.append("harrier_ft")

    rows0, groups = ltr_features(
        views, view_names, candidates, all_ids, channels
    )
    rows = {
        q: np.concatenate([rows0[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }
    dim = int(rows[all_ids[0]].shape[1])

    pred = {}
    per = {}
    for held in sorted(blocks):
        train = sum(
            (list(blocks[b]) for b in sorted(blocks) if b != held),
            [],
        )
        test = list(blocks[held])

        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X), y)

        for q in test:
            s = model.decision_function(scaler.transform(rows[q]))
            order = np.argsort(-s)
            pred[q] = [groups[q][i] for i in order[:5]]

    vals = []
    for q in all_ids:
        r = len(set(pred[q]) & gold[q]) / max(1, len(gold[q]))
        per[q] = r
        vals.append(r)

    recall = float(np.mean(vals))
    precision = float(np.mean([
        len(set(pred[q]) & gold[q]) / 5.0
        for q in all_ids
    ]))
    blocks_r = block_metrics(pred, gold, blocks)

    return {
        "name": name,
        "feature_dim": dim,
        "recall": recall,
        "precision": precision,
        "target_abs_error": abs(recall - EXPECTED_TARGET),
        "blocks": blocks_r,
        "predictions": pred,
        "per_query_recall": per,
        "view_names": view_names,
        "score_channels": sorted(channels),
    }


def compact(r):
    return {
        k: v for k, v in r.items()
        if k not in ("predictions", "per_query_recall")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    print("[1/5] Discovering historical Harrier artifacts/checkpoint...", flush=True)
    discovery = discover(root)
    for k, v in discovery["exact"].items():
        print(
            f"  {k:28s} {'FOUND' if v['exists'] else 'MISSING'} "
            f"{v['path']}",
            flush=True,
        )

    harrier_cv_path = root / "results/from_drive/harrier_ft_cv.pkl"
    jina_cv_path = root / "results/from_drive/jina_ft_cv.pkl"

    can_cal = harrier_cv_path.is_file() and jina_cv_path.is_file()
    if not can_cal:
        print(
            "\n[STOP-CAL] Need both historical CV caches to fingerprint recipe:",
            flush=True,
        )
        print("  ", harrier_cv_path, flush=True)
        print("  ", jina_cv_path, flush=True)

        adapters = [
            v for k, v in discovery["exact"].items()
            if k.startswith("adapter_") and v["exists"]
        ]
        base = discovery["exact"]["base_user_path"]["exists"]
        status = (
            "CHECKPOINT_READY_NEEDS_HARRIER_CV_RESCORING"
            if base and adapters
            else "BLOCKED_MISSING_HISTORICAL_CV_CACHE_AND_OR_CHECKPOINT"
        )
        report = {
            "schema": "manual.audit_jf_harrier_repro_preflight_v1",
            "status": status,
            "discovery": discovery,
            "cal_reproduction": None,
        }
        out = root / "results/manual/huy_jf_harrier_repro_preflight_v1"
        out.mkdir(parents=True, exist_ok=True)
        rp = out / "REPORT.json"
        rp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("STATUS:", status)
        print("REPORT:", rp)
        return

    print("[2/5] Loading authoritative historical CAL600 feature world...", flush=True)
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    (
        _queries,
        blocks,
        all_ids,
        candidates,
        local_views,
        full_channels,
        gold,
        _vnlegal,
        type_rows,
        cite_rows,
    ) = load_cal_inputs()

    # IMPORTANT: authoritative D1 full_channels has the three modern auxiliary
    # channels. Historical jf_harrier instead keeps the SAME underlying base
    # score stack but removes aiteamvn_ft/title_embed and supplies
    # jina_ft + harrier.
    historical_base = {
        k: v for k, v in full_channels.items()
        if k not in ("aiteamvn_ft", "jina_ft", "title_embed")
    }

    print(
        "  historical base score channels:",
        ", ".join(sorted(historical_base)),
        flush=True,
    )

    raw_jina = load_scores(jina_cv_path)
    raw_harrier = load_scores(harrier_cv_path)
    jina, jina_floor, jina_cov = align(
        raw_jina, all_ids, candidates
    )
    harrier, harrier_floor, harrier_cov = align(
        raw_harrier, all_ids, candidates
    )
    print(
        f"  jina_ft coverage={jina_cov:.2%} floor={jina_floor:.6f}",
        flush=True,
    )
    print(
        f"  harrier coverage={harrier_cov:.2%} floor={harrier_floor:.6f}",
        flush=True,
    )

    print("[3/5] Testing likely historical recipes...", flush=True)
    score_only = evaluate_recipe(
        name="JINA_HARRIER_SCORE_ONLY",
        all_ids=all_ids,
        blocks=blocks,
        candidates=candidates,
        local_views=local_views,
        base_channels=historical_base,
        jina=jina,
        harrier=harrier,
        type_rows=type_rows,
        cite_rows=cite_rows,
        gold=gold,
        harrier_rank_view=False,
    )
    score_rank = evaluate_recipe(
        name="JINA_HARRIER_SCORE_RANK",
        all_ids=all_ids,
        blocks=blocks,
        candidates=candidates,
        local_views=local_views,
        base_channels=historical_base,
        jina=jina,
        harrier=harrier,
        type_rows=type_rows,
        cite_rows=cite_rows,
        gold=gold,
        harrier_rank_view=True,
    )

    for r in (score_only, score_rank):
        print(
            f"  {r['name']:26s} dim={r['feature_dim']:2d} "
            f"R={r['recall']:.10f} P={r['precision']:.10f} "
            f"|target-0.9594|={r['target_abs_error']:.6f} "
            f"blocks={r['blocks']}",
            flush=True,
        )

    candidates_recipe = [score_only, score_rank]
    exact_dim48 = [r for r in candidates_recipe if r["feature_dim"] == 48]
    pool = exact_dim48 or candidates_recipe
    best = min(pool, key=lambda r: r["target_abs_error"])

    fingerprint_match = bool(
        best["feature_dim"] == 48
        and best["target_abs_error"] <= 0.0010
    )

    print("[4/5] Auditing private readiness...", flush=True)
    # Search for any already-computed private Harrier score cache.
    private_hits = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if "harrier" in low and "private" in low and p.suffix.lower() in (
            ".pkl", ".pickle", ".json"
        ):
            private_hits.append(p)

    adapter_paths = [
        root / "models/from_drive/vietlegal_finetuned_results_HNSW/"
               "vietlegal_finetuned/best_adapter",
        root / "models/from_drive/vietlegal_finetuned_results_HNSW/"
               "best_adapter",
        root / "results/from_drive/vietlegal/"
               "vietlegal_finetuned/best_adapter",
        root / "results/from_drive/vietlegal/best_adapter",
    ]
    adapter_ready = any(
        p.is_dir()
        and (p / "adapter_config.json").is_file()
        and (
            (p / "adapter_model.safetensors").is_file()
            or (p / "adapter_model.bin").is_file()
        )
        for p in adapter_paths
    )
    base_paths = [
        root / "models/vietlegal-harrier-0.6b",
        root / "models/mainguyen9_vietlegal-harrier-0.6b",
    ]
    base_ready = any(
        p.is_dir()
        and (
            any(p.glob("*.safetensors"))
            or any(p.glob("*.bin"))
        )
        for p in base_paths
    )

    if private_hits:
        private_status = "PRIVATE_HARRIER_CACHE_FOUND"
    elif adapter_ready:
        # Base can also be downloaded from HF if it is absent locally.
        private_status = (
            "CHECKPOINT_READY_NEEDS_PRIVATE_CANDIDATE_SCORING"
            if base_ready
            else "ADAPTER_READY_BASE_MODEL_MISSING_LOCALLY"
        )
    else:
        private_status = "BLOCKED_MISSING_HARRIER_ADAPTER"

    print(f"  adapter_ready={adapter_ready}", flush=True)
    print(f"  base_model_ready={base_ready}", flush=True)
    print(f"  private Harrier candidate caches found={len(private_hits)}", flush=True)
    for p in private_hits[:20]:
        print("   ", p, flush=True)
    print("  PRIVATE STATUS:", private_status, flush=True)

    print("[5/5] Writing report...", flush=True)
    status = (
        "HISTORICAL_RECIPE_REPRODUCED"
        if fingerprint_match
        else "HISTORICAL_RECIPE_NOT_EXACTLY_REPRODUCED"
    )

    report = {
        "schema": "manual.audit_jf_harrier_repro_preflight_v1",
        "status": status,
        "target": {
            "name": "burst_jf_harrier_maxrecall",
            "reported_recall": EXPECTED_TARGET,
            "reported_feature_dim": 48,
            "reported_aux_channels": ["jina_ft", "harrier"],
        },
        "discovery": discovery,
        "cal_cache": {
            "harrier_cv": str(harrier_cv_path),
            "jina_cv": str(jina_cv_path),
            "jina_coverage": jina_cov,
            "harrier_coverage": harrier_cov,
        },
        "recipes": {
            score_only["name"]: compact(score_only),
            score_rank["name"]: compact(score_rank),
        },
        "best_fingerprint_candidate": compact(best),
        "fingerprint_match": fingerprint_match,
        "private_readiness": {
            "status": private_status,
            "adapter_ready": adapter_ready,
            "base_model_ready": base_ready,
            "private_cache_hits": [str(p) for p in private_hits],
            "adapter_paths_checked": [str(p) for p in adapter_paths],
            "base_paths_checked": [str(p) for p in base_paths],
        },
        "interpretation": (
            "If the 48D score+rank arm reproduces ~0.9594, the historical "
            "recipe can be reconstructed from caches even though its original "
            "source script predates this repository's Git history."
        ),
    }

    out = root / "results/manual/huy_jf_harrier_repro_preflight_v1"
    out.mkdir(parents=True, exist_ok=True)
    rp = out / "REPORT.json"
    rp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 112)
    print("STATUS:", status)
    print(
        "BEST:",
        best["name"],
        f"dim={best['feature_dim']}",
        f"R={best['recall']:.10f}",
    )
    print("PRIVATE:", private_status)
    print("REPORT:", rp)
    print("=" * 112)


if __name__ == "__main__":
    main()
