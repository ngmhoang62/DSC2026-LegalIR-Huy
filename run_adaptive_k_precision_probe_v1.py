#!/usr/bin/env python
"""
Adaptive-K Precision Probe V1
=============================

CPU-only diagnostic for DSC2026 LegalIR.

Goal:
  Increase precision by pruning ONLY D1 rank-5 when it is judged very likely
  non-relevant, while preserving recall exactly.

Protocol:
  * Reconstruct exact D1 CAL600 4-block LOBO rankings + decision scores.
  * For each held block, calibrate a pruning rule ONLY on the other 3 blocks'
    already-OOF D1 outputs.
  * Calibration has a hard constraint: zero removed golds on development data.
  * Apply the frozen threshold/rule to the held block.
  * Aggregate 4 held blocks -> fully OOF adaptive-K evaluation.

Three pre-specified rule families:
  1) ABS_SCORE: prune rank5 if its D1 decision score <= threshold.
  2) GAP_4_5: prune rank5 if (score4 - score5) >= threshold.
  3) SUPPORT_TOP5: prune rank5 if it appears in <= N of the five D1 views' top5.

No GPU. No external model. No public labels.

Run:
  python ../run_adaptive_k_precision_probe_v1.py --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_P5 = 0.20566666666666666
SEED = 2026


def load_pkl(root: Path, rel_path: str):
    p = root / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_inputs(root: Path):
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    docs = DocumentStore(
        sorted(
            (
                root
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )

    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            root,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: set(map(str, queries[q][1])) for q in all_ids}

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(root, rel_path)
        fl = floor if floor is not None else min(
            v for q in obj for v in obj[q].values()
        )
        return {
            q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]}
            for q in all_ids
        }

    # Keep the exact D1 score-only vnlegal channel.
    vnlegal_path_a = root / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"
    vnlegal_path_b = root / "results/embedding_finetunc/vnlegal_lal_cv_scores.pkl"
    if vnlegal_path_a.is_file():
        vnlegal_cv = load_pkl(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    elif vnlegal_path_b.is_file():
        vnlegal_cv = load_pkl(root, "results/embedding_finetunc/vnlegal_lal_cv_scores.pkl")
    else:
        raise FileNotFoundError("vnlegal_lal_cv_scores.pkl not found")

    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)

    extra_paths = {
        "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
        "jina_ft": "results/from_drive/jina_ft_cv.pkl",
        "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
    }
    extra_cv = {name: load_aligned(rel) for name, rel in extra_paths.items()}

    full_channels = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(root, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    return (
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels,
        gold,
        type_rows,
        cite_rows,
    )


def compute_exact_d1_lobo(
    blocks,
    all_ids,
    extended,
    local_views,
    full_channels,
    gold,
    type_rows,
    cite_rows,
):
    from tune_expanded_fusion_selection import ltr_features

    rankings: Dict[str, List[str]] = {}
    scoremaps: Dict[str, Dict[str, float]] = {}

    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        eval_ids = train + blocks[held]

        rows, groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, full_channels
        )
        for q in rows:
            rows[q] = np.concatenate(
                [rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        X = np.vstack([rows[q] for q in train])
        y = np.concatenate(
            [[d in gold[q] for d in groups[q]] for q in train]
        ).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=SEED,
        )
        model.fit(scaler.transform(X), y)

        for q in blocks[held]:
            s = model.decision_function(scaler.transform(rows[q]))
            sm = {str(d): float(v) for d, v in zip(groups[q], s)}
            order = sorted(groups[q], key=lambda d: (-sm[d], str(d)))
            rankings[q] = [str(d) for d in order]
            scoremaps[q] = sm

    return rankings, scoremaps


def metrics(pred: Dict[str, List[str]], gold, ids):
    recalls = []
    macro_p = []
    total_hits = 0
    total_ret = 0

    for q in ids:
        docs = pred[q]
        hits = len(set(docs) & gold[q])
        recalls.append(hits / len(gold[q]))
        macro_p.append(hits / len(docs) if docs else 0.0)
        total_hits += hits
        total_ret += len(docs)

    return {
        "recall": float(np.mean(recalls)),
        "macro_precision_variable_k": float(np.mean(macro_p)),
        "micro_precision_variable_k": float(total_hits / total_ret),
        "mean_k": float(np.mean([len(pred[q]) for q in ids])),
        "total_predictions": int(total_ret),
        "total_hits": int(total_hits),
    }


def support_top5(local_views, q, doc):
    return sum(doc in local_views[v][q][:5] for v in D1_VIEWS)


def feature_value(family, q, rankings, scoremaps, local_views):
    top = rankings[q][:5]
    d4, d5 = top[3], top[4]
    if family == "ABS_SCORE":
        return scoremaps[q][d5]
    if family == "GAP_4_5":
        return scoremaps[q][d4] - scoremaps[q][d5]
    if family == "SUPPORT_TOP5":
        return support_top5(local_views, q, d5)
    raise ValueError(family)


def calibrate_rule(family, dev_ids, rankings, scoremaps, local_views, gold):
    """
    Pick the most aggressive threshold with ZERO dev gold removals.
    Returns a rule dict.
    """
    rows = []
    for q in dev_ids:
        d5 = rankings[q][4]
        x = feature_value(family, q, rankings, scoremaps, local_views)
        y = d5 in gold[q]
        rows.append((q, float(x), bool(y)))

    if family == "ABS_SCORE":
        pos = [x for _, x, y in rows if y]
        if not pos:
            return {"family": family, "enabled": False, "reason": "no positive rank5 on dev"}
        min_pos = min(pos)
        threshold = float(np.nextafter(min_pos, -np.inf))
        # prune if score <= threshold
        def fire(x): return x <= threshold

    elif family == "GAP_4_5":
        pos = [x for _, x, y in rows if y]
        if not pos:
            return {"family": family, "enabled": False, "reason": "no positive rank5 on dev"}
        max_pos = max(pos)
        threshold = float(np.nextafter(max_pos, np.inf))
        # prune if gap >= threshold
        def fire(x): return x >= threshold

    elif family == "SUPPORT_TOP5":
        # integer support 0..5. Choose largest N such that pruning support<=N
        # removes zero gold rank5 on dev.
        safe = []
        for n in range(0, 6):
            removed_gold = sum(1 for _, x, y in rows if x <= n and y)
            if removed_gold == 0:
                safe.append(n)
        if not safe:
            return {"family": family, "enabled": False, "reason": "no safe support threshold"}
        threshold = max(safe)
        def fire(x): return x <= threshold

    else:
        raise ValueError(family)

    dev_actions = [(q, x, y) for q, x, y in rows if fire(x)]
    dev_gold_removed = sum(y for _, _, y in dev_actions)

    return {
        "family": family,
        "enabled": True,
        "threshold": threshold,
        "dev_actions": len(dev_actions),
        "dev_gold_removed": int(dev_gold_removed),
    }


def rule_fires(rule, x):
    if not rule.get("enabled"):
        return False
    fam = rule["family"]
    t = rule["threshold"]
    if fam == "ABS_SCORE":
        return x <= t
    if fam == "GAP_4_5":
        return x >= t
    if fam == "SUPPORT_TOP5":
        return x <= t
    raise ValueError(fam)


def run_family(
    family,
    blocks,
    all_ids,
    rankings,
    scoremaps,
    local_views,
    gold,
):
    pred = {q: list(rankings[q][:5]) for q in all_ids}
    fold_reports = {}
    actions = []

    for held in sorted(blocks):
        dev = sum((blocks[b] for b in blocks if b != held), [])
        test = blocks[held]
        rule = calibrate_rule(
            family, dev, rankings, scoremaps, local_views, gold
        )

        fold_actions = []
        for q in test:
            x = feature_value(
                family, q, rankings, scoremaps, local_views
            )
            if rule_fires(rule, x):
                removed = pred[q][-1]
                was_gold = removed in gold[q]
                pred[q] = pred[q][:-1]
                row = {
                    "qid": q,
                    "removed_doc": removed,
                    "feature": float(x),
                    "removed_was_gold": bool(was_gold),
                }
                fold_actions.append(row)
                actions.append(row)

        fold_reports[held] = {
            "rule": rule,
            "actions": len(fold_actions),
            "gold_removed": sum(a["removed_was_gold"] for a in fold_actions),
            "metrics": metrics(pred, gold, test),
        }

    m = metrics(pred, gold, all_ids)
    removed_gold = sum(a["removed_was_gold"] for a in actions)

    return {
        "family": family,
        "metrics": m,
        "actions": len(actions),
        "gold_removed": int(removed_gold),
        "folds": fold_reports,
        "predictions": pred,
        "action_rows": actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()

    print("[1/4] Loading exact D1 CAL inputs...", flush=True)
    (
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels,
        gold,
        type_rows,
        cite_rows,
    ) = load_inputs(root)

    print("[2/4] Recomputing exact D1 4-block LOBO + decision scores...", flush=True)
    rankings, scoremaps = compute_exact_d1_lobo(
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels,
        gold,
        type_rows,
        cite_rows,
    )

    base_pred = {q: rankings[q][:5] for q in all_ids}
    base = metrics(base_pred, gold, all_ids)
    print("Baseline:", json.dumps(base, indent=2), flush=True)

    if abs(base["recall"] - EXPECTED_D1_R5) > 1e-12:
        raise RuntimeError(
            f"D1 recall parity failed: {base['recall']} != {EXPECTED_D1_R5}"
        )
    if abs(base["macro_precision_variable_k"] - EXPECTED_D1_P5) > 1e-12:
        raise RuntimeError(
            "D1 precision parity failed: "
            f"{base['macro_precision_variable_k']} != {EXPECTED_D1_P5}"
        )

    print("[3/4] Running fully OOF adaptive-K rank5 pruning...", flush=True)
    results = {}
    for family in ("ABS_SCORE", "GAP_4_5", "SUPPORT_TOP5"):
        r = run_family(
            family,
            blocks,
            all_ids,
            rankings,
            scoremaps,
            local_views,
            gold,
        )
        results[family] = r
        m = r["metrics"]
        print(
            f"{family:12s} "
            f"R={m['recall']:.10f} "
            f"Pmacro={m['macro_precision_variable_k']:.10f} "
            f"Pmicro={m['micro_precision_variable_k']:.10f} "
            f"meanK={m['mean_k']:.4f} "
            f"actions={r['actions']} "
            f"gold_removed={r['gold_removed']}",
            flush=True,
        )
        for held, fr in r["folds"].items():
            print(
                f"  {held}: actions={fr['actions']} "
                f"gold_removed={fr['gold_removed']} "
                f"threshold={fr['rule'].get('threshold')} "
                f"R={fr['metrics']['recall']:.10f} "
                f"P={fr['metrics']['macro_precision_variable_k']:.10f}",
                flush=True,
            )

    # Rank candidates: zero gold removals first, exact recall parity, then macro P.
    eligible = [
        r for r in results.values()
        if r["gold_removed"] == 0
        and abs(r["metrics"]["recall"] - base["recall"]) <= 1e-12
    ]
    best = max(
        eligible,
        key=lambda r: r["metrics"]["macro_precision_variable_k"],
        default=None,
    )

    out_dir = root / "results/manual/huy_adaptive_k_precision_probe_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = {
        "schema": "manual.huy_adaptive_k_precision_probe_v1",
        "baseline": base,
        "families": {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk not in ("predictions",)
            }
            for k, v in results.items()
        },
        "best_zero_loss_family": (
            best["family"] if best is not None else None
        ),
        "verdict": (
            "PROMOTE_TO_PUBLIC_MATERIALIZATION_PROBE"
            if best is not None
            and best["metrics"]["macro_precision_variable_k"]
                > base["macro_precision_variable_k"]
            else "NO_SAFE_PRECISION_GAIN"
        ),
    }
    (out_dir / "REPORT.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if best is not None:
        (out_dir / "BEST_OOF_PREDICTIONS.json").write_text(
            json.dumps(best["predictions"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("[4/4] DONE", flush=True)
    print("=" * 88)
    print(
        f"Baseline R={base['recall']:.10f} "
        f"Pmacro={base['macro_precision_variable_k']:.10f}"
    )
    if best is None:
        print("Best safe family: NONE")
        print("Verdict: NO_SAFE_PRECISION_GAIN")
    else:
        bm = best["metrics"]
        print(f"Best safe family: {best['family']}")
        print(
            f"Adaptive R={bm['recall']:.10f} "
            f"Pmacro={bm['macro_precision_variable_k']:.10f} "
            f"Pmicro={bm['micro_precision_variable_k']:.10f} "
            f"meanK={bm['mean_k']:.4f}"
        )
        print(
            f"Delta R={bm['recall']-base['recall']:+.10f} "
            f"Delta Pmacro="
            f"{bm['macro_precision_variable_k']-base['macro_precision_variable_k']:+.10f}"
        )
        print(
            f"Actions={best['actions']} "
            f"gold_removed={best['gold_removed']}"
        )
        print("Verdict: PROMOTE_TO_PUBLIC_MATERIALIZATION_PROBE")
    print(f"Report: {out_dir / 'REPORT.json'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
