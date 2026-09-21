#!/usr/bin/env python
"""
Adaptive-K Precision Probe V2 — Nested Safety Margin
====================================================

CPU-only. Reuses exact D1 CAL600 LOBO reconstruction.

Outer protocol:
  held block is never used to choose the pruning rule.

Inner protocol on the other 3 blocks:
  Choose a safety-margin multiplier lambda for rank5 D1 absolute score.
  For each inner-held dev block:
      threshold = min(gold rank5 score on the other 2 blocks)
                  - lambda * robust_scale(gold rank5 scores on those 2 blocks)
  A lambda is eligible only if it removes ZERO gold rank5 documents across
  ALL inner-held dev blocks.
  Among eligible lambdas, choose the one with most safe pruning actions
  (tie -> larger lambda, i.e. more conservative).
  Refit threshold on all 3 outer-dev blocks, then apply once to outer-held.

This is nested model/rule selection; outer labels are untouched until evaluation.

Run:
  python ../run_adaptive_k_precision_probe_v2.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, List, Set, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SEED = 2026
EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666

# Pre-specified before seeing V2 outer results.
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def load_pkl(root, rel):
    obj = pickle.loads((root / rel).read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_inputs(root: Path):
    sys.path.insert(0, str(root))
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")
    ))
    queries, blocks, ids, extended, views, base_scores = build_training_cap(
        root, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    def aligned(rel, floor=None):
        obj = load_pkl(root, rel)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in ids}

    a = root / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"
    b = root / "results/embedding_finetunc/vnlegal_lal_cv_scores.pkl"
    if a.is_file():
        vnlegal = load_pkl(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    elif b.is_file():
        vnlegal = load_pkl(root, "results/embedding_finetunc/vnlegal_lal_cv_scores.pkl")
    else:
        raise FileNotFoundError("vnlegal_lal_cv_scores.pkl")

    full = {
        **base_scores,
        "vnlegal_lal": vnlegal,
        "crossenc": aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5),
        "aiteamvn_ft": aligned("results/from_drive/aiteamvn_ft_cv.pkl"),
        "jina_ft": aligned("results/from_drive/jina_ft_cv.pkl"),
        "title_embed": aligned("results/burst_fresh_block/title_embed_scores.pkl"),
    }
    tt = build_type_table(root, docs, ids, extended)
    type_rows = type_features(extended, tt, queries, ids)
    own, cited = build_citation_table(docs, ids, extended)
    cite_rows = citation_features(extended, own, cited, ids)
    return queries, blocks, ids, extended, views, full, gold, type_rows, cite_rows


def reconstruct(blocks, ids, extended, views, full, gold, type_rows, cite_rows):
    from tune_expanded_fusion_selection import ltr_features
    rankings, smaps = {}, {}
    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        eval_ids = train + blocks[held]
        rows, groups = ltr_features(views, D1_VIEWS, extended, eval_ids, full)
        for q in rows:
            rows[q] = np.concatenate([rows[q], type_rows[q], cite_rows[q]], axis=1)
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([[d in gold[q] for d in groups[q]] for q in train]).astype(np.int8)
        sc = StandardScaler().fit(X)
        model = LogisticRegression(
            C=0.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=SEED
        ).fit(sc.transform(X), y)
        for q in blocks[held]:
            s = model.decision_function(sc.transform(rows[q]))
            sm = {str(d): float(v) for d, v in zip(groups[q], s)}
            order = sorted(groups[q], key=lambda d: (-sm[d], str(d)))
            rankings[q] = [str(d) for d in order]
            smaps[q] = sm
    return rankings, smaps


def metric(pred, gold, ids):
    rr, pp, hits, nret = [], [], 0, 0
    for q in ids:
        ds = pred[q]
        h = len(set(ds) & gold[q])
        rr.append(h / len(gold[q]))
        pp.append(h / len(ds))
        hits += h
        nret += len(ds)
    return {
        "recall": float(np.mean(rr)),
        "macro_precision": float(np.mean(pp)),
        "micro_precision": hits / nret,
        "mean_k": float(np.mean([len(pred[q]) for q in ids])),
        "hits": hits,
        "returned": nret,
    }


def rank5_score(q, rankings, smaps):
    d = rankings[q][4]
    return smaps[q][d], d, (d in GOLD[q])


def robust_scale(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    std = float(np.std(x))
    # Use the largest robust/spread estimator to avoid a deceptively tiny margin.
    return max(iqr / 1.349 if iqr > 0 else 0.0, mad, std, 1e-6)


def fit_threshold(cal_ids, rankings, smaps, gold, lam):
    positive_scores = []
    for q in cal_ids:
        d5 = rankings[q][4]
        if d5 in gold[q]:
            positive_scores.append(smaps[q][d5])
    if not positive_scores:
        return None
    return min(positive_scores) - lam * robust_scale(positive_scores)


def evaluate_threshold(ids, threshold, rankings, smaps, gold):
    actions, losses = 0, 0
    for q in ids:
        d5 = rankings[q][4]
        if smaps[q][d5] <= threshold:
            actions += 1
            losses += int(d5 in gold[q])
    return actions, losses


def choose_lambda(inner_blocks, rankings, smaps, gold):
    names = sorted(inner_blocks)
    candidates = []
    for lam in LAMBDAS:
        total_actions = total_losses = 0
        inner_detail = {}
        valid = True
        for ih in names:
            cal = sum((inner_blocks[b] for b in names if b != ih), [])
            test = inner_blocks[ih]
            t = fit_threshold(cal, rankings, smaps, gold, lam)
            if t is None:
                valid = False
                break
            a, l = evaluate_threshold(test, t, rankings, smaps, gold)
            total_actions += a
            total_losses += l
            inner_detail[ih] = {"threshold": t, "actions": a, "gold_removed": l}
        if valid:
            candidates.append({
                "lambda": lam, "actions": total_actions,
                "gold_removed": total_losses, "inner": inner_detail
            })

    safe = [c for c in candidates if c["gold_removed"] == 0]
    if not safe:
        return None, candidates

    # Maximize useful pruning among zero-loss inner-CV rules.
    # Tie-break toward larger lambda (more conservative extrapolation).
    best = sorted(safe, key=lambda c: (-c["actions"], -c["lambda"]))[0]
    return best, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()

    print("[1/4] Load + exact D1 reconstruction", flush=True)
    queries, blocks, ids, extended, views, full, gold, type_rows, cite_rows = load_inputs(root)

    global GOLD
    GOLD = gold

    rankings, smaps = reconstruct(
        blocks, ids, extended, views, full, gold, type_rows, cite_rows
    )
    base_pred = {q: rankings[q][:5] for q in ids}
    base = metric(base_pred, gold, ids)
    print("Baseline", base, flush=True)
    if abs(base["recall"] - EXPECTED_R) > 1e-12:
        raise RuntimeError("Recall parity failed")
    if abs(base["macro_precision"] - EXPECTED_P) > 1e-12:
        raise RuntimeError("Precision parity failed")

    print("[2/4] Nested OOF safety calibration", flush=True)
    pred = {q: list(rankings[q][:5]) for q in ids}
    outer_reports = {}
    action_rows = []

    for held in sorted(blocks):
        dev_blocks = {b: blocks[b] for b in blocks if b != held}
        best, all_candidates = choose_lambda(dev_blocks, rankings, smaps, gold)

        if best is None:
            outer_reports[held] = {
                "status": "ABSTAIN_NO_INNER_ZERO_LOSS_RULE",
                "actions": 0, "gold_removed": 0,
                "candidate_lambdas": all_candidates
            }
            print(f"  {held}: ABSTAIN no safe inner rule", flush=True)
            continue

        dev_ids = sum((dev_blocks[b] for b in sorted(dev_blocks)), [])
        lam = best["lambda"]
        threshold = fit_threshold(dev_ids, rankings, smaps, gold, lam)
        actions = losses = 0
        rows = []

        for q in blocks[held]:
            d5 = rankings[q][4]
            s5 = smaps[q][d5]
            if s5 <= threshold:
                was_gold = d5 in gold[q]
                pred[q] = pred[q][:-1]
                actions += 1
                losses += int(was_gold)
                rows.append({
                    "qid": q, "removed_doc": d5, "score": s5,
                    "removed_was_gold": bool(was_gold)
                })
                action_rows.append(rows[-1])

        outer_reports[held] = {
            "status": "APPLIED",
            "lambda": lam,
            "threshold": threshold,
            "inner_selected": best,
            "candidate_lambdas": all_candidates,
            "actions": actions,
            "gold_removed": losses,
            "held_metrics": metric(pred, gold, blocks[held]),
            "actions_detail": rows,
        }
        print(
            f"  {held}: lambda={lam} threshold={threshold:.6f} "
            f"actions={actions} gold_removed={losses}",
            flush=True
        )

    print("[3/4] Aggregate", flush=True)
    final = metric(pred, gold, ids)
    losses = sum(r["gold_removed"] for r in outer_reports.values())
    actions = sum(r["actions"] for r in outer_reports.values())

    report = {
        "schema": "manual.adaptive_k_precision_probe_v2_nested",
        "protocol": {
            "outer": "4-block LOBO",
            "inner": "nested leave-one-dev-block-out lambda selection",
            "lambdas": LAMBDAS,
            "hard_inner_constraint": "zero gold removals",
            "action": "rank5 prune only",
        },
        "baseline": base,
        "adaptive": final,
        "delta": {
            "recall": final["recall"] - base["recall"],
            "macro_precision": final["macro_precision"] - base["macro_precision"],
            "micro_precision": final["micro_precision"] - base["micro_precision"],
        },
        "actions": actions,
        "gold_removed": losses,
        "outer": outer_reports,
        "verdict": (
            "PROMOTE_ZERO_LOSS_ADAPTIVE_K"
            if losses == 0
            and abs(final["recall"] - base["recall"]) <= 1e-12
            and final["macro_precision"] > base["macro_precision"]
            else "KILL_OR_REFINE"
        ),
    }

    out = root / "results/manual/huy_adaptive_k_precision_probe_v2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "OOF_PREDICTIONS.json").write_text(
        json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[4/4] DONE")
    print("=" * 90)
    print(
        f"R {base['recall']:.10f} -> {final['recall']:.10f} "
        f"({final['recall']-base['recall']:+.10f})"
    )
    print(
        f"Pmacro {base['macro_precision']:.10f} -> {final['macro_precision']:.10f} "
        f"({final['macro_precision']-base['macro_precision']:+.10f})"
    )
    print(
        f"Pmicro {base['micro_precision']:.10f} -> {final['micro_precision']:.10f} "
        f"({final['micro_precision']-base['micro_precision']:+.10f})"
    )
    print(f"meanK {base['mean_k']:.4f} -> {final['mean_k']:.4f}")
    print(f"actions={actions} gold_removed={losses}")
    print("Verdict:", report["verdict"])
    print("Report:", out / "REPORT.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
