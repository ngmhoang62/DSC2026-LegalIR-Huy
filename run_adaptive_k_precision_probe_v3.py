#!/usr/bin/env python
"""
Adaptive-K Precision Probe V3 — Transferable normalized tail confidence
=======================================================================

CPU-only. Same exact D1 4-block LOBO baseline.

Motivation:
Raw LogisticRegression decision scores are not guaranteed to have the same scale
across LOBO models and the final all-CAL public model. This probe replaces raw
rank-5 score with query-normalized features.

Families:
  ROBUST_Z5:
      (score5 - median(all candidate scores)) / robust_scale(all candidate scores)
      prune very low z5.
  TOP5_REL_Z5:
      (score5 - mean(score1..score4)) / std(score1..score5)
      prune very negative values.
  BOTTOM_GAP_Z:
      (score4-score5) / robust_scale(all candidate scores)
      prune very large normalized boundary gaps.

Nested safety calibration:
same 4-block outer / 3-block inner protocol as V2.
No outer labels used for threshold/lambda selection.

Run:
 python ../run_adaptive_k_precision_probe_v3.py --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SEED = 2026
EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def load_pkl(root, rel):
    obj = pickle.loads((root / rel).read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def robust_scale(vals):
    x = np.asarray(vals, dtype=np.float64)
    if len(x) < 2:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    q25, q75 = np.percentile(x, [25, 75])
    iqrn = float(q75 - q25) / 1.349 if q75 > q25 else 0.0
    std = float(np.std(x))
    return max(mad, iqrn, std, 1e-6)


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
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=0.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=SEED
        ).fit(scaler.transform(X), y)

        for q in blocks[held]:
            s = model.decision_function(scaler.transform(rows[q]))
            sm = {str(d): float(v) for d, v in zip(groups[q], s)}
            order = sorted(groups[q], key=lambda d: (-sm[d], str(d)))
            rankings[q] = [str(d) for d in order]
            smaps[q] = sm

    return rankings, smaps


def metric(pred, gold, ids):
    rr, pp, hits, returned = [], [], 0, 0
    for q in ids:
        ds = pred[q]
        h = len(set(ds) & gold[q])
        rr.append(h / len(gold[q]))
        pp.append(h / len(ds))
        hits += h
        returned += len(ds)
    return {
        "recall": float(np.mean(rr)),
        "macro_precision": float(np.mean(pp)),
        "micro_precision": hits / returned,
        "mean_k": float(np.mean([len(pred[q]) for q in ids])),
        "hits": hits,
        "returned": returned,
    }


def feature(family, q, rankings, smaps):
    ordered = rankings[q]
    vals = np.asarray([smaps[q][d] for d in ordered], dtype=np.float64)
    top = vals[:5]
    s5 = top[4]

    if family == "ROBUST_Z5":
        med = float(np.median(vals))
        return float((s5 - med) / robust_scale(vals))

    if family == "TOP5_REL_Z5":
        denom = max(float(np.std(top)), 1e-6)
        return float((s5 - float(np.mean(top[:4]))) / denom)

    if family == "BOTTOM_GAP_Z":
        return float((top[3] - top[4]) / robust_scale(vals))

    raise ValueError(family)


def prune_direction(family):
    if family in ("ROBUST_Z5", "TOP5_REL_Z5"):
        return "LOW"
    if family == "BOTTOM_GAP_Z":
        return "HIGH"
    raise ValueError(family)


def threshold_from_positive(pos, lam, direction):
    pos = np.asarray(pos, dtype=np.float64)
    s = robust_scale(pos)
    if direction == "LOW":
        return float(np.min(pos) - lam * s)
    return float(np.max(pos) + lam * s)


def fires(x, t, direction):
    return x <= t if direction == "LOW" else x >= t


def fit_threshold(family, cal_ids, rankings, smaps, gold, lam):
    pos = []
    for q in cal_ids:
        d5 = rankings[q][4]
        if d5 in gold[q]:
            pos.append(feature(family, q, rankings, smaps))
    if not pos:
        return None
    return threshold_from_positive(pos, lam, prune_direction(family))


def eval_thr(family, ids, t, rankings, smaps, gold):
    a = l = 0
    for q in ids:
        x = feature(family, q, rankings, smaps)
        if fires(x, t, prune_direction(family)):
            a += 1
            l += int(rankings[q][4] in gold[q])
    return a, l


def choose_lambda(family, dev_blocks, rankings, smaps, gold):
    names = sorted(dev_blocks)
    rows = []

    for lam in LAMBDAS:
        total_a = total_l = 0
        detail = {}
        valid = True

        for ih in names:
            cal = sum((dev_blocks[b] for b in names if b != ih), [])
            test = dev_blocks[ih]
            t = fit_threshold(family, cal, rankings, smaps, gold, lam)
            if t is None:
                valid = False
                break
            a, l = eval_thr(family, test, t, rankings, smaps, gold)
            total_a += a
            total_l += l
            detail[ih] = {"threshold": t, "actions": a, "gold_removed": l}

        if valid:
            rows.append({
                "lambda": lam, "actions": total_a,
                "gold_removed": total_l, "inner": detail
            })

    safe = [r for r in rows if r["gold_removed"] == 0]
    if not safe:
        return None, rows
    best = sorted(safe, key=lambda r: (-r["actions"], -r["lambda"]))[0]
    return best, rows


def run_family(family, blocks, ids, rankings, smaps, gold):
    pred = {q: list(rankings[q][:5]) for q in ids}
    outer = {}
    actions = losses = 0

    for held in sorted(blocks):
        dev_blocks = {b: blocks[b] for b in blocks if b != held}
        best, candidates = choose_lambda(family, dev_blocks, rankings, smaps, gold)

        if best is None:
            outer[held] = {
                "status": "ABSTAIN", "actions": 0, "gold_removed": 0,
                "candidate_lambdas": candidates
            }
            continue

        dev_ids = sum((dev_blocks[b] for b in sorted(dev_blocks)), [])
        lam = best["lambda"]
        t = fit_threshold(family, dev_ids, rankings, smaps, gold, lam)
        held_rows = []

        for q in blocks[held]:
            x = feature(family, q, rankings, smaps)
            if fires(x, t, prune_direction(family)):
                d5 = rankings[q][4]
                bad = d5 in gold[q]
                pred[q] = pred[q][:-1]
                actions += 1
                losses += int(bad)
                held_rows.append({
                    "qid": q, "removed_doc": d5,
                    "feature": x, "removed_was_gold": bool(bad)
                })

        outer[held] = {
            "status": "APPLIED",
            "lambda": lam,
            "threshold": t,
            "actions": len(held_rows),
            "gold_removed": sum(r["removed_was_gold"] for r in held_rows),
            "inner_selected": best,
            "candidate_lambdas": candidates,
            "held_metrics": metric(pred, gold, blocks[held]),
            "actions_detail": held_rows,
        }

    return {
        "family": family,
        "adaptive": metric(pred, gold, ids),
        "actions": actions,
        "gold_removed": losses,
        "outer": outer,
        "predictions": pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()

    print("[1/4] Load + exact D1 reconstruction", flush=True)
    queries, blocks, ids, extended, views, full, gold, type_rows, cite_rows = load_inputs(root)
    rankings, smaps = reconstruct(blocks, ids, extended, views, full, gold, type_rows, cite_rows)

    base_pred = {q: rankings[q][:5] for q in ids}
    base = metric(base_pred, gold, ids)
    print("Baseline", base, flush=True)
    if abs(base["recall"] - EXPECTED_R) > 1e-12 or abs(base["macro_precision"] - EXPECTED_P) > 1e-12:
        raise RuntimeError("D1 parity failed")

    print("[2/4] Nested normalized adaptive-K", flush=True)
    results = {}
    for family in ("ROBUST_Z5", "TOP5_REL_Z5", "BOTTOM_GAP_Z"):
        r = run_family(family, blocks, ids, rankings, smaps, gold)
        results[family] = r
        m = r["adaptive"]
        print(
            f"{family:14s} R={m['recall']:.10f} "
            f"Pmacro={m['macro_precision']:.10f} "
            f"Pmicro={m['micro_precision']:.10f} "
            f"meanK={m['mean_k']:.4f} "
            f"actions={r['actions']} gold_removed={r['gold_removed']}",
            flush=True
        )
        for b, fr in r["outer"].items():
            print(
                f"  {b}: lambda={fr.get('lambda')} "
                f"threshold={fr.get('threshold')} "
                f"actions={fr['actions']} "
                f"gold_removed={fr['gold_removed']}",
                flush=True
            )

    print("[3/4] Select zero-loss transferable family", flush=True)
    safe = [
        r for r in results.values()
        if r["gold_removed"] == 0
        and abs(r["adaptive"]["recall"] - base["recall"]) <= 1e-12
        and r["adaptive"]["macro_precision"] > base["macro_precision"]
    ]
    best = max(
        safe,
        key=lambda r: r["adaptive"]["macro_precision"],
        default=None
    )

    out = root / "results/manual/huy_adaptive_k_precision_probe_v3"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "manual.adaptive_k_precision_probe_v3_normalized",
        "baseline": base,
        "families": {
            k: {kk: vv for kk, vv in r.items() if kk != "predictions"}
            for k, r in results.items()
        },
        "best": best["family"] if best else None,
        "verdict": "PROMOTE_NORMALIZED_ADAPTIVE_K" if best else "NO_ZERO_LOSS_NORMALIZED_GAIN"
    }
    (out / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if best:
        (out / "BEST_OOF_PREDICTIONS.json").write_text(
            json.dumps(best["predictions"], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print("[4/4] DONE")
    print("=" * 90)
    print(f"Baseline R={base['recall']:.10f} P={base['macro_precision']:.10f}")
    if best:
        m = best["adaptive"]
        print(f"Best={best['family']}")
        print(f"R={m['recall']:.10f} Pmacro={m['macro_precision']:.10f} Pmicro={m['micro_precision']:.10f}")
        print(f"meanK={m['mean_k']:.4f} actions={best['actions']} gold_removed={best['gold_removed']}")
        print(f"DeltaR={m['recall']-base['recall']:+.10f} DeltaP={m['macro_precision']-base['macro_precision']:+.10f}")
        print("Verdict=PROMOTE_NORMALIZED_ADAPTIVE_K")
    else:
        print("Verdict=NO_ZERO_LOSS_NORMALIZED_GAIN")
    print("Report:", out / "REPORT.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
