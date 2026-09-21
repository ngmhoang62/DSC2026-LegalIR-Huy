#!/usr/bin/env python
"""
AUDIT PRISM -> D1 FUSION ON CLEAN CAL600 HELD-OUT SCORES
========================================================

CPU-only. No private labels. No GPU inference.

Expected Prism artifact:
    best_channel_scores.pkl
from the pre-full-train Prism run trained on 1,050 labeled queries disjoint
from the 600-query CAL evaluation set.

Arms:
  d1                           48D
  prism_score                  50D
  prism_replace_jina_ft        48D
  prism_replace_crossenc       48D
  prism_score_rank             52D
"""

from __future__ import annotations
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_D1 = 0.9569444444444444
SEED = 2026
BASE_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load_pickle_scores(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        obj = obj["scores"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported Prism score artifact: {type(obj)}")
    return {str(q): {str(d): float(s) for d, s in row.items()} for q, row in obj.items()}

def aligned_scores(raw, ids, candidates):
    values = [v for q in raw.values() for v in q.values()]
    if not values:
        raise RuntimeError("Prism artifact contains no scores")
    floor = float(min(values))
    out = {q: {d: raw.get(q, {}).get(d, floor) for d in candidates[q]} for q in ids}
    total = sum(len(candidates[q]) for q in ids)
    present = sum(sum(d in raw.get(q, {}) for d in candidates[q]) for q in ids)
    perq = {
        q: sum(d in raw.get(q, {}) for d in candidates[q]) / max(1, len(candidates[q]))
        for q in ids
    }
    return out, floor, {
        "present": int(present),
        "total": int(total),
        "coverage": float(present / max(1, total)),
        "min_query_coverage": float(min(perq.values())),
        "mean_query_coverage": float(np.mean(list(perq.values()))),
        "queries_with_any_missing": int(sum(v < 1.0 for v in perq.values())),
    }

def metrics(pred, gold, ids):
    r, p = {}, {}
    for q in ids:
        hit = len(set(pred[q]) & set(gold[q]))
        r[q] = hit / max(1, len(gold[q]))
        p[q] = hit / 5.0
    return (
        float(np.mean([r[q] for q in ids])),
        float(np.mean([p[q] for q in ids])),
        r,
        p,
    )

def build_arm(name, full_channels, prism, local_views, candidates, ids):
    channels = dict(full_channels)
    views = dict(local_views)
    view_names = list(BASE_VIEWS)
    if name == "d1":
        pass
    elif name == "prism_score":
        channels["prism_ft"] = prism
    elif name == "prism_replace_jina_ft":
        channels.pop("jina_ft", None)
        channels["prism_ft"] = prism
    elif name == "prism_replace_crossenc":
        channels.pop("crossenc", None)
        channels["prism_ft"] = prism
    elif name == "prism_score_rank":
        channels["prism_ft"] = prism
        views["prism_ft"] = {
            q: sorted(candidates[q], key=lambda d: (-prism[q][d], d))
            for q in ids
        }
        view_names.append("prism_ft")
    else:
        raise ValueError(name)
    expected_dim = {
        "d1": 48,
        "prism_score": 50,
        "prism_replace_jina_ft": 48,
        "prism_replace_crossenc": 48,
        "prism_score_rank": 52,
    }[name]
    return channels, views, view_names, expected_dim

def lobo_arm(*, arm, blocks, all_ids, candidates, local_views, full_channels,
             prism, type_rows, cite_rows, gold):
    from tune_expanded_fusion_selection import ltr_features
    channels, views, view_names, expected_dim = build_arm(
        arm, full_channels, prism, local_views, candidates, all_ids
    )
    pred, score_maps = {}, {}
    for held in sorted(blocks):
        train = sum((list(blocks[b]) for b in sorted(blocks) if b != held), [])
        test = list(blocks[held])
        eval_ids = train + test
        rows0, groups = ltr_features(views, view_names, candidates, eval_ids, channels)
        rows = {
            q: np.concatenate([rows0[q], type_rows[q], cite_rows[q]], axis=1)
            for q in eval_ids
        }
        got_dim = rows[eval_ids[0]].shape[1]
        if got_dim != expected_dim:
            raise RuntimeError(f"{arm}: feature dim {got_dim} != expected {expected_dim}")
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]] for q in train
        ]).astype(np.int8)
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=SEED
        )
        model.fit(scaler.transform(X), y)
        for q in test:
            s = model.decision_function(scaler.transform(rows[q]))
            order = np.argsort(-s)
            pred[q] = [groups[q][i] for i in order[:5]]
            score_maps[q] = {groups[q][i]: float(s[i]) for i in range(len(groups[q]))}
    r, p, per_r, _ = metrics(pred, gold, all_ids)
    block_r = {b: float(np.mean([per_r[q] for q in blocks[b]])) for b in sorted(blocks)}
    single = float(np.mean([per_r[q] for q in all_ids if len(gold[q]) == 1]))
    multi = float(np.mean([per_r[q] for q in all_ids if len(gold[q]) > 1]))
    return {
        "arm": arm, "feature_dim": expected_dim, "views": view_names,
        "recall": r, "precision": p, "single_recall": single,
        "multi_recall": multi, "blocks": block_r,
        "predictions": pred, "decision_scores": score_maps,
        "per_query_recall": per_r,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prism-heldout-scores", type=Path, required=True)
    ap.add_argument("--min-coverage", type=float, default=.95)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    score_path = args.prism_heldout_scores.resolve()
    sys.path.insert(0, str(root))
    if not score_path.is_file():
        raise FileNotFoundError(score_path)

    print("[1/4] Loading authoritative D1 CAL600 bundle...")
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import load_cal_inputs
    (_queries, blocks, all_ids, extended, local_views, full_channels, gold,
     _vnlegal, type_rows, cite_rows) = load_cal_inputs()

    print("[2/4] Loading clean held-out Prism scores...")
    raw_prism = load_pickle_scores(score_path)
    prism, prism_floor, coverage = aligned_scores(raw_prism, all_ids, extended)
    print(
        f"  coverage={coverage['coverage']:.4%} "
        f"min-query={coverage['min_query_coverage']:.4%} "
        f"missing-queries={coverage['queries_with_any_missing']}"
    )
    if coverage["coverage"] < args.min_coverage:
        raise RuntimeError(
            f"Prism coverage {coverage['coverage']:.4%} < {args.min_coverage:.4%}"
        )

    arms = [
        "d1", "prism_score", "prism_replace_jina_ft",
        "prism_replace_crossenc", "prism_score_rank",
    ]
    print("[3/4] Running exact D1 LOBO family...")
    results = {}
    for arm in arms:
        res = lobo_arm(
            arm=arm, blocks=blocks, all_ids=all_ids, candidates=extended,
            local_views=local_views, full_channels=full_channels, prism=prism,
            type_rows=type_rows, cite_rows=cite_rows, gold=gold,
        )
        results[arm] = res
        print(
            f"  {arm:28s} R={res['recall']:.10f} "
            f"P={res['precision']:.10f} multi={res['multi_recall']:.6f}"
        )

    base = results["d1"]
    if abs(base["recall"] - EXPECTED_D1) > 1e-9:
        raise RuntimeError(f"D1 LOBO parity failed: {base['recall']} != {EXPECTED_D1}")

    print("[4/4] Paired comparison + recommendation...")
    comparisons = {}
    for arm in arms[1:]:
        r = results[arm]
        wins = losses = churn = 0
        for q in all_ids:
            a = base["per_query_recall"][q]
            b = r["per_query_recall"][q]
            wins += int(b > a + 1e-12)
            losses += int(b < a - 1e-12)
            churn += int(r["predictions"][q] != base["predictions"][q])
        comparisons[arm] = {
            "delta_recall": r["recall"] - base["recall"],
            "delta_precision": r["precision"] - base["precision"],
            "wins": wins, "losses": losses,
            "ordered_top5_churn": churn,
            "block_d_delta": r["blocks"]["d"] - base["blocks"]["d"],
        }
        print(
            f"  {arm:28s} dR={r['recall']-base['recall']:+.10f} "
            f"W/L={wins}/{losses} churn={churn} "
            f"dD={comparisons[arm]['block_d_delta']:+.6f}"
        )

    eligible = [
        a for a in arms[1:]
        if comparisons[a]["delta_recall"] > 0
        and comparisons[a]["wins"] > comparisons[a]["losses"]
    ]
    if eligible:
        recommended = max(
            eligible,
            key=lambda a: (
                results[a]["recall"],
                comparisons[a]["wins"] - comparisons[a]["losses"],
                -comparisons[a]["ordered_top5_churn"],
            ),
        )
        verdict = "PROMOTE_PRISM_PRIVATE"
    else:
        recommended = None
        verdict = "KILL_PRISM_D1_FUSION"

    out = root / "results/manual/huy_prism_d1_lobo_v1"
    out.mkdir(parents=True, exist_ok=True)
    compact = {
        arm: {k: v for k, v in res.items()
              if k not in ("predictions", "decision_scores", "per_query_recall")}
        for arm, res in results.items()
    }
    report = {
        "schema": "manual.prism_d1_lobo_v1",
        "status": verdict,
        "prism_artifact": {
            "path": str(score_path), "sha256": sha256(score_path),
            "floor": prism_floor, "coverage": coverage,
            "assumption": (
                "held-out Prism scores were produced by a checkpoint trained "
                "without CAL600 evaluation queries"
            ),
        },
        "baseline_expected_recall": EXPECTED_D1,
        "arms": compact,
        "comparisons_vs_d1": comparisons,
        "recommended_arm": recommended,
        "selection_rule": (
            "Recall-first: positive pooled Recall and wins>losses; maximize "
            "Recall, then net wins, then minimize churn"
        ),
    }
    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("=" * 100)
    print("VERDICT:", verdict)
    print("RECOMMENDED ARM:", recommended)
    print("Report:", report_path)
    print("=" * 100)

if __name__ == "__main__":
    main()
