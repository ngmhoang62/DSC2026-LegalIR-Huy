#!/usr/bin/env python
"""
Adaptive-K Precision Finalize V1
================================

CPU-only final calibration for the already-promoted ROBUST_Z5 family.

This script does NOT touch public test predictions. It:
  1) reconstructs exact D1 CAL600 LOBO,
  2) evaluates ONE GLOBAL lambda across all 4 held blocks,
  3) selects the most aggressive lambda with zero OOF gold removals,
  4) fits the final ROBUST_Z5 threshold on all CAL600,
  5) writes a frozen deployment contract.

The public materializer should consume this contract verbatim.

Run:
  python ../run_adaptive_k_precision_finalize_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SEED = 2026
EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pkl(root: Path, rel: str):
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
        fl = floor if floor is not None else min(
            v for q in obj for v in obj[q].values()
        )
        return {
            q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]}
            for q in ids
        }

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

    rankings = {}
    smaps = {}

    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        eval_ids = train + blocks[held]
        rows, groups = ltr_features(views, D1_VIEWS, extended, eval_ids, full)

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
        ).fit(scaler.transform(X), y)

        for q in blocks[held]:
            scores = model.decision_function(scaler.transform(rows[q]))
            sm = {
                str(d): float(v)
                for d, v in zip(groups[q], scores)
            }
            order = sorted(groups[q], key=lambda d: (-sm[d], str(d)))
            rankings[q] = [str(d) for d in order]
            smaps[q] = sm

    return rankings, smaps


def robust_z5(q, rankings, smaps):
    vals = np.asarray(
        [smaps[q][d] for d in rankings[q]],
        dtype=np.float64,
    )
    s5 = vals[4]
    med = float(np.median(vals))
    return float((s5 - med) / robust_scale(vals))


def metric(pred, gold, ids):
    rr, pp = [], []
    hits = returned = 0
    for q in ids:
        docs = pred[q]
        h = len(set(docs) & gold[q])
        rr.append(h / len(gold[q]))
        pp.append(h / len(docs))
        hits += h
        returned += len(docs)
    return {
        "recall": float(np.mean(rr)),
        "macro_precision": float(np.mean(pp)),
        "micro_precision": float(hits / returned),
        "mean_k": float(np.mean([len(pred[q]) for q in ids])),
        "hits": hits,
        "returned": returned,
    }


def fit_threshold(cal_ids, rankings, smaps, gold, lam):
    pos = [
        robust_z5(q, rankings, smaps)
        for q in cal_ids
        if rankings[q][4] in gold[q]
    ]
    if not pos:
        raise RuntimeError("No gold rank5 examples in calibration set")
    return float(min(pos) - lam * robust_scale(pos))


def eval_threshold(test_ids, threshold, rankings, smaps, gold):
    actions = losses = 0
    rows = []
    for q in test_ids:
        z = robust_z5(q, rankings, smaps)
        if z <= threshold:
            d5 = rankings[q][4]
            was_gold = d5 in gold[q]
            actions += 1
            losses += int(was_gold)
            rows.append({
                "qid": q,
                "removed_doc": d5,
                "robust_z5": z,
                "removed_was_gold": bool(was_gold),
            })
    return actions, losses, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()

    print("[1/4] Exact D1 CAL600 reconstruction...", flush=True)
    queries, blocks, ids, extended, views, full, gold, type_rows, cite_rows = load_inputs(root)
    rankings, smaps = reconstruct(
        blocks, ids, extended, views, full, gold, type_rows, cite_rows
    )

    base_pred = {q: rankings[q][:5] for q in ids}
    base = metric(base_pred, gold, ids)
    print("Baseline:", base, flush=True)
    if abs(base["recall"] - EXPECTED_R) > 1e-12:
        raise RuntimeError("D1 recall parity failed")
    if abs(base["macro_precision"] - EXPECTED_P) > 1e-12:
        raise RuntimeError("D1 precision parity failed")

    print("[2/4] Selecting one GLOBAL lambda by 4-fold OOF...", flush=True)
    candidates = []
    for lam in LAMBDAS:
        total_actions = total_losses = 0
        folds = {}

        for held in sorted(blocks):
            dev = sum((blocks[b] for b in blocks if b != held), [])
            threshold = fit_threshold(dev, rankings, smaps, gold, lam)
            a, l, _ = eval_threshold(
                blocks[held], threshold, rankings, smaps, gold
            )
            folds[held] = {
                "threshold": threshold,
                "actions": a,
                "gold_removed": l,
            }
            total_actions += a
            total_losses += l

        row = {
            "lambda": lam,
            "actions": total_actions,
            "gold_removed": total_losses,
            "folds": folds,
        }
        candidates.append(row)
        print(
            f"  lambda={lam:<4} actions={total_actions:<4} "
            f"gold_removed={total_losses}",
            flush=True,
        )

    safe = [r for r in candidates if r["gold_removed"] == 0]
    if not safe:
        raise RuntimeError("No global lambda has zero OOF gold removals")

    # Maximize safe pruning. Tie-break to larger lambda (conservative).
    selected = sorted(
        safe,
        key=lambda r: (-r["actions"], -r["lambda"])
    )[0]
    lam = selected["lambda"]

    print(
        f"[3/4] Selected GLOBAL lambda={lam} "
        f"with OOF actions={selected['actions']} and 0 gold removals",
        flush=True,
    )

    # Build the OOF candidate induced by the single selected lambda.
    oof_pred = {q: list(rankings[q][:5]) for q in ids}
    oof_rows = []
    for held in sorted(blocks):
        dev = sum((blocks[b] for b in blocks if b != held), [])
        threshold = fit_threshold(dev, rankings, smaps, gold, lam)
        for q in blocks[held]:
            z = robust_z5(q, rankings, smaps)
            if z <= threshold:
                d5 = rankings[q][4]
                oof_pred[q] = oof_pred[q][:-1]
                oof_rows.append({
                    "qid": q,
                    "held_block": held,
                    "removed_doc": d5,
                    "robust_z5": z,
                    "threshold": threshold,
                    "removed_was_gold": d5 in gold[q],
                })

    oof = metric(oof_pred, gold, ids)
    oof_gold_removed = sum(r["removed_was_gold"] for r in oof_rows)
    if oof_gold_removed != 0:
        raise RuntimeError("Selected global lambda unexpectedly removes OOF gold")
    if abs(oof["recall"] - base["recall"]) > 1e-12:
        raise RuntimeError("Selected global lambda changes recall")

    # Final threshold fit on all CAL600.
    final_threshold = fit_threshold(ids, rankings, smaps, gold, lam)

    contract = {
        "schema": "manual.adaptive_k_precision_deployment_contract_v1",
        "family": "ROBUST_Z5",
        "action": "prune D1 rank5 iff robust_z5 <= final_threshold",
        "robust_z5_definition": (
            "(D1_score_rank5 - median(all_candidate_D1_scores)) / "
            "max(MAD*1.4826, IQR/1.349, std, 1e-6)"
        ),
        "global_lambda": lam,
        "lambda_candidates": candidates,
        "selected_oof": {
            "actions": len(oof_rows),
            "gold_removed": oof_gold_removed,
            "metrics": oof,
            "delta_recall": oof["recall"] - base["recall"],
            "delta_macro_precision": (
                oof["macro_precision"] - base["macro_precision"]
            ),
        },
        "baseline": base,
        "final_threshold_all_cal600": final_threshold,
        "cal_query_count": len(ids),
        "training_policy": {
            "family_fixed_before_finalization": True,
            "lambda_selected_by_4fold_oof_only": True,
            "public_labels_used": False,
            "public_predictions_touched": False,
        },
    }

    out = root / "results/manual/huy_adaptive_k_precision_finalize_v1"
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "DEPLOYMENT_CONTRACT.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "OOF_PREDICTIONS.json").write_text(
        json.dumps(oof_pred, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    contract["contract_sha256"] = sha256_file(contract_path)
    # Write SHA-bearing receipt separately so contract hash remains stable.
    (out / "CONTRACT_RECEIPT.json").write_text(
        json.dumps({
            "deployment_contract": str(contract_path),
            "sha256": sha256_file(contract_path),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] DONE")
    print("=" * 92)
    print(f"Selected lambda={lam}")
    print(f"Final threshold={final_threshold:.12f}")
    print(
        f"OOF R {base['recall']:.10f} -> {oof['recall']:.10f} "
        f"({oof['recall']-base['recall']:+.10f})"
    )
    print(
        f"OOF Pmacro {base['macro_precision']:.10f} -> "
        f"{oof['macro_precision']:.10f} "
        f"({oof['macro_precision']-base['macro_precision']:+.10f})"
    )
    print(
        f"OOF actions={len(oof_rows)} gold_removed={oof_gold_removed} "
        f"meanK={oof['mean_k']:.4f}"
    )
    print(f"Contract: {contract_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
