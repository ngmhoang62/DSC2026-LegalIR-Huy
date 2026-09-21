#!/usr/bin/env python
"""
Adaptive-K Precision Safety Frontier V1
=======================================

Builds pre-registered conservative ROBUST_Z5 public candidates for:
  lambda = 0.5, 1.0, 1.5, 2.0

Uses:
  * exact OOF threshold computation from run_adaptive_k_precision_finalize_v1.py
  * original D1 champion Top5
  * already-sealed public ROBUST_Z5 actions from lambda=0.25 run

Because all requested lambdas are more conservative than 0.25, every prune they
can make is a subset of the already-scored lambda=0.25 public action set.
Therefore no public rescoring and no GPU inference are needed.

Run:
  python ../run_adaptive_k_precision_frontier_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

LAMBDAS = [0.5, 1.0, 1.5, 2.0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_finalize_module(script_dir: Path):
    p = script_dir / "run_adaptive_k_precision_finalize_v1.py"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing required prior script: {p}\n"
            "Place this frontier script in the same D:/Study/DSC2026 directory "
            "as run_adaptive_k_precision_finalize_v1.py."
        )
    spec = importlib.util.spec_from_file_location("adaptive_finalize_v1", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def metrics(pred, gold, ids):
    recalls = []
    macro_p = []
    hits = returned = 0
    for q in ids:
        docs = pred[q]
        h = len(set(docs) & gold[q])
        recalls.append(h / len(gold[q]))
        macro_p.append(h / len(docs))
        hits += h
        returned += len(docs)
    return {
        "recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(macro_p)),
        "micro_precision": float(hits / returned),
        "mean_k": float(np.mean([len(pred[q]) for q in ids])),
        "hits": int(hits),
        "returned": int(returned),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    script_dir = Path(__file__).resolve().parent

    fin = load_finalize_module(script_dir)

    print("[1/5] Reconstructing exact CAL600 D1 OOF once...", flush=True)
    (
        queries,
        blocks,
        ids,
        extended,
        views,
        full,
        gold,
        type_rows,
        cite_rows,
    ) = fin.load_inputs(root)

    rankings, smaps = fin.reconstruct(
        blocks, ids, extended, views, full, gold, type_rows, cite_rows
    )

    base_pred = {q: rankings[q][:5] for q in ids}
    base = metrics(base_pred, gold, ids)
    print(
        f"  baseline R={base['recall']:.10f} "
        f"P={base['macro_precision']:.10f}",
        flush=True,
    )

    print("[2/5] Freezing final thresholds for pre-registered lambdas...", flush=True)
    thresholds = {}
    oof_results = {}

    for lam in LAMBDAS:
        final_t = fin.fit_threshold(ids, rankings, smaps, gold, lam)
        thresholds[lam] = float(final_t)

        pred = {q: list(rankings[q][:5]) for q in ids}
        rows = []

        # Honest OOF evaluation for this global lambda:
        # threshold for each held block is fit only on the other 3 blocks.
        for held in sorted(blocks):
            dev = sum((blocks[b] for b in blocks if b != held), [])
            t = fin.fit_threshold(dev, rankings, smaps, gold, lam)
            for q in blocks[held]:
                z = fin.robust_z5(q, rankings, smaps)
                if z <= t:
                    d5 = rankings[q][4]
                    was_gold = d5 in gold[q]
                    pred[q] = pred[q][:-1]
                    rows.append(
                        {
                            "qid": q,
                            "held": held,
                            "z": float(z),
                            "threshold": float(t),
                            "removed_was_gold": bool(was_gold),
                        }
                    )

        m = metrics(pred, gold, ids)
        losses = sum(r["removed_was_gold"] for r in rows)
        oof_results[lam] = {
            "actions": len(rows),
            "gold_removed": int(losses),
            "metrics": m,
            "delta_recall": m["recall"] - base["recall"],
            "delta_macro_precision": m["macro_precision"] - base["macro_precision"],
        }

        print(
            f"  lambda={lam:<3} final_threshold={final_t:.12f} | "
            f"OOF actions={len(rows):<3} losses={losses} | "
            f"R={m['recall']:.10f} P={m['macro_precision']:.10f}",
            flush=True,
        )

        if losses != 0 or abs(m["recall"] - base["recall"]) > 1e-12:
            raise RuntimeError(
                f"Pre-registered lambda={lam} unexpectedly violates zero-loss OOF"
            )

    print("[3/5] Loading sealed lambda=0.25 public z-scores...", flush=True)
    prior_report_path = (
        root
        / "results/manual/huy_adaptive_k_precision_public_v1/"
        "PUBLIC_ADAPTIVE_K_REPORT.json"
    )
    report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    prior_threshold = float(report["threshold"])
    prior_actions = report["actions"]
    public_z = {str(a["qid"]): float(a["robust_z5"]) for a in prior_actions}

    if len(public_z) != 225:
        print(
            f"  warning: expected 225 prior actions from observed run, got {len(public_z)}",
            flush=True,
        )

    # All frontier thresholds must be <= lambda=.25 threshold, otherwise
    # the sealed action set would not cover all possible prunes.
    for lam, t in thresholds.items():
        if t > prior_threshold + 1e-12:
            raise RuntimeError(
                f"lambda={lam} threshold {t} is less conservative than "
                f"prior threshold {prior_threshold}; cannot subset safely"
            )

    champion_path = (
        root
        / "results/gemini/huy_vnlegal_rank_ablation_v1/"
        "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    )
    champion_raw = json.loads(champion_path.read_text(encoding="utf-8"))
    champion = {
        str(q): (
            [str(d) for d in row["answer"]]
            if isinstance(row, dict)
            else [str(d) for d in row]
        )
        for q, row in champion_raw.items()
    }
    if len(champion) != 1000:
        raise RuntimeError(f"Expected 1000 public queries, got {len(champion)}")

    print("[4/5] Materializing conservative public frontier...", flush=True)
    out_dir = root / "results/manual/huy_adaptive_k_precision_frontier_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = {}

    for lam in LAMBDAS:
        t = thresholds[lam]
        payload = {}
        actions = []

        for q, top5 in champion.items():
            if q in public_z and public_z[q] <= t:
                answer = top5[:4]
                actions.append(
                    {
                        "qid": q,
                        "removed_doc": top5[4],
                        "robust_z5": public_z[q],
                    }
                )
            else:
                answer = top5

            payload[q] = {"answer": answer}

        tag = str(lam).replace(".", "p")
        json_path = out_dir / f"submission_lambda_{tag}.json"
        zip_path = out_dir / f"submission_lambda_{tag}.zip"

        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zf:
            zf.writestr("submission.json", json_path.read_bytes())

        # self-check
        with zipfile.ZipFile(zip_path, "r") as zf:
            check = json.loads(zf.read("submission.json").decode("utf-8"))
        if len(check) != 1000:
            raise RuntimeError("ZIP qid count mismatch")
        if any(len(v["answer"]) not in (4, 5) for v in check.values()):
            raise RuntimeError("Unexpected K")

        mean_k = float(
            np.mean([len(payload[q]["answer"]) for q in payload])
        )
        candidates[str(lam)] = {
            "lambda": lam,
            "threshold": t,
            "public_actions": len(actions),
            "public_prune_rate": len(actions) / 1000.0,
            "mean_k": mean_k,
            "oof": oof_results[lam],
            "submission_json": str(json_path),
            "submission_json_sha256": sha256_file(json_path),
            "submission_zip": str(zip_path),
            "submission_zip_sha256": sha256_file(zip_path),
            "actions": actions,
        }

        print(
            f"  lambda={lam:<3} public_prunes={len(actions):<3} "
            f"rate={len(actions)/1000:.1%} meanK={mean_k:.4f} "
            f"| OOF ΔP={oof_results[lam]['delta_macro_precision']:+.6f}",
            flush=True,
        )

    frontier_report = {
        "schema": "manual.adaptive_k_precision_frontier_v1",
        "policy": (
            "All lambdas were pre-registered before observing these candidate "
            "leaderboard results; no per-public-query labels are used."
        ),
        "source_public_run": str(prior_report_path),
        "source_prior_threshold_lambda_0p25": prior_threshold,
        "source_champion": str(champion_path),
        "baseline_oof": base,
        "candidates": candidates,
        "recommended_submission_order_for_recall_first_objective": [
            "1.5",
            "1.0",
            "0.5",
            "2.0",
        ],
    }
    frontier_path = out_dir / "FRONTIER_REPORT.json"
    frontier_path.write_text(
        json.dumps(frontier_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] DONE")
    print("=" * 98)
    print("Recall-first suggested upload order:")
    print("  1) lambda=1.5  (very conservative, still nonzero OOF precision gain)")
    print("  2) lambda=1.0")
    print("  3) lambda=0.5  (largest remaining precision upside)")
    print("  4) lambda=2.0  (ultra-conservative diagnostic)")
    print(f"Report: {frontier_path}")
    print("GPU inference performed: FALSE")
    print("=" * 98)


if __name__ == "__main__":
    main()
