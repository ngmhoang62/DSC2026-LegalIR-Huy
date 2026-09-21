#!/usr/bin/env python
"""
PRISM PRIVATE TRANSFER-SHIFT AUDIT V1
=====================================

CPU-only. No private labels. No neural inference.

Context
-------
`prism_score_rank` improved clean held-out CAL600 but degraded private leaderboard.
Before spending another submission, this audit asks:

1) How much does each Prism fusion arm churn D1 on CAL OOF vs private?
2) Which positive-CAL arm is most conservative on private?
3) Did the raw Prism score/rank geometry shift from held-out CAL to full-train
   private inference?

Arms
----
  d1
  prism_score
  prism_replace_jina_ft
  prism_replace_crossenc
  prism_score_rank

This script DOES NOT package a submission and DOES NOT use private labels.

Outputs
-------
results/manual/huy_prism_private_transfer_shift_v1/
  REPORT.json
  PRIVATE_TOP5_<ARM>.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ARMS = [
    "prism_score",
    "prism_replace_jina_ft",
    "prism_replace_crossenc",
    "prism_score_rank",
]
EXPECTED_PRIVATE = 2080
EXPECTED_D1_CAL = 0.9569444444444444


def qstats(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "p10": None, "p90": None,
        }
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def aggregate_numeric(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return qstats(vals)


def raw_prism_geometry(ids, candidates, prism, d1_top5):
    rows = []
    for q in ids:
        docs = list(candidates[q])
        sc = np.asarray([float(prism[q][d]) for d in docs], dtype=float)
        order = np.argsort(-sc, kind="stable")
        raw_rank = [docs[i] for i in order]
        top1 = float(sc[order[0]])
        top5 = float(sc[order[min(4, len(order)-1)]])
        med = float(np.median(sc))
        std = float(np.std(sc))
        z_gap15 = (top1 - top5) / max(std, 1e-12)
        d1set = set(d1_top5[q])
        raw5 = set(raw_rank[:5])
        rows.append({
            "qid": q,
            "candidate_count": len(docs),
            "score_mean": float(sc.mean()),
            "score_std": std,
            "score_median": med,
            "top1_score": top1,
            "top5_score": top5,
            "top1_minus_top5": top1 - top5,
            "z_top1_minus_top5": z_gap15,
            "raw_prism_d1_top5_overlap": len(raw5 & d1set),
            "d1_docs_mean_prism": float(np.mean([prism[q][d] for d in d1_top5[q]])),
            "d1_rank5_prism_score": float(prism[q][d1_top5[q][4]]),
        })

    return {
        "candidate_count": aggregate_numeric(rows, "candidate_count"),
        "score_mean": aggregate_numeric(rows, "score_mean"),
        "score_std": aggregate_numeric(rows, "score_std"),
        "top1_minus_top5": aggregate_numeric(rows, "top1_minus_top5"),
        "z_top1_minus_top5": aggregate_numeric(rows, "z_top1_minus_top5"),
        "raw_prism_d1_top5_overlap": aggregate_numeric(
            rows, "raw_prism_d1_top5_overlap"
        ),
        "d1_docs_mean_prism": aggregate_numeric(rows, "d1_docs_mean_prism"),
        "d1_rank5_prism_score": aggregate_numeric(rows, "d1_rank5_prism_score"),
    }


def churn_stats(base, pred, ids):
    ordered = [q for q in ids if pred[q] != base[q]]
    setq = [q for q in ids if set(pred[q]) != set(base[q])]
    enters = [
        len(set(pred[q]) - set(base[q]))
        for q in ids
    ]
    leaves = [
        len(set(base[q]) - set(pred[q]))
        for q in ids
    ]
    overlap = [
        len(set(base[q]) & set(pred[q]))
        for q in ids
    ]
    exact_one_swap = sum(
        1 for q in ids
        if len(set(pred[q]) - set(base[q])) == 1
        and len(set(base[q]) - set(pred[q])) == 1
    )
    multi_swap = sum(
        1 for q in ids
        if len(set(pred[q]) - set(base[q])) >= 2
    )
    return {
        "ordered_churn_queries": len(ordered),
        "ordered_churn_rate": len(ordered) / len(ids),
        "set_churn_queries": len(setq),
        "set_churn_rate": len(setq) / len(ids),
        "entering_docs": int(sum(enters)),
        "leaving_docs": int(sum(leaves)),
        "mean_set_overlap": float(np.mean(overlap)),
        "exact_one_swap_queries": int(exact_one_swap),
        "multi_swap_queries": int(multi_swap),
        "ordered_churn_qids": ordered,
        "set_churn_qids": setq,
    }


def paired_cal(base_res, arm_res, ids):
    wins = losses = ties = 0
    for q in ids:
        a = base_res["per_query_recall"][q]
        b = arm_res["per_query_recall"][q]
        if b > a + 1e-12:
            wins += 1
        elif b < a - 1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "recall": arm_res["recall"],
        "precision": arm_res["precision"],
        "delta_recall": arm_res["recall"] - base_res["recall"],
        "delta_precision": arm_res["precision"] - base_res["precision"],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "single_recall": arm_res["single_recall"],
        "multi_recall": arm_res["multi_recall"],
        "blocks": arm_res["blocks"],
        "churn_vs_d1": churn_stats(
            base_res["predictions"], arm_res["predictions"], ids
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prism-heldout-scores", type=Path, required=True)
    ap.add_argument("--prism-private-scores", type=Path, required=True)
    ap.add_argument("--private-file", default="private-official.json")
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    heldout_path = args.prism_heldout_scores.expanduser().resolve()
    private_path_scores = args.prism_private_scores.expanduser().resolve()
    sys.path.insert(0, str(root))

    for p in (heldout_path, private_path_scores):
        if not p.is_file():
            raise FileNotFoundError(p)

    import audit_prism_d1_lobo_v1 as audit
    import materialize_private_prism_d1_v1 as mat

    print("[1/7] Loading CAL600 + held-out Prism...", flush=True)
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    from run_burst_expanded_fusion_submission import DocumentStore

    (
        _queries,
        blocks,
        cal_ids,
        extended,
        local_views,
        full_channels,
        gold,
        _vnlegal,
        type_rows,
        cite_rows,
    ) = load_cal_inputs()

    raw_cal = audit.load_pickle_scores(heldout_path)
    prism_cal, prism_cal_floor, coverage = audit.aligned_scores(
        raw_cal, cal_ids, extended
    )
    if coverage["coverage"] < .999999:
        raise RuntimeError(f"CAL Prism coverage not complete: {coverage}")

    print("[2/7] Running exact OOF CAL arms...", flush=True)
    cal_results = {}
    for arm in ["d1"] + ARMS:
        res = audit.lobo_arm(
            arm=arm,
            blocks=blocks,
            all_ids=cal_ids,
            candidates=extended,
            local_views=local_views,
            full_channels=full_channels,
            prism=prism_cal,
            type_rows=type_rows,
            cite_rows=cite_rows,
            gold=gold,
        )
        cal_results[arm] = res
        print(
            f"  {arm:28s} R={res['recall']:.10f} "
            f"P={res['precision']:.10f}",
            flush=True,
        )

    base_cal = cal_results["d1"]
    if abs(base_cal["recall"] - EXPECTED_D1_CAL) > 1e-9:
        raise RuntimeError(
            f"D1 CAL parity failed: {base_cal['recall']} != {EXPECTED_D1_CAL}"
        )

    cal_compare = {
        arm: paired_cal(base_cal, cal_results[arm], cal_ids)
        for arm in ARMS
    }

    print("[3/7] Loading private world once...", flush=True)
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    private_file = data / args.private_file
    raw_private_file = json.loads(private_file.read_text(encoding="utf-8"))
    ids = [str(q) for q in raw_private_file]
    questions = {
        str(q): str(v["question"] if isinstance(v, dict) else v)
        for q, v in raw_private_file.items()
    }
    if len(ids) != EXPECTED_PRIVATE:
        raise RuntimeError(f"Expected {EXPECTED_PRIVATE} private qids, got {len(ids)}")

    corpus_paths = sorted((data / "selected-contexts").glob("context_*.json"))
    documents = DocumentStore(corpus_paths)
    priv = mat.reconstruct_private(root, ids)

    raw_private_prism = mat.load_score_dict(private_path_scores)
    missing = [
        (q, d)
        for q in ids
        for d in priv["candidates"][q]
        if d not in raw_private_prism.get(q, {})
    ]
    if missing:
        raise RuntimeError(f"Private Prism cache incomplete sample={missing[:20]}")

    prism_private, prism_private_floor = mat.align(
        raw_private_prism, ids, priv["candidates"]
    )

    floors = {
        "crossenc": float(min(
            v for row in full_channels["crossenc"].values() for v in row.values()
        )),
        "aiteamvn_ft": float(min(
            v for row in full_channels["aiteamvn_ft"].values() for v in row.values()
        )),
        "jina_ft": float(min(
            v for row in full_channels["jina_ft"].values() for v in row.values()
        )),
        "title_embed": float(min(
            v for row in full_channels["title_embed"].values() for v in row.values()
        )),
    }

    print("[4/7] Reconstructing private D1 parity...", flush=True)
    d1_scaler, d1_model, _ = mat.train_full_cal(
        arm="d1",
        all_ids=cal_ids,
        extended=extended,
        local_views=local_views,
        full_channels=full_channels,
        prism_cal=prism_cal,
        type_rows=type_rows,
        cite_rows=cite_rows,
        gold=gold,
    )
    d1_rows, d1_groups, _ = mat.private_feature_bundle(
        root=root,
        documents=documents,
        ids=ids,
        questions=questions,
        priv=priv,
        prism_private=prism_private,
        arm="d1",
        floors=floors,
    )
    d1_private, _ = mat.infer(
        d1_rows, d1_groups, ids, d1_scaler, d1_model
    )

    v14_path = (
        root
        / "results/manual/huy_private_d1_rel_l0_exact_v1/"
          "D1_PRIVATE_V14_FAST.json"
    )
    v14 = json.loads(v14_path.read_text(encoding="utf-8"))
    mismatch = [
        q for q in ids
        if d1_private[q] != [str(d) for d in v14[q]["answer"]]
    ]
    if mismatch:
        raise RuntimeError(
            f"Private D1 parity failed {len(ids)-len(mismatch)}/{len(ids)} "
            f"sample={mismatch[:5]}"
        )
    print("  D1 PRIVATE PARITY: 2080/2080 PASS", flush=True)

    print("[5/7] Inferring ALL private Prism arms...", flush=True)
    private_compare = {}
    private_top5 = {}

    out = root / "results/manual/huy_prism_private_transfer_shift_v1"
    out.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        scaler, model, _ = mat.train_full_cal(
            arm=arm,
            all_ids=cal_ids,
            extended=extended,
            local_views=local_views,
            full_channels=full_channels,
            prism_cal=prism_cal,
            type_rows=type_rows,
            cite_rows=cite_rows,
            gold=gold,
        )
        rows, groups, _ = mat.private_feature_bundle(
            root=root,
            documents=documents,
            ids=ids,
            questions=questions,
            priv=priv,
            prism_private=prism_private,
            arm=arm,
            floors=floors,
        )
        pred, _decision = mat.infer(
            rows, groups, ids, scaler, model
        )
        private_top5[arm] = pred
        ch = churn_stats(d1_private, pred, ids)
        private_compare[arm] = ch

        (out / f"PRIVATE_TOP5_{arm.upper()}.json").write_text(
            json.dumps(
                {q: {"answer": pred[q]} for q in ids},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        print(
            f"  {arm:28s} ordered={ch['ordered_churn_queries']:4d} "
            f"set={ch['set_churn_queries']:4d} "
            f"enter={ch['entering_docs']:4d} "
            f"oneSwap={ch['exact_one_swap_queries']:4d} "
            f"multiSwap={ch['multi_swap_queries']:4d}",
            flush=True,
        )

    print("[6/7] Auditing held-out-CAL vs private-fulltrain Prism geometry...", flush=True)
    cal_geometry = raw_prism_geometry(
        cal_ids,
        extended,
        prism_cal,
        base_cal["predictions"],
    )
    private_geometry = raw_prism_geometry(
        ids,
        priv["candidates"],
        prism_private,
        d1_private,
    )

    shift = {}
    for key in cal_geometry:
        c = cal_geometry[key]
        p = private_geometry[key]
        cm = c["mean"]
        pm = p["mean"]
        shift[key] = {
            "cal": c,
            "private": p,
            "private_minus_cal_mean": (
                None if cm is None or pm is None else float(pm - cm)
            ),
            "private_over_cal_mean": (
                None
                if cm is None or pm is None or abs(cm) < 1e-12
                else float(pm / cm)
            ),
        }

    print("[7/7] Building transfer-risk table...", flush=True)
    transfer = OrderedDict()
    for arm in ARMS:
        cc = cal_compare[arm]
        pc = private_compare[arm]
        transfer[arm] = {
            "cal_delta_recall": cc["delta_recall"],
            "cal_wins": cc["wins"],
            "cal_losses": cc["losses"],
            "cal_net_wins": cc["wins"] - cc["losses"],
            "cal_set_churn_queries": cc["churn_vs_d1"]["set_churn_queries"],
            "cal_set_churn_rate": cc["churn_vs_d1"]["set_churn_rate"],
            "private_set_churn_queries": pc["set_churn_queries"],
            "private_set_churn_rate": pc["set_churn_rate"],
            "private_entering_docs": pc["entering_docs"],
            "private_exact_one_swap_queries": pc["exact_one_swap_queries"],
            "private_multi_swap_queries": pc["multi_swap_queries"],
            "private_to_cal_set_churn_ratio": (
                pc["set_churn_rate"]
                / max(cc["churn_vs_d1"]["set_churn_rate"], 1e-12)
            ),
        }
        print(
            f"  {arm:28s} "
            f"CAL dR={cc['delta_recall']:+.6f} "
            f"W/L={cc['wins']}/{cc['losses']} "
            f"CALset={cc['churn_vs_d1']['set_churn_rate']:.1%} "
            f"PRIVset={pc['set_churn_rate']:.1%} "
            f"ratio={transfer[arm]['private_to_cal_set_churn_ratio']:.2f}",
            flush=True,
        )

    # Label-free private choice among CAL-positive arms:
    # prefer positive CAL paired evidence, then minimize private set churn.
    eligible = [
        arm for arm in ARMS
        if cal_compare[arm]["delta_recall"] > 0
        and cal_compare[arm]["wins"] > cal_compare[arm]["losses"]
    ]
    conservative_candidate = (
        min(
            eligible,
            key=lambda a: (
                private_compare[a]["set_churn_rate"],
                private_compare[a]["entering_docs"],
                -cal_compare[a]["delta_recall"],
            ),
        )
        if eligible else None
    )

    report = {
        "schema": "manual.prism_private_transfer_shift_v1",
        "status": "AUDIT_ONLY_NO_PRIVATE_LABELS",
        "known_leaderboard_result": {
            "arm": "prism_score_rank",
            "precision": 0.203338393,
            "recall": 0.946256956,
            "baseline_d1_precision": 0.203540719,
            "baseline_d1_recall": 0.947858709,
            "delta_precision": 0.203338393 - 0.203540719,
            "delta_recall": 0.946256956 - 0.947858709,
            "note": (
                "These aggregate scores are externally supplied by the user; "
                "no private labels are read by this script."
            ),
        },
        "cal_oof": {
            arm: {
                k: v for k, v in cal_compare[arm].items()
                if k != "churn_vs_d1"
            } | {
                "churn_vs_d1": {
                    k: v for k, v in cal_compare[arm]["churn_vs_d1"].items()
                    if not k.endswith("_qids")
                }
            }
            for arm in ARMS
        },
        "private_label_free_churn": {
            arm: {
                k: v for k, v in private_compare[arm].items()
                if not k.endswith("_qids")
            }
            for arm in ARMS
        },
        "raw_prism_geometry_shift": shift,
        "transfer_table": transfer,
        "decision_support": {
            "eligible_cal_positive_arms": eligible,
            "most_conservative_private_arm_among_cal_positive": conservative_candidate,
            "rule": (
                "Among arms with positive CAL delta and wins>losses, choose "
                "minimum private set churn; this is decision support only, "
                "not evidence that the arm will improve private labels."
            ),
        },
        "artifacts": {
            "heldout_prism": str(heldout_path),
            "private_prism": str(private_path_scores),
            "v14": str(v14_path),
        },
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 112)
    print("KNOWN PRIVATE: prism_score_rank dR=-0.001601753")
    print("CAL-positive arms:", eligible)
    print("MOST CONSERVATIVE CANDIDATE:", conservative_candidate)
    print("REPORT:", report_path)
    print("=" * 112)


if __name__ == "__main__":
    main()
