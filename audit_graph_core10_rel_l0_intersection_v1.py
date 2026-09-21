#!/usr/bin/env python
"""
GRAPH_CORE10 x REL_L0 SAFE-SLOT INTERSECTION AUDIT V1
======================================================

CPU-only. No neural inference.

Purpose
-------
Before spending any GPU on graph candidates, answer:

  1) How many GRAPH_CORE10 rescued queries fall inside the frozen REL_L0
     safe population (96/600)?
  2) If graph candidates are only allowed on those REL_L0-safe queries,
     what is the one-slot oracle headroom on:
        - exact D1;
        - Prism score+rank?
  3) How many graph pairs would actually need BGE CE scoring?

Frozen REL_L0 contract
----------------------
  REL = BGE_CE(rank5) - median(BGE_CE(rank1..4))
  safe iff REL < -3.0393552780151367

The threshold was frozen from non-CAL Fold0 DEV. No threshold search here.

Inputs
------
  results/manual/huy_graph_core10_prism_headroom_v1/
    GRAPH_CORE10_CANDIDATES.jsonl

  results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/
    cal_exact_d1_top5_ce_scores_v1/<qid>.json

CAL gold is NOT read until:
  - safe population is sealed;
  - graph candidate order is sealed;
  - candidate workload is sealed.

Outputs
-------
results/manual/huy_graph_core10_rel_l0_intersection_v1/
  REPORT.json
  SAFE_GRAPH_CE_WORKLOAD.jsonl

Interpretation
--------------
This is a ceiling/triage audit, not a final admission policy.

If >=2 graph-rescue queries are REL_L0-safe and Prism safe-slot oracle headroom
is meaningfully positive, the next experiment should run the frozen Fold0 BGE
challenger certificate only on SAFE_GRAPH_CE_WORKLOAD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REL_L0 = -3.0393552780151367
EXPECTED_SAFE = 96
EXPECTED_D1 = 0.9569444444444444
EXPECTED_PRISM = 0.9636111111111111


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_top5_bge(root: Path, q: str):
    base = (
        root
        / "results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/"
          "cal_exact_d1_top5_ce_scores_v1"
    )
    for p in (base / f"{q}.json", base / f"{q}.jsonl"):
        if not p.is_file():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            return {str(d): float(s) for d, s in obj["scores"].items()}
        vals = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    vals[str(k)] = float(v)
                except Exception:
                    pass
        if vals:
            return vals
    return None


def load_gold(root: Path, ids):
    p = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
    )
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {
        q: {str(d) for d in raw[q]["answer"]}
        for q in ids
    }


def metrics(pred, gold, ids):
    per = {}
    p = []
    for q in ids:
        h = len(set(pred[q]) & gold[q])
        per[q] = h / max(1, len(gold[q]))
        p.append(h / 5.0)
    return {
        "recall": float(np.mean([per[q] for q in ids])),
        "precision": float(np.mean(p)),
        "single_recall": float(np.mean([
            per[q] for q in ids if len(gold[q]) == 1
        ])),
        "multi_recall": float(np.mean([
            per[q] for q in ids if len(gold[q]) > 1
        ])),
        "per_query_recall": per,
    }


def one_slot_oracle(base, candidates, allowed, gold, ids):
    out = {}
    gain_qids = []
    loss_possible_if_forced = []

    for q in ids:
        before = list(base[q])
        if q not in allowed or not candidates[q]:
            out[q] = before
            continue

        incumbent = before[4]
        choices = [incumbent] + list(candidates[q])
        chosen = next((d for d in choices if d in gold[q]), incumbent)
        after = before[:4] + [chosen]
        out[q] = after

        hb = len(set(before) & gold[q])
        ha = len(set(after) & gold[q])
        if ha > hb:
            gain_qids.append(q)

        if incumbent in gold[q]:
            loss_possible_if_forced.append(q)

    return out, gain_qids, loss_possible_if_forced


def first_candidate_policy(base, candidates, allowed, ids):
    out = {}
    actions = []
    for q in ids:
        before = list(base[q])
        if q not in allowed or not candidates[q]:
            out[q] = before
            continue
        c = candidates[q][0]
        if c in before:
            out[q] = before
            continue
        out[q] = before[:4] + [c]
        actions.append({
            "qid": q,
            "defender": before[4],
            "challenger": c,
        })
    return out, actions


def compare(base_m, cand_m, ids):
    wins = []
    losses = []
    for q in ids:
        a = base_m["per_query_recall"][q]
        b = cand_m["per_query_recall"][q]
        if b > a + 1e-12:
            wins.append(q)
        elif b < a - 1e-12:
            losses.append(q)
    return {
        "delta_recall": cand_m["recall"] - base_m["recall"],
        "delta_precision": cand_m["precision"] - base_m["precision"],
        "wins": len(wins),
        "losses": len(losses),
        "win_qids": wins,
        "loss_qids": losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--graph-core",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    core_path = (
        args.graph_core.expanduser().resolve()
        if args.graph_core is not None
        else (
            root
            / "results/manual/huy_graph_core10_prism_headroom_v1/"
              "GRAPH_CORE10_CANDIDATES.jsonl"
        )
    )
    if not core_path.is_file():
        raise FileNotFoundError(core_path)

    print("[1/6] Loading sealed GRAPH_CORE10 candidate artifact...", flush=True)
    rows = {}
    order = []
    for r in read_jsonl(core_path):
        q = str(r["qid"])
        order.append(q)
        rows[q] = r

    if len(order) != 600 or len(rows) != 600:
        raise RuntimeError(
            f"Expected GRAPH_CORE10 population 600, got {len(order)}/{len(rows)}"
        )

    ids = list(order)
    d1 = {
        q: [str(x) for x in rows[q]["d1_top5"]]
        for q in ids
    }
    prism = {
        q: [str(x) for x in rows[q]["prism_top5"]]
        for q in ids
    }
    graph = {
        q: [str(x["doc_id"]) for x in rows[q]["graph_candidates"]]
        for q in ids
    }

    print("[2/6] Sealing frozen REL_L0 safe population BEFORE gold reveal...", flush=True)
    safe = set()
    rel_values = {}
    missing = []

    for q in ids:
        sc = load_top5_bge(root, q)
        top = d1[q]
        if sc is None or any(d not in sc for d in top):
            missing.append(q)
            continue

        med = float(np.median([sc[d] for d in top[:4]]))
        rel = float(sc[top[4]] - med)
        rel_values[q] = rel
        if rel < REL_L0:
            safe.add(q)

    print(
        f"  safe={len(safe)} missing_ce={len(missing)} "
        f"graph-safe-queries={sum(bool(graph[q]) for q in safe)}",
        flush=True,
    )
    if len(safe) != EXPECTED_SAFE:
        raise RuntimeError(
            f"REL_L0 safe parity failed: {len(safe)} != {EXPECTED_SAFE}"
        )

    # Seal CE workload before gold.
    out = root / "results/manual/huy_graph_core10_rel_l0_intersection_v1"
    out.mkdir(parents=True, exist_ok=True)

    workload_path = out / "SAFE_GRAPH_CE_WORKLOAD.jsonl"
    pair_count = 0
    with workload_path.open("w", encoding="utf-8") as f:
        for q in ids:
            if q not in safe or not graph[q]:
                continue
            pair_count += len(graph[q])
            f.write(json.dumps({
                "qid": q,
                "question": str(rows[q].get("question", "")),
                "d1_top5": d1[q],
                "prism_top5": prism[q],
                "d1_rank5_rel_l0_value": rel_values[q],
                "candidate_doc_ids": graph[q],
                "candidate_details": rows[q]["graph_candidates"],
            }, ensure_ascii=False) + "\n")

    sealed = {
        "safe_qids": sorted(safe),
        "safe_count": len(safe),
        "missing_ce": missing,
        "graph_safe_queries": int(sum(bool(graph[q]) for q in safe)),
        "graph_safe_pairs": int(pair_count),
        "rel_l0": REL_L0,
        "workload": str(workload_path),
    }
    seal_path = out / "SAFE_POPULATION_SEALED.json"
    seal_path.write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"  sealed graph CE workload: queries="
        f"{sealed['graph_safe_queries']} pairs={pair_count}",
        flush=True,
    )

    print("[3/6] Revealing CAL gold for intersection/oracle diagnostics...", flush=True)
    gold = load_gold(root, ids)

    d1_m = metrics(d1, gold, ids)
    prism_m = metrics(prism, gold, ids)

    print(
        f"  D1 R={d1_m['recall']:.10f} "
        f"Prism R={prism_m['recall']:.10f}",
        flush=True,
    )
    if abs(d1_m["recall"] - EXPECTED_D1) > 1e-9:
        raise RuntimeError(
            f"D1 parity failed: {d1_m['recall']} != {EXPECTED_D1}"
        )
    if abs(prism_m["recall"] - EXPECTED_PRISM) > 1e-9:
        raise RuntimeError(
            f"Prism parity failed: {prism_m['recall']} != {EXPECTED_PRISM}"
        )

    outside_rescue_pairs = []
    graph_rescue_qids = set()

    # The core artifact contains only docs outside the original D1 pool.
    # For this audit "graph rescue" means graph candidate is gold.
    for q in ids:
        for d in graph[q]:
            if d in gold[q]:
                outside_rescue_pairs.append((q, d))
                graph_rescue_qids.add(q)

    rescue_safe = sorted(graph_rescue_qids & safe)
    rescue_unsafe = sorted(graph_rescue_qids - safe)

    print(
        f"  graph rescue queries={len(graph_rescue_qids)} "
        f"inside_REL_L0_safe={len(rescue_safe)} "
        f"outside={len(rescue_unsafe)}",
        flush=True,
    )
    print(f"  safe rescue qids={rescue_safe}", flush=True)

    print("[4/6] Measuring REL_L0-safe one-slot oracle headroom...", flush=True)
    d1_oracle_pred, d1_gain, d1_safe_gold_def = one_slot_oracle(
        d1, graph, safe, gold, ids
    )
    prism_oracle_pred, prism_gain, prism_safe_gold_def = one_slot_oracle(
        prism, graph, safe, gold, ids
    )

    d1_oracle_m = metrics(d1_oracle_pred, gold, ids)
    prism_oracle_m = metrics(prism_oracle_pred, gold, ids)

    print(
        f"  D1 safe-slot oracle R={d1_oracle_m['recall']:.10f} "
        f"dR={d1_oracle_m['recall']-d1_m['recall']:+.10f} "
        f"gainQ={len(d1_gain)}",
        flush=True,
    )
    print(
        f"  Prism safe-query slot oracle R={prism_oracle_m['recall']:.10f} "
        f"dR={prism_oracle_m['recall']-prism_m['recall']:+.10f} "
        f"gainQ={len(prism_gain)}",
        flush=True,
    )

    print("[5/6] Evaluating fully sealed FIRST_CORE sanity policy...", flush=True)
    d1_first_pred, d1_actions = first_candidate_policy(
        d1, graph, safe, ids
    )
    prism_first_pred, prism_actions = first_candidate_policy(
        prism, graph, safe, ids
    )
    d1_first_m = metrics(d1_first_pred, gold, ids)
    prism_first_m = metrics(prism_first_pred, gold, ids)

    d1_first_cmp = compare(d1_m, d1_first_m, ids)
    prism_first_cmp = compare(prism_m, prism_first_m, ids)

    print(
        f"  D1 FIRST_CORE dR={d1_first_cmp['delta_recall']:+.10f} "
        f"W/L={d1_first_cmp['wins']}/{d1_first_cmp['losses']} "
        f"actions={len(d1_actions)}",
        flush=True,
    )
    print(
        f"  Prism FIRST_CORE dR={prism_first_cmp['delta_recall']:+.10f} "
        f"W/L={prism_first_cmp['wins']}/{prism_first_cmp['losses']} "
        f"actions={len(prism_actions)}",
        flush=True,
    )

    print("[6/6] Writing report / next-stage decision...", flush=True)

    worth_bge = bool(
        len(rescue_safe) >= 2
        and (
            prism_oracle_m["recall"] - prism_m["recall"]
            >= 0.0015 - 1e-12
        )
        and pair_count > 0
    )

    report = {
        "schema": "manual.graph_core10_rel_l0_intersection_v1",
        "status": (
            "RUN_FROZEN_BGE_CERTIFICATE"
            if worth_bge
            else "REL_L0_GRAPH_INTERSECTION_TOO_SMALL"
        ),
        "frozen_contract": {
            "REL_L0": REL_L0,
            "safe_definition": (
                "BGE_CE(D1 rank5) - median(BGE_CE(D1 top1..4)) < REL_L0"
            ),
            "expected_safe_count": EXPECTED_SAFE,
        },
        "sealed_before_gold": sealed,
        "baselines": {
            "d1": {
                k: v for k, v in d1_m.items()
                if k != "per_query_recall"
            },
            "prism": {
                k: v for k, v in prism_m.items()
                if k != "per_query_recall"
            },
        },
        "graph_rescue": {
            "pairs": [
                {"qid": q, "gold_doc": d}
                for q, d in sorted(outside_rescue_pairs)
            ],
            "unique_queries": len(graph_rescue_qids),
            "safe_intersection_qids": rescue_safe,
            "safe_intersection_count": len(rescue_safe),
            "outside_safe_qids": rescue_unsafe,
        },
        "safe_slot_oracle": {
            "d1": {
                "recall": d1_oracle_m["recall"],
                "delta_recall": (
                    d1_oracle_m["recall"] - d1_m["recall"]
                ),
                "gain_qids": d1_gain,
                "safe_queries_where_current_rank5_is_gold": (
                    d1_safe_gold_def
                ),
            },
            "prism": {
                "recall": prism_oracle_m["recall"],
                "delta_recall": (
                    prism_oracle_m["recall"] - prism_m["recall"]
                ),
                "gain_qids": prism_gain,
                "D1_safe_queries_where_Prism_rank5_is_gold": (
                    prism_safe_gold_def
                ),
            },
        },
        "sealed_first_core_sanity": {
            "d1": {
                "actions": len(d1_actions),
                **d1_first_cmp,
            },
            "prism": {
                "actions": len(prism_actions),
                **prism_first_cmp,
            },
        },
        "next_stage": {
            "worth_running_frozen_bge_certificate": worth_bge,
            "workload": str(workload_path),
            "pairs": pair_count,
            "candidate_policy": (
                "On REL_L0-safe D1 queries only, score GRAPH_CORE10 with "
                "frozen non-CAL Fold0 BGE CE. Scan graph order and choose "
                "first candidate satisfying CE(candidate)>CE(D1 rank5) AND "
                "CE(candidate)-median(CE(D1 top1..4))>=REL_L0."
            ),
        },
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 116)
    print("STATUS:", report["status"])
    print(
        "GRAPH RESCUE ∩ REL_L0 SAFE:",
        f"{len(rescue_safe)}/{len(graph_rescue_qids)}",
        rescue_safe,
    )
    print(
        "D1 SAFE ORACLE:",
        f"{d1_oracle_m['recall']-d1_m['recall']:+.10f}",
    )
    print(
        "PRISM SAFE ORACLE:",
        f"{prism_oracle_m['recall']-prism_m['recall']:+.10f}",
    )
    print(
        "BGE NEXT-STAGE WORKLOAD:",
        f"{pair_count} pairs",
        workload_path,
    )
    print("REPORT:", report_path)
    print("=" * 116)


if __name__ == "__main__":
    main()
