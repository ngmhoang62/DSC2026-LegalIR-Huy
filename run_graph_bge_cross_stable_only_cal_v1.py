#!/usr/bin/env python
"""
GRAPH+BGE CROSS-STABLE-CHALLENGER ONLY — CAL V1
================================================

CPU-only. Reuses the already-sealed Graph+BGE actions.

ONE preregistered safety policy
-------------------------------
Keep a Stage-2 Graph+BGE action ONLY when its chosen challenger is marked
`cross_stable` in the previously sealed Stage-1 graph artifact.

cross_stable means:
  candidate appears in BOTH D1-seeded and Prism-seeded CORE10
  AND
  is supported by both the 1-hop and 2-hop graph rankings in at least one
  seed world.

Everything else abstains.

No threshold search.
No defender gate.
No alternate candidate search.
No query-id rules.
Gold is loaded only after this action subset is sealed.

This directly tests whether the Stage-1 structural-stability signal can improve
the precision of the semantic BGE certificate.

Outputs
-------
results/manual/huy_graph_bge_cross_stable_only_cal_v1/
  CAL_ACTIONS_SEALED.json
  REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

EXPECTED_PRISM_R = 0.9636111111111111
EXPECTED_POP = 600


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def metrics(pred, gold, ids):
    per = {}
    rec, prec = [], []
    for q in ids:
        h = len(set(pred[q]) & gold[q])
        per[q] = h / max(1, len(gold[q]))
        rec.append(per[q])
        prec.append(h / 5.0)
    return float(np.mean(rec)), float(np.mean(prec)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()

    stage2_path = (
        root
        / "results/manual/huy_graph_seed_intersection_bge_cert_cal_v1/"
          "CAL_ACTIONS_SEALED.json"
    )
    graph_path = (
        root
        / "results/manual/huy_prism_anchored_graph_stability_v1/"
          "SEALED_CANDIDATES.jsonl"
    )
    if not stage2_path.is_file():
        raise FileNotFoundError(stage2_path)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    out = (
        root
        / "results/manual/huy_graph_bge_cross_stable_only_cal_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading previously sealed Graph+BGE actions...", flush=True)
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    if stage2.get("status") != "SEALED_BEFORE_CAL_GOLD":
        raise RuntimeError(
            f"Unexpected Stage-2 status: {stage2.get('status')}"
        )
    old_actions = {
        str(q): dict(a)
        for q, a in stage2.get("actions", {}).items()
    }
    print(f"  Stage-2 actions={len(old_actions)}", flush=True)

    print("[2/5] Loading Stage-1 cross-stability provenance...", flush=True)
    graph_rows = {}
    ids = []
    for r in read_jsonl(graph_path):
        q = str(r["qid"])
        ids.append(q)
        graph_rows[q] = r
    if len(ids) != EXPECTED_POP or len(graph_rows) != EXPECTED_POP:
        raise RuntimeError(
            f"Expected {EXPECTED_POP} graph rows, got {len(ids)}/{len(graph_rows)}"
        )

    detail = {}
    for q in ids:
        detail[q] = {
            str(x["doc_id"]): x
            for x in graph_rows[q].get("candidates", [])
        }

    print("[3/5] Sealing CROSS_STABLE-only action subset...", flush=True)
    new_actions = {}
    rejected = {}
    diagnostics = {}

    for q in ids:
        if q not in old_actions:
            diagnostics[q] = {"has_stage2_action": False}
            continue

        a = old_actions[q]
        c = str(a["challenger"])
        d = detail[q].get(c)
        if d is None:
            raise RuntimeError(
                f"Stage-2 challenger absent from Stage-1 details q={q} doc={c}"
            )

        is_cross_stable = bool(d.get("cross_stable"))
        diagnostics[q] = {
            "has_stage2_action": True,
            "challenger": c,
            "cross_seed": bool(d.get("cross_seed")),
            "stable_graph_rank": bool(d.get("stable_graph_rank")),
            "cross_stable": is_cross_stable,
            "dual_direction": bool(d.get("dual_direction")),
        }

        if is_cross_stable:
            new_actions[q] = dict(a)
        else:
            rejected[q] = {
                **a,
                "reject_reason": "CHALLENGER_NOT_CROSS_STABLE",
            }

    seal = {
        "schema": "manual.graph_bge_cross_stable_only_cal_v1.actions",
        "status": "SEALED_BEFORE_CAL_GOLD",
        "policy": {
            "base_actions": (
                "previously sealed "
                "huy_graph_seed_intersection_bge_cert_cal_v1"
            ),
            "new_gate": "chosen challenger must have cross_stable=true",
            "no_threshold_search": True,
            "no_defender_gate": True,
            "no_alternate_candidate_after_reject": True,
            "no_qid_specific_rules": True,
            "K": 5,
        },
        "stage2_source": str(stage2_path),
        "stage2_sha256": sha256(stage2_path),
        "graph_source": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "stage2_actions": len(old_actions),
        "allowed_actions": len(new_actions),
        "rejected_actions": len(rejected),
        "actions": new_actions,
        "rejected": rejected,
        "diagnostics": diagnostics,
    }
    seal_path = out / "CAL_ACTIONS_SEALED.json"
    dump(seal_path, seal)

    print(
        f"  allowed={len(new_actions)} rejected={len(rejected)}",
        flush=True,
    )
    for q, a in new_actions.items():
        print(
            f"    ALLOW q={q} {a['defender']} -> {a['challenger']}",
            flush=True,
        )
    for q, a in rejected.items():
        print(
            f"    REJECT q={q} {a['defender']} -> {a['challenger']}",
            flush=True,
        )

    print("[4/5] Revealing CAL gold AFTER action seal...", flush=True)
    gold_path = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
    )
    raw = json.loads(gold_path.read_text(encoding="utf-8"))
    gold = {
        q: {str(d) for d in raw[q]["answer"]}
        for q in ids
    }

    baseline = {
        q: [str(d) for d in graph_rows[q]["prism_top5"]]
        for q in ids
    }
    candidate = {q: list(baseline[q]) for q in ids}
    for q, a in new_actions.items():
        candidate[q] = [str(d) for d in a["after"]]

    br, bp, bper = metrics(baseline, gold, ids)
    cr, cp, cper = metrics(candidate, gold, ids)
    if abs(br - EXPECTED_PRISM_R) > 1e-9:
        raise RuntimeError(
            f"Prism parity failed: {br} != {EXPECTED_PRISM_R}"
        )

    wins = [q for q in ids if cper[q] > bper[q] + 1e-12]
    losses = [q for q in ids if cper[q] < bper[q] - 1e-12]

    rejected_counterfactual = []
    for q, a in rejected.items():
        rejected_counterfactual.append({
            "qid": q,
            "defender": a["defender"],
            "challenger": a["challenger"],
            "defender_is_gold": a["defender"] in gold[q],
            "challenger_is_gold": a["challenger"] in gold[q],
            "would_have_been_win": (
                a["challenger"] in gold[q]
                and a["defender"] not in gold[q]
            ),
            "would_have_been_loss": (
                a["defender"] in gold[q]
                and a["challenger"] not in gold[q]
            ),
        })

    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    (
        _queries,
        blocks,
        all_ids2,
        _extended,
        _views,
        _channels,
        _gold2,
        _vnlegal,
        _type_rows,
        _cite_rows,
    ) = load_cal_inputs()
    if set(all_ids2) != set(ids):
        raise RuntimeError("CAL block population mismatch")

    block_delta = {
        b: float(
            np.mean([cper[q] for q in qs])
            - np.mean([bper[q] for q in qs])
        )
        for b, qs in blocks.items()
    }

    singles = [q for q in ids if len(gold[q]) == 1]
    multis = [q for q in ids if len(gold[q]) > 1]
    single_delta = float(
        np.mean([cper[q] for q in singles])
        - np.mean([bper[q] for q in singles])
    )
    multi_delta = float(
        np.mean([cper[q] for q in multis])
        - np.mean([bper[q] for q in multis])
    )

    strict_promote = bool(
        cr > br + 1e-12
        and len(wins) > len(losses)
        and len(losses) == 0
        and all(v >= -1e-12 for v in block_delta.values())
    )

    print("[5/5] Writing report...", flush=True)
    report = {
        "schema": "manual.graph_bge_cross_stable_only_cal_v1",
        "policy_sealed_before_gold": True,
        "private_labels_used": False,
        "baseline": {
            "recall": br,
            "precision": bp,
        },
        "candidate": {
            "recall": cr,
            "precision": cp,
            "delta_recall": cr - br,
            "delta_precision": cp - bp,
            "wins": len(wins),
            "losses": len(losses),
            "win_qids": wins,
            "loss_qids": losses,
            "single_delta": single_delta,
            "multi_delta": multi_delta,
            "block_delta": block_delta,
        },
        "action_accounting": {
            "stage2_actions": len(old_actions),
            "allowed": len(new_actions),
            "rejected": len(rejected),
            "rejected_counterfactual": rejected_counterfactual,
        },
        "decision": {
            "strict_promote": strict_promote,
            "status": (
                "PROMOTE_CROSS_STABLE_BGE"
                if strict_promote
                else "DO_NOT_PROMOTE_CROSS_STABLE_BGE"
            ),
        },
        "artifacts": {
            "action_seal": str(seal_path),
            "action_seal_sha256": sha256(seal_path),
        },
    }
    report_path = out / "REPORT.json"
    dump(report_path, report)

    print("=" * 118)
    print(
        f"Prism R={br:.10f} -> CrossStable+BGE R={cr:.10f} "
        f"dR={cr-br:+.10f}"
    )
    print(
        f"Stage2={len(old_actions)} allowed={len(new_actions)} "
        f"rejected={len(rejected)} W/L={len(wins)}/{len(losses)}"
    )
    print(
        f"singleΔ={single_delta:+.6f} "
        f"multiΔ={multi_delta:+.6f}"
    )
    print("blocks:", block_delta)
    print("STATUS:", report["decision"]["status"])
    print("SEALED:", seal_path)
    print("REPORT:", report_path)
    print("=" * 118)


if __name__ == "__main__":
    main()
