#!/usr/bin/env python
"""
GRAPH+BGE WITH D1 DEFENDER-STABILITY VETO — CAL V1
===================================================

CPU-only. Reuses the already-sealed Graph+BGE action set.

Motivation
----------
Stage-2 result:
  Prism baseline R = 0.9636111111
  Graph+BGE        = 0.9625000000
  W/L              = 1/1

Acquisition is still strong:
  SEED_INTERSECTION outside-pool oracle headroom = +0.004722

Therefore the remaining failure mode is not acquisition but unsafe eviction:
BGE can identify a plausible graph challenger yet still replace a genuinely
relevant Prism rank5 defender.

ONE preregistered safety policy
-------------------------------
Start from the exact Stage-2 Graph+BGE SEALED actions.

For each sealed action:
  - challenger already passed:
      BGE(challenger) > BGE(Prism rank5)
      AND challenger REL >= frozen REL_L0
  - NEW VETO:
      if the Prism rank5 defender also appears anywhere in exact OOF D1 Top5,
      ABSTAIN.
      Otherwise allow the existing sealed replacement unchanged.

Interpretation
--------------
The challenger is structurally stable (SEED_INTERSECTION) and semantically
certified by frozen Fold0 BGE. We now additionally require the incumbent to be
unstable across two selectors.

This is NOT the closed consensus-reranking branch:
  - D1 is not used to rerank the pool;
  - D1 does not choose the challenger;
  - D1 supplies only a binary veto on evicting an independently-selected
    Prism rank5 defender.

No threshold search.
No query-id rules.
No alternative candidate search after a veto.
Gold is loaded only AFTER this new action subset is sealed.

Inputs
------
results/manual/huy_graph_seed_intersection_bge_cert_cal_v1/
  CAL_ACTIONS_SEALED.json

results/manual/huy_prism_anchored_graph_stability_v1/
  SEALED_CANDIDATES.jsonl

Outputs
-------
results/manual/huy_graph_bge_d1_defender_veto_cal_v1/
  CAL_ACTIONS_SEALED.json
  REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    rec = []
    prec = []
    for q in ids:
        hit = len(set(pred[q]) & gold[q])
        per[q] = hit / max(1, len(gold[q]))
        rec.append(per[q])
        prec.append(hit / 5.0)
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
        / "results/manual/huy_graph_bge_d1_defender_veto_cal_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading previously SEALED Graph+BGE actions...", flush=True)
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    if stage2.get("status") != "SEALED_BEFORE_CAL_GOLD":
        raise RuntimeError(
            f"Unexpected Stage-2 seal status: {stage2.get('status')}"
        )
    old_actions = {
        str(q): dict(a)
        for q, a in stage2.get("actions", {}).items()
    }
    print(f"  Stage-2 sealed actions={len(old_actions)}", flush=True)

    print("[2/5] Loading label-free D1/Prism Top5 stability world...", flush=True)
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

    # Seal the new action subset WITHOUT touching gold.
    print("[3/5] Applying preregistered D1 defender-stability veto...", flush=True)
    new_actions = {}
    vetoed = {}
    diagnostics = {}

    for q in ids:
        row = graph_rows[q]
        prism_top5 = [str(d) for d in row["prism_top5"]]
        d1_top5 = [str(d) for d in row["d1_top5"]]

        diag = {
            "d1_top5": d1_top5,
            "prism_top5": prism_top5,
            "has_stage2_action": q in old_actions,
        }

        if q not in old_actions:
            diagnostics[q] = diag
            continue

        a = old_actions[q]
        defender = str(a["defender"])
        challenger = str(a["challenger"])

        if defender != prism_top5[4]:
            raise RuntimeError(
                f"Stage-2 defender != Prism rank5 q={q}: "
                f"{defender} != {prism_top5[4]}"
            )

        defender_in_d1_top5 = defender in set(d1_top5)
        defender_d1_rank = (
            d1_top5.index(defender) + 1
            if defender_in_d1_top5 else None
        )

        diag.update({
            "defender": defender,
            "challenger": challenger,
            "defender_in_d1_top5": defender_in_d1_top5,
            "defender_d1_rank_if_top5": defender_d1_rank,
        })

        if defender_in_d1_top5:
            vetoed[q] = {
                **a,
                "veto_reason": "PRISM_RANK5_ALSO_IN_D1_TOP5",
                "defender_d1_rank": defender_d1_rank,
            }
        else:
            new_actions[q] = dict(a)

        diagnostics[q] = diag

    seal = {
        "schema": "manual.graph_bge_d1_defender_veto_cal_v1.actions",
        "status": "SEALED_BEFORE_CAL_GOLD",
        "policy": {
            "base_actions": (
                "previously sealed "
                "huy_graph_seed_intersection_bge_cert_cal_v1 actions"
            ),
            "new_veto": (
                "abstain iff Prism rank5 defender appears anywhere in exact "
                "OOF D1 Top5"
            ),
            "no_threshold_search": True,
            "no_qid_specific_rules": True,
            "no_alternate_candidate_after_veto": True,
            "K": 5,
        },
        "stage2_source": str(stage2_path),
        "stage2_sha256": sha256(stage2_path),
        "graph_source": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "stage2_actions": len(old_actions),
        "allowed_actions": len(new_actions),
        "vetoed_actions": len(vetoed),
        "actions": new_actions,
        "vetoed": vetoed,
        "diagnostics": diagnostics,
    }
    seal_path = out / "CAL_ACTIONS_SEALED.json"
    dump(seal_path, seal)

    print(
        f"  allowed={len(new_actions)} vetoed={len(vetoed)}",
        flush=True,
    )
    for q, a in vetoed.items():
        print(
            f"    VETO q={q} defender={a['defender']} "
            f"D1rank={a['defender_d1_rank']} "
            f"challenger={a['challenger']}",
            flush=True,
        )
    for q, a in new_actions.items():
        print(
            f"    ALLOW q={q} defender={a['defender']} "
            f"challenger={a['challenger']}",
            flush=True,
        )

    print("[4/5] Revealing CAL gold AFTER the new action seal...", flush=True)
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
            f"Prism baseline parity failed: {br} != {EXPECTED_PRISM_R}"
        )

    wins = [q for q in ids if cper[q] > bper[q] + 1e-12]
    losses = [q for q in ids if cper[q] < bper[q] - 1e-12]

    # Outcome for both allowed and vetoed actions, for post-seal forensics.
    allowed_outcomes = []
    for q, a in new_actions.items():
        allowed_outcomes.append({
            "qid": q,
            "defender": a["defender"],
            "challenger": a["challenger"],
            "defender_is_gold": a["defender"] in gold[q],
            "challenger_is_gold": a["challenger"] in gold[q],
            "delta_query_recall": cper[q] - bper[q],
        })

    vetoed_counterfactual = []
    for q, a in vetoed.items():
        defender = str(a["defender"])
        challenger = str(a["challenger"])
        vetoed_counterfactual.append({
            "qid": q,
            "defender": defender,
            "challenger": challenger,
            "defender_is_gold": defender in gold[q],
            "challenger_is_gold": challenger in gold[q],
            "would_have_been_win": (
                challenger in gold[q] and defender not in gold[q]
            ),
            "would_have_been_loss": (
                defender in gold[q] and challenger not in gold[q]
            ),
            "defender_d1_rank": a["defender_d1_rank"],
        })

    # Load block information only after seal and outcome reveal.
    import sys
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

    block_delta = {}
    for b, qs in blocks.items():
        block_delta[b] = float(
            np.mean([cper[q] for q in qs])
            - np.mean([bper[q] for q in qs])
        )

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
        "schema": "manual.graph_bge_d1_defender_veto_cal_v1",
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
            "vetoed": len(vetoed),
            "allowed_outcomes": allowed_outcomes,
            "vetoed_counterfactual": vetoed_counterfactual,
        },
        "decision": {
            "strict_promote": strict_promote,
            "status": (
                "PROMOTE_D1_DEFENDER_VETO"
                if strict_promote
                else "DO_NOT_PROMOTE_D1_DEFENDER_VETO"
            ),
            "gate": (
                "positive Recall, wins>losses, zero losses, "
                "no negative block delta"
            ),
        },
        "artifacts": {
            "new_action_seal": str(seal_path),
            "new_action_seal_sha256": sha256(seal_path),
            "stage2_action_seal": str(stage2_path),
            "stage2_action_seal_sha256": sha256(stage2_path),
        },
    }
    report_path = out / "REPORT.json"
    dump(report_path, report)

    print("=" * 118)
    print(
        f"Prism R={br:.10f} -> D1-veto R={cr:.10f} "
        f"dR={cr-br:+.10f}"
    )
    print(
        f"Stage2 actions={len(old_actions)} "
        f"allowed={len(new_actions)} vetoed={len(vetoed)} "
        f"W/L={len(wins)}/{len(losses)}"
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
