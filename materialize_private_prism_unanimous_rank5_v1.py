#!/usr/bin/env python
"""
PRISM ROBUST UNANIMOUS RANK5 SWAP — CAL AUDIT + PRIVATE MATERIALIZER V1
======================================================================

CPU-only. No private labels. No neural inference.

Motivation
----------
Full `prism_score_rank` degraded private despite improving CAL, while all Prism
fusion arms churn ~72-76% of private Top5 sets. We therefore stop letting Prism
rewrite the full Top5.

ONE frozen conservative policy
------------------------------
Use the three strongest clean-CAL Prism arms:
  1) prism_score
  2) prism_replace_jina_ft
  3) prism_score_rank

For each query, start from exact D1 Top5 [d1,d2,d3,d4,d5].

Apply ONE swap only when ALL THREE arms:
  - differ from D1 by exactly one set member;
  - remove exactly D1 rank5 (d5);
  - add exactly the SAME challenger c.

Then output:
  [d1,d2,d3,d4,c]

Otherwise keep D1 unchanged.

No threshold search.
No query-id rules.
No private labels.
No score calibration assumptions at private time.
The weaker `prism_replace_crossenc` arm is excluded because its previously
sealed clean-CAL effect is materially weaker (+0.002222 vs +0.004444/+0.006667).

Promotion gate
--------------
CAL OOF policy must satisfy:
  delta Recall > 0
  wins > losses
  losses <= 1
  every block delta >= -0.005

If the gate fails, the script REFUSES to package private submission.

Inputs
------
- clean held-out CAL Prism score artifact
- private top5 artifacts already produced by
  audit_prism_private_transfer_shift_v1.py
- authoritative private D1 v14

Outputs
-------
results/manual/huy_private_prism_unanimous_rank5_v1/
  CAL_REPORT.json
  PRIVATE_ACTIONS.jsonl
  D1_PRIVATE_PRISM_UNANIMOUS_RANK5.json
  D1_PRIVATE_PRISM_UNANIMOUS_RANK5.zip
  REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ARMS = [
    "prism_score",
    "prism_replace_jina_ft",
    "prism_score_rank",
]
EXPECTED_D1_R = 0.9569444444444444
EXPECTED_PRIVATE = 2080


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def zip_exact(json_path: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path, "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.namelist() != ["submission.json"]:
            raise RuntimeError("ZIP member contract failed")
        if zf.read("submission.json") != json_path.read_bytes():
            raise RuntimeError("ZIP byte parity failed")


def metrics(pred, gold, ids):
    per = {}
    rec, prec = [], []
    for q in ids:
        hit = len(set(pred[q]) & set(gold[q]))
        per[q] = hit / max(1, len(gold[q]))
        rec.append(per[q])
        prec.append(hit / 5.0)
    return float(np.mean(rec)), float(np.mean(prec)), per


def consensus_rank5_action(base5, arm5s):
    """
    Return challenger doc iff all arms make the identical exact one-set swap:
      remove base rank5, add same challenger.
    Else return None.
    """
    if len(base5) != 5 or len(set(base5)) != 5:
        raise RuntimeError(f"Invalid D1 Top5: {base5}")

    bset = set(base5)
    defender = base5[4]
    challenger = None

    for top in arm5s:
        if len(top) != 5 or len(set(top)) != 5:
            return None
        aset = set(top)
        entering = aset - bset
        leaving = bset - aset
        if len(entering) != 1 or len(leaving) != 1:
            return None
        if leaving != {defender}:
            return None
        c = next(iter(entering))
        if challenger is None:
            challenger = c
        elif c != challenger:
            return None

    return challenger


def load_private_top5(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for q, row in raw.items():
        docs = row["answer"] if isinstance(row, dict) else row
        out[str(q)] = [str(d) for d in docs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prism-heldout-scores", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    heldout = args.prism_heldout_scores.expanduser().resolve()
    sys.path.insert(0, str(root))

    if not heldout.is_file():
        raise FileNotFoundError(heldout)

    import audit_prism_d1_lobo_v1 as audit

    out = root / "results/manual/huy_private_prism_unanimous_rank5_v1"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Reconstructing clean CAL OOF D1 + three Prism arms...", flush=True)
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )

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

    raw_prism = audit.load_pickle_scores(heldout)
    prism, prism_floor, coverage = audit.aligned_scores(
        raw_prism, cal_ids, extended
    )
    if coverage["coverage"] < .999999:
        raise RuntimeError(f"Prism heldout coverage incomplete: {coverage}")

    results = {}
    for arm in ["d1"] + ARMS:
        res = audit.lobo_arm(
            arm=arm,
            blocks=blocks,
            all_ids=cal_ids,
            candidates=extended,
            local_views=local_views,
            full_channels=full_channels,
            prism=prism,
            type_rows=type_rows,
            cite_rows=cite_rows,
            gold=gold,
        )
        results[arm] = res
        print(
            f"  {arm:28s} R={res['recall']:.10f} "
            f"P={res['precision']:.10f}",
            flush=True,
        )

    base = results["d1"]
    if abs(base["recall"] - EXPECTED_D1_R) > 1e-9:
        raise RuntimeError(
            f"D1 CAL parity failed: {base['recall']} != {EXPECTED_D1_R}"
        )

    print("[2/6] Sealing unanimous exact-rank5 CAL actions...", flush=True)
    cal_pred = {q: list(base["predictions"][q]) for q in cal_ids}
    cal_actions = {}

    for q in cal_ids:
        base5 = list(base["predictions"][q])
        c = consensus_rank5_action(
            base5,
            [results[a]["predictions"][q] for a in ARMS],
        )
        if c is None:
            continue
        cal_pred[q] = base5[:4] + [c]
        cal_actions[q] = {
            "qid": q,
            "defender": base5[4],
            "challenger": c,
            "before": base5,
            "after": cal_pred[q],
            "arm_top5": {
                a: list(results[a]["predictions"][q])
                for a in ARMS
            },
        }

    br, bp, bper = metrics(base["predictions"], gold, cal_ids)
    cr, cp, cper = metrics(cal_pred, gold, cal_ids)
    wins = [q for q in cal_ids if cper[q] > bper[q] + 1e-12]
    losses = [q for q in cal_ids if cper[q] < bper[q] - 1e-12]

    block_delta = {}
    for b, qs in blocks.items():
        block_delta[b] = float(
            np.mean([cper[q] for q in qs])
            - np.mean([bper[q] for q in qs])
        )

    singles = [q for q in cal_ids if len(gold[q]) == 1]
    multis = [q for q in cal_ids if len(gold[q]) > 1]
    single_delta = float(
        np.mean([cper[q] for q in singles])
        - np.mean([bper[q] for q in singles])
    )
    multi_delta = float(
        np.mean([cper[q] for q in multis])
        - np.mean([bper[q] for q in multis])
    )

    promote = bool(
        cr > br + 1e-12
        and len(wins) > len(losses)
        and len(losses) <= 1
        and min(block_delta.values()) >= -0.005 - 1e-12
    )

    cal_report = {
        "schema": "manual.prism_unanimous_rank5_cal_v1",
        "policy": {
            "arms": ARMS,
            "exact_one_set_swap_each_arm": True,
            "all_arms_same_challenger": True,
            "all_arms_remove_exact_d1_rank5": True,
            "freeze_d1_rank1_4": True,
            "no_threshold_search": True,
            "no_qid_rules": True,
        },
        "baseline": {"recall": br, "precision": bp},
        "candidate": {
            "recall": cr,
            "precision": cp,
            "delta_recall": cr - br,
            "delta_precision": cp - bp,
            "actions": len(cal_actions),
            "wins": len(wins),
            "losses": len(losses),
            "win_qids": wins,
            "loss_qids": losses,
            "single_delta": single_delta,
            "multi_delta": multi_delta,
            "block_delta": block_delta,
        },
        "promotion_gate_passed": promote,
        "heldout_prism": str(heldout),
        "heldout_prism_sha256": sha256(heldout),
        "prism_floor": prism_floor,
    }
    cal_report_path = out / "CAL_REPORT.json"
    dump(cal_report_path, cal_report)

    print(
        f"  CAL unanimous actions={len(cal_actions)} "
        f"R={br:.10f}->{cr:.10f} dR={cr-br:+.10f} "
        f"W/L={len(wins)}/{len(losses)}",
        flush=True,
    )
    print(
        f"  singleΔ={single_delta:+.6f} multiΔ={multi_delta:+.6f} "
        f"blocks={block_delta}",
        flush=True,
    )

    if not promote:
        print("=" * 108)
        print("STATUS: REFUSE_PRIVATE_MATERIALIZATION_CAL_GATE_FAILED")
        print("CAL REPORT:", cal_report_path)
        print("=" * 108)
        return

    print("[3/6] Loading exact private D1 + precomputed private Prism arm Top5...", flush=True)
    v14_path = (
        root
        / "results/manual/huy_private_d1_rel_l0_exact_v1/"
          "D1_PRIVATE_V14_FAST.json"
    )
    audit_dir = (
        root
        / "results/manual/huy_prism_private_transfer_shift_v1"
    )

    required = {
        "d1": v14_path,
        "prism_score": audit_dir / "PRIVATE_TOP5_PRISM_SCORE.json",
        "prism_replace_jina_ft":
            audit_dir / "PRIVATE_TOP5_PRISM_REPLACE_JINA_FT.json",
        "prism_score_rank":
            audit_dir / "PRIVATE_TOP5_PRISM_SCORE_RANK.json",
    }
    for name, p in required.items():
        if not p.is_file():
            raise FileNotFoundError(
                f"Required artifact missing [{name}]: {p}"
            )

    d1_private = load_private_top5(v14_path)
    arm_private = {
        a: load_private_top5(required[a])
        for a in ARMS
    }

    ids = list(d1_private)
    if len(ids) != EXPECTED_PRIVATE:
        raise RuntimeError(
            f"Expected {EXPECTED_PRIVATE} private qids, got {len(ids)}"
        )
    for a in ARMS:
        if set(arm_private[a]) != set(ids):
            raise RuntimeError(f"Private arm population mismatch: {a}")

    print("[4/6] Applying the frozen unanimous rank5 policy to private...", flush=True)
    private_pred = {q: list(d1_private[q]) for q in ids}
    private_actions = {}

    for q in ids:
        base5 = list(d1_private[q])
        c = consensus_rank5_action(
            base5,
            [arm_private[a][q] for a in ARMS],
        )
        if c is None:
            continue
        private_pred[q] = base5[:4] + [c]
        private_actions[q] = {
            "qid": q,
            "defender": base5[4],
            "challenger": c,
            "before": base5,
            "after": private_pred[q],
            "arm_top5": {
                a: arm_private[a][q]
                for a in ARMS
            },
        }

    ordered_churn = sum(
        private_pred[q] != d1_private[q]
        for q in ids
    )
    set_churn = sum(
        set(private_pred[q]) != set(d1_private[q])
        for q in ids
    )
    entering = sum(
        len(set(private_pred[q]) - set(d1_private[q]))
        for q in ids
    )

    print(
        f"  PRIVATE actions={len(private_actions)} "
        f"ordered_churn={ordered_churn}/{len(ids)} "
        f"set_churn={set_churn}/{len(ids)} "
        f"enter={entering}",
        flush=True,
    )

    print("[5/6] Validating documents + packaging...", flush=True)
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    valid_docs = {
        p.stem[len("context_"):]
        for p in (data / "selected-contexts").glob("context_*.json")
    }

    submission = {}
    for q in ids:
        ans = [str(d) for d in private_pred[q]]
        if (
            len(ans) != 5
            or len(set(ans)) != 5
            or any(d not in valid_docs for d in ans)
        ):
            raise RuntimeError(f"Invalid private output q={q}: {ans}")
        submission[q] = {"answer": ans}

    json_path = out / "D1_PRIVATE_PRISM_UNANIMOUS_RANK5.json"
    zip_path = out / "D1_PRIVATE_PRISM_UNANIMOUS_RANK5.zip"
    dump(json_path, submission)
    zip_exact(json_path, zip_path)

    actions_path = out / "PRIVATE_ACTIONS.jsonl"
    with actions_path.open("w", encoding="utf-8") as f:
        for q in ids:
            if q in private_actions:
                f.write(
                    json.dumps(
                        private_actions[q],
                        ensure_ascii=False,
                        sort_keys=True,
                    ) + "\n"
                )

    print("[6/6] Writing final report...", flush=True)
    report = {
        "schema": "manual.private_prism_unanimous_rank5_v1",
        "status": "READY_FOR_PRIVATE_SUBMISSION",
        "scientific_contract": {
            "private_labels_used": False,
            "private_leaderboard_scores_used_for_query_selection": False,
            "policy_selected_before_private_query outcomes": True,
            "D1_rank1_4_frozen": True,
            "only_exact_unanimous_rank5_swaps": True,
        },
        "cal": cal_report,
        "private": {
            "queries": len(ids),
            "actions": len(private_actions),
            "action_rate": len(private_actions) / len(ids),
            "ordered_churn": ordered_churn,
            "set_churn": set_churn,
            "entering_docs": entering,
            "d1_v14": str(v14_path),
            "d1_v14_sha256": sha256(v14_path),
            "source_top5_artifacts": {
                k: {
                    "path": str(p),
                    "sha256": sha256(p),
                }
                for k, p in required.items()
            },
        },
        "submission": {
            "json": str(json_path),
            "json_sha256": sha256(json_path),
            "zip": str(zip_path),
            "zip_sha256": sha256(zip_path),
            "actions": str(actions_path),
            "actions_sha256": sha256(actions_path),
        },
    }

    report_path = out / "REPORT.json"
    dump(report_path, report)

    print("=" * 108)
    print("STATUS: READY_FOR_PRIVATE_SUBMISSION")
    print(
        f"CAL dR={cr-br:+.10f} W/L={len(wins)}/{len(losses)} "
        f"| PRIVATE ACTIONS={len(private_actions)}/{len(ids)}"
    )
    print("ZIP:", zip_path)
    print("SHA256:", sha256(zip_path))
    print("REPORT:", report_path)
    print("=" * 108)


if __name__ == "__main__":
    main()
