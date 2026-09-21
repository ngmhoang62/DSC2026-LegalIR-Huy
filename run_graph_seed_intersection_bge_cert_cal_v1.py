#!/usr/bin/env python
"""
GRAPH SEED-INTERSECTION + FROZEN BGE CHALLENGER CERTIFICATE — CAL V1
====================================================================

Goal
----
Take the label-free SEED_INTERSECTION population sealed by
audit_prism_anchored_graph_stability_v1.py and test ONE preregistered admission
policy against the stronger Prism score+rank Top5 baseline.

Why this is materially different from REL_L0 x graph
-----------------------------------------------------
REL_L0 x graph used defender weakness as a QUERY GATE and lost 4/5 graph rescue
queries. Here REL_L0 is used only as a CHALLENGER ABSOLUTE CERTIFICATE:

    BGE(challenger) > BGE(Prism rank5 defender)
    AND
    BGE(challenger) - median(BGE(Prism ranks1..4)) >= REL_L0

No defender-weakness gate is used.

Candidate source
----------------
SEED_INTERSECTION:
    graph candidate appears in BOTH the D1-seeded CORE10 and Prism-seeded CORE10
    populations.

The Stage-1 CAL audit showed this frozen population preserved all five core
outside-pool rescues while reducing proposal count relative to the union.

Frozen candidate order
----------------------
Within SEED_INTERSECTION:
  1) CROSS_STABLE candidates first:
       cross-seed AND supported by both 1-hop / 2-hop graph rankings in at least
       one seed world;
  2) remaining SEED_INTERSECTION candidates;
  3) preserve original sealed order inside each group.

The first candidate passing the BGE certificate replaces Prism rank5. K=5.

No threshold search.
No query-id rules.
CAL gold is read only AFTER actions are sealed.

Expected baseline parity
------------------------
Prism score+rank Recall@5 = 0.9636111111111111

Inputs
------
results/manual/huy_prism_anchored_graph_stability_v1/
  SEALED_CANDIDATES.jsonl

Frozen CE infrastructure reused from previous endgame experiments:
  results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/training/model.pt
  ../LegalIR
  ../run_noncal_trainable_ce_boundary_v3_fixed.py

Outputs
-------
results/manual/huy_graph_seed_intersection_bge_cert_cal_v1/
  GRAPH_BGE_SCORES/*.json
  CAL_ACTIONS_SEALED.json
  REPORT.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REL_L0 = -3.0393552780151367
EXPECTED_PRISM_R = 0.9636111111111111
EXPECTED_POP = 600


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_sealed_graph(root: Path):
    path = (
        root
        / "results/manual/huy_prism_anchored_graph_stability_v1/"
          "SEALED_CANDIDATES.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = {}
    order = []
    for r in read_jsonl(path):
        q = str(r["qid"])
        order.append(q)
        rows[q] = r

    if len(order) != EXPECTED_POP or len(rows) != EXPECTED_POP:
        raise RuntimeError(
            f"Expected {EXPECTED_POP} sealed rows, got {len(order)}/{len(rows)}"
        )

    # Validate that the arm list stored in the row exactly matches the detail
    # flags. Then construct the ONE preregistered structural-priority order.
    candidate_order = {}
    for q in order:
        r = rows[q]
        prism_top5 = [str(d) for d in r["prism_top5"]]
        if len(prism_top5) != 5 or len(set(prism_top5)) != 5:
            raise RuntimeError(f"Invalid Prism Top5 q={q}: {prism_top5}")

        sealed_inter = [
            str(d) for d in r.get("arms", {}).get("SEED_INTERSECTION", [])
        ]

        detail_by_doc = {
            str(x["doc_id"]): x for x in r.get("candidates", [])
        }
        reconstructed = [
            str(x["doc_id"])
            for x in r.get("candidates", [])
            if bool(x.get("cross_seed"))
        ]
        if reconstructed != sealed_inter:
            raise RuntimeError(
                f"SEED_INTERSECTION detail/order drift q={q}"
            )

        stable = [
            d for d in sealed_inter
            if bool(detail_by_doc[d].get("cross_stable"))
        ]
        rest = [d for d in sealed_inter if d not in set(stable)]
        ordered = stable + rest

        if len(ordered) != len(set(ordered)):
            raise RuntimeError(f"Duplicate graph challenger q={q}")

        candidate_order[q] = ordered

    return order, rows, candidate_order, path


def load_existing_d1_top5_bge(root: Path, q: str):
    """
    Reuse already-computed exact-D1 top5 scores opportunistically.
    Prism Top5 may contain different docs; missing Prism docs are scored below
    with the identical frozen Fold0 BGE checkpoint.
    """
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
    return {}


@torch.inference_mode()
def score_needed_pairs(
    *,
    root: Path,
    sibling: Path,
    world,
    ids,
    rows,
    challengers,
    out: Path,
    pair_microbatch: int,
):
    sys.path[:0] = [str(sibling), str(sibling / "src")]
    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence

    ckpt = (
        root
        / "results/manual/huy_noncal_trainable_ce_boundary_v1/"
          "oof/fold_0/training/model.pt"
    )
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    model = CrossEncoder(ckpt)
    model.eval()
    evidence = Evidence(world["render"], model.tokenizer)

    score_dir = out / "GRAPH_BGE_SCORES"
    score_dir.mkdir(parents=True, exist_ok=True)

    eligible = set(map(str, world["cal_overlap"]))
    results = {}
    abstain = {}
    pair_total = 0
    pair_scored = 0
    started = time.perf_counter()

    try:
        for i, q in enumerate(ids, 1):
            if q not in eligible:
                abstain[q] = {"reason": "CAL_OUTSIDE_V2"}
                continue

            prism_top5 = [str(d) for d in rows[q]["prism_top5"]]
            graph_docs = list(challengers[q])
            needed_order = list(dict.fromkeys(prism_top5 + graph_docs))
            pair_total += len(needed_order)

            # Reuse exact-D1 Top5 BGE values if a Prism/graph doc overlaps them.
            scores = load_existing_d1_top5_bge(root, q)
            scores = {
                d: float(scores[d])
                for d in needed_order
                if d in scores
            }

            missing_evidence = []
            to_score = []
            for d in needed_order:
                if d in scores:
                    continue
                ok = evidence.db.execute(
                    "SELECT 1 FROM chunks WHERE doc=? LIMIT 1", (d,)
                ).fetchone() is not None
                if ok:
                    to_score.append(d)
                else:
                    missing_evidence.append(d)

            # Baseline Prism Top5 must be fully scoreable, otherwise no action.
            missing_baseline = [d for d in prism_top5 if d in missing_evidence]
            if missing_baseline:
                abstain[q] = {
                    "reason": "PRISM_TOP5_MISSING_FROZEN_EVIDENCE",
                    "docs": missing_baseline,
                }
                continue

            cache_path = score_dir / f"{q}.json"
            sig = hashlib.sha256(
                json.dumps(
                    [
                        "graph-seed-intersection-bge-cert-cal-v1",
                        sha256(ckpt),
                        q,
                        needed_order,
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            cached = {}
            if cache_path.is_file():
                obj = json.loads(cache_path.read_text(encoding="utf-8"))
                if obj.get("signature") != sig:
                    raise RuntimeError(f"BGE graph cache signature drift q={q}")
                cached = {
                    str(d): float(s)
                    for d, s in obj.get("scores", {}).items()
                }

            still = [d for d in to_score if d not in cached]
            if still:
                vals = []
                for st in range(0, len(still), pair_microbatch):
                    docs = still[st:st + pair_microbatch]
                    pairs = [evidence.package(q, d) for d in docs]
                    vals.extend(model(pairs).detach().cpu().tolist())
                cached.update({
                    d: float(s) for d, s in zip(still, vals)
                })
                pair_scored += len(still)

            # Cache only model-computed values; reused D1-cache values remain
            # external but same frozen checkpoint.
            dump(
                cache_path,
                {
                    "signature": sig,
                    "scores": cached,
                    "missing_evidence": missing_evidence,
                },
            )
            scores.update(cached)

            # Missing graph challenger evidence -> skip only that challenger.
            results[q] = {
                "scores": {
                    d: float(scores[d])
                    for d in needed_order
                    if d in scores
                },
                "missing_graph_evidence": [
                    d for d in graph_docs if d not in scores
                ],
            }

            if i % 50 == 0 or i == len(ids):
                elapsed = time.perf_counter() - started
                print(
                    f"  BGE {i}/{len(ids)} usable={len(results)} "
                    f"abstain={len(abstain)} newly_scored={pair_scored} "
                    f"qps={i/max(elapsed,1e-9):.2f}",
                    flush=True,
                )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect()
        torch.cuda.empty_cache()

    return results, abstain, ckpt, pair_total, pair_scored


def metrics(pred, gold, ids):
    per = {}
    recalls = []
    precisions = []
    for q in ids:
        h = len(set(pred[q]) & gold[q])
        per[q] = h / max(1, len(gold[q]))
        recalls.append(per[q])
        precisions.append(h / 5.0)
    return float(np.mean(recalls)), float(np.mean(precisions)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--pair-microbatch", type=int, default=4)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sibling = root.parent / "LegalIR"
    base_script = root.parent / "run_noncal_trainable_ce_boundary_v3_fixed.py"

    if not sibling.is_dir():
        raise FileNotFoundError(
            f"Expected sibling LegalIR directory: {sibling}"
        )
    if not base_script.is_file():
        raise FileNotFoundError(
            f"Expected frozen CE world loader: {base_script}"
        )

    out = (
        root
        / "results/manual/huy_graph_seed_intersection_bge_cert_cal_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading sealed Stage-1 graph artifact...", flush=True)
    ids, rows, candidate_order, sealed_path = load_sealed_graph(root)

    total = sum(len(candidate_order[q]) for q in ids)
    stable_first = sum(
        1
        for q in ids
        for d in candidate_order[q]
        if next(
            x for x in rows[q]["candidates"]
            if str(x["doc_id"]) == d
        ).get("cross_stable")
    )
    print(
        f"  population={len(ids)} "
        f"SEED_INTERSECTION pairs={total} "
        f"CROSS_STABLE members={stable_first}",
        flush=True,
    )
    if total != 2729:
        print(
            f"  WARNING: expected historical Stage-1 pair count 2729, got {total}. "
            "Continuing because the sealed artifact is authoritative.",
            flush=True,
        )

    print("[2/7] Loading frozen non-CAL Fold0 BGE world...", flush=True)
    m = loadmod(base_script, "cebase_graph_intersection")
    cal_ids, _ = m.get_cal_ids_label_free(root)
    world = m.load_noncal_world(root, sibling, set(cal_ids))
    eligible = set(map(str, world["cal_overlap"]))
    print(
        f"  frozen BGE CAL-overlap={len(eligible)}/{len(ids)}",
        flush=True,
    )

    print("[3/7] Scoring Prism Top5 + graph challengers...", flush=True)
    (
        score_rows,
        abstain,
        ckpt,
        pair_total,
        pair_scored,
    ) = score_needed_pairs(
        root=root,
        sibling=sibling,
        world=world,
        ids=ids,
        rows=rows,
        challengers=candidate_order,
        out=out,
        pair_microbatch=args.pair_microbatch,
    )

    print("[4/7] Sealing ONE challenger policy BEFORE CAL gold...", flush=True)
    actions = {}
    diagnostics = {}

    for q in ids:
        prism_top5 = [str(d) for d in rows[q]["prism_top5"]]
        diag = {
            "prism_top5": prism_top5,
            "candidate_order": candidate_order[q],
            "candidate_order_rule": (
                "CROSS_STABLE members first, then remaining SEED_INTERSECTION; "
                "preserve sealed order within each group"
            ),
        }

        if q in abstain:
            diag["abstain"] = abstain[q]
            diagnostics[q] = diag
            continue
        if q not in score_rows:
            diag["abstain"] = {"reason": "NO_BGE_SCORE_ROW"}
            diagnostics[q] = diag
            continue

        scores = score_rows[q]["scores"]
        if any(d not in scores for d in prism_top5):
            diag["abstain"] = {
                "reason": "INCOMPLETE_PRISM_TOP5_BGE",
                "missing": [d for d in prism_top5 if d not in scores],
            }
            diagnostics[q] = diag
            continue

        med = float(np.median([scores[d] for d in prism_top5[:4]]))
        defender = prism_top5[4]
        defender_score = float(scores[defender])
        defender_rel = defender_score - med

        cert_rows = []
        for idx, d in enumerate(candidate_order[q]):
            if d not in scores:
                cert_rows.append({
                    "index": idx,
                    "doc": d,
                    "has_frozen_evidence": False,
                    "certified": False,
                })
                continue

            s = float(scores[d])
            rel = s - med
            beats = s > defender_score
            absolute = rel >= REL_L0
            cert_rows.append({
                "index": idx,
                "doc": d,
                "bge_score": s,
                "bge_rel_to_prism_top4_median": rel,
                "beats_prism_rank5": bool(beats),
                "passes_frozen_rel_l0_candidate_certificate": bool(absolute),
                "has_frozen_evidence": True,
                "certified": bool(beats and absolute),
            })

        first = next((x for x in cert_rows if x["certified"]), None)

        diag.update({
            "median_prism_top4_bge": med,
            "defender": defender,
            "defender_bge": defender_score,
            "defender_rel": defender_rel,
            "candidate_certificates": cert_rows,
            "missing_graph_evidence": score_rows[q]["missing_graph_evidence"],
        })

        if first is not None:
            c = first["doc"]
            actions[q] = {
                "qid": q,
                "defender": defender,
                "challenger": c,
                "before": prism_top5,
                "after": prism_top5[:4] + [c],
                "candidate_index": int(first["index"]),
                "defender_rel": defender_rel,
                "challenger_rel": first["bge_rel_to_prism_top4_median"],
                "challenger_bge": first["bge_score"],
            }

        diagnostics[q] = diag

    seal = {
        "schema": "manual.graph_seed_intersection_bge_cert_cal_v1.actions",
        "status": "SEALED_BEFORE_CAL_GOLD",
        "policy": {
            "baseline": "Prism score+rank OOF Top5",
            "candidate_source": "sealed SEED_INTERSECTION",
            "candidate_order": (
                "CROSS_STABLE first; then remaining SEED_INTERSECTION; "
                "preserve Stage-1 sealed order"
            ),
            "defender_weakness_gate": None,
            "require_challenger_bge_gt_prism_rank5": True,
            "challenger_rel_threshold": REL_L0,
            "K": 5,
            "first_certified_candidate_wins": True,
            "no_threshold_search": True,
            "no_qid_specific_rules": True,
        },
        "sealed_graph_source": str(sealed_path),
        "sealed_graph_sha256": sha256(sealed_path),
        "bge_checkpoint": str(ckpt),
        "bge_checkpoint_sha256": sha256(ckpt),
        "eligible_frozen_bge_queries": len(eligible),
        "pair_total_requested": pair_total,
        "pair_newly_scored": pair_scored,
        "actions_count": len(actions),
        "actions": actions,
        "diagnostics": diagnostics,
    }
    seal_path = out / "CAL_ACTIONS_SEALED.json"
    dump(seal_path, seal)

    print(
        f"  SEALED actions={len(actions)} "
        f"BGE-eligible={len(eligible)}",
        flush=True,
    )
    for q, a in list(actions.items())[:30]:
        print(
            f"    q={q} Prism-r5 {a['defender']} -> {a['challenger']} "
            f"rel {a['defender_rel']:+.3f}->{a['challenger_rel']:+.3f} "
            f"idx={a['candidate_index']}",
            flush=True,
        )
    if len(actions) > 30:
        print(f"    ... +{len(actions)-30} more actions", flush=True)

    print("[5/7] Revealing CAL gold AFTER action seal...", flush=True)
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
        q: [str(d) for d in rows[q]["prism_top5"]]
        for q in ids
    }
    candidate = {q: list(baseline[q]) for q in ids}
    for q, a in actions.items():
        candidate[q] = list(a["after"])

    br, bp, bper = metrics(baseline, gold, ids)
    cr, cp, cper = metrics(candidate, gold, ids)

    print(
        f"  Prism baseline R={br:.10f} P={bp:.10f}",
        flush=True,
    )
    if abs(br - EXPECTED_PRISM_R) > 1e-9:
        raise RuntimeError(
            f"Prism baseline parity failed: {br} != {EXPECTED_PRISM_R}"
        )

    wins = [q for q in ids if cper[q] > bper[q] + 1e-12]
    losses = [q for q in ids if cper[q] < bper[q] - 1e-12]

    # Forensic action outcome only after policy was sealed.
    action_outcomes = []
    for q, a in actions.items():
        action_outcomes.append({
            **a,
            "defender_is_gold": a["defender"] in gold[q],
            "challenger_is_gold": a["challenger"] in gold[q],
            "delta_query_recall": cper[q] - bper[q],
        })

    print("[6/7] Measuring block / single-multi behavior...", flush=True)
    # Obtain authoritative CAL blocks only now, after seal.
    sys.path.insert(0, str(root))
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    (
        _queries,
        blocks,
        all_ids2,
        _extended,
        _local_views,
        _channels,
        _gold2,
        _vnlegal,
        _type_rows,
        _cite_rows,
    ) = load_cal_inputs()
    if set(all_ids2) != set(ids):
        raise RuntimeError("CAL population drift when loading blocks")

    block_delta = {}
    for b, qs in blocks.items():
        base_b = float(np.mean([bper[q] for q in qs]))
        cand_b = float(np.mean([cper[q] for q in qs]))
        block_delta[b] = cand_b - base_b

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
        and len(losses) <= 1
        and min(block_delta.values()) >= -0.005 - 1e-12
    )

    print("[7/7] Writing report...", flush=True)
    report = {
        "schema": "manual.graph_seed_intersection_bge_cert_cal_v1",
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
        "actions": {
            "count": len(actions),
            "outcomes": action_outcomes,
        },
        "decision": {
            "strict_promote": strict_promote,
            "status": (
                "PROMOTE_GRAPH_BGE_CERTIFICATE"
                if strict_promote else
                "DO_NOT_PROMOTE_GRAPH_BGE_CERTIFICATE"
            ),
            "rule": (
                "positive global recall; wins>losses; at most one loss; "
                "worst block delta >= -0.005"
            ),
        },
        "artifacts": {
            "actions_seal": str(seal_path),
            "actions_seal_sha256": sha256(seal_path),
            "sealed_graph": str(sealed_path),
            "sealed_graph_sha256": sha256(sealed_path),
            "gold_path": str(gold_path),
            "gold_sha256": sha256(gold_path),
        },
    }

    report_path = out / "REPORT.json"
    dump(report_path, report)

    print("=" * 118)
    print(
        f"Prism R={br:.10f} -> Graph+BGE R={cr:.10f} "
        f"dR={cr-br:+.10f}"
    )
    print(
        f"W/L={len(wins)}/{len(losses)} "
        f"singleΔ={single_delta:+.6f} multiΔ={multi_delta:+.6f}"
    )
    print("blocks:", block_delta)
    print("STATUS:", report["decision"]["status"])
    print("SEALED:", seal_path)
    print("REPORT:", report_path)
    print("=" * 118)


if __name__ == "__main__":
    main()
