#!/usr/bin/env python
"""
HUY DEADLINE — EXACT REL_L0 SAFE-SLOT COMPREHENSIVE V2
======================================================

Authoritative D1 reconstruction only:
  common.load_cal_data()
  evaluate_s0_s1.run_lobo_pipeline()

This is deliberately NOT a hand-reimplementation of D1.

Before any intervention it requires:
  - 48D
  - exact pooled Recall/Precision
  - exact four block recalls
  - ordered Top-5 parity 600/600 with S0_S1_CAL_PREDICTIONS.jsonl
  - exact REL_L0 safe population = 96

Policies on REL_L0-safe queries:
  R6
  R7
  R8
  DIRECT_REF_ONLY
  DIRECT_REF_THEN_R6
  SW_ANY
  SW_POOL

All policies keep authoritative D1 top1-4 unchanged.
The original rank5 is removed only on REL_L0-safe queries.
No public labels are used.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

REL_L0 = -3.0393552780151367
EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
EXPECTED_BLOCKS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "bị","các","có","của","cho","được","để","đến","đối","gì","hay","khi","không",
    "là","làm","một","nào","những","như","phải","ra","sẽ","theo","thì","thế",
    "trong","trên","từ","và","về","với","việc","bao","nhiêu","người","quy","định",
}


def score_file(root: Path, q: str):
    b = root / (
        "results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/"
        "cal_exact_d1_top5_ce_scores_v1"
    )
    for p in (b / f"{q}.json", b / f"{q}.jsonl"):
        if p.is_file():
            return p
    return None


def read_scores(p: Path):
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return {str(k): float(v) for k, v in obj["scores"].items()}
    vals = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                vals[str(k)] = float(v)
            except Exception:
                pass
    if vals:
        return vals
    raise RuntimeError(f"Unknown CE schema: {p}")


def locate_fts_db(root: Path):
    for p in [
        root / "benchmarks/legalir_full_fts.sqlite",
        root.parent / "LegalIR/benchmarks/legalir_full_fts.sqlite",
        root / (
            "results/manual/huy_sparse_fts_query_tokenization_v2/"
            "legalir_full_fts_audit.sqlite"
        ),
    ]:
        if p.is_file():
            return p
    raise FileNotFoundError("No FTS DB found")


def stopword_terms(text: str):
    seen, out = set(), []
    for t in TOKEN_RE.findall((text or "").lower()):
        if t in STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def fts_expr(ts):
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in ts)


def load_authoritative_top5_artifact(root: Path, ids):
    p = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "S0_S1_CAL_PREDICTIONS.jsonl"
    )
    out = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = str(r["qid"])
            if q in ids:
                out[q] = [str(x) for x in r["s0_top5"]]
    return out


def load_direct_refs(root: Path):
    p = (
        root
        / "results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/"
        "CAL_PER_QUERY_EXPANSION.jsonl"
    )
    direct = {}
    if not p.is_file():
        return direct
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = str(r["qid"])
            direct[q] = [
                str(x["doc_id"])
                for x in r.get("addition_details", [])
                if x.get("addition_type") == "DIRECT_REFERENCE_MATCH"
            ]
    return direct


def evaluate_predictions(pred, ids, gold):
    per = {}
    precisions = []
    for q in ids:
        hit = len(set(pred[q]) & set(gold[q]))
        per[q] = hit / max(1, len(gold[q]))
        precisions.append(hit / max(1, len(pred[q])))
    return (
        float(np.mean([per[q] for q in ids])),
        float(np.mean(precisions)),
        per,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from benchmark_burst_v4_full_sqlite import (
        retrieve_docs,
        retrieve_local,
        fuse,
        load_dataset,
    )
    from src.gemini.huy_d1_legal_section_evidence_v1.common import (
        load_cal_data,
        seed_everything,
    )
    from src.gemini.huy_d1_legal_section_evidence_v1.evaluate_s0_s1 import (
        run_lobo_pipeline,
    )

    seed_everything(2026)

    print("[1/6] Loading authoritative CAL world...", flush=True)
    (
        _docs,
        queries,
        blocks,
        ids,
        extended,
        local_views,
        full_channels,
        gold,
        type_rows,
        cite_rows,
    ) = load_cal_data()

    print("[2/6] Running AUTHORITATIVE S0 D1 producer...", flush=True)
    (
        preds,
        dim,
        metrics,
        _scores,
        full_rankings,
        _full_scores,
    ) = run_lobo_pipeline(
        "S0_D1_BASELINE",
        full_channels,
        local_views,
        extended,
        type_rows,
        cite_rows,
        blocks,
        ids,
        gold,
    )

    if dim != 48:
        raise RuntimeError(f"D1 feature dim mismatch: {dim}")
    if abs(metrics["recall_at_5"] - EXPECTED_R) > 1e-12:
        raise RuntimeError(
            f"D1 Recall parity failed: {metrics['recall_at_5']}"
        )
    if abs(metrics["precision_at_5"] - EXPECTED_P) > 1e-12:
        raise RuntimeError(
            f"D1 Precision parity failed: {metrics['precision_at_5']}"
        )
    for b, exp in EXPECTED_BLOCKS.items():
        got = metrics["block_recalls"][b]
        if abs(got - exp) > 1e-9:
            raise RuntimeError(
                f"D1 block {b} parity failed: got={got} expected={exp}"
            )

    artifact = load_authoritative_top5_artifact(root, set(ids))
    bad_order = [q for q in ids if artifact.get(q) != preds[q]]
    bad_set = [
        q for q in ids
        if set(artifact.get(q, [])) != set(preds[q])
    ]
    print(
        f"  D1 artifact parity: ordered={len(ids)-len(bad_order)}/{len(ids)} "
        f"set={len(ids)-len(bad_set)}/{len(ids)}",
        flush=True,
    )
    if bad_order:
        raise RuntimeError(
            "Authoritative producer still disagrees with stored artifact: "
            f"{len(bad_order)} ordered mismatches; sample={bad_order[:5]}"
        )

    print("[3/6] Sealing exact REL_L0 safe population...", flush=True)
    safe = set()
    ce_abstain = []
    for q in ids:
        sf = score_file(root, q)
        if sf is None:
            ce_abstain.append(q)
            continue
        sc = read_scores(sf)
        top = preds[q]
        if any(d not in sc for d in top):
            ce_abstain.append(q)
            continue
        rel = float(
            sc[top[4]] - np.median([sc[d] for d in top[:4]])
        )
        if rel < REL_L0:
            safe.add(q)

    print(
        f"  safe={len(safe)} ce_abstain={len(ce_abstain)}",
        flush=True,
    )
    if len(safe) != 96:
        raise RuntimeError(
            f"REL_L0 safe-population parity failed: {len(safe)} != 96"
        )

    print("[4/6] Loading direct refs + stopword sparse rankings...", flush=True)
    direct = load_direct_refs(root)
    print(
        f"  direct-ref qids={sum(bool(v) for v in direct.values())} "
        f"direct docs={sum(len(v) for v in direct.values())}",
        flush=True,
    )

    db = locate_fts_db(root)
    docs_all, _ = load_dataset(
        root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    )
    doc_ids = [str(d) for d, _ in docs_all]

    sw = {}
    conn = sqlite3.connect(
        f"file:{db.resolve().as_posix()}?mode=ro",
        uri=True,
    )
    try:
        safe_ordered = [q for q in ids if q in safe]
        for i, q in enumerate(safe_ordered, 1):
            e = fts_expr(stopword_terms(queries[q][0]))
            full = retrieve_docs(conn, e, 500)
            local = retrieve_local(conn, e, 2000, second_weight=.3)
            fused = fuse(full, local, local_weight=.9, rrf_k=20)
            sw[q] = [doc_ids[d] for d in fused[:200]]
            if i % 25 == 0 or i == len(safe_ordered):
                print(
                    f"  sparse safe queries {i}/{len(safe_ordered)}",
                    flush=True,
                )
    finally:
        conn.close()

    print("[5/6] Sealing safe-slot policies WITHOUT consulting test gold...", flush=True)
    names = (
        "R6",
        "R7",
        "R8",
        "DIRECT_REF_ONLY",
        "DIRECT_REF_THEN_R6",
        "SW_ANY",
        "SW_POOL",
    )
    policies = {
        name: {q: list(preds[q]) for q in ids}
        for name in names
    }
    provenance = {name: [] for name in names}

    for q in ids:
        if q not in safe:
            continue

        top4 = list(preds[q][:4])
        top5_set = set(preds[q])
        pool = set(extended[q])

        # Authoritative D1 tail.
        for name, pos in (("R6", 5), ("R7", 6), ("R8", 7)):
            if len(full_rankings[q]) > pos:
                c = str(full_rankings[q][pos])
                policies[name][q] = top4 + [c]
                provenance[name].append({
                    "qid": q,
                    "challenger": c,
                    "source": name,
                })

        # Direct reference only: K4 abstain if unavailable.
        dref = next(
            (d for d in direct.get(q, []) if d not in top5_set),
            None,
        )
        if dref is not None:
            policies["DIRECT_REF_ONLY"][q] = top4 + [dref]
            provenance["DIRECT_REF_ONLY"].append({
                "qid": q,
                "challenger": dref,
                "source": "DIRECT_REFERENCE_MATCH",
            })
        else:
            policies["DIRECT_REF_ONLY"][q] = top4

        # Direct ref first, otherwise authoritative rank6.
        if dref is not None:
            c = dref
            src = "DIRECT_REFERENCE_MATCH"
        elif len(full_rankings[q]) > 5:
            c = str(full_rankings[q][5])
            src = "R6_FALLBACK"
        else:
            c = None
            src = None
        if c is not None:
            policies["DIRECT_REF_THEN_R6"][q] = top4 + [c]
            provenance["DIRECT_REF_THEN_R6"].append({
                "qid": q,
                "challenger": c,
                "source": src,
            })

        # Stopword sparse sources.
        sw_any = next(
            (d for d in sw.get(q, []) if d not in top5_set),
            None,
        )
        sw_pool = next(
            (
                d for d in sw.get(q, [])
                if d not in top5_set and d in pool
            ),
            None,
        )

        if sw_any is not None:
            policies["SW_ANY"][q] = top4 + [sw_any]
            provenance["SW_ANY"].append({
                "qid": q,
                "challenger": sw_any,
                "source": "STOPWORD_SPARSE_ANY",
            })
        else:
            policies["SW_ANY"][q] = top4

        if sw_pool is not None:
            policies["SW_POOL"][q] = top4 + [sw_pool]
            provenance["SW_POOL"].append({
                "qid": q,
                "challenger": sw_pool,
                "source": "STOPWORD_SPARSE_POOL",
            })
        else:
            policies["SW_POOL"][q] = top4

    print("[6/6] Evaluating sealed policies...", flush=True)

    br, bp, bper = evaluate_predictions(preds, ids, gold)
    if abs(br - EXPECTED_R) > 1e-12:
        raise RuntimeError(f"Final baseline parity failed: {br}")

    results = {}
    for name, pred in policies.items():
        r, p, per = evaluate_predictions(pred, ids, gold)
        diff = np.asarray(
            [per[q] - bper[q] for q in ids],
            dtype=np.float64,
        )
        wins = [q for q in ids if per[q] > bper[q] + 1e-12]
        losses = [q for q in ids if per[q] < bper[q] - 1e-12]
        block_deltas = {
            b: float(
                np.mean([per[q] for q in qs])
                - np.mean([bper[q] for q in qs])
            )
            for b, qs in blocks.items()
        }
        results[name] = {
            "recall": r,
            "precision_variable_k": p,
            "delta_recall": r - br,
            "wins": len(wins),
            "losses": len(losses),
            "ties": len(ids) - len(wins) - len(losses),
            "win_qids": wins,
            "loss_qids": losses,
            "actions": len(provenance[name]),
            "block_deltas": block_deltas,
            "strict_promote": bool(
                r > br + 1e-12
                and len(losses) == 0
                and all(v >= -1e-12 for v in block_deltas.values())
            ),
        }

    out = (
        root
        / "results/manual/huy_rel_l0_safe_slot_comprehensive_v2_exact"
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps({
            "schema": "manual.rel_l0_safe_slot_comprehensive_v2_exact",
            "authoritative_producer": (
                "huy_d1_legal_section_evidence_v1.run_lobo_pipeline"
            ),
            "safe_queries": len(safe),
            "ce_abstain": ce_abstain,
            "baseline": {
                "recall": br,
                "precision": bp,
                "metrics": metrics,
            },
            "results": results,
            "action_provenance": provenance,
            "public_labels_used": False,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 124)
    print(
        f"D1 R={br:.10f} P={bp:.10f} | "
        f"REL_L0 safe={len(safe)} | parity=600/600"
    )
    for name in names:
        x = results[name]
        print(
            f"{name:<20s} "
            f"R={x['recall']:.10f} "
            f"dR={x['delta_recall']:+.10f} "
            f"W/L/T={x['wins']}/{x['losses']}/{x['ties']} "
            f"actions={x['actions']} "
            f"blocks={x['block_deltas']} "
            f"promote={x['strict_promote']}"
        )
    print("Report:", path)
    print("=" * 124)


if __name__ == "__main__":
    main()
