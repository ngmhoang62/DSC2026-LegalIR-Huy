#!/usr/bin/env python
"""
OUTSIDE-POOL E5/AITEAM CONSENSUS + SYMMETRIC REL_L0 CERTIFICATE — CAL V1
========================================================================

ONE preregistered policy:

1) exact D1 rank5 defender must be REL_L0-weak:
      CE(defender) - median(CE(rank1..4)) < REL_L0

2) challenger must be OUTSIDE exact D1 candidate pool and retrieved by BOTH
   adapted-E5 full corpus and AITeam-FT full corpus:
      rank_e5 <= 50
      rank_ai <= 50
      min(rank_e5, rank_ai) <= 10

3) challenger must pass the SAME semantic certificate:
      CE(challenger) - median(CE(rank1..4)) >= REL_L0

4) exactly one challenger may satisfy all rules; otherwise abstain.

Then replace exact D1 rank5 only. K remains exactly 5.

No CAL labels are read until CAL_ACTIONS_SEALED.json is written.
No threshold search / no alternate arm selection.

REL_L0 is frozen from nonCAL Fold0 DEV:
  -3.0393552780151367
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REL_L0 = -3.0393552780151367
EXPECTED_D1_R = 0.9569444444444444
EXPECTED_D1_P = 0.20566666666666666

MAX_E5_RANK = 50
MAX_AI_RANK = 50
MIN_SOURCE_RANK_MAX = 10


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def load_exact_d1_top5_whitelist(root: Path):
    """
    Authoritative historical artifact co-locates gold, so whitelist-extract only
    qid and s0_top5 from raw text; never json.loads() whole lines pre-seal.
    """
    path = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "S0_S1_CAL_PREDICTIONS.jsonl"
    )
    qid_re = re.compile(r'"qid"\s*:\s*"([^"]+)"')
    top5_re = re.compile(r'"s0_top5"\s*:\s*(\[[^\]]*\])')
    out = {}
    order = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            qm = qid_re.search(line)
            tm = top5_re.search(line)
            if qm is None or tm is None:
                raise RuntimeError(f"Whitelist parse failed {path}:{lineno}")
            q = str(qm.group(1))
            docs = [str(x) for x in json.loads(tm.group(1))]
            if len(docs) != 5 or len(set(docs)) != 5:
                raise RuntimeError(f"Invalid D1 Top5 q={q}: {docs}")
            out[q] = docs
            order.append(q)
    if len(out) != 600:
        raise RuntimeError(f"Expected 600 exact D1 queries, got {len(out)}")
    return order, out, path


def load_e5_full_corpus_whitelist(root: Path):
    """
    E5 OOF full-corpus rows co-locate gold. Whitelist-extract qid and
    adapted_order_top150 only.
    """
    qid_re = re.compile(r'"qid"\s*:\s*"([^"]+)"')
    rank_re = re.compile(r'"adapted_order_top150"\s*:\s*(\[[^\]]*\])')
    out = {}
    paths = []
    for i in range(5):
        p = (
            root
            / f"results/research_v2_open_rl/fold_{i}/"
            "FULL_CORPUS_PREDICTIONS.jsonl"
        )
        if not p.is_file():
            raise FileNotFoundError(p)
        paths.append(p)
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                qm = qid_re.search(line)
                rm = rank_re.search(line)
                if qm is None or rm is None:
                    raise RuntimeError(f"E5 whitelist parse failed {p}:{lineno}")
                q = str(qm.group(1))
                docs = [str(x) for x in json.loads(rm.group(1))]
                if len(docs) < 50:
                    raise RuntimeError(f"E5 <50 docs q={q}")
                if q in out:
                    raise RuntimeError(f"Duplicate E5 qid={q}")
                out[q] = docs
    return out, paths


def load_aiteam(root: Path):
    p = root / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_TOP50.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    raw = json.loads(p.read_text(encoding="utf-8"))
    out = {
        str(q): [str(d) for d in docs]
        for q, docs in raw.items()
    }
    if any(len(v) < 50 for v in out.values()):
        bad = [q for q, v in out.items() if len(v) < 50][:10]
        raise RuntimeError(f"AITeam Top50 short rows sample={bad}")
    return out, p


def load_label_free_pool(root: Path):
    """
    Reuse the prior label-free CAL loader; it reconstructs exact extended pool
    from retrieval caches while using CAL_QUESTIONS_LABEL_FREE.
    """
    sys.path.insert(0, str(root))
    from src.gemini.huy_d1_aiteam_novel_consensus_v1.common import (
        load_cal_data_label_free,
    )

    (
        docs,
        queries_label_free,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
    ) = load_cal_data_label_free()

    return {
        "docs": docs,
        "questions": {q: queries_label_free[q][0] for q in all_ids},
        "blocks": blocks,
        "all_ids": [str(q) for q in all_ids],
        "pool": {str(q): [str(d) for d in extended[q]] for q in all_ids},
    }


def load_top5_bge(root: Path, q: str):
    p = (
        root
        / "results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/"
        "cal_exact_d1_top5_ce_scores_v1"
        / f"{q}.json"
    )
    if not p.is_file():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {str(d): float(s) for d, s in obj["scores"].items()}


@torch.inference_mode()
def score_candidate_bge(
    *,
    root: Path,
    sibling: Path,
    sanitized_world,
    requests,
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
    model = CrossEncoder(ckpt)
    model.eval()
    evidence = Evidence(sanitized_world["render"], model.tokenizer)

    score_dir = out / "outside_candidate_bge"
    score_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    abstain = {}
    started = time.perf_counter()

    try:
        for i, (q, req) in enumerate(requests.items(), 1):
            docs = list(req["candidate_docs"])
            missing = [
                d for d in docs
                if evidence.db.execute(
                    "SELECT 1 FROM chunks WHERE doc=? LIMIT 1", (d,)
                ).fetchone() is None
            ]
            if missing:
                abstain[q] = {
                    "reason": "OUTSIDE_CANDIDATE_MISSING_EVIDENCE",
                    "docs": missing,
                }
                continue

            sig = hashlib.sha256(
                json.dumps(
                    ["outside-consensus-rel-l0-v1", sha256(ckpt), q, docs],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            cp = score_dir / f"{q}.json"

            if cp.is_file():
                obj = json.loads(cp.read_text(encoding="utf-8"))
                if obj.get("signature") != sig:
                    raise RuntimeError(f"BGE cache signature mismatch q={q}")
                scores = {str(d): float(s) for d, s in obj["scores"].items()}
            else:
                vals = []
                for st in range(0, len(docs), pair_microbatch):
                    batch_docs = docs[st:st + pair_microbatch]
                    pairs = [evidence.package(q, d) for d in batch_docs]
                    vals.extend(model(pairs).detach().cpu().tolist())
                scores = {d: float(s) for d, s in zip(docs, vals)}
                dump(cp, {"signature": sig, "scores": scores})

            rows[q] = scores

            if i % 25 == 0 or i == len(requests):
                print(
                    f"  candidate CE {i}/{len(requests)} "
                    f"scored={len(rows)} abstain={len(abstain)} "
                    f"qps={i/max(time.perf_counter()-started,1e-9):.2f}",
                    flush=True,
                )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect()
        torch.cuda.empty_cache()

    return rows, abstain, ckpt


def metrics(pred, gold, ids):
    recalls, precisions = [], []
    for q in ids:
        h = len(set(pred[q]) & gold[q])
        recalls.append(h / len(gold[q]))
        precisions.append(h / 5.0)
    return float(np.mean(recalls)), float(np.mean(precisions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--pair-microbatch", type=int, default=4)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"
    base_script = root.parent / "run_noncal_trainable_ce_boundary_v3_fixed.py"
    out = (
        root
        / "results/manual/"
        "huy_outside_consensus_symmetric_rel_l0_cal_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading label-free exact D1/pool/source rankings...")
    ids, d1, d1_path = load_exact_d1_top5_whitelist(root)
    lf = load_label_free_pool(root)
    if set(ids) != set(lf["all_ids"]):
        raise RuntimeError("D1 / label-free CAL population mismatch")

    e5, e5_paths = load_e5_full_corpus_whitelist(root)
    ai, ai_path = load_aiteam(root)

    print(
        f"  D1=600 | pool=600 | E5 qids={len(e5)} | AITeam qids={len(ai)}",
        flush=True,
    )

    print("[2/7] Building cross-source OUTSIDE-pool proposal sets...")
    source_rows = {}
    proposal_requests = {}
    source_triggered = 0
    weak_defender_queries = 0

    for q in ids:
        pool = set(lf["pool"][q])
        top5 = d1[q]

        top5_scores = load_top5_bge(root, q)
        if top5_scores is None or any(d not in top5_scores for d in top5):
            source_rows[q] = {"abstain": "MISSING_TOP5_BGE_CACHE"}
            continue

        median_top4 = float(np.median([top5_scores[d] for d in top5[:4]]))
        defender = top5[4]
        defender_rel = top5_scores[defender] - median_top4
        if defender_rel >= REL_L0:
            source_rows[q] = {
                "defender": defender,
                "defender_rel": defender_rel,
                "weak_defender": False,
            }
            continue

        weak_defender_queries += 1
        e5_order = e5.get(q, [])
        ai_order = ai.get(q, [])
        e5_rank = {d: i + 1 for i, d in enumerate(e5_order[:MAX_E5_RANK])}
        ai_rank = {d: i + 1 for i, d in enumerate(ai_order[:MAX_AI_RANK])}

        common = set(e5_rank) & set(ai_rank)
        proposals = []
        for d in common:
            if d in pool or d in top5:
                continue
            re5 = e5_rank[d]
            rai = ai_rank[d]
            if min(re5, rai) > MIN_SOURCE_RANK_MAX:
                continue
            proposals.append({
                "doc": d,
                "e5_rank": re5,
                "aiteam_rank": rai,
                "rrf20": 1.0 / (20 + re5) + 1.0 / (20 + rai),
            })

        proposals.sort(
            key=lambda x: (
                -x["rrf20"],
                max(x["e5_rank"], x["aiteam_rank"]),
                x["doc"],
            )
        )

        source_rows[q] = {
            "defender": defender,
            "defender_score": top5_scores[defender],
            "median_top4": median_top4,
            "defender_rel": defender_rel,
            "weak_defender": True,
            "proposals": proposals,
        }

        if proposals:
            source_triggered += 1
            proposal_requests[q] = {
                "candidate_docs": [x["doc"] for x in proposals]
            }

    print(
        f"  REL_L0-weak defenders={weak_defender_queries} | "
        f"queries with cross-source outside proposals={source_triggered}",
        flush=True,
    )

    print("[3/7] Loading sanitized renderer and scoring outside challengers...")
    m = loadmod(base_script, "cebase_outside_rel")
    cal_ids, _ = m.get_cal_ids_label_free(root)
    sanitized = m.load_noncal_world(root, sibling, set(cal_ids))

    ce_rows, ce_abstain, ckpt = score_candidate_bge(
        root=root,
        sibling=sibling,
        sanitized_world=sanitized,
        requests=proposal_requests,
        out=out,
        pair_microbatch=args.pair_microbatch,
    )

    print("[4/7] Applying symmetric REL_L0 certificate...")
    actions = {}
    diagnostics = {}

    for q in ids:
        row = source_rows[q]
        diag = dict(row)

        if not row.get("weak_defender"):
            diagnostics[q] = diag
            continue
        if q not in proposal_requests:
            diagnostics[q] = diag
            continue
        if q not in ce_rows:
            diag["ce_abstain"] = ce_abstain.get(q, {"reason": "NO_CE_ROW"})
            diagnostics[q] = diag
            continue

        eligible = []
        proposal_details = []
        med = row["median_top4"]

        for p in row["proposals"]:
            d = p["doc"]
            score = ce_rows[q][d]
            rel = score - med
            cert = rel >= REL_L0

            item = {
                **p,
                "bge_score": score,
                "bge_rel_to_d1_top4_median": rel,
                "passes_symmetric_rel_l0_certificate": bool(cert),
            }
            proposal_details.append(item)
            if cert:
                eligible.append(d)

        diag["proposal_details"] = proposal_details
        diag["eligible_docs"] = eligible

        # Deliberate ambiguity abstention.
        if len(eligible) == 1:
            c = eligible[0]
            info = next(x for x in proposal_details if x["doc"] == c)
            actions[q] = {
                "qid": q,
                "defender": row["defender"],
                "challenger": c,
                "before": d1[q],
                "after": d1[q][:4] + [c],
                "e5_rank": info["e5_rank"],
                "aiteam_rank": info["aiteam_rank"],
                "defender_rel": row["defender_rel"],
                "challenger_rel": info["bge_rel_to_d1_top4_median"],
            }
        elif len(eligible) > 1:
            diag["abstain"] = {
                "reason": "MULTIPLE_CERTIFIED_CHALLENGERS",
                "docs": eligible,
            }

        diagnostics[q] = diag

    print(f"  sealed action candidates={len(actions)}")

    print("[5/7] Sealing actions BEFORE CAL outcome reveal...")
    seal = {
        "schema": "manual.outside_consensus_symmetric_rel_l0_cal_v1.actions",
        "status": "SEALED_BEFORE_CAL_GOLD",
        "policy": {
            "defender_rel_threshold": REL_L0,
            "e5_max_rank": MAX_E5_RANK,
            "aiteam_max_rank": MAX_AI_RANK,
            "min_source_rank_must_be_lte": MIN_SOURCE_RANK_MAX,
            "challenger_must_be_outside_exact_d1_pool": True,
            "challenger_must_appear_in_both_sources": True,
            "challenger_rel_threshold": REL_L0,
            "unique_certified_challenger_required": True,
            "K": 5,
            "no_threshold_search": True,
        },
        "d1_source": str(d1_path),
        "d1_source_sha256": sha256(d1_path),
        "aiteam_source": str(ai_path),
        "aiteam_sha256": sha256(ai_path),
        "e5_sources": [str(p) for p in e5_paths],
        "e5_sha256": {str(p): sha256(p) for p in e5_paths},
        "bge_checkpoint": str(ckpt),
        "bge_checkpoint_sha256": sha256(ckpt),
        "actions_count": len(actions),
        "actions": actions,
        "diagnostics": diagnostics,
    }
    seal_path = out / "CAL_ACTIONS_SEALED.json"
    dump(seal_path, seal)

    for q, a in actions.items():
        print(
            f"    q={q} D1r5 {a['defender']} -> outside {a['challenger']} "
            f"(E5={a['e5_rank']}, AI={a['aiteam_rank']}, "
            f"rel {a['defender_rel']:+.3f}->{a['challenger_rel']:+.3f})",
            flush=True,
        )

    print("[6/7] Revealing CAL gold and evaluating sealed policy...")
    gold_raw = json.loads(
        (
            root
            / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
        ).read_text(encoding="utf-8")
    )
    gold = {
        q: {str(d) for d in gold_raw[q]["answer"]}
        for q in ids
    }

    baseline = {q: list(d1[q]) for q in ids}
    candidate = {q: list(d1[q]) for q in ids}
    for q, a in actions.items():
        candidate[q] = list(a["after"])

    br, bp = metrics(baseline, gold, ids)
    cr, cp = metrics(candidate, gold, ids)
    if abs(br - EXPECTED_D1_R) > 1e-12:
        raise RuntimeError(f"D1 recall parity failed: {br}")
    if abs(bp - EXPECTED_D1_P) > 1e-12:
        raise RuntimeError(f"D1 precision parity failed: {bp}")

    wins, losses, neutrals = [], [], []
    for q, a in actions.items():
        rb = len(set(a["before"]) & gold[q]) / len(gold[q])
        ra = len(set(a["after"]) & gold[q]) / len(gold[q])
        item = {
            "qid": q,
            "defender": a["defender"],
            "challenger": a["challenger"],
            "defender_is_gold": a["defender"] in gold[q],
            "challenger_is_gold": a["challenger"] in gold[q],
            "before_recall": rb,
            "after_recall": ra,
            "delta": ra - rb,
            "e5_rank": a["e5_rank"],
            "aiteam_rank": a["aiteam_rank"],
        }
        if ra > rb:
            wins.append(item)
        elif ra < rb:
            losses.append(item)
        else:
            neutrals.append(item)

    verdict = (
        "PROMOTE_TO_PUBLIC_MATERIALIZATION"
        if len(actions) >= 1 and len(wins) > len(losses) and cr > br
        else "KILL_OUTSIDE_CONSENSUS_POLICY"
    )

    report = {
        "schema": "manual.outside_consensus_symmetric_rel_l0_cal_v1.report",
        "verdict": verdict,
        "baseline": {"recall": br, "precision": bp},
        "candidate": {"recall": cr, "precision": cp},
        "delta": {"recall": cr - br, "precision": cp - bp},
        "paired": {
            "actions": len(actions),
            "wins": len(wins),
            "losses": len(losses),
            "neutral": len(neutrals),
        },
        "wins": wins,
        "losses": losses,
        "neutrals": neutrals,
        "seal_sha256": sha256(seal_path),
    }
    dump(out / "REPORT.json", report)

    print("[7/7] RESULT")
    print("=" * 108)
    print(f"D1        R={br:.10f} P={bp:.10f}")
    print(
        f"CANDIDATE R={cr:.10f} ({cr-br:+.10f}) "
        f"P={cp:.10f} ({cp-bp:+.10f})"
    )
    print(
        f"actions={len(actions)} W/L/N="
        f"{len(wins)}/{len(losses)}/{len(neutrals)}"
    )
    print("VERDICT:", verdict)
    print("Report:", out / "REPORT.json")
    print("=" * 108)


if __name__ == "__main__":
    main()
