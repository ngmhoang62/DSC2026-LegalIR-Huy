#!/usr/bin/env python
"""
QUERY-ANCHORED LEGAL-REF RESCUE + SYMMETRIC REL_L0 CERTIFICATE — CAL V1
=======================================================================

ONE preregistered policy:

- Reuse the already-sealed label-free candidate generator artifact:
    results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/
    CAL_PER_QUERY_EXPANSION.jsonl

- Preserve its frozen deterministic candidate order:
    DIRECT_REFERENCE_MATCH
    -> HEADER_RELATION_NEIGHBOR
    -> BODY_RELATION_NEIGHBOR

- Exact D1 rank5 defender must be REL_L0-weak:
      CE(defender) - median(CE(rank1..4)) < REL_L0

- Scan generated additions IN THEIR SEALED ORDER and choose the FIRST candidate
  with frozen Evidence coverage that passes the same semantic certificate:
      CE(candidate) - median(CE(rank1..4)) >= REL_L0

- Replace D1 rank5 with that first certified addition. K stays exactly 5.
- If none pass, abstain.
- No threshold search. No CAL labels are read until actions are sealed.

REL_L0 = -3.0393552780151367
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


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
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


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_exact_d1_top5_whitelist(root: Path):
    """
    Historical artifact co-locates gold. Read raw text and whitelist-extract
    ONLY qid + s0_top5 before outcome reveal.
    """
    path = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "S0_S1_CAL_PREDICTIONS.jsonl"
    )
    qid_re = re.compile(r'"qid"\s*:\s*"([^"]+)"')
    top5_re = re.compile(r'"s0_top5"\s*:\s*(\[[^\]]*\])')

    ids, base = [], {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            qm = qid_re.search(line)
            tm = top5_re.search(line)
            if qm is None or tm is None:
                raise RuntimeError(f"D1 whitelist parse failed {path}:{lineno}")
            q = str(qm.group(1))
            docs = [str(d) for d in json.loads(tm.group(1))]
            if len(docs) != 5 or len(set(docs)) != 5:
                raise RuntimeError(f"Invalid D1 Top5 q={q}: {docs}")
            ids.append(q)
            base[q] = docs

    if len(ids) != 600 or len(base) != 600:
        raise RuntimeError(f"Exact D1 population mismatch: {len(ids)}/{len(base)}")
    return ids, base, path


def load_sealed_legal_ref_additions(root: Path, ids):
    res = root / "results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1"
    path = res / "CAL_PER_QUERY_EXPANSION.jsonl"
    seal_path = res / "CAL600_PER_QUERY_EXPANSION_SEAL.json"

    if not path.is_file():
        raise FileNotFoundError(path)
    if not seal_path.is_file():
        raise FileNotFoundError(seal_path)

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    actual_sha = sha256(path)
    sealed_sha = seal.get("generated_additions_artifact_sha256")
    if actual_sha != sealed_sha:
        raise RuntimeError(
            f"Legal-ref generator seal mismatch: actual={actual_sha} sealed={sealed_sha}"
        )

    rows = {}
    for r in read_jsonl(path):
        q = str(r["qid"])
        additions = [str(d) for d in r.get("newly_added_doc_ids", [])]
        details = list(r.get("addition_details", []))
        if len(additions) != len(details):
            raise RuntimeError(f"Addition/details length mismatch q={q}")
        rows[q] = {
            "qid": q,
            "query_text": str(r.get("query_text", "")),
            "extracted_references": list(r.get("extracted_references", [])),
            "additions": additions,
            "details": details,
        }

    if set(rows) != set(ids):
        raise RuntimeError("Legal-ref addition population != exact D1 population")

    triggered = [q for q in ids if rows[q]["additions"]]
    total_additions = sum(len(rows[q]["additions"]) for q in ids)

    return rows, path, seal_path, triggered, total_additions


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
def score_generated_candidates(
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

    score_dir = out / "legal_ref_candidate_bge"
    score_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    missing_evidence = {}
    started = time.perf_counter()

    try:
        for i, (q, docs) in enumerate(requests.items(), 1):
            scores = {}
            missing = []

            # Keep sealed generator order.
            scorable = []
            for d in docs:
                ok = evidence.db.execute(
                    "SELECT 1 FROM chunks WHERE doc=? LIMIT 1", (d,)
                ).fetchone() is not None
                if ok:
                    scorable.append(d)
                else:
                    missing.append(d)

            if missing:
                missing_evidence[q] = missing

            if scorable:
                sig = hashlib.sha256(
                    json.dumps(
                        [
                            "legal-ref-symmetric-rel-l0-v1",
                            sha256(ckpt),
                            q,
                            scorable,
                        ],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                cp = score_dir / f"{q}.json"

                if cp.is_file():
                    obj = json.loads(cp.read_text(encoding="utf-8"))
                    if obj.get("signature") != sig:
                        raise RuntimeError(f"CE cache signature mismatch q={q}")
                    scores = {
                        str(d): float(s)
                        for d, s in obj["scores"].items()
                    }
                else:
                    vals = []
                    for st in range(0, len(scorable), pair_microbatch):
                        batch_docs = scorable[st:st + pair_microbatch]
                        pairs = [evidence.package(q, d) for d in batch_docs]
                        vals.extend(model(pairs).detach().cpu().tolist())
                    scores = {
                        d: float(s)
                        for d, s in zip(scorable, vals)
                    }
                    dump(cp, {"signature": sig, "scores": scores})

            rows[q] = scores

            print(
                f"  CE legal-ref {i}/{len(requests)} "
                f"scored_docs={len(scores)} missing={len(missing)} "
                f"qps={i/max(time.perf_counter()-started,1e-9):.2f}",
                flush=True,
            )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect()
        torch.cuda.empty_cache()

    return rows, missing_evidence, ckpt


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
        "huy_legal_ref_symmetric_rel_l0_rescue_cal_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading exact D1 label-free Top5...")
    ids, base, d1_path = load_exact_d1_top5_whitelist(root)

    print("[2/7] Verifying previously sealed legal-ref generator artifact...")
    (
        generator,
        generator_path,
        generator_seal_path,
        triggered,
        total_additions,
    ) = load_sealed_legal_ref_additions(root, ids)

    print(
        f"  triggered_queries={len(triggered)} "
        f"total_additions={total_additions}",
        flush=True,
    )
    if len(triggered) != 4 or total_additions != 21:
        raise RuntimeError(
            "Frozen generator population drift: "
            f"triggered={len(triggered)} additions={total_additions}"
        )

    print("[3/7] Applying frozen REL_L0 defender gate...")
    requests = {}
    prelim = {}

    for q in ids:
        additions = generator[q]["additions"]
        if not additions:
            prelim[q] = {"triggered": False}
            continue

        top5 = base[q]
        top5_scores = load_top5_bge(root, q)
        if top5_scores is None or any(d not in top5_scores for d in top5):
            prelim[q] = {
                "triggered": True,
                "abstain": "MISSING_TOP5_BGE_CACHE",
            }
            continue

        med = float(np.median([top5_scores[d] for d in top5[:4]]))
        defender = top5[4]
        defender_rel = top5_scores[defender] - med
        weak = defender_rel < REL_L0

        prelim[q] = {
            "triggered": True,
            "defender": defender,
            "defender_score": top5_scores[defender],
            "median_top4": med,
            "defender_rel": defender_rel,
            "defender_rel_l0_pass": bool(weak),
            "generator_additions": additions,
            "generator_details": generator[q]["details"],
            "extracted_references": generator[q]["extracted_references"],
        }

        if weak:
            requests[q] = additions

    print(
        f"  triggered={len(triggered)} | "
        f"REL_L0-weak defender queries={len(requests)}",
        flush=True,
    )

    print("[4/7] Scoring generated additions with frozen Fold0 BGE CE...")
    m = loadmod(base_script, "cebase_legal_ref_rel")
    cal_ids, _ = m.get_cal_ids_label_free(root)
    sanitized = m.load_noncal_world(root, sibling, set(cal_ids))

    cal_overlap = set(sanitized["cal_overlap"])
    outside_v2_requests = sorted(set(requests) - cal_overlap)
    if outside_v2_requests:
        for q in outside_v2_requests:
            prelim[q]["abstain"] = "CAL_OUTSIDE_V2_NO_FROZEN_QUERY_VECTOR"
            requests.pop(q, None)
        print(
            f"  conservative CAL-outside-V2 abstentions={outside_v2_requests}",
            flush=True,
        )

    ce_rows, missing_evidence, ckpt = score_generated_candidates(
        root=root,
        sibling=sibling,
        sanitized_world=sanitized,
        requests=requests,
        out=out,
        pair_microbatch=args.pair_microbatch,
    )

    print("[5/7] Selecting FIRST certified addition in frozen generator order...")
    actions = {}
    diagnostics = {}

    for q in ids:
        row = dict(prelim[q])

        if q not in requests:
            diagnostics[q] = row
            continue

        med = row["median_top4"]
        scores = ce_rows.get(q, {})
        candidates = []

        for idx, (d, detail) in enumerate(
            zip(row["generator_additions"], row["generator_details"])
        ):
            if d not in scores:
                candidates.append({
                    "generator_index": idx,
                    "doc": d,
                    "addition_type": detail.get("addition_type"),
                    "relation_family": detail.get("relation_family"),
                    "has_frozen_evidence": False,
                    "certified": False,
                })
                continue

            score = float(scores[d])
            rel = score - med
            cert = rel >= REL_L0
            candidates.append({
                "generator_index": idx,
                "doc": d,
                "addition_type": detail.get("addition_type"),
                "relation_family": detail.get("relation_family"),
                "relation_direction": detail.get("relation_direction"),
                "is_header": detail.get("is_header"),
                "anchor_ref": detail.get("anchor_ref"),
                "anchor_doc": detail.get("anchor_doc"),
                "bge_score": score,
                "bge_rel_to_d1_top4_median": rel,
                "has_frozen_evidence": True,
                "certified": bool(cert),
            })

        row["candidate_certificates"] = candidates
        row["missing_evidence_docs"] = missing_evidence.get(q, [])

        first = next((c for c in candidates if c["certified"]), None)
        if first is not None:
            c = first["doc"]
            actions[q] = {
                "qid": q,
                "defender": row["defender"],
                "challenger": c,
                "before": list(base[q]),
                "after": list(base[q][:4]) + [c],
                "defender_rel": row["defender_rel"],
                "challenger_rel": first["bge_rel_to_d1_top4_median"],
                "generator_index": first["generator_index"],
                "addition_type": first.get("addition_type"),
                "relation_family": first.get("relation_family"),
                "relation_direction": first.get("relation_direction"),
                "is_header": first.get("is_header"),
                "anchor_ref": first.get("anchor_ref"),
                "anchor_doc": first.get("anchor_doc"),
            }

        diagnostics[q] = row

    print(f"  sealed actions={len(actions)}")
    for q, a in actions.items():
        print(
            f"    q={q} {a['addition_type']} "
            f"D1r5 {a['defender']} -> {a['challenger']} "
            f"rel {a['defender_rel']:+.3f}->{a['challenger_rel']:+.3f}",
            flush=True,
        )

    seal = {
        "schema": "manual.legal_ref_symmetric_rel_l0_rescue_cal_v1.actions",
        "status": "SEALED_BEFORE_CAL_GOLD",
        "policy": {
            "candidate_source": (
                "previously-sealed HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1"
            ),
            "candidate_order": (
                "use exact order in CAL_PER_QUERY_EXPANSION.jsonl; "
                "first certified candidate wins"
            ),
            "defender_rel_threshold": REL_L0,
            "challenger_rel_threshold": REL_L0,
            "K": 5,
            "no_threshold_search": True,
            "no_qid_specific_rules": True,
        },
        "d1_source": str(d1_path),
        "d1_sha256": sha256(d1_path),
        "generator_source": str(generator_path),
        "generator_sha256": sha256(generator_path),
        "generator_seal": str(generator_seal_path),
        "generator_seal_sha256": sha256(generator_seal_path),
        "bge_checkpoint": str(ckpt),
        "bge_checkpoint_sha256": sha256(ckpt),
        "triggered_queries": len(triggered),
        "total_generator_additions": total_additions,
        "actions_count": len(actions),
        "actions": actions,
        "diagnostics": diagnostics,
    }

    seal_path = out / "CAL_ACTIONS_SEALED.json"
    dump(seal_path, seal)

    print("[6/7] Revealing CAL gold AFTER action seal...")
    gold_path = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
    )
    raw = json.loads(gold_path.read_text(encoding="utf-8"))
    gold = {q: {str(d) for d in raw[q]["answer"]} for q in ids}

    baseline = {q: list(base[q]) for q in ids}
    candidate = {q: list(base[q]) for q in ids}
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
            "addition_type": a["addition_type"],
            "relation_family": a["relation_family"],
            "generator_index": a["generator_index"],
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
        else "KILL_LEGAL_REF_CERTIFICATE_POLICY"
    )

    report = {
        "schema": "manual.legal_ref_symmetric_rel_l0_rescue_cal_v1.report",
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
