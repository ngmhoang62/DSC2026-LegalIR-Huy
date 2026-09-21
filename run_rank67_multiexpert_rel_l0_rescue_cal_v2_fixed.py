#!/usr/bin/env python
"""
D1 RANK6-7 MULTI-EXPERT RESCUE + REL_L0 DEFENDER VETO — CAL CONFIRMATION V1
===========================================================================

Purpose
-------
Test ONE preregistered recall-rescue policy on exact D1 LOBO:

  challenger ∈ exact D1 ranks {6,7}
  AND rank5 defender is weak under frozen BGE CE REL_L0
  AND BGE CE prefers challenger > defender
  AND frozen legal-section CE prefers challenger > defender
  AND >=4/5 independent external score families prefer challenger
  AND >=2/5 D1 rank-view families prefer challenger
  AND exactly one challenger satisfies all gates
  -> replace exact D1 rank5 with that challenger

No threshold search is performed. No alternate policy is selected after seeing CAL
outcomes. The action set is sealed before outcome evaluation.

External score families:
  vnlegal_lal, crossenc, aiteamvn_ft, jina_ft, title_embed

D1 rank-view families:
  base, expanded, jina, dense, corpus

The BGE REL_L0 threshold is frozen from nonCAL Fold0 DEV:
  -3.0393552780151367

Outputs
-------
results/manual/huy_d1_rank67_multiexpert_rel_l0_rescue_v1/
  CAL_ACTIONS_SEALED.json
  REPORT.json
  bge_rank6_7_scores/*.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REL_L0 = -3.0393552780151367
EXPECTED_D1_R = 0.9569444444444444
EXPECTED_D1_P = 0.20566666666666666

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXTERNAL = [
    "vnlegal_lal",
    "crossenc",
    "aiteamvn_ft",
    "jina_ft",
    "title_embed",
]

# ONE preregistered policy.
MAX_CHALLENGER_RANK = 7
MIN_EXTERNAL_VOTES = 4
MIN_RANKVIEW_VOTES = 2


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_authoritative_d1_top5(root: Path):
    p = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "S0_S1_CAL_PREDICTIONS.jsonl"
    )
    out = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["qid"])] = [str(x) for x in r["s0_top5"]]
    if len(out) != 600:
        raise RuntimeError(f"authoritative D1 population !=600: {len(out)}")
    return out, p


def reconstruct_exact_d1_oof(root: Path):
    """
    Reconstruct D1 full LOBO ranking. Every query is scored by a D1 model that
    excludes its whole block from model training.
    """
    sys.path.insert(0, str(root))
    from tune_expanded_fusion_selection import ltr_features
    from src.gemini.huy_d1_legal_section_evidence_v1.common import (
        D1_VIEWS as SOURCE_D1_VIEWS,
        load_cal_data,
    )

    if list(SOURCE_D1_VIEWS) != D1_VIEWS:
        raise RuntimeError(f"D1 view contract drift: {SOURCE_D1_VIEWS}")

    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels,
        gold,
        type_rows,
        cite_rows,
    ) = load_cal_data()

    eval_rows, eval_groups = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, full_channels
    )
    for q in all_ids:
        eval_rows[q] = np.concatenate(
            [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
        )

    rankings = {}
    scores = {}

    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        test = list(blocks[held])

        X = np.vstack([eval_rows[q] for q in train])
        y = np.concatenate(
            [[d in gold[q] for d in eval_groups[q]] for q in train]
        ).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X), y)

        for q in test:
            dec = model.decision_function(scaler.transform(eval_rows[q]))
            # Match the authoritative D1 LOBO implementation exactly.
            order = sorted(
                range(len(dec)),
                key=lambda i: dec[i],
                reverse=True,
            )
            rankings[q] = [eval_groups[q][i] for i in order]
            scores[q] = {
                eval_groups[q][i]: float(dec[i])
                for i in range(len(dec))
            }

    return {
        "docs": docs,
        "queries": queries,
        "blocks": blocks,
        "all_ids": all_ids,
        "extended": extended,
        "local_views": local_views,
        "full_channels": full_channels,
        "gold": gold,
        "rankings": rankings,
        "scores": scores,
    }


def rank_map(order):
    return {d: i + 1 for i, d in enumerate(order)}


def load_section_scores(root: Path):
    p = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "legal_section_ce_cv.pkl"
    )
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"], p
    return obj, p


def get_top5_bge_score_cache(root: Path, q: str):
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
def score_rank67_bge(
    *,
    root: Path,
    sibling: Path,
    base_script: Path,
    world,
    all_ids,
    rankings,
    output_dir: Path,
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
    evidence = Evidence(world["render"], model.tokenizer)

    score_dir = output_dir / "bge_rank6_7_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    eligible_v2 = set(world["cal_overlap"])
    result = {}
    abstain = {}
    started = time.perf_counter()

    try:
        for i, q in enumerate(all_ids, 1):
            if q not in eligible_v2:
                abstain[q] = {"reason": "CAL_OUTSIDE_V2"}
                continue

            top7 = rankings[q][:7]
            top5_cache = get_top5_bge_score_cache(root, q)
            if top5_cache is None or any(d not in top5_cache for d in top7[:5]):
                abstain[q] = {"reason": "MISSING_TOP5_BGE_CACHE"}
                continue

            defender = top7[4]
            s14 = [top5_cache[d] for d in top7[:4]]
            defender_rel = top5_cache[defender] - float(np.median(s14))

            # If defender is not REL_L0-weak, this query can never fire.
            if not (defender_rel < REL_L0):
                result[q] = {
                    "defender": defender,
                    "defender_score": top5_cache[defender],
                    "defender_rel_top4_med": defender_rel,
                    "challenger_scores": {},
                }
                continue

            challengers = top7[5:7]
            missing = [
                d for d in challengers
                if evidence.db.execute(
                    "SELECT 1 FROM chunks WHERE doc=? LIMIT 1", (d,)
                ).fetchone() is None
            ]
            if missing:
                abstain[q] = {
                    "reason": "RANK67_MISSING_EVIDENCE",
                    "docs": missing,
                }
                continue

            cache_path = score_dir / f"{q}.json"
            sig = hashlib.sha256(
                json.dumps(
                    [
                        "rank67-bge-v1",
                        sha256(ckpt),
                        q,
                        challengers,
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            if cache_path.is_file():
                obj = json.loads(cache_path.read_text(encoding="utf-8"))
                if obj.get("signature") != sig:
                    raise RuntimeError(f"BGE rank67 cache drift q={q}")
                ch_scores = {
                    str(d): float(s)
                    for d, s in obj["scores"].items()
                }
            else:
                vals = []
                for st in range(0, len(challengers), pair_microbatch):
                    docs = challengers[st:st + pair_microbatch]
                    pairs = [evidence.package(q, d) for d in docs]
                    vals.extend(model(pairs).detach().cpu().tolist())
                ch_scores = {
                    d: float(s)
                    for d, s in zip(challengers, vals)
                }
                dump(
                    cache_path,
                    {
                        "signature": sig,
                        "scores": ch_scores,
                    },
                )

            result[q] = {
                "defender": defender,
                "defender_score": top5_cache[defender],
                "defender_rel_top4_med": defender_rel,
                "challenger_scores": ch_scores,
            }

            if i % 100 == 0 or i == len(all_ids):
                print(
                    f"  BGE boundary {i}/{len(all_ids)} "
                    f"usable={len(result)} abstain={len(abstain)} "
                    f"qps={i/max(time.perf_counter()-started,1e-9):.2f}",
                    flush=True,
                )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect()
        torch.cuda.empty_cache()

    return result, abstain, ckpt


def metrics(pred, gold, ids):
    recalls = []
    precisions = []
    for q in ids:
        h = len(set(pred[q]) & set(gold[q]))
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
        / "results/manual/huy_d1_rank67_multiexpert_rel_l0_rescue_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Reconstructing exact D1 LOBO full rankings...")
    world = reconstruct_exact_d1_oof(root)
    ids = list(world["all_ids"])
    rankings = world["rankings"]

    auth, auth_path = load_authoritative_d1_top5(root)
    parity = sum(rankings[q][:5] == auth[q] for q in ids)
    print(f"  Exact D1 ordered Top5 parity={parity}/600")
    if parity != 600:
        bad = [q for q in ids if rankings[q][:5] != auth[q]][:10]
        raise RuntimeError(f"D1 parity failed sample={bad}")

    print("[2/6] Loading frozen expert caches...")
    sec, sec_path = load_section_scores(root)
    full = world["full_channels"]
    for name in EXTERNAL:
        if name not in full:
            raise RuntimeError(f"Missing external score family: {name}")

    rankmaps = {
        name: {q: rank_map(world["local_views"][name][q]) for q in ids}
        for name in D1_VIEWS
    }

    print("[3/6] Loading sanitized V2 renderer and scoring rank6-7 BGE only where REL_L0 fires...")
    m = loadmod(base_script, "cebase_rank67")
    cal_ids, _ = m.get_cal_ids_label_free(root)
    v2 = m.load_noncal_world(root, sibling, set(cal_ids))
    bge, bge_abstain, ckpt = score_rank67_bge(
        root=root,
        sibling=sibling,
        base_script=base_script,
        world=v2,
        all_ids=ids,
        rankings=rankings,
        output_dir=out,
        pair_microbatch=args.pair_microbatch,
    )

    print("[4/6] Applying ONE preregistered policy and sealing actions...")
    actions = {}
    diagnostic = {}

    for q in ids:
        top = rankings[q]
        top5 = top[:5]
        defender = top5[4]

        row = {
            "qid": q,
            "defender": defender,
            "defender_rel_l0_pass": False,
            "candidates": [],
            "eligible": [],
        }

        if q not in bge:
            row["abstain"] = bge_abstain.get(q, {"reason": "NO_BGE_ROW"})
            diagnostic[q] = row
            continue

        brow = bge[q]
        rel_pass = brow["defender_rel_top4_med"] < REL_L0
        row["defender_rel_top4_med"] = brow["defender_rel_top4_med"]
        row["defender_rel_l0_pass"] = bool(rel_pass)
        if not rel_pass:
            diagnostic[q] = row
            continue

        for rank in range(6, MAX_CHALLENGER_RANK + 1):
            c = top[rank - 1]
            if c not in brow["challenger_scores"]:
                continue

            bge_pass = brow["challenger_scores"][c] > brow["defender_score"]
            sec_c = float(sec.get(q, {}).get(c, -1e12))
            sec_d = float(sec.get(q, {}).get(defender, -1e12))
            sec_pass = sec_c > sec_d

            ext_votes = {}
            for name in EXTERNAL:
                qs = full[name].get(q, {})
                ext_votes[name] = (
                    float(qs.get(c, -1e12))
                    > float(qs.get(defender, -1e12))
                )
            ext_n = sum(ext_votes.values())

            rvotes = {}
            for name in D1_VIEWS:
                rm = rankmaps[name][q]
                rvotes[name] = rm.get(c, 10**9) < rm.get(defender, 10**9)
            rv_n = sum(rvotes.values())

            eligible = (
                bge_pass
                and sec_pass
                and ext_n >= MIN_EXTERNAL_VOTES
                and rv_n >= MIN_RANKVIEW_VOTES
            )

            cr = {
                "doc": c,
                "d1_rank": rank,
                "d1_score": world["scores"][q][c],
                "defender_d1_score": world["scores"][q][defender],
                "bge_challenger": brow["challenger_scores"][c],
                "bge_defender": brow["defender_score"],
                "bge_prefers_challenger": bool(bge_pass),
                "section_challenger": sec_c,
                "section_defender": sec_d,
                "section_prefers_challenger": bool(sec_pass),
                "external_votes": ext_votes,
                "external_vote_count": int(ext_n),
                "rankview_votes": rvotes,
                "rankview_vote_count": int(rv_n),
                "eligible": bool(eligible),
            }
            row["candidates"].append(cr)
            if eligible:
                row["eligible"].append(c)

        # Ambiguity abstention is deliberate.
        if len(row["eligible"]) == 1:
            c = row["eligible"][0]
            actions[q] = {
                "qid": q,
                "defender": defender,
                "challenger": c,
                "before": top5,
                "after": top5[:4] + [c],
                "policy": {
                    "max_challenger_rank": MAX_CHALLENGER_RANK,
                    "rel_l0_threshold": REL_L0,
                    "bge_prefers_challenger": True,
                    "section_prefers_challenger": True,
                    "min_external_votes": MIN_EXTERNAL_VOTES,
                    "min_rankview_votes": MIN_RANKVIEW_VOTES,
                    "unique_eligible_required": True,
                },
            }
        elif len(row["eligible"]) > 1:
            row["abstain"] = {
                "reason": "MULTIPLE_ELIGIBLE_CHALLENGERS",
                "docs": row["eligible"],
            }

        diagnostic[q] = row

    sealed = {
        "schema": "manual.d1_rank67_multiexpert_rel_l0_rescue_v1.actions",
        "status": "SEALED_BEFORE_OUTCOME_EVALUATION",
        "policy": {
            "challenger_rank_range": [6, MAX_CHALLENGER_RANK],
            "defender_rel_l0_threshold": REL_L0,
            "require_bge_challenger_gt_defender": True,
            "require_section_challenger_gt_defender": True,
            "external_families": EXTERNAL,
            "min_external_votes": MIN_EXTERNAL_VOTES,
            "rankview_families": D1_VIEWS,
            "min_rankview_votes": MIN_RANKVIEW_VOTES,
            "unique_eligible_required": True,
            "no_threshold_search": True,
        },
        "exact_d1_top5_source": str(auth_path),
        "exact_d1_top5_sha256": sha256(auth_path),
        "bge_checkpoint": str(ckpt),
        "bge_checkpoint_sha256": sha256(ckpt),
        "section_cache": str(sec_path),
        "section_cache_sha256": sha256(sec_path),
        "actions_count": len(actions),
        "actions": actions,
        "diagnostic": diagnostic,
    }
    seal_path = out / "CAL_ACTIONS_SEALED.json"
    dump(seal_path, sealed)
    print(f"  sealed actions={len(actions)}")
    for q, a in actions.items():
        print(
            f"    q={q} rank5 {a['defender']} -> {a['challenger']}",
            flush=True,
        )

    print("[5/6] Evaluating sealed policy outcomes...")
    gold = world["gold"]
    baseline = {q: rankings[q][:5] for q in ids}
    candidate = {q: list(baseline[q]) for q in ids}
    for q, a in actions.items():
        candidate[q] = list(a["after"])

    br, bp = metrics(baseline, gold, ids)
    cr, cp = metrics(candidate, gold, ids)
    if abs(br - EXPECTED_D1_R) > 1e-12 or abs(bp - EXPECTED_D1_P) > 1e-12:
        raise RuntimeError(f"Baseline metric parity failed R={br} P={bp}")

    wins = []
    losses = []
    neutrals = []
    for q, a in actions.items():
        before = len(set(a["before"]) & gold[q]) / len(gold[q])
        after = len(set(a["after"]) & gold[q]) / len(gold[q])
        item = {
            "qid": q,
            "challenger": a["challenger"],
            "defender": a["defender"],
            "challenger_is_gold": a["challenger"] in gold[q],
            "defender_is_gold": a["defender"] in gold[q],
            "recall_before": before,
            "recall_after": after,
            "delta": after - before,
        }
        if after > before:
            wins.append(item)
        elif after < before:
            losses.append(item)
        else:
            neutrals.append(item)

    verdict = (
        "PROMOTE_TO_PUBLIC_MATERIALIZATION"
        if len(actions) >= 1 and len(wins) > len(losses) and cr > br
        else "KILL_RANK67_RESCUE_POLICY"
    )

    report = {
        "schema": "manual.d1_rank67_multiexpert_rel_l0_rescue_v1.report",
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

    print("[6/6] RESULT")
    print("=" * 104)
    print(f"D1       R={br:.10f} P={bp:.10f}")
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
    print("=" * 104)


if __name__ == "__main__":
    main()
