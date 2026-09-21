#!/usr/bin/env python
"""
FOLD0 CE MARGIN SALVAGE V1
==========================
Deadline-oriented salvage of the already-trained fold_0 CE.

Scientific contract:
- No more training.
- Fold_0 was entirely held out from CE training.
- Split fold_0 deterministically BEFORE using outcomes:
    DEV / CONFIRM by SHA256(salt|qid) parity.
- Candidate proposal per query:
    best CE-margin challenger among E5@50 ∩ LAL@50 novel docs.
- DEV learns ONE scalar threshold only.
- Threshold must yield >=1 beneficial, 0 harmful, <=32 actions on DEV.
- Lock threshold.
- CONFIRM must yield >0 recall delta, >=1 beneficial, 0 harmful.
- Only if CONFIRM passes:
    use frozen fold_0 model + locked threshold on CAL599 V2-supported queries.
    CAL-only non-V2 query abstains.
    seal CAL actions before reading CAL gold.
"""

from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import torch

SALT = "dsc2026-fold0-ce-margin-salvage-v1"
MAX_DEV_ACTIONS = 32
EXPECTED_D1_R = 0.9569444444444444
EXPECTED_D1_P = 0.20566666666666666


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_base_module(root: Path):
    candidates = [
        root.parent / "run_noncal_trainable_ce_boundary_v3_fixed.py",
        root.parent / "run_noncal_trainable_ce_boundary_v1_fixed.py",
    ]
    p = next((x for x in candidates if x.is_file()), None)
    if p is None:
        raise FileNotFoundError(f"Base training script not found; tried {candidates}")
    spec = importlib.util.spec_from_file_location("cebase", p)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m, p


def split_fold0(qids: List[str]):
    dev, confirm = [], []
    for q in qids:
        x = int(hashlib.sha256(f"{SALT}|{q}".encode()).hexdigest(), 16)
        (dev if x % 2 == 0 else confirm).append(q)
    return dev, confirm


def build_rows(m, world, qids, score_dir: Path):
    rows = {}
    missing = []
    for q in qids:
        path = score_dir / f"{q}.json"
        if not path.is_file():
            missing.append(q)
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        scores = {str(d): float(s) for d, s in obj["scores"].items()}
        novels = m.overlap_novel(world["e5"][q], world["lal"][q], world["pool"][q])
        defender = world["base"][q][4]
        if not novels:
            rows[q] = None
            continue
        best = max(novels, key=lambda d: (scores[d] - scores[defender], -int(d) if d.isdigit() else 0))
        margin = scores[best] - scores[defender]
        gold = world["gold"][q]
        utility = (int(best in gold) - int(defender in gold)) / len(gold)
        rows[q] = {
            "qid": q,
            "defender": defender,
            "challenger": best,
            "margin": float(margin),
            "challenger_gold": best in gold,
            "defender_gold": defender in gold,
            "utility": float(utility),
            "gold_count": len(gold),
        }
    if missing:
        raise RuntimeError(f"Missing {len(missing)} fold0 score files; sample={missing[:10]}")
    return rows


def evaluate_threshold(rows, ids, threshold):
    b = h = n = 0
    mass = 0.0
    acted = []
    for q in ids:
        r = rows[q]
        if r is None or r["margin"] < threshold:
            continue
        u = r["utility"]
        mass += u
        if u > 0:
            b += 1
            kind = "beneficial"
        elif u < 0:
            h += 1
            kind = "harmful"
        else:
            n += 1
            kind = "neutral"
        acted.append({**r, "outcome": kind})
    return {
        "threshold": float(threshold),
        "actions": len(acted),
        "beneficial": b,
        "harmful": h,
        "neutral": n,
        "recall_delta": mass / len(ids),
        "acted": acted,
    }


def choose_dev_threshold(rows, dev):
    # Positive semantic direction only.
    margins = sorted(
        {rows[q]["margin"] for q in dev if rows[q] is not None and rows[q]["margin"] > 0},
        reverse=True,
    )
    candidates = []
    for t in margins:
        r = evaluate_threshold(rows, dev, t)
        if r["actions"] > MAX_DEV_ACTIONS:
            continue
        if r["beneficial"] >= 1 and r["harmful"] == 0 and r["recall_delta"] > 0:
            candidates.append(r)
    if not candidates:
        return None

    # Max utility first; then fewer actions; then stricter threshold.
    candidates.sort(
        key=lambda x: (x["recall_delta"], x["beneficial"], -x["actions"], x["threshold"]),
        reverse=True,
    )
    return candidates[0]


@torch.inference_mode()
def score_cal_margin_policy(m, root, sibling, world, model_path, threshold, pair_microbatch, out):
    ids, cal_questions, blocks, pool, base, e5, ai = m.load_cal_frontier_label_free(root)
    eligible = [q for q in ids if q in set(world["cal_overlap"])]
    abstain = [q for q in ids if q not in set(eligible)]

    sys.path.insert(0, str(sibling))
    sys.path.insert(0, str(sibling / "src"))
    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence

    model = CrossEncoder(model_path)
    model.eval()
    evidence = Evidence(world["render"], model.tokenizer)

    score_dir = out / "cal_scores_label_free"
    score_dir.mkdir(parents=True, exist_ok=True)

    modified = {q: list(base[q][:5]) for q in ids}
    actions = {}
    started = time.perf_counter()

    try:
        for i, q in enumerate(eligible, 1):
            novels = m.overlap_novel(e5[q], ai[q], pool[q])
            defender = base[q][4]
            docs = list(dict.fromkeys([defender] + novels))
            path = score_dir / f"{q}.json"

            sig = hashlib.sha256(
                json.dumps(
                    [sha256_file(model_path), q, docs, "best-margin-v1"],
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            if path.is_file():
                obj = json.loads(path.read_text(encoding="utf-8"))
                if obj["signature"] != sig:
                    raise RuntimeError(f"CAL score cache mismatch q={q}")
                scores = {str(d): float(s) for d, s in obj["scores"].items()}
            else:
                vals = []
                for st in range(0, len(docs), pair_microbatch):
                    dd = docs[st:st + pair_microbatch]
                    pairs = [evidence.package(q, d) for d in dd]
                    vals.extend(model(pairs).detach().cpu().tolist())
                scores = {d: float(s) for d, s in zip(docs, vals)}
                dump(path, {"signature": sig, "scores": scores})

            if novels:
                best = max(novels, key=lambda d: scores[d] - scores[defender])
                margin = scores[best] - scores[defender]
                if margin >= threshold:
                    modified[q] = list(base[q][:4]) + [best]
                    actions[q] = {
                        "qid": q,
                        "defender": defender,
                        "challenger": best,
                        "margin": float(margin),
                        "new_top5": modified[q],
                    }

            if i % 50 == 0 or i == len(eligible):
                print(
                    f"    CAL score {i}/{len(eligible)} "
                    f"qps={i/max(time.perf_counter()-started,1e-9):.3f}",
                    flush=True,
                )
    finally:
        evidence.db.close()
        del evidence, model
        torch.cuda.empty_cache()

    action_doc = {
        "schema": "manual.fold0_ce_margin_salvage_v1.cal_actions",
        "status": "SEALED_BEFORE_CAL_GOLD",
        "threshold": float(threshold),
        "threshold_source": "fold0 DEV only; confirmed on disjoint fold0 CONFIRM",
        "model_sha256": sha256_file(model_path),
        "eligible_cal_v2": len(eligible),
        "cal_outside_v2_abstentions": abstain,
        "actions_count": len(actions),
        "actions": actions,
        "predictions": modified,
    }
    action_path = out / "CAL_ACTIONS_LABEL_FREE.json"
    dump(action_path, action_doc)

    # Gold reveal AFTER seal.
    gold_path = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
    raw = json.loads(gold_path.read_text(encoding="utf-8"))
    gold = {q: {str(d) for d in raw[q]["answer"]} for q in ids}

    bm = m.metrics(base, gold, ids)
    mm = m.metrics(modified, gold, ids)
    po = m.action_outcomes(ids, base, modified, gold)
    dr = mm["recall_at_5"] - bm["recall_at_5"]
    dp = mm["precision_at_5"] - bm["precision_at_5"]

    if abs(bm["recall_at_5"] - EXPECTED_D1_R) > 1e-12:
        raise RuntimeError(f"D1 recall parity failed: {bm}")
    if abs(bm["precision_at_5"] - EXPECTED_D1_P) > 5e-10:
        raise RuntimeError(f"D1 precision parity failed: {bm}")

    block_delta = {}
    for b, qids in blocks.items():
        block_delta[b] = m.metrics(modified, gold, qids)["recall_at_5"] - m.metrics(base, gold, qids)["recall_at_5"]

    verdict = (
        "PROMOTE_FOLD0_CE_MARGIN_SALVAGE_V1"
        if dr > 0 and dp >= -1e-12 and po["beneficial"] > po["harmful"]
        else "KILL_FOLD0_CE_MARGIN_SALVAGE_V1"
    )
    rep = {
        "baseline": bm,
        "modified": mm,
        "delta_recall": dr,
        "delta_precision": dp,
        "paired": po,
        "block_delta": block_delta,
        "verdict": verdict,
        "action_artifact_sha256": sha256_file(action_path),
        "gold_reveal_time": datetime.now(timezone.utc).isoformat(),
    }
    dump(out / "CAL_REPORT.json", rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--pair-microbatch", type=int, default=2)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"
    sys.path[:0] = [str(root), str(root / "src"), str(sibling), str(sibling / "src")]

    m, base_script = load_base_module(root)
    out_base = root / "results/manual/huy_noncal_trainable_ce_boundary_v1"
    model_path = out_base / "oof/fold_0/training/model.pt"
    score_dir = out_base / "oof/fold_0/scores"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    print("[1/5] Loading sanitized nonCAL world...", flush=True)
    cal_ids_list, _ = m.get_cal_ids_label_free(root)
    world = m.load_noncal_world(root, sibling, set(cal_ids_list))

    fold0 = [q for q in world["noncal"] if world["folds"][q] == "fold_0"]
    dev, confirm = split_fold0(fold0)
    print(f"  fold0={len(fold0)} dev={len(dev)} confirm={len(confirm)}", flush=True)

    print("[2/5] Reading existing fold0 CE scores; NO GPU inference...", flush=True)
    rows = build_rows(m, world, fold0, score_dir)

    oracle_rows = [r for r in rows.values() if r is not None and r["utility"] > 0]
    print(
        f"  best-challenger positive-utility opportunities={len(oracle_rows)} "
        f"(DEV={sum(r['qid'] in set(dev) for r in oracle_rows)}, "
        f"CONFIRM={sum(r['qid'] in set(confirm) for r in oracle_rows)})",
        flush=True,
    )

    print("[3/5] Calibrating ONE CE-margin threshold on DEV...", flush=True)
    chosen = choose_dev_threshold(rows, dev)
    diag = {
        "split_salt": SALT,
        "dev_count": len(dev),
        "confirm_count": len(confirm),
        "best_challenger_positive_utility_count": len(oracle_rows),
        "chosen_dev": chosen,
    }
    if chosen is None:
        dump(root / "results/manual/huy_fold0_ce_margin_salvage_v1/DEV_CONFIRM_REPORT.json", diag)
        print("  No DEV threshold achieves >=1 beneficial, 0 harmful, positive delta.")
        print("=" * 92)
        print("Verdict : KILL_NO_SAFE_DEV_THRESHOLD")
        print("=" * 92)
        return

    threshold = chosen["threshold"]
    confirm_result = evaluate_threshold(rows, confirm, threshold)
    diag["confirm"] = confirm_result
    confirm_pass = (
        confirm_result["recall_delta"] > 0
        and confirm_result["beneficial"] >= 1
        and confirm_result["harmful"] == 0
    )
    diag["confirm_pass"] = confirm_pass

    out = root / "results/manual/huy_fold0_ce_margin_salvage_v1"
    dump(out / "DEV_CONFIRM_REPORT.json", diag)

    print(
        f"  DEV threshold={threshold:+.6f} actions={chosen['actions']} "
        f"B/H/N={chosen['beneficial']}/{chosen['harmful']}/{chosen['neutral']} "
        f"delta={chosen['recall_delta']:+.6f}",
        flush=True,
    )
    print(
        f"  CONFIRM actions={confirm_result['actions']} "
        f"B/H/N={confirm_result['beneficial']}/{confirm_result['harmful']}/{confirm_result['neutral']} "
        f"delta={confirm_result['recall_delta']:+.6f} pass={confirm_pass}",
        flush=True,
    )

    if not confirm_pass:
        print("[4/5] STOP: disjoint confirmation failed; CAL gold NOT read.")
        print("=" * 92)
        print("Verdict : KILL_AT_FOLD0_CONFIRM")
        print(f"Report  : {out / 'DEV_CONFIRM_REPORT.json'}")
        print("=" * 92)
        return

    print("[4/5] CONFIRM PASS. Applying frozen fold0 model to CAL label-free...", flush=True)
    cal = score_cal_margin_policy(
        m, root, sibling, world, model_path, threshold,
        args.pair_microbatch, out,
    )

    print("[5/5] DONE")
    print("=" * 92)
    print(
        f"D1       R@5={cal['baseline']['recall_at_5']:.10f} "
        f"P@5={cal['baseline']['precision_at_5']:.10f}"
    )
    print(
        f"MarginCE R@5={cal['modified']['recall_at_5']:.10f} "
        f"P@5={cal['modified']['precision_at_5']:.10f}"
    )
    print(
        f"Delta    R={cal['delta_recall']:+.10f} "
        f"P={cal['delta_precision']:+.10f}"
    )
    p = cal["paired"]
    print(
        f"Actions  {p['actions']} beneficial={p['beneficial']} "
        f"harmful={p['harmful']} neutral={p['neutral']}"
    )
    print("Blocks   " + " ".join(f"{k}:{v:+.6f}" for k, v in sorted(cal["block_delta"].items())))
    print(f"Verdict  {cal['verdict']}")
    print(f"Report   {out / 'CAL_REPORT.json'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
