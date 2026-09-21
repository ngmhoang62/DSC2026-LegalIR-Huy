#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DEPTH = 50
RRF_K = 32.0
RIDGE_ALPHA = 10.0
EXPECTED_CAL_D1_R5 = 0.9569444444444444
EXPECTED_CAL_D1_P5 = 0.20566666666666666

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def recall(top5: Sequence[str], gold: Set[str]) -> float:
    return len(set(top5[:5]) & gold) / len(gold)

def precision(top5: Sequence[str], gold: Set[str]) -> float:
    return len(set(top5[:5]) & gold) / 5.0

def metrics(preds, gold, ids):
    return {
        "recall_at_5": float(np.mean([recall(preds[q], gold[q]) for q in ids])),
        "precision_at_5": float(np.mean([precision(preds[q], gold[q]) for q in ids])),
    }

def rank_map(order: Sequence[str]) -> Dict[str, int]:
    return {str(d): i + 1 for i, d in enumerate(order[:DEPTH])}

def doc_phi(doc: str, a: Dict[str, int], b: Dict[str, int]) -> np.ndarray:
    ranks = [a.get(doc), b.get(doc)]
    present = sorted(r for r in ranks if r is not None)
    rr = [1.0 / (RRF_K + r) if r is not None else 0.0 for r in ranks]
    rr_sorted = sorted(rr, reverse=True)
    best_q = (DEPTH + 1 - present[0]) / DEPTH if present else 0.0
    if len(present) == 2:
        second_q = (DEPTH + 1 - present[1]) / DEPTH
        close = 1.0 - abs(present[0] - present[1]) / (DEPTH - 1)
    else:
        second_q = 0.0
        close = 0.0
    return np.asarray([
        rr[0] + rr[1],
        rr_sorted[0],
        rr_sorted[1],
        best_q,
        second_q,
        sum(r is not None and r <= 5 for r in ranks) / 2.0,
        sum(r is not None and r <= 10 for r in ranks) / 2.0,
        sum(r is not None and r <= 20 for r in ranks) / 2.0,
        sum(r is not None and r <= 50 for r in ranks) / 2.0,
        close,
    ], dtype=np.float64)

def pair_feature(chal: str, defend: str, a: Dict[str, int], b: Dict[str, int]) -> np.ndarray:
    return doc_phi(chal, a, b) - doc_phi(defend, a, b)

def novel_union(a: Sequence[str], b: Sequence[str], pool: Sequence[str]) -> List[str]:
    poolset = set(map(str, pool))
    merged = list(dict.fromkeys([str(x) for x in a[:DEPTH]] + [str(x) for x in b[:DEPTH]]))
    return [d for d in merged if d not in poolset]

def action_outcomes(base, modified, gold, ids):
    ben = harm = neu = 0
    mass = 0.0
    details = []
    for q in ids:
        if base[q][:5] == modified[q][:5]:
            continue
        before, after = recall(base[q], gold[q]), recall(modified[q], gold[q])
        d = after - before
        if d > 0:
            ben += 1; kind = "beneficial"
        elif d < 0:
            harm += 1; kind = "harmful"
        else:
            neu += 1; kind = "neutral"
        mass += d
        details.append({"qid": q, "delta": d, "outcome": kind,
                        "base_top5": base[q][:5], "modified_top5": modified[q][:5]})
    return {
        "actions": len(details),
        "beneficial": ben,
        "harmful": harm,
        "neutral": neu,
        "weighted_query_recall_mass": mass,
        "intervention_precision_excluding_neutral": ben / max(ben + harm, 1),
        "details": details,
    }

def load_v2(root: Path):
    bundle = root / "cache/research_v2_e5_confirmation/bundle-v1"
    pool_path = bundle / "V2_CANDIDATE_POOL.jsonl"
    anchor_path = root / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
    source_db = root.parent / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"
    for p in (pool_path, anchor_path, source_db):
        if not p.is_file():
            raise FileNotFoundError(p)

    pool = {str(r["qid"]): [str(d) for d in r["doc_ids"]] for r in read_jsonl(pool_path)}
    anchor_rows = {str(r["qid"]): r for r in read_jsonl(anchor_path)}
    if len(pool) != 6991 or set(pool) != set(anchor_rows):
        raise RuntimeError("V2 pool/anchor population mismatch")
    gold = {q: {str(d) for d in r["gold"]} for q, r in anchor_rows.items()}
    fold_for = {q: str(r["fold"]) for q, r in anchor_rows.items()}
    base = {q: [str(d) for d in r["fused_top5"]] for q, r in anchor_rows.items()}

    e5 = {}
    e5_hash = {}
    for i in range(5):
        p = root / f"results/research_v2_open_rl/fold_{i}/FULL_CORPUS_PREDICTIONS.jsonl"
        if not p.is_file():
            raise FileNotFoundError(p)
        e5_hash[f"fold_{i}"] = sha256_file(p)
        for r in read_jsonl(p):
            q = str(r["qid"])
            if q in e5:
                raise RuntimeError(f"duplicate V2 E5 qid {q}")
            e5[q] = [str(d) for d in r["adapted_order_top150"]]
    if set(e5) != set(pool):
        raise RuntimeError("V2 E5 population mismatch")

    lal = {}
    con = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("LAL source DB integrity failed")
    for q in sorted(pool, key=int):
        row = con.execute("SELECT payload FROM sources WHERE q=? AND source='lal'", (q,)).fetchone()
        if row is None:
            raise RuntimeError(f"missing LAL source qid={q}")
        lal[q] = [str(x["doc_id"]) for x in json.loads(row[0])]
    con.close()

    qids = sorted(pool, key=int)
    prov = {
        "pool_sha256": sha256_file(pool_path),
        "anchor_sha256": sha256_file(anchor_path),
        "e5_fold_sha256": e5_hash,
        "lal_source_db_sha256": sha256_file(source_db),
    }
    return qids, pool, gold, fold_for, base, e5, lal, prov

def build_rows(ids, pool, gold, base, sa, sb, labels=True):
    xs, ys, ws = [], [], []
    for q in ids:
        defend = base[q][4]
        ra, rb = rank_map(sa[q]), rank_map(sb[q])
        chals = novel_union(sa[q], sb[q], pool[q])
        if not chals:
            continue
        w = 1.0 / len(chals)
        for c in chals:
            xs.append(pair_feature(c, defend, ra, rb))
            ws.append(w)
            if labels:
                ys.append((int(c in gold[q]) - int(defend in gold[q])) / len(gold[q]))
    X = np.vstack(xs) if xs else np.empty((0, 10), np.float64)
    return X, np.asarray(ys, np.float64), np.asarray(ws, np.float64)

def apply_model(ids, pool, base, sa, sb, scaler, model):
    out = {q: list(base[q][:5]) for q in ids}
    actions = {}
    for q in ids:
        defend = base[q][4]
        ra, rb = rank_map(sa[q]), rank_map(sb[q])
        chals = novel_union(sa[q], sb[q], pool[q])
        if not chals:
            continue
        X = np.vstack([pair_feature(c, defend, ra, rb) for c in chals])
        pred = model.predict(scaler.transform(X))
        order = sorted(range(len(chals)), key=lambda i: (-float(pred[i]), chals[i]))
        i = order[0]
        if float(pred[i]) > 0.0:
            c = chals[i]
            out[q] = list(base[q][:4]) + [c]
            actions[q] = {
                "qid": q, "defender": defend, "challenger": c,
                "predicted_utility": float(pred[i]),
                "novel_count": len(chals),
                "challenger_source_ranks": {"source_1": ra.get(c), "source_2": rb.get(c)},
                "defender_source_ranks": {"source_1": ra.get(defend), "source_2": rb.get(defend)},
            }
    return out, actions

def v2_oof(root: Path, outdir: Path):
    qids, pool, gold, fold_for, base, e5, lal, prov = load_v2(root)
    folds = [f"fold_{i}" for i in range(5)]
    final = {}
    all_actions = {}
    fold_reports = {}
    for held in folds:
        train = [q for q in qids if fold_for[q] != held]
        test = [q for q in qids if fold_for[q] == held]
        X, y, w = build_rows(train, pool, gold, base, e5, lal, True)
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(X), y, sample_weight=w)
        mod, actions = apply_model(test, pool, base, e5, lal, scaler, model)
        final.update(mod); all_actions.update(actions)
        bm, mm = metrics(base, gold, test), metrics(mod, gold, test)
        oc = action_outcomes(base, mod, gold, test); oc.pop("details")
        fold_reports[held] = {
            "queries": len(test), "training_rows": len(X),
            "target_positive": int(np.sum(y > 0)),
            "target_negative": int(np.sum(y < 0)),
            "target_neutral": int(np.sum(y == 0)),
            "baseline": bm, "modified": mm,
            "recall_delta": mm["recall_at_5"] - bm["recall_at_5"],
            "actions": oc,
        }
    bm, mm = metrics(base, gold, qids), metrics(final, gold, qids)
    outcomes = action_outcomes(base, final, gold, qids)
    deltas = [fold_reports[f]["recall_delta"] for f in folds]
    gate = {
        "delta_gte_0_001": mm["recall_at_5"] - bm["recall_at_5"] >= 0.001,
        "wins_gt_losses": outcomes["beneficial"] > outcomes["harmful"],
        "at_least_4_of_5_folds_nonnegative": sum(d >= 0 for d in deltas) >= 4,
        "worst_fold_gte_minus_0_005": min(deltas) >= -0.005,
    }
    report = {
        "schema": "manual.strict_v2_transfer_boundary_verifier_v1.v2",
        "status": "PASS_V2_GATE" if all(gate.values()) else "KILL_AT_V2_GATE",
        "mechanism": {
            "depth": DEPTH, "feature_dim": 10,
            "features": "source-agnostic challenger-minus-defender rank support",
            "target": "exact one-swap Recall@5 utility",
            "model": f"Ridge(alpha={RIDGE_ALPHA})",
            "weighting": "query-balanced 1/n_novel",
            "action_rule": "best predicted utility > 0",
        },
        "provenance": prov,
        "baseline": bm, "modified": mm,
        "delta_recall_at_5": mm["recall_at_5"] - bm["recall_at_5"],
        "paired_actions": outcomes,
        "folds": fold_reports, "gate_checks": gate, "gate_pass": all(gate.values()),
    }
    write_json(outdir / "V2_OOF_REPORT.json", report)
    return dict(report=report, qids=qids, pool=pool, gold=gold, base=base, e5=e5, lal=lal)

def fit_all(v2):
    X, y, w = build_rows(v2["qids"], v2["pool"], v2["gold"], v2["base"], v2["e5"], v2["lal"], True)
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(X), y, sample_weight=w)
    meta = {
        "rows": len(X), "positive": int(np.sum(y > 0)), "negative": int(np.sum(y < 0)),
        "neutral": int(np.sum(y == 0)), "coef": model.coef_.astype(float).tolist(),
        "intercept": float(model.intercept_), "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
    }
    return scaler, model, meta

def load_cal_label_free(root: Path):
    from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import load_cal_data_label_free
    _, _, blocks, all_ids, pool, _, _, _, _ = load_cal_data_label_free()
    all_ids = [str(q) for q in all_ids]
    pool = {str(q): [str(d) for d in ds] for q, ds in pool.items()}
    blocks = {str(b): [str(q) for q in ids] for b, ids in blocks.items()}

    bp = root / "results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json"
    ep = root / "results/manual/huy_cal600_adapted_e5_full_corpus_v1/CAL600_ADAPTED_E5_FULL_CORPUS_TOP150.json"
    ap = root / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_TOP50.json"
    for p in (bp, ep, ap):
        if not p.is_file():
            raise FileNotFoundError(p)
    base = {str(q): [str(d) for d in row] for q, row in json.loads(bp.read_text(encoding="utf-8")).items()}
    e5 = {str(q): [str(d) for d in row] for q, row in json.loads(ep.read_text(encoding="utf-8")).items()}
    ai = {str(q): [str(d) for d in row] for q, row in json.loads(ap.read_text(encoding="utf-8")).items()}
    s = set(all_ids)
    if not (set(base) == set(e5) == set(ai) == set(pool) == s):
        raise RuntimeError("CAL population mismatch")
    return all_ids, blocks, pool, base, e5, ai

def eval_cal(root: Path, outdir: Path, scaler, model, train_meta):
    ids, blocks, pool, base, e5, ai = load_cal_label_free(root)
    modified, actions = apply_model(ids, pool, base, e5, ai, scaler, model)

    action_doc = {
        "schema": "manual.strict_v2_transfer_boundary_verifier_v1.cal_actions",
        "status": "SEALED_BEFORE_CAL_GOLD_REVEAL",
        "training_population": "Strict-V2 6991 only",
        "transfer": "source-agnostic Adapted-E5+LAL -> Adapted-E5+AITeamVN-FT",
        "depth": DEPTH, "actions_count": len(actions), "actions": actions,
        "predictions": modified, "final_v2_model": train_meta,
        "hashes": {
            "d1": sha256_file(root / "results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json"),
            "e5": sha256_file(root / "results/manual/huy_cal600_adapted_e5_full_corpus_v1/CAL600_ADAPTED_E5_FULL_CORPUS_TOP150.json"),
            "aiteam": sha256_file(root / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_TOP50.json"),
        },
    }
    action_path = outdir / "CAL_ACTIONS_LABEL_FREE.json"
    write_json(action_path, action_doc)
    seal = sha256_file(action_path)

    from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import load_cal_gold_labels
    gold, reveal_time = load_cal_gold_labels(ids)
    bm, mm = metrics(base, gold, ids), metrics(modified, gold, ids)
    if abs(bm["recall_at_5"] - EXPECTED_CAL_D1_R5) > 1e-12:
        raise RuntimeError(f"D1 recall parity failed {bm}")
    if abs(bm["precision_at_5"] - EXPECTED_CAL_D1_P5) > 5e-10:
        raise RuntimeError(f"D1 precision parity failed {bm}")
    outcomes = action_outcomes(base, modified, gold, ids)
    block_delta = {}
    for b, qids in blocks.items():
        block_delta[b] = metrics(modified, gold, qids)["recall_at_5"] - metrics(base, gold, qids)["recall_at_5"]
    single = [q for q in ids if len(gold[q]) == 1]
    multi = [q for q in ids if len(gold[q]) > 1]
    delta_r = mm["recall_at_5"] - bm["recall_at_5"]
    delta_p = mm["precision_at_5"] - bm["precision_at_5"]
    gates = {
        "recall_positive": delta_r > 0,
        "precision_no_decrease": delta_p >= -1e-12,
        "wins_gt_losses": outcomes["beneficial"] > outcomes["harmful"],
        "no_block_decrease": all(v >= -1e-12 for v in block_delta.values()),
    }
    if all(gates.values()) and mm["recall_at_5"] >= 0.96:
        verdict = "STRONG_PROMOTE_TRANSFER_BOUNDARY_VERIFIER_V1"
    elif delta_r > 0 and delta_p >= -1e-12 and outcomes["beneficial"] > outcomes["harmful"]:
        verdict = "PROMISING_TRANSFER_BOUNDARY_VERIFIER_V1"
    else:
        verdict = "KILL_TRANSFER_BOUNDARY_VERIFIER_V1"
    report = {
        "schema": "manual.strict_v2_transfer_boundary_verifier_v1.cal",
        "gold_reveal_time": reveal_time, "action_seal_sha256": seal,
        "baseline": bm, "modified": mm,
        "delta": {
            "recall_at_5": delta_r, "precision_at_5": delta_p,
            "single_gold_recall": metrics(modified, gold, single)["recall_at_5"] - metrics(base, gold, single)["recall_at_5"],
            "multi_gold_recall": metrics(modified, gold, multi)["recall_at_5"] - metrics(base, gold, multi)["recall_at_5"],
            "blocks": block_delta,
        },
        "paired_actions": outcomes, "promotion_gates": gates, "verdict": verdict,
    }
    write_json(outdir / "CAL_FINAL_REPORT.json", report)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    outdir = root / "results/manual/huy_strict_v2_transfer_boundary_verifier_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Strict-V2 OOF transfer gate...", flush=True)
    v2 = v2_oof(root, outdir)
    r = v2["report"]
    print(f"  V2 {r['baseline']['recall_at_5']:.10f} -> {r['modified']['recall_at_5']:.10f} ({r['delta_recall_at_5']:+.10f})")
    p = r["paired_actions"]
    print(f"  actions={p['actions']} beneficial/harmful/neutral={p['beneficial']}/{p['harmful']}/{p['neutral']}")
    print(f"  gate={r['gate_pass']} {r['gate_checks']}")
    if not r["gate_pass"]:
        print("[2/5] STOP: Strict-V2 gate failed; CAL gold NOT read.")
        print("=" * 88)
        print("Verdict : KILL_AT_STRICT_V2_GATE")
        print(f"Report  : {outdir / 'V2_OOF_REPORT.json'}")
        print("=" * 88)
        return

    print("[2/5] Fit final verifier on all Strict-V2...", flush=True)
    scaler, model, meta = fit_all(v2)
    write_json(outdir / "FINAL_V2_MODEL.json", meta)

    print("[3/5] Transfer to CAL label-free E5+AITeam frontier...", flush=True)
    print("[4/5] Seal actions, then reveal CAL utility...", flush=True)
    cal = eval_cal(root, outdir, scaler, model, meta)

    print("[5/5] DONE")
    print("=" * 88)
    print(f"D1        R@5={cal['baseline']['recall_at_5']:.10f} P@5={cal['baseline']['precision_at_5']:.10f}")
    print(f"Transfer  R@5={cal['modified']['recall_at_5']:.10f} P@5={cal['modified']['precision_at_5']:.10f}")
    print(f"Delta     R={cal['delta']['recall_at_5']:+.10f} P={cal['delta']['precision_at_5']:+.10f}")
    print(f"Single Δ  {cal['delta']['single_gold_recall']:+.10f} | Multi Δ {cal['delta']['multi_gold_recall']:+.10f}")
    print("Blocks    " + " ".join(f"{b}:{d:+.6f}" for b, d in sorted(cal["delta"]["blocks"].items())))
    p = cal["paired_actions"]
    print(f"Actions   {p['actions']} | beneficial={p['beneficial']} harmful={p['harmful']} neutral={p['neutral']}")
    print(f"Verdict   {cal['verdict']}")
    print(f"Report    {outdir / 'CAL_FINAL_REPORT.json'}")
    print("=" * 88)

if __name__ == "__main__":
    main()
