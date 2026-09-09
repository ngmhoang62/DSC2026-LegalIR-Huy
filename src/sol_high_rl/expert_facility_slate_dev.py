"""F2 fixed label-free submodular expert-facility slate, DEV outcomes only."""
from __future__ import annotations
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from forensic_world_model import metrics, prepare_contract  # noqa: E402

OUT = ROOT / "results/sol_high_rl"
DEV_FOLDS = ("fold_0", "fold_1", "fold_2")
LAMBDA = .20


def paired(candidate, baseline, queries, ids):
    cv = {q: len(set(candidate[q][:5]) & queries[q][1]) / len(queries[q][1]) for q in ids}
    bv = {q: len(set(baseline[q][:5]) & queries[q][1]) / len(queries[q][1]) for q in ids}
    return {"wins": sum(cv[q] > bv[q] for q in ids), "losses": sum(cv[q] < bv[q] for q in ids),
            "ties": sum(cv[q] == bv[q] for q in ids),
            "changed_top5_sets": sum(set(candidate[q][:5]) != set(baseline[q][:5]) for q in ids)}


def quality(rank):
    return 1.0 / math.log2(rank + 1.0)


def select(base, expert_lists, candidates):
    base_pos = {d: i + 1 for i, d in enumerate(base)}
    expert_pos = [{d: i + 1 for i, d in enumerate(row)} for row in expert_lists]
    covered = [0.0] * len(expert_pos)
    selected = []
    remaining = set(candidates)
    for _ in range(5):
        best = None
        for d in remaining:
            # Linear rank relevance makes lambda=0 exactly preserve baseline Top-5.
            rel = max(0.0, (33.0 - base_pos.get(d, 33)) / 32.0)
            marginal = sum(max(0.0, quality(pos.get(d, 10**6)) - covered[i])
                           for i, pos in enumerate(expert_pos)) / len(expert_pos)
            gain = rel + LAMBDA * 5.0 * marginal
            key = (gain, -base_pos.get(d, 10**6), d)
            if best is None or key > best[0]:
                best = (key, d)
        d = best[1]
        selected.append(d); remaining.remove(d)
        for i, pos in enumerate(expert_pos):
            covered[i] = max(covered[i], quality(pos.get(d, 10**6)))
    return selected + [d for d in base if d not in set(selected)]


def main():
    started = time.perf_counter()
    split_bytes = (OUT / "CAL600_STRATIFIED_5FOLD_SEED42.json").read_bytes()
    split = json.loads(split_bytes)
    report0 = json.loads((OUT / "CAL600_CANONICAL_BASELINE_REPORT.json").read_text(encoding="utf-8"))
    if hashlib.sha256(split_bytes).hexdigest() != report0["protocol"]["split_sha256"]:
        raise RuntimeError("split mismatch")
    baseline = json.loads((OUT / "CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    queries, _, all_ids, candidates, views, scores, names, _, _, _ = prepare_contract()
    cbytes = (json.dumps({q: candidates[q] for q in all_ids}, ensure_ascii=False, indent=2) + "\n").encode()
    if hashlib.sha256(cbytes).hexdigest() != report0["candidate_contract_sha256"]:
        raise RuntimeError("candidate mismatch")
    # Exactly the canonical LTR information sources: six rank views and ten score channels.
    rankers = [{q: list(views[n][q]) for q in all_ids} for n in names]
    for n in sorted(scores):
        rankers.append({q: sorted(candidates[q], key=lambda d: (-scores[n][q].get(d, -1e30), d)) for q in all_ids})
    dev_ids = sum((split["folds"][f] for f in DEV_FOLDS), [])
    pred = {q: select(baseline[q], [r[q] for r in rankers], candidates[q]) for q in dev_ids}
    bm = metrics(baseline, queries, dev_ids)[0]; cm = metrics(pred, queries, dev_ids)[0]
    multi = [q for q in dev_ids if len(queries[q][1]) > 1]
    bmulti = metrics(baseline, queries, multi)[0]; cmulti = metrics(pred, queries, multi)[0]
    folds = {}
    for f in DEV_FOLDS:
        ids = split["folds"][f]; b = metrics(baseline, queries, ids)[0]; c = metrics(pred, queries, ids)[0]
        folds[f] = {"baseline": b, "candidate": c, "delta": c["recall_at_5"] - b["recall_at_5"],
                    "paired": paired(pred, baseline, queries, ids)}
    delta = cm["recall_at_5"] - bm["recall_at_5"]
    p = paired(pred, baseline, queries, dev_ids)
    multi_delta = cmulti["recall_at_5"] - bmulti["recall_at_5"]
    promote = delta >= .005 and sum(x["delta"] > 0 for x in folds.values()) >= 2 and p["wins"] > p["losses"] and multi_delta >= -.02
    report = {"status": "PROMOTE_TO_CONFIRMATION" if promote else "REJECT", "family": "F2_expert_facility_slate",
              "protocol": {"development_folds": list(DEV_FOLDS), "confirmation_outcomes_inspected": False,
                           "split_sha256": report0["protocol"]["split_sha256"],
                           "candidate_contract_sha256": report0["candidate_contract_sha256"]},
              "objective": {"lambda": LAMBDA, "slate_size": 5, "experts": names + ["score:" + n for n in sorted(scores)],
                            "lambda_rationale": "one-of-five-slot equivalent; preregistered, not tuned"},
              "baseline_dev": bm, "candidate_dev": cm, "dev_delta": delta, "paired": p,
              "folds": folds, "multi_gold": {"baseline": bmulti, "candidate": cmulti, "delta": multi_delta},
              "promotion_gate_passed": promote, "runtime_seconds": time.perf_counter() - started}
    (OUT / "F2_EXPERT_FACILITY_DEV_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "F2_EXPERT_FACILITY_DEV_PREDICTIONS.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "delta": delta, "paired": p,
                      "fold_deltas": {f: x["delta"] for f, x in folds.items()}, "multi_delta": multi_delta,
                      "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
