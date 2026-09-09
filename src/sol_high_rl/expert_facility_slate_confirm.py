"""Open locked F2 confirmation folds, then pooled evidence only if gate passes."""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from forensic_world_model import metrics, prepare_contract  # noqa: E402
from expert_facility_slate_dev import LAMBDA, paired, select  # noqa: E402
from seal_cal600_protocol import paired_bootstrap  # noqa: E402

OUT = ROOT / "results/sol_high_rl"
CONF = ("fold_3", "fold_4")


def main():
    started = time.perf_counter()
    split_bytes = (OUT / "CAL600_STRATIFIED_5FOLD_SEED42.json").read_bytes()
    split = json.loads(split_bytes)
    base_report = json.loads((OUT / "CAL600_CANONICAL_BASELINE_REPORT.json").read_text(encoding="utf-8"))
    dev_report = json.loads((OUT / "F2_EXPERT_FACILITY_DEV_REPORT.json").read_text(encoding="utf-8"))
    if dev_report["status"] != "PROMOTE_TO_CONFIRMATION" or dev_report["objective"]["lambda"] != LAMBDA:
        raise RuntimeError("F2 DEV promotion/objective lock absent")
    if hashlib.sha256(split_bytes).hexdigest() != base_report["protocol"]["split_sha256"]:
        raise RuntimeError("split mismatch")
    baseline = json.loads((OUT / "CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    dev_pred = json.loads((OUT / "F2_EXPERT_FACILITY_DEV_PREDICTIONS.json").read_text(encoding="utf-8"))
    queries, old_blocks, all_ids, candidates, views, scores, names, _, _, _ = prepare_contract()
    cbytes = (json.dumps({q: candidates[q] for q in all_ids}, ensure_ascii=False, indent=2) + "\n").encode()
    if hashlib.sha256(cbytes).hexdigest() != base_report["candidate_contract_sha256"]:
        raise RuntimeError("candidate mismatch")
    rankers = [{q: list(views[n][q]) for q in all_ids} for n in names]
    for n in sorted(scores):
        rankers.append({q: sorted(candidates[q], key=lambda d: (-scores[n][q].get(d, -1e30), d)) for q in all_ids})
    conf_ids = sum((split["folds"][f] for f in CONF), [])
    conf_pred = {q: select(baseline[q], [r[q] for r in rankers], candidates[q]) for q in conf_ids}
    bm = metrics(baseline, queries, conf_ids)[0]; cm = metrics(conf_pred, queries, conf_ids)[0]
    fold_rows = {}
    for f in CONF:
        ids = split["folds"][f]; b = metrics(baseline, queries, ids)[0]; c = metrics(conf_pred, queries, ids)[0]
        fold_rows[f] = {"baseline": b, "candidate": c, "delta": c["recall_at_5"] - b["recall_at_5"],
                        "paired": paired(conf_pred, baseline, queries, ids)}
    multi = [q for q in conf_ids if len(queries[q][1]) > 1]
    mb = metrics(baseline, queries, multi)[0]; mc = metrics(conf_pred, queries, multi)[0]
    delta = cm["recall_at_5"] - bm["recall_at_5"]
    pp = paired(conf_pred, baseline, queries, conf_ids)
    gate = delta >= 0 and min(x["delta"] for x in fold_rows.values()) >= -.005 and pp["wins"] >= pp["losses"] and mc["recall_at_5"] - mb["recall_at_5"] >= -.02
    report = {"status": "CONFIRM_PASS" if gate else "CONFIRM_FAIL", "family": "F2_expert_facility_slate",
              "locked_objective_sha256": hashlib.sha256((HERE / "expert_facility_slate_dev.py").read_bytes()).hexdigest(),
              "confirmation": {"baseline": bm, "candidate": cm, "delta": delta, "paired": pp,
                               "folds": fold_rows, "multi_gold": {"baseline": mb, "candidate": mc,
                               "delta": mc["recall_at_5"] - mb["recall_at_5"]}},
              "confirmation_gate_passed": gate}
    if gate:
        full = dict(dev_pred); full.update(conf_pred)
        bmet, bpq = metrics(baseline, queries, all_ids); cmet, cpq = metrics(full, queries, all_ids)
        singles = [q for q in all_ids if len(queries[q][1]) == 1]
        multis = [q for q in all_ids if len(queries[q][1]) > 1]
        report["pooled"] = {"baseline": bmet, "candidate": cmet,
                            "delta": cmet["recall_at_5"] - bmet["recall_at_5"],
                            "paired": paired(full, baseline, queries, all_ids),
                            "bootstrap": paired_bootstrap([cpq[q] for q in all_ids], [bpq[q] for q in all_ids]),
                            "single_gold": {"baseline": metrics(baseline, queries, singles)[0], "candidate": metrics(full, queries, singles)[0]},
                            "multi_gold": {"baseline": metrics(baseline, queries, multis)[0], "candidate": metrics(full, queries, multis)[0]},
                            "old_block_stress": {b: {"baseline": metrics(baseline, queries, ids)[0],
                                                     "candidate": metrics(full, queries, ids)[0],
                                                     "delta": metrics(full, queries, ids)[0]["recall_at_5"] - metrics(baseline, queries, ids)[0]["recall_at_5"],
                                                     "paired": paired(full, baseline, queries, ids)} for b, ids in old_blocks.items()}}
        (OUT / "F2_EXPERT_FACILITY_POOLED_PREDICTIONS.json").write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["runtime_seconds"] = time.perf_counter() - started
    (OUT / "F2_EXPERT_FACILITY_CONFIRM_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "confirmation": {"delta": delta, "paired": pp,
               "fold_deltas": {f: x["delta"] for f, x in fold_rows.items()},
               "multi_delta": report["confirmation"]["multi_gold"]["delta"]}}
    if gate:
        summary["pooled"] = {"delta": report["pooled"]["delta"], "paired": report["pooled"]["paired"],
                             "bootstrap": report["pooled"]["bootstrap"],
                             "single_delta": report["pooled"]["single_gold"]["candidate"]["recall_at_5"] - report["pooled"]["single_gold"]["baseline"]["recall_at_5"],
                             "multi_delta": report["pooled"]["multi_gold"]["candidate"]["recall_at_5"] - report["pooled"]["multi_gold"]["baseline"]["recall_at_5"],
                             "old_block_deltas": {b: x["delta"] for b, x in report["pooled"]["old_block_stress"].items()}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
