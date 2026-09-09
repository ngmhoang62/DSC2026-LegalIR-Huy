"""F1 preregistered DEV-only audit of two fixed historical ranking mechanisms."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from forensic_world_model import metrics, prepare_contract  # noqa: E402
from tune_expanded_fusion_selection import ltr_features  # noqa: E402

OUT = ROOT / "results/sol_high_rl"
DEST = OUT / "F1_FIXED_CAPACITY_DEV_REPORT.json"
PRED = OUT / "F1_FIXED_CAPACITY_DEV_PREDICTIONS.json"
DEV_FOLDS = ("fold_0", "fold_1", "fold_2")


def paired(candidate, baseline, queries, ids):
    c = {q: len(set(candidate[q][:5]) & queries[q][1]) / len(queries[q][1]) for q in ids}
    b = {q: len(set(baseline[q][:5]) & queries[q][1]) / len(queries[q][1]) for q in ids}
    return {
        "wins": sum(c[q] > b[q] for q in ids), "losses": sum(c[q] < b[q] for q in ids),
        "ties": sum(c[q] == b[q] for q in ids),
        "changed_top5_sets": sum(set(candidate[q][:5]) != set(baseline[q][:5]) for q in ids),
    }


def pairwise_fit_predict(rows, groups, queries, base_view, train_ids, test_ids):
    xs, ys, weights = [], [], []
    for q in train_ids:
        index = {d: i for i, d in enumerate(groups[q])}
        positives = [index[d] for d in queries[q][1] if d in index]
        negatives = [index[d] for d in base_view[q] if d in index and d not in queries[q][1]][:20]
        if not positives or not negatives:
            continue
        w = 1.0 / (2 * len(positives) * len(negatives))
        for pi in positives:
            for ni in negatives:
                diff = rows[q][pi] - rows[q][ni]
                xs.extend((diff, -diff)); ys.extend((1, 0)); weights.extend((w, w))
    x = np.asarray(xs, dtype=np.float32)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.03, solver="liblinear", max_iter=2000, random_state=2026)
    model.fit(scaler.transform(x), np.asarray(ys, dtype=np.int8), sample_weight=np.asarray(weights))
    out = {}
    for q in test_ids:
        score = model.decision_function(scaler.transform(rows[q]))
        out[q] = [groups[q][i] for i in np.argsort(-score, kind="stable")]
    return out, {"pair_rows": len(ys), "positive_rows": int(sum(ys))}


def xgb_fit_predict(rows, groups, queries, train_ids, test_ids):
    x = np.vstack([rows[q] for q in train_ids])
    y = np.concatenate([[d in queries[q][1] for d in groups[q]] for q in train_ids]).astype(np.int8)
    sizes = [len(groups[q]) for q in train_ids]
    model = XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@5", tree_method="hist",
        n_estimators=180, max_depth=2, learning_rate=.03, min_child_weight=3,
        subsample=.85, colsample_bytree=.9, reg_lambda=8.0, reg_alpha=.05,
        n_jobs=2, lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=10,
        random_state=2026,
    )
    model.fit(x, y, group=sizes, verbose=False)
    out = {}
    for q in test_ids:
        score = model.predict(rows[q])
        out[q] = [groups[q][i] for i in np.argsort(-score, kind="stable")]
    return out, {"train_rows": len(y), "positive_rows": int(y.sum())}


def main():
    started = time.perf_counter()
    split_bytes = (OUT / "CAL600_STRATIFIED_5FOLD_SEED42.json").read_bytes()
    split = json.loads(split_bytes)
    base_report = json.loads((OUT / "CAL600_CANONICAL_BASELINE_REPORT.json").read_text(encoding="utf-8"))
    if hashlib.sha256(split_bytes).hexdigest() != base_report["protocol"]["split_sha256"]:
        raise RuntimeError("split lock mismatch")
    baseline = json.loads((OUT / "CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    queries, _, all_ids, candidates, views, scores, names, _, extras, _ = prepare_contract()
    candidate_bytes = (json.dumps({q: candidates[q] for q in all_ids}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if hashlib.sha256(candidate_bytes).hexdigest() != base_report["candidate_contract_sha256"]:
        raise RuntimeError("candidate lock mismatch")
    rows, groups = ltr_features(views, names, candidates, all_ids, scores)
    for extra in extras:
        for q in all_ids:
            rows[q] = np.concatenate([rows[q], extra[q]], axis=1)

    predictions = {"pairwise_logistic": {}, "shallow_lambdarank": {}}
    fit_diagnostics = {}
    for fold in DEV_FOLDS:
        test_ids = split["folds"][fold]
        train_ids = [q for q in all_ids if q not in set(test_ids)]
        p, pd = pairwise_fit_predict(rows, groups, queries, views["base"], train_ids, test_ids)
        x, xd = xgb_fit_predict(rows, groups, queries, train_ids, test_ids)
        predictions["pairwise_logistic"].update(p)
        predictions["shallow_lambdarank"].update(x)
        fit_diagnostics[fold] = {"pairwise_logistic": pd, "shallow_lambdarank": xd}

    dev_ids = sum((split["folds"][f] for f in DEV_FOLDS), [])
    single = [q for q in dev_ids if len(queries[q][1]) == 1]
    multi = [q for q in dev_ids if len(queries[q][1]) > 1]
    base_dev = metrics(baseline, queries, dev_ids)[0]
    known = {(r["qid"], r["docid"]) for r in json.loads(
        (OUT / "RANKING_FAILURE_ANATOMY_V2.json").read_text(encoding="utf-8"))
        ["miss_decomposition"]["rows"] if r["known_by_any_expert_top5"] and r["qid"] in set(dev_ids)}
    variants = {}
    for name, pred in predictions.items():
        overall = metrics(pred, queries, dev_ids)[0]
        corrected = sum(d in pred[q][:5] for q, d in known)
        harmed = sum(d not in pred[q][:5] for q in dev_ids for d in queries[q][1] if d in baseline[q][:5])
        variants[name] = {
            "config": ({"negative_depth": 20, "C": .03, "objective": "symmetric_pairwise_logistic"}
                       if name == "pairwise_logistic" else
                       {"depth": 2, "rate": .03, "trees": 180, "min_child": 3, "pairs": 10,
                        "objective": "rank:ndcg@5"}),
            "dev": overall,
            "dev_recall_delta": overall["recall_at_5"] - base_dev["recall_at_5"],
            "folds": {f: {
                "metrics": metrics(pred, queries, split["folds"][f])[0],
                "recall_delta": metrics(pred, queries, split["folds"][f])[0]["recall_at_5"] - metrics(baseline, queries, split["folds"][f])[0]["recall_at_5"],
                "paired": paired(pred, baseline, queries, split["folds"][f]),
            } for f in DEV_FOLDS},
            "single_gold": metrics(pred, queries, single)[0],
            "multi_gold": metrics(pred, queries, multi)[0],
            "multi_gold_delta": metrics(pred, queries, multi)[0]["recall_at_5"] - metrics(baseline, queries, multi)[0]["recall_at_5"],
            "paired": paired(pred, baseline, queries, dev_ids),
            "targeted_known_expert_missed_occurrences": len(known),
            "targeted_occurrences_corrected": corrected,
            "previously_selected_gold_occurrences_harmed": harmed,
        }
    order = sorted(variants, key=lambda n: (
        variants[n]["dev"]["recall_at_5"], variants[n]["dev"]["precision_at_5_fixed5"],
        variants[n]["paired"]["wins"] - variants[n]["paired"]["losses"]), reverse=True)
    champion = order[0]
    v = variants[champion]
    positive_folds = sum(x["recall_delta"] > 0 for x in v["folds"].values())
    promotion = v["dev_recall_delta"] >= .005 and positive_folds >= 2 and v["multi_gold_delta"] >= -.02
    report = {
        "status": "PROMOTE_TO_CONFIRMATION" if promotion else "REJECT",
        "family": "F1_fixed_historical_ranking_objective_capacity",
        "protocol": {"development_folds": list(DEV_FOLDS), "confirmation_outcomes_inspected": False,
                     "candidate_contract_sha256": base_report["candidate_contract_sha256"],
                     "split_sha256": base_report["protocol"]["split_sha256"]},
        "baseline_dev": base_dev, "variants": variants, "predeclared_champion": champion,
        "promotion_gate": {"delta_ge": .005, "positive_folds_ge": 2, "multi_delta_ge": -.02,
                           "passed": promotion},
        "fit_diagnostics": fit_diagnostics, "runtime_seconds": time.perf_counter() - started,
    }
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PRED.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "baseline_dev": base_dev,
                      "variants": {n: {"delta": variants[n]["dev_recall_delta"],
                                           "fold_deltas": {f: x["recall_delta"] for f, x in variants[n]["folds"].items()},
                                           "paired": variants[n]["paired"],
                                           "multi_delta": variants[n]["multi_gold_delta"],
                                           "targeted_corrected": variants[n]["targeted_occurrences_corrected"],
                                           "selected_harmed": variants[n]["previously_selected_gold_occurrences_harmed"]}
                                   for n in variants},
                      "champion": champion, "runtime_seconds": report["runtime_seconds"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
