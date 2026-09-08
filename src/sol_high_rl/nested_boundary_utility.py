"""Strict nested direct-utility KEEP/SWAP model for Huy's CV environment."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from forensic_world_model import metrics, prepare_contract
from tune_expanded_fusion_selection import ltr_features


OUT = ROOT / "results/sol_high_rl"
REPORT = OUT / "nested_boundary_utility_report.json"
PREDICTIONS = OUT / "NESTED_BOUNDARY_UTILITY_PREDICTIONS.json"
ALPHA = 10.0
SCORE_CHANNELS = ["jina_ft", "crossenc", "aiteamvn_ft", "vnlegal_lal", "title_embed"]


def candidate_rows(candidates, views, scores, names, all_ids, extras):
    rows, groups = ltr_features(views, names, candidates, all_ids, scores)
    for extra in extras:
        for qid in all_ids:
            rows[qid] = np.concatenate([rows[qid], extra[qid]], axis=1)
    return rows, groups


def fit_base(rows, groups, queries, train_ids, test_ids):
    x = np.vstack([rows[q] for q in train_ids])
    y = np.concatenate(
        [[d in queries[q][1] for d in groups[q]] for q in train_ids]
    ).astype(np.int8)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    ).fit(scaler.transform(x), y)
    ranking, probability = {}, {}
    for qid in test_ids:
        p = model.predict_proba(scaler.transform(rows[qid]))[:, 1]
        order = np.argsort(-p)
        ranking[qid] = [groups[qid][i] for i in order]
        probability[qid] = {groups[qid][i]: float(p[i]) for i in range(len(p))}
    return ranking, probability


def zscore_tables(scores, candidates, all_ids):
    result = {name: {} for name in SCORE_CHANNELS}
    for name in SCORE_CHANNELS:
        for qid in all_ids:
            raw = scores[name].get(qid, {})
            vals = np.asarray([raw.get(d, np.nan) for d in candidates[qid]], dtype=np.float64)
            present = vals[~np.isnan(vals)]
            mean = present.mean() if present.size else 0.0
            std = (present.std() or 1.0) if present.size else 1.0
            vals = np.where(np.isnan(vals), mean - 2.0 * std, vals)
            result[name][qid] = {
                d: float((value - mean) / std) for d, value in zip(candidates[qid], vals)
            }
    return result


def action_feature(qid, defender, challenger, challenger_rank, probability, zscores, views, names):
    pdef = probability[qid][defender]
    pchal = probability[qid][challenger]
    feature = [
        pchal - pdef,
        math.log(max(pchal, 1e-8)) - math.log(max(pdef, 1e-8)),
    ]
    feature.extend(zscores[name][qid][challenger] - zscores[name][qid][defender] for name in SCORE_CHANNELS)
    agreements = 0
    for name in names:
        pos = {d: i + 1 for i, d in enumerate(views[name][qid])}
        rd, rc = pos.get(defender, 60), pos.get(challenger, 60)
        feature.append((rd - rc) / 60.0)
        agreements += rc < rd
    feature.extend([agreements / len(names), challenger_rank / 20.0])
    return feature


def build_actions(ids, ranking, probability, queries, zscores, views, names, with_labels):
    x, y, owners = [], [], []
    for qid in ids:
        defender = ranking[qid][4]
        for challenger_rank, challenger in enumerate(ranking[qid][5:20], 6):
            x.append(action_feature(qid, defender, challenger, challenger_rank, probability, zscores, views, names))
            if with_labels:
                gold = queries[qid][1]
                y.append(((challenger in gold) - (defender in gold)) / len(gold))
            owners.append((qid, defender, challenger))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), owners


def apply_actions(ids, baseline, owners, predicted):
    by_q = {qid: [] for qid in ids}
    for owner, score in zip(owners, predicted):
        by_q[owner[0]].append((float(score), owner[1], owner[2]))
    output = {qid: list(baseline[qid]) for qid in ids}
    selected = {}
    for qid in ids:
        score, defender, challenger = max(by_q[qid], key=lambda x: x[0])
        if score > 0.0:
            assert output[qid][4] == defender
            output[qid][4] = challenger
            selected[qid] = {"predicted_utility": score, "defender": defender, "challenger": challenger}
    return output, selected


def per_action_outcomes(selected, queries):
    result = {"beneficial": 0, "harmful": 0, "neutral": 0, "weighted_utility": 0.0}
    details = []
    for qid, row in selected.items():
        gold = queries[qid][1]
        utility = ((row["challenger"] in gold) - (row["defender"] in gold)) / len(gold)
        key = "beneficial" if utility > 0 else "harmful" if utility < 0 else "neutral"
        result[key] += 1
        result["weighted_utility"] += utility
        details.append({"qid": qid, **row, "true_utility": utility, "outcome": key})
    result["actions"] = len(selected)
    result["intervention_precision_excluding_neutral"] = (
        result["beneficial"] / max(result["beneficial"] + result["harmful"], 1)
    )
    result["details"] = details
    return result


def choice_oracle(baseline, queries, all_ids):
    output = {q: list(baseline[q]) for q in all_ids}
    actions = 0
    for qid in all_ids:
        defender = baseline[qid][4]
        if defender in queries[qid][1]:
            continue
        challenger = next((d for d in baseline[qid][5:20] if d in queries[qid][1]), None)
        if challenger is not None:
            output[qid][4] = challenger
            actions += 1
    return output, actions


def main():
    started = time.perf_counter()
    (queries, blocks, all_ids, candidates, views, scores, names, _, extras, _) = prepare_contract()
    rows, groups = candidate_rows(candidates, views, scores, names, all_ids, extras)
    zscores = zscore_tables(scores, candidates, all_ids)
    frozen_baseline = json.loads((OUT / "BASELINE_LOBO_PREDICTIONS.json").read_text(encoding="utf-8"))

    final_baseline, final_prediction, all_selected = {}, {}, {}
    fold_reports = {}
    for held, test_ids in blocks.items():
        outer_train_blocks = [name for name in blocks if name != held]
        outer_train_ids = [q for name in outer_train_blocks for q in blocks[name]]
        test_base, test_prob = fit_base(rows, groups, queries, outer_train_ids, test_ids)
        assert all(test_base[q] == frozen_baseline[q] for q in test_ids)

        nested_rank, nested_prob = {}, {}
        for inner_held in outer_train_blocks:
            inner_train = [q for name in outer_train_blocks if name != inner_held for q in blocks[name]]
            rank, prob = fit_base(rows, groups, queries, inner_train, blocks[inner_held])
            nested_rank.update(rank)
            nested_prob.update(prob)

        train_x, train_y, _ = build_actions(
            outer_train_ids, nested_rank, nested_prob, queries, zscores, views, names, True
        )
        scaler = StandardScaler().fit(train_x)
        action_model = Ridge(alpha=ALPHA).fit(scaler.transform(train_x), train_y)
        test_x, _, owners = build_actions(
            test_ids, test_base, test_prob, queries, zscores, views, names, False
        )
        predicted = action_model.predict(scaler.transform(test_x))
        modified, selected = apply_actions(test_ids, test_base, owners, predicted)
        final_baseline.update(test_base)
        final_prediction.update(modified)
        all_selected.update(selected)
        base_m, _ = metrics(test_base, queries, test_ids)
        mod_m, _ = metrics(modified, queries, test_ids)
        outcomes = per_action_outcomes(selected, queries)
        outcomes.pop("details")
        fold_reports[held] = {
            "baseline": base_m,
            "modified": mod_m,
            "recall_delta": mod_m["recall_at_5"] - base_m["recall_at_5"],
            "train_target": {
                "positive": int(np.sum(train_y > 0)),
                "negative": int(np.sum(train_y < 0)),
                "neutral": int(np.sum(train_y == 0)),
            },
            "actions": outcomes,
        }

    assert final_baseline == frozen_baseline
    baseline_metrics, _ = metrics(final_baseline, queries, all_ids)
    modified_metrics, _ = metrics(final_prediction, queries, all_ids)
    outcomes = per_action_outcomes(all_selected, queries)
    oracle, oracle_actions = choice_oracle(final_baseline, queries, all_ids)
    oracle_metrics, _ = metrics(oracle, queries, all_ids)
    single = [q for q in all_ids if len(queries[q][1]) == 1]
    multi = [q for q in all_ids if len(queries[q][1]) > 1]
    report = {
        "status": "COMPLETE",
        "model": {"family": "Ridge direct utility", "alpha_fixed": ALPHA, "decision_boundary": 0.0, "features": 15},
        "nesting": "outer LOBO; inner OOF upstream LTR slates on outer-training blocks",
        "baseline": baseline_metrics,
        "modified": modified_metrics,
        "recall_delta": modified_metrics["recall_at_5"] - baseline_metrics["recall_at_5"],
        "choice_oracle": {"metrics": oracle_metrics, "actions": oracle_actions, "recall_delta": oracle_metrics["recall_at_5"] - baseline_metrics["recall_at_5"]},
        "paired_actions": outcomes,
        "slices": {
            "single": {"baseline": metrics(final_baseline, queries, single)[0], "modified": metrics(final_prediction, queries, single)[0]},
            "multi": {"baseline": metrics(final_baseline, queries, multi)[0], "modified": metrics(final_prediction, queries, multi)[0]},
        },
        "folds": fold_reports,
        "runtime_seconds": time.perf_counter() - started,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    PREDICTIONS.write_text(json.dumps(final_prediction, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = dict(report)
    compact["paired_actions"] = {k: v for k, v in outcomes.items() if k != "details"}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print(REPORT)


if __name__ == "__main__":
    main()
