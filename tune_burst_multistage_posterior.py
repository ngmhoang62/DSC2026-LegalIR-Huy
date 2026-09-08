"""Stack the complementary safe BURST posterior branches."""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path

from tune_burst_empirical_bayes_ltr import label_frequency, load_caches
from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
from tune_burst_graph_posterior import build_graph, graph_rerank
from tune_burst_kernel_posterior import load_queries, robust_rankings
from tune_burst_pairwise import fixed_metrics
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank


def weighted_rrf(rankings, weights, k):
    output = {}
    for q in rankings[0]:
        rankmaps = [{d: i + 1 for i, d in enumerate(branch[q])}
                    for branch in rankings]
        docs = set().union(*(r.keys() for r in rankmaps))
        output[q] = sorted(docs, key=lambda d: (
            -sum(w / (k + ranks.get(d, 100000))
                 for w, ranks in zip(weights, rankmaps)), d
        ))[:100]
    return output


def main():
    root = Path(__file__).resolve().parent
    queries = load_queries(root)
    qids = list(queries)
    splits = {
        "tune_a": qids[700:750],
        "validation_a": qids[750:850],
        "tune_b": qids[1150:1250],
        "validation_b": qids[1250:1350],
    }
    eval_ids = sum(splits.values(), [])
    baseline = robust_rankings(root, queries, eval_ids)
    cache = load_caches(root)

    saved_pair = pickle.loads(
        (root / "results" / "burst_empirical_pairwise" / "model.pkl").read_bytes()
    )
    pair_model = saved_pair["model"]
    pair_scaler = saved_pair["scaler"]
    pair_raw = {}
    profile_raw = {}
    graph = {}
    for name, ids in splits.items():
        held = set(ids)
        freq = label_frequency(queries, held)
        pair_features = features_for(cache, queries, ids, freq)
        pair_raw.update(pairwise_rank(pair_model, pair_scaler, pair_features, ids))

        print(f"Building profile/graph branch for {name}", flush=True)
        memory = [q for q in qids if q not in held]
        profile_model = build_profiles(queries, memory)
        for q in ids:
            profile_raw[q] = profile_rank(queries[q][0], profile_model,
                                          2, 1.2, .75, .3)
        graph_freq, adjacency, _ = build_graph(queries, memory)
        for q in ids:
            graph[q] = graph_rerank(baseline[q], graph_freq, adjacency,
                                    3, .5, "conditional", 3, .4, 0)
        del profile_model, pair_features

    # Branch 0 remains the anchor. Pair/profile/graph weights are independently
    # tuned but capped so no weak posterior can dominate the robust baseline.
    tune_ids = splits["tune_a"] + splits["tune_b"]
    tune_gold = {q: queries[q] for q in tune_ids}
    trials = []
    for pw, sw, gw in itertools.product((0.0, .05, .10, .15, .20, .30, .40), repeat=3):
        if pw + sw + gw > .65 or pw + sw + gw == 0:
            continue
        bw = 1.0 - pw - sw - gw
        weights = (bw, pw, sw, gw)
        for k in (0, 2, 5, 10, 20, 40, 80):
            ranked = weighted_rrf(
                [{q: baseline[q] for q in tune_ids},
                 {q: pair_raw[q] for q in tune_ids},
                 {q: profile_raw[q] for q in tune_ids},
                 {q: graph[q] for q in tune_ids}], weights, k,
            )
            metrics, _ = fixed_metrics(ranked, tune_gold)
            trials.append((metrics["Recall@5"], metrics["Precision@5"],
                           metrics["nDCG@10"], weights, k, metrics))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    reports = []
    for _, _, _, weights, k, tune_m in trials[:400]:
        report = {"weights_baseline_pairwise_profile_graph": weights,
                  "rrf_k": k, "tune": tune_m, "splits": {}}
        safe = True
        gain = 0.0
        for name in ("validation_a", "validation_b"):
            ids = splits[name]
            gold = {q: queries[q] for q in ids}
            ranked = weighted_rrf(
                [{q: baseline[q] for q in ids}, {q: pair_raw[q] for q in ids},
                 {q: profile_raw[q] for q in ids}, {q: graph[q] for q in ids}],
                weights, k,
            )
            bm, bp = fixed_metrics({q: baseline[q] for q in ids}, gold)
            fm, fp = fixed_metrics(ranked, gold)
            paired = {"wins": sum(a > b for a, b in zip(fp, bp)),
                      "ties": sum(a == b for a, b in zip(fp, bp)),
                      "losses": sum(a < b for a, b in zip(fp, bp))}
            safe &= paired["losses"] == 0 and fm["Precision@5"] >= bm["Precision@5"]
            gain += fm["Recall@5"] - bm["Recall@5"]
            report["splits"][name] = {"baseline": bm, "multistage": fm,
                                       "paired": paired}
        report["safe"] = bool(safe)
        report["total_validation_recall_gain"] = gain
        reports.append(report)
    reports.sort(key=lambda x: (x["safe"], x["total_validation_recall_gain"],
                                x["tune"]["Recall@5"], x["tune"]["nDCG@10"]),
                 reverse=True)
    path = root / "burst_multistage_posterior_validation.json"
    path.write_text(json.dumps({"best": reports[0], "top_trials": reports[:30]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports[0], ensure_ascii=False, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
