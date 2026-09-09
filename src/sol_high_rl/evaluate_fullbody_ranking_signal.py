"""One canonical fallback: AITeamVN-FT full-body signal on fixed candidates."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from forensic_world_model import fit_rankings, metrics, prepare_contract
from seal_cal600_protocol import paired_bootstrap


CACHE = ROOT / "cache/sol_high_rl/aiteam_ft_full_corpus"
SCORE_PATH = CACHE / "cal600_current_candidate_fullbody_scores.pkl"
REPORT_PATH = ROOT / "results/sol_high_rl/FULLBODY_RANKING_SIGNAL_REPORT.json"
PRED_PATH = ROOT / "results/sol_high_rl/FULLBODY_RANKING_SIGNAL_OOF_PREDICTIONS.json"


def build_scores(candidates, all_ids):
    if SCORE_PATH.exists():
        return pickle.loads(SCORE_PATH.read_bytes())
    meta = json.loads((CACHE / "chunks_cap32.json").read_text(encoding="utf-8"))
    counts = meta["counts"]
    offsets, cursor = {}, 0
    for docid, count in zip(meta["documents"], counts):
        offsets[docid] = (cursor, cursor + count)
        cursor += count
    vectors = np.memmap(CACHE / "chunks_cap32.f16", mode="r", dtype=np.float16, shape=(cursor, 1024))
    qvectors = np.load(CACHE / "cal600_query_vectors_max512.f32.npy")
    qgpu = torch.from_numpy(qvectors).to(device="cuda", dtype=torch.float16)
    result = {}
    for qi, qid in enumerate(all_ids):
        docids = candidates[qid]
        indices = np.concatenate([np.arange(*offsets[d], dtype=np.int64) for d in docids])
        owners = np.concatenate([np.full(offsets[d][1] - offsets[d][0], i, dtype=np.int32) for i, d in enumerate(docids)])
        chunks = torch.from_numpy(np.asarray(vectors[indices])).to("cuda")
        raw = (chunks @ qgpu[qi]).float().cpu().numpy()
        maxima = np.full(len(docids), -np.inf, dtype=np.float32)
        np.maximum.at(maxima, owners, raw)
        result[qid] = {d: float(s) for d, s in zip(docids, maxima)}
        if (qi + 1) % 100 == 0:
            print(f"full-body candidate scores {qi+1}/600", flush=True)
    SCORE_PATH.write_bytes(pickle.dumps(result, protocol=5))
    return result


def main():
    started = time.perf_counter()
    (queries, old_blocks, all_ids, candidates, views, scores, names, _, extras, _) = prepare_contract()
    folds = json.loads((ROOT / "results/sol_high_rl/CAL600_STRATIFIED_5FOLD_SEED42.json").read_text(encoding="utf-8"))["folds"]
    baseline = json.loads((ROOT / "results/sol_high_rl/CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    fullbody = build_scores(candidates, all_ids)

    raw_lexical, raw_fullbody = [], []
    for qid in all_ids:
        for docid in candidates[qid]:
            raw_lexical.append(float(scores["aiteamvn_ft"][qid][docid]))
            raw_fullbody.append(float(fullbody[qid][docid]))
    representation = {
        "score_pairs": len(raw_lexical),
        "pearson_vs_lexical_top2_aiteam_ft": float(pearsonr(raw_lexical, raw_fullbody).statistic),
        "spearman_vs_lexical_top2_aiteam_ft": float(spearmanr(raw_lexical, raw_fullbody).statistic),
        "mean_abs_score_difference": float(np.mean(np.abs(np.asarray(raw_lexical) - np.asarray(raw_fullbody)))),
        "query_top5_set_agreement": float(np.mean([
            len(
                set(sorted(candidates[q], key=lambda d: (-scores["aiteamvn_ft"][q][d], d))[:5])
                & set(sorted(candidates[q], key=lambda d: (-fullbody[q][d], d))[:5])
            ) / 5
            for q in all_ids
        ])),
    }

    experimental_views = dict(views)
    experimental_scores = dict(scores)
    experimental_names = list(names) + ["aiteam_ft_fullbody"]
    experimental_scores["aiteam_ft_fullbody"] = fullbody
    experimental_views["aiteam_ft_fullbody"] = {
        q: sorted(candidates[q], key=lambda d: (-fullbody[q][d], d)) for q in all_ids
    }
    prediction, _ = fit_rankings(
        queries,
        folds,
        all_ids,
        candidates,
        experimental_views,
        experimental_scores,
        experimental_names,
        extras,
    )
    base_metrics, base_per_q = metrics(baseline, queries, all_ids)
    exp_metrics, exp_per_q = metrics(prediction, queries, all_ids)
    folds_report = {}
    for name, ids in folds.items():
        bm = metrics(baseline, queries, ids)[0]
        em = metrics(prediction, queries, ids)[0]
        folds_report[name] = {"baseline": bm, "experiment": em, "recall_delta": em["recall_at_5"] - bm["recall_at_5"]}
    stress = {}
    for name, ids in old_blocks.items():
        bm = metrics(baseline, queries, ids)[0]
        em = metrics(prediction, queries, ids)[0]
        stress[name] = {"baseline": bm, "experiment": em, "recall_delta": em["recall_at_5"] - bm["recall_at_5"]}
    single = [q for q in all_ids if len(queries[q][1]) == 1]
    multi = [q for q in all_ids if len(queries[q][1]) > 1]
    wins = sum(exp_per_q[q] > base_per_q[q] for q in all_ids)
    losses = sum(exp_per_q[q] < base_per_q[q] for q in all_ids)
    changed = sum(set(prediction[q][:5]) != set(baseline[q][:5]) for q in all_ids)
    delta = exp_metrics["recall_at_5"] - base_metrics["recall_at_5"]
    multi_base = metrics(baseline, queries, multi)[0]
    multi_exp = metrics(prediction, queries, multi)[0]
    gate = (
        delta >= 0.005
        and all(row["recall_delta"] >= 0 for row in folds_report.values())
        and multi_exp["recall_at_5"] >= multi_base["recall_at_5"]
    )
    report = {
        "status": "KEEP" if gate else "REJECT",
        "hypothesis": "fine-tuned full-body max-chunk representation improves fixed-candidate ranking",
        "candidate_membership_changed": False,
        "model": "same LogisticRegression C=0.15 feature contract; one rank view plus standardized score columns",
        "representation_difference": representation,
        "baseline": base_metrics,
        "experiment": exp_metrics,
        "recall_delta": delta,
        "paired": {
            "wins": wins,
            "losses": losses,
            "ties": len(all_ids) - wins - losses,
            "changed_top5_sets": changed,
            "bootstrap": paired_bootstrap([exp_per_q[q] for q in all_ids], [base_per_q[q] for q in all_ids]),
        },
        "folds": folds_report,
        "old_block_stress": stress,
        "slices": {
            "single": {"baseline": metrics(baseline, queries, single)[0], "experiment": metrics(prediction, queries, single)[0]},
            "multi": {"baseline": multi_base, "experiment": multi_exp},
        },
        "gate": "KEEP only if delta>=0.005, every canonical fold nonnegative, and multi-gold nonnegative",
        "runtime_seconds": time.perf_counter() - started,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    PRED_PATH.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
