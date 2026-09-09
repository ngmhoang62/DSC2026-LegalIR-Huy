"""Canonical CAL600 ranking failure anatomy, cache-only and label-diagnostic.

This script never fits a model.  It verifies the sealed split/candidate contract,
then decomposes the already-sealed canonical OOF predictions.  Expert ranks are
frozen score/rank views and are used only for diagnosis, not selection.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from forensic_world_model import metrics, prepare_contract  # noqa: E402
from tune_doctype_features import doc_type  # noqa: E402
from tune_title_features import extract_title, tokenize  # noqa: E402

OUT = ROOT / "results/sol_high_rl"
SPLIT = OUT / "CAL600_STRATIFIED_5FOLD_SEED42.json"
BASE_REPORT = OUT / "CAL600_CANONICAL_BASELINE_REPORT.json"
BASE_PRED = OUT / "CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json"
DEST = OUT / "RANKING_FAILURE_ANATOMY_V2.json"


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def mean(values):
    return float(np.mean(values)) if values else None


def query_type(question):
    q = question.lower()
    if re.search(r"\bđiều\s+\d|\b(?:luật|nghị định|thông tư|quyết định)\s+số\b|\d{1,4}/\d{4}", q):
        return "explicit_citation_or_article"
    if any(x in q for x in ("xử phạt", "mức phạt", "bị phạt", "phạt tiền")):
        return "penalty"
    if any(x in q for x in ("thủ tục", "hồ sơ", "trình tự")):
        return "procedure"
    if any(x in q for x in ("được hưởng", "điều kiện", "quyền", "nghĩa vụ")):
        return "entitlement_or_eligibility"
    return "other"


def grouped_summary(ids, per_q, oracle_q, failure_occurrences):
    return {
        "queries": len(ids),
        "baseline_recall": mean([per_q[q] for q in ids]),
        "candidate_oracle_recall": mean([oracle_q[q] for q in ids]),
        "missed_gold_occurrences": int(sum(1 for row in failure_occurrences if row["qid"] in set(ids))),
    }


def main():
    started = time.perf_counter()
    split_raw = SPLIT.read_bytes()
    split = json.loads(split_raw)
    base_report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    baseline = json.loads(BASE_PRED.read_text(encoding="utf-8"))
    split_sha = hashlib.sha256(split_raw).hexdigest()
    if split_sha != base_report["protocol"]["split_sha256"]:
        raise RuntimeError("sealed split checksum mismatch")

    queries, old_blocks, all_ids, candidates, views, scores, _, docs, _, _ = prepare_contract()
    candidate_sha = hashlib.sha256(canonical_bytes({q: candidates[q] for q in all_ids})).hexdigest()
    if candidate_sha != base_report["candidate_contract_sha256"]:
        raise RuntimeError("candidate membership/order contract mismatch")
    if set(baseline) != set(all_ids):
        raise RuntimeError("baseline prediction population mismatch")

    fold_of = {q: fold for fold, ids in split["folds"].items() for q in ids}
    block_of = {q: block for block, ids in old_blocks.items() for q in ids}
    rankers = {f"rank:{name}": {q: list(table[q]) for q in all_ids}
               for name, table in views.items()}
    for name, table in scores.items():
        rankers[f"score:{name}"] = {
            q: sorted(candidates[q], key=lambda d: (-table[q].get(d, -1e30), d))
            for q in all_ids
        }
    expert_positions = {
        name: {q: {d: i + 1 for i, d in enumerate(rows[q])} for q in all_ids}
        for name, rows in rankers.items()
    }
    baseline_pos = {q: {d: i + 1 for i, d in enumerate(baseline[q])} for q in all_ids}

    baseline_metrics, per_q = metrics(baseline, queries, all_ids)
    oracle_q = {q: len(set(candidates[q]) & queries[q][1]) / len(queries[q][1]) for q in all_ids}
    expert_union_q = {}
    failure_rows = []
    weighted_bins = Counter()
    occurrence_bins = Counter()
    class_weight = Counter()
    class_occurrences = Counter()
    for q in all_ids:
        union5 = set().union(*(set(table[q][:5]) for table in rankers.values()))
        gold = queries[q][1]
        expert_union_q[q] = len(union5 & gold) / len(gold)
        for d in gold:
            br = baseline_pos[q].get(d)
            if br is not None and br <= 5:
                continue
            if d not in candidates[q]:
                bucket = "out_of_pool"
            elif br <= 10:
                bucket = "rank_6_10"
            elif br <= 20:
                bucket = "rank_11_20"
            else:
                bucket = "rank_gt20"
            ranks = {name: pos[q].get(d) for name, pos in expert_positions.items()}
            finite = [r for r in ranks.values() if r is not None]
            min_rank = min(finite) if finite else None
            top5 = sorted(name for name, r in ranks.items() if r is not None and r <= 5)
            top10 = sorted(name for name, r in ranks.items() if r is not None and r <= 10)
            other_gold_selected = bool((set(baseline[q][:5]) & gold) - {d})
            if bucket == "out_of_pool":
                failure_class = "out_of_pool"
            elif len(gold) > 1 and other_gold_selected:
                failure_class = "slate_multigold_coverage"
            elif min_rank is not None and min_rank <= 5:
                failure_class = "fusion_failure"
            elif min_rank is None or min_rank > 10:
                failure_class = "representation_failure"
            else:
                failure_class = "midrank_consensus_gap"
            weight = 1.0 / len(gold)
            occurrence_bins[bucket] += 1
            weighted_bins[bucket] += weight
            class_occurrences[failure_class] += 1
            class_weight[failure_class] += weight
            failure_rows.append({
                "qid": q, "docid": d, "gold_count": len(gold), "canonical_fold": fold_of[q],
                "old_block": block_of[q], "baseline_rank": br, "rank_bucket": bucket,
                "query_recall_weight": weight, "minimum_expert_rank": min_rank,
                "experts_top5": top5, "experts_top10": top10,
                "known_by_any_expert_top5": bool(top5),
                "known_by_any_expert_top10": bool(top10),
                "other_gold_already_selected": other_gold_selected,
                "exclusive_failure_class": failure_class,
            })

    # Boundary disagreement uses frozen expert order only; labels are diagnostics.
    boundary = {}
    for challenger_rank in (6, 7):
        rows = []
        vote_hist = Counter()
        state = defaultdict(list)
        for q in all_ids:
            defender, challenger = baseline[q][4], baseline[q][challenger_rank - 1]
            votes = sum(
                expert_positions[name][q].get(challenger, 10**9)
                < expert_positions[name][q].get(defender, 10**9)
                for name in rankers
            )
            gold = queries[q][1]
            key = ("defender_gold" if defender in gold else "defender_nongold") + "__" + (
                "challenger_gold" if challenger in gold else "challenger_nongold")
            vote_hist[str(votes)] += 1
            state[key].append(votes)
            rows.append({"qid": q, "defender": defender, "challenger": challenger,
                         "experts_preferring_challenger": votes, "expert_count": len(rankers),
                         "gold_state": key})
        boundary[f"rank5_vs_rank{challenger_rank}"] = {
            "expert_count": len(rankers), "vote_histogram": dict(sorted(vote_hist.items(), key=lambda x: int(x[0]))),
            "gold_state": {k: {"queries": len(v), "mean_challenger_votes": mean(v)} for k, v in state.items()},
            "rows": rows,
        }

    single_ids = [q for q in all_ids if len(queries[q][1]) == 1]
    multi_ids = [q for q in all_ids if len(queries[q][1]) > 1]
    multi_rows = []
    title_cache, type_cache = {}, {}
    def document_meta(d):
        if d not in title_cache:
            text = docs[d]
            title_cache[d] = tokenize(extract_title(text))
            type_cache[d] = doc_type(text)
        return title_cache[d], type_cache[d]
    for q in multi_ids:
        gold = queries[q][1]
        reachable = set(candidates[q]) & gold
        selected_gold = set(baseline[q][:5]) & gold
        missed_reachable = reachable - selected_gold
        nongold = [d for d in baseline[q][:5] if d not in gold]
        similarities, same_types = [], []
        for gd in missed_reachable:
            gt, gy = document_meta(gd)
            for nd in nongold:
                nt, ny = document_meta(nd)
                similarities.append(len(gt & nt) / max(len(gt | nt), 1))
                same_types.append(float(gy == ny))
        top_types = [document_meta(d)[1] for d in baseline[q][:5]]
        multi_rows.append({
            "qid": q, "gold_count": len(gold), "reachable_gold_count": len(reachable),
            "selected_gold_count": len(selected_gold), "missed_reachable_gold_count": len(missed_reachable),
            "candidate_oracle_recall": len(reachable) / len(gold), "baseline_recall": len(selected_gold) / len(gold),
            "gold_baseline_ranks": {d: baseline_pos[q].get(d) for d in sorted(gold)},
            "top5_unique_doctypes": len(set(top_types)),
            "top5_max_doctype_share": max(Counter(top_types).values()) / 5,
            "missed_gold_vs_top5_nongold_max_title_jaccard": max(similarities) if similarities else None,
            "missed_gold_vs_top5_nongold_same_doctype_fraction": mean(same_types),
        })

    def slice_map(key_fn):
        groups = defaultdict(list)
        for q in all_ids:
            groups[key_fn(q)].append(q)
        return {str(k): grouped_summary(v, per_q, oracle_q, failure_rows) for k, v in sorted(groups.items())}

    missed_weight = float(sum(r["query_recall_weight"] for r in failure_rows) / len(all_ids))
    known5_weight = float(sum(r["query_recall_weight"] for r in failure_rows if r["known_by_any_expert_top5"]) / len(all_ids))
    report = {
        "status": "COMPLETE",
        "protocol_lock": {
            "split_sha256": split_sha, "candidate_contract_sha256": candidate_sha,
            "baseline_predictions_sha256": hashlib.sha256(BASE_PRED.read_bytes()).hexdigest(),
            "population": len(all_ids), "diagnostic_only_no_model_fit": True,
        },
        "baseline": baseline_metrics,
        "candidate_oracle": {
            "all": mean([oracle_q[q] for q in all_ids]),
            "single_gold": mean([oracle_q[q] for q in single_ids]),
            "multi_gold": mean([oracle_q[q] for q in multi_ids]),
            "average_reachable_gold_count_all": mean([len(set(candidates[q]) & queries[q][1]) for q in all_ids]),
            "average_reachable_gold_count_multi": mean([len(set(candidates[q]) & queries[q][1]) for q in multi_ids]),
        },
        "headroom": {
            "candidate_ceiling_minus_baseline": mean([oracle_q[q] - per_q[q] for q in all_ids]),
            "total_missed_query_macro_recall_mass": missed_weight,
            "existing_expert_top5_union_oracle": mean([expert_union_q[q] for q in all_ids]),
            "existing_expert_union_minus_baseline": mean([expert_union_q[q] - per_q[q] for q in all_ids]),
            "final_missed_mass_known_by_any_expert_top5": known5_weight,
            "fraction_of_final_missed_mass_known_by_any_expert_top5": known5_weight / missed_weight,
        },
        "miss_decomposition": {
            "occurrence_counts": dict(occurrence_bins),
            "query_macro_recall_mass": {k: float(v / len(all_ids)) for k, v in weighted_bins.items()},
            "exclusive_failure_class_occurrences": dict(class_occurrences),
            "exclusive_failure_class_recall_mass": {k: float(v / len(all_ids)) for k, v in class_weight.items()},
            "classification_rules": [
                "out-of-pool first", "multi-gold with another gold already selected => slate_multigold_coverage",
                "else any expert top5 => fusion_failure", "else all experts beyond top10 => representation_failure",
                "remaining expert rank6-10 => midrank_consensus_gap",
            ],
            "rows": failure_rows,
        },
        "expert_diagnostics": {
            "expert_names": sorted(rankers), "expert_count": len(rankers),
            "top5_union_recall": mean([expert_union_q[q] for q in all_ids]),
            "per_expert_top5_recall": {name: metrics(table, queries, all_ids)[0]["recall_at_5"] for name, table in rankers.items()},
        },
        "boundary_disagreement": boundary,
        "multi_gold": {
            "queries": len(multi_ids), "candidate_oracle_recall": mean([oracle_q[q] for q in multi_ids]),
            "baseline_recall": mean([per_q[q] for q in multi_ids]),
            "expert_top5_union_recall": mean([expert_union_q[q] for q in multi_ids]),
            "average_reachable_gold_count": mean([len(set(candidates[q]) & queries[q][1]) for q in multi_ids]),
            "queries_with_reachable_missed_gold": sum(r["missed_reachable_gold_count"] > 0 for r in multi_rows),
            "aspect_redundancy_proxy_warning": "Title Jaccard and doctype concentration are structural proxies, not proof of semantic aspect collapse.",
            "rows": multi_rows,
        },
        "failure_concentration": {
            "canonical_fold": {k: grouped_summary(v, per_q, oracle_q, failure_rows) for k, v in split["folds"].items()},
            "old_block": {k: grouped_summary(v, per_q, oracle_q, failure_rows) for k, v in old_blocks.items()},
            "gold_count": slice_map(lambda q: len(queries[q][1])),
            "query_type": slice_map(lambda q: query_type(queries[q][0])),
            "query_length": slice_map(lambda q: "short_le12" if len(queries[q][0].split()) <= 12 else ("medium_13_25" if len(queries[q][0].split()) <= 25 else "long_gt25")),
            "query_type_rules": "citation > penalty > procedure > entitlement > other; lexical diagnostic only",
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    DEST.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "candidate_oracle": report["candidate_oracle"], "headroom": report["headroom"],
        "classes": report["miss_decomposition"]["exclusive_failure_class_recall_mass"],
        "multi_gold": {k: report["multi_gold"][k] for k in (
            "queries", "candidate_oracle_recall", "baseline_recall", "expert_top5_union_recall",
            "queries_with_reachable_missed_gold")},
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
