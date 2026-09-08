"""Build the Phase-0 world model on Huy's exact 600-query CV protocol.

This is intentionally cache-only. It reconstructs the shipped feature contract,
produces pooled four-block LOBO predictions, decomposes the candidate/ranking
headroom, and audits the lexical evidence selector without loading model weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark_burst_v4_full_sqlite import tokens  # noqa: E402
from benchmark_jina_reranker_holdouts import SPACE_RE, STOPWORDS  # noqa: E402
from run_burst_expanded_fusion_submission import DocumentStore  # noqa: E402
from tune_citation_graph import build_citation_table, citation_features  # noqa: E402
from tune_corpus_cap32_fusion import build_training_cap  # noqa: E402
from tune_doctype_features import build_type_table, type_features  # noqa: E402
from tune_expanded_fusion_selection import ltr_features  # noqa: E402


OUT = ROOT / "results/sol_high_rl"
CACHE = ROOT / "cache/sol_high_rl"
ARTICLE_RE = re.compile(r"\b(?:Điều|DIEU|Dieu|điều|dieu)\s+\d+[a-zA-Z]?\b")


def load_pickle(rel: str):
    return pickle.loads((ROOT / rel).read_bytes())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def metrics(rankings, queries, ids):
    recall, precision, single, multi, per_query = [], [], [], [], {}
    for q in ids:
        gold = queries[q][1]
        top = rankings[q][:5]
        value = len(set(top) & gold) / len(gold)
        recall.append(value)
        precision.append(len(set(top) & gold) / 5)
        (single if len(gold) == 1 else multi).append(value)
        per_query[q] = value
    return {
        "recall_at_5": float(np.mean(recall)),
        "precision_at_5_fixed5": float(np.mean(precision)),
        "single_gold_recall_at_5": float(np.mean(single)) if single else None,
        "multi_gold_recall_at_5": float(np.mean(multi)) if multi else None,
        "queries": len(ids),
        "gold_occurrences": int(sum(len(queries[q][1]) for q in ids)),
    }, per_query


def top_passage_details(question: str, text: str, count: int = 2,
                        window: int = 220, overlap: int = 70):
    """Exact lexical selector logic, augmented with stable offsets and diagnostics."""
    words = SPACE_RE.findall(text or "")
    if len(words) <= window + 80:
        part = " ".join(words)
        return [{"start_word": 0, "end_word": len(words), "lexical_density": None,
                 "coverage_score": None, "part": part}]
    query_tokens = tokens(question)
    content = {t for t in query_tokens if len(t) >= 3 and t not in STOPWORDS}
    numbers = {t for t in query_tokens if any(c.isdigit() for c in t)}
    bigrams = {" ".join(query_tokens[i:i + 2]) for i in range(len(query_tokens) - 1)}
    scored = []
    step = window - overlap
    for start in range(0, len(words), step):
        end = min(start + window, len(words))
        part = " ".join(words[start:end])
        normalized = tokens(part)
        token_set = set(normalized)
        norm_text = " ".join(normalized)
        coverage = sum(1.0 + .20 * min(normalized.count(t), 3)
                       for t in content if t in token_set)
        numeric = 3.0 * sum(t in token_set for t in numbers)
        phrase = 1.8 * sum(p in norm_text for p in bigrams)
        raw = coverage + numeric + phrase
        density = raw / math.sqrt(max(len(normalized), 1))
        scored.append((density, raw, -start, start, end, part))
        if end == len(words):
            break
    scored.sort(reverse=True)
    chosen = []
    for density, raw, _, start, end, part in scored:
        if not any(x["part"] == part for x in chosen):
            chosen.append({"start_word": start, "end_word": end,
                           "lexical_density": float(density),
                           "coverage_score": float(raw), "part": part})
        if len(chosen) >= count:
            break
    return chosen


def enrich_passage_detail(detail):
    part = detail.pop("part")
    detail["contains_article_marker"] = bool(ARTICLE_RE.search(part))
    detail["sha256"] = hashlib.sha256(part.encode("utf-8")).hexdigest()
    detail["snippet"] = part[:240]
    return detail


def prepare_contract():
    queries, blocks, all_ids, candidates, views, scores = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)
    names = ["base", "expanded", "jina", "dense", "corpus"]

    vnlegal = load_pickle("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    views = dict(views)
    views["vnlegal_lal"] = {
        q: sorted(candidates[q], key=lambda d: (-vnlegal.get(q, {}).get(d, -1e9), d))
        for q in all_ids
    }
    names.append("vnlegal_lal")
    scores = dict(scores)
    scores["vnlegal_lal"] = vnlegal

    crossenc = load_pickle("results/crossenc_fullpool/cv_scores.pkl")["scores"]
    cross_floor = min(v for row in crossenc.values() for v in row.values())
    scores["crossenc"] = {
        q: {d: crossenc.get(q, {}).get(d, cross_floor) for d in candidates[q]}
        for q in all_ids
    }

    extra_specs = {
        "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
        "jina_ft": "results/from_drive/jina_ft_cv.pkl",
        "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
    }
    channel_floors = {"crossenc": float(cross_floor)}
    for name, rel in extra_specs.items():
        table = load_pickle(rel)
        floor = min(v for row in table.values() for v in row.values())
        channel_floors[name] = float(floor)
        scores[name] = {
            q: {d: table.get(q, {}).get(d, floor) for d in candidates[q]}
            for q in all_ids
        }

    docs = DocumentStore(sorted(
        (ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    type_table = build_type_table(ROOT, docs, all_ids, candidates)
    type_rows = type_features(candidates, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, candidates)
    cite_rows = citation_features(candidates, own, cited, all_ids)
    return queries, blocks, all_ids, candidates, views, scores, names, docs, [type_rows, cite_rows], channel_floors


def fit_rankings(queries, blocks, all_ids, candidates, views, scores, names, extras):
    rows, groups = ltr_features(views, names, candidates, all_ids, scores)
    for extra in extras:
        for q in all_ids:
            rows[q] = np.concatenate([rows[q], extra[q]], axis=1)

    def train_predict(train_ids, test_ids):
        x = np.vstack([rows[q] for q in train_ids])
        y = np.concatenate([[d in queries[q][1] for d in groups[q]]
                            for q in train_ids]).astype(np.int8)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=.15, class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=2026)
        model.fit(scaler.transform(x), y)
        result = {}
        for q in test_ids:
            score = model.predict_proba(scaler.transform(rows[q]))[:, 1]
            order = np.argsort(-score)
            result[q] = [groups[q][i] for i in order]
        return result

    lobo = {}
    for held in blocks:
        train = sum((blocks[name] for name in blocks if name != held), [])
        lobo.update(train_predict(train, blocks[held]))
    apparent = train_predict(all_ids, all_ids)
    return lobo, apparent


def source_audit(queries, blocks, all_ids, candidates, views, scores, baseline):
    corpus_saved = load_pickle("results/corpus_index/holdout_dense_rank_cap32.pkl")
    membership = {
        "base": {q: list(views["base"][q]) for q in all_ids},
        "expanded": {q: list(views["expanded"][q]) for q in all_ids},
        "corpus_top20": {q: list(corpus_saved["ranking"][q][:20]) for q in all_ids},
    }
    rankers = {name: {q: list(table[q]) for q in all_ids}
               for name, table in views.items()}
    for name, table in scores.items():
        rankers[f"score:{name}"] = {
            q: sorted(candidates[q], key=lambda d: (-table.get(q, {}).get(d, -1e9), d))
            for q in all_ids
        }

    missed = {(q, d) for q in all_ids for d in queries[q][1] if d not in baseline[q][:5]}
    top5_rescue_sets = {}
    result = {}
    for name, table in rankers.items():
        rescued = {(q, d) for q, d in missed if d in table[q][:5]}
        top5_rescue_sets[name] = rescued
        gold_ranks = []
        for q in all_ids:
            pos = {d: i + 1 for i, d in enumerate(table[q])}
            gold_ranks.extend(pos.get(d) for d in queries[q][1] if d in pos)
        result[name] = {
            "top5_recall": metrics(table, queries, all_ids)[0]["recall_at_5"],
            "missed_gold_occurrences_rescued_at_top5": len(rescued),
            "gold_occurrences_ranked": len(gold_ranks),
            "gold_rank_median": float(np.median(gold_ranks)) if gold_ranks else None,
            "gold_rank_p90": float(np.percentile(gold_ranks, 90)) if gold_ranks else None,
        }
    for name in result:
        others = set().union(*(v for n, v in top5_rescue_sets.items() if n != name))
        result[name]["unique_missed_gold_rescue_at_top5"] = len(top5_rescue_sets[name] - others)

    membership_report = {}
    occurrence_sets = {}
    for name, table in membership.items():
        occurrences = {(q, d) for q in all_ids for d in queries[q][1] if d in table[q]}
        occurrence_sets[name] = occurrences
    for name, occurrences in occurrence_sets.items():
        others = set().union(*(v for n, v in occurrence_sets.items() if n != name))
        membership_report[name] = {
            "gold_occurrences_present": len(occurrences),
            "unique_gold_membership_contribution": len(occurrences - others),
            "mean_documents": float(np.mean([len(membership[name][q]) for q in all_ids])),
            "per_block_gold_occurrences_present": {
                block: sum((q, d) in occurrences for q in ids for d in queries[q][1])
                for block, ids in blocks.items()
            },
        }
    return result, membership_report


def main():
    started = time.perf_counter()
    (OUT / "evidence_audit").mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (queries, blocks, all_ids, candidates, views, scores, names, docs, extras,
     channel_floors) = prepare_contract()
    baseline, apparent = fit_rankings(
        queries, blocks, all_ids, candidates, views, scores, names, extras)
    baseline_metrics, baseline_per_query = metrics(baseline, queries, all_ids)
    apparent_metrics, _ = metrics(apparent, queries, all_ids)
    per_block = {name: metrics(baseline, queries, ids)[0] for name, ids in blocks.items()}

    oracle_values = {}
    candidate_ceiling_by_block = {}
    for block, ids in blocks.items():
        vals = [len(set(candidates[q]) & queries[q][1]) / len(queries[q][1]) for q in ids]
        candidate_ceiling_by_block[block] = float(np.mean(vals))
        oracle_values.update({q: value for q, value in zip(ids, vals)})
    candidate_ceiling = float(np.mean([oracle_values[q] for q in all_ids]))

    rank_bins = Counter()
    out_of_pool = []
    weighted = Counter()
    in_pool_missed = []
    for q in all_ids:
        position = {d: i + 1 for i, d in enumerate(baseline[q])}
        for d in queries[q][1]:
            weight = 1.0 / len(queries[q][1])
            rank = position.get(d)
            if rank is None:
                rank_bins["out_of_pool"] += 1
                weighted["out_of_pool"] += weight
                out_of_pool.append({"qid": q, "doc_id": d, "block": next(k for k, v in blocks.items() if q in v)})
            elif rank <= 5:
                rank_bins["top5"] += 1
                weighted["top5"] += weight
            else:
                if rank <= 10:
                    key = "r6_10"
                elif rank <= 20:
                    key = "r11_20"
                else:
                    key = "gt20"
                rank_bins[key] += 1
                weighted[key] += weight
                in_pool_missed.append((q, d, rank, weight))

    boundary = Counter()
    boundary_by_block = {name: Counter() for name in blocks}
    block_of = {q: name for name, ids in blocks.items() for q in ids}
    for q in all_ids:
        gold = queries[q][1]
        r5, r6 = baseline[q][4], baseline[q][5]
        a, b = r5 in gold, r6 in gold
        key = ("r5_gold_r6_not" if a and not b else
               "r5_not_r6_gold" if b and not a else
               "both_gold" if a and b else "neither_gold")
        boundary[key] += 1
        boundary_by_block[block_of[q]][key] += 1
    decisive = boundary["r5_gold_r6_not"] + boundary["r5_not_r6_gold"]

    evidence_rows = []
    evidence_summary = Counter()
    suspect_weight = 0.0
    for q, d, rank, weight in in_pool_missed:
        text = docs[d]
        word_count = len(SPACE_RE.findall(text or ""))
        selections = [enrich_passage_detail(x) for x in top_passage_details(queries[q][0], text)]
        long_doc = word_count > 300
        substantive = sum(x["contains_article_marker"] for x in selections)
        header_only = all(x["start_word"] < 300 for x in selections)
        suspected = bool(long_doc and substantive == 0)
        evidence_summary["missed_gold_documents"] += 1
        evidence_summary["long_documents"] += int(long_doc)
        evidence_summary["at_least_one_article_marker"] += int(substantive > 0)
        evidence_summary["header_region_only"] += int(header_only)
        evidence_summary["suspected_bad_evidence"] += int(suspected)
        suspect_weight += weight if suspected else 0.0
        evidence_rows.append({
            "qid": q, "doc_id": d, "block": block_of[q], "baseline_rank": rank,
            "gold_count": len(queries[q][1]), "document_words": word_count,
            "long_document": long_doc, "header_region_only": header_only,
            "suspected_bad_evidence": suspected, "selected_passages": selections,
            "passage_score_stability": "unavailable: shipped caches retain only max document score"
        })
    evidence_path = OUT / "evidence_audit/missed_gold_lexical_passages.jsonl"
    temp = evidence_path.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evidence_rows),
                    encoding="utf-8")
    temp.replace(evidence_path)

    source_report, membership_report = source_audit(
        queries, blocks, all_ids, candidates, views, scores, baseline)
    in_pool_weight = weighted["r6_10"] + weighted["r11_20"] + weighted["gt20"]
    n_queries = len(all_ids)
    ranking_residual_weight = max(0.0, in_pool_weight - suspect_weight)
    world = {
        "status": "PHASE0_COMPLETE",
        "protocol": {
            "queries": len(all_ids), "blocks": {k: len(v) for k, v in blocks.items()},
            "pooling": "pooled query-level macro Recall@5 across unequal blocks",
            "baseline": "cache-only four-block LOBO reconstruction of burst_userft_maxrecall feature contract",
            "ltr": "LogisticRegression(C=0.15,class_weight=balanced,liblinear,seed=2026)",
            "rank_views": names,
            "score_channels": sorted(scores),
            "channel_floors": channel_floors,
        },
        "baseline_recall": baseline_metrics["recall_at_5"],
        "baseline_metrics": baseline_metrics,
        "apparent_full_fit_cv_metrics": apparent_metrics,
        "per_block": per_block,
        "cv_block_variance": float(np.var([x["recall_at_5"] for x in per_block.values()])),
        "candidate_ceiling": candidate_ceiling,
        "candidate_ceiling_per_block": candidate_ceiling_by_block,
        "candidate_pool_size": {
            "min": min(len(candidates[q]) for q in all_ids),
            "mean": float(np.mean([len(candidates[q]) for q in all_ids])),
            "max": max(len(candidates[q]) for q in all_ids),
        },
        "gold_occurrence_decomposition": dict(rank_bins),
        "weighted_recall_mass_decomposition": {k: v / n_queries for k, v in weighted.items()},
        "out_of_pool_gold": len(out_of_pool),
        "misses_r6_10": rank_bins["r6_10"],
        "misses_r11_20": rank_bins["r11_20"],
        "misses_gt20": rank_bins["gt20"],
        "suspected_bad_evidence": int(evidence_summary["suspected_bad_evidence"]),
        "evidence_quality": {
            **dict(evidence_summary),
            "heuristic": "suspect iff missed in-pool gold is >300 words and neither selected lexical window contains an explicit Article marker",
            "detailed_artifact": str(evidence_path.relative_to(ROOT)),
            "per_passage_score_stability": "not measurable from shipped max-only caches",
        },
        "boundary_defender_base_rate": {
            **dict(boundary),
            "decisive_cases": decisive,
            "r5_defender_accuracy_when_decisive": (boundary["r5_gold_r6_not"] / decisive if decisive else None),
            "per_block": {k: dict(v) for k, v in boundary_by_block.items()},
        },
        "source_rank_and_rescue": source_report,
        "candidate_membership_contribution": membership_report,
        "unique_rescue_by_source": {k: v["unique_missed_gold_rescue_at_top5"] for k, v in source_report.items()},
        "headroom_estimates": {
            "H_candidate_missing_pool_recall_mass": 1.0 - candidate_ceiling,
            "H_evidence_suspected_recoverable_recall_mass": suspect_weight / n_queries,
            "H_ranking_residual_recall_mass": ranking_residual_weight / n_queries,
            "H_total_in_pool": candidate_ceiling - baseline_metrics["recall_at_5"],
            "warning": "H_evidence vs H_ranking split is a lexical-selector heuristic; their sum equals in-pool headroom but only H_candidate and H_total_in_pool are oracle-exact."
        },
        "known_public_transfer_failures": [
            "vnlegal-lal last-token pooling improved local F2 but reduced real leaderboard Recall; Huy reverted to CLS cache",
            "raw query_cites yielded about +0.0014 local Recall but reduced leaderboard performance",
            "README warns N=600 CV differences around +/-0.005 have repeatedly failed to transfer"
        ],
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "oof_predictions": "results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json",
            "out_of_pool_gold": "results/sol_high_rl/OUT_OF_POOL_GOLD.json",
            "evidence_details": str(evidence_path.relative_to(ROOT)),
        },
    }
    atomic_json(OUT / "BASELINE_LOBO_PREDICTIONS.json", baseline)
    atomic_json(OUT / "OUT_OF_POOL_GOLD.json", out_of_pool)
    atomic_json(OUT / "WORLD_MODEL.json", world)
    print(json.dumps(world, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
