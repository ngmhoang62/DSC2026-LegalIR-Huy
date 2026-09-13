"""Strict-V2 failure anatomy for the current adapted-E5 + frozen-LAL RRF32 anchor."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "research_v2_open_rl"
BUNDLE = ROOT / "cache" / "research_v2_e5_confirmation" / "bundle-v1"
E5 = ROOT / "results" / "research_v2_e5_confirmation"
E5_F0 = ROOT / "results" / "research_v2_e5_transfer" / "research_v2_e5_transfer_fold0"
ANCHOR = ROOT / "results" / "research_v2_post_e5" / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
SOURCE_DB = ROOT.parent / "LegalIR" / "cache" / "exp112_task_adaptive_retrieval" / "sources.sqlite"
CONTEXTS = ROOT / "cache" / "research_v2_forensic" / "kaggle_input" / "research-v2-jina-boundary-v4" / "V2_CONTEXTS.jsonl"


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket(rank: int | None) -> str:
    if rank is None:
        return "out_of_pool"
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return ">50"


def query_tags(text: str) -> list[str]:
    q = text.lower()
    patterns = {
        "explicit_citation": r"\b(điều|khoản|điểm)\s+\d+|\b\d{1,4}/\d{4}/(?:nđ|tt|qđ|qh|ubtvqh)",
        "penalty": r"xử phạt|mức phạt|truy cứu|tội |hình phạt",
        "procedure": r"thủ tục|trình tự|hồ sơ|cấp giấy|đăng ký",
        "amount_or_duration": r"mức |bao nhiêu|thời hạn|thời gian|tỷ lệ|phần trăm|%",
        "authority_or_duty": r"thẩm quyền|do ai|cơ quan nào|nhiệm vụ|trách nhiệm|quyền hạn",
        "eligibility": r"điều kiện|đối tượng|trường hợp nào|có được|được phép",
        "form_or_template": r"mẫu |phụ lục|biểu mẫu",
        "definition": r"là gì|được hiểu|khái niệm|như thế nào theo quy định",
    }
    tags = [name for name, pattern in patterns.items() if re.search(pattern, q)]
    return tags or ["other"]


def doc_type(text: str) -> str:
    d = re.sub(r"\s+", " ", text.lower()[:5000])
    for name, pattern in (
        ("decree", r"\bnghị định\b"),
        ("circular", r"\bthông tư\b"),
        ("decision", r"\bquyết định\b"),
        ("law_or_code", r"\b(?:bộ luật|luật)\b"),
        ("resolution", r"\bnghị quyết\b"),
        ("standard", r"\b(?:qcvn|tcvn|quy chuẩn|tiêu chuẩn)\b"),
        ("guidance", r"\b(?:hướng dẫn|quy định)\b"),
    ):
        if re.search(pattern, d):
            return name
    return "unknown"


def summarize_counts(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "gt20": sum(value > 20 for value in values),
        "gt50": sum(value > 50 for value in values),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor = {str(row["qid"]): row for row in rows(ANCHOR)}
    pools = {str(row["qid"]): list(map(str, row["doc_ids"])) for row in rows(BUNDLE / "V2_CANDIDATE_POOL.jsonl")}
    queries = {str(row["qid"]): row for row in rows(BUNDLE / "V2_TRANSFER_QUERIES.jsonl")}
    contexts = {str(row["doc_id"]): str(row["passage"]) for row in rows(CONTEXTS)}
    chunk_counts = Counter()
    for row in rows(BUNDLE / "chunk_ids.jsonl"):
        chunk_counts[str(row["doc_id"])] += 1

    e5_paths = {
        0: E5_F0 / "score" / "E5_TRANSFER_FOLD0_PREDICTIONS.jsonl",
        **{fold: E5 / f"fold_{fold}" / "score" / f"E5_CONFIRMATION_FOLD_{fold}_PREDICTIONS.jsonl" for fold in range(1, 5)},
    }
    e5 = {}
    for fold, path in e5_paths.items():
        for row in rows(path):
            row["fold"] = f"fold_{fold}"
            e5[str(row["qid"])] = row
    if set(anchor) != set(pools) or set(anchor) != set(queries) or set(anchor) != set(e5):
        raise RuntimeError("Strict-V2 population mismatch")

    connection = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("source DB integrity failure")

    occurrence_buckets: Counter[str] = Counter()
    query_failures: Counter[str] = Counter()
    missed_source_nearest: Counter[str] = Counter()
    missed_tags: Counter[str] = Counter()
    all_tags: Counter[str] = Counter()
    missed_doc_types: Counter[str] = Counter()
    all_gold_doc_types: Counter[str] = Counter()
    fold_failures: Counter[str] = Counter()
    fold_queries: Counter[str] = Counter()
    missed_chunks: list[int] = []
    hit_chunks: list[int] = []
    all_gold_chunks: list[int] = []
    score_margins = []
    failure_rows = []
    query_macro = []
    candidate_macro = []
    fusion_harms = fusion_rescues = 0
    expert_known_missed_occurrences = consensus_blind_occurrences = 0
    total_gold_occurrences = missed_occurrences = 0

    for qid in sorted(anchor, key=int):
        a = anchor[qid]
        erow = e5[qid]
        gold = set(map(str, a["gold"]))
        top5 = set(map(str, a["fused_top5"]))
        e5_top5 = set(map(str, erow["ft_order"][:5]))
        fold = str(a["fold"])
        fold_queries[fold] += 1
        qrec = len(gold & top5) / len(gold)
        crec = len(gold & set(pools[qid])) / len(gold)
        query_macro.append(qrec)
        candidate_macro.append(crec)
        if qrec < 1:
            query_failures["queries_with_any_miss"] += 1
            fold_failures[fold] += 1
        if qrec == 0:
            query_failures["zero_hit_queries"] += 1
        if len(gold) > 1 and qrec < 1:
            query_failures["multi_gold_queries_with_any_miss"] += 1

        if len(gold & e5_top5) > len(gold & top5):
            fusion_harms += 1
        if len(gold & top5) > len(gold & e5_top5):
            fusion_rescues += 1

        source_rows = connection.execute("SELECT source,payload FROM sources WHERE q=?", (qid,)).fetchall()
        source_orders = {source: [str(item["doc_id"]) for item in json.loads(payload)] for source, payload in source_rows}
        source_ranks = {source: {doc: idx for idx, doc in enumerate(order, 1)} for source, order in source_orders.items()}
        adapted_rank = {doc: idx for idx, doc in enumerate(map(str, erow["ft_order"]), 1)}
        adapted_score = {doc: float(score) for doc, score in zip(map(str, erow["ft_order"]), erow["ft_scores"])}
        boundary_score = float(erow["ft_scores"][4])

        tags = query_tags(str(queries[qid]["question"]))
        for tag in tags:
            all_tags[tag] += 1

        for doc in gold:
            total_gold_occurrences += 1
            count = chunk_counts.get(doc, 0)
            all_gold_chunks.append(count)
            all_gold_doc_types[doc_type(contexts.get(doc, ""))] += 1
            if doc in top5:
                hit_chunks.append(count)
                occurrence_buckets["hit_top5"] += 1
                continue

            missed_occurrences += 1
            rank = a["fused_gold_ranks"].get(doc)
            occurrence_buckets[bucket(rank)] += 1
            missed_chunks.append(count)
            dtype = doc_type(contexts.get(doc, ""))
            missed_doc_types[dtype] += 1
            for tag in tags:
                missed_tags[tag] += 1

            ranks = {source: mapping.get(doc) for source, mapping in source_ranks.items()}
            top5_sources = sorted(source for source, value in ranks.items() if value is not None and value <= 5)
            if doc in e5_top5:
                top5_sources.append("adapted_e5")
            if top5_sources:
                expert_known_missed_occurrences += 1
                for source in set(top5_sources):
                    missed_source_nearest[f"top5:{source}"] += 1
            else:
                consensus_blind_occurrences += 1
                best_source, best_rank = min(
                    ((source, value) for source, value in ranks.items() if value is not None),
                    key=lambda item: item[1], default=("none", None),
                )
                missed_source_nearest[f"nearest:{best_source}:{bucket(best_rank)}"] += 1

            e5_rank = adapted_rank.get(doc)
            margin = None if e5_rank is None else adapted_score[doc] - boundary_score
            if margin is not None:
                score_margins.append(margin)
            failure_rows.append({
                "qid": qid,
                "fold": fold,
                "gold_doc": doc,
                "gold_count": len(gold),
                "query": queries[qid]["question"],
                "query_tags": tags,
                "anchor_rank": rank,
                "adapted_e5_rank": e5_rank,
                "adapted_e5_margin_to_rank5": margin,
                "source_ranks": ranks,
                "known_by_top5_experts": sorted(set(top5_sources)),
                "candidate_member": doc in set(pools[qid]),
                "document_chunks": count,
                "document_characters": len(contexts.get(doc, "")),
                "document_type": dtype,
            })

    connection.close()

    def rate_table(numerators: Counter[str], denominators: Counter[str]) -> dict[str, Any]:
        return {
            key: {
                "failures": numerators[key],
                "queries": denominators[key],
                "rate": numerators[key] / denominators[key],
            }
            for key in sorted(denominators)
        }

    report = {
        "schema_version": "dsc2026.research_v2.anchor_failure_anatomy.v1",
        "status": "COMPLETE_CACHE_ONLY",
        "population": {"queries": len(anchor), "gold_occurrences": total_gold_occurrences},
        "metrics": {
            "anchor_recall_at_5": mean(query_macro),
            "candidate_ceiling": mean(candidate_macro),
            "queries_with_any_miss": query_failures["queries_with_any_miss"],
            "zero_hit_queries": query_failures["zero_hit_queries"],
            "missed_gold_occurrences": missed_occurrences,
        },
        "missed_occurrence_rank_buckets": dict(occurrence_buckets),
        "failure_classes": {
            "expert_top5_known_but_anchor_missed_occurrences": expert_known_missed_occurrences,
            "consensus_blind_occurrences": consensus_blind_occurrences,
            "adapted_e5_to_rrf_query_harms": fusion_harms,
            "adapted_e5_to_rrf_query_rescues": fusion_rescues,
        },
        "failure_by_fold": rate_table(fold_failures, fold_queries),
        "query_tag_failure_enrichment": {
            tag: {
                "all_queries": all_tags[tag],
                "missed_gold_occurrences": missed_tags[tag],
                "misses_per_tagged_query": missed_tags[tag] / all_tags[tag],
            }
            for tag in sorted(all_tags)
        },
        "document_length_chunks": {
            "all_gold": summarize_counts(all_gold_chunks),
            "hit_gold": summarize_counts(hit_chunks),
            "missed_gold": summarize_counts(missed_chunks),
        },
        "document_type": {
            "all_gold_occurrences": dict(all_gold_doc_types),
            "missed_gold_occurrences": dict(missed_doc_types),
        },
        "adapted_e5_missed_gold_margin_to_rank5": {
            "n": len(score_margins),
            "mean": mean(score_margins) if score_margins else None,
            "median": median(score_margins) if score_margins else None,
            "within_0_01_below": sum(-0.01 <= value < 0 for value in score_margins),
            "within_0_03_below": sum(-0.03 <= value < 0 for value in score_margins),
        },
        "source_signals_on_missed_gold": dict(missed_source_nearest),
        "interpretation_guard": "Patterns are descriptive. They may justify one preregistered causal falsifier but may not be converted into query/doc rules.",
        "failures": failure_rows,
    }
    json_path = OUT / "ANCHOR_FAILURE_ANATOMY.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    chunk_stats = report["document_length_chunks"]
    md = f"""# Research V2 anchor failure anatomy

## Fixed state

- Queries: `{len(anchor)}`; gold occurrences: `{total_gold_occurrences}`.
- Current adapted-E5 + frozen-LAL RRF32 Recall@5: `{mean(query_macro):.9f}`.
- Immutable-pool oracle: `{mean(candidate_macro):.9f}`.
- Queries with at least one missed gold: `{query_failures['queries_with_any_miss']}`; zero-hit: `{query_failures['zero_hit_queries']}`.

## Error classes

- Missed gold occurrences: `{missed_occurrences}`.
- At least one existing expert places the missed gold in its Top-5: `{expert_known_missed_occurrences}`.
- No current expert places it in Top-5: `{consensus_blind_occurrences}`.
- Adding frozen LAL to adapted-E5 rescues `{fusion_rescues}` queries but harms `{fusion_harms}`.

## Parent-length signal

- All gold parents: median `{chunk_stats['all_gold']['median']}` chunks; p90 `{chunk_stats['all_gold']['p90']}`.
- Hit gold parents: median `{chunk_stats['hit_gold']['median']}` chunks; p90 `{chunk_stats['hit_gold']['p90']}`.
- Missed gold parents: median `{chunk_stats['missed_gold']['median']}` chunks; p90 `{chunk_stats['missed_gold']['p90']}`; `>50` chunks `{chunk_stats['missed_gold']['gt50']}`.

## Guard

This is descriptive anatomy, not a source of manual rules. Any next action must alter one causal mechanism globally and be preregistered before metric inspection.
"""
    (OUT / "ANCHOR_FAILURE_ANATOMY.md").write_text(md, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "metrics", "missed_occurrence_rank_buckets", "failure_classes", "document_length_chunks", "adapted_e5_missed_gold_margin_to_rank5")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
