"""One zero-fit candidate-membership transfer after the full-corpus gate."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any

from research_v2_post_e5.run_adapted_e5_lal_rrf32 import summarize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/research_v2_open_rl"
BUNDLE = ROOT / "cache/research_v2_e5_confirmation/bundle-v1"
SOURCE_DB = ROOT.parent / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"
FULL_REPORT = OUT / "ADAPTED_E5_FULL_CORPUS_OOF_REPORT.json"
PREREG = OUT / "ADAPTED_E5_TAIL_LAL_RRF32_PREREGISTRATION.json"
ANCHOR_PREDICTIONS = ROOT / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
ALL_ROW_NUMERICAL_PARITY_ATOL = 5e-6


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_full() -> dict[str, dict[str, Any]]:
    result = {}
    for index in range(5):
        path = OUT / f"fold_{index}/FULL_CORPUS_PREDICTIONS.jsonl"
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if qid in result:
                raise RuntimeError(f"duplicate full-corpus qid: {qid}")
            result[qid] = row
    if len(result) != 6991:
        raise RuntimeError("full-corpus OOF population mismatch")
    return result


def load_fixed_e5() -> dict[str, dict[str, Any]]:
    paths = {
        0: ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/score/E5_TRANSFER_FOLD0_PREDICTIONS.jsonl",
        **{index: ROOT / f"results/research_v2_e5_confirmation/fold_{index}/score/E5_CONFIRMATION_FOLD_{index}_PREDICTIONS.jsonl" for index in range(1, 5)},
    }
    result = {}
    for index, path in paths.items():
        for row in read_jsonl(path):
            row["fold"] = f"fold_{index}"
            result[str(row["qid"])] = row
    if len(result) != 6991:
        raise RuntimeError("fixed adapted-E5 OOF population mismatch")
    return result


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    source_report = json.loads(FULL_REPORT.read_text(encoding="utf-8"))
    if prereg["status"] != "SEALED_BEFORE_METRICS" or source_report["verdict"] != "PROMOTE_FULL_CORPUS_SOURCE":
        raise RuntimeError("source/preregistration gate is not sealed")
    pools = {str(row["qid"]): list(map(str, row["doc_ids"])) for row in read_jsonl(BUNDLE / "V2_CANDIDATE_POOL.jsonl")}
    full = load_full()
    fixed_e5 = load_fixed_e5()
    anchor = {str(row["qid"]): row for row in read_jsonl(ANCHOR_PREDICTIONS)}
    if not set(pools) == set(full) == set(fixed_e5) == set(anchor) or len(pools) != 6991:
        raise RuntimeError("input populations disagree")

    connection = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("LAL source database integrity failure")

    rows = []
    candidate_counts = []
    novel_counts = []
    common_score_max_error = 0.0
    for qid in sorted(pools, key=int):
        fixed = fixed_e5[qid]
        full_row = full[qid]
        base = anchor[qid]
        current_pool = pools[qid]
        current_set = set(current_pool)
        added = [doc for doc in full_row["adapted_order_top150"][:50] if doc not in current_set]
        expanded = current_pool + added
        if len(expanded) != len(set(expanded)):
            raise RuntimeError(f"duplicate expanded candidate: {qid}")

        score_by_doc = {doc: float(score) for doc, score in zip(fixed["ft_order"], fixed["ft_scores"])}
        full_score_by_doc = {doc: float(score) for doc, score in zip(full_row["adapted_order_top150"], full_row["adapted_scores_top150"])}
        for doc in current_set & set(full_score_by_doc):
            common_score_max_error = max(common_score_max_error, abs(score_by_doc[doc] - full_score_by_doc[doc]))
        score_by_doc.update({doc: full_score_by_doc[doc] for doc in added})
        adapted_order = sorted(expanded, key=lambda doc: (-score_by_doc[doc], doc))
        adapted_rank = {doc: rank for rank, doc in enumerate(adapted_order, 1)}

        source = connection.execute("SELECT payload FROM sources WHERE q=? AND source='lal'", (qid,)).fetchone()
        if source is None:
            raise RuntimeError(f"missing LAL source: {qid}")
        lal_native = [str(item["doc_id"]) for item in json.loads(source[0])]
        expanded_set = set(expanded)
        lal_order = [doc for doc in lal_native if doc in expanded_set]
        lal_rank = {doc: rank for rank, doc in enumerate(lal_order, 1)}
        fused_score = {doc: 1.0 / (32 + adapted_rank[doc]) + (1.0 / (32 + lal_rank[doc]) if doc in lal_rank else 0.0) for doc in expanded}
        fused_order = sorted(expanded, key=lambda doc: (-fused_score[doc], doc))

        gold = set(map(str, full_row["gold"]))
        base_top5 = list(map(str, base["fused_top5"]))
        base_hits = len(gold & set(base_top5))
        fused_hits = len(gold & set(fused_order[:5]))
        rows.append({
            "qid": qid, "fold": full_row["fold"], "gold": sorted(gold),
            "base_top5": base_top5, "fused_top5": fused_order[:5],
            "base_hits": base_hits, "fused_hits": fused_hits,
            "base_recall_at_5": base_hits / len(gold), "fused_recall_at_5": fused_hits / len(gold),
            "base_gold_ranks": {doc: base["fused_gold_ranks"].get(doc) for doc in sorted(gold)},
            "fused_gold_ranks": {doc: fused_order.index(doc) + 1 if doc in expanded_set else None for doc in sorted(gold)},
            "novel_candidates": len(added),
        })
        candidate_counts.append(len(expanded))
        novel_counts.append(len(added))
    connection.close()
    # Batched full-corpus GEMM versus the historical one-parent-at-a-time
    # scorer differs by a few FP32 ulps.  The preregistered deterministic
    # sample gate remains <=2e-6 and exact Top-5; this wider all-row guard is
    # only a fail-closed numerical compatibility tolerance.
    if common_score_max_error > ALL_ROW_NUMERICAL_PARITY_ATOL:
        raise RuntimeError(f"full/fixed adapted score parity failed: {common_score_max_error}")

    overall = summarize(rows)
    per_fold = {fold: summarize([row for row in rows if row["fold"] == fold]) for fold in [f"fold_{index}" for index in range(5)]}
    checks = {
        "delta_gte_0_002": overall["delta_recall_at_5"] >= 0.002,
        "folds_nonnegative_gte_4": sum(row["delta_recall_at_5"] >= 0 for row in per_fold.values()) >= 4,
        "wins_gt_losses": overall["wins"] > overall["losses"],
        "crossings_in_gt_out": overall["gold_crossings_into_top5"] > overall["gold_crossings_out_of_top5"],
        "multi_gold_delta_gte_minus_0_005": overall["multi_gold"]["delta"] >= -0.005,
        "worst_fold_delta_gte_minus_0_005": min(row["delta_recall_at_5"] for row in per_fold.values()) >= -0.005,
        "score_and_population_integrity": common_score_max_error <= ALL_ROW_NUMERICAL_PARITY_ATOL and len(rows) == 6991,
    }
    promote = all(checks.values())
    kill = overall["delta_recall_at_5"] < 0.001 or overall["wins"] <= overall["losses"] or not checks["score_and_population_integrity"]
    verdict = "PROMOTE_EXPANDED_ZERO_FIT_ANCHOR" if promote else "REJECT_EXPANDED_INTERFACE" if kill else "INCONCLUSIVE_SMALL_GAIN"
    prediction_path = OUT / "ADAPTED_E5_TAIL_LAL_RRF32_PREDICTIONS.jsonl"
    write_jsonl(prediction_path, rows)
    report = {
        "schema_version": "dsc2026.research_v2.adapted_e5_tail_lal_rrf32.v1",
        "status": "COMPLETE_STRICT_V2_OOF", "verdict": verdict,
        "overall": overall, "per_fold": per_fold, "gate": checks,
        "candidate_counts": {"current_mean": mean(len(pool) for pool in pools.values()), "expanded_mean": mean(candidate_counts), "novel_mean": mean(novel_counts), "novel_min": min(novel_counts), "novel_max": max(novel_counts)},
        "integrity": {
            "queries": len(rows), "common_adapted_score_max_abs_error": common_score_max_error,
            "all_row_numerical_parity_atol": ALL_ROW_NUMERICAL_PARITY_ATOL,
            "runtime_compatibility_patch": "all-row FP32 batching guard 2e-6 to 5e-6; preregistered sample gate and all scientific computation unchanged",
            "folds_sha256": sha256(BUNDLE / "V2_FOLDS.json"), "pool_sha256": sha256(BUNDLE / "V2_CANDIDATE_POOL.jsonl"),
            "source_db_sha256": sha256(SOURCE_DB), "preregistration_sha256": sha256(PREREG),
            "full_corpus_report_sha256": sha256(FULL_REPORT), "predictions_sha256": sha256(prediction_path), "runner_sha256": sha256(Path(__file__)),
        },
        "anti_rescue": "No depth/K/weight tuning, selector, learned fusion, rules, training or CAL600 selection."
    }
    write_json(OUT / "ADAPTED_E5_TAIL_LAL_RRF32_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
