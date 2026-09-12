"""Execute the preregistered label-free adapted-E5 + LAL RRF32 falsifier."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/research_v2_post_e5"
CONFIRM = ROOT / "results/research_v2_e5_confirmation"
FOLD0 = ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0"
BUNDLE = ROOT / "cache/research_v2_e5_confirmation/bundle-v1"
SOURCE_DB = ROOT.parent / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"
PREREG = OUT / "NEXT_V2_HYPOTHESIS_PREREGISTRATION.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51+"


def load_reference() -> dict[str, dict[str, Any]]:
    paths = {
        0: FOLD0 / "score/E5_TRANSFER_FOLD0_PREDICTIONS.jsonl",
        **{
            index: CONFIRM / f"fold_{index}/score/E5_CONFIRMATION_FOLD_{index}_PREDICTIONS.jsonl"
            for index in range(1, 5)
        },
    }
    result = {}
    for index, path in paths.items():
        for row in read_jsonl(path):
            qid = str(row["qid"])
            row["fold"] = f"fold_{index}"
            if qid in result:
                raise RuntimeError(f"duplicate qid {qid}")
            result[qid] = row
    if len(result) != 6991:
        raise RuntimeError(f"reference count {len(result)} != 6991")
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = [row["base_recall_at_5"] for row in rows]
    fused = [row["fused_recall_at_5"] for row in rows]
    single = [row for row in rows if len(row["gold"]) == 1]
    multi = [row for row in rows if len(row["gold"]) > 1]
    transitions: Counter[str] = Counter()
    into = out = 0
    for row in rows:
        for gold, base_rank in row["base_gold_ranks"].items():
            fused_rank = row["fused_gold_ranks"][gold]
            before, after = bucket(base_rank), bucket(fused_rank)
            transitions[f"{before}->{after}"] += 1
            into += before != "1-5" and after == "1-5"
            out += before == "1-5" and after != "1-5"

    def sliced(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(values),
            "base": mean(row["base_recall_at_5"] for row in values),
            "fused": mean(row["fused_recall_at_5"] for row in values),
            "delta": mean(row["fused_recall_at_5"] - row["base_recall_at_5"] for row in values),
            "wins": sum(row["fused_recall_at_5"] > row["base_recall_at_5"] for row in values),
            "losses": sum(row["fused_recall_at_5"] < row["base_recall_at_5"] for row in values),
        }

    return {
        "queries": len(rows),
        "base_recall_at_5": mean(base),
        "fused_recall_at_5": mean(fused),
        "delta_recall_at_5": mean(f - b for b, f in zip(base, fused)),
        "base_precision_at_5": mean(row["base_hits"] / 5 for row in rows),
        "fused_precision_at_5": mean(row["fused_hits"] / 5 for row in rows),
        "wins": sum(f > b for b, f in zip(base, fused)),
        "losses": sum(f < b for b, f in zip(base, fused)),
        "ties": sum(f == b for b, f in zip(base, fused)),
        "changed_top5_sets": sum(set(row["base_top5"]) != set(row["fused_top5"]) for row in rows),
        "single_gold": sliced(single),
        "multi_gold": sliced(multi),
        "gold_crossings_into_top5": into,
        "gold_crossings_out_of_top5": out,
        "gold_rank_bucket_movements": dict(sorted(transitions.items())),
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "PREREGISTERED_BEFORE_RESULT":
        raise RuntimeError("preregistration is not sealed")
    reference = load_reference()
    pools = {
        str(row["qid"]): list(map(str, row["doc_ids"]))
        for row in read_jsonl(BUNDLE / "V2_CANDIDATE_POOL.jsonl")
    }
    if set(reference) != set(pools):
        raise RuntimeError("pool/reference population mismatch")

    connection = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("source database integrity failure")

    rows = []
    signatures: Counter[str] = Counter()
    for qid in sorted(reference, key=int):
        row = reference[qid]
        pool = pools[qid]
        if set(row["ft_order"]) != set(pool):
            raise RuntimeError(f"adapted candidate mismatch: {qid}")
        source = connection.execute(
            "SELECT payload,signature FROM sources WHERE q=? AND source='lal'", (qid,)
        ).fetchone()
        if source is None:
            raise RuntimeError(f"missing LAL source: {qid}")
        payload, signature = source
        signatures[signature] += 1
        native_order = [str(item["doc_id"]) for item in json.loads(payload)]
        pool_set = set(pool)
        lal_order = [doc for doc in native_order if doc in pool_set]
        lal_rank = {doc: index for index, doc in enumerate(lal_order, 1)}
        adapted_order = list(map(str, row["ft_order"]))
        adapted_rank = {doc: index for index, doc in enumerate(adapted_order, 1)}
        scores = {
            doc: 1.0 / (32 + adapted_rank[doc])
            + (1.0 / (32 + lal_rank[doc]) if doc in lal_rank else 0.0)
            for doc in pool
        }
        fused_order = sorted(pool, key=lambda doc: (-scores[doc], doc))
        if set(fused_order) != pool_set or len(fused_order) != len(pool):
            raise RuntimeError(f"fusion membership failure: {qid}")
        gold = set(map(str, row["gold"]))
        base_hits = len(gold & set(adapted_order[:5]))
        fused_hits = len(gold & set(fused_order[:5]))
        rows.append({
            "qid": qid,
            "fold": row["fold"],
            "gold": sorted(gold),
            "base_top5": adapted_order[:5],
            "fused_top5": fused_order[:5],
            "base_hits": base_hits,
            "fused_hits": fused_hits,
            "base_recall_at_5": base_hits / len(gold),
            "fused_recall_at_5": fused_hits / len(gold),
            "base_gold_ranks": {doc: adapted_rank.get(doc) for doc in sorted(gold)},
            "fused_gold_ranks": {
                doc: fused_order.index(doc) + 1 if doc in pool_set else None for doc in sorted(gold)
            },
        })
    connection.close()

    prediction_path = OUT / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    overall = summarize(rows)
    per_fold = {
        fold: summarize([row for row in rows if row["fold"] == fold])
        for fold in [f"fold_{index}" for index in range(5)]
    }
    checks = {
        "pooled_delta_recall_at_5_gte_0_005": overall["delta_recall_at_5"] >= 0.005,
        "at_least_4_of_5_folds_nonnegative": sum(value["delta_recall_at_5"] >= 0 for value in per_fold.values()) >= 4,
        "wins_exceed_losses": overall["wins"] > overall["losses"],
        "crossings_into_exceed_out": overall["gold_crossings_into_top5"] > overall["gold_crossings_out_of_top5"],
        "multi_gold_delta_gte_minus_0_005": overall["multi_gold"]["delta"] >= -0.005,
        "worst_fold_delta_gte_minus_0_005": min(value["delta_recall_at_5"] for value in per_fold.values()) >= -0.005,
        "candidate_and_hash_integrity": (
            sha256(BUNDLE / "V2_FOLDS.json") == prereg["fixed_population"]["folds_sha256"]
            and sha256(BUNDLE / "V2_CANDIDATE_POOL.jsonl") == prereg["fixed_population"]["candidate_pool_sha256"]
        ),
    }
    status = "PASS" if all(checks.values()) else (
        "KILL" if overall["delta_recall_at_5"] < 0.002 else "INCONCLUSIVE_NO_RESCUE"
    )
    report = {
        "schema_version": "dsc2026.research_v2.adapted_e5_lal_equal_rrf32.v1",
        "status": status,
        "hypothesis": prereg["hypothesis"],
        "scientific_contract": prereg["mechanism"],
        "overall": overall,
        "per_fold": per_fold,
        "gate_checks": checks,
        "integrity": {
            "queries": len(rows),
            "candidate_membership_unchanged": True,
            "calibration_or_label_fit": False,
            "lal_signature_counts": dict(sorted(signatures.items())),
            "folds_sha256": sha256(BUNDLE / "V2_FOLDS.json"),
            "candidate_pool_sha256": sha256(BUNDLE / "V2_CANDIDATE_POOL.jsonl"),
            "source_database_sha256": sha256(SOURCE_DB),
            "preregistration_sha256": sha256(PREREG),
            "predictions_sha256": sha256(prediction_path),
            "runner_sha256": sha256(Path(__file__)),
        },
        "anti_rescue": "No RRF constant/weight grid, learned fusion, routing, threshold, candidate append, CAL600 selection, or submission.",
    }
    report_path = OUT / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_REPORT.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
