"""Run the preregistered zero-fit three-expert Research V2 RRF32 test."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from research_v2_post_e5 import run_adapted_e5_lal_rrf32 as base


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/research_v2_post_e5"
PREREG = OUT / "NEXT_V2_HYPOTHESIS_PREREGISTRATION_02.json"
JINA_DB = ROOT / "cache/research_v2_forensic/evidence_ab_scores.sqlite"
EXPECTED_JINA_DB_SHA = "e2b00064a5bcce90cf6558027ff90c30aeb3baa67a78cb97ea81f5445eb95f7f"


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "PREREGISTERED_BEFORE_RESULT":
        raise RuntimeError("preregistration is not sealed")
    if base.sha256(JINA_DB) != EXPECTED_JINA_DB_SHA:
        raise RuntimeError("Jina lexical score database hash mismatch")
    reference_rows = base.load_reference()
    pools = {
        str(row["qid"]): list(map(str, row["doc_ids"]))
        for row in base.read_jsonl(base.BUNDLE / "V2_CANDIDATE_POOL.jsonl")
    }
    saved_two = {
        str(row["qid"]): row
        for row in base.read_jsonl(OUT / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl")
    }
    if set(reference_rows) != set(pools) or set(reference_rows) != set(saved_two):
        raise RuntimeError("population mismatch")

    source = sqlite3.connect(f"file:{base.SOURCE_DB.as_posix()}?mode=ro", uri=True)
    source.execute("PRAGMA query_only=ON")
    jina = sqlite3.connect(f"file:{JINA_DB.as_posix()}?mode=ro", uri=True)
    jina.execute("PRAGMA query_only=ON")
    if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("source DB integrity failure")
    if jina.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("Jina DB integrity failure")

    rows = []
    lal_signatures: Counter[str] = Counter()
    two_expert_parity = 0
    for qid in sorted(reference_rows, key=int):
        source_row = reference_rows[qid]
        pool = pools[qid]
        pool_set = set(pool)
        adapted_order = list(map(str, source_row["ft_order"]))
        adapted_rank = {doc: index for index, doc in enumerate(adapted_order, 1)}
        payload, signature = source.execute(
            "SELECT payload,signature FROM sources WHERE q=? AND source='lal'", (qid,)
        ).fetchone()
        lal_signatures[signature] += 1
        lal_native = [str(item["doc_id"]) for item in json.loads(payload)]
        lal_order = [doc for doc in lal_native if doc in pool_set]
        lal_rank = {doc: index for index, doc in enumerate(lal_order, 1)}
        jina_rows = jina.execute(
            "SELECT doc_id,score FROM scores WHERE qid=? AND arm='lexical'", (qid,)
        ).fetchall()
        jina_scores = {str(doc): float(score) for doc, score in jina_rows}
        if set(jina_scores) != pool_set:
            raise RuntimeError(f"Jina candidate membership mismatch: {qid}")
        jina_order = sorted(pool, key=lambda doc: (-jina_scores[doc], doc))
        jina_rank = {doc: index for index, doc in enumerate(jina_order, 1)}

        score_two = {
            doc: 1.0 / (32 + adapted_rank[doc])
            + (1.0 / (32 + lal_rank[doc]) if doc in lal_rank else 0.0)
            for doc in pool
        }
        order_two = sorted(pool, key=lambda doc: (-score_two[doc], doc))
        if order_two[:5] == list(map(str, saved_two[qid]["fused_top5"])):
            two_expert_parity += 1
        score_three = {
            doc: score_two[doc] + 1.0 / (32 + jina_rank[doc]) for doc in pool
        }
        order_three = sorted(pool, key=lambda doc: (-score_three[doc], doc))
        if set(order_three) != pool_set or len(order_three) != len(pool):
            raise RuntimeError(f"candidate membership changed: {qid}")
        gold = set(map(str, source_row["gold"]))
        base_hits = len(gold & set(order_two[:5]))
        fused_hits = len(gold & set(order_three[:5]))
        rows.append({
            "qid": qid,
            "fold": source_row["fold"],
            "gold": sorted(gold),
            "base_top5": order_two[:5],
            "fused_top5": order_three[:5],
            "base_hits": base_hits,
            "fused_hits": fused_hits,
            "base_recall_at_5": base_hits / len(gold),
            "fused_recall_at_5": fused_hits / len(gold),
            "base_gold_ranks": {
                doc: order_two.index(doc) + 1 if doc in pool_set else None for doc in sorted(gold)
            },
            "fused_gold_ranks": {
                doc: order_three.index(doc) + 1 if doc in pool_set else None for doc in sorted(gold)
            },
        })
    source.close()
    jina.close()
    if two_expert_parity != 6991:
        raise RuntimeError(f"two-expert reference parity failed: {two_expert_parity}/6991")

    prediction = OUT / "V2_ADAPTED_E5_LAL_JINAV2_EQUAL_RRF32_PREDICTIONS.jsonl"
    with prediction.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    overall = base.summarize(rows)
    per_fold = {
        fold: base.summarize([row for row in rows if row["fold"] == fold])
        for fold in [f"fold_{index}" for index in range(5)]
    }
    checks = {
        "pooled_delta_gte_0_005": overall["delta_recall_at_5"] >= 0.005,
        "at_least_4_of_5_folds_nonnegative": sum(value["delta_recall_at_5"] >= 0 for value in per_fold.values()) >= 4,
        "wins_exceed_losses": overall["wins"] > overall["losses"],
        "crossings_into_exceed_out": overall["gold_crossings_into_top5"] > overall["gold_crossings_out_of_top5"],
        "multi_gold_delta_gte_minus_0_005": overall["multi_gold"]["delta"] >= -0.005,
        "worst_fold_delta_gte_minus_0_005": min(value["delta_recall_at_5"] for value in per_fold.values()) >= -0.005,
        "integrity": two_expert_parity == 6991,
    }
    status = "PASS" if all(checks.values()) else (
        "KILL" if overall["delta_recall_at_5"] < 0.002 else "INCONCLUSIVE_NO_RESCUE"
    )
    report = {
        "schema_version": "dsc2026.research_v2.adapted_e5_lal_jina_equal_rrf32.v1",
        "status": status,
        "hypothesis": prereg["hypothesis"],
        "scientific_contract": prereg["mechanism"],
        "overall": overall,
        "per_fold": per_fold,
        "gate_checks": checks,
        "integrity": {
            "queries": 6991,
            "two_expert_top5_parity": two_expert_parity,
            "candidate_membership_unchanged": True,
            "calibration_or_label_fit": False,
            "lal_signature_counts": dict(sorted(lal_signatures.items())),
            "candidate_pool_sha256": base.sha256(base.BUNDLE / "V2_CANDIDATE_POOL.jsonl"),
            "jina_score_database_sha256": base.sha256(JINA_DB),
            "source_database_sha256": base.sha256(base.SOURCE_DB),
            "preregistration_sha256": base.sha256(PREREG),
            "predictions_sha256": base.sha256(prediction),
            "runner_sha256": base.sha256(Path(__file__)),
        },
        "anti_rescue": "No weight/k/subset grid, learned fusion, routing, threshold, candidate append, CAL600, or submission.",
    }
    base.write_json(OUT / "V2_ADAPTED_E5_LAL_JINAV2_EQUAL_RRF32_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
