"""Cache-only complementarity anatomy for strict Research V2.

No trainable model is fit here.  The reference is the five-fold OOF adapted-E5
expert. Native-source and immutable-pool-restricted views are reported
separately so candidate information is never mistaken for ranking information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/research_v2_post_e5"
CONFIRM = ROOT / "results/research_v2_e5_confirmation"
FOLD0 = ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0"
FORENSIC = ROOT / "results/research_v2_forensic"
BUNDLE = ROOT / "cache/research_v2_e5_confirmation/bundle-v1"
SOURCE_DB = ROOT.parent / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def rank_bucket(rank: int | None) -> str:
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
    if rank <= 100:
        return "51-100"
    return "101+"


def recall(top5: list[str], gold: set[str]) -> float:
    return len(set(top5) & gold) / len(gold)


def load_adapted() -> dict[str, dict[str, Any]]:
    paths = {
        0: FOLD0 / "score/E5_TRANSFER_FOLD0_PREDICTIONS.jsonl",
        **{
            index: CONFIRM / f"fold_{index}/score/E5_CONFIRMATION_FOLD_{index}_PREDICTIONS.jsonl"
            for index in range(1, 5)
        },
    }
    rows: dict[str, dict[str, Any]] = {}
    for index, path in paths.items():
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if qid in rows:
                raise RuntimeError(f"duplicate adapted prediction qid: {qid}")
            row["fold"] = f"fold_{index}"
            rows[qid] = row
    if len(rows) != 6991:
        raise RuntimeError(f"adapted OOF population mismatch: {len(rows)}")
    return rows


def top5_map(path: Path) -> dict[str, list[str]]:
    result = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        result[qid] = list(map(str, row["top5"]))
    if len(result) != 6991:
        raise RuntimeError(f"prediction population mismatch for {path}: {len(result)}")
    return result


def source_views(
    qids: list[str], pools: dict[str, list[str]], golds: dict[str, set[str]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, dict[str, int]]], dict[str, Any]]:
    keys = ("e5", "lal", "bm25", "trigram", "jina")
    views = {f"{key}_native": {} for key in keys}
    views.update({f"{key}_pool": {} for key in keys})
    gold_ranks: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    signatures: dict[str, Counter[str]] = {key: Counter() for key in keys}
    uri = f"file:{SOURCE_DB.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    for index, qid in enumerate(qids, 1):
        rows = connection.execute(
            "SELECT source,payload,signature FROM sources WHERE q=?", (qid,)
        ).fetchall()
        payloads = {source: (json.loads(payload), signature) for source, payload, signature in rows}
        missing = set(keys) - set(payloads)
        if missing:
            raise RuntimeError(f"missing source rows for {qid}: {sorted(missing)}")
        pool = set(pools[qid])
        for key in keys:
            payload, signature = payloads[key]
            signatures[key][signature] += 1
            order = [str(item["doc_id"]) for item in payload]
            if len(order) != len(set(order)):
                raise RuntimeError(f"duplicate docs in {qid}/{key}")
            views[f"{key}_native"][qid] = order[:5]
            views[f"{key}_pool"][qid] = [doc for doc in order if doc in pool][:5]
            rank = {doc: pos for pos, doc in enumerate(order, 1) if doc in golds[qid]}
            gold_ranks[qid][key] = rank
        if index % 1000 == 0:
            print(json.dumps({"stage": "read_sources", "completed": index, "total": len(qids)}), flush=True)
    connection.close()
    provenance = {
        "database": str(SOURCE_DB),
        "database_sha256": sha256(SOURCE_DB),
        "integrity_check": integrity,
        "rows": len(qids) * len(keys),
        "signature_counts": {
            key: dict(sorted(counter.items())) for key, counter in signatures.items()
        },
    }
    return views, gold_ranks, provenance


def analyze(
    name: str,
    predictions: dict[str, list[str]],
    reference: dict[str, dict[str, Any]],
    pools: dict[str, list[str]],
    source_gold_ranks: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    per_fold: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    ref_values, expert_values, union_values = [], [], []
    ref_better = expert_better = changed = 0
    ref_exclusive_mass = expert_exclusive_mass = 0.0
    single, multi = [], []
    rescued_locations: Counter[str] = Counter()
    rescued_candidate_membership: Counter[str] = Counter()
    rescued_source_ranks: dict[str, Counter[str]] = defaultdict(Counter)
    rescued_gold_occurrences = 0

    for qid, row in reference.items():
        gold = set(map(str, row["gold"]))
        ref_top5 = list(map(str, row["ft_order"][:5]))
        expert_top5 = list(map(str, predictions[qid][:5]))
        ref = recall(ref_top5, gold)
        expert = recall(expert_top5, gold)
        union = recall(list(set(ref_top5) | set(expert_top5)), gold)
        ref_values.append(ref)
        expert_values.append(expert)
        union_values.append(union)
        ref_better += ref > expert
        expert_better += expert > ref
        changed += set(ref_top5) != set(expert_top5)
        ref_only = (gold & set(ref_top5)) - set(expert_top5)
        expert_only = (gold & set(expert_top5)) - set(ref_top5)
        ref_exclusive_mass += len(ref_only) / len(gold)
        expert_exclusive_mass += len(expert_only) / len(gold)
        fold = str(row["fold"])
        per_fold[fold].append((ref, expert, union))
        (single if len(gold) == 1 else multi).append((ref, expert, union))
        adapted_order = list(map(str, row["ft_order"]))
        pool = set(pools[qid])
        for doc_id in expert_only:
            rescued_gold_occurrences += 1
            try:
                adapted_rank = adapted_order.index(doc_id) + 1
            except ValueError:
                adapted_rank = None
            rescued_locations[rank_bucket(adapted_rank)] += 1
            rescued_candidate_membership["in_pool" if doc_id in pool else "out_of_pool"] += 1
            for source, ranks in source_gold_ranks[qid].items():
                rescued_source_ranks[source][rank_bucket(ranks.get(doc_id))] += 1

    def slice_metrics(values: list[tuple[float, float, float]]) -> dict[str, float | int]:
        return {
            "queries": len(values),
            "adapted_e5": mean(value[0] for value in values),
            "expert": mean(value[1] for value in values),
            "union_oracle": mean(value[2] for value in values),
            "union_increment_over_adapted_e5": mean(value[2] - value[0] for value in values),
        }

    result = {
        "expert": name,
        "queries": len(reference),
        "standalone_recall_at_5": mean(expert_values),
        "adapted_e5_recall_at_5": mean(ref_values),
        "delta_vs_adapted_e5": mean(e - r for r, e in zip(ref_values, expert_values)),
        "paired": {
            "expert_wins": expert_better,
            "adapted_e5_wins": ref_better,
            "ties": len(reference) - expert_better - ref_better,
            "changed_top5_sets": changed,
        },
        "top5_set_union_oracle": mean(union_values),
        "union_increment_over_adapted_e5": mean(u - r for r, u in zip(ref_values, union_values)),
        "exclusive_gold_recall_mass": {
            "expert_over_adapted_e5": expert_exclusive_mass / len(reference),
            "adapted_e5_over_expert": ref_exclusive_mass / len(reference),
            "expert_rescued_gold_occurrences": rescued_gold_occurrences,
        },
        "per_fold": {
            fold: slice_metrics(values) for fold, values in sorted(per_fold.items())
        },
        "single_gold": slice_metrics(single),
        "multi_gold": slice_metrics(multi),
        "expert_rescued_gold_location_in_adapted_e5_ranking": dict(sorted(rescued_locations.items())),
        "expert_rescued_gold_candidate_membership": dict(sorted(rescued_candidate_membership.items())),
        "expert_rescued_gold_native_source_rank_buckets": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(rescued_source_ranks.items())
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "V2_COMPLEMENTARITY_AUDIT.json")
    args = parser.parse_args()

    reference = load_adapted()
    qids = sorted(reference, key=int)
    pools = {
        str(row["qid"]): list(map(str, row["doc_ids"]))
        for row in read_jsonl(BUNDLE / "V2_CANDIDATE_POOL.jsonl")
    }
    if set(reference) != set(pools):
        raise RuntimeError("reference/pool qid mismatch")
    golds = {qid: set(map(str, reference[qid]["gold"])) for qid in qids}

    predictions: dict[str, dict[str, list[str]]] = {
        "adapted_e5_oof": {qid: list(map(str, reference[qid]["ft_order"][:5])) for qid in qids},
        "frozen_e5_exact_pool": {qid: list(map(str, reference[qid]["base_order"][:5])) for qid in qids},
        "jina_v2_lexical_pool": top5_map(FORENSIC / "V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl"),
        "jina_v2_structural_pool": top5_map(FORENSIC / "V2_ZERO_SHOT_STRUCTURAL_PREDICTIONS.jsonl"),
    }
    source_predictions, source_gold_ranks, source_provenance = source_views(
        qids, pools, golds
    )
    predictions.update(source_predictions)

    classifications = {
        "adapted_e5_oof": {"class": "STRICT_V2_OOF_READY", "decision_eligible": True},
        "frozen_e5_exact_pool": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True},
        "jina_v2_lexical_pool": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True},
        "jina_v2_structural_pool": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": False, "reason": "matched evidence contract rejected on all five folds"},
        "lal_native": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True, "information_role": "candidate_and_ranking"},
        "lal_pool": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True, "information_role": "ranking_only_on_immutable_pool"},
        "jina_native": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True, "information_role": "candidate_and_late_interaction_ranking"},
        "jina_pool": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True, "information_role": "late_interaction_ranking_on_immutable_pool"},
        "e5_native": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True},
        "e5_pool": {"class": "FROZEN_LABEL_FREE_V2_READY", "decision_eligible": True},
        "bm25_native": {"class": "HISTORICAL_ONLY", "decision_eligible": False, "reason": "per-query old-fold OOF configuration lacks V2 duplicate-linked isolation"},
        "bm25_pool": {"class": "HISTORICAL_ONLY", "decision_eligible": False, "reason": "per-query old-fold OOF configuration lacks V2 duplicate-linked isolation"},
        "trigram_native": {"class": "HISTORICAL_ONLY", "decision_eligible": False, "reason": "EXP-111 old-fold supervised configuration, not strict V2 outer isolated"},
        "trigram_pool": {"class": "HISTORICAL_ONLY", "decision_eligible": False, "reason": "EXP-111 old-fold supervised configuration, not strict V2 outer isolated"},
    }

    analyses = {
        name: analyze(name, values, reference, pools, source_gold_ranks)
        for name, values in predictions.items()
    }

    eligible_pool = [
        "frozen_e5_exact_pool", "jina_v2_lexical_pool", "lal_pool", "jina_pool"
    ]
    eligible_native = ["e5_native", "lal_native", "jina_native"]

    def all_union(names: list[str]) -> dict[str, Any]:
        values = []
        out_of_pool_rescues = 0
        for qid in qids:
            gold = golds[qid]
            docs = set(reference[qid]["ft_order"][:5])
            for name in names:
                docs.update(predictions[name][qid][:5])
            values.append(len(gold & docs) / len(gold))
            out_of_pool_rescues += len((gold & docs) - set(pools[qid]))
        return {
            "experts": ["adapted_e5_oof", *names],
            "set_union_oracle": mean(values),
            "increment_over_adapted_e5": mean(values) - analyses["adapted_e5_oof"]["standalone_recall_at_5"],
            "out_of_pool_gold_occurrences_rescued": out_of_pool_rescues,
        }

    pool_ceiling = mean(
        len(golds[qid] & set(pools[qid])) / len(golds[qid]) for qid in qids
    )
    report = {
        "schema_version": "dsc2026.research_v2.post_e5_complementarity.v1",
        "status": "COMPLETE_CACHE_ONLY_NO_FIT",
        "population": {"queries": 6991, "parents": 8507},
        "folds_sha256": sha256(BUNDLE / "V2_FOLDS.json"),
        "candidate_pool_sha256": sha256(BUNDLE / "V2_CANDIDATE_POOL.jsonl"),
        "candidate_ceiling": pool_ceiling,
        "adapted_e5_reference_recall_at_5": analyses["adapted_e5_oof"]["standalone_recall_at_5"],
        "within_pool_ranking_headroom": pool_ceiling - analyses["adapted_e5_oof"]["standalone_recall_at_5"],
        "classifications": classifications,
        "experts": analyses,
        "clean_pool_restricted_all_expert_union": all_union(eligible_pool),
        "clean_native_source_all_expert_union": all_union(eligible_native),
        "source_database_provenance": source_provenance,
        "input_hashes": {
            "confirmation_report": sha256(CONFIRM / "E5_STRICT_CONFIRMATION_REPORT.json"),
            "lexical_predictions": sha256(FORENSIC / "V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl"),
            "structural_predictions": sha256(FORENSIC / "V2_ZERO_SHOT_STRUCTURAL_PREDICTIONS.jsonl"),
            "source_database": source_provenance["database_sha256"],
        },
        "interpretation_contract": {
            "set_union_is_diagnostic_only": True,
            "pool_views_isolate_ranking_information": True,
            "native_views_mix_candidate_and_ranking_information": True,
            "no_learned_fusion_or_threshold_fit": True,
        },
    }
    write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "candidate_ceiling": pool_ceiling,
        "adapted_e5": report["adapted_e5_reference_recall_at_5"],
        "ranking_headroom": report["within_pool_ranking_headroom"],
        "eligible_pool_union": report["clean_pool_restricted_all_expert_union"],
        "eligible_native_union": report["clean_native_source_all_expert_union"],
        "experts": {
            name: {
                "recall": analyses[name]["standalone_recall_at_5"],
                "union_increment": analyses[name]["union_increment_over_adapted_e5"],
            }
            for name in analyses
        },
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
