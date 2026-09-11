"""Read-only reconciliation of LegalIR EXP-112 with Research V2.

This script never modifies LegalIR artifacts.  It reconstructs the direct
frozen-E5 versus epoch-2 adapted-E5 comparison from the sealed EXP-112 ranks
and canonical labels, then writes one compact JSON audit in the Research V2
namespace.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


LEGALIR = Path(r"D:\Study\DSC2026\LegalIR")
SOTA = Path(r"D:\Study\DSC2026\sota")
EXP112_RESULTS = LEGALIR / "results" / "exp112_task_adaptive_retrieval"
EXP112_CACHE = LEGALIR / "cache" / "exp112_task_adaptive_retrieval"
FROZEN_RANKS = (
    LEGALIR
    / "cache"
    / "exp021_e5_dense_candidates"
    / "config_rankings"
    / "top2_mean.jsonl"
)
OUT_DIR = SOTA / "results" / "research_v2_kaggle_audit"
OUT_PATH = OUT_DIR / "EXP112_V2_E5_TRANSFER_AUDIT.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = list(values)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
    }


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


def selected_calibration_summary(fold: int) -> dict[str, Any]:
    rows = read_json(EXP112_RESULTS / "outer" / f"fold_{fold}" / "CALIBRATION_SCREEN.json")
    lock = read_json(EXP112_RESULTS / "outer" / f"fold_{fold}" / "SELECTION_LOCK.json")

    def same_recipe(recipe: dict[str, Any], selected: dict[str, Any]) -> bool:
        keys = ("epoch", "beta", "pool", "family", "block", "alpha", "confidence")
        return all(recipe.get(key) == selected.get(key) for key in keys)

    selected_row = next(row for row in rows if same_recipe(row["recipe"], lock))
    active = [
        row
        for row in rows
        if int(row["recipe"].get("epoch", 0)) > 0
        and float(row["recipe"].get("beta", 0.0)) > 0.0
    ]
    best_active = max(active, key=lambda row: float(row["metrics"]["canonical"]["recall@5"]))
    selected_r5 = float(selected_row["metrics"]["canonical"]["recall@5"])
    active_r5 = float(best_active["metrics"]["canonical"]["recall@5"])
    return {
        "fold": fold,
        "calibration_fold": f"fold_{(fold - 1) % 5}",
        "selected_recipe": {key: lock.get(key) for key in (
            "epoch", "nominal_epochs", "beta", "pool", "family", "block", "alpha", "confidence"
        )},
        "selected_recall_at_5": selected_r5,
        "best_adapter_active_recipe": best_active["recipe"],
        "best_adapter_active_recall_at_5": active_r5,
        "adapter_active_delta_vs_selected": active_r5 - selected_r5,
    }


def load_frozen(gold: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with FROZEN_RANKS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row["qid"])
            if qid not in gold or not gold[qid]:
                continue
            candidates = row["candidates"]
            order = [str(item["doc_id"]) for item in candidates]
            rank_for = {doc_id: index + 1 for index, doc_id in enumerate(order)}
            rows[qid] = {
                "top5": order[:5],
                "gold_ranks": {doc_id: rank_for.get(doc_id) for doc_id in gold[qid]},
            }
    return rows


def load_adapted(qid: str, fold: int, gold_docs: set[str]) -> dict[str, Any]:
    path = EXP112_CACHE / "outer" / f"fold_{fold}" / "test-query" / f"{qid}.json"
    row = read_json(path)
    order = [str(value) for value in row["order"]]
    rank_for = {doc_id: index + 1 for index, doc_id in enumerate(order)}
    return {
        "top5": order[:5],
        "gold_ranks": {doc_id: rank_for.get(doc_id) for doc_id in gold_docs},
        "frozen_query_cosine": float(row["frozen_query_cosine"]),
    }


def evaluate_fold(
    fold: int,
    qids: list[str],
    gold: dict[str, set[str]],
    frozen: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_rows: list[dict[str, Any]] = []
    crossing_counter: Counter[str] = Counter()
    gold_rank_deltas: list[float] = []
    for qid in qids:
        gold_docs = gold[qid]
        if not gold_docs:
            continue
        base = frozen[qid]
        adapted = load_adapted(qid, fold, gold_docs)
        base_hits = len(set(base["top5"]) & gold_docs)
        adapted_hits = len(set(adapted["top5"]) & gold_docs)
        base_recall = base_hits / len(gold_docs)
        adapted_recall = adapted_hits / len(gold_docs)
        for doc_id in gold_docs:
            old_rank = base["gold_ranks"][doc_id]
            new_rank = adapted["gold_ranks"][doc_id]
            old_bucket = rank_bucket(old_rank)
            new_bucket = rank_bucket(new_rank)
            crossing_counter[f"{old_bucket}->{new_bucket}"] += 1
            if old_rank is not None and new_rank is not None:
                gold_rank_deltas.append(float(old_rank - new_rank))
        query_rows.append(
            {
                "qid": qid,
                "fold": fold,
                "gold_count": len(gold_docs),
                "frozen_recall_at_5": base_recall,
                "adapted_recall_at_5": adapted_recall,
                "delta": adapted_recall - base_recall,
                "changed_top5_set": set(base["top5"]) != set(adapted["top5"]),
                "frozen_query_cosine": adapted["frozen_query_cosine"],
            }
        )

    wins = sum(row["delta"] > 0 for row in query_rows)
    losses = sum(row["delta"] < 0 for row in query_rows)
    single = [row for row in query_rows if row["gold_count"] == 1]
    multi = [row for row in query_rows if row["gold_count"] > 1]

    def slice_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(rows),
            "frozen_recall_at_5": mean(row["frozen_recall_at_5"] for row in rows),
            "adapted_recall_at_5": mean(row["adapted_recall_at_5"] for row in rows),
            "delta": mean(row["delta"] for row in rows),
            "wins": sum(row["delta"] > 0 for row in rows),
            "losses": sum(row["delta"] < 0 for row in rows),
            "ties": sum(row["delta"] == 0 for row in rows),
        }

    summary = {
        "fold": fold,
        "all": slice_stats(query_rows),
        "single_gold": slice_stats(single),
        "multi_gold": slice_stats(multi),
        "changed_top5_sets": sum(row["changed_top5_set"] for row in query_rows),
        "gold_rank_improvement": metric_summary(gold_rank_deltas),
        "gold_rank_bucket_movements": dict(sorted(crossing_counter.items())),
        "gold_crossings_into_top5": sum(
            count
            for key, count in crossing_counter.items()
            if not key.startswith("1-5->") and key.endswith("->1-5")
        ),
        "gold_crossings_out_of_top5": sum(
            count
            for key, count in crossing_counter.items()
            if key.startswith("1-5->") and not key.endswith("->1-5")
        ),
        "query_drift_cosine": metric_summary(row["frozen_query_cosine"] for row in query_rows),
        "wins": wins,
        "losses": losses,
    }
    return summary, query_rows


def main() -> None:
    sys.path.insert(0, str(LEGALIR / "src"))
    from exp109b_encoder_complementarity import canonical_labels  # type: ignore

    gold, label_audit = canonical_labels()
    folds = read_json(LEGALIR / "cache" / "cv_folds.json")
    frozen = load_frozen(gold)
    expected = {qid for qid, values in gold.items() if values}
    if set(frozen) != expected:
        raise RuntimeError(
            f"frozen rank population mismatch: missing={len(expected-set(frozen))}, "
            f"extra={len(set(frozen)-expected)}"
        )

    fold_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for fold in range(5):
        summary, rows = evaluate_fold(
            fold,
            [str(value) for value in folds[f"fold_{fold}"]],
            gold,
            frozen,
        )
        fold_summaries.append(summary)
        all_rows.extend(rows)

    def pooled_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(rows),
            "frozen_recall_at_5": mean(row["frozen_recall_at_5"] for row in rows),
            "adapted_recall_at_5": mean(row["adapted_recall_at_5"] for row in rows),
            "delta": mean(row["delta"] for row in rows),
            "wins": sum(row["delta"] > 0 for row in rows),
            "losses": sum(row["delta"] < 0 for row in rows),
            "ties": sum(row["delta"] == 0 for row in rows),
            "changed_top5_sets": sum(row["changed_top5_set"] for row in rows),
        }

    training_runs = []
    for fold in range(5):
        marker_path = EXP112_CACHE / "outer" / f"fold_{fold}" / "query-outer" / "_SUCCESS.json"
        marker = read_json(marker_path)
        training_runs.append(
            {
                "fold": fold,
                "training_queries": len(marker["training_qids"]),
                "epochs": marker["epochs"],
                "updates": marker["updates"],
                "seconds": marker["seconds"],
                "contract_hash": marker["contract"],
                "checkpoint_sha256": marker["files"],
            }
        )

    key_files = [
        LEGALIR / "docs" / "EXP-112_PLAN.md",
        EXP112_RESULTS / "EXP112_SOURCE_SNAPSHOT_20260905.zip",
        EXP112_RESULTS / "OOF_REPORT.json",
        EXP112_RESULTS / "FINAL_CONFIG_LOCK.json",
        LEGALIR / "results" / "exp_final_retrieval" / "adapter_audit" / "ADAPTER_AUDIT.json",
        LEGALIR / "results" / "exp_final_retrieval" / "adapter_objective_probe" / "ADAPTER_OBJECTIVE_REPORT.json",
        LEGALIR / "cache" / "cv_folds.json",
        SOTA / "results" / "research_v2_forensic" / "V2_FOLDS.json",
    ]

    report = {
        "schema_version": "dsc2026.research_v2.exp112_transfer_audit.v1",
        "status": "COMPLETE_READ_ONLY_PRIOR_ART_RECONCILIATION",
        "classification": "PARTIAL_DUPLICATE_WITH_POSITIVE_PRIOR",
        "label_audit": label_audit,
        "historical_exp112_contract": {
            "base_model": "mainguyen9/vietlegal-e5",
            "adaptation_scope": "query encoder only; frozen document chunk bank",
            "prefixes": {"query": "query: ", "document": "passage: "},
            "pooling": "attention-mask mean pooling followed by L2 normalization",
            "max_length": 512,
            "lora": {
                "targets": ["query", "value"],
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "bias": "none",
            },
            "parent_aggregation": "mean of top two chunk cosines; singleton uses top one",
            "negative_mining_per_query": {
                "total_unique": 64,
                "current_hardest": 16,
                "adapted_ranks_17_100_rotated": 16,
                "sparse_lal_disagreement": 16,
                "jina_or_dense_confusers": 8,
                "random": 8,
                "gold_and_alias_exclusion": True,
            },
            "loss": {
                "core": "independent-positive temperature-scaled contrastive loss over every positive versus selected negatives",
                "temperature": 0.05,
                "positive_weighting": "per-query normalized; every canonical gold positive",
                "sibling_gold_as_negative": False,
                "query_drift_penalty_weight": 0.05,
                "epoch_2_boundary_softplus_weight": 0.25,
                "boundary_definition": "positive and negative lie on opposite sides of full-corpus parent rank 5",
            },
            "optimization": {
                "epochs": 2,
                "optimizer": "AdamW",
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "effective_batch_queries": 16,
                "warmup_ratio": 0.1,
                "schedule": "cosine",
                "seed": 112,
            },
            "fold_isolation": {
                "outer_folds": 5,
                "selection_calibration_map": {
                    "fold_0": "fold_4",
                    "fold_1": "fold_0",
                    "fold_2": "fold_1",
                    "fold_3": "fold_2",
                    "fold_4": "fold_3",
                },
                "inner_training": "three folds excluding outer-held and calibration",
                "outer_refit": "four non-held folds",
                "outer_scoring": "held fold only",
            },
        },
        "reconstructed_epoch2_standalone": {
            "per_fold": fold_summaries,
            "pooled": {
                "all": pooled_slice(all_rows),
                "single_gold": pooled_slice([row for row in all_rows if row["gold_count"] == 1]),
                "multi_gold": pooled_slice([row for row in all_rows if row["gold_count"] > 1]),
            },
        },
        "selection_reconciliation": {
            "per_outer_fold_calibration": [selected_calibration_summary(fold) for fold in range(5)],
            "all_outer_locks": "epoch=0, beta=0, pool=frozen",
            "final_config_lock": "epoch=0, beta=0, pool=frozen",
            "causal_interpretation": (
                "The adapted E5 standalone retriever improved, but EXP-112 selected a much stronger supervised "
                "ML ranking recipe. On every calibration fold, the best recipe that actively blended adapted-E5 "
                "ranks (epoch>0, beta>0) scored below the frozen beta=0 recipe. Standalone gains therefore did "
                "not establish complementary residual value for that fusion interface; outer standalone "
                "diagnostics were deliberately non-selective and could not change the locks."
            ),
        },
        "training_runs": training_runs,
        "transfer_decision": {
            "classification": "PARTIAL_DUPLICATE_WITH_POSITIVE_PRIOR",
            "preserve_from_exp112": [
                "query-only LoRA with frozen document bank",
                "native E5 prefixes, mean pooling, normalization, and 512-token contract",
                "top2_mean parent aggregation",
                "64-way negative mixture and multi-positive sibling-safe loss",
                "two-epoch schedule including drift and epoch-2 boundary term",
                "fixed seed and optimization settings",
            ],
            "single_causal_variable_to_change": (
                "downstream scoring role/population: score only the immutable V2 E5@50 union novel BM25@10 "
                "candidate pool and take direct deterministic Top-5, instead of adding adapted full-corpus "
                "retrieval/rank fusion to EXP-112's supervised ML system"
            ),
            "required_noncausal_protocol_changes": [
                "use sealed V2 folds",
                "exclude exact/near-duplicate-linked Fold-0 queries from training and mining",
            ],
            "gpu_training_launched": False,
        },
        "provenance_sha256": {str(path): sha256(path) for path in key_files},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(OUT_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
