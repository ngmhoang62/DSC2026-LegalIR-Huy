"""Cache-first Huy-style 5-fold reconstruction, ports, and ablations.

The ranker/feature contract is inherited from burst_userft_maxrecall:
per-view reciprocal/raw ranks, within-query standardized score margins,
doctype/citation metadata, StandardScaler, and balanced LR(C=.15).

Only channels with complete 6,991-query coverage are decision eligible.  The
historical Huy task-finetuned channels are recorded but never silently used.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT.parent
OUT = ROOT / "results/huy_fasttrack"
FOLDS_PATH = ROOT / "results/research_v2_forensic/V2_FOLDS.json"
POOL_PATH = ROOT / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"
BOUNDARY_MANIFEST = ROOT / "results/research_v2_forensic/V2_BOUNDARY_GROUPS_MANIFEST.json"
SOURCE_DB = WORKSPACE / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"
JINA_DB = ROOT / "results/research_v2_forensic/research_v2_jina_boundary/evidence_ab_scores.sqlite"
DATA = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def adapted_paths() -> list[tuple[int, Path]]:
    return [
        (0, ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/score/E5_TRANSFER_FOLD0_PREDICTIONS.jsonl"),
        *[
            (i, ROOT / f"results/research_v2_e5_confirmation/fold_{i}/score/E5_CONFIRMATION_FOLD_{i}_PREDICTIONS.jsonl")
            for i in range(1, 5)
        ],
    ]


def load_inputs():
    folds_payload = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    folds = {name: [str(x) for x in ids if str(x) not in set(folds_payload["population"]["non_evaluable_qids"])]
             for name, ids in folds_payload["folds"].items()}
    pools, questions = {}, {}
    for row in read_jsonl(POOL_PATH):
        qid = str(row["qid"])
        pools[qid] = [str(x) for x in row["doc_ids"]]
        questions[qid] = str(row["query"])

    golds: dict[str, set[str]] = {}
    e5_orders: dict[str, dict[str, list[str]]] = {"frozen_e5": {}, "adapted_e5": {}}
    e5_scores: dict[str, dict[str, dict[str, float]]] = {"frozen_e5": {}, "adapted_e5": {}}
    prediction_hashes = {}
    for fold, path in adapted_paths():
        prediction_hashes[f"fold_{fold}"] = sha256(path)
        for row in read_jsonl(path):
            qid = str(row["qid"])
            golds[qid] = set(map(str, row["gold"]))
            for channel, order_key, score_key in (
                ("frozen_e5", "base_order", "base_scores"),
                ("adapted_e5", "ft_order", "ft_scores"),
            ):
                order = list(map(str, row[order_key]))
                e5_orders[channel][qid] = order
                e5_scores[channel][qid] = {
                    doc: float(score) for doc, score in zip(order, row[score_key])
                }
    if set(pools) != set(golds) or len(golds) != 6991:
        raise RuntimeError("V2 pool/gold population mismatch")

    duplicate_exclusions = json.loads(BOUNDARY_MANIFEST.read_text(encoding="utf-8"))[
        "held_fold_duplicate_exclusions"
    ]
    return folds, pools, questions, golds, e5_orders, e5_scores, duplicate_exclusions, prediction_hashes


def load_jina(pools: dict[str, list[str]]):
    con = sqlite3.connect(f"file:{JINA_DB.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for qid, doc, score in con.execute(
        "SELECT qid,doc_id,score FROM scores WHERE arm='lexical'"
    ):
        scores[str(qid)][str(doc)] = float(score)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    missing = sum(doc not in scores[qid] for qid, docs in pools.items() for doc in docs)
    if missing:
        raise RuntimeError(f"Jina lexical score cache misses {missing} pool rows")
    orders = {
        qid: sorted(docs, key=lambda d: (-scores[qid][d], d))
        for qid, docs in pools.items()
    }
    return orders, dict(scores), integrity


def load_source_channel(channel: str, pools: dict[str, list[str]]):
    con = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    orders, scores = {}, {}
    for qid, payload in con.execute(
        "SELECT q,payload FROM sources WHERE source=?", (channel,)
    ):
        qid = str(qid)
        if qid not in pools:
            continue
        wanted = set(pools[qid])
        values = json.loads(payload)
        score_map = {
            str(row["doc_id"]): float(row["score"])
            for row in values if str(row["doc_id"]) in wanted
        }
        native_rank = {
            str(row["doc_id"]): int(row["rank"])
            for row in values if str(row["doc_id"]) in wanted
        }
        docs = pools[qid]
        orders[qid] = sorted(docs, key=lambda d: (native_rank.get(d, 10**9), d))
        scores[qid] = score_map
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    if set(orders) != set(pools):
        raise RuntimeError(f"source {channel} population mismatch")
    return orders, scores, integrity


def document_heads(pools: dict[str, list[str]]) -> dict[str, str]:
    needed = {d for docs in pools.values() for d in docs}
    result = {}
    for doc in needed:
        path = DATA / "selected-contexts" / f"context_{doc}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        result[doc] = str(row.get("passage") or "")[:3000]
    return result


def metadata_arrays(pools, questions, heads):
    # Reuse Huy's implementation exactly for parsing and feature semantics.
    import sys
    sys.path.insert(0, str(ROOT))
    from tune_citation_graph import cited_numbers, own_number
    from tune_doctype_features import TYPES, doc_type, question_type_hints

    types = {doc: doc_type(text) for doc, text in heads.items()}
    own = {doc: own_number(text) for doc, text in heads.items()}
    cited = {}
    for doc, text in heads.items():
        values = cited_numbers(text)
        values.discard(own[doc])
        cited[doc] = values

    doctype, citation = {}, {}
    for qid, docs in pools.items():
        counts = {name: 0 for name in TYPES}
        for doc in docs:
            counts[types[doc]] += 1
        hints = question_type_hints(questions[qid])
        drows, crows = [], []
        pool_numbers = {own[d] for d in docs if own[d]}
        for doc in docs:
            kind = types[doc]
            drows.append(
                [float(kind == name) for name in TYPES]
                + [counts[kind] / max(len(docs), 1)]
                + hints.tolist()
            )
            outgoing = cited[doc] & pool_numbers
            incoming = [other for other in docs if other != doc and own[doc] and own[doc] in cited[other]]
            crows.append([float(bool(outgoing)), float(len(outgoing)), float(bool(incoming)), float(len(incoming))])
        doctype[qid] = np.asarray(drows, dtype=np.float32)
        citation[qid] = np.asarray(crows, dtype=np.float32)
    return doctype, citation


def rank_columns(order_map, pools):
    out = {}
    for qid, docs in pools.items():
        ranks = {doc: i + 1 for i, doc in enumerate(order_map[qid])}
        values = np.asarray([ranks.get(doc, 60) for doc in docs], dtype=np.float32)
        out[qid] = np.column_stack((1.0 / (10.0 + values), values / 60.0)).astype(np.float32)
    return out


def score_columns(score_map, pools):
    out = {}
    for qid, docs in pools.items():
        raw = score_map[qid]
        values = np.asarray([raw.get(doc, np.nan) for doc in docs], dtype=np.float64)
        present = values[~np.isnan(values)]
        if present.size:
            mean = float(present.mean())
            std = float(present.std()) or 1.0
            top = float(present.max())
        else:
            mean, std, top = 0.0, 1.0, 0.0
        filled = np.where(np.isnan(values), mean - 2 * std, values)
        out[qid] = np.column_stack(((filled - mean) / std, (filled - top) / std)).astype(np.float32)
    return out


def make_rows(config, pools, rank_features, score_features, metadata):
    rank_names = config["rank_views"]
    score_names = config["score_channels"]
    meta_names = config["metadata"]
    rows = {}
    for qid, docs in pools.items():
        parts = []
        if rank_names:
            parts.extend(rank_features[name][qid] for name in rank_names)
            raw_ranks = np.column_stack([
                rank_features[name][qid][:, 1] * 60.0 for name in rank_names
            ])
            parts.append(np.column_stack((raw_ranks.min(axis=1), raw_ranks.mean(axis=1))).astype(np.float32))
        parts.extend(score_features[name][qid] for name in score_names)
        parts.extend(metadata[name][qid] for name in meta_names)
        if not parts:
            raise RuntimeError("empty feature configuration")
        rows[qid] = np.column_stack(parts).astype(np.float32)
    return rows


def metrics(predictions, golds, folds):
    values, precisions = [], []
    single, multi = [], []
    by_fold = {}
    for fold, qids in folds.items():
        fold_values = []
        for qid in qids:
            gold = golds[qid]
            hits = len(set(predictions[qid][:5]) & gold)
            value = hits / len(gold)
            values.append(value)
            precisions.append(hits / 5.0)
            fold_values.append(value)
            (single if len(gold) == 1 else multi).append(value)
        by_fold[fold] = float(np.mean(fold_values))
    return {
        "queries": len(values),
        "recall_at_5": float(np.mean(values)),
        "precision_at_5": float(np.mean(precisions)),
        "single_gold_recall_at_5": float(np.mean(single)),
        "multi_gold_recall_at_5": float(np.mean(multi)),
        "per_fold_recall_at_5": by_fold,
    }


def evaluate_config(name, config, pools, golds, folds, duplicate_exclusions,
                    rank_features, score_features, metadata):
    started = time.perf_counter()
    rows = make_rows(config, pools, rank_features, score_features, metadata)
    all_qids = set(pools)
    predictions = {}
    feature_count = next(iter(rows.values())).shape[1]
    fold_runtime = {}
    for fold, test_ids in folds.items():
        held = set(test_ids)
        blocked = set(map(str, duplicate_exclusions.get(fold, [])))
        train_ids = sorted(all_qids - held - blocked, key=int)
        x = np.vstack([rows[q] for q in train_ids])
        y = np.concatenate([
            np.asarray([doc in golds[q] for doc in pools[q]], dtype=np.int8)
            for q in train_ids
        ])
        fold_started = time.perf_counter()
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=2026,
        ).fit(scaler.transform(x), y)
        for qid in test_ids:
            values = model.decision_function(scaler.transform(rows[qid]))
            predictions[qid] = [
                pools[qid][i] for i in np.lexsort((np.asarray(pools[qid]), -values))
            ]
        fold_runtime[fold] = time.perf_counter() - fold_started
    result = {
        "name": name,
        "config": config,
        "feature_count": feature_count,
        "metrics": metrics(predictions, golds, folds),
        "runtime_seconds": time.perf_counter() - started,
        "fold_fit_score_seconds": fold_runtime,
    }
    return result, predictions


def compare(candidate, reference, golds, folds):
    wins = losses = ties = churn = crossings_in = crossings_out = 0
    fold_delta = {}
    for fold, qids in folds.items():
        deltas = []
        for qid in qids:
            gold = golds[qid]
            a, b = set(reference[qid][:5]), set(candidate[qid][:5])
            ra, rb = len(a & gold) / len(gold), len(b & gold) / len(gold)
            deltas.append(rb - ra)
            wins += rb > ra
            losses += rb < ra
            ties += rb == ra
            churn += a != b
            crossings_in += len((b & gold) - a)
            crossings_out += len((a & gold) - b)
        fold_delta[fold] = float(np.mean(deltas))
    return {
        "delta_recall_at_5": float(np.mean([
            len(set(candidate[q][:5]) & golds[q]) / len(golds[q])
            - len(set(reference[q][:5]) & golds[q]) / len(golds[q])
            for q in golds
        ])),
        "worst_fold_delta": min(fold_delta.values()),
        "per_fold_delta": fold_delta,
        "wins": wins, "losses": losses, "ties": ties,
        "top5_churn": churn,
        "gold_crossings_into_top5": crossings_in,
        "gold_crossings_out_of_top5": crossings_out,
    }


def model_complexity(config):
    models = set()
    for name in config["rank_views"] + config["score_channels"]:
        if name == "jina_ce":
            models.add("jinaai/jina-reranker-v2-base-multilingual")
        elif name in {"frozen_e5", "adapted_e5"}:
            models.add("mainguyen9/vietlegal-e5")
        elif name == "lal_native":
            models.add("darklethelong/vnlegal-lal")
        elif name == "legalir_jina":
            models.add("jinaai/jina-embeddings-v3")
    estimates = {
        "jinaai/jina-reranker-v2-base-multilingual": 278_000_000,
        "mainguyen9/vietlegal-e5": 560_000_000,
        "darklethelong/vnlegal-lal": 560_000_000,
        "jinaai/jina-embeddings-v3": 572_000_000,
    }
    return {
        "model_inference_count": len(models),
        "models": sorted(models),
        "approx_original_parameters": sum(estimates[x] for x in models),
        "score_channels": len(config["score_channels"]),
        "rank_views": len(config["rank_views"]),
        "ltr": True,
    }


def write_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    folds, pools, questions, golds, e5_orders, e5_scores, dup, pred_hashes = load_inputs()
    jina_order, jina_scores, jina_integrity = load_jina(pools)
    lal_order, lal_scores, source_integrity = load_source_channel("lal", pools)
    legalir_jina_order, legalir_jina_scores, _ = load_source_channel("jina", pools)

    heads = document_heads(pools)
    doctype, citation = metadata_arrays(pools, questions, heads)
    orders = {
        "jina_ce": jina_order,
        "frozen_e5": e5_orders["frozen_e5"],
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
        "legalir_jina": legalir_jina_order,
    }
    score_maps = {
        "jina_ce": jina_scores,
        "frozen_e5": e5_scores["frozen_e5"],
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
        "legalir_jina": legalir_jina_scores,
    }
    rank_features = {name: rank_columns(values, pools) for name, values in orders.items()}
    score_features = {name: score_columns(values, pools) for name, values in score_maps.items()}
    metadata = {"doctype": doctype, "citation": citation}

    exact_huy = {
        "rank_views": ["jina_ce"],
        "score_channels": ["jina_ce"],
        "metadata": ["doctype", "citation"],
    }
    variants = {
        "huy_exact_safe_available": exact_huy,
        "huy_no_citation": {**exact_huy, "metadata": ["doctype"]},
        "huy_no_doctype": {**exact_huy, "metadata": ["citation"]},
        "huy_neural_only": {**exact_huy, "metadata": []},
        "huy_plus_frozen_e5": {
            "rank_views": ["jina_ce", "frozen_e5"],
            "score_channels": ["jina_ce", "frozen_e5"],
            "metadata": ["doctype", "citation"],
        },
        "huy_replace_with_adapted_e5": {
            "rank_views": ["jina_ce", "adapted_e5"],
            "score_channels": ["jina_ce", "adapted_e5"],
            "metadata": ["doctype", "citation"],
        },
        "huy_both_e5": {
            "rank_views": ["jina_ce", "frozen_e5", "adapted_e5"],
            "score_channels": ["jina_ce", "frozen_e5", "adapted_e5"],
            "metadata": ["doctype", "citation"],
        },
        "huy_plus_frozen_e5_lal": {
            "rank_views": ["jina_ce", "frozen_e5", "lal_native"],
            "score_channels": ["jina_ce", "frozen_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "huy_plus_adapted_e5_lal": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "all_safe_available": {
            "rank_views": ["jina_ce", "frozen_e5", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "frozen_e5", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_jina": {
            "rank_views": ["adapted_e5", "lal_native"],
            "score_channels": ["adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_lal": {
            "rank_views": ["jina_ce", "adapted_e5"],
            "score_channels": ["jina_ce", "adapted_e5"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_doctype": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["citation"],
        },
        "best_no_citation": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["doctype"],
        },
        "best_neural_only": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": [],
        },
        "adapted_lal_neural_only": {
            "rank_views": ["adapted_e5", "lal_native"],
            "score_channels": ["adapted_e5", "lal_native"],
            "metadata": [],
        },
        "adapted_only_huy_ltr": {
            "rank_views": ["adapted_e5"],
            "score_channels": ["adapted_e5"],
            "metadata": ["doctype", "citation"],
        },
        "lal_only_huy_ltr": {
            "rank_views": ["lal_native"],
            "score_channels": ["lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_rank_only": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": [],
            "metadata": ["doctype", "citation"],
        },
        "best_score_only": {
            "rank_views": [],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_jina_score": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_e5_score": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_lal_score": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_jina_rank": {
            "rank_views": ["adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_e5_rank": {
            "rank_views": ["jina_ce", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_no_lal_rank": {
            "rank_views": ["jina_ce", "adapted_e5"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_plus_legalir_jina": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native", "legalir_jina"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native", "legalir_jina"],
            "metadata": ["doctype", "citation"],
        },
        "best_plus_legalir_jina_rank_only": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native", "legalir_jina"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
        "best_plus_legalir_jina_score_only": {
            "rank_views": ["jina_ce", "adapted_e5", "lal_native"],
            "score_channels": ["jina_ce", "adapted_e5", "lal_native", "legalir_jina"],
            "metadata": ["doctype", "citation"],
        },
        "replace_huy_jina_with_legalir_jina": {
            "rank_views": ["legalir_jina", "adapted_e5", "lal_native"],
            "score_channels": ["legalir_jina", "adapted_e5", "lal_native"],
            "metadata": ["doctype", "citation"],
        },
    }

    results, predictions = {}, {}
    for name, config in variants.items():
        print(json.dumps({"stage": "evaluate", "name": name}), flush=True)
        result, pred = evaluate_config(
            name, config, pools, golds, folds, dup,
            rank_features, score_features, metadata,
        )
        result["complexity"] = model_complexity(config)
        results[name], predictions[name] = result, pred
        print(json.dumps({"name": name, **result["metrics"], "runtime": result["runtime_seconds"]}), flush=True)

    baseline_name = "huy_exact_safe_available"
    for name in results:
        results[name]["paired_vs_huy_baseline"] = compare(
            predictions[name], predictions[baseline_name], golds, folds
        )
    best_name = max(results, key=lambda n: results[n]["metrics"]["recall_at_5"])
    for name in results:
        results[name]["paired_vs_best"] = compare(
            predictions[name], predictions[best_name], golds, folds
        )

    candidate_ceiling = float(np.mean([
        len(set(pools[q]) & golds[q]) / len(golds[q]) for q in pools
    ]))
    common = {
        "status": "COMPLETE_CACHE_ONLY_5FOLD_OOF",
        "population": {"queries": 6991, "parents": 8507},
        "protocol": {
            "folds": str(FOLDS_PATH),
            "folds_sha256": sha256(FOLDS_PATH),
            "candidate_pool": str(POOL_PATH),
            "candidate_pool_sha256": sha256(POOL_PATH),
            "held_fold_duplicate_training_exclusions": dup,
            "candidate_ceiling": candidate_ceiling,
            "ltr": "Huy StandardScaler + balanced liblinear LogisticRegression C=0.15 seed=2026",
        },
        "cache_integrity": {"jina_score_db": jina_integrity, "source_db": source_integrity},
        "prediction_input_hashes": pred_hashes,
        "limitations": {
            "reconstruction": "Exact Huy feature/LR/metadata contract on every complete safe Huy channel available under the V2 pool; not a claim that unavailable historical channels were reproduced.",
            "candidate_pool": "Sealed V2 E5@50 union BM25@10 pool, held fixed for all variants.",
            "unsafe_for_5fold": [
                "aiteamvn_ft: historical checkpoint/cache lacks clean V2 fold lineage",
                "jina_ft: historical checkpoint/cache lacks clean V2 fold lineage",
                "title_embed: only 599 evaluable V2 qids cached",
                "crossenc: only 899 evaluable V2 qids cached",
                "expanded/dense/corpus historical Huy views: only 899 evaluable V2 qids cached",
                "Huy CLS LAL interface: only 899 evaluable V2 qids cached",
            ],
        },
    }
    baseline = {
        "schema_version": "dsc2026.huy_fasttrack.5fold_baseline.v1",
        **common,
        "baseline": results[baseline_name],
        "available_component_screen": results,
        "best_screened_variant": best_name,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "HUY_5FOLD_BASELINE.json", baseline)

    # Greedy/Pareto is represented by every nested or ablated checkpoint above;
    # order by performance then complexity and mark nondominated points.
    points = []
    for name, result in results.items():
        c = result["complexity"]
        points.append({
            "name": name,
            "recall_at_5": result["metrics"]["recall_at_5"],
            "precision_at_5": result["metrics"]["precision_at_5"],
            "model_inference_count": c["model_inference_count"],
            "feature_count": result["feature_count"],
            "score_channels": c["score_channels"],
            "runtime_seconds": result["runtime_seconds"],
            "config": result["config"],
        })
    frontier = []
    for point in points:
        dominated = any(
            other["recall_at_5"] >= point["recall_at_5"]
            and other["model_inference_count"] <= point["model_inference_count"]
            and other["feature_count"] <= point["feature_count"]
            and (
                other["recall_at_5"] > point["recall_at_5"]
                or other["model_inference_count"] < point["model_inference_count"]
                or other["feature_count"] < point["feature_count"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda x: (-x["recall_at_5"], x["model_inference_count"], x["feature_count"]))
    best_recall = results[best_name]["metrics"]["recall_at_5"]
    def minimal_within(tol):
        eligible = [p for p in points if p["recall_at_5"] >= best_recall - tol]
        return min(eligible, key=lambda p: (p["model_inference_count"], p["feature_count"], -p["recall_at_5"]))
    pareto = {
        "schema_version": "dsc2026.huy_fasttrack.ablation_pareto.v1",
        **common,
        "full": next(p for p in points if p["name"] == "all_safe_available"),
        "best_absolute_score": next(p for p in points if p["name"] == best_name),
        "minimal_within_minus_0_001": minimal_within(.001),
        "minimal_within_minus_0_003": minimal_within(.003),
        "minimal_within_minus_0_005": minimal_within(.005),
        "pareto_frontier": frontier,
        "all_checkpoints": points,
    }
    write_json(OUT / "HUY_ABLATION_PARETO.json", pareto)

    pred_path = OUT / "BEST_5FOLD_PREDICTIONS.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(pools, key=int):
            f.write(json.dumps({"qid": qid, "top5": predictions[best_name][qid][:5]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "baseline": results[baseline_name]["metrics"],
        "best": best_name,
        "best_metrics": results[best_name]["metrics"],
        "candidate_ceiling": candidate_ceiling,
        "output": str(OUT),
        "runtime_seconds": time.perf_counter() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
