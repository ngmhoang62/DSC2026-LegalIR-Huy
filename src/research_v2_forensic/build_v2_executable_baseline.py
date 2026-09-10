"""Seal Research V2 candidates and audit retrieval/duplicate provenance.

This consumes immutable LegalIR artifacts but writes only to the Research V2
namespace.  It does not retrieve again or alter the historical candidate file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


DEPTHS = (5, 20, 50, 100, 150)
LOCK_E5 = 50
LOCK_BM25 = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def norm(text: str, *, accents: bool) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    if not accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def canonical_answers(train: dict, exclusions: list[dict]) -> dict[str, set[str]]:
    alias = {str(row["doc_id"]): str(row["duplicate_retained_id"])
             for row in exclusions if row.get("duplicate_retained_id")}
    excluded = {str(row["doc_id"]) for row in exclusions
                if "empty_passage" in row.get("reasons", [])}
    out = {}
    for qid, row in train.items():
        gold = {alias.get(str(x), str(x)) for x in row["answer"]}
        out[str(qid)] = gold - excluded
    return out


def recall(ranking: list[str], gold: set[str], k: int | None = None) -> float:
    if not gold:
        return float("nan")
    docs = ranking if k is None else ranking[:k]
    return len(set(docs) & gold) / len(gold)


def duplicate_audit(train: dict, fold_for: dict[str, str], answers: dict[str, set[str]]) -> dict:
    queries = {str(q): str(row["question"]) for q, row in train.items()}
    result: dict[str, object] = {}
    for label, accents in (("normalized", True), ("accentless", False)):
        groups: dict[str, list[str]] = defaultdict(list)
        for qid, text in queries.items():
            groups[norm(text, accents=accents)].append(qid)
        dup = [ids for ids in groups.values() if len(ids) > 1]
        result[f"exact_{label}"] = {
            "groups": len(dup),
            "queries": sum(map(len, dup)),
            "cross_fold_groups": sum(len({fold_for[q] for q in ids}) > 1 for ids in dup),
            "examples": [
                {"qids": ids, "folds": [fold_for[q] for q in ids],
                 "question": queries[ids[0]],
                 "canonical_answers": [sorted(answers[q]) for q in ids],
                 "same_canonical_answers": len({tuple(sorted(answers[q])) for q in ids}) == 1}
                for ids in dup[:25]
            ],
        }

    qids = sorted(queries, key=int)
    texts = [norm(queries[q], accents=False) for q in qids]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, dtype=np.float32).fit_transform(texts)
    distances, indices = NearestNeighbors(n_neighbors=6, metric="cosine",
                                          algorithm="brute", n_jobs=-1).fit(matrix).kneighbors(matrix)
    pairs = {}
    for i, qid in enumerate(qids):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            other = qids[int(j)]
            key = tuple(sorted((qid, other), key=int))
            similarity = 1.0 - float(distance)
            if similarity >= 0.92 and fold_for[qid] != fold_for[other]:
                pairs[key] = max(similarity, pairs.get(key, -1.0))
    ordered = sorted(pairs.items(), key=lambda item: (-item[1], int(item[0][0]), int(item[0][1])))
    result["near_duplicate_char_tfidf"] = {
        "definition": "accentless char_wb TF-IDF 3-5 grams; cosine >= 0.92; nearest 5; cross-fold only",
        "pairs": len(ordered),
        "queries": len({q for pair, _ in ordered for q in pair}),
        "examples": [
            {"qid_a": a, "fold_a": fold_for[a], "qid_b": b, "fold_b": fold_for[b],
             "similarity": score, "question_a": queries[a], "question_b": queries[b],
             "answers_a": sorted(answers[a]), "answers_b": sorted(answers[b]),
             "answer_jaccard": (len(answers[a] & answers[b]) / len(answers[a] | answers[b])
                                if answers[a] | answers[b] else None)}
            for (a, b), score in ordered[:50]
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    folds_payload = json.loads(args.folds.read_text(encoding="utf-8"))
    assert folds_payload["status"] == "SEALED"
    fold_for = {str(q): fold for fold, ids in folds_payload["folds"].items() for q in ids}
    train = json.loads(args.train.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    answers = canonical_answers(train, exclusions)
    evaluable = {q for q, gold in answers.items() if gold}
    if len(train) != 7000 or len(evaluable) != 6991 or set(train) != set(fold_for):
        raise RuntimeError("V2 population contract mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "V2_CANDIDATE_POOL.jsonl"
    sums = {"e5": {str(k): 0.0 for k in DEPTHS},
            "bm25": {str(k): 0.0 for k in DEPTHS},
            "set_union": {str(k): 0.0 for k in DEPTHS}}
    fold_sums = {fold: {"locked_pool": 0.0, "n": 0} for fold in folds_payload["folds"]}
    locked_sum = 0.0
    count = 0
    seen = set()
    pool_sizes = []
    with args.candidates.open("r", encoding="utf-8") as source, pool_path.open("w", encoding="utf-8", newline="\n") as sink:
        for line in source:
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in seen:
                raise RuntimeError(f"duplicate candidate qid {qid}")
            seen.add(qid)
            if qid not in evaluable:
                continue
            by_e5, by_bm25 = {}, {}
            for cand in row["candidates"]:
                doc = str(cand["doc_id"])
                sources = cand.get("sources", {})
                if "e5" in sources:
                    by_e5[int(sources["e5"]["rank"])] = doc
                if "bm25" in sources:
                    by_bm25[int(sources["bm25"]["rank"])] = doc
            e5 = [by_e5[r] for r in sorted(by_e5)]
            bm25 = [by_bm25[r] for r in sorted(by_bm25)]
            gold = answers[qid]
            for k in DEPTHS:
                sums["e5"][str(k)] += recall(e5, gold, k)
                sums["bm25"][str(k)] += recall(bm25, gold, k)
                union = list(dict.fromkeys(e5[:k] + bm25[:k]))
                sums["set_union"][str(k)] += recall(union, gold)
            locked = list(dict.fromkeys(e5[:LOCK_E5] + bm25[:LOCK_BM25]))
            locked_value = recall(locked, gold)
            locked_sum += locked_value
            fold_sums[fold_for[qid]]["locked_pool"] += locked_value
            fold_sums[fold_for[qid]]["n"] += 1
            count += 1
            pool_sizes.append(len(locked))
            sink.write(canonical_json({
                "qid": qid, "fold": fold_for[qid], "query": row["query"],
                "candidate_policy": f"e5@{LOCK_E5}_union_bm25@{LOCK_BM25}",
                "doc_ids": locked,
            }) + "\n")

    if seen != set(train) or count != 6991:
        raise RuntimeError("candidate coverage mismatch")
    metrics = {source: {f"recall@{k}": total / count for k, total in values.items()}
               for source, values in sums.items()}
    report = {
        "schema_version": "dsc2026.research_v2.executable_baseline.v1",
        "status": "SEALED",
        "population": {"all": 7000, "evaluable": count, "canonical_parents": 8507},
        "input_sha256": {
            "v2_folds": sha256(args.folds), "train": sha256(args.train),
            "exclusions": sha256(args.exclusions), "historical_candidates": sha256(args.candidates),
            "historical_candidate_manifest": sha256(args.source_manifest),
        },
        "retrieval_metrics_query_macro": metrics,
        "union_ceiling_definition": "set union of available E5@150 and BM25@50",
        "union_ceiling": metrics["set_union"]["recall@150"],
        "locked_pool": {
            "policy": f"E5@{LOCK_E5} set-union BM25@{LOCK_BM25}, E5 order then novel BM25 order",
            "recall_ceiling": locked_sum / count,
            "size": {"min": min(pool_sizes), "mean": sum(pool_sizes) / len(pool_sizes), "max": max(pool_sizes)},
            "per_fold": {fold: values["locked_pool"] / values["n"] for fold, values in fold_sums.items()},
            "path": str(pool_path),
        },
        "duplicate_contamination": duplicate_audit(train, fold_for, answers),
        "provenance_notes": {
            "e5": "Frozen mainguyen9/vietlegal-e5; query:/passage: prefixes; 512 tokens; 1024 dimensions; no DSC task fine-tuning in this lineage.",
            "bm25": "Each query uses the nested-OOF EXP-021 configuration selected without that query's old-fold labels. Re-grouping rows into V2 folds does not make the per-query retrieval in-sample.",
            "adaptive_reuse_caveat": "The fixed E5/BM25 family and quotas were historically developed on this 7k population; treat the sealed V2 split as comparative OOF for learned stages, not as a pristine untouched benchmark for candidate-policy invention.",
        },
    }
    report_path = args.output_dir / "V2_EXECUTABLE_BASELINE.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "dsc2026.research_v2.executable_baseline_manifest.v1",
        "status": "SEALED", "report_sha256": sha256(report_path),
        "candidate_pool_sha256": sha256(pool_path), "candidate_rows": count,
    }
    (args.output_dir / "V2_EXECUTABLE_BASELINE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "manifest": manifest,
                      "locked_recall": report["locked_pool"]["recall_ceiling"],
                      "union_ceiling": report["union_ceiling"]}, indent=2))


if __name__ == "__main__":
    main()
