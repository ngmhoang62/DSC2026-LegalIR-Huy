"""Materialize fold-safe boundary groups after the evidence renderer is sealed.

The output is an immutable Kaggle input: every canonical gold is represented as
a positive parent and no sibling gold can enter the negative list.  Negatives
come from the frozen base-Jina ordering, primarily ranks 4--20.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from run_evidence_contract_ab import ChunkReader, canonical_answers, canonical_json, read_jsonl, sha256


def linked_duplicates(baseline: dict) -> set[tuple[str, str]]:
    links: set[tuple[str, str]] = set()
    duplicate = baseline["duplicate_contamination"]
    for row in duplicate["exact_normalized"]["examples"]:
        qids = [str(x) for x in row["qids"]]
        for a in qids:
            for b in qids:
                if a != b:
                    links.add((a, b))
    for row in duplicate["near_duplicate_char_tfidf"]["examples"]:
        a, b = str(row["qid_a"]), str(row["qid_b"])
        links.add((a, b)); links.add((b, a))
    return links


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--score-db", type=Path, required=True)
    p.add_argument("--renderer-lock", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--historical-candidates", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--exclusions", type=Path, required=True)
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--chunk-lookup", type=Path, required=True)
    p.add_argument("--doc-to-chunks", type=Path, required=True)
    p.add_argument("--huy-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--negatives", type=int, default=4)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lock = json.loads(args.renderer_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "SEALED" or lock.get("winner") not in {"lexical", "structural"}:
        raise RuntimeError("evidence renderer is not sealed")
    renderer = str(lock["winner"])
    train = json.loads(args.train.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    answers = canonical_answers(train, exclusions)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    links = linked_duplicates(baseline)
    fold_for = {str(row["qid"]): str(row["fold"]) for row in read_jsonl(args.pool)}

    sys.path.insert(0, str(args.huy_root.resolve()))
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import DocumentStore
    docs = DocumentStore(sorted(args.contexts.glob("context_*.json")), cache_size=9000)
    selection = {str(row["qid"]): row for row in read_jsonl(args.selection)}
    historical_e5 = {}
    for row in read_jsonl(args.historical_candidates):
        qid = str(row["qid"])
        historical_e5[qid] = {
            str(candidate["doc_id"]): str(candidate["sources"]["e5"]["evidence"][0]["chunk_id"])
            for candidate in row["candidates"]
            if candidate.get("sources", {}).get("e5", {}).get("evidence")
        }
    doc_to_chunks = json.loads(args.doc_to_chunks.read_text(encoding="utf-8"))
    reader = ChunkReader(args.chunks, args.chunk_lookup)

    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro", uri=True)
    score_rows = db.execute("SELECT qid,doc_id,score FROM scores WHERE arm=?", (renderer,)).fetchall()
    db.close()
    scores = {(str(q), str(d)): float(s) for q, d, s in score_rows}

    output = args.output_dir / "V2_BOUNDARY_GROUPS.jsonl"
    counts = Counter()
    gold_outside_pool = 0
    fallback_positive = 0
    try:
        with output.open("w", encoding="utf-8", newline="\n") as sink:
            for row in read_jsonl(args.pool):
                qid = str(row["qid"])
                query = str(row["query"])
                candidates = [str(x) for x in row["doc_ids"]]
                gold = sorted(answers[qid])
                if not gold:
                    continue
                ordered = sorted(candidates, key=lambda d: (-scores[(qid, d)], d))
                band = [d for d in ordered[3:20] if d not in set(gold)]
                fillers = [d for d in ordered if d not in set(gold) and d not in set(band)]
                negatives = (band + fillers)[:args.negatives]
                if any(d in set(gold) for d in negatives):
                    raise AssertionError("sibling gold entered negative pool")

                selected = selection[qid]
                selected_map = dict(zip(map(str, selected["doc_ids"]), map(str, selected["chunk_ids"])))
                needed = list(dict.fromkeys(gold + negatives))
                if renderer == "structural":
                    chosen = {}
                    for doc in needed:
                        chunk = selected_map.get(doc) or historical_e5.get(qid, {}).get(doc)
                        if chunk is None:
                            fallback = doc_to_chunks.get(doc, [])
                            if not fallback:
                                raise RuntimeError(f"no structural chunk for parent {doc}")
                            chunk = str(fallback[0])
                            fallback_positive += int(doc in set(gold))
                        chosen[doc] = chunk
                    records = reader.load(list(chosen.values()))
                    evidence = {doc: [str(records[chunk]["retrieval_text"])]
                                for doc, chunk in chosen.items()}
                else:
                    evidence = {doc: list(top_passages(query, docs[doc], count=2)) for doc in needed}

                positives = [{"doc_id": d, "passages": evidence[d]} for d in gold]
                neg_rows = [{"doc_id": d, "passages": evidence[d],
                             "base_rank": ordered.index(d) + 1,
                             "base_score": scores[(qid, d)]} for d in negatives]
                gold_outside_pool += sum(d not in set(candidates) for d in gold)
                sink.write(canonical_json({
                    "qid": qid, "fold": fold_for[qid], "query": query,
                    "positives": positives, "negatives": neg_rows,
                }) + "\n")
                counts[fold_for[qid]] += 1
    finally:
        reader.close()

    held_exclusions: dict[str, list[str]] = defaultdict(list)
    for held_fold in sorted(set(fold_for.values())):
        held = {q for q, f in fold_for.items() if f == held_fold}
        held_exclusions[held_fold] = sorted({train_q for train_q, held_q in links
                                             if held_q in held and fold_for.get(train_q) != held_fold})
    manifest = {
        "schema_version": "dsc2026.research_v2.boundary_groups.v1",
        "status": "SEALED",
        "renderer": renderer,
        "policy": {
            "positives": "all canonical gold parents, including out-of-pool gold",
            "negatives": f"first {args.negatives} non-gold parents, prioritizing frozen base-Jina ranks 4-20",
            "multi_gold_safety": "all sibling gold excluded before negative selection",
            "validation": "deterministic qid SHA256 bucket 0 mod 10 inside the four non-held folds",
            "duplicate_safety": "exclude exact/near-duplicate-linked training qids for each held fold",
        },
        "groups": sum(counts.values()), "per_fold": dict(counts),
        "gold_occurrences_outside_locked_pool": gold_outside_pool,
        "structural_first_chunk_positive_fallbacks": fallback_positive,
        "held_fold_duplicate_exclusions": held_exclusions,
        "inputs_sha256": {
            "pool": sha256(args.pool), "score_db": sha256(args.score_db),
            "renderer_lock": sha256(args.renderer_lock), "selection": sha256(args.selection),
            "historical_candidates": sha256(args.historical_candidates),
            "train": sha256(args.train), "exclusions": sha256(args.exclusions),
            "chunks": sha256(args.chunks), "doc_to_chunks": sha256(args.doc_to_chunks),
        },
        "artifact_sha256": sha256(output),
    }
    (args.output_dir / "V2_BOUNDARY_GROUPS_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output),
                      "renderer": renderer, "groups": manifest["groups"],
                      "sha256": manifest["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
