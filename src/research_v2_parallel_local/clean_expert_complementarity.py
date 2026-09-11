"""Cache-only clean expert complementarity audit on the exact V2 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED = {
    "pool": "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277",
    "folds": "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19",
    "train": "c39cde9e74977e350f1456e7d487aafe67d2bcbaa4fa26fcabd557fe635635b7",
    "exclusions": "d10aef3d891746cd9f874b32e9cb20940cb4fef0928c0fe239fdb7ca16337616",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def answers(train: dict, exclusions: list[dict]) -> dict[str, set[str]]:
    alias = {str(x["doc_id"]): str(x["duplicate_retained_id"])
             for x in exclusions if x.get("duplicate_retained_id")}
    empty = {str(x["doc_id"]) for x in exclusions if "empty_passage" in x.get("reasons", [])}
    return {str(q): {alias.get(str(d), str(d)) for d in row["answer"]} - empty
            for q, row in train.items()}


def recall(top: list[str], gold: set[str]) -> float:
    return len(set(top[:5]) & gold) / len(gold)


def main(args: argparse.Namespace) -> None:
    for path, key in ((args.pool, "pool"), (args.folds, "folds"),
                      (args.train, "train"), (args.exclusions, "exclusions")):
        actual = sha256(path)
        if actual != EXPECTED[key]:
            raise RuntimeError(f"{key} SHA mismatch: {actual}")
    golds = answers(json.loads(args.train.read_text(encoding="utf-8")),
                    json.loads(args.exclusions.read_text(encoding="utf-8")))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    fold_for = {str(q): f for f, ids in folds.items() for q in ids}
    source = {str(r["qid"]): r for r in read_jsonl(args.tail_pools)}
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro", uri=True)
    jina_scores = {(str(q), str(d)): float(s) for q, d, s in
                   db.execute("SELECT qid,doc_id,score FROM scores WHERE arm='lexical'")}
    db.close()

    ranked = {"jina_v2_base": {}, "vietlegal_e5": {}, "bm25_oof": {}}
    for row in read_jsonl(args.pool):
        qid, docs = str(row["qid"]), [str(x) for x in row["doc_ids"]]
        meta = source[qid]["source_meta"]
        ranked["jina_v2_base"][qid] = sorted(docs, key=lambda d: (-jina_scores[(qid, d)], d))[:5]
        ranked["vietlegal_e5"][qid] = sorted(
            docs, key=lambda d: (meta.get(d, {}).get("e5_rank", 10**9), d))[:5]
        ranked["bm25_oof"][qid] = sorted(
            docs, key=lambda d: (meta.get(d, {}).get("bm25_rank", 10**9), d))[:5]

    qids = sorted(ranked["jina_v2_base"], key=int)
    base = ranked["jina_v2_base"]
    summaries = {}
    for name, table in ranked.items():
        per_q = {q: recall(table[q], golds[q]) for q in qids}
        base_q = {q: recall(base[q], golds[q]) for q in qids}
        delta = {q: per_q[q] - base_q[q] for q in qids}
        union_q = {q: len((set(table[q]) | set(base[q])) & golds[q]) / len(golds[q]) for q in qids}
        rescue_q = {q: len((set(table[q]) - set(base[q])) & golds[q]) / len(golds[q]) for q in qids}
        summaries[name] = {
            "standalone_recall_at_5": float(np.mean(list(per_q.values()))),
            "per_fold_recall": {f: float(np.mean([per_q[q] for q in qids if fold_for[q] == f])) for f in folds},
            "single_gold_recall": float(np.mean([per_q[q] for q in qids if len(golds[q]) == 1])),
            "multi_gold_recall": float(np.mean([per_q[q] for q in qids if len(golds[q]) > 1])),
            "paired_vs_jina": {"delta": float(np.mean(list(delta.values()))),
                               "wins": sum(x > 0 for x in delta.values()),
                               "losses": sum(x < 0 for x in delta.values()),
                               "ties": sum(x == 0 for x in delta.values())},
            "top5_set_union_with_jina_oracle": float(np.mean(list(union_q.values()))),
            "union_increment_over_jina": float(np.mean([union_q[q] - base_q[q] for q in qids])),
            "exclusive_rescue_mass_over_jina": float(np.mean(list(rescue_q.values()))),
            "queries_rescuing_any_jina_miss": sum(x > 0 for x in rescue_q.values()),
            "multi_gold_exclusive_rescue_mass": float(np.mean([rescue_q[q] for q in qids if len(golds[q]) > 1])),
            "changed_top5_sets_vs_jina": sum(set(table[q]) != set(base[q]) for q in qids),
        }

    all_union = {q: set().union(*(set(ranked[name][q]) for name in ranked)) for q in qids}
    combined = float(np.mean([len(all_union[q] & golds[q]) / len(golds[q]) for q in qids]))
    eligible = [(name, info["union_increment_over_jina"])
                for name, info in summaries.items() if name != "jina_v2_base"
                and info["union_increment_over_jina"] >= .005]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    report = {
        "schema_version": "dsc2026.research_v2.clean_expert_complementarity.v1",
        "status": "COMPLETE_CACHE_ONLY_NO_TRAINING",
        "scope": "Exact immutable V2 candidate membership; source rank experts and locked lexical base-Jina scores.",
        "experts": summaries,
        "three_expert_top5_set_union_oracle": combined,
        "ranked_next_hypothesis_inputs": [{"expert": n, "jina_union_increment": v} for n, v in eligible[:2]],
        "interpretation": "Set-union oracle may contain up to 10 or 15 documents and is diagnostic only; it is not a valid Top-5 system.",
        "provenance": {
            "jina_v2_base": "Frozen pretrained HF revision 9cfeff2; locked lexical evidence; no V2 label training.",
            "vietlegal_e5": "Frozen mainguyen9/vietlegal-e5 source ranks from EXP-021; no DSC task fine-tuning in this lineage.",
            "bm25_oof": "Existing EXP-021/022 per-query OOF sparse rankings; no rescoring or label fitting in this audit.",
            "skipped": [
                "Jina-v3.5 and BGE are exclusion-registry families.",
                "EXP-031 GTE title-base cache covers only bounded 64-query screens per historical fold, not all exact V2 rows.",
                "Huy title/Jina-FT/AITeam artifacts do not provide clean matched 6991-query V2 score coverage; rescoring was intentionally not opened."
            ]
        },
        "inputs_sha256": {"pool": sha256(args.pool), "tail_pools": sha256(args.tail_pools),
                          "score_db": sha256(args.score_db), "folds": sha256(args.folds)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "CLEAN_EXPERT_COMPLEMENTARITY_AUDIT.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("results/research_v2_parallel_local"))
    p.add_argument("--pool", type=Path, default=Path("results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"))
    p.add_argument("--tail-pools", type=Path, default=Path("results/research_v2_parallel_local/V2_CANDIDATE_TAIL_POOLS.jsonl"))
    p.add_argument("--score-db", type=Path, default=Path("cache/research_v2_forensic/evidence_ab_scores.sqlite"))
    p.add_argument("--folds", type=Path, default=Path("results/research_v2_forensic/V2_FOLDS.json"))
    p.add_argument("--train", type=Path, default=Path("../LegalIR/public_test_dataset/train.json"))
    p.add_argument("--exclusions", type=Path, default=Path("../LegalIR/cache/final_preprocessed_v2/exclusions.json"))
    main(p.parse_args())
