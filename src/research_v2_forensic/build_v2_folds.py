"""Seal the Research V2 7,000-query stratified five-fold laboratory split.

All 7,000 query ids receive fold membership.  Duplicate gold document ids are
canonicalised to their retained identity and empty-passage gold ids are removed,
matching LegalIR's audited preprocessing policy.  Queries left without a
retrievable gold document remain in the split artifact but are explicitly
quarantined from training losses and evaluation denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


SCHEMA = "dsc2026.research_v2.stratified_5fold.v1"
LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
SEED = 20260909


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def canonical_gold(train: dict, exclusions: list[dict]) -> dict[str, set[str]]:
    excluded = {str(row["doc_id"]): row for row in exclusions}
    if len(excluded) != len(exclusions):
        raise ValueError("duplicate exclusion doc id")
    result: dict[str, set[str]] = {}
    for qid, row in train.items():
        gold: set[str] = set()
        for raw_docid in row["answer"]:
            docid = str(raw_docid)
            item = excluded.get(docid)
            if item is None:
                gold.add(docid)
                continue
            reasons = set(map(str, item.get("reasons", [])))
            replacement = item.get("duplicate_retained_id")
            if "exact_duplicate_raw_passage" in reasons:
                if not replacement:
                    raise ValueError(f"missing retained alias for {docid}")
                gold.add(str(replacement))
            elif reasons != {"empty_passage"}:
                raise ValueError(f"unsupported exclusion for {docid}: {sorted(reasons)}")
        result[str(qid)] = gold
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path,
                        default=workspace / "LegalIR/public_test_dataset/train.json")
    parser.add_argument("--exclusions", type=Path, default=workspace /
                        "LegalIR/cache/final_preprocessed_v2/exclusions.json")
    parser.add_argument("--output", type=Path, default=root /
                        "results/research_v2_forensic/V2_FOLDS.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train = json.loads(args.train.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    if len(train) != 7000:
        raise ValueError(f"expected 7000 queries, found {len(train)}")
    answers = canonical_gold(train, exclusions)
    qids = sorted(answers, key=lambda value: (int(value) if value.isdigit() else value))
    strata = ["empty" if not answers[qid] else
              "3plus" if len(answers[qid]) >= 3 else str(len(answers[qid]))
              for qid in qids]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds: dict[str, list[str]] = {}
    per_fold: dict[str, dict[str, object]] = {}
    for index, (_, held) in enumerate(splitter.split(qids, strata)):
        name = f"fold_{index}"
        members = [qids[int(i)] for i in held]
        folds[name] = members
        counts: dict[str, int] = {}
        for qid in members:
            key = "empty" if not answers[qid] else "3plus" if len(answers[qid]) >= 3 else str(len(answers[qid]))
            counts[key] = counts.get(key, 0) + 1
        per_fold[name] = {
            "queries": len(members),
            "evaluable_queries": sum(bool(answers[qid]) for qid in members),
            "strata": counts,
        }

    flat = [qid for members in folds.values() for qid in members]
    if len(flat) != len(set(flat)) or set(flat) != set(qids):
        raise RuntimeError("fold partition is not an exact cover")
    non_evaluable = sorted(qid for qid in qids if not answers[qid])
    payload = {
        "schema_version": SCHEMA,
        "status": "SEALED",
        "seed": args.seed,
        "splitter": "sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)",
        "stratification": "canonical gold count buckets: empty, 1, 2, 3plus",
        "label_policy": LABEL_POLICY,
        "population": {
            "all_queries": len(qids),
            "evaluable_queries": len(qids) - len(non_evaluable),
            "non_evaluable_queries": len(non_evaluable),
            "non_evaluable_qids": non_evaluable,
        },
        "rules": {
            "fold_membership_covers_all_7000": True,
            "empty_canonical_gold_in_training_loss": False,
            "empty_canonical_gold_in_metrics": False,
            "held_fold_labels_visible_to_supervised_stage": False,
        },
        "input_sha256": {
            "train_json": sha256(args.train),
            "exclusions_json": sha256(args.exclusions),
        },
        "per_fold": per_fold,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    checksum = sha256(args.output)
    args.output.with_suffix(".sha256").write_text(
        f"{checksum}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({"output": str(args.output), "sha256": checksum,
                      "per_fold": per_fold}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

