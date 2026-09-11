"""Independent integrity and failure audit for the downloaded Research V2 pilot.

This script is intentionally read-only with respect to the Kaggle input and
downloaded output.  It writes derived audit artifacts only to a separate
Research V2 audit namespace.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def recall(top5: list[str], gold: set[str]) -> float:
    return len(set(top5) & gold) / len(gold)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def notebook_audit(input_path: Path, output_path: Path) -> dict:
    before = read_json(input_path)
    after = read_json(output_path)
    cells = []
    for index, cell in enumerate(after["cells"]):
        source = "".join(cell.get("source", []))
        cells.append(
            {
                "index": index,
                "cell_type": cell.get("cell_type"),
                "execution_count": cell.get("execution_count"),
                "first_line": source.splitlines()[0] if source.splitlines() else "",
                "contains_full_launch_flag": "USER_REVIEWED_PILOT_AND_APPROVES_FULL" in source,
            }
        )
    executed = [c["index"] for c in cells if c["execution_count"] is not None]
    full_cells = [c for c in cells if c["contains_full_launch_flag"]]
    return {
        "input_cells": len(before["cells"]),
        "output_cells": len(after["cells"]),
        "executed_cell_indices": executed,
        "full_launch_cells": full_cells,
        "full_launch_cell_executed": any(c["execution_count"] is not None for c in full_cells),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    inp = args.input.resolve()
    out = args.output.resolve()
    audit_dir = args.audit_dir.resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    report = read_json(out / "PILOT_REPORT.json")
    metrics_reported = read_json(out / "pilot_fold_0/score/fold_0_METRICS.json")
    training = read_json(out / "pilot_fold_0/TRAINING_MANIFEST.json")
    group_manifest = read_json(inp / "V2_BOUNDARY_GROUPS_MANIFEST.json")
    input_manifest = read_json(inp / "KAGGLE_INPUT_MANIFEST.json")
    folds_doc = read_json(inp / "V2_FOLDS.json")
    folds = {name: {str(q) for q in qids} for name, qids in folds_doc["folds"].items()}
    fold_of = {qid: fold for fold, qids in folds.items() for qid in qids}

    # Verify every sealed top-level input file that is directly present.
    input_hash_checks = {}
    for rel, expected in input_manifest["files_sha256"].items():
        path = inp / rel
        if path.is_file():
            actual = sha256(path)
            input_hash_checks[rel] = {"expected": expected, "actual": actual, "match": actual == expected}

    output_hash_checks = {}
    output_expected = {
        "jina_v2_boundary_train_kaggle.py": report["artifacts"]["runner_sha256"],
        "pilot_fold_0/score/fold_0_PREDICTIONS.jsonl": report["artifacts"]["fold0_predictions_sha256"],
        "pilot_fold_0/score/fold_0_METRICS.json": report["artifacts"]["fold0_metrics_sha256"],
        "pilot_fold_0/TRAINING_MANIFEST.json": report["artifacts"]["training_manifest_sha256"],
    }
    for rel, expected in output_expected.items():
        actual = sha256(out / rel)
        output_hash_checks[rel] = {"expected": expected, "actual": actual, "match": actual == expected}

    adapter_dir = out / "pilot_fold_0/best/adapter"
    adapter_hash_checks = {}
    for name, expected in report["metrics"]["adapter_files_sha256"].items():
        actual = sha256(adapter_dir / name)
        adapter_hash_checks[name] = {"expected": expected, "actual": actual, "match": actual == expected}

    # Stream group validation; this independently checks the scientific labels.
    group_count = 0
    group_fold_counts = Counter()
    duplicate_qids: list[str] = []
    invalid_fold: list[str] = []
    pos_neg_overlap: list[str] = []
    wrong_negative_count: list[str] = []
    missing_passages: list[str] = []
    sibling_gold_negative: list[str] = []
    group_by_qid = {}
    for row in read_jsonl(inp / "V2_BOUNDARY_GROUPS.jsonl"):
        qid = str(row["qid"])
        group_count += 1
        if qid in group_by_qid:
            duplicate_qids.append(qid)
        group_by_qid[qid] = row
        group_fold_counts[row["fold"]] += 1
        if fold_of.get(qid) != row["fold"]:
            invalid_fold.append(qid)
        positives = {str(item["doc_id"]) for item in row["positives"]}
        negatives = {str(item["doc_id"]) for item in row["negatives"]}
        if positives & negatives:
            pos_neg_overlap.append(qid)
            sibling_gold_negative.append(qid)
        if len(row["negatives"]) != 4:
            wrong_negative_count.append(qid)
        if any(not item.get("passages") for item in row["positives"] + row["negatives"]):
            missing_passages.append(qid)

    # Reproduce the exact fold-0 train/validation split and pilot cap.
    held = "fold_0"
    forbidden = set(group_manifest["held_fold_duplicate_exclusions"][held])
    eligible = [
        row for qid, row in group_by_qid.items()
        if row["fold"] != held and qid not in forbidden
    ]
    valid = [row for row in eligible if int(hashlib.sha256(str(row["qid"]).encode()).hexdigest(), 16) % 10 == 0]
    train_all = [row for row in eligible if int(hashlib.sha256(str(row["qid"]).encode()).hexdigest(), 16) % 10 != 0]
    train_all.sort(key=lambda row: hashlib.sha256(str(row["qid"]).encode()).hexdigest())
    pilot_train = train_all[: training["train_groups"]]
    train_pairs = sum(len(row["positives"]) * len(row["negatives"]) for row in pilot_train)
    valid_pairs = sum(len(row["positives"]) * len(row["negatives"]) for row in valid)

    # Candidate pool is also the exact scoring population.
    pool_by_qid = {}
    pool_fold_counts = Counter()
    pool_duplicate_qids = []
    duplicate_docs = []
    for row in read_jsonl(inp / "V2_CANDIDATE_POOL.jsonl"):
        qid = str(row["qid"])
        if qid in pool_by_qid:
            pool_duplicate_qids.append(qid)
        pool_by_qid[qid] = row
        pool_fold_counts[row["fold"]] += 1
        if len(row["doc_ids"]) != len(set(map(str, row["doc_ids"]))):
            duplicate_docs.append(qid)

    # SQLite integrity and exact arm/cardinality checks.
    conn = sqlite3.connect(f"file:{(out / 'evidence_ab_scores.sqlite').as_posix()}?mode=ro", uri=True)
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    score_counts = {str(arm): int(count) for arm, count in conn.execute("SELECT arm,COUNT(*) FROM scores GROUP BY arm")}
    score_dup_count = int(conn.execute(
        "SELECT COUNT(*) FROM (SELECT arm,qid,doc_id,COUNT(*) c FROM scores GROUP BY arm,qid,doc_id HAVING c>1)"
    ).fetchone()[0])
    base_scores = defaultdict(dict)
    for qid, doc_id, score in conn.execute("SELECT qid,doc_id,score FROM scores WHERE arm='lexical'"):
        if str(qid) in folds[held]:
            base_scores[str(qid)][str(doc_id)] = float(score)
    conn.close()

    # Recalculate all reported metrics from immutable predictions.
    predictions = list(read_jsonl(out / "pilot_fold_0/score/fold_0_PREDICTIONS.jsonl"))
    pred_duplicate_qids = []
    seen = set()
    rec_base, rec_ft = [], []
    wins = losses = ties = changed = 0
    slice_stats = defaultdict(lambda: {"queries": 0, "base_sum": 0.0, "ft_sum": 0.0, "wins": 0, "losses": 0})
    jaccards = []
    replacements = []
    base_gold_ranks_win = []
    base_gold_ranks_loss = []
    base_margin_win = []
    base_margin_loss = []
    entered_gold_docs = 0
    exited_gold_docs = 0
    for row in predictions:
        qid = str(row["qid"])
        if qid in seen:
            pred_duplicate_qids.append(qid)
        seen.add(qid)
        gold = set(map(str, row["gold"]))
        base = list(map(str, row["base_top5"]))
        ft = list(map(str, row["ft_top5"]))
        rb, rf = recall(base, gold), recall(ft, gold)
        rec_base.append(rb)
        rec_ft.append(rf)
        key = "single" if len(gold) == 1 else "multi"
        stat = slice_stats[key]
        stat["queries"] += 1
        stat["base_sum"] += rb
        stat["ft_sum"] += rf
        if rf > rb:
            wins += 1
            stat["wins"] += 1
        elif rf < rb:
            losses += 1
            stat["losses"] += 1
        else:
            ties += 1
        sb, sf = set(base), set(ft)
        if sb != sf:
            changed += 1
        union = sb | sf
        jaccards.append(len(sb & sf) / len(union))
        replacements.append(len(sb - sf))
        entered_gold_docs += len((sf - sb) & gold)
        exited_gold_docs += len((sb - sf) & gold)

        scores = base_scores[qid]
        ranked = sorted(pool_by_qid[qid]["doc_ids"], key=lambda doc: (-scores[str(doc)], str(doc)))
        rank_of = {str(doc): rank + 1 for rank, doc in enumerate(ranked)}
        fifth = scores[str(ranked[4])]
        sixth = scores[str(ranked[5])] if len(ranked) > 5 else math.nan
        if rf > rb:
            for doc in gold - sb:
                if doc in sf:
                    base_gold_ranks_win.append(rank_of[doc])
                    base_margin_win.append(scores[doc] - fifth)
        elif rf < rb:
            for doc in gold & (sb - sf):
                base_gold_ranks_loss.append(rank_of[doc])
                base_margin_loss.append(scores[doc] - sixth)

    recomputed_slices = {}
    for key, stat in slice_stats.items():
        recomputed_slices[key] = {
            "queries": stat["queries"],
            "base_recall": stat["base_sum"] / stat["queries"],
            "ft_recall": stat["ft_sum"] / stat["queries"],
            "delta": (stat["ft_sum"] - stat["base_sum"]) / stat["queries"],
            "wins": stat["wins"],
            "losses": stat["losses"],
        }

    adapter_config = read_json(adapter_dir / "adapter_config.json")
    input_runner = (inp / "jina_v2_boundary_train.py").read_text(encoding="utf-8").splitlines()
    output_runner = (out / "jina_v2_boundary_train_kaggle.py").read_text(encoding="utf-8").splitlines()
    runner_diff = list(difflib.unified_diff(input_runner, output_runner, fromfile="sealed_input", tofile="executed_kaggle", lineterm=""))
    (audit_dir / "KAGGLE_RUNNER_PATCH.diff").write_text("\n".join(runner_diff) + "\n", encoding="utf-8")

    nb = notebook_audit(inp / "RESEARCH_V2_JINA_BOUNDARY_KAGGLE.ipynb", out / "research-v2-jina-boundary-kaggle.ipynb")
    expected_scored_qids = {qid for qid, row in pool_by_qid.items() if row["fold"] == held}
    all_integrity = (
        all(v["match"] for v in input_hash_checks.values())
        and all(v["match"] for v in output_hash_checks.values())
        and all(v["match"] for v in adapter_hash_checks.values())
        and integrity == "ok"
        and score_dup_count == 0
        and not duplicate_qids
        and not invalid_fold
        and not pos_neg_overlap
        and not wrong_negative_count
        and not missing_passages
        and not pool_duplicate_qids
        and not duplicate_docs
        and not pred_duplicate_qids
        and seen == expected_scored_qids
        and not nb["full_launch_cell_executed"]
        and report["full_five_fold_launched"] is False
    )

    audit = {
        "schema_version": "dsc2026.research_v2.kaggle_pilot_integrity_audit.v1",
        "status": "PASS_OFFICIAL_PILOT" if all_integrity else "FAIL_INTEGRITY",
        "read_only_sources": {"input": str(inp), "downloaded_output": str(out)},
        "hashes": {
            "input_files": input_hash_checks,
            "output_files": output_hash_checks,
            "adapter_files": adapter_hash_checks,
            "executed_notebook_sha256": sha256(out / "research-v2-jina-boundary-kaggle.ipynb"),
        },
        "fold_and_group_contract": {
            "groups": group_count,
            "group_fold_counts": dict(sorted(group_fold_counts.items())),
            "unique_group_qids": len(group_by_qid),
            "duplicate_qids": duplicate_qids,
            "invalid_fold_qids": invalid_fold,
            "positive_negative_overlap_qids": pos_neg_overlap,
            "sibling_gold_negative_qids": sibling_gold_negative,
            "wrong_negative_count_qids": wrong_negative_count,
            "missing_passage_qids": missing_passages,
            "fold0_reproduced": {
                "forbidden_duplicate_qids": sorted(forbidden),
                "eligible_nonheld_groups": len(eligible),
                "train_groups_before_cap": len(train_all),
                "pilot_train_groups": len(pilot_train),
                "valid_groups": len(valid),
                "train_pairs": train_pairs,
                "valid_pairs": valid_pairs,
                "matches_training_manifest": len(pilot_train) == training["train_groups"] and len(valid) == training["valid_groups"],
            },
        },
        "candidate_and_evidence_contract": {
            "candidate_policy": next(iter(pool_by_qid.values()))["candidate_policy"],
            "pool_queries": len(pool_by_qid),
            "pool_fold_counts": dict(sorted(pool_fold_counts.items())),
            "duplicate_pool_qids": pool_duplicate_qids,
            "duplicate_docs_within_query": duplicate_docs,
            "renderer": report["immutable_contract"]["renderer"],
            "parent_aggregation": report["immutable_contract"]["parent_aggregation"],
            "max_length": report["immutable_contract"]["max_length"],
            "sqlite_integrity": integrity,
            "sqlite_score_counts": score_counts,
            "sqlite_duplicate_keys": score_dup_count,
        },
        "training_contract": {
            "objective": "pairwise softplus(-(max_positive_passage_logit-max_negative_passage_logit)); every positive x every negative",
            "lora_rank": adapter_config["r"],
            "lora_alpha": adapter_config["lora_alpha"],
            "lora_dropout": adapter_config["lora_dropout"],
            "target_modules_count": len(adapter_config["target_modules"]),
            "target_modules": sorted(adapter_config["target_modules"]),
            "modules_to_save": adapter_config["modules_to_save"],
            "best_step": read_json(out / "pilot_fold_0/best/trainer_state.json")["step"],
            "resume_tensor_parity": report["resume_probe"]["tensor_parity"],
        },
        "execution": {
            "notebook": nb,
            "runner_input_sha256": sha256(inp / "jina_v2_boundary_train.py"),
            "runner_executed_sha256": sha256(out / "jina_v2_boundary_train_kaggle.py"),
            "runner_diff_lines": len(runner_diff),
            "patch_classification": "RUNTIME_COMPATIBILITY_PATCH_ONLY",
            "scientific_contract_change": False,
            "patch_summary": [
                "Resolve exact 24 Wqkv/out_proj module names to avoid PEFT suffix collision with classifier.out_proj.",
                "Replace unavailable enable_input_require_grads with an exact embedding-output requires_grad hook.",
                "Add tuple-safe LoRA forward dtype fallback for PEFT 0.11.1 while preserving the same LoRA delta computation.",
            ],
            "full_five_fold_launched": report["full_five_fold_launched"],
        },
        "independently_recomputed_metrics": {
            "queries": len(predictions),
            "unique_qids": len(seen),
            "prediction_qids_exact_evaluable_fold0": seen == expected_scored_qids,
            "non_evaluable_fold0_qids_not_scored": sorted(folds[held] - expected_scored_qids),
            "base_recall_at_5": mean(rec_base),
            "ft_recall_at_5": mean(rec_ft),
            "delta_recall_at_5": mean(rec_ft) - mean(rec_base),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "changed_top5_sets": changed,
            "slices": recomputed_slices,
            "matches_report": {
                "base_recall": abs(mean(rec_base) - metrics_reported["base_recall_at_5"]) < 1e-15,
                "ft_recall": abs(mean(rec_ft) - metrics_reported["ft_recall_at_5"]) < 1e-15,
                "wins_losses_ties": [wins, losses, ties] == [metrics_reported["wins"], metrics_reported["losses"], metrics_reported["ties"]],
                "changed_top5_sets": changed == metrics_reported["changed_top5_sets"],
            },
        },
    }
    (audit_dir / "KAGGLE_PILOT_INTEGRITY_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    forensic = {
        "schema_version": "dsc2026.research_v2.jina_boundary_failure_forensic.v1",
        "decision": "REJECTED_PILOT_STOP_FAMILY_WITHOUT_GRID",
        "exact_family": "Jina-v2 + locked lexical top-2 evidence + pairwise parent-max LoRA + deterministic Top-5",
        "reported_gate_status": report["status"],
        "gate_checks": report["gate_checks"],
        "effect": audit["independently_recomputed_metrics"],
        "top5_churn": {
            "changed_sets": changed,
            "mean_jaccard": mean(jaccards),
            "median_replaced_docs": median(replacements),
            "mean_replaced_docs": mean(replacements),
            "gold_docs_entered": entered_gold_docs,
            "gold_docs_exited": exited_gold_docs,
        },
        "boundary_movements_from_available_artifacts": {
            "winning_new_gold_base_ranks": {
                "count": len(base_gold_ranks_win),
                "median": median(base_gold_ranks_win) if base_gold_ranks_win else None,
                "ranks": dict(sorted(Counter(base_gold_ranks_win).items())),
            },
            "losing_gold_base_ranks": {
                "count": len(base_gold_ranks_loss),
                "median": median(base_gold_ranks_loss) if base_gold_ranks_loss else None,
                "ranks": dict(sorted(Counter(base_gold_ranks_loss).items())),
            },
            "winning_gold_base_score_minus_base_rank5": {
                "count": len(base_margin_win),
                "mean": mean(base_margin_win),
                "median": median(base_margin_win) if base_margin_win else None,
                "min": min(base_margin_win) if base_margin_win else None,
                "max": max(base_margin_win) if base_margin_win else None,
            },
            "losing_gold_base_score_minus_base_rank6": {
                "count": len(base_margin_loss),
                "mean": mean(base_margin_loss),
                "median": median(base_margin_loss) if base_margin_loss else None,
                "min": min(base_margin_loss) if base_margin_loss else None,
                "max": max(base_margin_loss) if base_margin_loss else None,
            },
            "ft_raw_margin_availability": "NOT_EMITTED_BY_EXECUTED_SCORER; no environment-drifted local rescore used as Kaggle evidence",
        },
        "causal_interpretation": {
            "primary": "PAIRWISE_SURROGATE_MISALIGNMENT_AND_ORDER_DESTRUCTION",
            "evidence": [
                "Held-fold Recall@5 and boundary-pair accuracy both fell, so the learned ordering did not transfer to the target boundary.",
                "Top-5 membership changed broadly while losses exceeded wins, indicating destructive reordering rather than a small calibration-only shift.",
                "Multi-gold Recall regressed more strongly than the permitted gate, consistent with a pairwise objective that does not preserve slate coverage.",
                "Training validation pair accuracy selected step 96, but held-fold boundary accuracy fell; this is direct surrogate/generalization misalignment.",
            ],
            "not_claimed": [
                "The result does not falsify all task-specific rerankers.",
                "It does not falsify a different architecture, objective, or evidence representation.",
                "It does not establish that raw Jina-v2 scores are unusable; the frozen base remains the comparator.",
            ],
        },
    }
    (audit_dir / "KAGGLE_PILOT_FAILURE_FORENSIC.json").write_text(
        json.dumps(forensic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": audit["status"],
        "queries": len(predictions),
        "delta": forensic["effect"]["delta_recall_at_5"],
        "wins": wins,
        "losses": losses,
        "changed": changed,
        "audit_dir": str(audit_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
