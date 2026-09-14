"""Variant A: Conservative Explicit-Relation Resolver (RABR_SAFE).

Deterministic ultra-conservative resolver:
- Ranks 1-4 immutable
- At most one action per query
- Safe rule A: duplicate old/new in Top-5 -> remove old document unless query cites it
- Safe rule B: challenger 6-8 explicitly replaces/repeals rank-5 defender -> promote to slot 5 unless query cites old doc
Outputs: RABR_SAFE_REPORT.json, RABR_SAFE_PREDICTIONS.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
BASE_SNAPSHOT_DIR = CURRENT_DIR / "baseline_snapshot"
sys.path.insert(0, str(BASE_SNAPSHOT_DIR))

import run_huy_5fold_fasttrack as core

RESULTS_DIR = REPO_ROOT / "results/gemini/rabr_v1"
CACHE_DIR = RESULTS_DIR / "cache"
GRAPH_FILE = RESULTS_DIR / "RELATION_GRAPH.jsonl"
PREDICTIONS_FILE = CACHE_DIR / "BASELINE_PREDICTIONS_AND_SCORES.jsonl"
META_FILE = CACHE_DIR / "DOCUMENT_METADATA.json"


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text.replace("đ", "d").replace("Đ", "D")


def query_mentions_doc(query: str, doc_meta: Dict[str, Any]) -> bool:
    if not query or not doc_meta:
        return False
    q_norm = strip_accents(query).upper()
    q_norm_clean = re.sub(r"[\s\-_]+", " ", q_norm)

    ref = doc_meta.get("official_number")
    if ref:
        ref_norm = strip_accents(ref).upper()
        ref_clean = re.sub(r"[\s\-_]+", " ", ref_norm)
        # Check full ref match or number segment match
        if ref_clean in q_norm_clean:
            return True
        # Check number/year match e.g. "78/2015"
        m_ny = re.search(r"(\d+\/\d{4})", ref)
        if m_ny and m_ny.group(1) in q_norm:
            return True

    year = doc_meta.get("year")
    if year and 1945 <= year <= 2026:
        # Check explicit year in query e.g. "năm 2015", "2015"
        if re.search(rf"\b{year}\b", q_norm):
            return True

    return False


def main():
    started = time.perf_counter()
    print("=== Step 6: Variant A (RABR_SAFE) ===", flush=True)

    folds, pools, questions, golds, _, _, _, _ = core.load_inputs()

    # Load metadata
    with META_FILE.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load relation graph: store REPLACES and REPEALS edges
    replaces_edges = set()  # (newer, older)
    repeals_edges = set()   # (repealing, repealed)

    with GRAPH_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src, dst, rel = str(row["src_doc"]), str(row["dst_doc"]), row["relation_type"]
            if rel == "REPLACES":
                replaces_edges.add((src, dst))
            elif rel == "REPEALS":
                repeals_edges.add((src, dst))

    supersedes_edges = replaces_edges | repeals_edges

    # Load baseline predictions
    base_preds = {}
    with PREDICTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            base_preds[str(row["qid"])] = row["order"]

    safe_preds = {}
    actions = []

    for qid in sorted(base_preds, key=int):
        orig_order = list(base_preds[qid])
        query = questions[qid]
        top4 = orig_order[:4]  # Immutable!
        defender = orig_order[4]  # Rank 5
        challengers = orig_order[5:8]  # Ranks 6, 7, 8

        modified = False
        action_detail = None

        # Safe Rule A: Duplicate old/new inside current Top-5
        top5 = orig_order[:5]
        # Check if any doc in top5 is superseded by another doc in top5
        for i, doc_new in enumerate(top5):
            for j, doc_old in enumerate(top5):
                if i != j and (doc_new, doc_old) in supersedes_edges:
                    # doc_old is superseded by doc_new in top5!
                    # Check query mentions doc_old
                    if not query_mentions_doc(query, meta.get(doc_old, {})):
                        # Remove doc_old from top5, preserve ranks 1-4
                        # If doc_old is among top4, can we remove it?
                        # RULE: "Ranks 1-4 are immutable."
                        # "If current Top-5 contains BOTH: an old doc and doc that replaces it... remove old doc from Top-5"
                        # Wait, if doc_old is rank 5, removing it is safe!
                        # If doc_old is in rank 1-4, removing it would change ranks 1-4!
                        # The specification strictly states: "Ranks 1-4 are immutable."
                        # "Freeze ranks 1-4 of the current best Huy-fasttrack output. Only decide which document among current ranks 5-8 should occupy slot 5."
                        # Therefore, if doc_old is rank 5 (defender), it is removed and replaced by the highest challenger not superseded.
                        if j == 4:
                            # Defender is superseded by someone in Top-4!
                            # Pick highest challenger from 5:8 not superseded
                            replacement = None
                            for c in orig_order[5:]:
                                if not any((t, c) in supersedes_edges for t in top4):
                                    replacement = c
                                    break
                            if replacement and replacement != defender:
                                new_order = list(top4) + [replacement] + [doc for doc in orig_order[4:] if doc != replacement]
                                safe_preds[qid] = new_order
                                modified = True
                                action_detail = {
                                    "qid": qid,
                                    "rule": "SAFE_RULE_A_DEFENDER_SUPERSEDED_BY_TOP4",
                                    "old_doc": defender,
                                    "new_doc": replacement,
                                    "superseding_top4_doc": doc_new,
                                    "orig_defender_rank": 5,
                                    "replacement_orig_rank": orig_order.index(replacement) + 1,
                                }
                                break
            if modified:
                break

        # Safe Rule B: Challenger in 6, 7, 8 explicitly replaces/repeals rank-5 defender
        if not modified:
            for c_idx, c in enumerate(challengers):
                if (c, defender) in supersedes_edges:
                    # Challenger c replaces/repeals defender!
                    if not query_mentions_doc(query, meta.get(defender, {})):
                        # Promote challenger c to slot 5!
                        new_order = list(top4) + [c] + [doc for doc in orig_order[4:] if doc != c]
                        safe_preds[qid] = new_order
                        modified = True
                        action_detail = {
                            "qid": qid,
                            "rule": "SAFE_RULE_B_CHALLENGER_REPLACES_DEFENDER",
                            "old_doc": defender,
                            "new_doc": c,
                            "orig_defender_rank": 5,
                            "replacement_orig_rank": orig_order.index(c) + 1,
                        }
                        break

        if not modified:
            safe_preds[qid] = orig_order
        else:
            actions.append(action_detail)

    # Evaluate metrics
    base_metrics = core.metrics(base_preds, golds, folds)
    safe_metrics = core.metrics(safe_preds, golds, folds)

    # Wins / losses / ties
    wins = 0
    losses = 0
    ties = 0

    for qid in base_preds:
        gold = golds[qid]
        base_hit = len(set(base_preds[qid][:5]) & gold) / len(gold)
        safe_hit = len(set(safe_preds[qid][:5]) & gold) / len(gold)
        if safe_hit > base_hit + 1e-9:
            wins += 1
        elif safe_hit < base_hit - 1e-9:
            losses += 1
        else:
            ties += 1

    per_fold_deltas = {}
    for fold in folds:
        per_fold_deltas[fold] = safe_metrics["per_fold_recall_at_5"][fold] - base_metrics["per_fold_recall_at_5"][fold]

    delta_recall_at_5 = safe_metrics["recall_at_5"] - base_metrics["recall_at_5"]

    print(f"RABR_SAFE Actions taken: {len(actions)}")
    print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}")
    print(f"Recall@5: {safe_metrics['recall_at_5']:.9f} (delta: {delta_recall_at_5:+.9f})")
    print(f"Per-fold deltas: {per_fold_deltas}")

    report = {
        "schema_version": "dsc2026.gemini.rabr_v1.rabr_safe_report.v1",
        "variant": "RABR_SAFE",
        "action_count": len(actions),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "metrics": safe_metrics,
        "baseline_metrics": base_metrics,
        "delta_recall_at_5": delta_recall_at_5,
        "delta_precision_at_5": safe_metrics["precision_at_5"] - base_metrics["precision_at_5"],
        "delta_single_gold_recall_at_5": safe_metrics["single_gold_recall_at_5"] - base_metrics["single_gold_recall_at_5"],
        "delta_multi_gold_recall_at_5": safe_metrics["multi_gold_recall_at_5"] - base_metrics["multi_gold_recall_at_5"],
        "per_fold_deltas": per_fold_deltas,
        "actions": actions,
        "runtime_seconds": time.perf_counter() - started,
    }

    report_path = RESULTS_DIR / "RABR_SAFE_REPORT.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {report_path}", flush=True)

    pred_path = RESULTS_DIR / "RABR_SAFE_PREDICTIONS.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for qid in sorted(safe_preds, key=int):
            record = {"qid": qid, "order": safe_preds[qid][:20]}
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
