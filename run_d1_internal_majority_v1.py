#!/usr/bin/env python
"""
HUY_D1_INTERNAL_MAJORITY_V1
===========================

Hypothesis
----------
Exact D1 sometimes makes the wrong Top-5 boundary decision because a few
feature families have very large LR contribution magnitudes even when a
majority of D1's internal families prefer the challenger.

This experiment does NOT add features or models. It decomposes exact D1's
48-dimensional linear decision function into its 18 pre-existing semantic
families:

  6 rank families:
    rank_base, rank_expanded, rank_jina, rank_dense, rank_corpus, rank_global

  10 score families:
    one family for each existing score channel (2 standardized columns each)

  2 structural families:
    doctype, citation

For every query:
  * D1 ranks 1-4 are immutable.
  * D1 rank 5 is the defender.
  * Every other document in the exact D1 candidate pool is a challenger.
  * A family votes challenger iff its LR contribution(challenger) is greater
    than contribution(defender).
  * A challenger is eligible iff positive_votes > negative_votes.
    Exact ties abstain.
  * If multiple challengers are eligible, choose the one with:
      1) largest majority margin = positive - negative
      2) largest positive vote count
      3) highest exact D1 decision score
      4) earlier exact D1 rank
  * Maximum one replacement at rank 5.

There is NO numeric threshold and NO gold-based configuration.

The complete action set is written and hashed before CAL utility evaluation.

Dependency
----------
Keep this script beside:
    run_d1_query_bootstrap_bagging_v1.py

Run from Git Bash:
    python ../run_d1_internal_majority_v1.py \
      --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_FAMILY_COUNT = 18


def load_base(script_dir: Path):
    path = script_dir / "run_d1_query_bootstrap_bagging_v1.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dependency: {path}\n"
            "Keep both scripts in D:/Study/DSC2026/."
        )

    spec = importlib.util.spec_from_file_location("bagging_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def json_dump(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_family_groups(score_channel_names: Sequence[str]) -> Dict[str, List[int]]:
    """
    Exact layout of tune_expanded_fusion_selection.ltr_features():

      reciprocal ranks for 5 views       [0:5]
      normalized ranks for 5 views       [5:10]
      global min/mean rank               [10:12]
      2 columns per sorted score channel [12:32] for D1's 10 channels
      doctype                             [32:44]
      citation                            [44:48]

    We derive score positions dynamically and assert the final dimension.
    """
    rank_names = ["base", "expanded", "jina", "dense", "corpus"]

    groups: Dict[str, List[int]] = {}
    for i, name in enumerate(rank_names):
        groups[f"rank_{name}"] = [i, 5 + i]

    groups["rank_global"] = [10, 11]

    cursor = 12
    for channel in sorted(score_channel_names):
        groups[f"score_{channel}"] = [cursor, cursor + 1]
        cursor += 2

    # D1 must have exactly ten score channels here.
    if cursor != 32:
        raise RuntimeError(
            "Unexpected D1 score-channel layout: "
            f"{len(score_channel_names)} channels -> score cursor {cursor}, expected 32. "
            f"channels={sorted(score_channel_names)}"
        )

    groups["doctype"] = list(range(32, 44))
    groups["citation"] = list(range(44, 48))

    covered = [i for idxs in groups.values() for i in idxs]
    if sorted(covered) != list(range(48)):
        raise RuntimeError(
            "Family grouping does not partition exact 48D once: "
            f"covered={sorted(covered)}"
        )

    if len(groups) != EXPECTED_FAMILY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FAMILY_COUNT} families, got {len(groups)}: "
            f"{list(groups)}"
        )

    return groups


def exact_rank(score: np.ndarray) -> np.ndarray:
    """Match authoritative D1: np.argsort(-score)."""
    return np.argsort(-np.asarray(score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base = load_base(script_dir)

    root = args.repo_root.resolve()
    if not (root / "tune_corpus_cap32_fusion.py").exists():
        raise RuntimeError(
            f"{root} does not look like sota repo root. "
            "Use --repo-root /d/Study/DSC2026/sota"
        )

    out = root / "results/manual/huy_d1_internal_majority_v1"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Reconstructing exact D1 48D state...", flush=True)
    state = base.reconstruct_feature_state(root)

    groups = build_family_groups(state["score_channel_names"])
    print(
        f"  feature_dim={state['feature_dim']} "
        f"families={len(groups)}"
    )
    print("  " + ", ".join(groups.keys()))

    blocks = state["blocks"]
    all_ids = state["all_ids"]
    rows = state["rows"]
    docs_by_q = state["groups"]
    gold = state["gold"]

    baseline_ranked: Dict[str, List[str]] = {}
    candidate_ranked: Dict[str, List[str]] = {}
    label_free_rows: Dict[str, Any] = {}

    print("[2/5] Fitting exact D1 LOBO and building internal-majority actions...", flush=True)

    for held in sorted(blocks):
        train_ids = [
            q
            for b in sorted(blocks)
            if b != held
            for q in blocks[b]
        ]
        held_ids = blocks[held]

        scaler, model = base.fit_exact_d1(state, train_ids)

        coef = np.asarray(model.coef_[0], dtype=np.float64)
        if coef.shape != (48,):
            raise RuntimeError(f"Unexpected LR coef shape: {coef.shape}")

        for q in held_ids:
            X = rows[q]
            Z = scaler.transform(X)
            score = np.asarray(
                model.decision_function(Z),
                dtype=np.float64,
            )

            order_idx = exact_rank(score)
            ranking = [docs_by_q[q][i] for i in order_idx]
            baseline_ranked[q] = ranking

            index_of = {d: i for i, d in enumerate(docs_by_q[q])}
            rank_of = {d: r + 1 for r, d in enumerate(ranking)}

            defender = ranking[4]
            defender_i = index_of[defender]

            # Exact LR contribution of each semantic family per document.
            contribution: Dict[str, np.ndarray] = {}
            for family, idxs in groups.items():
                contribution[family] = (
                    Z[:, idxs] @ coef[idxs]
                ).astype(np.float64)

            candidates = []
            for challenger in ranking[5:]:
                ci = index_of[challenger]

                vote_rows = []
                positive = negative = ties = 0

                for family in groups:
                    delta = float(
                        contribution[family][ci]
                        - contribution[family][defender_i]
                    )
                    if delta > 0.0:
                        vote = "CHALLENGER"
                        positive += 1
                    elif delta < 0.0:
                        vote = "DEFENDER"
                        negative += 1
                    else:
                        vote = "TIE"
                        ties += 1

                    vote_rows.append({
                        "family": family,
                        "delta_challenger_minus_defender": delta,
                        "vote": vote,
                    })

                eligible = positive > negative
                candidates.append({
                    "doc_id": challenger,
                    "d1_rank": rank_of[challenger],
                    "d1_score": float(score[ci]),
                    "positive_votes": positive,
                    "negative_votes": negative,
                    "ties": ties,
                    "majority_margin": positive - negative,
                    "eligible": eligible,
                    "family_votes": vote_rows,
                })

            eligible = [x for x in candidates if x["eligible"]]
            eligible.sort(
                key=lambda x: (
                    -x["majority_margin"],
                    -x["positive_votes"],
                    -x["d1_score"],
                    x["d1_rank"],
                    x["doc_id"],
                )
            )

            selected = eligible[0] if eligible else None
            new_top5 = (
                ranking[:4] + [selected["doc_id"]]
                if selected is not None
                else ranking[:5]
            )

            candidate_ranked[q] = (
                new_top5
                + [d for d in ranking if d not in set(new_top5)]
            )

            label_free_rows[q] = {
                "qid": q,
                "held_block": held,
                "d1_top5": ranking[:5],
                "defender": defender,
                "defender_d1_score": float(score[defender_i]),
                "family_count": len(groups),
                "eligible_count": len(eligible),
                "action": selected is not None,
                "selected": selected,
                "candidate_top5": new_top5,
                # Retain all challengers so the rule is fully auditable.
                "challengers": candidates,
            }

    print("[3/5] Exact D1 parity check...", flush=True)
    base_metrics = base.metrics(
        baseline_ranked,
        gold,
        all_ids,
        blocks,
    )

    errors = []
    if abs(base_metrics["recall_at_5"] - EXPECTED_D1_R5) > 1e-12:
        errors.append(
            f"Recall={base_metrics['recall_at_5']} "
            f"expected={EXPECTED_D1_R5}"
        )
    for b, expected in base.EXPECTED_BLOCKS.items():
        got = base_metrics["block_recalls"][b]
        if abs(got - expected) > 1e-12:
            errors.append(f"Block {b}={got} expected={expected}")

    if errors:
        raise RuntimeError(
            "BLOCKED_D1_PARITY:\n  - " + "\n  - ".join(errors)
        )

    print(
        f"  PASS D1 R@5={base_metrics['recall_at_5']:.12f}",
        flush=True,
    )

    actions_count = sum(
        row["action"] for row in label_free_rows.values()
    )

    action_payload = {
        "schema": "manual.d1_internal_majority_v1.label_free",
        "rule": {
            "families": list(groups.keys()),
            "family_count": len(groups),
            "ranks_1_to_4_immutable": True,
            "defender": "exact D1 rank5",
            "challengers": "all exact D1 pool documents outside Top5",
            "eligibility": "positive_votes > negative_votes; exact ties abstain",
            "selection": [
                "largest majority_margin",
                "largest positive_votes",
                "highest exact D1 decision score",
                "earlier exact D1 rank",
                "doc id deterministic tie-break",
            ],
            "max_replacements_per_query": 1,
            "numeric_thresholds": None,
            "new_features": None,
            "new_models": None,
        },
        "summary": {
            "queries": len(all_ids),
            "actions": actions_count,
        },
        "rows": label_free_rows,
    }

    action_path = out / "INTERNAL_MAJORITY_ACTIONS_LABEL_FREE.json"
    json_dump(action_path, action_payload)

    seal = {
        "action_sha256": sha256_file(action_path),
        "actions": actions_count,
    }
    json_dump(out / "LABEL_FREE_SEAL.json", seal)

    print(
        f"[4/5] Sealed label-free actions: "
        f"actions={actions_count}, sha={seal['action_sha256'][:12]}...",
        flush=True,
    )

    # --------------------------------------------------------------
    # Gold utility begins ONLY after label-free action seal.
    # --------------------------------------------------------------
    print("[5/5] Evaluating sealed actions...", flush=True)

    cand_metrics = base.metrics(
        candidate_ranked,
        gold,
        all_ids,
        blocks,
    )

    wins = losses = ties = 0
    beneficial = harmful = neutral = 0
    gold_in = gold_out = 0
    changed = []

    for q in all_ids:
        r0 = base_metrics["per_query_recall"][q]
        r1 = cand_metrics["per_query_recall"][q]

        if r1 > r0:
            wins += 1
            query_effect = "WIN"
        elif r1 < r0:
            losses += 1
            query_effect = "LOSS"
        else:
            ties += 1
            query_effect = "TIE"

        row = label_free_rows[q]
        if row["action"]:
            if r1 > r0:
                beneficial += 1
                action_effect = "BENEFICIAL"
            elif r1 < r0:
                harmful += 1
                action_effect = "HARMFUL"
            else:
                neutral += 1
                action_effect = "NEUTRAL"

            before = set(row["d1_top5"])
            after = set(row["candidate_top5"])
            gold_in += len((after - before) & gold[q])
            gold_out += len((before - after) & gold[q])

            changed.append({
                "qid": q,
                "block": row["held_block"],
                "effect": action_effect,
                "recall_before": r0,
                "recall_after": r1,
                "defender": row["defender"],
                "challenger": row["selected"]["doc_id"],
                "challenger_d1_rank": row["selected"]["d1_rank"],
                "positive_votes": row["selected"]["positive_votes"],
                "negative_votes": row["selected"]["negative_votes"],
                "ties": row["selected"]["ties"],
                "majority_margin": row["selected"]["majority_margin"],
            })

    delta_r = (
        cand_metrics["recall_at_5"]
        - base_metrics["recall_at_5"]
    )
    delta_p = (
        cand_metrics["precision_at_5"]
        - base_metrics["precision_at_5"]
    )
    delta_single = (
        cand_metrics["single_gold_recall_at_5"]
        - base_metrics["single_gold_recall_at_5"]
    )
    delta_multi = (
        cand_metrics["multi_gold_recall_at_5"]
        - base_metrics["multi_gold_recall_at_5"]
    )
    block_delta = {
        b: (
            cand_metrics["block_recalls"][b]
            - base_metrics["block_recalls"][b]
        )
        for b in blocks
    }

    if (
        delta_r > 0
        and delta_p >= -1e-12
        and harmful == 0
        and losses == 0
        and all(v >= -1e-12 for v in block_delta.values())
    ):
        verdict = (
            "STRONG_PROMOTE_D1_INTERNAL_MAJORITY_V1"
            if cand_metrics["recall_at_5"] >= 0.96
            else "PROMISING_D1_INTERNAL_MAJORITY_V1"
        )
    else:
        verdict = "KILL_D1_INTERNAL_MAJORITY_V1"

    report = {
        "schema": "manual.d1_internal_majority_v1.report",
        "label_free_seal": seal,
        "family_groups": groups,
        "baseline": {
            k: v
            for k, v in base_metrics.items()
            if k != "per_query_recall"
        },
        "candidate": {
            k: v
            for k, v in cand_metrics.items()
            if k != "per_query_recall"
        },
        "delta": {
            "recall_at_5": delta_r,
            "precision_at_5": delta_p,
            "single_gold_recall_at_5": delta_single,
            "multi_gold_recall_at_5": delta_multi,
            "blocks": block_delta,
        },
        "actions": {
            "total": actions_count,
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
        },
        "verdict": verdict,
        "changed_actions": changed,
    }

    json_dump(out / "FINAL_REPORT.json", report)

    print("=" * 80)
    print(
        f"D1       R@5={base_metrics['recall_at_5']:.10f} "
        f"P@5={base_metrics['precision_at_5']:.10f}"
    )
    print(
        f"Majority R@5={cand_metrics['recall_at_5']:.10f} "
        f"P@5={cand_metrics['precision_at_5']:.10f}"
    )
    print(
        f"Delta    R={delta_r:+.10f} P={delta_p:+.10f}"
    )
    print(
        f"Single   {base_metrics['single_gold_recall_at_5']:.10f} "
        f"-> {cand_metrics['single_gold_recall_at_5']:.10f} "
        f"({delta_single:+.10f})"
    )
    print(
        f"Multi    {base_metrics['multi_gold_recall_at_5']:.10f} "
        f"-> {cand_metrics['multi_gold_recall_at_5']:.10f} "
        f"({delta_multi:+.10f})"
    )
    print(
        "Blocks   "
        + " ".join(
            f"{b}:{block_delta[b]:+.6f}"
            for b in sorted(block_delta)
        )
    )
    print(
        f"Actions  {actions_count} | "
        f"beneficial={beneficial} harmful={harmful} neutral={neutral}"
    )
    print(
        f"W/L/T    {wins}/{losses}/{ties} | "
        f"gold in/out={gold_in}/{gold_out}"
    )
    print(f"Verdict  {verdict}")
    print(f"Report   {out / 'FINAL_REPORT.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
