#!/usr/bin/env python
"""
Manual experiment: HUY_D1_SPARSE_MUTUAL_TOP5_V1

Purpose
-------
Test a calibration-free selective repair on exact D1:
replace D1 rank-5 only when frozen LegalIR BM25 and trigram both place the
same non-Top5 challenger in their Top-5 AND both rank it above the defender.

This script:
  1) reconstructs exact D1 four-block LOBO,
  2) asserts exact D1 parity,
  3) loads frozen sparse source rankings from LegalIR sources.sqlite,
  4) writes a LABEL-FREE action seal,
  5) only then evaluates CAL gold utility,
  6) writes a compact authoritative report.

Run from repo root:
    python run_sparse_mutual_top5_v1.py

Optional:
    python run_sparse_mutual_top5_v1.py --repo-root D:/Study/DSC2026/sota
    python run_sparse_mutual_top5_v1.py --sources-db D:/Study/DSC2026/LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_D1_DIM = 48
EXPECTED_BLOCKS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}

DUPLICATE_MAP = {
    "121575": "84226",
    "158189": "206810",
    "184972": "206810",
    "254937": "280171",
    "35337": "277743",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: Path) -> Dict[str, Any]:
    def cmd(*args):
        try:
            return subprocess.check_output(
                list(args), cwd=str(root), text=True, stderr=subprocess.STDOUT
            ).strip()
        except Exception as e:
            return f"ERROR: {e}"

    return {
        "head": cmd("git", "rev-parse", "HEAD"),
        "origin_main": cmd("git", "rev-parse", "origin/main"),
        "porcelain": cmd("git", "status", "--porcelain"),
    }


def load_pkl(root: Path, rel_path: str):
    p = root / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_aligned(
    root: Path,
    rel_path: str,
    candidate_pool: Dict[str, List[str]],
    all_ids: List[str],
    floor=None,
):
    obj = load_pkl(root, rel_path)
    if floor is None:
        vals = [v for q in obj.values() for v in q.values()]
        if not vals:
            raise RuntimeError(f"No score values in {rel_path}")
        floor = min(vals)
    return {
        q: {d: obj.get(q, {}).get(d, floor) for d in candidate_pool[q]}
        for q in all_ids
    }


def load_sparse_ranks(
    sources_db: Path,
    candidate_pool: Dict[str, List[str]],
    qids: List[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, List[str]]]]:
    if not sources_db.exists():
        raise FileNotFoundError(
            f"Missing LegalIR sparse DB: {sources_db}\n"
            "Pass --sources-db if your LegalIR repo is elsewhere."
        )

    con = sqlite3.connect(f"file:{sources_db.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")

    scores = {"legalir_bm25": {}, "legalir_trigram": {}}
    ranks = {"legalir_bm25": {}, "legalir_trigram": {}}

    for source_key, channel in [
        ("legalir_bm25", "bm25"),
        ("legalir_trigram", "trigram"),
    ]:
        cur = con.cursor()
        for qid in qids:
            docs = candidate_pool[qid]
            wanted = set(docs)

            row = cur.execute(
                "SELECT payload FROM sources WHERE q=? AND source=?",
                (qid, channel),
            ).fetchone()

            score_map: Dict[str, float] = {}
            native_rank: Dict[str, int] = {}
            values = json.loads(row[0]) if row else []

            by_doc = {str(r["doc_id"]): r for r in values}

            for r in values:
                d = str(r["doc_id"])
                if d in wanted:
                    score_map[d] = float(r["score"])
                    native_rank[d] = int(r["rank"])

            # Preserve historical duplicate mapping semantics.
            for d in docs:
                if d not in score_map and d in DUPLICATE_MAP:
                    twin = DUPLICATE_MAP[d]
                    r = by_doc.get(twin)
                    if r is not None:
                        score_map[d] = float(r["score"])
                        native_rank[d] = int(r["rank"])

            order = sorted(docs, key=lambda d: (native_rank.get(d, 10**9), d))
            scores[source_key][qid] = score_map
            ranks[source_key][qid] = order

    con.close()
    return scores, ranks


def metrics(
    pred: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    all_ids: List[str],
    blocks: Dict[str, List[str]],
) -> Dict[str, Any]:
    per_r = {
        q: len(set(pred[q]) & gold[q]) / len(gold[q])
        for q in all_ids
    }
    per_p = {
        q: len(set(pred[q]) & gold[q]) / 5.0
        for q in all_ids
    }

    singles = [per_r[q] for q in all_ids if len(gold[q]) == 1]
    multis = [per_r[q] for q in all_ids if len(gold[q]) > 1]

    return {
        "recall_at_5": float(np.mean(list(per_r.values()))),
        "precision_at_5": float(np.mean(list(per_p.values()))),
        "single_gold_recall_at_5": float(np.mean(singles)) if singles else None,
        "multi_gold_recall_at_5": float(np.mean(multis)) if multis else None,
        "block_recalls": {
            b: float(np.mean([per_r[q] for q in ids]))
            for b, ids in blocks.items()
        },
        "per_query_recall": per_r,
    }


def reconstruct_d1(root: Path):
    # Imports intentionally mirror the old authoritative D1 sparse evaluation.
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    ctx_dir = (
        root
        / "DSC2026-LegalIR-main"
        / "v4_run"
        / "public_test_dataset"
        / "selected-contexts"
    )
    docs = DocumentStore(sorted(ctx_dir.glob("context_*.json")))

    queries, blocks, all_ids, extended, local_views, base_scores = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    gold = {q: set(queries[q][1]) for q in all_ids}

    vnlegal_cv = load_pkl(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned(
        root, "results/crossenc_fullpool/cv_scores.pkl", extended, all_ids, -11.5
    )
    extra_cv = {
        name: load_aligned(root, rel, extended, all_ids)
        for name, rel in EXTRA_CV_PATHS.items()
    }

    d1_channels = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(root, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    preds_top5: Dict[str, List[str]] = {}
    full_rankings: Dict[str, List[str]] = {}
    feature_dim = None

    for held in sorted(blocks.keys()):
        train_ids = sum((blocks[b] for b in blocks if b != held), [])
        eval_ids = train_ids + blocks[held]

        rows, groups = ltr_features(
            local_views,
            D1_VIEWS,
            extended,
            eval_ids,
            d1_channels,
        )
        for q in rows:
            rows[q] = np.concatenate(
                [rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        feature_dim = int(rows[all_ids[0]].shape[1])
        X_train = np.vstack([rows[q] for q in train_ids])
        y_train = np.concatenate(
            [[d in gold[q] for d in groups[q]] for q in train_ids]
        ).astype(np.int8)

        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X_train), y_train)

        for q in blocks[held]:
            s = model.decision_function(scaler.transform(rows[q]))
            order_idx = np.argsort(-s)
            ranked = [groups[q][i] for i in order_idx]
            full_rankings[q] = ranked
            preds_top5[q] = ranked[:5]

    d1_metrics = metrics(preds_top5, gold, all_ids, blocks)

    return {
        "queries": queries,
        "blocks": blocks,
        "all_ids": all_ids,
        "extended": extended,
        "gold": gold,
        "preds_top5": preds_top5,
        "full_rankings": full_rankings,
        "feature_dim": feature_dim,
        "metrics": d1_metrics,
    }


def assert_d1_parity(d1: Dict[str, Any]):
    m = d1["metrics"]
    errors = []

    if d1["feature_dim"] != EXPECTED_D1_DIM:
        errors.append(
            f"feature_dim={d1['feature_dim']} expected={EXPECTED_D1_DIM}"
        )

    if abs(m["recall_at_5"] - EXPECTED_D1_R5) > 1e-12:
        errors.append(
            f"Recall@5={m['recall_at_5']} expected={EXPECTED_D1_R5}"
        )

    for b, exp in EXPECTED_BLOCKS.items():
        got = m["block_recalls"][b]
        if abs(got - exp) > 1e-12:
            errors.append(f"Block {b}={got} expected={exp}")

    if errors:
        raise RuntimeError(
            "BLOCKED_D1_PARITY:\n  - " + "\n  - ".join(errors)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to D:/Study/DSC2026/sota. Default: script directory.",
    )
    ap.add_argument(
        "--sources-db",
        type=Path,
        default=Path(
            "D:/Study/DSC2026/LegalIR/cache/"
            "exp112_task_adaptive_retrieval/sources.sqlite"
        ),
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sources_db = args.sources_db.resolve()

    if not (root / "tune_corpus_cap32_fusion.py").exists():
        raise RuntimeError(
            f"{root} does not look like the sota repo root. "
            "Put this script in D:/Study/DSC2026/sota or pass --repo-root."
        )

    out_dir = root / "results" / "manual" / "huy_d1_sparse_mutual_top5_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("[1/5] Reconstructing exact D1 LOBO...", flush=True)
    d1 = reconstruct_d1(root)
    assert_d1_parity(d1)
    print(
        f"  PASS D1 parity: R@5={d1['metrics']['recall_at_5']:.12f}, "
        f"dim={d1['feature_dim']}"
    )

    provenance = {
        "experiment": "HUY_D1_SPARSE_MUTUAL_TOP5_V1_MANUAL",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git": git_info(root),
        "sources_db": str(sources_db),
        "sources_db_sha256": sha256_file(sources_db),
        "rule": {
            "defender": "exact D1 rank5",
            "challenger_pool": "same doc must be BM25 Top5 AND trigram Top5, outside D1 Top5",
            "bm25_condition": "challenger rank < defender rank",
            "trigram_condition": "challenger rank < defender rank",
            "unique_only": True,
            "max_actions_per_query": 1,
        },
    }
    (out_dir / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    d1_parity = {
        "status": "PASS",
        "feature_dim": d1["feature_dim"],
        "metrics": {
            k: v
            for k, v in d1["metrics"].items()
            if k != "per_query_recall"
        },
        "expected_recall_at_5": EXPECTED_D1_R5,
        "expected_blocks": EXPECTED_BLOCKS,
    }
    (out_dir / "D1_PARITY.json").write_text(
        json.dumps(d1_parity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[2/5] Loading frozen BM25/trigram rankings...", flush=True)
    sparse_scores, sparse_ranks = load_sparse_ranks(
        sources_db, d1["extended"], d1["all_ids"]
    )

    # ------------------------------------------------------------------
    # LABEL-FREE action construction.
    # IMPORTANT: no gold is read anywhere in this block.
    # ------------------------------------------------------------------
    print("[3/5] Building label-free mutual-Top5 action seal...", flush=True)
    actions: Dict[str, Any] = {}
    candidate_preds = {
        q: list(d1["preds_top5"][q])
        for q in d1["all_ids"]
    }

    unique_actions = 0
    ambiguity_abstentions = 0
    queries_with_mutual = 0
    total_mutual_candidates = 0

    for q in d1["all_ids"]:
        top5 = list(d1["preds_top5"][q])
        top5_set = set(top5)
        defender = top5[4]

        bm_order = sparse_ranks["legalir_bm25"][q]
        tri_order = sparse_ranks["legalir_trigram"][q]
        bm_rank = {d: i + 1 for i, d in enumerate(bm_order)}
        tri_rank = {d: i + 1 for i, d in enumerate(tri_order)}

        bm_top5 = bm_order[:5]
        tri_top5 = tri_order[:5]
        mutual = sorted(
            (set(bm_top5) & set(tri_top5)) - top5_set
        )

        eligible = []
        details = []
        for d in mutual:
            br = bm_rank[d]
            tr = tri_rank[d]
            bdef = bm_rank[defender]
            tdef = tri_rank[defender]
            ok_b = br < bdef
            ok_t = tr < tdef
            ok = ok_b and ok_t
            if ok:
                eligible.append(d)
            details.append(
                {
                    "doc_id": d,
                    "bm25_rank": br,
                    "trigram_rank": tr,
                    "bm25_above_defender": ok_b,
                    "trigram_above_defender": ok_t,
                    "eligible": ok,
                }
            )

        if mutual:
            queries_with_mutual += 1
            total_mutual_candidates += len(mutual)

        action_fire = len(eligible) == 1
        selected = eligible[0] if action_fire else None

        if len(eligible) > 1:
            ambiguity_abstentions += 1

        new_top5 = list(top5)
        if action_fire:
            new_top5 = top5[:4] + [selected]
            candidate_preds[q] = new_top5
            unique_actions += 1

        actions[q] = {
            "qid": q,
            "d1_top5": top5,
            "defender": defender,
            "defender_bm25_rank": bm_rank[defender],
            "defender_trigram_rank": tri_rank[defender],
            "bm25_top5_in_d1_pool": bm_top5,
            "trigram_top5_in_d1_pool": tri_top5,
            "mutual_top5_candidates_outside_d1_top5": mutual,
            "candidate_details": details,
            "eligible_count": len(eligible),
            "action_fire": action_fire,
            "selected_challenger": selected,
            "candidate_top5": new_top5,
        }

    action_payload = {
        "schema": "manual.sparse_mutual_top5.actions.label_free.v1",
        "rule": provenance["rule"],
        "summary": {
            "queries_with_mutual_top5_candidates": queries_with_mutual,
            "total_mutual_top5_candidates": total_mutual_candidates,
            "unique_action_queries": unique_actions,
            "ambiguity_abstentions": ambiguity_abstentions,
        },
        "actions": actions,
    }

    action_path = out_dir / "SPARSE_MUTUAL_TOP5_ACTIONS_LABEL_FREE.json"
    action_path.write_text(
        json.dumps(action_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    action_sha = sha256_file(action_path)
    print(
        f"  sealed: actions={unique_actions}, "
        f"ambiguous={ambiguity_abstentions}, sha={action_sha[:12]}..."
    )

    # ------------------------------------------------------------------
    # Gold utility evaluation happens ONLY after action artifact is sealed.
    # ------------------------------------------------------------------
    print("[4/5] Evaluating sealed actions against CAL gold...", flush=True)
    gold = d1["gold"]
    baseline_m = d1["metrics"]
    cand_m = metrics(
        candidate_preds,
        gold,
        d1["all_ids"],
        d1["blocks"],
    )

    wins = losses = ties = 0
    beneficial = harmful = neutral = 0
    gold_in = gold_out = 0
    set_churn = ordered_churn = 0

    for q in d1["all_ids"]:
        r0 = baseline_m["per_query_recall"][q]
        r1 = cand_m["per_query_recall"][q]
        if r1 > r0:
            wins += 1
        elif r1 < r0:
            losses += 1
        else:
            ties += 1

        if actions[q]["action_fire"]:
            if r1 > r0:
                beneficial += 1
            elif r1 < r0:
                harmful += 1
            else:
                neutral += 1

        p0 = d1["preds_top5"][q]
        p1 = candidate_preds[q]
        s0, s1 = set(p0), set(p1)

        if p0 != p1:
            ordered_churn += 1
        if s0 != s1:
            set_churn += 1

        g = gold[q]
        gold_in += len((s1 - s0) & g)
        gold_out += len((s0 - s1) & g)

    # Post-hoc sparse complementarity diagnostic.
    residual = {
        "total_d1_missed_gold_occurrences": 0,
        "bm25_top5_residual_gold_count": 0,
        "trigram_top5_residual_gold_count": 0,
        "mutual_top5_residual_gold_count": 0,
        "eligible_mutual_residual_gold_count": 0,
        "admitted_residual_gold_count": 0,
        "cases": [],
    }

    for q in d1["all_ids"]:
        missed = gold[q] - set(d1["preds_top5"][q])
        if not missed:
            continue

        bm_top = set(sparse_ranks["legalir_bm25"][q][:5])
        tri_top = set(sparse_ranks["legalir_trigram"][q][:5])
        details_by_doc = {
            x["doc_id"]: x for x in actions[q]["candidate_details"]
        }

        for gdoc in sorted(missed):
            residual["total_d1_missed_gold_occurrences"] += 1
            b = gdoc in bm_top
            t = gdoc in tri_top
            mut = b and t
            elig = bool(details_by_doc.get(gdoc, {}).get("eligible", False))
            admitted = actions[q]["selected_challenger"] == gdoc

            residual["bm25_top5_residual_gold_count"] += int(b)
            residual["trigram_top5_residual_gold_count"] += int(t)
            residual["mutual_top5_residual_gold_count"] += int(mut)
            residual["eligible_mutual_residual_gold_count"] += int(elig)
            residual["admitted_residual_gold_count"] += int(admitted)

            if b or t:
                residual["cases"].append(
                    {
                        "qid": q,
                        "gold_doc_id": gdoc,
                        "bm25_top5": b,
                        "trigram_top5": t,
                        "mutual_top5": mut,
                        "eligible": elig,
                        "admitted": admitted,
                    }
                )

    block_deltas = {
        b: cand_m["block_recalls"][b] - baseline_m["block_recalls"][b]
        for b in d1["blocks"]
    }

    delta_recall = cand_m["recall_at_5"] - baseline_m["recall_at_5"]
    delta_precision = cand_m["precision_at_5"] - baseline_m["precision_at_5"]
    delta_single = (
        cand_m["single_gold_recall_at_5"]
        - baseline_m["single_gold_recall_at_5"]
    )
    delta_multi = (
        cand_m["multi_gold_recall_at_5"]
        - baseline_m["multi_gold_recall_at_5"]
    )

    gates = {
        "candidate_recall_ge_0_96": cand_m["recall_at_5"] >= 0.96 - 1e-12,
        "precision_no_decrease": cand_m["precision_at_5"] >= baseline_m["precision_at_5"] - 1e-12,
        "beneficial_ge_2": beneficial >= 2,
        "harmful_eq_0": harmful == 0,
        "paired_losses_eq_0": losses == 0,
        "actions_ge_2": unique_actions >= 2,
        "actions_le_12": unique_actions <= 12,
        "no_block_decrease": all(v >= -1e-12 for v in block_deltas.values()),
        "single_no_decrease": delta_single >= -1e-12,
        "multi_no_decrease": delta_multi >= -1e-12,
    }
    promoted = all(gates.values())

    report = {
        "schema": "manual.sparse_mutual_top5.cal_report.v1",
        "runtime_seconds": float(time.perf_counter() - t0),
        "action_seal_sha256": action_sha,
        "d1": {
            k: v
            for k, v in baseline_m.items()
            if k != "per_query_recall"
        },
        "candidate": {
            k: v
            for k, v in cand_m.items()
            if k != "per_query_recall"
        },
        "deltas": {
            "recall_at_5": delta_recall,
            "precision_at_5": delta_precision,
            "single_gold_recall_at_5": delta_single,
            "multi_gold_recall_at_5": delta_multi,
            "blocks": block_deltas,
        },
        "actions": {
            "total": unique_actions,
            "beneficial": beneficial,
            "harmful": harmful,
            "neutral": neutral,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "gold_crossings_in": gold_in,
            "gold_crossings_out": gold_out,
            "set_churn": set_churn,
            "ordered_churn": ordered_churn,
            "ambiguity_abstentions": ambiguity_abstentions,
        },
        "posthoc_residual_sparse_diagnostic": residual,
        "promotion_gates": gates,
        "verdict": (
            "LOCAL_PROMOTE_SPARSE_MUTUAL_TOP5_V1"
            if promoted
            else "KILL_SPARSE_MUTUAL_TOP5_V1"
        ),
    }

    report_path = out_dir / "SPARSE_MUTUAL_TOP5_CAL_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[5/5] DONE")
    print("=" * 72)
    print(
        f"D1       : R@5={baseline_m['recall_at_5']:.10f}  "
        f"P@5={baseline_m['precision_at_5']:.10f}"
    )
    print(
        f"Candidate: R@5={cand_m['recall_at_5']:.10f}  "
        f"P@5={cand_m['precision_at_5']:.10f}"
    )
    print(
        f"Delta    : R={delta_recall:+.10f}  "
        f"P={delta_precision:+.10f}"
    )
    print(
        f"Actions  : {unique_actions} | "
        f"beneficial={beneficial} harmful={harmful} neutral={neutral}"
    )
    print(f"W/L/T    : {wins}/{losses}/{ties}")
    print(
        "Residual sparse golds: "
        f"BM25@5={residual['bm25_top5_residual_gold_count']} "
        f"TRI@5={residual['trigram_top5_residual_gold_count']} "
        f"MUTUAL@5={residual['mutual_top5_residual_gold_count']} "
        f"ELIGIBLE={residual['eligible_mutual_residual_gold_count']} "
        f"ADMITTED={residual['admitted_residual_gold_count']}"
    )
    print(f"Verdict  : {report['verdict']}")
    print(f"Report   : {report_path}")
    print(f"Actions  : {action_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
