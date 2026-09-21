#!/usr/bin/env python
"""
HUY PIPELINE AUDIT S — EXACT/NORMALIZED DUPLICATE SLOT RESCUE V1
===============================================================

CPU-only exact D1 LOBO.

Corpus-integrity audit found:
  - 4 exact/normalized duplicate-text groups in the 8,532-doc corpus
  - exactly 1 CAL query where D1 Top5 contains two normalized-duplicate docs
  - 0 CAL gold queries where two normalized-duplicate docs are simultaneously gold

Label-free policy:
  take the exact D1 full ranking;
  keep the higher-ranked occurrence of each normalized passage;
  skip later normalized duplicates;
  continue down the SAME D1 ranking until K=5.

No gold is used to construct actions. Gold is opened only after the deduplicated
Top5 predictions are sealed, for CAL evaluation.

This tests whether a duplicate corpus slot is wasting K.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}
WS_RE = re.compile(r"\s+", re.UNICODE)


def norm_text(s: str) -> str:
    return WS_RE.sub(
        " ",
        unicodedata.normalize("NFKC", s or "").lower(),
    ).strip()


def load_pkl(root, rel):
    obj = pickle.loads((root / rel).read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    print("[1/5] Loading exact current D1 feature world...", flush=True)

    context_paths = sorted(
        (
            root
            / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
        ).glob("context_*.json")
    )
    docs_store = DocumentStore(context_paths)

    queries, blocks, ids, extended, views, base_scores = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )

    # Keep gold inaccessible to policy construction until actions are sealed.
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    def aligned(rel, floor=None):
        obj = load_pkl(root, rel)
        if floor is None:
            floor = min(v for q in obj for v in obj[q].values())
        return {
            q: {d: obj.get(q, {}).get(d, floor) for d in extended[q]}
            for q in ids
        }

    scores = {
        **base_scores,
        "vnlegal_lal": load_pkl(
            root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"
        ),
        "crossenc": aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5),
        **{k: aligned(v) for k, v in EXTRA.items()},
    }

    type_table = build_type_table(root, docs_store, ids, extended)
    type_rows = type_features(extended, type_table, queries, ids)
    own, cited = build_citation_table(docs_store, ids, extended)
    cite_rows = citation_features(extended, own, cited, ids)

    rows, groups = ltr_features(
        views, D1_VIEWS, extended, ids, scores
    )
    for q in ids:
        rows[q] = np.concatenate(
            [rows[q], type_rows[q], cite_rows[q]], axis=1
        )
    if rows[ids[0]].shape[1] != 48:
        raise RuntimeError("Expected exact 48D D1 contract")

    print("[2/5] Recomputing full LOBO D1 rankings...", flush=True)

    full_rank = {}
    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        sc = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(sc.transform(X), y)

        for q in blocks[held]:
            s = model.decision_function(sc.transform(rows[q]))
            order = np.argsort(-s, kind="stable")
            full_rank[q] = [groups[q][i] for i in order]

    baseline = {q: full_rank[q][:5] for q in ids}

    def metric(pred):
        rr = []
        pp = []
        per = {}
        for q in ids:
            h = len(set(pred[q]) & gold[q])
            r = h / len(gold[q])
            p = h / 5.0
            rr.append(r); pp.append(p); per[q] = r
        return float(np.mean(rr)), float(np.mean(pp)), per

    br, bp, bper = metric(baseline)
    if abs(br - EXPECTED_R) > 1e-12 or abs(bp - EXPECTED_P) > 1e-12:
        raise RuntimeError(
            f"D1 parity failed: R={br} P={bp}"
        )

    print("[3/5] Sealing normalized-text dedup actions WITHOUT gold...", flush=True)

    # Precompute normalized text only for documents that actually appear in D1 pools.
    needed_docs = sorted({d for q in ids for d in full_rank[q]})
    norm = {}
    for i, d in enumerate(needed_docs, 1):
        norm[d] = norm_text(docs_store[d])
        if i % 1000 == 0 or i == len(needed_docs):
            print(f"  normalized docs {i}/{len(needed_docs)}", flush=True)

    candidate = {}
    actions = []

    for q in ids:
        chosen = []
        seen = set()
        skipped = []

        for d in full_rank[q]:
            sig = norm[d]
            # Empty passages are not collapsed: their link-title fallback can be
            # distinct despite empty passage text.
            if sig and sig in seen:
                skipped.append(d)
                continue
            chosen.append(d)
            if sig:
                seen.add(sig)
            if len(chosen) == 5:
                break

        if len(chosen) != 5:
            raise RuntimeError(f"Could not fill K=5 after dedup q={q}")

        candidate[q] = chosen
        if chosen != baseline[q]:
            actions.append({
                "qid": q,
                "baseline": baseline[q],
                "candidate": chosen,
                "skipped_duplicates": skipped,
                "promoted_docs": [d for d in chosen if d not in baseline[q]],
            })

    print(
        f"  sealed actions={len(actions)} before outcome evaluation",
        flush=True,
    )

    print("[4/5] Opening CAL gold for post-seal evaluation...", flush=True)

    cr, cp, cper = metric(candidate)
    wins = losses = neutral = 0
    action_eval = []
    for a in actions:
        q = a["qid"]
        delta = cper[q] - bper[q]
        if delta > 1e-12:
            wins += 1
            outcome = "WIN"
        elif delta < -1e-12:
            losses += 1
            outcome = "LOSS"
        else:
            neutral += 1
            outcome = "NEUTRAL"
        action_eval.append({
            **a,
            "outcome": outcome,
            "delta_recall": delta,
            "gold": sorted(gold[q]),
        })

    block_delta = {}
    for b, qids in blocks.items():
        bb = float(np.mean([bper[q] for q in qids]))
        cc = float(np.mean([cper[q] for q in qids]))
        block_delta[b] = cc - bb

    promote = (
        len(actions) >= 1
        and losses == 0
        and cr >= br - 1e-12
        and all(v >= -1e-12 for v in block_delta.values())
        and wins > 0
    )

    report = {
        "schema": "manual.normalized_duplicate_slot_rescue_v1",
        "policy": {
            "normalization": "NFKC + lowercase + collapse whitespace",
            "keep": "higher D1-ranked normalized-text occurrence",
            "fill": "first later D1-ranked document with unseen normalized text",
            "K": 5,
            "gold_access_during_action_construction": False,
        },
        "baseline": {"recall": br, "precision": bp},
        "candidate": {"recall": cr, "precision": cp},
        "delta": {"recall": cr-br, "precision": cp-bp},
        "actions": len(actions),
        "wins": wins,
        "losses": losses,
        "neutral": neutral,
        "block_deltas": block_delta,
        "action_details_postseal": action_eval,
        "verdict": (
            "PROMOTE_DUPLICATE_SLOT_POLICY"
            if promote else "KILL_DUPLICATE_SLOT_POLICY"
        ),
    }

    out = root / "results/manual/huy_normalized_duplicate_slot_rescue_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT")
    print("=" * 116)
    print(f"D1        R={br:.10f} P={bp:.10f}")
    print(
        f"CANDIDATE R={cr:.10f} ({cr-br:+.10f}) "
        f"P={cp:.10f} ({cp-bp:+.10f})"
    )
    print(
        f"actions={len(actions)} W/L/N={wins}/{losses}/{neutral} "
        f"blocks={block_delta}"
    )
    for x in action_eval:
        print(
            f"  q={x['qid']} outcome={x['outcome']} "
            f"skip={x['skipped_duplicates']} promote={x['promoted_docs']}"
        )
    print("VERDICT:", report["verdict"])
    print("Report:", path)
    print("=" * 116)


if __name__ == "__main__":
    main()
