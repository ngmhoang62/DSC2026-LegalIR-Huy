#!/usr/bin/env python
"""
HUY PIPELINE AUDIT K — JINA OVERFLOW HEADER REPAIR V1
======================================================

GPU experiment. Run AFTER cap48 / parent-aggregation jobs.

Audit finding:
  15.77% of current Jina query+top-passage pairs exceed max_length=512.

Current top_passages() prepends ~70 words of document header to a selected
non-header lexical window:
    HEADER + "[ĐOẠN PHÙ HỢP]" + SELECTED_WINDOW

Frozen repair:
  - if pair length <=512: unchanged, reuse cached Jina score;
  - if pair >512 AND passage contains the prepend marker:
      remove ONLY the prepended header and score SELECTED_WINDOW;
  - if overflow has no prepend marker: leave unchanged;
  - score both passages for any affected doc and MAX aggregate exactly as current.

This isolates one packaging bug without changing lexical selection/model/LTR.

Only current core `jina` view+score channel is patched. All other D1 channels,
including historical `jina_ft`, remain frozen.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


EXPECTED_R = 0.9569444444444444
MARKER = "\n[ĐOẠN PHÙ HỢP]\n"
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def load_pickle(root, rel):
    return pickle.loads((root / rel).read_bytes())


def align(raw, candidates, ids, floor=None):
    if isinstance(raw, dict) and isinstance(raw.get("scores"), dict):
        raw = raw["scores"]
    if floor is None:
        vals = [v for q in raw.values() for v in q.values()]
        floor = min(vals) if vals else -1e9
    return {
        q: {d: float(raw.get(q, {}).get(d, floor)) for d in candidates[q]}
        for q in ids
    }


def pair_len(tok, q, p):
    return len(tok(
        q, p,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"])


def prepare_world(root):
    from benchmark_jina_reranker_holdouts import load_documents, top_passages
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    text_docs = load_documents(data)
    docs = DocumentStore(sorted((data / "selected-contexts").glob("context_*.json")))

    queries, blocks, ids, candidates, views, base_scores = build_training_cap(
        root, 32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    channels = {
        **base_scores,
        "vnlegal_lal": align(
            load_pickle(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"),
            candidates, ids
        ),
        "crossenc": align(
            load_pickle(root, "results/crossenc_fullpool/cv_scores.pkl"),
            candidates, ids, -11.5
        ),
        **{
            k: align(load_pickle(root, rel), candidates, ids)
            for k, rel in EXTRA.items()
        },
    }

    type_table = build_type_table(root, docs, ids, candidates)
    type_rows = type_features(candidates, type_table, queries, ids)
    own, cited = build_citation_table(docs, ids, candidates)
    cite_rows = citation_features(candidates, own, cited, ids)

    return {
        "queries": queries,
        "blocks": blocks,
        "ids": ids,
        "candidates": candidates,
        "views": views,
        "channels": channels,
        "gold": gold,
        "type_rows": type_rows,
        "cite_rows": cite_rows,
        "text_docs": text_docs,
        "top_passages": top_passages,
    }


@torch.inference_mode()
def patch_jina(root, world, batch_size=16):
    model_path = root / "models/jina-reranker-v2-base-multilingual"
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
        local_files_only=True,
    )
    # suppress warning only during untruncated length audit
    tok.model_max_length = 10**9

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    state = torch.load(
        root / "results/jina_reranker/burst_pairwise_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state["state_dict"], strict=False)
    model._tokenizer = tok
    model.eval().to("cuda")

    patched = {
        q: dict(world["channels"]["jina"][q])
        for q in world["ids"]
    }

    affected_docs = []
    overflow_pairs = 0
    repairable_pairs = 0
    unrepairable_pairs = 0
    repaired_still_over = 0

    print("[2/5] Discovering overflow docs...", flush=True)
    started = time.perf_counter()

    for qi, q in enumerate(world["ids"], 1):
        question = world["queries"][q][0]
        for d in world["candidates"][q]:
            passages = world["top_passages"](
                question, world["text_docs"][d], count=2
            )
            lens = [pair_len(tok, question, p) for p in passages]
            over = [L > 512 for L in lens]

            if not any(over):
                continue

            overflow_pairs += sum(over)
            repaired = []
            doc_repairable = False

            for p, is_over in zip(passages, over):
                if is_over and MARKER in p:
                    _, tail = p.split(MARKER, 1)
                    rp = tail
                    repairable_pairs += 1
                    doc_repairable = True
                    if pair_len(tok, question, rp) > 512:
                        repaired_still_over += 1
                    repaired.append(rp)
                else:
                    if is_over:
                        unrepairable_pairs += 1
                    repaired.append(p)

            if doc_repairable:
                affected_docs.append((q, d, question, repaired))

        if qi % 50 == 0 or qi == len(world["ids"]):
            print(
                f"  scanned q={qi}/{len(world['ids'])} "
                f"affected_docs={len(affected_docs)} "
                f"overflow_pairs={overflow_pairs} "
                f"({qi/max(time.perf_counter()-started,1e-9):.2f} q/s)",
                flush=True,
            )

    print(
        f"  affected docs={len(affected_docs)} | "
        f"repairable pairs={repairable_pairs} | "
        f"unrepairable pairs={unrepairable_pairs} | "
        f"repaired still >512={repaired_still_over}",
        flush=True,
    )

    print("[3/5] Rescoring only affected docs...", flush=True)

    cache_path = (
        root
        / "results/manual/huy_jina_overflow_header_repair_v1/"
        "PATCHED_JINA_SCORES.pkl"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    completed = {}
    if cache_path.is_file():
        completed = pickle.loads(cache_path.read_bytes())

    for i, (q, d, question, passages) in enumerate(affected_docs, 1):
        key = (q, d)
        if key in completed:
            patched[q][d] = float(completed[key])
            continue

        pairs = [(question, p) for p in passages]
        raw = model.compute_score(
            pairs,
            batch_size=min(batch_size, len(pairs)),
            max_length=512,
        )
        if np.isscalar(raw):
            raw = [raw]
        score = max(float(x) for x in raw)
        completed[key] = score
        patched[q][d] = score

        if i % 100 == 0 or i == len(affected_docs):
            cache_path.write_bytes(pickle.dumps(completed, protocol=5))
            print(
                f"  rescored docs {i}/{len(affected_docs)}",
                flush=True,
            )

    cache_path.write_bytes(pickle.dumps(completed, protocol=5))

    del model
    torch.cuda.empty_cache()

    audit = {
        "overflow_pairs": overflow_pairs,
        "repairable_pairs": repairable_pairs,
        "unrepairable_pairs": unrepairable_pairs,
        "affected_docs": len(affected_docs),
        "repaired_still_over512": repaired_still_over,
    }
    return patched, audit


def lobo(world, jina_scores):
    from tune_expanded_fusion_selection import ltr_features

    channels = dict(world["channels"])
    channels["jina"] = jina_scores

    views = dict(world["views"])
    # IMPORTANT: the historical Jina score table is intentionally sparse.
    # build_training_cap() reconstructs the production Jina view with
    # table[q].get(doc, -1e9).  Reproduce that exact missing-score semantics
    # here; indexing jina_scores[q][d] directly is incorrect and caused the
    # KeyError seen on doc 73336.
    views["jina"] = {
        q: sorted(
            world["candidates"][q],
            key=lambda d: (-float(jina_scores[q].get(d, -1e9)), d),
        )
        for q in world["ids"]
    }

    # For the frozen control scores, this must exactly reproduce the shipped
    # production Jina view.  For patched scores it is expected to differ.
    if jina_scores is world["channels"]["jina"]:
        mismatched = [
            q for q in world["ids"]
            if views["jina"][q] != world["views"]["jina"][q]
        ]
        if mismatched:
            raise RuntimeError(
                f"Control Jina view parity failed on {len(mismatched)} qids; "
                f"sample={mismatched[:5]}"
            )

    rows, groups = ltr_features(
        views, D1_VIEWS, world["candidates"], world["ids"], channels
    )
    for q in world["ids"]:
        rows[q] = np.concatenate(
            [rows[q], world["type_rows"][q], world["cite_rows"][q]],
            axis=1,
        )

    pred = {}
    per_q = {}

    for held in sorted(world["blocks"]):
        train = sum(
            (world["blocks"][b] for b in world["blocks"] if b != held),
            [],
        )
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in world["gold"][q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X), y)

        for q in world["blocks"][held]:
            s = model.decision_function(scaler.transform(rows[q]))
            order = np.argsort(-s, kind="stable")
            pred[q] = [groups[q][i] for i in order[:5]]

    for q in world["ids"]:
        per_q[q] = (
            len(set(pred[q]) & world["gold"][q])
            / len(world["gold"][q])
        )

    return {
        "recall": float(np.mean([per_q[q] for q in world["ids"]])),
        "precision": float(np.mean([
            len(set(pred[q]) & world["gold"][q]) / 5.0
            for q in world["ids"]
        ])),
        "blocks": {
            b: float(np.mean([per_q[q] for q in world["blocks"][b]]))
            for b in sorted(world["blocks"])
        },
        "pred": pred,
        "per_q": per_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    print("[1/5] Loading exact current D1 world...", flush=True)
    world = prepare_world(root)

    # Control from frozen cached core Jina scores.
    control = lobo(world, world["channels"]["jina"])
    if abs(control["recall"] - EXPECTED_R) > 1e-12:
        raise RuntimeError(f"D1 control parity failed: {control['recall']}")

    patched, audit = patch_jina(
        root, world, batch_size=args.batch_size
    )

    print("[4/5] Running exact LOBO with repaired Jina evidence...", flush=True)
    candidate = lobo(world, patched)

    d = np.asarray([
        candidate["per_q"][q] - control["per_q"][q]
        for q in world["ids"]
    ])
    block_delta = {
        b: candidate["blocks"][b] - control["blocks"][b]
        for b in control["blocks"]
    }

    comparison = {
        "delta_recall": candidate["recall"] - control["recall"],
        "delta_precision": candidate["precision"] - control["precision"],
        "wins": int(np.sum(d > 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
        "block_deltas": block_delta,
        "top5_exact_matches": int(sum(
            candidate["pred"][q] == control["pred"][q]
            for q in world["ids"]
        )),
    }

    promote = (
        comparison["delta_recall"] > 1e-12
        and all(x >= -1e-12 for x in block_delta.values())
        and comparison["wins"] > comparison["losses"]
    )

    report = {
        "schema": "manual.jina_overflow_header_repair_v1",
        "audit": audit,
        "control": {
            "recall": control["recall"],
            "precision": control["precision"],
            "blocks": control["blocks"],
        },
        "candidate": {
            "recall": candidate["recall"],
            "precision": candidate["precision"],
            "blocks": candidate["blocks"],
        },
        "comparison": comparison,
        "policy": {
            "repair_only_if_pair_gt512": True,
            "repair": "drop prepended document header only; keep selected lexical window",
            "model_unchanged": True,
            "lexical_selection_unchanged": True,
            "aggregation": "MAX of two passages",
            "public_labels_used": False,
        },
        "verdict": (
            "PROMOTE_JINA_HEADER_REPAIR"
            if promote else "KILL_JINA_HEADER_REPAIR"
        ),
    }

    out = root / "results/manual/huy_jina_overflow_header_repair_v1"
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT")
    print("=" * 112)
    print(
        f"overflow_pairs={audit['overflow_pairs']} "
        f"repairable={audit['repairable_pairs']} "
        f"affected_docs={audit['affected_docs']} "
        f"still_over512={audit['repaired_still_over512']}"
    )
    print(
        f"D1        R={control['recall']:.10f} P={control['precision']:.10f}"
    )
    print(
        f"CANDIDATE R={candidate['recall']:.10f} "
        f"({comparison['delta_recall']:+.10f}) "
        f"P={candidate['precision']:.10f} "
        f"({comparison['delta_precision']:+.10f})"
    )
    print(
        f"W/L/T={comparison['wins']}/{comparison['losses']}/{comparison['ties']} "
        f"blocks={comparison['block_deltas']}"
    )
    print("VERDICT:", report["verdict"])
    print("Report:", path)
    print("=" * 112)


if __name__ == "__main__":
    main()
