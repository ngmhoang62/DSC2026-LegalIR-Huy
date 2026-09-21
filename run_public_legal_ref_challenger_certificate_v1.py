#!/usr/bin/env python
"""
PUBLIC LEGAL-REF CHALLENGER CERTIFICATE RESCUE V1
=================================================

Materialize the CAL-promoted recall policy on public test WITHOUT any tuning.

Frozen policy (identical to promoted CAL arm)
---------------------------------------------
1) Rebuild the original query-anchored legal-reference generator:
   DIRECT_REFERENCE_MATCH -> HEADER_RELATION_NEIGHBOR -> BODY_RELATION_NEIGHBOR
   cap=8, deterministic canonical doc-id tie-breaking.

2) Candidate generation is relative to the exact public D1 candidate pool.
   Public D1 Top5 itself is loaded from the exact champion JSON.

3) For every triggered query, scan generated additions IN FROZEN ORDER and take
   the FIRST candidate satisfying:
       BGE_CE(candidate) > BGE_CE(D1 rank5 defender)
       AND
       BGE_CE(candidate) - median(BGE_CE(D1 ranks1..4)) >= REL_L0

   REL_L0 = -3.0393552780151367, frozen from nonCAL Fold0 DEV.

4) Replace D1 rank5 only. K remains exactly 5.
   No defender-weak gate.
   No public labels.
   No threshold search.
   No query-id rules.

The script reuses the already-built public E5 query-vector cache and exact D1
Top5 CE cache from:
  results/manual/huy_public_d1_ce_rank5_veto_v1/

It reconstructs only D1 candidate MEMBERSHIP; it does not retrain/re-score D1.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REL_L0 = -3.0393552780151367
EXPECTED_PUBLIC = 1000
MAX_ADDITIONS = 8
PUBLIC_ACTION_CAP = 10
EXPECTED_CHUNKS = 343347
EXPECTED_DOCS_EVIDENCE = 8507
E5_DIM = 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_public_champion(root: Path):
    p = (
        root
        / "results/gemini/huy_vnlegal_rank_ablation_v1/"
        "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    )
    raw = json.loads(p.read_text(encoding="utf-8"))
    pred = {}
    for q, row in raw.items():
        docs = [str(d) for d in row["answer"]]
        if len(docs) != 5 or len(set(docs)) != 5:
            raise RuntimeError(f"Invalid champion Top5 q={q}: {docs}")
        pred[str(q)] = docs
    if len(pred) != EXPECTED_PUBLIC:
        raise RuntimeError(f"Champion population={len(pred)} != {EXPECTED_PUBLIC}")
    return pred, p


def load_public_questions(root: Path):
    p = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/"
        "public-official.json"
    )
    raw = json.loads(p.read_text(encoding="utf-8"))
    questions = {
        str(q): str(v["question"])
        for q, v in raw.items()
    }
    ids = list(map(str, raw.keys()))
    if len(ids) != EXPECTED_PUBLIC:
        raise RuntimeError(f"Public questions={len(ids)} != {EXPECTED_PUBLIC}")
    return ids, questions, p


def reconstruct_public_candidate_pool(root: Path, questions, ids):
    """
    Reconstruct exact D1 public candidate membership only.
    Mirrors public_d1_control_parity.py stages up through public_candidates.
    """
    sys.path.insert(0, str(root))
    from run_burst_expanded_fusion_submission import (
        CORPUS_CAP,
        CORPUS_DEPTH,
        EXPANSION_CONFIG,
        RERANK_CONFIG,
        DocumentStore,
        corpus_dense,
        dense_expansion,
        load_public_retrieval,
        raw_union,
        weighted_rrf,
    )
    from run_burst_multistage_submission import load_metadata
    from src.gemini.huy_d1_exact_citation_public_v1.common import (
        compute_candidate_fingerprint,
        load_pkl,
    )

    public_dataset = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    contexts = public_dataset / "selected-contexts"
    docs_store = DocumentStore(sorted(contexts.glob("context_*.json")))

    class DummyArgs:
        db = root / "benchmarks/legalir_full_fts.sqlite"
        workers = 4
        cache_dir = root / "results/burst_expanded_fusion"
        output_dir = root / "results/burst_userft_maxrecall"

    _, doc_ids_meta, train_meta, public_meta = load_metadata(public_dataset)
    if list(public_meta) != ids:
        # Population equality is sufficient, but preserve official public order.
        if set(public_meta) != set(ids):
            raise RuntimeError("load_metadata public population mismatch")

    base_pub = load_pkl("results/burst_gpu_threeview/cpu_top20.pkl")["rankings"]

    pub_retrieval = load_public_retrieval(
        root, DummyArgs, doc_ids_meta, train_meta, public_meta, ids
    )
    raw_pub = {
        q: raw_union(pub_retrieval[q], EXPANSION_CONFIG["depth"])
        for q in ids
    }
    pub_retrieval.clear()

    expansion_scores = dense_expansion(
        root,
        DummyArgs.cache_dir,
        public_meta,
        ids,
        raw_pub,
        docs_store,
        "cuda",
    )
    dense_rank = {
        q: sorted(
            raw_pub[q],
            key=lambda d: (-expansion_scores[q][d], d),
        )
        for q in ids
    }
    expanded = weighted_rrf(
        [raw_pub, dense_rank],
        RERANK_CONFIG["expansion_weights"],
        RERANK_CONFIG["expansion_rrf_k"],
    )

    corpus_rank, _ = corpus_dense(
        root,
        DummyArgs.cache_dir,
        public_meta,
        ids,
        "cuda",
        cap=CORPUS_CAP,
        depth=CORPUS_DEPTH,
    )

    pools = {
        q: list(
            dict.fromkeys(
                list(base_pub[q])
                + expanded[q][: RERANK_CONFIG["expanded_depth"]]
                + corpus_rank[q][:CORPUS_DEPTH]
            )
        )
        for q in ids
    }

    fp = compute_candidate_fingerprint(ids, pools)

    # Strong parity contract against the previous exact-citation public audit,
    # if available. That experiment reproduced exact D1 1000/1000.
    old_parity = (
        root
        / "results/gemini/huy_d1_exact_citation_public_v1/"
        "PUBLIC_D1_CONTROL_PARITY.json"
    )
    parity_check = {
        "previous_parity_artifact": str(old_parity),
        "available": old_parity.is_file(),
        "candidate_fingerprint": fp,
    }
    if old_parity.is_file():
        old = json.loads(old_parity.read_text(encoding="utf-8"))
        if not old.get("control_parity_passed"):
            raise RuntimeError("Prior public D1 control parity artifact is not PASS")
        expected_fp = old["public_candidate_pool_fingerprint"]
        parity_check["expected_candidate_fingerprint"] = expected_fp
        parity_check["matches"] = (fp == expected_fp)
        if fp != expected_fp:
            raise RuntimeError(
                f"Public candidate fingerprint drift: {fp} != {expected_fp}"
            )

    return pools, contexts, parity_check


def load_corpus(contexts: Path):
    corpus = {}
    for p in sorted(contexts.glob("context_*.json")):
        d = p.stem[len("context_"):]
        corpus[d] = json.loads(p.read_text(encoding="utf-8"))
    if not corpus:
        raise RuntimeError(f"No context documents found under {contexts}")
    return corpus


class PublicRenderData:
    def __init__(
        self,
        *,
        bundle: Path,
        questions,
        qvec_path: Path,
        qids_path: Path,
        evidence_db: Path,
    ):
        self.questions = dict(questions)

        con = sqlite3.connect(
            f"file:{evidence_db.resolve().as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        row = con.execute(
            "SELECT signature FROM complete WHERE kind='inventory'"
        ).fetchone()
        con.close()
        if row is None:
            raise RuntimeError("Evidence inventory signature missing")
        self.fingerprint = str(row[0])

        chunk_rows = []
        with (bundle / "chunk_ids.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk_rows.append(json.loads(line))
        self.chunk_ids = [str(r["chunk_id"]) for r in chunk_rows]
        parents = [str(r["doc_id"]) for r in chunk_rows]

        if len(self.chunk_ids) != EXPECTED_CHUNKS:
            raise RuntimeError(f"Chunk-count drift: {len(self.chunk_ids)}")

        self.doc_ids = sorted(set(parents))
        if len(self.doc_ids) != EXPECTED_DOCS_EVIDENCE:
            raise RuntimeError(f"Evidence parent-count drift: {len(self.doc_ids)}")

        self.doc_row = {d: i for i, d in enumerate(self.doc_ids)}
        positions = [[] for _ in self.doc_ids]
        for i, d in enumerate(parents):
            positions[self.doc_row[d]].append(i)
        self.positions = [
            np.asarray(v, dtype=np.int64)
            for v in positions
        ]

        self._matrix = np.load(
            bundle / "embeddings.f16.npy", mmap_mode="r"
        )
        if self._matrix.shape != (EXPECTED_CHUNKS, E5_DIM):
            raise RuntimeError(f"Chunk matrix drift: {self._matrix.shape}")

        qids = [
            str(x)
            for x in json.loads(qids_path.read_text(encoding="utf-8"))
        ]
        self._qvec = np.load(qvec_path, mmap_mode="r")
        self._qrow = {q: i for i, q in enumerate(qids)}

        if set(self.questions) != set(self._qrow):
            raise RuntimeError("Public qvec population mismatch")
        if self._qvec.shape != (len(qids), E5_DIM):
            raise RuntimeError(f"Public qvec shape drift: {self._qvec.shape}")

    def matrix(self, source: str):
        if source != "e5":
            raise ValueError(source)
        return self._matrix

    def query_vector(self, qid: str, source: str):
        if source != "e5":
            raise ValueError(source)
        v = np.asarray(
            self._qvec[self._qrow[qid]],
            dtype=np.float32,
        )
        return v / max(float(np.linalg.norm(v)), 1e-12)


def load_top5_ce_cache(root: Path, q: str):
    p = (
        root
        / "results/manual/huy_public_d1_ce_rank5_veto_v1/"
        "ce_scores"
        / f"{q}.json"
    )
    if not p.is_file():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {str(d): float(s) for d, s in obj["scores"].items()}


@torch.inference_mode()
def score_additions(
    *,
    root: Path,
    sibling: Path,
    render,
    requests,
    out: Path,
    pair_microbatch: int,
):
    sys.path[:0] = [str(sibling), str(sibling / "src")]
    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence

    ckpt = (
        root
        / "results/manual/huy_noncal_trainable_ce_boundary_v1/"
        "oof/fold_0/training/model.pt"
    )
    model = CrossEncoder(ckpt)
    model.eval()
    evidence = Evidence(render, model.tokenizer)

    score_dir = out / "public_legal_ref_candidate_ce"
    score_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    missing = {}
    started = time.perf_counter()

    try:
        for i, (q, docs) in enumerate(requests.items(), 1):
            scorable = []
            miss = []
            for d in docs:
                ok = evidence.db.execute(
                    "SELECT 1 FROM chunks WHERE doc=? LIMIT 1",
                    (d,),
                ).fetchone() is not None
                if ok:
                    scorable.append(d)
                else:
                    miss.append(d)

            if miss:
                missing[q] = miss

            sig = hashlib.sha256(
                json.dumps(
                    [
                        "public-legal-ref-challenger-cert-v1",
                        sha256(ckpt),
                        q,
                        scorable,
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            cp = score_dir / f"{q}.json"

            scores = {}
            if scorable:
                if cp.is_file():
                    obj = json.loads(cp.read_text(encoding="utf-8"))
                    if obj.get("signature") != sig:
                        raise RuntimeError(f"Candidate CE cache drift q={q}")
                    scores = {
                        str(d): float(s)
                        for d, s in obj["scores"].items()
                    }
                else:
                    vals = []
                    for st in range(0, len(scorable), pair_microbatch):
                        batch_docs = scorable[st:st + pair_microbatch]
                        pairs = [evidence.package(q, d) for d in batch_docs]
                        vals.extend(model(pairs).detach().cpu().tolist())
                    scores = {
                        d: float(s)
                        for d, s in zip(scorable, vals)
                    }
                    dump(cp, {"signature": sig, "scores": scores})

            rows[q] = scores
            print(
                f"  CE additions {i}/{len(requests)} "
                f"scorable={len(scores)} missing={len(miss)} "
                f"qps={i/max(time.perf_counter()-started,1e-9):.2f}",
                flush=True,
            )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect()
        torch.cuda.empty_cache()

    return rows, missing, ckpt


def zip_submission(json_path: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.namelist() != ["submission.json"]:
            raise RuntimeError("ZIP member contract failed")
        if zf.read("submission.json") != json_path.read_bytes():
            raise RuntimeError("ZIP byte parity failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--pair-microbatch", type=int, default=4)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"
    out = (
        root
        / "results/manual/"
        "huy_public_legal_ref_challenger_certificate_rescue_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading exact public D1 champion + public queries...")
    d1, champion_path = load_public_champion(root)
    ids, questions, questions_path = load_public_questions(root)
    if set(d1) != set(ids):
        raise RuntimeError("Champion/public population mismatch")

    print("[2/8] Reconstructing exact public D1 candidate MEMBERSHIP...")
    pools, contexts, pool_parity = reconstruct_public_candidate_pool(
        root, questions, ids
    )
    if any(not set(d1[q]) <= set(pools[q]) for q in ids):
        bad = [q for q in ids if not set(d1[q]) <= set(pools[q])][:10]
        raise RuntimeError(f"Champion Top5 not contained in public pool sample={bad}")
    print(
        f"  mean_pool={np.mean([len(pools[q]) for q in ids]):.2f} "
        f"fingerprint={pool_parity['candidate_fingerprint']}",
        flush=True,
    )

    print("[3/8] Rebuilding frozen legal-reference index + relation graph...")
    from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
        build_legal_reference_index,
    )
    from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.relation_graph import (
        build_explicit_relation_graph,
    )
    from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.query_anchored_generator import (
        generate_query_additions,
    )

    corpus = load_corpus(contexts)
    ref_to_docs, doc_to_own_ref, index_audit = build_legal_reference_index(
        corpus,
        audit_filename="",
    )
    out_edges, in_edges, relation_audit = build_explicit_relation_graph(
        corpus,
        ref_to_docs,
        doc_to_own_ref,
        audit_filename="",
    )

    print("[4/8] Generating public additions with frozen CAL-promoted policy...")
    generated = {}
    triggered = []
    total_additions = 0

    for q in ids:
        adds, details, refs = generate_query_additions(
            query_text=questions[q],
            existing_pool=set(pools[q]),
            ref_to_docs=ref_to_docs,
            out_edges=out_edges,
            in_edges=in_edges,
            cap=MAX_ADDITIONS,
        )
        generated[q] = {
            "qid": q,
            "query_text": questions[q],
            "extracted_references": refs,
            "additions": [str(d) for d in adds],
            "details": details,
        }
        if adds:
            triggered.append(q)
            total_additions += len(adds)

    gen_path = out / "PUBLIC_LEGAL_REF_GENERATOR_LABEL_FREE.json"
    dump(
        gen_path,
        {
            "schema": "manual.public_legal_ref_generator_label_free.v1",
            "policy": {
                "max_additions": MAX_ADDITIONS,
                "ordering": [
                    "DIRECT_REFERENCE_MATCH",
                    "HEADER_RELATION_NEIGHBOR",
                    "BODY_RELATION_NEIGHBOR",
                ],
                "tie_breaking": "CANONICAL_DOC_ID_NUMERIC_ASC",
                "traversal_hops": 1,
            },
            "triggered_queries": len(triggered),
            "total_additions": total_additions,
            "rows": generated,
        },
    )
    print(
        f"  triggered_queries={len(triggered)} "
        f"total_additions={total_additions}",
        flush=True,
    )

    print("[5/8] Loading public CE contract + D1 Top5 CE caches...")
    ce_root = root / "results/manual/huy_public_d1_ce_rank5_veto_v1"
    qvec = ce_root / "public_query_vectors/public_queries.f32.npy"
    qids_path = ce_root / "public_query_vectors/public_query_ids.json"
    bundle = root / "cache/research_v2_e5_confirmation/bundle-v1"
    evidence_db = sibling / "cache/exp_final_retrieval/evidence.sqlite"

    for p in (qvec, qids_path, bundle / "embeddings.f16.npy", evidence_db):
        if not p.exists():
            raise FileNotFoundError(p)

    render = PublicRenderData(
        bundle=bundle,
        questions=questions,
        qvec_path=qvec,
        qids_path=qids_path,
        evidence_db=evidence_db,
    )

    prelim = {}
    requests = {}
    missing_top5_ce = []

    for q in triggered:
        top5 = d1[q]
        scores = load_top5_ce_cache(root, q)
        if scores is None or any(d not in scores for d in top5):
            missing_top5_ce.append(q)
            prelim[q] = {"abstain": "MISSING_D1_TOP5_CE_CACHE"}
            continue

        med = float(np.median([scores[d] for d in top5[:4]]))
        defender = top5[4]
        prelim[q] = {
            "defender": defender,
            "defender_score": scores[defender],
            "median_top4": med,
            "defender_rel": scores[defender] - med,
        }
        requests[q] = generated[q]["additions"]

    print(
        f"  triggered={len(triggered)} "
        f"CE-eligible-triggered={len(requests)} "
        f"missing-D1-CE={len(missing_top5_ce)}",
        flush=True,
    )

    print("[6/8] Scoring legal-ref additions with frozen Fold0 BGE CE...")
    ce_rows, missing_evidence, ckpt = score_additions(
        root=root,
        sibling=sibling,
        render=render,
        requests=requests,
        out=out,
        pair_microbatch=args.pair_microbatch,
    )

    print("[7/8] Applying EXACT promoted policy and sealing public actions...")
    actions = {}
    diagnostics = {}

    for q in ids:
        if q not in generated or not generated[q]["additions"]:
            continue

        row = {
            **generated[q],
            **prelim.get(q, {}),
            "candidate_certificates": [],
            "missing_evidence_docs": missing_evidence.get(q, []),
        }

        if q not in requests or q not in ce_rows:
            diagnostics[q] = row
            continue

        med = prelim[q]["median_top4"]
        defender_score = prelim[q]["defender_score"]

        for idx, (d, detail) in enumerate(
            zip(generated[q]["additions"], generated[q]["details"])
        ):
            if d not in ce_rows[q]:
                row["candidate_certificates"].append({
                    "generator_index": idx,
                    "doc": d,
                    "addition_type": detail.get("addition_type"),
                    "relation_family": detail.get("relation_family"),
                    "has_frozen_evidence": False,
                    "certified": False,
                })
                continue

            score = float(ce_rows[q][d])
            rel = score - med
            beats_defender = score > defender_score
            abs_cert = rel >= REL_L0
            cert = beats_defender and abs_cert

            item = {
                "generator_index": idx,
                "doc": d,
                "addition_type": detail.get("addition_type"),
                "relation_family": detail.get("relation_family"),
                "relation_direction": detail.get("relation_direction"),
                "is_header": detail.get("is_header"),
                "anchor_ref": detail.get("anchor_ref"),
                "anchor_doc": detail.get("anchor_doc"),
                "bge_score": score,
                "bge_rel_to_d1_top4_median": rel,
                "bge_beats_defender": bool(beats_defender),
                "passes_rel_l0_candidate_certificate": bool(abs_cert),
                "certified": bool(cert),
            }
            row["candidate_certificates"].append(item)

        first = next(
            (x for x in row["candidate_certificates"] if x["certified"]),
            None,
        )
        if first is not None:
            c = first["doc"]
            actions[q] = {
                "qid": q,
                "challenger": c,
                "defender": d1[q][4],
                "before": list(d1[q]),
                "after": list(d1[q][:4]) + [c],
                "generator_index": first["generator_index"],
                "addition_type": first.get("addition_type"),
                "relation_family": first.get("relation_family"),
                "relation_direction": first.get("relation_direction"),
                "anchor_ref": first.get("anchor_ref"),
                "anchor_doc": first.get("anchor_doc"),
                "challenger_rel": first["bge_rel_to_d1_top4_median"],
                "defender_rel": prelim[q]["defender_rel"],
            }

        diagnostics[q] = row

    action_doc = {
        "schema": "manual.public_legal_ref_challenger_certificate_rescue_v1.actions",
        "status": "SEALED_LABEL_FREE_PUBLIC",
        "policy": {
            "candidate_source": "query-anchored legal-reference generator",
            "candidate_order": (
                "DIRECT_REFERENCE_MATCH -> HEADER_RELATION_NEIGHBOR "
                "-> BODY_RELATION_NEIGHBOR; first certified candidate"
            ),
            "cap": MAX_ADDITIONS,
            "require_challenger_bge_gt_defender": True,
            "challenger_rel_threshold": REL_L0,
            "defender_rel_threshold": None,
            "replace_slot": 5,
            "K": 5,
            "no_public_labels": True,
            "no_threshold_search": True,
            "no_qid_specific_rules": True,
        },
        "champion_path": str(champion_path),
        "champion_sha256": sha256(champion_path),
        "public_questions_path": str(questions_path),
        "public_questions_sha256": sha256(questions_path),
        "public_candidate_pool_parity": pool_parity,
        "generator_sha256": sha256(gen_path),
        "bge_checkpoint": str(ckpt),
        "bge_checkpoint_sha256": sha256(ckpt),
        "triggered_queries": len(triggered),
        "total_generator_additions": total_additions,
        "actions_count": len(actions),
        "actions": actions,
        "diagnostics": diagnostics,
    }
    actions_path = out / "PUBLIC_ACTIONS_LABEL_FREE.json"
    dump(actions_path, action_doc)

    for q, a in actions.items():
        print(
            f"    q={q} {a['addition_type']} "
            f"D1r5 {a['defender']} -> {a['challenger']} "
            f"rel {a['defender_rel']:+.3f}->{a['challenger_rel']:+.3f}",
            flush=True,
        )

    print("[8/8] Materializing exact K=5 submission ZIP...")
    submission = {}
    for q in ids:
        ans = actions[q]["after"] if q in actions else d1[q]
        if len(ans) != 5 or len(set(ans)) != 5:
            raise RuntimeError(f"Invalid final K=5 q={q}: {ans}")
        submission[q] = {"answer": [str(d) for d in ans]}

    changed = sum(
        submission[q]["answer"] != d1[q]
        for q in ids
    )
    if changed != len(actions):
        raise RuntimeError(
            f"Churn mismatch: changed={changed} actions={len(actions)}"
        )
    if any(
        submission[q]["answer"][:4] != d1[q][:4]
        for q in actions
    ):
        raise RuntimeError("Ranks 1-4 changed on an intervention query")

    sub_path = out / "submission.json"
    zip_path = out / "submission_LEGAL_REF_CHALLENGER_CERT.zip"
    dump(sub_path, submission)
    zip_submission(sub_path, zip_path)

    status = (
        "READY_TO_SUBMIT"
        if 1 <= len(actions) <= PUBLIC_ACTION_CAP
        else "NO_ACTIONS"
        if len(actions) == 0
        else "BLOCKED_ACTION_SHIFT"
    )

    report = {
        "schema": "manual.public_legal_ref_challenger_certificate_rescue_v1.report",
        "status": status,
        "cal_promoted_policy_reference": {
            "cal_recall_before": 0.9569444444,
            "cal_recall_after": 0.9575,
            "cal_delta_recall": 0.0005555556,
            "cal_actions": 3,
            "cal_wins": 1,
            "cal_losses": 0,
            "cal_neutral": 2,
        },
        "public": {
            "queries": len(ids),
            "generator_triggered_queries": len(triggered),
            "generator_total_additions": total_additions,
            "missing_top5_ce_triggered_queries": len(missing_top5_ce),
            "actions": len(actions),
            "action_rate": len(actions) / len(ids),
            "action_cap": PUBLIC_ACTION_CAP,
            "changed_queries": changed,
        },
        "artifacts": {
            "submission_json": str(sub_path),
            "submission_json_sha256": sha256(sub_path),
            "submission_zip": str(zip_path),
            "submission_zip_sha256": sha256(zip_path),
            "actions": str(actions_path),
            "actions_sha256": sha256(actions_path),
            "generator": str(gen_path),
            "generator_sha256": sha256(gen_path),
        },
    }
    report_path = out / "REPORT.json"
    dump(report_path, report)

    print("=" * 112)
    print("PUBLIC LEGAL-REF CHALLENGER CERTIFICATE")
    print(
        f"generator triggered={len(triggered)} | "
        f"additions={total_additions} | "
        f"actions={len(actions)} | status={status}"
    )
    print(f"ZIP: {zip_path}")
    print(f"Report: {report_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
