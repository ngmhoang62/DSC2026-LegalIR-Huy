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
import pickle
import re
import unicodedata
from collections import defaultdict
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



# ---------------------------------------------------------------------------
# SELF-CONTAINED FROZEN LEGAL-REFERENCE LOGIC
# Copied from the audited V1 source to avoid importing its Windows stdout/stderr
# wrapper side effects. Logic/constants intentionally unchanged.
# ---------------------------------------------------------------------------

REF_REGEX = re.compile(
    r'(?:'
    r'\b\d+/(?:\d{4}/)?(?:[A-ZĐa-zđ\d]+[-/])*[A-ZĐa-zđ\d]+\b'
    r'|\b\d+-(?:CT|NQ|QĐ|QD|TT)/[A-ZĐa-zđ\d]+\b'
    r'|\b(?:QCVN|TCVN)\s*[\d\.\-]+(?::\d{4}|/\d{4})?(?:/[A-ZĐa-zđ\d]+)?\b'
    r')',
    re.UNICODE,
)
SO_PAT = re.compile(
    r'S[ốoỐ]\s*[:\.]\s*([0-9]+[0-9a-zA-ZĐđ\.\-_/]+(?:/[0-9a-zA-ZĐđ\.\-_/]+)*)',
    re.UNICODE | re.IGNORECASE,
)
LINK_PAT = re.compile(
    r'/(?:Thong-tu|Nghi-dinh|Quyet-dinh|Luat|Nghi-quyet|Chi-thi|Cong-van|Thong-bao)-'
    r'([0-9]+(?:-[0-9]+)?-[0-9a-zA-ZĐđ\-]+)-',
    re.UNICODE | re.IGNORECASE,
)

AMEND_TRIGGERS = [
    "sửa đổi", "bổ sung", "thay thế", "bãi bỏ",
    "sua doi", "bo sung", "thay the", "bai bo",
]
GUIDE_TRIGGERS = [
    "hướng dẫn thi hành", "quy định chi tiết và hướng dẫn thi hành",
    "quy định chi tiết", "hướng dẫn", "thi hành",
    "huong dan thi hanh", "quy dinh chi tiet", "huong dan", "thi hanh",
]


def canonicalize_ref(ref_str: str) -> str:
    s = unicodedata.normalize("NFC", (ref_str or "").strip()).upper()
    s = re.sub(r"\s*([/\-:])\s*", r"\1", s)
    s = s.replace("ND-CP", "NĐ-CP")
    s = s.replace("QD-TTG", "QĐ-TTg").replace("QĐ-TTG", "QĐ-TTg")
    s = s.replace("BGDDT", "BGDĐT")
    s = s.rstrip(".,;:()")
    return s


def extract_doc_own_reference(passage: str, link: str):
    m_so = SO_PAT.search(passage[:800])
    if m_so:
        raw_val = m_so.group(1).split()[0]
        ref = canonicalize_ref(raw_val)
        if len(ref) >= 3 and any(c.isdigit() for c in ref):
            return ref
    if link:
        m_link = LINK_PAT.search(link)
        if m_link:
            raw_slug = m_link.group(1)
            slug_norm = re.sub(r"^(\d+)-(\d{4})-", r"\1/\2/", raw_slug)
            slug_norm = re.sub(r"^(\d+)-([A-ZĐa-zđ]+)-", r"\1/\2-", slug_norm)
            ref = canonicalize_ref(slug_norm)
            if len(ref) >= 3 and any(c.isdigit() for c in ref):
                return ref
    return None


def build_legal_reference_index_local(corpus):
    ref_to_docs = defaultdict(list)
    doc_to_own_ref = {}
    for doc_id, item in corpus.items():
        own_ref = extract_doc_own_reference(
            item.get("passage", ""),
            item.get("link", ""),
        )
        if own_ref:
            doc_to_own_ref[doc_id] = own_ref
            ref_to_docs[own_ref].append(doc_id)
    for r in ref_to_docs:
        ref_to_docs[r].sort(key=lambda x: int(x) if x.isdigit() else x)
    collision_groups = {
        r: docs for r, docs in ref_to_docs.items() if len(docs) > 1
    }
    audit = {
        "status": "PASS",
        "total_documents": len(corpus),
        "indexed_documents_count": len(doc_to_own_ref),
        "unique_canonical_references_count": len(ref_to_docs),
        "collision_groups_count": len(collision_groups),
    }
    return dict(ref_to_docs), doc_to_own_ref, audit


def build_explicit_relation_graph_local(corpus, ref_to_docs, doc_to_own_ref):
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    amend_count = 0
    guide_count = 0
    header_edge_count = 0

    for doc_id, item in corpus.items():
        passage = item.get("passage", "")
        own_ref = doc_to_own_ref.get(doc_id)

        for m in REF_REGEX.finditer(passage):
            target_ref = canonicalize_ref(m.group(0))
            if own_ref and target_ref == own_ref:
                continue
            target_doc_ids = ref_to_docs.get(target_ref, [])
            if not target_doc_ids:
                continue

            start, end = m.span()
            ctx_before = passage[max(0, start - 80):start].lower()
            ctx_after = passage[end:min(len(passage), end + 80)].lower()
            snippet = (
                passage[max(0, start - 50):min(len(passage), end + 50)]
                .strip()
                .replace("\n", " ")
            )

            rel_family = None
            if (
                any(t in ctx_before for t in AMEND_TRIGGERS)
                or any(t in ctx_after for t in AMEND_TRIGGERS)
            ):
                rel_family = "AMENDMENT_REPLACEMENT_REPEAL"
                amend_count += 1
            elif (
                any(t in ctx_before for t in GUIDE_TRIGGERS)
                or any(t in ctx_after for t in GUIDE_TRIGGERS)
            ):
                rel_family = "IMPLEMENTATION_GUIDANCE"
                guide_count += 1

            if rel_family:
                is_header = start < 1200
                if is_header:
                    header_edge_count += 1
                for td in target_doc_ids:
                    if td == doc_id:
                        continue
                    edge = (td, rel_family, is_header, snippet)
                    rev_edge = (doc_id, rel_family, is_header, snippet)
                    if edge not in out_edges[doc_id]:
                        out_edges[doc_id].append(edge)
                    if rev_edge not in in_edges[td]:
                        in_edges[td].append(rev_edge)

    audit = {
        "status": "PASS",
        "nodes_with_out_edges": len(out_edges),
        "nodes_with_in_edges": len(in_edges),
        "total_directed_edges": sum(len(v) for v in out_edges.values()),
        "relation_family_breakdown": {
            "amendment_replacement_repeal_mentions": amend_count,
            "implementation_guidance_mentions": guide_count,
            "header_evidence_count": header_edge_count,
        },
    }
    return dict(out_edges), dict(in_edges), audit


def extract_query_references_local(query_text: str):
    raw_matches = REF_REGEX.findall(query_text or "")
    refs = []
    for m in raw_matches:
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", m):
            continue
        c_ref = canonicalize_ref(m)
        if (
            len(c_ref) >= 3
            and any(c.isdigit() for c in c_ref)
            and c_ref not in refs
        ):
            refs.append(c_ref)
    return refs


def generate_query_additions_local(
    query_text: str,
    existing_pool,
    ref_to_docs,
    out_edges,
    in_edges,
    cap: int = MAX_ADDITIONS,
):
    query_refs = extract_query_references_local(query_text)
    if not query_refs:
        return [], [], []

    exact_docs = []
    exact_details = {}
    for r in query_refs:
        for d in ref_to_docs.get(r, []):
            if d not in exact_docs:
                exact_docs.append(d)
                exact_details[d] = {
                    "doc_id": d,
                    "addition_type": "DIRECT_REFERENCE_MATCH",
                    "anchor_ref": r,
                    "relation_family": "EXACT_OWN_REFERENCE",
                    "evidence_snippet": f"Matched own legal reference: {r}",
                }

    header_neighbors = []
    body_neighbors = []
    neighbor_details = {}

    for d in exact_docs:
        for td, rel_fam, is_h, snip in out_edges.get(d, []):
            if td not in exact_docs:
                if is_h and td not in header_neighbors:
                    header_neighbors.append(td)
                elif not is_h and td not in body_neighbors:
                    body_neighbors.append(td)
                if td not in neighbor_details:
                    neighbor_details[td] = {
                        "doc_id": td,
                        "addition_type": "RELATION_NEIGHBOR",
                        "anchor_doc": d,
                        "relation_direction": "OUTGOING",
                        "relation_family": rel_fam,
                        "is_header": is_h,
                        "evidence_snippet": snip,
                    }

        for sd, rel_fam, is_h, snip in in_edges.get(d, []):
            if sd not in exact_docs:
                if is_h and sd not in header_neighbors:
                    header_neighbors.append(sd)
                elif not is_h and sd not in body_neighbors:
                    body_neighbors.append(sd)
                if sd not in neighbor_details:
                    neighbor_details[sd] = {
                        "doc_id": sd,
                        "addition_type": "RELATION_NEIGHBOR",
                        "anchor_doc": d,
                        "relation_direction": "INCOMING",
                        "relation_family": rel_fam,
                        "is_header": is_h,
                        "evidence_snippet": snip,
                    }

    header_neighbors.sort(key=lambda x: int(x) if x.isdigit() else x)
    body_neighbors.sort(key=lambda x: int(x) if x.isdigit() else x)

    candidates_stream = exact_docs + header_neighbors + body_neighbors
    final_additions = []
    final_details = []

    for d in candidates_stream:
        if d not in final_additions and d not in existing_pool:
            final_additions.append(d)
            final_details.append(
                exact_details.get(d)
                or neighbor_details.get(d)
                or {"doc_id": d}
            )
        if len(final_additions) >= cap:
            break

    return final_additions, final_details, query_refs


def candidate_fingerprint(ids, pools):
    h = hashlib.sha256()
    for q in sorted(ids):
        h.update(f"{q}:".encode("utf-8"))
        for d in sorted(pools[q]):
            h.update(f"{d},".encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_pkl_local(root: Path, rel: str):
    obj = pickle.loads((root / rel).read_bytes())
    return obj


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

    base_pub = load_pkl_local(
        root, "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]

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

    fp = candidate_fingerprint(ids, pools)

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
    corpus = load_corpus(contexts)
    ref_to_docs, doc_to_own_ref, index_audit = (
        build_legal_reference_index_local(corpus)
    )
    out_edges, in_edges, relation_audit = (
        build_explicit_relation_graph_local(
            corpus,
            ref_to_docs,
            doc_to_own_ref,
        )
    )
    print(
        f"  corpus_docs={len(corpus)} "
        f"indexed_refs={index_audit['unique_canonical_references_count']} "
        f"relation_edges={relation_audit['total_directed_edges']}",
        flush=True,
    )

    print("[4/8] Generating public additions with frozen CAL-promoted policy...")
    generated = {}
    triggered = []
    total_additions = 0

    for q in ids:
        adds, details, refs = generate_query_additions_local(
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
        "frozen_logic_source_provenance": {
            "legal_ref_indexer": {
                "path": str(root / "src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/legal_ref_indexer.py"),
                "sha256": sha256(root / "src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/legal_ref_indexer.py"),
            },
            "relation_graph": {
                "path": str(root / "src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/relation_graph.py"),
                "sha256": sha256(root / "src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/relation_graph.py"),
            },
            "query_anchored_generator": {
                "path": str(root / "src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/query_anchored_generator.py"),
                "sha256": sha256(root / "src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/query_anchored_generator.py"),
            },
        },
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
