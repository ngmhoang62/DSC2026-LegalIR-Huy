#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, pickle, sys
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--workload", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import DocumentStore

    workload = args.workload.resolve() if args.workload else (
        root / "results/manual/huy_private_prism_v1/PRISM_PRIVATE_WORKLOAD.jsonl"
    )
    if not workload.is_file():
        raise FileNotFoundError(workload)

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    corpus_paths = sorted((data / "selected-contexts").glob("context_*.json"))
    documents = DocumentStore(corpus_paths)

    rows = [json.loads(x) for x in workload.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 2080:
        raise RuntimeError(f"Expected 2080 rows, got {len(rows)}")

    queries = {}
    total_docs = 0
    total_passages = 0
    for i, row in enumerate(rows, 1):
        qid = str(row["qid"])
        question = str(row["question"])
        qdocs = {}
        for d in map(str, row["candidate_doc_ids"]):
            ps = [str(p) for p in top_passages(question, documents[d], count=2) if str(p).strip()]
            if not ps:
                ps = [str(documents[d] or "")]
            qdocs[d] = ps
            total_docs += 1
            total_passages += len(ps)
        queries[qid] = {"question": question, "docs": qdocs}
        if i % 100 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} queries | docs={total_docs:,} passages={total_passages:,}", flush=True)

    obj = {
        "schema": "manual.prism_private_passages.v2",
        "mode": "top2_max",
        "max_length": 1024,
        "source_workload_sha256": sha256(workload),
        "queries": queries,
        "stats": {"queries": len(queries), "docs": total_docs, "passages": total_passages},
    }

    outdir = root / "results/manual/huy_private_prism_v1"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "PRISM_PRIVATE_PASSAGES.pkl"
    out.write_bytes(pickle.dumps(obj, protocol=5))
    meta = {
        "path": str(out),
        "sha256": sha256(out),
        "queries": len(queries),
        "docs": total_docs,
        "passages": total_passages,
        "mode": "top2_max",
        "max_length": 1024,
    }
    (outdir / "PRISM_PRIVATE_PASSAGES_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("="*96)
    print("READY:", out)
    print(json.dumps(meta, indent=2))
    print("="*96)

if __name__ == "__main__":
    main()
