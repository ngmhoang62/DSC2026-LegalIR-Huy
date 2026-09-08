"""Title-embedding similarity for the PUBLIC candidate pool.

Mirrors tune_title_embedding.py exactly (AITeamVN CLS encoder, 128-token
window, question vector dotted with each candidate's extracted title vector),
but for the public queries. The pool is taken from the cached full-pool
cross-encoder scores, whose key set is the public candidate set the runner
builds, so candidate generation is not repeated.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from run_burst_expanded_fusion_submission import DocumentStore
from tune_title_features import title_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/burst_fresh_block/title_embed_public.pkl")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    docs = DocumentStore(sorted((data / "selected-contexts").glob("context_*.json")))
    public = json.loads((data / "public-official.json").read_text(encoding="utf-8"))
    public = {q: (v["question"] if isinstance(v, dict) else v) for q, v in public.items()}
    pool = pickle.loads(
        (root / "results/crossenc_fullpool/public_scores.pkl").read_bytes())["scores"]
    ids = [q for q in public if q in pool]
    candidates = {q: list(pool[q]) for q in ids}
    print(f"public queries {len(ids)}, mean pool "
          f"{sum(len(candidates[q]) for q in ids) / len(ids):.1f}", flush=True)

    titles = title_table(docs, ids, candidates)
    print(f"titles extracted for {len(titles)} documents", flush=True)

    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to("cuda")
    print(f"encoder ready on {torch.cuda.get_device_name(0)}", flush=True)

    doc_ids = list(titles)
    started = time.perf_counter()
    title_vecs = encode_cls(model, tokenizer,
                            [titles[d] or "khong co tieu de" for d in doc_ids], 64, 128)
    print(f"embedded {len(doc_ids)} titles in {time.perf_counter()-started:.1f}s",
          flush=True)
    tmap = dict(zip(doc_ids, title_vecs))
    q_vecs = encode_cls(model, tokenizer, [public[q] for q in ids], 32, 128)
    qmap = dict(zip(ids, q_vecs))
    del model
    torch.cuda.empty_cache()

    out = {q: {d: float(qmap[q] @ tmap[d]) for d in candidates[q]} for q in ids}
    path = root / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out, protocol=5))
    print(f"Saved {path} ({len(out)} queries)", flush=True)


if __name__ == "__main__":
    main()
