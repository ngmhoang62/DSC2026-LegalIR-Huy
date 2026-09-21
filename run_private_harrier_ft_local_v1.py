#!/usr/bin/env python
"""
LOCAL RTX4050 HAR(R)IER-FT FULL-CORPUS PRIVATE RETRIEVAL
========================================================

Source-faithful inference contract recovered from
results/from_drive/fine_tune_source_code/vietlegal-tune-kaggle-hnsw.ipynb:

  base: mainguyen9/vietlegal-harrier-0.6b
  max_length: 512
  fp16 base/model
  last-token pooling
  L2-normalized float32 embedding -> stored fp16
  query prefix:
    "Instruct: Given a Vietnamese legal question, retrieve relevant legal "
    "passages that answer the question\\nQuery: "
  document chunks: no prefix
  HNSW: cosine, M=16, ef_construction=200, ef_search=300
  retrieve top-200 CHUNKS (not deduplicated parents)

This script is designed for one local GPU (e.g. RTX4050 6GB) and is resumable.

FAIL-CLOSED corpus rule
-----------------------
The exact fine-tune source says the original chunk-context signature must be:
    8532 JSON files, total bytes = 380681028

This runner DOES NOT regenerate chunk-context approximately. If your folder is
missing or has a different signature, it stops unless you explicitly pass
--allow-signature-mismatch.

Outputs
-------
results/manual/huy_private_harrier_ft_v1/
  cache/corpus_emb.fp16.npy
  cache/corpus_manifest.npz
  cache/corpus_encode_progress.json
  cache/harrier_ft_hnsw.bin
  cache/HNSW_META.json
  PRIVATE_HARRIER_TOP200.json
  PRIVATE_HARRIER_PARENT_RANKS.pkl
  PRIVATE_HARRIER_REPORT.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

EXPECTED_FILES = 8532
EXPECTED_BYTES = 380_681_028
MODEL_NAME = "mainguyen9/vietlegal-harrier-0.6b"
MAX_SEQ_LEN = 512
QUERY_INSTRUCT = (
    "Instruct: Given a Vietnamese legal question, retrieve relevant legal "
    "passages that answer the question\nQuery: "
)
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 300
TOP_SAVE = 200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_pickle(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=5))
    os.replace(tmp, path)


def corpus_signature(chunk_dir: Path):
    files = sorted(
        p for p in chunk_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".json"
    )
    return files, (len(files), sum(p.stat().st_size for p in files))


def find_chunk_dir(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p

    candidates = [
        root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/chunk-context",
        root / "DSC2026-LegalIR-main/v4_run/chunk-context",
        root / "DSC2026-LegalIR-main/data/chunk-context",
        root / "data/chunk-context",
        root / "chunk-context",
    ]
    hits = [p for p in candidates if p.is_dir()]
    if len(hits) == 1:
        return hits[0].resolve()
    if len(hits) > 1:
        raise RuntimeError(
            "Multiple chunk-context directories found; pass --chunk-dir:\n"
            + "\n".join(f"  {p}" for p in hits)
        )
    raise FileNotFoundError(
        "Exact chunk-context directory not found. Pass --chunk-dir. "
        "Do NOT substitute selected-contexts."
    )


def find_private_file(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    for name in (
        "private-official.json",
        "private_official.json",
        "private.json",
    ):
        p = data / name
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"Private query file not found under {data}; pass --private-file."
    )


def find_optional_train_results(root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    hits = list(root.rglob("train_finetuned_best.json"))
    if len(hits) == 1:
        return hits[0].resolve()
    return None


def find_merged_model(root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p

    names = [
        root / "results/from_drive/vietlegal/vietlegal_finetuned/final_model_merged",
        root / "results/from_drive/vietlegal/final_model_merged",
        root / "results/from_drive/vietlegal_finetuned/final_model_merged",
    ]
    for p in names:
        if p.is_dir() and any(p.glob("*.safetensors")):
            return p.resolve()
    return None


def find_adapter(root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    names = [
        root / "results/from_drive/vietlegal/vietlegal_finetuned/best_adapter",
        root / "results/from_drive/vietlegal/best_adapter",
        root / "results/from_drive/vietlegal_finetuned/best_adapter",
    ]
    for p in names:
        if p.is_dir() and (p / "adapter_config.json").is_file():
            return p.resolve()
    return None


def load_model_and_tokenizer(root, merged_dir, adapter_dir, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()

    if merged_dir is not None:
        print(f"[model] loading merged checkpoint: {merged_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            merged_dir,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            merged_dir,
            dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        identity = {
            "mode": "merged_model",
            "path": str(merged_dir),
            "model_file": next(
                (str(p) for p in merged_dir.glob("*.safetensors")),
                None,
            ),
        }
    else:
        print(f"[model] loading base from HF: {MODEL_NAME}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        base = AutoModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        if adapter_dir is None:
            raise RuntimeError(
                "No merged model or best_adapter found. "
                "Pass --merged-model-dir or --adapter-dir."
            )
        print(f"[model] attaching adapter: {adapter_dir}", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base,
            adapter_dir,
            is_trainable=False,
        ).to(device)
        identity = {
            "mode": "base_plus_adapter",
            "base": MODEL_NAME,
            "adapter": str(adapter_dir),
            "adapter_sha256": (
                sha256(adapter_dir / "adapter_model.safetensors")
                if (adapter_dir / "adapter_model.safetensors").is_file()
                else None
            ),
        }

    model.eval()
    return model, tokenizer, identity


def last_token_pool(last_hidden, attention_mask):
    import torch
    left_padding = (
        attention_mask[:, -1].sum() == attention_mask.shape[0]
    )
    if bool(left_padding):
        return last_hidden[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(
        last_hidden.shape[0],
        device=last_hidden.device,
    )
    return last_hidden[batch_idx, seq_lengths]


def encode_texts_adaptive(
    texts,
    *,
    model,
    tokenizer,
    device,
    batch_size,
    desc,
    max_length=MAX_SEQ_LEN,
):
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    out = []
    i = 0
    bs = int(batch_size)
    bar = tqdm(total=len(texts), desc=desc, leave=False)

    with torch.inference_mode():
        while i < len(texts):
            chunk = texts[i:i + bs]
            try:
                enc = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                hidden = model(**enc).last_hidden_state
                emb = last_token_pool(hidden, enc["attention_mask"])
                emb = F.normalize(emb.float(), p=2, dim=1)
                arr = emb.half().cpu().numpy()
                out.append(arr)

                i += len(chunk)
                bar.update(len(chunk))
                del enc, hidden, emb
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 1:
                    raise
                old = bs
                bs = max(1, bs // 2)
                print(
                    f"\n[OOM] {desc}: batch {old} -> {bs}; retry",
                    flush=True,
                )

    bar.close()
    return np.concatenate(out, axis=0), bs


def build_manifest(chunk_files: list[Path], cache_dir: Path):
    manifest_path = cache_dir / "corpus_manifest.npz"
    meta_path = cache_dir / "CORPUS_MANIFEST_META.json"

    signature = (
        len(chunk_files),
        sum(p.stat().st_size for p in chunk_files),
    )

    if manifest_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if tuple(meta.get("signature", [])) == signature:
            m = np.load(manifest_path)
            print(
                f"[manifest] reuse {len(m['ctx_ids']):,} chunks",
                flush=True,
            )
            return (
                np.asarray(m["ctx_ids"], dtype=np.int32),
                np.asarray(m["chunk_idx"], dtype=np.int32),
                signature,
            )

    ctx_ids = []
    chunk_idx = []
    n_files = len(chunk_files)

    print("[manifest] scanning exact chunk order...", flush=True)
    for i, p in enumerate(chunk_files, 1):
        row = json.loads(p.read_text(encoding="utf-8"))
        cid = int(row["id"])
        for k, text in row["chunk"].items():
            if not str(text).strip():
                continue
            ctx_ids.append(cid)
            chunk_idx.append(int(k))
        if i % 500 == 0 or i == n_files:
            print(
                f"  files {i:,}/{n_files:,} | chunks={len(ctx_ids):,}",
                flush=True,
            )

    ctx_ids = np.asarray(ctx_ids, dtype=np.int32)
    chunk_idx = np.asarray(chunk_idx, dtype=np.int32)
    np.savez(manifest_path, ctx_ids=ctx_ids, chunk_idx=chunk_idx)
    atomic_json(
        meta_path,
        {
            "schema": "manual.harrier_corpus_manifest.v1",
            "signature": list(signature),
            "chunks": int(len(ctx_ids)),
        },
    )
    return ctx_ids, chunk_idx, signature


def iter_chunk_texts(chunk_files):
    global_idx = 0
    for p in chunk_files:
        row = json.loads(p.read_text(encoding="utf-8"))
        for k, text in row["chunk"].items():
            text = str(text)
            if not text.strip():
                continue
            yield global_idx, text
            global_idx += 1


def encode_corpus_resume(
    *,
    chunk_files,
    n_chunks,
    model,
    tokenizer,
    device,
    cache_dir,
    model_identity,
    signature,
    batch_size,
):
    import torch

    emb_path = cache_dir / "corpus_emb.fp16.npy"
    progress_path = cache_dir / "corpus_encode_progress.json"
    meta_path = cache_dir / "CORPUS_EMB_META.json"

    expected_contract = {
        "model_identity": model_identity,
        "signature": list(signature),
        "max_seq_len": MAX_SEQ_LEN,
        "pooling": "last_token",
        "normalize": "L2_float32_then_fp16",
        "query_prefix_for_corpus": None,
    }

    # Need one small forward to infer embedding dimension if cache absent.
    if emb_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if {
            k: meta.get(k) for k in expected_contract
        } == expected_contract:
            mmap = np.load(emb_path, mmap_mode="r+")
            if mmap.shape[0] != n_chunks:
                raise RuntimeError(
                    f"Embedding cache row count {mmap.shape[0]} != {n_chunks}"
                )
            next_index = 0
            if progress_path.is_file():
                prog = json.loads(progress_path.read_text(encoding="utf-8"))
                next_index = int(prog.get("next_index", 0))
            if bool(meta.get("complete")):
                print(
                    f"[corpus] complete cache reuse: {emb_path} {mmap.shape}",
                    flush=True,
                )
                return mmap
            print(
                f"[corpus] resume at {next_index:,}/{n_chunks:,}",
                flush=True,
            )
            dim = mmap.shape[1]
        else:
            raise RuntimeError(
                "Existing corpus embedding cache has different contract. "
                "Refusing overwrite; move/delete that cache manually."
            )
    else:
        first_text = next(iter_chunk_texts(chunk_files))[1]
        sample, _ = encode_texts_adaptive(
            [first_text],
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=1,
            desc="infer dim",
        )
        dim = int(sample.shape[1])
        mmap = np.lib.format.open_memmap(
            emb_path,
            mode="w+",
            dtype=np.float16,
            shape=(n_chunks, dim),
        )
        next_index = 0
        atomic_json(
            meta_path,
            {
                "schema": "manual.harrier_corpus_emb.v1",
                **expected_contract,
                "shape": [n_chunks, dim],
                "dtype": "float16",
                "complete": False,
            },
        )
        atomic_json(
            progress_path,
            {"next_index": 0, "batch_size": batch_size},
        )

    current_bs = int(batch_size)
    pending_texts = []
    pending_indices = []
    started = time.perf_counter()
    done_this_run = 0

    def flush():
        nonlocal current_bs, done_this_run
        if not pending_texts:
            return
        arr, current_bs = encode_texts_adaptive(
            pending_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=current_bs,
            desc="corpus batch",
        )
        idx = np.asarray(pending_indices, dtype=np.int64)
        mmap[idx] = arr
        mmap.flush()
        done_this_run += len(idx)
        next_i = int(idx[-1] + 1)
        atomic_json(
            progress_path,
            {
                "next_index": next_i,
                "batch_size": current_bs,
                "elapsed_seconds_this_run": (
                    time.perf_counter() - started
                ),
            },
        )
        elapsed = time.perf_counter() - started
        rate = done_this_run / max(elapsed, 1e-9)
        eta = (n_chunks - next_i) / max(rate, 1e-9) / 60
        print(
            f"[corpus] {next_i:,}/{n_chunks:,} "
            f"({100*next_i/n_chunks:5.1f}%) | "
            f"{rate:.1f} chunks/s | ETA {eta:.1f}m | batch={current_bs}",
            flush=True,
        )
        pending_texts.clear()
        pending_indices.clear()

    # Encode in checkpoint chunks of about 4096 texts, but inner GPU batches
    # remain adaptive.
    flush_every = max(1024, current_bs * 32)
    for idx, text in iter_chunk_texts(chunk_files):
        if idx < next_index:
            continue
        pending_indices.append(idx)
        pending_texts.append(text)
        if len(pending_texts) >= flush_every:
            flush()
    flush()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["complete"] = True
    meta["completed_at_unix"] = time.time()
    atomic_json(meta_path, meta)
    atomic_json(
        progress_path,
        {"next_index": n_chunks, "batch_size": current_bs},
    )
    print("[corpus] COMPLETE", flush=True)
    return np.load(emb_path, mmap_mode="r")


def build_or_load_hnsw(
    corpus_emb,
    cache_dir: Path,
    contract_hash: str,
):
    import hnswlib

    index_path = cache_dir / "harrier_ft_hnsw.bin"
    meta_path = cache_dir / "HNSW_META.json"
    dim = int(corpus_emb.shape[1])
    n = int(corpus_emb.shape[0])

    if index_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("contract_hash") == contract_hash
            and int(meta.get("n", -1)) == n
            and int(meta.get("dim", -1)) == dim
        ):
            print(f"[HNSW] loading cached index: {index_path}", flush=True)
            index = hnswlib.Index(space="cosine", dim=dim)
            index.load_index(str(index_path), max_elements=n)
            index.set_ef(HNSW_EF_SEARCH)
            return index

    print(
        f"[HNSW] building n={n:,} dim={dim} "
        f"M={HNSW_M} efc={HNSW_EF_CONSTRUCTION}",
        flush=True,
    )
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=n,
        ef_construction=HNSW_EF_CONSTRUCTION,
        M=HNSW_M,
    )

    # Add in original corpus order, in blocks to cap temporary float32 RAM.
    block = 25_000
    t0 = time.perf_counter()
    for start in range(0, n, block):
        end = min(start + block, n)
        x = np.asarray(corpus_emb[start:end], dtype=np.float32)
        ids = np.arange(start, end, dtype=np.int64)
        index.add_items(x, ids)
        if start == 0 or end == n or (start // block) % 4 == 0:
            print(
                f"  [HNSW] add {end:,}/{n:,} "
                f"| {time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    index.set_ef(HNSW_EF_SEARCH)
    index.save_index(str(index_path))
    atomic_json(
        meta_path,
        {
            "schema": "manual.harrier_hnsw.v1",
            "contract_hash": contract_hash,
            "n": n,
            "dim": dim,
            "M": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "ef_search": HNSW_EF_SEARCH,
            "index_path": str(index_path),
        },
    )
    print(f"[HNSW] saved: {index_path}", flush=True)
    return index


def retrieve_queries(
    *,
    questions,
    model,
    tokenizer,
    device,
    index,
    ctx_ids,
    chunk_idx,
    query_batch,
    k=TOP_SAVE,
):
    qids = sorted(questions)
    qtexts = [QUERY_INSTRUCT + questions[q] for q in qids]
    qemb, final_bs = encode_texts_adaptive(
        qtexts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=query_batch,
        desc="encode private queries",
    )

    labels, distances = index.knn_query(
        np.asarray(qemb, dtype=np.float32),
        k=min(k, len(ctx_ids)),
    )

    out = {}
    for j, qid in enumerate(qids):
        out[qid] = {
            "question": questions[qid],
            "results": [
                {
                    "ctx_id": int(ctx_ids[int(idx)]),
                    "chunk": int(chunk_idx[int(idx)]),
                    "score": round(1.0 - float(distances[j, t]), 4),
                }
                for t, idx in enumerate(labels[j])
            ],
        }
    return out, final_bs


def parent_ranks(raw):
    out = {}
    for q, row in raw.items():
        rank = {}
        pos = 0
        for x in row.get("results", []):
            d = str(x["ctx_id"])
            if d in rank:
                continue
            pos += 1
            rank[d] = pos
        out[str(q)] = rank
    return out


def calibration_audit(
    *,
    root,
    train_results_path,
    model,
    tokenizer,
    device,
    index,
    ctx_ids,
    chunk_idx,
    query_batch,
    n_queries,
):
    if train_results_path is None or n_queries <= 0:
        return {"status": "SKIPPED"}

    saved = json.loads(train_results_path.read_text(encoding="utf-8"))
    data_dir = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    train_path = data_dir / "train.json"
    if not train_path.is_file():
        return {"status": "SKIPPED_TRAIN_JSON_MISSING"}

    train = json.loads(train_path.read_text(encoding="utf-8"))
    common = [
        q for q in sorted(saved)
        if q in train
        and str(saved[q].get("question", "")).strip()
        == str(train[q].get("question", "")).strip()
    ][:n_queries]
    if not common:
        return {"status": "SKIPPED_NO_MATCHED_QIDS"}

    questions = {q: train[q]["question"] for q in common}
    fresh, _ = retrieve_queries(
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        device=device,
        index=index,
        ctx_ids=ctx_ids,
        chunk_idx=chunk_idx,
        query_batch=query_batch,
        k=TOP_SAVE,
    )

    rows = []
    for q in common:
        a = list(parent_ranks({q: saved[q]})[q])
        b = list(parent_ranks({q: fresh[q]})[q])
        rows.append({
            "qid": q,
            "saved_top1": a[0] if a else None,
            "fresh_top1": b[0] if b else None,
            "fresh_top1_in_saved_top5": bool(
                b and b[0] in set(a[:5])
            ),
            "top5_overlap": len(set(a[:5]) & set(b[:5])) / 5.0,
            "top10_overlap": len(set(a[:10]) & set(b[:10])) / 10.0,
        })

    rate = float(np.mean([r["fresh_top1_in_saved_top5"] for r in rows]))
    top5 = float(np.mean([r["top5_overlap"] for r in rows]))
    top10 = float(np.mean([r["top10_overlap"] for r in rows]))
    status = "PASS" if rate >= 0.50 else "FAIL"

    result = {
        "status": status,
        "queries": len(rows),
        "fresh_top1_in_saved_top5_rate": rate,
        "mean_parent_top5_overlap": top5,
        "mean_parent_top10_overlap": top10,
        "rows": rows,
    }
    if status == "FAIL":
        raise RuntimeError(
            "Harrier calibration parity catastrophically low: "
            f"fresh_top1_in_saved_top5={rate:.3f}. "
            "Likely wrong checkpoint/chunk corpus/contract."
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--chunk-dir", type=Path, default=None)
    ap.add_argument("--private-file", type=Path, default=None)
    ap.add_argument("--merged-model-dir", type=Path, default=None)
    ap.add_argument("--adapter-dir", type=Path, default=None)
    ap.add_argument("--train-results", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--query-batch-size", type=int, default=64)
    ap.add_argument("--calibration-queries", type=int, default=20)
    ap.add_argument("--allow-signature-mismatch", action="store_true")
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    out = root / "results/manual/huy_private_harrier_ft_v1"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = args.device
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        props = torch.cuda.get_device_properties(0)
        print(
            f"[GPU] {props.name} | {props.total_memory/2**30:.2f} GiB",
            flush=True,
        )

    chunk_dir = find_chunk_dir(root, args.chunk_dir)
    chunk_files, signature = corpus_signature(chunk_dir)
    print(f"[corpus] {chunk_dir}", flush=True)
    print(
        f"[corpus] signature files={signature[0]} bytes={signature[1]}",
        flush=True,
    )
    if signature != (EXPECTED_FILES, EXPECTED_BYTES):
        msg = (
            f"Exact chunk signature mismatch: got={signature}, "
            f"expected={(EXPECTED_FILES, EXPECTED_BYTES)}"
        )
        if not args.allow_signature_mismatch:
            raise RuntimeError(
                msg + ". Refusing approximate corpus. "
                "Pass --allow-signature-mismatch only if intentional."
            )
        print("WARNING:", msg, flush=True)

    private_path = find_private_file(root, args.private_file)
    private_raw = json.loads(private_path.read_text(encoding="utf-8"))
    questions = {
        str(q): str(v["question"])
        for q, v in private_raw.items()
    }
    if len(questions) != 2080:
        raise RuntimeError(
            f"Expected 2080 private queries, got {len(questions)}"
        )

    merged_dir = find_merged_model(root, args.merged_model_dir)
    adapter_dir = None if merged_dir else find_adapter(root, args.adapter_dir)
    if merged_dir is None and adapter_dir is None:
        raise RuntimeError(
            "No Harrier FT checkpoint found. Download either final_model_merged "
            "or best_adapter locally, then pass --merged-model-dir or --adapter-dir."
        )

    model, tokenizer, model_identity = load_model_and_tokenizer(
        root, merged_dir, adapter_dir, device
    )

    ctx_ids, chunk_idx, signature = build_manifest(chunk_files, cache)
    n_chunks = len(ctx_ids)
    print(f"[corpus] non-empty chunks={n_chunks:,}", flush=True)

    corpus_emb = encode_corpus_resume(
        chunk_files=chunk_files,
        n_chunks=n_chunks,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache_dir=cache,
        model_identity=model_identity,
        signature=signature,
        batch_size=args.batch_size,
    )

    emb_meta_path = cache / "CORPUS_EMB_META.json"
    contract_hash = hashlib.sha256(
        emb_meta_path.read_bytes()
    ).hexdigest()

    # Free transient CUDA cache before HNSW CPU build.
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    index = build_or_load_hnsw(
        corpus_emb=corpus_emb,
        cache_dir=cache,
        contract_hash=contract_hash,
    )

    train_results = find_optional_train_results(root, args.train_results)
    print("[audit] checkpoint/corpus parity sample...", flush=True)
    calibration = calibration_audit(
        root=root,
        train_results_path=train_results,
        model=model,
        tokenizer=tokenizer,
        device=device,
        index=index,
        ctx_ids=ctx_ids,
        chunk_idx=chunk_idx,
        query_batch=args.query_batch_size,
        n_queries=args.calibration_queries,
    )
    print(
        "[audit]",
        json.dumps(
            {k: v for k, v in calibration.items() if k != "rows"},
            ensure_ascii=False,
        ),
        flush=True,
    )

    print("[private] retrieving exact-source top200 chunks...", flush=True)
    private_out, final_qbs = retrieve_queries(
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        device=device,
        index=index,
        ctx_ids=ctx_ids,
        chunk_idx=chunk_idx,
        query_batch=args.query_batch_size,
        k=TOP_SAVE,
    )

    result_path = out / "PRIVATE_HARRIER_TOP200.json"
    atomic_json(result_path, private_out)

    pr = parent_ranks(private_out)
    parent_rank_path = out / "PRIVATE_HARRIER_PARENT_RANKS.pkl"
    atomic_pickle(parent_rank_path, pr)

    counts = [len(v) for v in pr.values()]
    report = {
        "schema": "manual.private_harrier_ft_v1",
        "status": "READY_FOR_D1_HARRIER_MATERIALIZATION",
        "source_contract": {
            "base_model": MODEL_NAME,
            "max_seq_len": MAX_SEQ_LEN,
            "pooling": "last_token",
            "normalization": "L2",
            "query_instruction": QUERY_INSTRUCT,
            "document_prefix": None,
            "hnsw": {
                "space": "cosine",
                "M": HNSW_M,
                "ef_construction": HNSW_EF_CONSTRUCTION,
                "ef_search": HNSW_EF_SEARCH,
                "top_save_chunks": TOP_SAVE,
            },
        },
        "model_identity": model_identity,
        "corpus": {
            "chunk_dir": str(chunk_dir),
            "signature": list(signature),
            "chunks": n_chunks,
            "embedding_shape": list(corpus_emb.shape),
        },
        "private": {
            "file": str(private_path),
            "sha256": sha256(private_path),
            "queries": len(questions),
            "query_batch_final": final_qbs,
            "distinct_parent_rank_min": int(min(counts)),
            "distinct_parent_rank_mean": float(np.mean(counts)),
            "distinct_parent_rank_max": int(max(counts)),
        },
        "calibration": calibration,
        "outputs": {
            "top200_chunks": str(result_path),
            "top200_chunks_sha256": sha256(result_path),
            "parent_ranks": str(parent_rank_path),
            "parent_ranks_sha256": sha256(parent_rank_path),
        },
    }
    report_path = out / "PRIVATE_HARRIER_REPORT.json"
    atomic_json(report_path, report)

    print("=" * 108)
    print("HAR(R)IER PRIVATE RETRIEVAL READY")
    print("Top200 chunks:", result_path)
    print("Parent ranks:", parent_rank_path)
    print("Report:", report_path)
    print("=" * 108)


if __name__ == "__main__":
    main()
