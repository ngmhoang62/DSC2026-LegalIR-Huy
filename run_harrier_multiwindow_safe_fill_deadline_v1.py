#!/usr/bin/env python
"""
HUY DEADLINE — HAR(R)IER MULTI-WINDOW RERANK ON REL_L0 SAFE SLOTS V1
===================================================================

Motivation:
  The first public Harrier experiment used one prefix vector per parent document.
  Audit against saved exact warmup all-chunk HNSW retrieval showed only ~0.42
  Top-5 parent overlap, so that proxy is not faithful enough.

This script:
  1) Starts from the already-built proxy Harrier Top-50 parent candidates.
  2) For only docs needed by:
       - 127 clean REL_L0-safe public queries
       - public queries that exactly match saved warmup questions (teacher audit)
     generate up to N evenly-spaced 220-word windows from the full selected-context
     passage, encode them with the fine-tuned Harrier, and parent-aggregate by MAX.
  3) Re-rank the Top-50 parents per query by max-window cosine.
  4) Compare the new rerank to the SAVED exact all-chunk warmup parent ranking on
     identical questions.  This is retrieval-fidelity validation, NOT label tuning.
  5) Materialize clean safe-slot submissions using reranked outsider positions
     1, 2, and 3.

No public labels are read.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

QUERY_INSTRUCT = (
    "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages "
    "that answer the question\nQuery: "
)
WS_RE = re.compile(r"\s+")


def norm(s):
    return WS_RE.sub(
        " ",
        unicodedata.normalize("NFKC", str(s or "")).lower().strip(),
    )


def get_answer(v):
    if isinstance(v, dict):
        return [str(x) for x in v.get("answer", [])]
    return [str(x) for x in v]


def qtext(v):
    if isinstance(v, dict):
        return str(v.get("question") or v.get("query") or v.get("text") or "")
    return str(v)


def find_named(base: Path, name: str):
    p = base / name
    if p.is_file():
        return p
    hits = list(base.rglob(name))
    return hits[0] if hits else None


def find_merged_model(model_root: Path):
    for p in (
        model_root / "vietlegal_finetuned" / "final_model_merged",
        model_root / "final_model_merged",
        model_root,
    ):
        if (p / "model.safetensors").is_file() and (p / "config.json").is_file():
            return p
    for m in model_root.rglob("model.safetensors"):
        if (m.parent / "config.json").is_file():
            return m.parent
    raise FileNotFoundError("Merged Harrier model not found")


def find_train_config(model_root: Path):
    for p in [model_root / "config.json", *model_root.rglob("config.json")]:
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
            if "max_seq_len" in x and "model_name" in x:
                return p, x
        except Exception:
            pass
    return None, {}


def last_token_pool(last_hidden, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if bool(left_padding):
        return last_hidden[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_idx, seq_lengths]


@torch.inference_mode()
def encode(model, tok, texts, max_length):
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
    h = model(**enc).last_hidden_state
    e = last_token_pool(h, enc["attention_mask"])
    return F.normalize(e.float(), p=2, dim=1)


def sample_windows(text: str, window_words=220, max_windows=6):
    words = str(text).split()
    n = len(words)
    if n <= window_words:
        return [" ".join(words)] if words else [""]
    max_start = max(0, n - window_words)
    starts = np.linspace(
        0, max_start, num=min(max_windows, math.ceil(n / window_words)),
        dtype=np.int64,
    )
    starts = sorted(set(int(x) for x in starts))
    return [" ".join(words[s:s + window_words]) for s in starts]


def load_doc_text(path: Path):
    row = json.loads(path.read_text(encoding="utf-8"))
    text = row.get("passage")
    if text is None:
        raise KeyError(f"'passage' missing in {path}")
    return str(text)


def dedup_parent_results(results, limit=200):
    out = []
    for x in results:
        d = str(x.get("ctx_id", x.get("doc_id", ""))) if isinstance(x, dict) else str(x)
        if d and d not in out:
            out.append(d)
        if len(out) >= limit:
            break
    return out


def write_submission(payload, jp: Path, zp: Path):
    jp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("submission.json", jp.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-windows", type=int, default=3)
    ap.add_argument("--window-words", type=int, default=220)
    ap.add_argument("--proxy-depth", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    root = args.repo_root.resolve()
    model_root = args.model_root.resolve()

    d1 = json.loads(
        (
            root
            / "results/gemini/huy_vnlegal_rank_ablation_v1/"
            "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
        ).read_text(encoding="utf-8")
    )

    clean_report = json.loads(
        (
            root
            / "results/manual/huy_public_rel_l0_harrier_safe_fill_v2_clean/"
            "REPORT.json"
        ).read_text(encoding="utf-8")
    )
    clean_safe = [str(x["qid"]) for x in clean_report["actions"]]
    if len(clean_safe) != 127:
        raise RuntimeError(f"Expected 127 clean safe qids, got {len(clean_safe)}")

    proxy = json.loads(
        (
            root
            / "results/manual/huy_public_rel_l0_harrier_safe_fill_v1/"
            "PUBLIC_HARRIER_GLOBAL_TOP50.json"
        ).read_text(encoding="utf-8")
    )
    proxy_rank = {
        str(q): dedup_parent_results(v.get("results", []), args.proxy_depth)
        for q, v in proxy.items()
    }

    public = json.loads(
        (
            root
            / "DSC2026-LegalIR-main/v4_run/public_test_dataset/"
            "public-official.json"
        ).read_text(encoding="utf-8")
    )
    public_q = {str(q): qtext(v) for q, v in public.items()}

    warm_path = find_named(model_root, "warmup_finetuned_best.json")
    if warm_path is None:
        raise FileNotFoundError("warmup_finetuned_best.json")
    warm = json.loads(warm_path.read_text(encoding="utf-8"))

    warm_by_text = {}
    for wq, v in warm.items():
        nt = norm(qtext(v))
        if nt:
            warm_by_text.setdefault(nt, []).append((str(wq), v))

    teacher_public_qids = [
        q for q, t in public_q.items()
        if norm(t) in warm_by_text
    ]
    print(
        f"Clean safe qids={len(clean_safe)} | "
        f"teacher identical-public qids={len(teacher_public_qids)}",
        flush=True,
    )

    target_qids = sorted(set(clean_safe) | set(teacher_public_qids))
    needed_docs = []
    for q in target_qids:
        for d in proxy_rank[q][:args.proxy_depth]:
            if d not in needed_docs:
                needed_docs.append(d)
    print(f"Unique proxy candidate docs to multi-window encode: {len(needed_docs)}", flush=True)

    ctx_dir = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    path_by_doc = {
        p.stem[len("context_"):]: p
        for p in ctx_dir.glob("context_*.json")
    }

    model_dir = find_merged_model(model_root)
    cfg_path, cfg = find_train_config(model_root)
    max_length = int(cfg.get("max_seq_len", 512))
    print(f"Model={model_dir} | max_length={max_length}", flush=True)

    tok = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_dir,
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")

    # Build windows and ownership.
    windows = []
    owners = []
    doc_slices = {}
    for d in needed_docs:
        if d not in path_by_doc:
            raise KeyError(f"Missing context for doc {d}")
        text = load_doc_text(path_by_doc[d])
        ws = sample_windows(
            text,
            window_words=args.window_words,
            max_windows=args.max_windows,
        )
        st = len(windows)
        windows.extend(ws)
        owners.extend([d] * len(ws))
        doc_slices[d] = (st, len(windows))

    print(
        f"Total windows={len(windows)} "
        f"(mean={len(windows)/max(1,len(needed_docs)):.2f}/doc)",
        flush=True,
    )

    dim = int(model.config.hidden_size)
    outdir = root / "results/manual/huy_harrier_multiwindow_safe_fill_v1"
    outdir.mkdir(parents=True, exist_ok=True)
    emb_path = outdir / (
        f"candidate_windows_w{args.window_words}_n{args.max_windows}.f16"
    )

    emb = np.memmap(
        emb_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(windows), dim),
    )

    t0 = time.time()
    for i in range(0, len(windows), args.batch_size):
        j = min(i + args.batch_size, len(windows))
        try:
            e = encode(model, tok, windows[i:j], max_length)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"OOM with --batch-size {args.batch_size}; rerun with a smaller batch."
            )
        emb[i:j] = e.half().cpu().numpy()
        if j % 256 < args.batch_size or j == len(windows):
            emb.flush()
            sec_per = (time.time() - t0) / max(j, 1)
            eta = sec_per * (len(windows) - j) / 60
            print(
                f"Encoded {j}/{len(windows)} windows | "
                f"{sec_per:.3f}s/window | eta={eta:.1f}m | "
                f"VRAM={torch.cuda.memory_allocated()/2**30:.2f}GB",
                flush=True,
            )

    # Encode only target queries.
    qvec = {}
    tq = target_qids
    for i in range(0, len(tq), 16):
        bq = tq[i:i+16]
        e = encode(
            model, tok,
            [QUERY_INSTRUCT + public_q[q] for q in bq],
            max_length,
        ).cpu().numpy()
        for k, q in enumerate(bq):
            qvec[q] = e[k].astype(np.float32)

    del model
    torch.cuda.empty_cache()

    emb_arr = np.asarray(emb, dtype=np.float32)

    reranked = {}
    scores = {}
    for q in target_qids:
        qv = qvec[q]
        ds = proxy_rank[q][:args.proxy_depth]
        sdict = {}
        for d in ds:
            a, b = doc_slices[d]
            sc = emb_arr[a:b] @ qv
            sdict[d] = float(np.max(sc))
        ds2 = sorted(ds, key=lambda d: (-sdict[d], d))
        reranked[q] = ds2
        scores[q] = sdict

    # Retrieval-fidelity teacher audit on identical public<->warmup questions.
    rows = []
    for pq in teacher_public_qids:
        nt = norm(public_q[pq])
        for wq, wv in warm_by_text[nt]:
            exact = dedup_parent_results(wv.get("results", []), 200)
            approx = reranked[pq]
            if not exact or not approx:
                continue
            rec = {"public_qid": pq, "warmup_qid": wq}
            for k in (1, 5, 10, 20, 50):
                rec[f"overlap_at_{k}"] = (
                    len(set(approx[:k]) & set(exact[:k])) / max(1, k)
                )
            rec["approx_top1_in_exact_top5"] = approx[0] in set(exact[:5])
            rows.append(rec)

    teacher_summary = {
        "matched_pairs": len(rows),
        "mean_overlap": {
            str(k): float(np.mean([r[f"overlap_at_{k}"] for r in rows]))
            for k in (1, 5, 10, 20, 50)
        },
        "approx_top1_in_exact_top5_rate": float(np.mean([
            r["approx_top1_in_exact_top5"] for r in rows
        ])),
    }

    print("=" * 108)
    print("MULTI-WINDOW vs saved exact warmup retrieval")
    for k in (1, 5, 10, 20, 50):
        print(
            f"Mean parent Top{k:>2} overlap: "
            f"{teacher_summary['mean_overlap'][str(k)]:.4f}"
        )
    print(
        "Approx top1 in exact top5 rate:",
        f"{teacher_summary['approx_top1_in_exact_top5_rate']:.4f}",
    )
    print("=" * 108)

    # Produce clean safe-slot variants from reranked outsider positions 1/2/3.
    valid_docs = set(path_by_doc)
    variants = {}
    for pos in (1, 2, 3):
        payload = {
            q: {"answer": list(get_answer(v))}
            for q, v in d1.items()
        }
        actions = []
        for q in clean_safe:
            base = get_answer(d1[q])
            outsiders = [d for d in reranked[q] if d not in set(base)]
            if len(outsiders) < pos:
                continue
            c = outsiders[pos - 1]
            payload[q] = {"answer": base[:4] + [c]}
            actions.append({
                "qid": q,
                "challenger": c,
                "reranked_outsider_pos": pos,
                "multiwindow_score": scores[q][c],
            })

        for q, v in payload.items():
            a = get_answer(v)
            assert len(a) == 5 and len(set(a)) == 5
            assert all(d in valid_docs for d in a)

        stem = f"CANDIDATE_D1_REL_L0_HARRIER_MULTIWINDOW_POS{pos}"
        jp = outdir / f"{stem}.json"
        zp = outdir / f"{stem}.zip"
        write_submission(payload, jp, zp)
        variants[str(pos)] = {
            "actions": len(actions),
            "json": str(jp),
            "zip": str(zp),
        }
        print(f"POS{pos}: actions={len(actions)} -> {zp}")

    report = {
        "schema": "manual.harrier_multiwindow_safe_fill_v1",
        "window_words": args.window_words,
        "max_windows": args.max_windows,
        "proxy_depth": args.proxy_depth,
        "unique_docs": len(needed_docs),
        "total_windows": len(windows),
        "teacher_fidelity": teacher_summary,
        "baseline_prefix_proxy_teacher_fidelity": {
            "top1": 0.4340,
            "top5": 0.4226,
            "top10": 0.4189,
            "top20": 0.4264,
            "top50": 0.3683,
            "prefix_top1_in_exact_top5": 0.6981,
        },
        "variants": variants,
        "public_labels_used": False,
    }
    rp = outdir / "REPORT.json"
    rp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Report:", rp)


if __name__ == "__main__":
    main()
