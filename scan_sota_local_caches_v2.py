#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

KEYWORDS = (
    "embed", "embedding", "vector", "vectors", "score", "scores",
    "passage", "chunk", "dense", "e5", "vnlegal", "aiteam",
    "jina", "crossenc", "rerank", "title", "public", "private",
)

EXTS = {
    ".pkl", ".pickle", ".npy", ".npz", ".f16", ".bin", ".pt",
    ".pth", ".safetensors", ".json", ".jsonl", ".sqlite", ".db",
}

SKIP_DIR_NAMES = {
    ".git", "dsc_env_huy", "__pycache__", ".venv", "venv",
    "node_modules",
}

def human(n):
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024

def is_interesting(path: Path, size: int):
    name = path.as_posix().lower()
    return (
        path.suffix.lower() in EXTS
        and (
            any(k in name for k in KEYWORDS)
            or size >= 50 * 1024 * 1024
        )
    )

def npy_meta(path: Path):
    try:
        import numpy as np
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return f"shape={tuple(arr.shape)} dtype={arr.dtype}"
    except Exception:
        return ""

def inspect_known(root: Path):
    known = [
        # Public score caches from reproduce.py
        "results/crossenc_fullpool/public_scores.pkl",
        "results/from_drive/aiteamvn_ft_public.pkl",
        "results/from_drive/jina_ft_public.pkl",
        "results/burst_fresh_block/title_embed_public.pkl",
        "results/burst_userft_maxrecall/vnlegal_scores.pkl",
        "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl",

        # Generic/fixed representation caches known from historical source
        "results/corpus_index/chunks_cap32.f16",
        "results/corpus_index/chunks_cap32.json",

        # Private caches from current runner (v13 exact paths)
        "results/manual/huy_private_d1_rel_l0_exact_v1/cache/d1_e5_small_threeview_scores.pkl",
        "results/manual/huy_private_d1_rel_l0_exact_v1/cache/vnlegal/vnlegal_scores_exact64_prefetch.pkl",
        "results/manual/huy_private_d1_rel_l0_exact_v1/cache/crossenc_scores.pkl",
        "results/manual/huy_private_d1_rel_l0_exact_v1/cache/aiteamvn_ft_scores.pkl",
        "results/manual/huy_private_d1_rel_l0_exact_v1/cache/jina_ft_scores.pkl",
        "results/manual/huy_private_d1_rel_l0_exact_v1/cache/title_embed_scores.pkl",

        # REL generic embeddings
        "cache/research_v2_e5_confirmation/bundle-v1/embeddings.f16.npy",
        "cache/research_v2_e5_confirmation/bundle-v1/chunk_ids.jsonl",
    ]

    print("=== KNOWN / EXPECTED CACHES ===")
    for rel in known:
        p = root / rel
        if p.exists():
            size = p.stat().st_size if p.is_file() else 0
            meta = npy_meta(p) if p.suffix.lower() == ".npy" else ""
            print(f"[FOUND] {human(size):>9} {rel} {meta}")
        else:
            print(f"[miss ] {'':>9} {rel}")
    print()

def walk_roots(root: Path):
    """Scan only this repo's ignored/local artifact roots."""
    roots = []
    for p in [
        root / "results",
        root / "cache",
        root / "models",
    ]:
        p = p.resolve()
        if p.exists() and p not in roots:
            roots.append(p)
    return roots

def main():
    ap = argparse.ArgumentParser(
        description="Read-only inventory of ignored/local caches inside the current SOTA repo only."
    )
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--min-size-mb", type=float, default=1.0,
        help="Minimum size for keyword-matching files to print.",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()

    inspect_known(root)

    print("=== LOCAL IGNORED/UNTRACKED CACHE INVENTORY ===")
    print("Read-only: filenames, sizes, and .npy headers only.\n")

    rows = []
    min_size = int(args.min_size_mb * 1024 * 1024)

    for scan_root in walk_roots(root):
        print(f"Scanning: {scan_root}", flush=True)
        for dirpath, dirnames, filenames in os.walk(scan_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            dp = Path(dirpath)
            for fn in filenames:
                p = dp / fn
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size < min_size:
                    continue
                if not is_interesting(p, size):
                    continue
                try:
                    rel = p.relative_to(root).as_posix()
                except ValueError:
                    rel = str(p)
                meta = npy_meta(p) if p.suffix.lower() == ".npy" else ""
                rows.append((size, rel, meta))

    rows.sort(reverse=True)

    print(f"\nFound {len(rows)} interesting files >= {args.min_size_mb:.1f} MB.\n")
    for size, rel, meta in rows:
        print(f"{human(size):>9}  {rel}" + (f"  [{meta}]" if meta else ""))

    print("\n=== LIKELY REUSABLE REPRESENTATION CACHES ===")
    rep_rows = []
    for size, rel, meta in rows:
        low = rel.lower()
        if (
            any(k in low for k in ("embed", "embedding", "vector", "chunk"))
            and not any(k in low for k in ("model.safetensors", "pytorch_model", "optimizer"))
        ):
            rep_rows.append((size, rel, meta))

    if not rep_rows:
        print("No obvious representation cache found by filename.")
    else:
        for size, rel, meta in rep_rows:
            print(f"{human(size):>9}  {rel}" + (f"  [{meta}]" if meta else ""))

    print("\nNOTE:")
    print("- *_scores.pkl are usually query-specific score caches; public scores cannot be reused for private queries.")
    print("- corpus/chunk embeddings can be reusable across query sets if text/model/chunking contracts match.")
    print("- cross-encoder/Jina pair scores are query-passage dependent and cannot be converted into reusable passage embeddings.")

if __name__ == "__main__":
    main()
