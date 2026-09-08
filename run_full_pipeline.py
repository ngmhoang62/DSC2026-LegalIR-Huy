"""Recompute every extra-channel score from the model weights, then rebuild.

`reproduce.py` is the fast path: it fuses the bundled score caches and finishes
in minutes with no GPU. This is the slow path -- it deletes those three caches
and regenerates them by actually running the models, which is what you want if
you changed a model, or want to prove the caches were not hand-edited.

Requires the weights first:  python download_models.py

Roughly 2 hours on an 8 GB GPU: 600 CV + 1000 public queries x ~39 documents
x 2 passages, for each of three channels.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (label, output cache, command)
STEPS = [
    ("aiteamvn_ft / CV", "results/from_drive/aiteamvn_ft_cv.pkl",
     [PY, "-u", "score_cv_custom_encoder.py",
      "--model-path", "models/from_drive/AITeamVN_Vietnamese_Embedding",
      "--out", "results/from_drive/aiteamvn_ft_cv.pkl"]),
    ("aiteamvn_ft / public", "results/from_drive/aiteamvn_ft_public.pkl",
     [PY, "-u", "score_public_models.py", "--kind", "bi",
      "--model-path", "models/from_drive/AITeamVN_Vietnamese_Embedding",
      "--out", "results/from_drive/aiteamvn_ft_public.pkl"]),
    ("jina_ft / CV", "results/from_drive/jina_ft_cv.pkl",
     [PY, "-u", "score_cv_jina_ft.py",
      "--weights", "models/from_drive/jina_finetuned/model.safetensors",
      "--out", "results/from_drive/jina_ft_cv.pkl"]),
    ("jina_ft / public", "results/from_drive/jina_ft_public.pkl",
     [PY, "-u", "score_public_models.py", "--kind", "jina",
      "--weights", "models/from_drive/jina_finetuned/model.safetensors",
      "--out", "results/from_drive/jina_ft_public.pkl"]),
    ("title_embed / CV", "results/burst_fresh_block/title_embed_scores.pkl",
     [PY, "-u", "tune_title_embedding.py"]),
    ("title_embed / public", "results/burst_fresh_block/title_embed_public.pkl",
     [PY, "-u", "score_public_title_embed.py"]),
]

REQUIRED_MODELS = [
    "models/from_drive/AITeamVN_Vietnamese_Embedding/model.safetensors",
    "models/from_drive/jina_finetuned/model.safetensors",
    "models/jina-reranker-v2-base-multilingual",
    "models/AITeamVN_Vietnamese_Embedding",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-caches", action="store_true",
                    help="giu cache cu, chi tinh phan con thieu")
    ap.add_argument("--only", default="", help="loc buoc theo ten, vd 'jina_ft'")
    args = ap.parse_args()

    missing = [m for m in REQUIRED_MODELS if not (ROOT / m).exists()]
    if missing:
        print("Thieu model weight:")
        for m in missing:
            print("   ", m)
        print("\nChay truoc:  python download_models.py")
        return 1

    steps = [s for s in STEPS if not args.only or args.only in s[0]]
    if not args.keep_caches:
        for label, cache, _ in steps:
            p = ROOT / cache
            if p.exists():
                p.unlink()
                print(f"da xoa cache cu: {cache}")

    for i, (label, cache, cmd) in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps)}] {label} ===", flush=True)
        t = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"buoc '{label}' that bai (exit {rc})")
            return rc
        print(f"  xong trong {(time.time() - t) / 60:.1f} phut", flush=True)

    print("\n=== dung lai submission tu diem vua tinh ===", flush=True)
    return subprocess.call([PY, "-u", "reproduce.py"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
