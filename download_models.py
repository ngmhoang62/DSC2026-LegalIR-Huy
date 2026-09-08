"""Fetch every model weight this pipeline can use.

Two sources:

  * Google Drive -- the fine-tuned checkpoints (not published on the Hub).
    Downloaded with gdown from the shared folder.
  * Hugging Face -- the public base models.

Nothing here is needed to rebuild the shipped submission: `reproduce.py` runs
from the bundled score caches alone. Models are only required to RECOMPUTE
those scores from scratch (see run_full_pipeline.py), or to score a new test
set.

    python download_models.py              # everything (~5.5 GB)
    python download_models.py --drive      # only the fine-tuned checkpoints
    python download_models.py --hf         # only the public base models
    python download_models.py --check      # report what is already present
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1ahUyUuHcegozzSoWOBj3E8OuGvgtcLUV"

# subfolder inside the Drive folder -> what it is used for
DRIVE_EXPECTED = {
    "AITeamVN_Vietnamese_Embedding": "kenh aiteamvn_ft (bi-encoder fine-tune)",
    "jina_finetuned": "kenh jina_ft (cross-encoder fine-tune)",
    "vietlegal_finetuned_results_HNSW": "LoRA harrier (khong dung trong cau hinh nay)",
}

# local dir -> (hugging face repo id, what it is used for)
HF_MODELS = {
    "models/AITeamVN_Vietnamese_Embedding":
        ("AITeamVN/Vietnamese_Embedding", "kenh dense + title_embed"),
    "models/jina-reranker-v2-base-multilingual":
        ("jinaai/jina-reranker-v2-base-multilingual",
         "kenh jina; cung la code nen de nap weight jina_ft"),
    "models/AITeamVN_Vietnamese_Reranker":
        ("AITeamVN/Vietnamese_Reranker", "kenh crossenc"),
    "models/vnlegal-lal":
        ("darklethelong/vnlegal-lal", "kenh vnlegal_lal"),
}

HF_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.bin", "*.model", "*.jinja",
               "tokenizer*", "vocab*", "merges*", "special_tokens_map*", "*.py"]
HF_IGNORE = ["onnx/*", "openvino/*", "*.onnx", "*.sig", "*.h5", "*.msgpack"]


def size_of(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def report() -> None:
    print("=== fine-tuned checkpoints (Google Drive) ===")
    for name, why in DRIVE_EXPECTED.items():
        p = ROOT / "models/from_drive" / name
        s = size_of(p)
        print(f"  {'co   ' if s else 'THIEU'} {s / 2**20:8.1f} MB  "
              f"models/from_drive/{name}\n           {why}")
    print("\n=== base models (Hugging Face) ===")
    for rel, (repo, why) in HF_MODELS.items():
        p = ROOT / rel
        s = size_of(p)
        print(f"  {'co   ' if s else 'THIEU'} {s / 2**20:8.1f} MB  {rel}"
              f"\n           {repo} -- {why}")


def clean_drive_junk(target: Path) -> int:
    """The shared folder carries Windows 'Zone.Identifier' companion files."""
    removed = 0
    for f in list(target.rglob("*Zone.Identifier*")):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def fetch_drive() -> bool:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown chua cai. Chay: pip install gdown", file=sys.stderr)
        return False
    target = ROOT / "models/from_drive"
    target.mkdir(parents=True, exist_ok=True)
    print(f"tai tu Google Drive -> {target}", flush=True)
    import subprocess
    rc = subprocess.call([sys.executable, "-m", "gdown", "--folder", "--continue",
                          DRIVE_FOLDER, "-O", str(target)])
    if rc != 0:
        print(f"gdown tra ve {rc}. Neu bao loi quyen, mo link bang trinh duyet "
              f"va tai thu cong vao {target}", file=sys.stderr)
    n = clean_drive_junk(target)
    if n:
        print(f"da xoa {n} file rac Zone.Identifier")
    ok = True
    for name in ("AITeamVN_Vietnamese_Embedding", "jina_finetuned"):
        p = target / name / "model.safetensors"
        if not p.exists():
            print(f"  ! thieu {p.relative_to(ROOT)}", file=sys.stderr)
            ok = False
    return ok


def fetch_hf() -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub chua cai. Chay: pip install huggingface_hub",
              file=sys.stderr)
        return False
    ok = True
    for rel, (repo, why) in HF_MODELS.items():
        target = ROOT / rel
        if size_of(target) > 100 * 2**20:
            print(f"  bo qua {rel} (da co)", flush=True)
            continue
        print(f"\ntai {repo} -> {rel}  [{why}]", flush=True)
        try:
            snapshot_download(repo, local_dir=str(target),
                              allow_patterns=HF_PATTERNS, ignore_patterns=HF_IGNORE)
        except Exception as exc:  # network, gated repo, renamed repo
            print(f"  ! that bai: {exc}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true", help="chi tai tu Google Drive")
    ap.add_argument("--hf", action="store_true", help="chi tai tu Hugging Face")
    ap.add_argument("--check", action="store_true", help="chi bao cao, khong tai")
    args = ap.parse_args()

    if args.check:
        report()
        return 0

    both = not (args.drive or args.hf)
    ok = True
    if args.drive or both:
        ok &= fetch_drive()
    if args.hf or both:
        ok &= fetch_hf()

    print("\n=== ket qua ===")
    report()
    if not ok:
        print("\nMot so model chua tai duoc -- xem thong bao loi o tren.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
