"""Discover and verify all required VietLegal-E5 assets on disk.

Produces results/gemini/huy_e5_adaptation_delta_v1/E5_ASSET_MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "results/gemini/huy_e5_adaptation_delta_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = [
    {
        "name": "vietlegal-e5_base_weights",
        "path": "cache/research_v2_e5_confirmation/bundle-v1/vietlegal-e5/model.safetensors",
        "scope": "base_model",
        "type": "safetensors",
    },
    {
        "name": "vietlegal-e5_chunk_embeddings",
        "path": "cache/research_v2_e5_confirmation/bundle-v1/embeddings.f16.npy",
        "scope": "corpus_chunk_bank",
        "type": "numpy_f16",
    },
    {
        "name": "vietlegal-e5_chunk_ids",
        "path": "cache/research_v2_e5_confirmation/bundle-v1/chunk_ids.jsonl",
        "scope": "corpus_metadata",
        "type": "jsonl",
    },
    {
        "name": "v2_folds",
        "path": "cache/research_v2_e5_confirmation/bundle-v1/V2_FOLDS.json",
        "scope": "cv_split",
        "type": "json",
    },
    {
        "name": "adapter_fold_0",
        "path": "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt",
        "scope": "held_out_fold_0",
        "type": "pytorch_adapter",
    },
    {
        "name": "adapter_fold_1",
        "path": "results/research_v2_e5_confirmation/fold_1/training/epoch-2.pt",
        "scope": "held_out_fold_1",
        "type": "pytorch_adapter",
    },
    {
        "name": "adapter_fold_2",
        "path": "results/research_v2_e5_confirmation/fold_2/training/epoch-2.pt",
        "scope": "held_out_fold_2",
        "type": "pytorch_adapter",
    },
    {
        "name": "adapter_fold_3",
        "path": "results/research_v2_e5_confirmation/fold_3/training/epoch-2.pt",
        "scope": "held_out_fold_3",
        "type": "pytorch_adapter",
    },
    {
        "name": "adapter_fold_4",
        "path": "results/research_v2_e5_confirmation/fold_4/training/epoch-2.pt",
        "scope": "held_out_fold_4",
        "type": "pytorch_adapter",
    },
    {
        "name": "adapter_full_data",
        "path": "results/research_v2_open_rl/v2_anchor_submission_candidate/full_data_adapter/epoch-2.pt",
        "scope": "public_deployment_all_6991",
        "type": "pytorch_adapter",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    print("=== Building E5_ASSET_MANIFEST.json ===", flush=True)
    manifest = {
        "schema_version": "dsc2026.gemini.huy_e5_adaptation_delta_v1.e5_asset_manifest.v1",
        "status": "VERIFIED_ON_DISK",
        "assets": [],
    }

    for item in ASSETS:
        p = ROOT / item["path"]
        if not p.is_file():
            raise FileNotFoundError(f"Missing required asset: {p}")
        file_sha256 = sha256(p)
        entry = {
            "name": item["name"],
            "relative_path": item["path"],
            "absolute_path": str(p.resolve()),
            "bytes": p.stat().st_size,
            "sha256": file_sha256,
            "scope": item["scope"],
            "type": item["type"],
            "status": "VALID",
        }
        manifest["assets"].append(entry)
        print(f"  {item['name']:30s} | {entry['bytes']/1e6:8.2f} MB | {file_sha256[:16]}...")

    out_path = OUTPUT_DIR / "E5_ASSET_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
