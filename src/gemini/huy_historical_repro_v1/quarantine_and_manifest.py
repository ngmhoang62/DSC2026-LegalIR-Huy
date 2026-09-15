"""Quarantine historical caches and build CLEANROOM_INPUT_MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
BACKUP_DIR = RESULTS_DIR / f"_pre_cleanroom_historical_backup_{int(time.time())}"
MANIFEST_PATH = REPO_ROOT / "results/gemini/huy_historical_repro_v1/CLEANROOM_INPUT_MANIFEST.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=== Scanning historical artifacts under results/ for quarantine and manifest ===")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Specific targets that will be freshly recomputed from weights
    targets_to_quarantine = {
        "results/from_drive/aiteamvn_ft_cv.pkl": "REPRODUCED_FRESH_FROM_WEIGHTS",
        "results/from_drive/aiteamvn_ft_public.pkl": "REPRODUCED_FRESH_FROM_WEIGHTS",
        "results/from_drive/jina_ft_cv.pkl": "REPRODUCED_FRESH_FROM_WEIGHTS",
        "results/from_drive/jina_ft_public.pkl": "REPRODUCED_FRESH_FROM_WEIGHTS",
        "results/burst_fresh_block/title_embed_scores.pkl": "REPRODUCED_FRESH_FROM_WEIGHTS",
        "results/burst_fresh_block/title_embed_public.pkl": "REPRODUCED_FRESH_FROM_WEIGHTS",
        "results/burst_userft_maxrecall/vnlegal_scores.pkl": "REPRODUCED_FRESH_FROM_WEIGHTS",
        "results/burst_userft_maxrecall/submission.json": "FINAL_OUTPUT_TO_BE_REFIT",
        "results/burst_userft_maxrecall/submission.zip": "FINAL_OUTPUT_TO_BE_REFIT",
        "results/burst_userft_maxrecall/reproduce_audit.json": "FINAL_OUTPUT_TO_BE_REFIT",
    }

    manifest_entries = []
    quarantined_count = 0

    for file_path in RESULTS_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(REPO_ROOT).as_posix()
        if rel_path.startswith("results/gemini") or rel_path.startswith("results/_pre_cleanroom"):
            continue

        size = file_path.stat().st_size
        digest = sha256_file(file_path)

        if rel_path in targets_to_quarantine:
            classification = targets_to_quarantine[rel_path]
            action = "QUARANTINED"
            dest = BACKUP_DIR / rel_path
            manifest_entries.append({
                "path": rel_path,
                "size_bytes": size,
                "sha256": digest,
                "classification": classification,
                "action": action,
                "target_quarantine": dest.as_posix()
            })
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(dest))
            quarantined_count += 1
        else:
            if "state.pt" in file_path.name or "checkpoint" in file_path.name:
                classification = "FROZEN_UPSTREAM_CHECKPOINT"
            elif "extended_scores" in file_path.name or "cpu_top20" in file_path.name or "fullpool" in rel_path:
                classification = "FROZEN_UPSTREAM_CANDIDATE_POOL_UNRESOLVED_GENERATOR"
            else:
                classification = "FROZEN_UPSTREAM_BASELINE_CACHE"
            manifest_entries.append({
                "path": rel_path,
                "size_bytes": size,
                "sha256": digest,
                "classification": classification,
                "action": "PRESERVED_FROZEN_INPUT",
                "target_quarantine": None
            })

    print(f"Recorded {len(manifest_entries)} artifacts in manifest.")
    print(f"Quarantined {quarantined_count} target score/submission files to {BACKUP_DIR}")

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "timestamp": time.time(),
        "backup_directory": BACKUP_DIR.relative_to(REPO_ROOT).as_posix(),
        "total_tracked": len(manifest_entries),
        "total_quarantined": quarantined_count,
        "artifacts": manifest_entries
    }
    MANIFEST_PATH.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest to {MANIFEST_PATH}")

    # Verify zero target files exist at pipeline paths
    remaining_targets = [p for p in targets_to_quarantine if (REPO_ROOT / p).exists()]
    if remaining_targets:
        print(f"ERROR: Quarantined files still exist: {remaining_targets}")
        sys.exit(1)
    else:
        print("VERIFICATION PASS: All target score and submission paths are empty and clean.")


if __name__ == "__main__":
    main()
