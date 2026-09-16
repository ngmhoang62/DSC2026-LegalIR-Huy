"""Stage 1: Audit and seal LoRA adapter provenance and upstream training artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .common import (
    ADAPTER_DIR,
    RESULTS_DIR,
    ROOT,
    SOURCE_DIR,
    UPSTREAM_COMMIT,
    UPSTREAM_RESULTS_DIR,
    get_git_status,
    sha256_file,
)


def run_provenance_audit() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("=== STAGE 1: AUDIT PROVENANCE AND SEAL ADAPTER ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_status()

    # 1. Audit and seal adapter files
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Missing adapter directory: {ADAPTER_DIR}")

    adapter_files = {}
    for f in sorted(ADAPTER_DIR.glob("*")):
        if f.is_file():
            adapter_files[f.name] = {
                "path": str(f.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            }

    required_adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    for req in required_adapter_files:
        if req not in adapter_files:
            raise FileNotFoundError(f"Missing required adapter file: {req}")

    # 2. Inspect and verify upstream pilot authoritative artifacts
    stability_file = UPSTREAM_RESULTS_DIR / "TRAINING_STABILITY.json"
    update_proof_file = UPSTREAM_RESULTS_DIR / "NEURAL_UPDATE_PROOF.json"
    reload_parity_file = UPSTREAM_RESULTS_DIR / "ADAPTER_RELOAD_PARITY.json"

    for p in [stability_file, update_proof_file, reload_parity_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing upstream artifact: {p}")

    stability_data = json.loads(stability_file.read_text(encoding="utf-8"))
    update_proof_data = json.loads(update_proof_file.read_text(encoding="utf-8"))
    reload_parity_data = json.loads(reload_parity_file.read_text(encoding="utf-8"))

    # Assertions on upstream artifacts
    upstream_commit_match = (
        stability_data.get("git_commit") == UPSTREAM_COMMIT
        and update_proof_data.get("git_commit") == UPSTREAM_COMMIT
        and reload_parity_data.get("git_commit") == UPSTREAM_COMMIT
    )
    if not upstream_commit_match:
        raise ValueError(
            f"Upstream artifacts git commit mismatch! Expected {UPSTREAM_COMMIT}, got "
            f"{stability_data.get('git_commit')}, {update_proof_data.get('git_commit')}, {reload_parity_data.get('git_commit')}"
        )

    classifier_delta_zero = update_proof_data.get("classifier_integrity", {}).get("delta_is_strictly_zero") is True
    frozen_base_delta_zero = update_proof_data.get("frozen_base_integrity", {}).get("delta_is_strictly_zero") is True
    reload_parity_passed = reload_parity_data.get("parity_passed") is True
    nan_inf_grad_zero = stability_data.get("gradient_statistics", {}).get("nan_inf_grad_count") == 0
    total_l2_drift = update_proof_data.get("lora_update_summary", {}).get("total_l2_drift", 0.0)

    if not classifier_delta_zero:
        raise AssertionError("Classifier integrity check failed: delta is not zero!")
    if not frozen_base_delta_zero:
        raise AssertionError("Frozen base integrity check failed: delta is not zero!")
    if not reload_parity_passed:
        raise AssertionError("Adapter reload parity check failed!")
    if not nan_inf_grad_zero:
        raise AssertionError("Upstream training had NaN/Inf gradients!")
    if total_l2_drift <= 0.0:
        raise AssertionError(f"Total LoRA drift is not positive: {total_l2_drift}")

    adapter_manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.adapter_provenance.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "upstream_commit": UPSTREAM_COMMIT,
        "status": "PASS",
        "adapter_dir": str(ADAPTER_DIR.relative_to(ROOT)).replace("\\", "/"),
        "adapter_files": adapter_files,
        "upstream_verifications": {
            "upstream_commit_match": upstream_commit_match,
            "classifier_head_delta_strictly_zero": classifier_delta_zero,
            "frozen_base_delta_strictly_zero": frozen_base_delta_zero,
            "reload_parity_passed": reload_parity_passed,
            "max_reload_difference": reload_parity_data.get("max_absolute_pre_save_vs_reload_difference"),
            "total_lora_l2_drift": total_l2_drift,
            "optimizer_steps": stability_data.get("total_optimizer_steps"),
            "nan_inf_grad_count": 0,
        },
    }

    adapter_manifest_path = RESULTS_DIR / "ADAPTER_PROVENANCE.json"
    adapter_manifest_path.write_text(json.dumps(adapter_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {adapter_manifest_path}", flush=True)

    # 3. Source provenance
    source_files = {}
    for f in sorted(SOURCE_DIR.glob("*.py")):
        source_files[f.name] = {
            "path": str(f.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": f.stat().st_size,
            "sha256": sha256_file(f),
        }

    source_manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.source_provenance.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "source_files": source_files,
    }
    source_manifest_path = RESULTS_DIR / "SOURCE_PROVENANCE.json"
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {source_manifest_path}", flush=True)

    return adapter_manifest, source_manifest


if __name__ == "__main__":
    run_provenance_audit()
