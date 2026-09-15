import sys
import json
import hashlib
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path("d:/Study/DSC2026/sota").resolve()
sys.path.insert(0, str(REPO_ROOT))

from src.research_v2_e5_transfer.e5_transfer_runner import QueryEncoder

BASE_MODEL_DIR = REPO_ROOT / "cache" / "research_v2_e5_confirmation" / "bundle-v1" / "vietlegal-e5"
RESULTS_DIR = REPO_ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ADAPTER_CHECKPOINTS = {
    "fold_0": REPO_ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt",
    "fold_1": REPO_ROOT / "results/research_v2_e5_confirmation/fold_1/training/epoch-2.pt",
    "fold_2": REPO_ROOT / "results/research_v2_e5_confirmation/fold_2/training/epoch-2.pt",
    "fold_3": REPO_ROOT / "results/research_v2_e5_confirmation/fold_3/training/epoch-2.pt",
    "fold_4": REPO_ROOT / "results/research_v2_e5_confirmation/fold_4/training/epoch-2.pt",
    "full_data": REPO_ROOT / "results/research_v2_open_rl/v2_anchor_submission_candidate/full_data_adapter/epoch-2.pt",
}

def fingerprint_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

def audit_checkpoint(name: str, ckpt_path: Path):
    print(f"\nAuditing {name}: {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    adapter_dict = ckpt["adapter"]
    ckpt_keys = set(adapter_dict.keys())

    # Instantiate QueryEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Outer QueryEncoder
    outer_encoder = QueryEncoder(BASE_MODEL_DIR, device=device, checkpoint_path=None)
    outer_state_keys = set(outer_encoder.state_dict().keys())
    
    # Inner PEFT model
    inner_model = outer_encoder.model
    inner_state_keys = set(inner_model.state_dict().keys())

    # Keys analysis
    sample_ckpt_keys = sorted(list(ckpt_keys))[:5]
    sample_outer_keys = sorted([k for k in outer_state_keys if "lora" in k])[:5]
    sample_inner_keys = sorted([k for k in inner_state_keys if "lora" in k])[:5]

    # Target 1: Outer QueryEncoder (canonical: self.load_state_dict(ckpt["adapter"], strict=False))
    # Target 2: Inner PEFT model (old gemini: self.model.load_state_dict(ckpt["adapter"], strict=False))
    
    # Test loading into inner PEFT model (old gemini scorer method)
    # Save baseline weights of inner trainable parameters
    trainable_inner_baseline = {k: v.clone() for k, v in inner_model.named_parameters() if v.requires_grad}
    inner_res = inner_model.load_state_dict(adapter_dict, strict=False)
    
    # Calculate parameter delta for old method
    inner_changed_tensors = 0
    inner_l2_delta = 0.0
    for k, v in inner_model.named_parameters():
        if v.requires_grad:
            diff = (v - trainable_inner_baseline[k]).float()
            norm = float(torch.norm(diff).item())
            if norm > 1e-7:
                inner_changed_tensors += 1
                inner_l2_delta += norm

    # Re-instantiate a fresh QueryEncoder to test canonical load into outer QueryEncoder
    outer_encoder_canonical = QueryEncoder(BASE_MODEL_DIR, device=device, checkpoint_path=None)
    trainable_outer_baseline = {k: v.clone() for k, v in outer_encoder_canonical.named_parameters() if v.requires_grad}
    canonical_res = outer_encoder_canonical.load_state_dict(adapter_dict, strict=False)

    canonical_changed_tensors = 0
    canonical_l2_delta = 0.0
    for k, v in outer_encoder_canonical.named_parameters():
        if v.requires_grad:
            diff = (v - trainable_outer_baseline[k]).float()
            norm = float(torch.norm(diff).item())
            if norm > 1e-7:
                canonical_changed_tensors += 1
                canonical_l2_delta += norm

    # Key overlap analysis
    matching_inner = ckpt_keys.intersection(inner_state_keys)
    matching_outer = ckpt_keys.intersection(outer_state_keys)

    audit_entry = {
        "checkpoint_name": name,
        "checkpoint_path": str(ckpt_path),
        "total_checkpoint_adapter_keys": len(ckpt_keys),
        "sample_checkpoint_keys": sample_ckpt_keys,
        "sample_outer_lora_keys": sample_outer_keys,
        "sample_inner_lora_keys": sample_inner_keys,
        "old_gemini_inner_peft_load": {
            "target": "self.model.load_state_dict(ckpt['adapter'], strict=False)",
            "matching_keys_count": len(matching_inner),
            "missing_keys_count": len(inner_res.missing_keys),
            "unexpected_keys_count": len(inner_res.unexpected_keys),
            "changed_trainable_tensors_count": inner_changed_tensors,
            "total_parameter_l2_delta": inner_l2_delta,
            "status": "FAILED_NO_WEIGHTS_LOADED" if inner_changed_tensors == 0 else "PARTIAL_LOAD",
        },
        "canonical_outer_encoder_load": {
            "target": "self.load_state_dict(ckpt['adapter'], strict=False)",
            "matching_keys_count": len(matching_outer),
            "missing_keys_count": len(canonical_res.missing_keys),
            "unexpected_keys_count": len(canonical_res.unexpected_keys),
            "changed_trainable_tensors_count": canonical_changed_tensors,
            "total_parameter_l2_delta": canonical_l2_delta,
            "status": "SUCCESS_WEIGHTS_LOADED" if canonical_changed_tensors == len(ckpt_keys) else "PARTIAL",
        }
    }
    print(f"  Old method matching keys: {len(matching_inner)}/{len(ckpt_keys)}, changed tensors: {inner_changed_tensors}, L2 delta: {inner_l2_delta:.4f}")
    print(f"  Canonical matching keys: {len(matching_outer)}/{len(ckpt_keys)}, changed tensors: {canonical_changed_tensors}, L2 delta: {canonical_l2_delta:.4f}")
    return audit_entry

def main():
    print("=== RUNNING ADAPTER LOAD AUDIT ===")
    results = {}
    for name, path in ADAPTER_CHECKPOINTS.items():
        results[name] = audit_checkpoint(name, path)

    # Determine overall verdict
    all_old_zero = all(res["old_gemini_inner_peft_load"]["changed_trainable_tensors_count"] == 0 for res in results.values())
    all_canonical_full = all(res["canonical_outer_encoder_load"]["changed_trainable_tensors_count"] == res["total_checkpoint_adapter_keys"] for res in results.values())

    verdict = "PREVIOUS_E5_EXPERIMENT_INVALID_ADAPTER_LOAD" if all_old_zero else "MIXED"
    
    full_audit = {
        "schema_version": "dsc2026.gemini.huy_e5_transplant_corrected_v2.adapter_load_audit.v1",
        "verdict": verdict,
        "all_old_gemini_changed_tensors_zero": all_old_zero,
        "all_canonical_changed_tensors_full": all_canonical_full,
        "root_cause_explanation": (
            "The checkpoint 'adapter' dictionary was saved by QueryEncoder.adapter_state() using "
            "{name: param for name, param in self.named_parameters() if param.requires_grad}. "
            "Because self is QueryEncoder wrapping self.model, all parameter keys are prefixed with 'model.'. "
            "Calling self.model.load_state_dict(ckpt['adapter'], strict=False) caused every key to be treated as "
            "an unexpected key by the inner model (which does not have 'model.' prefix in its local namespace). "
            "Consequently, 0 keys matched, 0 weights were updated (L2 delta = 0.0), and the previous experiment "
            "evaluated frozen VietLegal-E5 weights in both Arm H1 and Arm H2, invalidating the conclusion of the prior run."
        ),
        "checkpoints": results
    }

    out_file = RESULTS_DIR / "ADAPTER_LOAD_AUDIT.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2, ensure_ascii=False)
    print(f"\nWrote audit to {out_file}")
    print(f"VERDICT: {verdict}")

if __name__ == "__main__":
    main()
