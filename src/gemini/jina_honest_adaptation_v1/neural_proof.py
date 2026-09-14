"""Neural training proof scaffold enforcing Section 7 and Section 3 requirements.
Captures pre-training, in-training, and post-training parameter shifts,
all-tensor fingerprints, gradient norms, audit logits with microbatch=4,
and hardware utilization.
Produces results/gemini/jina_honest_adaptation_v1/NEURAL_TRAINING_PROOF.json.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from common import EXP_RESULTS, get_git_info, sha256_file


def tensor_sha256(t: torch.Tensor) -> str:
    """Compute SHA256 hash of tensor byte representation."""
    cpu_bytes = t.detach().cpu().to(torch.float32).numpy().tobytes()
    return hashlib.sha256(cpu_bytes).hexdigest()


class NeuralTrainingProofTracker:
    def __init__(self, stage_name: str, audit_pairs: List[Tuple[str, str]]):
        self.stage_name = stage_name
        self.audit_pairs = audit_pairs[:256]  # exactly 256 audit pairs
        self.pre_record: Dict[str, Any] = {}
        self.step_records: List[Dict[str, Any]] = []
        self.post_record: Dict[str, Any] = {}
        self.pre_weights: Dict[str, torch.Tensor] = {}
        self.pre_fingerprints: Dict[str, str] = {}

        # Section 3.3 Bookkeeping: track separately
        self.actual_optimizer_steps = 0
        self.gradient_log_samples = 0
        self.nonzero_gradient_samples = 0

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Execute optimizer.step() and reliably track actual optimizer steps."""
        optimizer.step()
        self.actual_optimizer_steps += 1

    def before_training(self, model: torch.nn.Module, tok: Any, device: str = "cuda") -> None:
        """Capture pre-training state, parameter counts, fingerprints, and frozen audit logits."""
        torch.cuda.reset_peak_memory_stats()
        original_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Trainable tensor fingerprints for ALL trainable parameters (Section 3.3)
        self.pre_fingerprints = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                h = tensor_sha256(param)
                self.pre_fingerprints[name] = h
                self.pre_weights[name] = param.detach().clone().cpu().float()

        # Classifier head fingerprint
        classifier_fingerprint = {}
        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name:
                classifier_fingerprint[name] = tensor_sha256(param)

        # Frozen audit logits on 256 pairs with microbatch <= 4 (Section 3.1)
        model.eval()
        audit_logits = []
        micro_bs = 4
        with torch.no_grad():
            for i in range(0, len(self.audit_pairs), micro_bs):
                batch = self.audit_pairs[i : i + micro_bs]
                inputs = tok(
                    [p[0] for p in batch],
                    [p[1] for p in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                logits = model(**inputs).logits.view(-1).float().cpu().tolist()
                audit_logits.extend(logits)

        # Audit-pair diversity verification (Section 3.4)
        unique_queries = len(set(p[0] for p in self.audit_pairs))
        unique_passages = len(set(hashlib.sha256(p[1].encode("utf-8")).hexdigest() for p in self.audit_pairs))
        frozen_logit_std = float(np.std(audit_logits)) if audit_logits else 0.0

        self.pre_record = {
            "model_class": model.__class__.__name__,
            "total_parameter_count": original_params,
            "trainable_parameter_count": trainable_params,
            "trainable_parameter_fraction": trainable_params / original_params,
            "trainable_tensor_count": len(self.pre_fingerprints),
            "trainable_fingerprints_sample": {
                k: self.pre_fingerprints[k] for k in list(self.pre_fingerprints.keys())[:10]
            },
            "classifier_fingerprint": classifier_fingerprint,
            "audit_pairs_count": len(self.audit_pairs),
            "audit_pairs_diversity": {
                "unique_query_count": unique_queries,
                "unique_passage_count": unique_passages,
                "frozen_logit_std": frozen_logit_std,
                "diversity_verified": unique_queries >= 200 and unique_passages >= 200,
            },
            "frozen_audit_logits_sample": audit_logits[:10],
            "_frozen_audit_logits": audit_logits,
        }

    def record_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        model: torch.nn.Module,
        lr: float,
        optimizer_class: str,
    ) -> None:
        """Record gradient norm, non-zero gradient fraction, and memory."""
        grad_norms = []
        nonzero_grads = 0
        total_grads = 0

        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach()
                grad_norms.append(float(g.norm(2).cpu().item()))
                nonzero_grads += int((g != 0).sum().cpu().item())
                total_grads += g.numel()

        total_grad_norm = float(np.sqrt(sum(g**2 for g in grad_norms))) if grad_norms else 0.0
        nonzero_fraction = (nonzero_grads / total_grads) if total_grads > 0 else 0.0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

        self.gradient_log_samples += 1
        if total_grad_norm > 0:
            self.nonzero_gradient_samples += 1

        self.step_records.append(
            {
                "step": step,
                "epoch": epoch,
                "loss": float(loss),
                "total_grad_norm": total_grad_norm,
                "nonzero_grad_fraction": nonzero_fraction,
                "learning_rate": lr,
                "optimizer": optimizer_class,
                "peak_vram_mb": round(peak_vram, 1),
            }
        )

    def after_training(
        self,
        model: torch.nn.Module,
        tok: Any,
        checkpoint_path: Optional[Path] = None,
        val_loss: Optional[float] = None,
        val_recall_at_5: Optional[float] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Capture post-training state, compute L2 parameter delta and logit changes."""
        model.eval()
        post_fingerprints = {}
        tensors_changed = 0
        l2_delta_sq = 0.0

        # Compare ALL trainable tensors (Section 3.3)
        for name, param in model.named_parameters():
            if param.requires_grad:
                h = tensor_sha256(param)
                post_fingerprints[name] = h
                if name in self.pre_weights:
                    pre = self.pre_weights[name]
                    post = param.detach().clone().cpu().float()
                    diff = (post - pre).norm(2).item()
                    l2_delta_sq += diff**2
                    if name in self.pre_fingerprints and h != self.pre_fingerprints[name]:
                        tensors_changed += 1

        total_l2_delta = float(np.sqrt(l2_delta_sq))

        # Adapted audit logits on 256 pairs with microbatch <= 4 (Section 3.1)
        adapted_audit_logits = []
        micro_bs = 4
        with torch.no_grad():
            for i in range(0, len(self.audit_pairs), micro_bs):
                batch = self.audit_pairs[i : i + micro_bs]
                inputs = tok(
                    [p[0] for p in batch],
                    [p[1] for p in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                logits = model(**inputs).logits.view(-1).float().cpu().tolist()
                adapted_audit_logits.extend(logits)

        frozen_logits = self.pre_record.get("_frozen_audit_logits", [])
        logit_diffs = [abs(a - f) for a, f in zip(adapted_audit_logits, frozen_logits)]
        mean_abs_logit_change = float(np.mean(logit_diffs)) if logit_diffs else 0.0
        max_abs_logit_change = float(np.max(logit_diffs)) if logit_diffs else 0.0

        mean_grad_norm = (
            float(np.mean([s["total_grad_norm"] for s in self.step_records]))
            if self.step_records
            else 0.0
        )

        neural_proof_passed = (
            self.actual_optimizer_steps > 0
            and self.nonzero_gradient_samples > 0
            and total_l2_delta > 1e-4
            and mean_abs_logit_change > 1e-3
            and tensors_changed > 0
        )

        report = {
            "schema_version": "dsc2026.gemini.neural_training_proof.v2",
            "status": "PASS" if neural_proof_passed else "NEURAL_TRAINING_NOT_EXECUTED",
            "stage": self.stage_name,
            "device": torch.cuda.get_device_name(0),
            "pre_training": {
                "model_class": self.pre_record["model_class"],
                "total_parameters": self.pre_record["total_parameter_count"],
                "trainable_parameters": self.pre_record["trainable_parameter_count"],
                "trainable_fraction": self.pre_record["trainable_parameter_fraction"],
                "trainable_tensor_count": self.pre_record["trainable_tensor_count"],
                "classifier_fingerprint": self.pre_record["classifier_fingerprint"],
                "audit_pairs_diversity": self.pre_record.get("audit_pairs_diversity"),
            },
            "training_dynamics": {
                "actual_optimizer_steps": self.actual_optimizer_steps,
                "gradient_log_samples": self.gradient_log_samples,
                "nonzero_gradient_samples": self.nonzero_gradient_samples,
                "mean_grad_norm": mean_grad_norm,
                "initial_loss": self.step_records[0]["loss"] if self.step_records else None,
                "final_loss": self.step_records[-1]["loss"] if self.step_records else None,
                "validation_loss": val_loss,
                "validation_recall_at_5": val_recall_at_5,
                "peak_vram_mb": max([s["peak_vram_mb"] for s in self.step_records])
                if self.step_records
                else 0.0,
                "step_trace_sample": self.step_records[:5] + self.step_records[-5:],
            },
            "parameter_shift_verification": {
                "total_trainable_tensors": len(self.pre_fingerprints),
                "tensors_changed": tensors_changed,
                "all_tensors_changed": tensors_changed == len(self.pre_fingerprints),
                "total_l2_parameter_delta": total_l2_delta,
                "audit_pairs_count": len(self.audit_pairs),
                "mean_abs_logit_change": mean_abs_logit_change,
                "max_abs_logit_change": max_abs_logit_change,
                "frozen_vs_adapted_logits_sample": [
                    {"frozen": f, "adapted": a, "diff": abs(a - f)}
                    for f, a in zip(frozen_logits[:5], adapted_audit_logits[:5])
                ],
            },
            "checkpoint": {
                "path": str(checkpoint_path) if checkpoint_path else "IN_MEMORY",
                "sha256": sha256_file(checkpoint_path)
                if checkpoint_path and checkpoint_path.exists()
                else "N/A",
            },
            "git": get_git_info(),
        }

        out_path = EXP_RESULTS / "NEURAL_TRAINING_PROOF.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} (Status: {report['status']}, L2 delta: {total_l2_delta:.6f})")
        return report
