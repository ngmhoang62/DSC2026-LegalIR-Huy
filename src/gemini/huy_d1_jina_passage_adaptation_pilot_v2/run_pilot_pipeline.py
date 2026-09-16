"""Master pipeline orchestrator for HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .audit_shipped_jina_parity import run_shipped_jina_parity
from .audit_split import run_split_audit
from .audit_trainable_contract import run_trainable_parameter_audit
from .build_decision_and_artifacts import run_decision_and_artifacts
from .build_teacher_cache import build_teacher_cache
from .common import RES_DIR, ROOT, SRC_DIR, get_git_status
from .evaluate_cal_zero_shot import evaluate_cal_zero_shot
from .evaluate_held_v2 import evaluate_held_v2
from .test_adapter_reload_parity import run_reload_parity_test
from .train_passage_adaptation import train_adaptation


def main():
    print("==================================================================", flush=True)
    print("  EXPERIMENT: HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2", flush=True)
    print("==================================================================", flush=True)
    t_start = time.time()

    # Stage 0: Git hard gate
    print("\n>>> STAGE 0: GIT REPOSITORY HARD GATE <<<", flush=True)
    git_info = get_git_status()
    print(f"HEAD commit:        {git_info.get('head_commit')}", flush=True)
    print(f"origin/main commit: {git_info.get('origin_main_commit')}", flush=True)
    print(f"Commit parity:      {git_info.get('parity')}", flush=True)
    print(f"Tree clean:         {git_info.get('status_clean')}", flush=True)

    if not git_info.get("parity") or not git_info.get("status_clean"):
        raise RuntimeError(
            f"GIT_HARD_GATE_BLOCKED: Source must be committed and pushed to origin/main before authoritative execution!\n"
            f"Parity: {git_info.get('parity')}, Clean: {git_info.get('status_clean')}\n"
            f"Porcelain: {git_info.get('porcelain_output')}"
        )

    # Stage 1: Split & Anti-contamination audit
    print("\n>>> STAGE 1: DATA SPLIT AUDIT <<<", flush=True)
    run_split_audit()

    # Stage 2: Shipped Jina parity audit
    print("\n>>> STAGE 2: SHIPPED JINA PARITY AUDIT <<<", flush=True)
    run_shipped_jina_parity()

    # Stage 3: Trainable parameter contract audit
    print("\n>>> STAGE 3: TRAINABLE PARAMETER CONTRACT AUDIT <<<", flush=True)
    run_trainable_parameter_audit()

    # Stage 4: Teacher score precomputation and sealing
    print("\n>>> STAGE 4: TEACHER SCORE CACHE BUILD & SEAL <<<", flush=True)
    build_teacher_cache()

    # Stage 5: Pure-LoRA Adaptation Training
    print("\n>>> STAGE 5: PURE-LORA ADAPTATION TRAINING <<<", flush=True)
    train_adaptation()

    # Stage 6: Adapter reload parity test
    print("\n>>> STAGE 6: ADAPTER RELOAD PARITY TEST <<<", flush=True)
    run_reload_parity_test()

    # Stage 7: Held non-CAL V2 evaluation (Fold 0)
    print("\n>>> STAGE 7: HELD NON-CAL V2 EVALUATION <<<", flush=True)
    evaluate_held_v2()

    # Stage 8: CAL600 zero-shot transfer evaluation & D1 diagnostic
    print("\n>>> STAGE 8: CAL600 ZERO-SHOT TRANSFER EVALUATION <<<", flush=True)
    evaluate_cal_zero_shot()

    # Stage 9: Score collapse audit, decision synthesis & report consistency
    print("\n>>> STAGE 9: DECISION & ARTIFACT SYNTHESIS <<<", flush=True)
    result = run_decision_and_artifacts()

    t_total = time.time() - t_start
    print("\n==================================================================", flush=True)
    print(f"  PIPELINE COMPLETE: {result['verdict']} (Total: {t_total:.1f}s)", flush=True)
    print("==================================================================", flush=True)


if __name__ == "__main__":
    main()
