"""Seal the post-LAL Research V2 world state without mutating prior artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "research_v2_next_world_state"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict:
    try:
        display_path = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    post_dir = ROOT / "results" / "research_v2_post_e5"
    lal_dir = ROOT / "results" / "research_v2_lal_transfer"
    e5_dir = ROOT / "results" / "research_v2_e5_confirmation"
    legalir = ROOT.parent / "LegalIR"

    previous = read_json(post_dir / "V2_POST_E5_WORLD_MODEL.json")
    lal = read_json(lal_dir / "fold_0" / "score" / "LAL_TRANSFER_FOLD_0_REPORT.json")
    prereg = read_json(lal_dir / "V2_LAL_TASK_ADAPTATION_PREREGISTRATION.json")
    data_audit = read_json(lal_dir / "fold_0" / "DATA_AUDIT.json")
    frozen_parity = read_json(lal_dir / "fold_0" / "preflight" / "FROZEN_LAL_PARITY.json")
    mining_parity = read_json(lal_dir / "fold_0" / "preflight" / "MINING_POLICY_PARITY.json")
    smoke_parity = read_json(lal_dir / "fold_0" / "preflight" / "TRAINING_SMOKE_RESUME_PARITY.json")
    train_success = read_json(lal_dir / "fold_0" / "train" / "_SUCCESS.json")

    required_statuses = {
        "data_audit": data_audit.get("status"),
        "frozen_parity": frozen_parity.get("status"),
        "mining_parity": mining_parity.get("status"),
        "smoke_resume_parity": smoke_parity.get("status"),
    }
    if any(status != "PASS" for status in required_statuses.values()):
        raise RuntimeError(f"Fail-closed: LAL integrity status mismatch: {required_statuses}")
    if not all(lal["expert_gate_checks"].values()):
        raise RuntimeError("Fail-closed: standalone LAL expert gate did not pass")
    if lal["system_gate_checks"]["delta_gte_0_003"]:
        raise RuntimeError("Fail-closed: expected LAL system endpoint gate to fail")
    if previous["current_best_research_v2_system"]["recall_at_5"] != 0.917954036141706:
        raise RuntimeError("Fail-closed: previous V2 anchor changed")

    selector = {
        "current_anchor_recall_at_5": 0.917954036141706,
        "clean_expert_union_oracle": 0.9638392218566728,
        "target_recall_at_5": 0.96,
    }
    selector["available_oracle_residual"] = selector["clean_expert_union_oracle"] - selector["current_anchor_recall_at_5"]
    selector["gain_required_to_target"] = selector["target_recall_at_5"] - selector["current_anchor_recall_at_5"]
    selector["required_fraction_of_oracle_residual"] = selector["gain_required_to_target"] / selector["available_oracle_residual"]
    selector["oracle_margin_above_target"] = selector["clean_expert_union_oracle"] - selector["target_recall_at_5"]
    selector["gate"] = "FAIL_NOT_JUSTIFIED"
    selector["reason"] = "Reaching 0.96 would require capturing 91.63% of the clean-expert oracle residual; fully nested selection has an implausible EV/overfit profile at this geometry."

    rehab = {
        "hypothesis": "V2_SUPERVISED_EXPERT_REHABILITATION",
        "verdict": "CLOSED_NO_EXACT_RECONSTRUCTABLE_HIGH_EV_RECIPE",
        "components": [
            {
                "name": "Huy AITeamVN-FT bi-encoder",
                "checkpoint": "fine_tune/AITeamVN_Vietnamese_Embedding/model.safetensors",
                "known": "XLM-R 24-layer 1024-hidden bi-encoder; CLS+L2; no prefix; max512 in Huy scoring; full checkpoint",
                "unknown_required_fields": ["training qids", "fold exclusions", "chunk policy", "loss", "negative mining", "optimizer", "scheduler", "checkpoint selection"],
                "decision": "CLOSED_RECIPE_UNRECONSTRUCTABLE",
                "causal_reason": "A new strict-V2 recipe would be an invented adaptation experiment, not reconstruction of Huy's task-specific channel.",
            },
            {
                "name": "Huy Jina-FT cross-encoder",
                "checkpoint": "fine_tune/jina_finetuned/model.safetensors",
                "known": "Jina reranker-v2 compatible 12-layer 768-hidden scalar cross-encoder; lexical top2 parent max at inference",
                "unknown_required_fields": ["training qids", "fold exclusions", "passage construction", "loss", "negative mining", "optimizer", "scheduler", "checkpoint selection"],
                "decision": "CLOSED_RECIPE_UNRECONSTRUCTABLE_AND_NEAREST_V2_FAMILY_REJECTED",
                "causal_reason": "The exact checkpoint cannot be reconstructed, while the auditable Jina-v2 lexical parent-max pairwise LoRA family already failed strict V2 Fold-0.",
            },
            {
                "name": "LegalIR AITeamVN/Vietnamese_Reranker hard-negative LoRA",
                "known": "Historical LoRA r16/alpha32 on query/value, rank-boundary hard negatives, old folds; later high-fidelity/fusion lineage",
                "historical_evidence": "Fold-0 multi-gold increased 0.004587, with documented single-gold preamble losses; no matched strict-V2 standalone endpoint evidence",
                "decision": "CLOSED_BY_PRIOR_ART_AND_EXCLUSION_REGISTRY",
                "causal_reason": "It is a historical hard-negative cross-encoder lineage, not reconstruction of Huy AITeamVN-FT; reopening it would repeat a closed family under new folds without a new information source.",
            },
            {
                "name": "LegalIR BGE and generic reranker rehabilitation",
                "decision": "EXCLUDED_MODEL_ZOO_OR_CLOSED_HARD_NEGATIVE_LINEAGE",
                "causal_reason": "No new supervision source or representation interface distinguishes it from already screened/falsified reranker families.",
            },
        ],
        "local_proxy": "NONE_VALID",
        "reason_no_proxy": "Unknown Huy supervision contracts cannot be approximated without changing the causal hypothesis.",
    }

    lal_summary = {
        "hypothesis": "V2_LAL_TASK_ADAPTATION_TRANSFER",
        "prior_art_classification": "PARTIAL_DUPLICATE_WITH_POSITIVE_PRIOR",
        "integrity": required_statuses,
        "fold": "fold_0",
        "train_queries_after_duplicate_exclusion": data_audit.get("train_queries_after_duplicate_exclusions", data_audit.get("train_queries")),
        "held_queries": data_audit.get("held_queries", data_audit.get("heldout_queries")),
        "epochs": 2,
        "training_runtime_seconds": train_success.get(
            "runtime_seconds_this_process",
            train_success.get("runtime_seconds", train_success.get("elapsed_seconds")),
        ),
        "peak_reserved_mib": train_success.get("peak_reserved_mib"),
        "checkpoint_sha256": lal["checkpoint_sha256"],
        "expert": lal["expert"],
        "system_endpoint": lal["system_endpoint"],
        "expert_gate_checks": lal["expert_gate_checks"],
        "system_gate_checks": lal["system_gate_checks"],
        "verdict": "EXPERT_IMPROVES_BUT_COMPLEMENTARITY_COLLAPSES",
        "action": "STOP_WITHOUT_FOLDS_1_4_OR_TUNING",
        "materialization_fraction": lal["system_endpoint"]["delta"] / lal["expert"]["delta"],
    }

    now = datetime.now(timezone.utc).isoformat()
    world = {
        "schema_version": "dsc2026.research_v2.next_world_model.v1",
        "created_at_utc": now,
        "status": "SCIENTIFICALLY_CONVERGED_NO_HIGH_EV_LOCAL_BRANCH",
        "scientific_ground_truth": previous["scientific_ground_truth"],
        "authoritative_anchor": previous["current_best_research_v2_system"],
        "confirmed_adapted_e5": previous["e5_confirmation"],
        "new_hypothesis_results": [lal_summary, rehab],
        "selector_gate": selector,
        "external_supervision": {
            "status": "NOT_READY",
            "reason": "No authoritative local DSC2026 terms source or prepared external entailment dataset/recipe establishes legality, provenance, and a bounded V2 falsifier. A public-search third-party copy is insufficient for authorization.",
            "action": "Do not train or import external labels until official rules and dataset provenance are sealed.",
        },
        "closed_exact_hypotheses": previous["closed_exact_hypotheses"] + [
            "LAL query-only task adaptation for the adapted-E5+LAL equal-RRF32 endpoint",
            "Huy AITeamVN-FT reconstruction from incomplete checkpoint provenance",
            "Huy Jina-FT reconstruction from incomplete checkpoint provenance",
            "fully nested selector on the current clean expert set",
        ],
        "belief_updates": [
            "LAL task adaptation transfers strongly as a standalone expert on V2 Fold-0 (+0.017704), corroborating that query-side dense adaptation is robust across both E5 and LAL geometries.",
            "Only +0.000656 reaches the fixed adapted-E5+LAL endpoint, so most LAL adaptation gain replaces rather than complements adapted-E5 signal.",
            "The current limitation is not absence of trainable expert improvement; it is absence of sufficiently orthogonal, provenance-clean information that a simple deployment interface can materialize.",
            "Checkpoint weights without training provenance cannot be rehabilitated as strict V2 OOF experts by inventing a plausible recipe.",
            "Nested selection remains unjustified because target 0.96 needs 91.63% of the current oracle residual.",
        ],
        "next_decision": {
            "final_state": "SCIENTIFICALLY_CONVERGED_NO_HIGH_EV_LOCAL_BRANCH",
            "submission": "NOT_CREATED",
            "training": "NO_FURTHER_LOCAL_TRAINING",
            "unlock_condition": "A genuinely new, competition-legal information source with sealed provenance and >=0.005 strict-V2 recoverable headroom, or an exact reconstructable fold-isolated expert recipe distinct from closed families.",
        },
    }

    lal_report = {
        "schema_version": "dsc2026.research_v2.lal_transfer_decision.v1",
        "created_at_utc": now,
        "preregistration_sha256": sha256(lal_dir / "V2_LAL_TASK_ADAPTATION_PREREGISTRATION.json"),
        "integrity": {
            "data_audit": data_audit,
            "frozen_baseline_parity": frozen_parity,
            "mining_policy_parity": mining_parity,
            "smoke_resume_parity": smoke_parity,
        },
        "result": lal_summary,
        "gate_verdict": "STOP_EXPERT_IMPROVES_BUT_COMPLEMENTARITY_COLLAPSES",
        "full_confirmation_launched": False,
        "anti_rescue": "No rank/LR/epoch/negative/aggregation/seed/RRF weight/K/subset tuning.",
    }

    write_json(OUT / "V2_LAL_TASK_ADAPTATION_DECISION.json", lal_report)
    write_json(OUT / "V2_SUPERVISED_EXPERT_REHABILITATION_AUDIT.json", rehab)
    write_json(OUT / "V2_NEXT_WORLD_MODEL.json", world)

    lal_md = f"""# V2 LAL task-adaptation decision

## Integrity and causal contract

- Fold-0 data, frozen-score parity, exact 64-negative mining parity, and smoke/resume tensor parity: **PASS**.
- Training used the preregistered native LAL query-only Q/V LoRA contract for two epochs; no Fold-0 labels entered training/mining/model selection.
- Frozen comparator Top-5 parity: `{frozen_parity['cached_top5_exact']}/{frozen_parity['queries']}`; score max absolute error `{frozen_parity['cached_score_max_abs_error']:.3e}`.

## Result

- Standalone LAL: `{lal['expert']['base']:.9f} -> {lal['expert']['new']:.9f}` (`{lal['expert']['delta']:+.9f}`), W/L `{lal['expert']['wins']}/{lal['expert']['losses']}`, crossings `{lal['expert']['gold_crossings_into_top5']}/{lal['expert']['gold_crossings_out_of_top5']}`, multi-gold `{lal['expert']['multi_gold']['delta']:+.9f}`.
- Fixed adapted-E5 + LAL equal-RRF32 endpoint: `{lal['system_endpoint']['base']:.9f} -> {lal['system_endpoint']['new']:.9f}` (`{lal['system_endpoint']['delta']:+.9f}`), W/L `{lal['system_endpoint']['wins']}/{lal['system_endpoint']['losses']}`.
- Only `{lal_summary['materialization_fraction']:.2%}` of the standalone gain reaches the deployment endpoint.

## Verdict

`EXPERT_IMPROVES_BUT_COMPLEMENTARITY_COLLAPSES`. The expert gate passes, but the preregistered system-relevance gate (`>= +0.003`) fails. Folds 1-4 were not launched; no tuning or rescue is permitted.
"""
    (OUT / "V2_LAL_TASK_ADAPTATION_DECISION.md").write_text(lal_md, encoding="utf-8")

    rehab_md = """# V2 supervised-expert rehabilitation audit

## Decision

`CLOSED_NO_EXACT_RECONSTRUCTABLE_HIGH_EV_RECIPE`.

### Huy AITeamVN-FT

The full bi-encoder checkpoint and tokenizer establish architecture and inference behavior, but no training-query manifest, fold exclusions, chunk policy, loss, negative-mining recipe, optimizer/scheduler, or checkpoint-selection record exists. Git history does not recover those fields. Reusing its weights is provenance-unsafe; inventing a new recipe would not reconstruct Huy's channel.

### Huy Jina-FT

The shipped full cross-encoder state is compatible with Jina reranker-v2, but the same supervision fields are absent. Moreover, the auditable strict-V2 Jina-v2 lexical parent-max pairwise-LoRA family has already failed. No reconstruction or proxy is admissible.

### Historical LegalIR rerankers

LegalIR contains AITeamVN/Vietnamese_Reranker and BGE hard-negative LoRA lineages. They are historical mechanism evidence, not Huy-checkpoint recipes. The AITeam boundary lineage reported only a `+0.004587` Fold-0 multi-gold change with documented single-gold preamble losses before later high-fidelity/fusion complexity. Reopening these closed reranker families under V2 folds would add no new information source.

### External supervision

No authoritative local DSC2026 rules receipt and no prepared external entailment dataset with sealed provenance were found. External-label training is therefore not ready and was not launched.
"""
    (OUT / "V2_SUPERVISED_EXPERT_REHABILITATION_AUDIT.md").write_text(rehab_md, encoding="utf-8")

    world_md = f"""# Research V2 next world model

## Authoritative state

- Strict V2 population: 6,991 evaluable queries, 8,507 canonical parents, sealed 5 folds and duplicate isolation.
- Confirmed adapted-E5 OOF: `0.909610` Recall@5.
- Current deployable anchor remains adapted-E5 + frozen-LAL equal RRF32: `{selector['current_anchor_recall_at_5']:.9f}`.
- Clean-expert union oracle: `{selector['clean_expert_union_oracle']:.9f}`.

## New evidence

LAL query-only adaptation is a strong standalone transfer (`{lal['expert']['delta']:+.6f}` on Fold-0), but it contributes only `{lal['system_endpoint']['delta']:+.6f}` to the fixed system endpoint. This is replacement signal, not enough new complementary information, so strict confirmation was not launched.

Huy AITeamVN-FT and Jina-FT cannot be reconstructed as strict-V2 experts because their training recipes and label/fold provenance are absent. Historical LegalIR AITeam/BGE reranker families are already-covered hard-negative mechanisms, not valid proxies for those checkpoints.

## Selector gate

Moving from `{selector['current_anchor_recall_at_5']:.6f}` to `0.960000` requires `{selector['gain_required_to_target']:.6f}`. The oracle residual is only `{selector['available_oracle_residual']:.6f}`, so a selector would need to capture `{selector['required_fraction_of_oracle_residual']:.2%}` of all available oracle gain. That is not a credible generalization premise for a fully nested selector on 6,991 queries.

## Final state

`SCIENTIFICALLY_CONVERGED_NO_HIGH_EV_LOCAL_BRANCH`.

No submission was created. Reopen research only when a genuinely new, competition-legal information source has sealed provenance and at least `+0.005` strict-V2 headroom, or when an exact fold-isolated expert recipe distinct from the closed families becomes available.
"""
    (OUT / "V2_NEXT_WORLD_MODEL.md").write_text(world_md, encoding="utf-8")

    manifest_inputs = [
        e5_dir / "E5_STRICT_CONFIRMATION_REPORT.json",
        post_dir / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_REPORT.json",
        post_dir / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl",
        post_dir / "V2_POST_RRF_COMPLEMENTARITY_ANATOMY.json",
        lal_dir / "V2_LAL_TASK_ADAPTATION_COMPATIBILITY_AUDIT.md",
        lal_dir / "V2_LAL_TASK_ADAPTATION_PREREGISTRATION.json",
        lal_dir / "fold_0" / "DATA_AUDIT.json",
        lal_dir / "fold_0" / "preflight" / "FROZEN_LAL_PARITY.json",
        lal_dir / "fold_0" / "preflight" / "MINING_POLICY_PARITY.json",
        lal_dir / "fold_0" / "preflight" / "TRAINING_SMOKE_RESUME_PARITY.json",
        lal_dir / "fold_0" / "train" / "epoch-2.pt",
        lal_dir / "fold_0" / "score" / "LAL_TRANSFER_FOLD_0_PREDICTIONS.jsonl",
        lal_dir / "fold_0" / "score" / "LAL_TRANSFER_FOLD_0_REPORT.json",
        ROOT / "src" / "research_v2_lal_transfer" / "lal_transfer_runner.py",
        ROOT / "fine_tune" / "AITeamVN_Vietnamese_Embedding" / "model.safetensors",
        ROOT / "fine_tune" / "jina_finetuned" / "model.safetensors",
        legalir / "src" / "exp029_nested_lora_reranker.py",
        legalir / "src" / "exp106_cross_encoder_reranker.py",
    ]
    outputs = [
        OUT / "V2_LAL_TASK_ADAPTATION_DECISION.json",
        OUT / "V2_LAL_TASK_ADAPTATION_DECISION.md",
        OUT / "V2_SUPERVISED_EXPERT_REHABILITATION_AUDIT.json",
        OUT / "V2_SUPERVISED_EXPERT_REHABILITATION_AUDIT.md",
        OUT / "V2_NEXT_WORLD_MODEL.json",
        OUT / "V2_NEXT_WORLD_MODEL.md",
    ]
    manifest = {
        "schema_version": "dsc2026.research_v2.next_world_state_manifest.v1",
        "created_at_utc": now,
        "inputs": [artifact(path) for path in manifest_inputs],
        "outputs": [artifact(path) for path in outputs],
        "worker_state": "No Research V2/E5/LAL worker observed at seal time; unrelated user Anaconda python PID 12300 was not touched.",
        "full_lal_confirmation_launched": False,
        "submission_created": False,
    }
    write_json(OUT / "ARTIFACT_MANIFEST.json", manifest)

    print(json.dumps({
        "status": world["status"],
        "lal_expert_delta": lal["expert"]["delta"],
        "lal_system_delta": lal["system_endpoint"]["delta"],
        "selector_required_fraction": selector["required_fraction_of_oracle_residual"],
        "outputs": [str(path) for path in outputs] + [str(OUT / "ARTIFACT_MANIFEST.json")],
    }, indent=2))


if __name__ == "__main__":
    main()
