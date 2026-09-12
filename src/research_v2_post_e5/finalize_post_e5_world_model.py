"""Seal the authoritative post-E5 Research V2 state from immutable reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/research_v2_post_e5"
CONFIRM = ROOT / "results/research_v2_e5_confirmation"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def component(name: str, group: str, classification: str, role: str, evidence: str, usable: bool) -> dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "classification": classification,
        "role": role,
        "evidence": evidence,
        "usable_for_new_v2_decisions": usable,
    }


def main() -> None:
    confirmation = read(CONFIRM / "E5_STRICT_CONFIRMATION_REPORT.json")
    complementarity = read(RESULTS / "V2_COMPLEMENTARITY_AUDIT.json")
    two = read(RESULTS / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_REPORT.json")
    three = read(RESULTS / "V2_ADAPTED_E5_LAL_JINAV2_EQUAL_RRF32_REPORT.json")
    post_two = read(RESULTS / "V2_POST_RRF_COMPLEMENTARITY_ANATOMY.json")

    complementarity["bounded_integration_results"] = {
        "adapted_e5_plus_lal_equal_rrf32": {
            "status": two["status"],
            "overall": two["overall"],
            "per_fold": two["per_fold"],
            "gate_checks": two["gate_checks"],
            "report_sha256": sha(RESULTS / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_REPORT.json"),
        },
        "adapted_e5_plus_lal_plus_jinav2_equal_rrf32": {
            "status": three["status"],
            "overall": three["overall"],
            "per_fold": three["per_fold"],
            "gate_checks": three["gate_checks"],
            "report_sha256": sha(RESULTS / "V2_ADAPTED_E5_LAL_JINAV2_EQUAL_RRF32_REPORT.json"),
        },
    }
    complementarity["post_two_expert_incremental_anatomy"] = post_two
    complementarity["final_diagnosis"] = {
        "primary": "A_RANKING_INFORMATION_AND_LOW_CAPACITY_INTEGRATION",
        "secondary": "C_EVIDENCE_INFORMATION_EXISTS_BUT_NAIVE_THREE_WAY_INTERFACE_IS_UNSTABLE",
        "tertiary": "B_CANDIDATE_INFORMATION_NOT_CURRENT_PRIMARY",
        "evidence": [
            "Immutable pool ceiling 0.9819315 leaves 0.0639775 above the best deployable zero-fit system.",
            "Clean pool-restricted Top-5 set union reaches 0.9638392, but it is diagnostic and may contain up to 25 documents.",
            "Adapted-E5+LAL fixed RRF32 materializes +0.008344 robustly.",
            "Adding base Jina-v2 equally materializes only +0.002842 and is negative on folds 3 and 4, so oracle complementarity alone does not authorize expert accumulation.",
            "Broad candidate-tail expansion previously increased ceiling but harmed realized ranking; current missing-candidate mass is not the highest-EV bottleneck."
        ],
    }
    write(RESULTS / "V2_COMPLEMENTARITY_AUDIT.json", complementarity)

    components = [
        component("canonical_8507_parent_corpus", "data", "FROZEN_LABEL_FREE_V2_READY", "fixed corpus and canonical mapping", "6,991 evaluable queries; nine filtered-ground-truth queries excluded by sealed contract", True),
        component("immutable_e5_50_union_novel_bm25_10_pool", "candidate", "FROZEN_LABEL_FREE_V2_READY", "fixed candidate membership", "ceiling 0.9819315310; SHA 96a44e...e277", True),
        component("frozen_vietlegal_e5_full_corpus", "retrieval", "FROZEN_LABEL_FREE_V2_READY", "dense candidate source and exact pool rank", "R@5 0.8861632; R@150 0.9877724", True),
        component("frozen_vnlegal_lal", "retrieval", "FROZEN_LABEL_FREE_V2_READY", "independent dense candidate/rank source", "pool-restricted R@5 0.9050732; +0.0265389 union headroom over adapted-E5", True),
        component("exp021_bm25_old_fold_oof", "retrieval", "HISTORICAL_ONLY", "locked pool contributor and diagnostic sparse rank", "per-query old-fold OOF but no V2 duplicate-linked isolation; do not fit new architecture from it", False),
        component("exp111_trigram", "retrieval", "HISTORICAL_ONLY", "historical sparse expert", "old-fold supervised configuration, not strict V2 outer isolated", False),
        component("broad_cached_tail_expansion", "candidate", "HISTORICAL_ONLY", "rejected candidate expansion", "ceiling +0.004892 but zero-shot R@5 -0.007271", False),
        component("huy_lexical_top_passages", "evidence", "FROZEN_LABEL_FREE_V2_READY", "locked Jina-v2 renderer", "matched V2 A/B winner; R@5 0.8516593", True),
        component("structural_v3", "evidence", "FROZEN_LABEL_FREE_V2_READY", "available but rejected renderer", "lost lexical by -0.0688743 on all folds", False),
        component("e5_all_chunks_top2_mean", "evidence", "FROZEN_LABEL_FREE_V2_READY", "dense parent evidence/aggregation", "exact frozen parity and five-fold adapted scoring; singleton top1", True),
        component("base_jina_v2_lexical_parent_max", "frozen_neural_expert", "FROZEN_LABEL_FREE_V2_READY", "cross-attention evidence rank on immutable pool", "R@5 0.8516593; +0.0361107 union headroom over adapted-E5", True),
        component("jina_colbert_config_e_approximate", "frozen_neural_expert", "FROZEN_LABEL_FREE_V2_READY", "late-interaction rank; native and pool views", "pool R@5 0.8501049; +0.0333524 union headroom", True),
        component("adapted_vietlegal_e5_five_fold", "adapted_neural_expert", "STRICT_V2_OOF_READY", "confirmed query-only task-adapted rank", "F1-F4 unseen +0.0241820; full OOF 0.9096100", True),
        component("adapted_e5_lal_equal_rrf32", "deterministic_system", "STRICT_V2_OOF_READY", "current best zero-fit V2 system", "R@5 0.9179540; +0.0083441; 5/5 positive", True),
        component("adapted_e5_lal_jinav2_equal_rrf32", "deterministic_system", "STRICT_V2_OOF_READY", "rejected/inconclusive fixed interface", "+0.0028417 only; folds 3/4 negative; no rescue", False),
        component("jina_v2_pairwise_lora_fold0", "adapted_neural_expert", "HISTORICAL_ONLY", "rejected V2 pilot family", "Fold0 delta -0.010372; no five-fold launch", False),
        component("legalir_old_fold_aiteam_or_other_adapters", "adapted_neural_expert", "HISTORICAL_ONLY", "mechanism evidence only", "not trained/scored under V2 folds and duplicate isolation", False),
        component("huy_aiteamvn_ft_and_jina_ft", "supervised_channel", "PROVENANCE_UNSAFE", "historical Huy components only", "training-query overlap with the 6,991 V2 population is not excluded under V2 folds", False),
        component("huy_memory_profile_graph_pairwise_ltr", "supervised_channel", "PROVENANCE_UNSAFE", "historical mechanism evidence only", "CAL600/Huy lineage; not a V2 score or selection protocol", False),
        component("legalir_memory_profile_kernel_graph_router_rules", "supervised_channel", "HISTORICAL_ONLY", "negative-results/prior-art registry", "old folds and rejected complexity; no strict V2 port", False),
        component("gte_title_bge_jina_v35_encoder_screens", "frozen_neural_expert", "HISTORICAL_ONLY", "bounded historical screens", "no complete exact-V2 matched cache and model-screen family closed", False),
    ]

    world = {
        "schema_version": "dsc2026.research_v2.post_e5_world_model.v1",
        "status": "SEALED_POST_E5_LOCAL_RL",
        "scientific_ground_truth": {
            "population": "6991 evaluable queries over 8507 canonical parents",
            "folds_sha256": "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19",
            "candidate_pool_sha256": "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277",
            "duplicate_isolation": "fold-specific exact/near-duplicate-linked exclusions",
            "forbidden_validation": "CAL600 and Huy four-block LOBO are historical evidence only, never V2 selection metrics",
        },
        "e5_confirmation": {
            "verdict": confirmation["verdict"],
            "primary_folds_1_4": confirmation["folds_1_4_primary_confirmation"],
            "complete_five_fold_development_oof": confirmation["complete_five_fold_development_oof"],
            "report_sha256": sha(CONFIRM / "E5_STRICT_CONFIRMATION_REPORT.json"),
        },
        "architecture_components": components,
        "headroom": {
            "immutable_candidate_ceiling": complementarity["candidate_ceiling"],
            "adapted_e5_recall_at_5": complementarity["adapted_e5_reference_recall_at_5"],
            "best_simple_system_recall_at_5": two["overall"]["fused_recall_at_5"],
            "best_simple_system_to_candidate_ceiling": complementarity["candidate_ceiling"] - two["overall"]["fused_recall_at_5"],
            "clean_pool_expert_set_union_oracle": complementarity["clean_pool_restricted_all_expert_union"]["set_union_oracle"],
            "oracle_increment_over_adapted_e5": complementarity["clean_pool_restricted_all_expert_union"]["increment_over_adapted_e5"],
        },
        "bounded_local_experiments": {
            "two_expert_equal_rrf32": {"verdict": two["status"], "overall": two["overall"], "gate_checks": two["gate_checks"]},
            "three_expert_equal_rrf32": {"verdict": three["status"], "overall": three["overall"], "gate_checks": three["gate_checks"]},
        },
        "current_best_research_v2_system": {
            "name": "V2_ADAPTED_E5_LAL_EQUAL_RRF32",
            "architecture": "immutable pool -> strict OOF adapted-E5 rank + frozen LAL pool rank -> equal RRF32 -> deterministic Top-5",
            "recall_at_5": two["overall"]["fused_recall_at_5"],
            "precision_at_5": two["overall"]["fused_precision_at_5"],
            "single_gold_recall": two["overall"]["single_gold"]["fused"],
            "multi_gold_recall": two["overall"]["multi_gold"]["fused"],
            "submission_authorized": False,
        },
        "bottleneck_order": [
            "ranking/integration: primary; 0.0639775 within-pool gap remains above best simple system",
            "evidence/expert choice: secondary; clean cross-attention and late-interaction experts expose oracle rescue but equal three-way fusion is unstable",
            "candidate retrieval: tertiary; pool ceiling is already 0.9819315 and broad expansion harmed realized ranking",
            "new representation: only if it brings independently auditable information; model-name screening remains excluded"
        ],
        "belief_updates": [
            "EXP-112 query-only E5 adaptation transfers strongly and robustly to strict V2 fixed-pool ranking.",
            "Frozen LAL is a strong complementary dense geometry; a zero-fit interface materializes +0.008344 without fold regression.",
            "Large Top-5 union oracle does not imply equal-weight fusion success: adding Jina-v2 yields only +0.002842 and fold instability.",
            "The next difficulty is a scientifically clean selection interface, not absence of candidate or expert signal.",
        ],
        "closed_exact_hypotheses": [
            "structural-v3 evidence contract",
            "broad cached candidate-tail expansion",
            "Jina-v2 pairwise lexical parent-max LoRA",
            "adapted-E5+LAL+Jina-v2 equal RRF32",
            "weight/k/subset rescue for either fixed RRF experiment",
        ],
        "next_decision": {
            "status": "SCIENTIFIC_STOP_AFTER_TWO_BOUNDED_INTERFACES",
            "reason": "The first simple integration passed; the only admitted fallback failed its preregistered robustness gate. Further exploitation requires either a nested strict-V2 low-capacity selection design or a genuinely new external information source. Neither is authorized as another opportunistic local variant in this trajectory.",
            "submission": "not authorized; strict-V2 best is 0.917954 and no matched public inference package has been sealed",
        },
    }
    write(RESULTS / "V2_POST_E5_WORLD_MODEL.json", world)

    seal_names = [
        "V2_POST_E5_WORLD_MODEL.json",
        "V2_POST_E5_WORLD_MODEL.md",
        "V2_COMPLEMENTARITY_AUDIT.json",
        "V2_COMPLEMENTARITY_AUDIT.md",
        "NEXT_V2_HYPOTHESIS_PREREGISTRATION.json",
        "NEXT_V2_HYPOTHESIS_PREREGISTRATION.md",
        "V2_ADAPTED_E5_LAL_EQUAL_RRF32_REPORT.json",
        "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl",
        "NEXT_V2_HYPOTHESIS_PREREGISTRATION_02.json",
        "NEXT_V2_HYPOTHESIS_PREREGISTRATION_02.md",
        "V2_ADAPTED_E5_LAL_JINAV2_EQUAL_RRF32_REPORT.json",
        "V2_ADAPTED_E5_LAL_JINAV2_EQUAL_RRF32_PREDICTIONS.jsonl",
        "V2_POST_RRF_COMPLEMENTARITY_ANATOMY.json",
    ]
    source_names = [
        "v2_complementarity_audit.py",
        "audit_post_rrf_headroom.py",
        "run_adapted_e5_lal_rrf32.py",
        "run_adapted_e5_lal_jina_rrf32.py",
        "finalize_post_e5_world_model.py",
    ]
    artifacts = []
    for name in seal_names:
        path = RESULTS / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        })
    for name in source_names:
        path = ROOT / "src/research_v2_post_e5" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        })
    write(RESULTS / "V2_POST_E5_FINAL_SEAL.json", {
        "schema_version": "dsc2026.research_v2.post_e5_final_seal.v1",
        "status": "SEALED",
        "population_queries": 6991,
        "canonical_parents": 8507,
        "folds_sha256": world["scientific_ground_truth"]["folds_sha256"],
        "candidate_pool_sha256": world["scientific_ground_truth"]["candidate_pool_sha256"],
        "best_system": world["current_best_research_v2_system"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    })
    print(json.dumps({
        "status": world["status"],
        "best_system": world["current_best_research_v2_system"],
        "bottleneck_order": world["bottleneck_order"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
