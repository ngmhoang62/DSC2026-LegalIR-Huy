"""Master pipeline orchestrator for HUY_D1_SELECTIVE_REPAIR_V1."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_expanded_fusion_selection import ltr_features

from src.gemini.huy_d1_selective_repair_v1.common import (
    BOOTSTRAP_REPLICAS,
    BOOTSTRAP_THRESHOLD,
    CAL_CONTEXTS_DIR,
    CAL_FROZEN_SECTION_CACHE_PATH,
    CAL_QUESTIONS_LABEL_FREE_PATH,
    CANONICAL_V2_CONTEXTS_JSONL,
    CANONICAL_V2_QUERIES_JSONL,
    D1_VIEWS,
    EXPECTED_BLOCK_RECALLS,
    EXPECTED_D1_R5,
    EXPECTED_OLD_JINA_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    OLD_JINA_CACHE_PATH,
    RESULTS_DIR,
    SEED,
    SRC_DIR,
    get_git_status,
    get_source_files_sha256,
    load_cal_data_label_free,
    load_cal_gold_labels,
    seed_everything,
    sha256_file,
)
from src.gemini.huy_d1_selective_repair_v1.evaluate_arms import (
    compute_arm_metrics,
    compute_intervention_churn_and_utility,
    evaluate_parity,
)
from src.gemini.huy_d1_selective_repair_v1.tier1_citation import (
    apply_tier1_repair,
    index_corpus_own_references,
    index_v2_corpus_own_references,
    resolve_tier1_anchor,
    run_cal_citation_audit,
    run_v2_citation_shadow,
)
from src.gemini.huy_d1_selective_repair_v1.tier2_bootstrap import (
    evaluate_bootstrap_committee,
    find_rank6_proposals,
    load_tier2_score_caches,
)


def run_experiment() -> Dict[str, Any]:
    print("==================================================================", flush=True)
    print("STARTING EXPERIMENT: HUY_D1_SELECTIVE_REPAIR_V1", flush=True)
    print("==================================================================", flush=True)

    start_time_utc = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    # 1. Source Provenance Audit
    git_info = get_git_status()
    source_files = get_source_files_sha256()

    print(f"Git HEAD Commit:         {git_info.get('head_commit')}", flush=True)
    print(f"Git Origin/Main Commit:  {git_info.get('origin_main_commit')}", flush=True)
    print(f"Remote Parity:           {git_info.get('parity')}", flush=True)
    print(f"Working Tree Clean:      {git_info.get('status_clean')}", flush=True)

    provenance_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.source_provenance.v1",
        "experiment_id": "HUY_D1_SELECTIVE_REPAIR_V1",
        "timestamp_utc": start_time_utc,
        "git": git_info,
        "source_files": source_files,
    }
    (RESULTS_DIR / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Verify frozen cache hashes
    actual_sec_sha = sha256_file(CAL_FROZEN_SECTION_CACHE_PATH)
    actual_jina_sha = sha256_file(OLD_JINA_CACHE_PATH)
    print(f"Section CE Cache SHA256: {actual_sec_sha}", flush=True)
    print(f"Old Jina Cache SHA256:   {actual_jina_sha}", flush=True)
    if actual_sec_sha != EXPECTED_SECTION_CE_SHA256:
        raise RuntimeError(f"Section CE cache SHA mismatch: expected {EXPECTED_SECTION_CE_SHA256}, got {actual_sec_sha}")
    if actual_jina_sha != EXPECTED_OLD_JINA_SHA256:
        raise RuntimeError(f"Old Jina cache SHA mismatch: expected {EXPECTED_OLD_JINA_SHA256}, got {actual_jina_sha}")

    # 2. Load CAL Data (label-free)
    print("\n--- LOADING CAL DATA (LABEL-FREE) ---", flush=True)
    (
        docs,
        queries_label_free,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
    ) = load_cal_data_label_free()

    # 3. Fit Exact D1 LOBO (R0 Baseline)
    print("\n--- FITTING D1 LOBO MODELS (48D LIBLINEAR) ---", flush=True)
    # Note: Gold labels are strictly required for fitting the supervised LOBO models
    gold, reveal_time_utc = load_cal_gold_labels(all_ids)

    d1_rankings: Dict[str, List[str]] = {}
    d1_scores: Dict[str, np.ndarray] = {}
    eval_rows_all: Dict[str, np.ndarray] = {}
    eval_groups_all: Dict[str, List[str]] = {}

    for held in sorted(blocks.keys()):
        train_ids = sum((blocks[n] for n in blocks if n != held), [])
        test_ids = blocks[held]
        eval_ids = train_ids + test_ids

        eval_rows, eval_groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, full_channels_cv
        )
        for q in eval_rows:
            eval_rows[q] = np.concatenate(
                [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        X_train = np.vstack([eval_rows[q] for q in train_ids])
        y_train = np.concatenate(
            [[d in gold[q] for d in eval_groups[q]] for q in train_ids]
        ).astype(np.int8)

        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X_train), y_train)

        for q in test_ids:
            X_test = scaler.transform(eval_rows[q])
            dec_scores = model.decision_function(X_test)
            order = sorted(
                range(len(dec_scores)), key=lambda i: dec_scores[i], reverse=True
            )
            ranking = [eval_groups[q][i] for i in order]
            d1_rankings[q] = ranking
            d1_scores[q] = dec_scores
            eval_rows_all[q] = eval_rows[q]
            eval_groups_all[q] = eval_groups[q]

    r0_predictions = {q: d1_rankings[q][:5] for q in all_ids}
    r0_metrics = compute_arm_metrics("R0_EXACT_D1", r0_predictions, gold, all_ids, blocks)

    # 4. Parity Verification Gate
    print("\n--- VERIFYING D1 EXACT PARITY GATE ---", flush=True)
    parity_doc = evaluate_parity(r0_metrics)
    (RESULTS_DIR / "D1_PARITY.json").write_text(
        json.dumps(parity_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"R0 Recall@5: {r0_metrics['recall_at_5']:.16f} (Expected: {EXPECTED_D1_R5:.16f})", flush=True)
    print(f"Parity Status: {parity_doc['status']}", flush=True)
    if not parity_doc["parity_pass"]:
        print("BLOCKED_D1_PARITY: Parity verification failed! Stopping.", flush=True)
        return {"status": "BLOCKED_D1_PARITY"}

    # 5. Tier 1: Exact Legal Citation Protected Injection
    print("\n--- TIER 1: EXACT LEGAL CITATION REPAIR & AUDIT ---", flush=True)
    cal_ref_to_docs, cal_doc_to_own_ref = index_corpus_own_references(CAL_CONTEXTS_DIR)
    v2_ref_to_docs, v2_doc_to_own_ref = index_v2_corpus_own_references(CANONICAL_V2_CONTEXTS_JSONL)

    cal_citation_audit = run_cal_citation_audit(
        all_ids=all_ids,
        queries_label_free=queries_label_free,
        d1_predictions=r0_predictions,
        gold=gold,
        ref_to_docs=cal_ref_to_docs,
    )
    (RESULTS_DIR / "EXACT_CITATION_ANCHOR_AUDIT.json").write_text(
        json.dumps(cal_citation_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    v2_citation_shadow = run_v2_citation_shadow(v2_ref_to_docs)
    (RESULTS_DIR / "STRICTV2_CITATION_SHADOW.json").write_text(
        json.dumps(v2_citation_shadow, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"CAL Citation Queries with parsed refs: {cal_citation_audit['queries_containing_parsed_references']}", flush=True)
    print(f"CAL Unique exact matches:              {cal_citation_audit['unique_exact_corpus_matches']}", flush=True)
    print(f"CAL Interventions:                     {cal_citation_audit['intervention_count']}", flush=True)
    print(f"CAL Anchor Precision:                  {cal_citation_audit['anchor_gold_precision']:.4f}", flush=True)
    print(f"V2 Shadow Evaluable Matches:           {v2_citation_shadow.get('sample_size', 0)}", flush=True)
    print(f"V2 Shadow Anchor Precision:            {v2_citation_shadow.get('matched_anchor_gold_precision', 0.0):.4f}", flush=True)

    # Materialize Arm R1
    r1_predictions = {}
    r1_action_records = {}
    for q in all_ids:
        qval = queries_label_free[q]
        qtext = qval[0] if isinstance(qval, (list, tuple)) else str(qval)
        anchor_doc, status, refs = resolve_tier1_anchor(qtext, cal_ref_to_docs)
        new_top5, act, inj, evict = apply_tier1_repair(d1_rankings[q], anchor_doc)
        r1_predictions[q] = new_top5
        if act == "EXACT_CITATION_INJECTION":
            r1_action_records[q] = {
                "qid": q,
                "tier": "TIER_1",
                "action": "EXACT_CITATION_INJECTION",
                "anchor_doc": inj,
                "evicted_doc": evict,
                "anchor_is_gold": inj in gold[q],
                "evicted_is_gold": evict in gold[q],
            }

    # 6. Tier 2: Bootstrap-Consensus Rank-6 Boundary Repair
    print("\n--- TIER 2: BOOTSTRAP-CONSENSUS RANK-6 BOUNDARY REPAIR ---", flush=True)
    sec_scores, jina_scores = load_tier2_score_caches()
    proposals_by_block = find_rank6_proposals(
        all_ids=all_ids,
        blocks=blocks,
        d1_rankings=d1_rankings,
        d1_scores=d1_scores,
        eval_groups=eval_groups_all,
        sec_scores=sec_scores,
        jina_scores=jina_scores,
    )

    all_proposals, tier2_diagnostics = evaluate_bootstrap_committee(
        blocks=blocks,
        proposals_by_block=proposals_by_block,
        eval_rows=eval_rows_all,
        eval_groups=eval_groups_all,
        gold=gold,
        replicas=BOOTSTRAP_REPLICAS,
        threshold=BOOTSTRAP_THRESHOLD,
    )
    (RESULTS_DIR / "BOOTSTRAP_BOUNDARY_DIAGNOSTICS.json").write_text(
        json.dumps(tier2_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    passing_proposals = tier2_diagnostics["actions"]
    print(f"Total Rank-6 Proposals:                {tier2_diagnostics['summary']['total_rank6_proposals_before_bootstrap_gate']}", flush=True)
    print(f"Proposals Passing >= {BOOTSTRAP_THRESHOLD}/{BOOTSTRAP_REPLICAS} Votes:   {len(passing_proposals)}", flush=True)

    # Materialize Arm R2
    passing_by_qid = {p["qid"]: p for p in passing_proposals}
    r2_predictions = {}
    r2_action_records = {}
    for q in all_ids:
        top5 = list(d1_rankings[q][:5])
        if q in passing_by_qid:
            p = passing_by_qid[q]
            chal = p["challenger"]
            defend = p["defender"]
            # Swap rank 5
            r2_top5 = top5[:4] + [chal]
            r2_predictions[q] = r2_top5
            r2_action_records[q] = {
                "qid": q,
                "tier": "TIER_2",
                "action": "BOOTSTRAP_RANK6_SWAP",
                "defender": defend,
                "challenger": chal,
                "votes": p["bootstrap_votes_for_challenger"],
                "defender_is_gold": defend in gold[q],
                "challenger_is_gold": chal in gold[q],
            }
        else:
            r2_predictions[q] = top5

    # 7. Combined Arm R3
    print("\n--- ARM R3: COMBINED SELECTIVE REPAIR ---", flush=True)
    r3_predictions = {}
    r3_action_records = {}
    for q in all_ids:
        # Tier 1 precedence
        if q in r1_action_records:
            r3_predictions[q] = r1_predictions[q]
            r3_action_records[q] = {
                "qid": q,
                "active_tier": "TIER_1",
                "tier1_action": r1_action_records[q],
                "tier2_action": None,
                "tier2_suppressed": q in passing_by_qid,
            }
        elif q in r2_action_records:
            r3_predictions[q] = r2_predictions[q]
            r3_action_records[q] = {
                "qid": q,
                "active_tier": "TIER_2",
                "tier1_action": None,
                "tier2_action": r2_action_records[q],
                "tier2_suppressed": False,
            }
        else:
            r3_predictions[q] = list(r0_predictions[q])

    # 8. Compute Arm Metrics & Churn/Utility
    print("\n--- EVALUATING ALL ARMS (R0, R1, R2, R3) ---", flush=True)
    r1_metrics = compute_arm_metrics("R1_CITATION_ONLY", r1_predictions, gold, all_ids, blocks)
    r2_metrics = compute_arm_metrics("R2_BOOTSTRAP_ONLY", r2_predictions, gold, all_ids, blocks)
    r3_metrics = compute_arm_metrics("R3_COMBINED_SELECTIVE_REPAIR", r3_predictions, gold, all_ids, blocks)

    r1_utility = compute_intervention_churn_and_utility(r0_predictions, r1_predictions, gold, all_ids)
    r2_utility = compute_intervention_churn_and_utility(r0_predictions, r2_predictions, gold, all_ids)
    r3_utility = compute_intervention_churn_and_utility(r0_predictions, r3_predictions, gold, all_ids)

    # 9. Action Log
    action_log_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.action_log.v1",
        "experiment_id": "HUY_D1_SELECTIVE_REPAIR_V1",
        "r1_actions": r1_action_records,
        "r2_actions": r2_action_records,
        "r3_actions": r3_action_records,
        "r3_action_cases_detailed": [
            {
                "qid": q,
                "query_text": queries_label_free[q][0] if isinstance(queries_label_free[q], (list, tuple)) else str(queries_label_free[q]),
                "action_info": r3_action_records[q],
                "r0_top5": r0_predictions[q],
                "r3_top5": r3_predictions[q],
                "gold_set": list(gold[q]),
                "r0_hits": len(set(r0_predictions[q]) & gold[q]),
                "r3_hits": len(set(r3_predictions[q]) & gold[q]),
                "recall_gain": float((len(set(r3_predictions[q]) & gold[q]) - len(set(r0_predictions[q]) & gold[q])) / max(1, len(gold[q]))),
            }
            for q in sorted(r3_action_records.keys())
        ],
    }
    (RESULTS_DIR / "SELECTIVE_REPAIR_ACTION_LOG.json").write_text(
        json.dumps(action_log_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 10. Evaluate Generalization / Safety Gates for R3
    print("\n--- EVALUATING PRODUCTION GATES FOR R3 ---", flush=True)

    r0_r5 = r0_metrics["recall_at_5"]
    r3_r5 = r3_metrics["recall_at_5"]

    gate_r0_parity = parity_doc["parity_pass"]
    gate_recall_gain = (r3_r5 >= r0_r5 + 0.0015)
    gate_paired_losses = (r3_utility["pairwise_vs_r0"]["losses"] == 0)
    gate_precision = (r3_metrics["precision_at_5"] >= r0_metrics["precision_at_5"])
    gate_block_monotonic = all(
        r3_metrics["block_recalls"][b] >= r0_metrics["block_recalls"][b]
        for b in sorted(blocks.keys())
    )
    gate_single_gold = (r3_metrics["single_gold_recall_at_5"] >= r0_metrics["single_gold_recall_at_5"])
    gate_multi_gold = (r3_metrics["multi_gold_recall_at_5"] >= r0_metrics["multi_gold_recall_at_5"])

    # Tier 2 specific gates
    t2_actions_count = r2_utility["total_interventions"]
    t2_harmful_count = r2_utility["harmful_interventions"]
    t2_beneficial_count = r2_utility["beneficial_interventions"]
    t2_fraction_gate = (
        (t2_beneficial_count / t2_actions_count >= 0.25)
        if t2_actions_count > 0
        else True
    )

    gate_t2_count = t2_actions_count <= 15
    gate_t2_harmful = t2_harmful_count == 0
    gate_t2_beneficial_fraction = t2_fraction_gate

    # Tier 1 specific gates
    t1_harmful_count = cal_citation_audit["top5_interventions"]["harmful"]
    t1_anchor_prec = cal_citation_audit["anchor_gold_precision"]

    gate_t1_harmful = t1_harmful_count == 0
    gate_t1_prec = t1_anchor_prec >= 0.95

    # Strict-V2 shadow gate
    v2_sample = v2_citation_shadow.get("sample_size", 0)
    v2_prec = v2_citation_shadow.get("matched_anchor_gold_precision", 0.0)
    if v2_sample >= 20:
        gate_v2_shadow = v2_prec >= 0.90
    else:
        gate_v2_shadow = True

    all_gates = {
        "gate_01_r0_parity_pass": gate_r0_parity,
        "gate_02_r3_recall_gain_ge_0015": gate_recall_gain,
        "gate_03_r3_paired_losses_eq_0": gate_paired_losses,
        "gate_04_r3_precision_no_decrease": gate_precision,
        "gate_05_no_cal_block_decreases": gate_block_monotonic,
        "gate_06_single_gold_no_decrease": gate_single_gold,
        "gate_07_multi_gold_no_decrease": gate_multi_gold,
        "gate_08_tier2_total_actions_le_15": gate_t2_count,
        "gate_09_tier2_harmful_actions_eq_0": gate_t2_harmful,
        "gate_10_tier2_beneficial_fraction_ge_025": gate_t2_beneficial_fraction,
        "gate_11_tier1_harmful_interventions_eq_0": gate_t1_harmful,
        "gate_12_tier1_anchor_precision_ge_095": gate_t1_prec,
        "gate_13_v2_shadow_anchor_precision_ge_090": gate_v2_shadow,
    }

    all_passed = all(all_gates.values())
    final_verdict = "LOCAL_PROMOTE_SELECTIVE_REPAIR_V1" if all_passed else "KILL_SELECTIVE_REPAIR_V1"

    print(f"\nFinal Verdict: {final_verdict}", flush=True)
    for g_name, g_val in all_gates.items():
        print(f"  {g_name}: {'PASS' if g_val else 'FAIL'}", flush=True)

    # 11. Write LOCAL_SELECTIVE_REPAIR_ARMS.json
    arms_summary_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.local_arms.v1",
        "experiment_id": "HUY_D1_SELECTIVE_REPAIR_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_verdict": final_verdict,
        "all_gates_pass": all_passed,
        "gates_evaluation": all_gates,
        "arms_metrics": {
            "R0_EXACT_D1": r0_metrics,
            "R1_CITATION_ONLY": r1_metrics,
            "R2_BOOTSTRAP_ONLY": r2_metrics,
            "R3_COMBINED_SELECTIVE_REPAIR": r3_metrics,
        },
        "utility_and_churn_vs_r0": {
            "R1_CITATION_ONLY": r1_utility,
            "R2_BOOTSTRAP_ONLY": r2_utility,
            "R3_COMBINED_SELECTIVE_REPAIR": r3_utility,
        },
        "block_deltas_vs_r0": {
            arm_id: {
                b: float(m["block_recalls"][b] - r0_metrics["block_recalls"][b])
                for b in sorted(blocks.keys())
            }
            for arm_id, m in [("R1", r1_metrics), ("R2", r2_metrics), ("R3", r3_metrics)]
        },
    }
    (RESULTS_DIR / "LOCAL_SELECTIVE_REPAIR_ARMS.json").write_text(
        json.dumps(arms_summary_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 12. Write EXPERIMENT_MANIFEST.json
    manifest_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.manifest.v1",
        "experiment_id": "HUY_D1_SELECTIVE_REPAIR_V1",
        "source_head_commit": git_info.get("head_commit"),
        "source_origin_commit": git_info.get("origin_main_commit"),
        "git_clean": git_info.get("status_clean"),
        "critical_input_hashes": {
            "CAL_FROZEN_SECTION_CACHE_PATH": actual_sec_sha,
            "OLD_JINA_CACHE_PATH": actual_jina_sha,
            "CAL_QUESTIONS_LABEL_FREE_PATH": sha256_file(CAL_QUESTIONS_LABEL_FREE_PATH),
            "CANONICAL_V2_CONTEXTS_JSONL": sha256_file(CANONICAL_V2_CONTEXTS_JSONL),
            "CANONICAL_V2_QUERIES_JSONL": sha256_file(CANONICAL_V2_QUERIES_JSONL),
        },
        "exact_d1_learner_config": {
            "model": "StandardScaler() + LogisticRegression",
            "C": 0.15,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 3000,
            "random_state": 2026,
            "rank_views": D1_VIEWS,
            "feature_dim": 48,
        },
        "citation_tier1_config": {
            "logic": "Exact query reference regex extraction + corpus own-number resolution",
            "relation_expansion": False,
            "action": "Replace D1 rank 5 if unique exact anchor outside Top-5",
        },
        "bootstrap_tier2_config": {
            "proposals": "rank6 challenger in frozen Section Top 5, Section(chal)>Section(def), Jina(chal)>Jina(def)",
            "replicas_count": BOOTSTRAP_REPLICAS,
            "threshold_votes": BOOTSTRAP_THRESHOLD,
            "threshold_rate": float(BOOTSTRAP_THRESHOLD / BOOTSTRAP_REPLICAS),
            "seed_policy": "2026000 + outer_fold_index * 1000 + b",
        },
        "arms_evaluated": ["R0", "R1", "R2", "R3"],
        "final_verdict": final_verdict,
        "submission_zip_generated": False,
    }
    (RESULTS_DIR / "EXPERIMENT_MANIFEST.json").write_text(
        json.dumps(manifest_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== EXPERIMENT COMPLETE ===", flush=True)
    print(f"Artifacts saved in: {RESULTS_DIR.resolve()}", flush=True)
    return arms_summary_doc


if __name__ == "__main__":
    run_experiment()
