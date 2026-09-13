"""Fail-closed post-metric closure for the sealed ViMRC Fold-0 experiment."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def recall(docs: set[str], gold: set[str]) -> float:
    return len(docs & gold) / len(gold)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    output = root / "results/research_v2_open_rl"
    paths = {
        "folds": root / "results/research_v2_forensic/V2_FOLDS.json",
        "pool": root / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl",
        "train": workspace / "LegalIR/public_test_dataset/train.json",
        "exclusions": output / "VIMRC_INPUT_EXCLUSIONS.json",
        "anchor": root / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl",
        "e5_predictions": root / "results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl",
        "jina_predictions": root / "results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl",
        "sources_db": workspace / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite",
        "score_db": root / "cache/research_v2_open_rl/vimrc_answerability_fold0_scores.sqlite",
        "preregistration": output / "VIMRC_ANSWERABILITY_FOLD0_PREREGISTRATION.json",
        "parity": output / "VIMRC_ANSWERABILITY_PARITY.json",
        "batch_parity": output / "VIMRC_FP32_BATCH_PARITY.json",
        "sdpa_parity": output / "VIMRC_SDPA_PARITY.json",
        "runner": root / "src/research_v2_open_rl/score_vimrc_answerability_fold0.py",
        "report": output / "VIMRC_ANSWERABILITY_FOLD0_REPORT.json",
        "predictions": output / "VIMRC_ANSWERABILITY_FOLD0_PREDICTIONS.jsonl",
    }
    expected = {
        "folds": "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19",
        "pool": "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277",
        "train": "c39cde9e74977e350f1456e7d487aafe67d2bcbaa4fa26fcabd557fe635635b7",
        "exclusions": "d10aef3d891746cd9f874b32e9cb20940cb4fef0928c0fe239fdb7ca16337616",
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    failures = [f"{name}_sha256" for name, value in expected.items() if observed[name] != value]

    folds = json.loads(paths["folds"].read_text(encoding="utf-8"))
    fold0_all = set(map(str, folds["folds"]["fold_0"]))
    pool_rows = [row for row in read_jsonl(paths["pool"]) if str(row["qid"]) in fold0_all]
    pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in pool_rows}
    if len(pool) != 1398:
        failures.append("fold0_pool_count")

    predictions = {str(row["qid"]): row for row in read_jsonl(paths["predictions"])}
    anchor = {str(row["qid"]): row for row in read_jsonl(paths["anchor"]) if str(row["qid"]) in pool}
    e5 = {str(row["qid"]): row for row in read_jsonl(paths["e5_predictions"]) if str(row["qid"]) in pool}
    jina_v2 = {str(row["qid"]): list(map(str, row["top5"])) for row in read_jsonl(paths["jina_predictions"]) if str(row["qid"]) in pool}
    for name, mapping in (("predictions", predictions), ("anchor", anchor), ("e5", e5), ("jina_v2", jina_v2)):
        if set(mapping) != set(pool):
            failures.append(f"{name}_qid_set")

    source_db = sqlite3.connect(f"file:{paths['sources_db'].resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    native: dict[tuple[str, str], list[str]] = {}
    for qid, source, payload in source_db.execute(
        "SELECT q,source,payload FROM sources WHERE source IN ('e5','lal','jina')"
    ):
        qid = str(qid)
        if qid in pool:
            native[(qid, str(source))] = [
                str(row["doc_id"]) for row in json.loads(payload) if str(row["doc_id"]) in pool[qid]
            ][:5]
    source_db.close()

    component_failures: dict[str, int] = {name: 0 for name in ("adapted_e5", "frozen_e5", "lal", "jina_native", "jina_v2")}
    clean: dict[str, set[str]] = {}
    adapted_anchor_matches = 0
    frozen_native_matches = 0
    for qid in pool:
        components = {
            "adapted_e5": list(map(str, e5[qid]["ft_order"][:5])),
            "frozen_e5": list(map(str, e5[qid]["base_order"][:5])),
            "lal": native[(qid, "lal")],
            "jina_native": native[(qid, "jina")],
            "jina_v2": jina_v2[qid],
        }
        for name, docs in components.items():
            component_failures[name] += int(len(docs) != 5 or len(set(docs)) != 5 or not set(docs) <= pool[qid])
        clean[qid] = set().union(*(set(docs) for docs in components.values()))
        adapted_anchor_matches += components["adapted_e5"] == list(map(str, anchor[qid]["base_top5"]))
        frozen_native_matches += components["frozen_e5"] == native[(qid, "e5")]
    if any(component_failures.values()) or adapted_anchor_matches != 1398 or frozen_native_matches != 1398:
        failures.append("clean_expert_construction")

    score_db = sqlite3.connect(f"file:{paths['score_db'].resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    db_integrity = score_db.execute("PRAGMA integrity_check").fetchone()[0]
    progress = score_db.execute(
        "SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress"
    ).fetchone()
    score_stats = score_db.execute(
        "SELECT COUNT(*),COUNT(DISTINCT qid),COUNT(DISTINCT qid||char(0)||doc_id),SUM(score IS NULL) FROM scores"
    ).fetchone()
    db_qids = {str(row[0]) for row in score_db.execute("SELECT qid FROM progress")}
    score_pairs = {(str(q), str(d)) for q, d in score_db.execute("SELECT DISTINCT qid,doc_id FROM scores")}
    per_query_mismatches = score_db.execute(
        "SELECT COUNT(*) FROM (SELECT p.qid,p.sequences,COUNT(s.doc_id) n FROM progress p "
        "LEFT JOIN scores s ON s.qid=p.qid GROUP BY p.qid HAVING p.sequences!=n)"
    ).fetchone()[0]
    score_db.close()
    expected_pairs = {(qid, doc) for qid, docs in pool.items() for doc in docs}
    db_ok = (
        db_integrity == "ok"
        and progress[0] == progress[1] == 1398
        and progress[2] == score_stats[0] == 146249
        and score_stats[1] == 1398
        and score_stats[2] == len(expected_pairs) == 73128
        and score_stats[3] == 0
        and db_qids == set(pool)
        and score_pairs == expected_pairs
        and per_query_mismatches == 0
    )
    if not db_ok:
        failures.append("score_database_completeness")

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    qa_scores: list[float] = []
    anchor_scores: list[float] = []
    candidate_scores: list[float] = []
    clean_scores: list[float] = []
    clean_plus_scores: list[float] = []
    crossings_in = crossings_out = 0
    qid_top5_rows: list[str] = []
    for qid in sorted(pool, key=int):
        row = predictions[qid]
        ranking = list(map(str, row["ranking"]))
        top5 = list(map(str, row["top5"]))
        values = list(map(float, row["scores"]))
        base = set(map(str, anchor[qid]["fused_top5"]))
        gold = set(map(str, anchor[qid]["gold"]))
        if top5 != ranking[:5] or set(ranking) != pool[qid] or len(ranking) != len(pool[qid]) or len(values) != len(ranking):
            failures.append(f"prediction_contract_{qid}")
        if any((values[i], ranking[i]) < (values[i + 1], ranking[i + 1]) for i in range(len(values) - 1)):
            # Scores must descend. Tied documents are ordered by ascending string id.
            for i in range(len(values) - 1):
                if values[i] < values[i + 1] or (values[i] == values[i + 1] and ranking[i] > ranking[i + 1]):
                    failures.append(f"prediction_sort_{qid}")
                    break
        pred = set(top5)
        qa_scores.append(recall(pred, gold))
        anchor_scores.append(recall(base, gold))
        candidate_scores.append(recall(pool[qid], gold))
        clean_scores.append(recall(clean[qid], gold))
        clean_plus_scores.append(recall(clean[qid] | pred, gold))
        crossings_in += len((pred - base) & gold)
        crossings_out += len((base - pred) & gold)
        qid_top5_rows.append(canonical_json({"qid": qid, "top5": top5}))

    recomputed = {
        "qa_recall_at_5": float(np.mean(qa_scores)),
        "anchor_recall_at_5": float(np.mean(anchor_scores)),
        "candidate_ceiling": float(np.mean(candidate_scores)),
        "clean_expert_union": float(np.mean(clean_scores)),
        "clean_plus_qa_union": float(np.mean(clean_plus_scores)),
        "clean_union_delta": float(np.mean(clean_plus_scores) - np.mean(clean_scores)),
        "gold_crossings_into_top5": crossings_in,
        "gold_crossings_out_of_top5": crossings_out,
    }
    metric_checks = {
        "qa_recall": abs(recomputed["qa_recall_at_5"] - report["metrics"]["recall_at_5"]) <= 1e-12,
        "anchor_recall": abs(recomputed["anchor_recall_at_5"] - report["metrics"]["current_anchor_recall_at_5"]) <= 1e-12,
        "clean_union": abs(recomputed["clean_expert_union"] - report["metrics"]["existing_clean_experts_union"]) <= 1e-12,
        "clean_plus": abs(recomputed["clean_plus_qa_union"] - report["metrics"]["clean_experts_plus_qa_union"]) <= 1e-12,
    }
    if not all(metric_checks.values()):
        failures.append("metric_recompute")

    gate = {
        "pass_standalone": report["metrics"]["recall_at_5"] >= 0.90 and report["metrics"]["clean_union_delta"] >= 0.006,
        "pass_orthogonal": report["metrics"]["recall_at_5"] >= 0.84 and report["metrics"]["clean_union_delta"] >= 0.005,
        "kill_standalone_lt_0_78": report["metrics"]["recall_at_5"] < 0.78,
        "kill_clean_union_delta_lt_0_003": report["metrics"]["clean_union_delta"] < 0.003,
    }
    sealed_verdict = "KILL" if gate["kill_standalone_lt_0_78"] or gate["kill_clean_union_delta_lt_0_003"] else "OTHER"
    if report["verdict"] != sealed_verdict or report["verdict"] != "KILL":
        failures.append("sealed_gate_verdict")

    model_dir = root / "cache/research_v2_open_rl/models/vi-mrc-large"
    model_hashes = {path.name: sha256(path) for path in sorted(model_dir.iterdir()) if path.is_file()}
    parity = json.loads(paths["parity"].read_text(encoding="utf-8"))
    model_matches_parity = model_hashes == parity["model_files"]
    if not model_matches_parity:
        failures.append("model_hashes")

    input_manifest = {
        "schema_version": "dsc2026.research_v2.vimrc_input_manifest.v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "population": {"fold": "fold_0", "queries": len(pool)},
        "files": {name: {"path": str(paths[name]), "sha256": observed[name]} for name in (
            "folds", "pool", "train", "exclusions", "anchor", "e5_predictions", "jina_predictions",
            "sources_db", "score_db", "preregistration", "parity", "batch_parity", "sdpa_parity", "runner"
        )},
        "expected_hash_checks": {name: observed[name] == value for name, value in expected.items()},
        "model": {
            "path": str(model_dir),
            "files": model_hashes,
            "matches_premetric_parity_manifest": model_matches_parity,
            "license": "CC-BY-NC-4.0",
            "research_only": True,
        },
        "clean_expert_union": {
            "components": ["adapted_e5", "frozen_e5", "frozen_lal", "native_jina", "jina_v2_lexical"],
            "component_contract_failures": component_failures,
            "adapted_e5_matches_anchor_all_queries": adapted_anchor_matches == 1398,
            "frozen_e5_matches_native_source_all_queries": frozen_native_matches == 1398,
            "e5_byte_hash_reconciliation": "runner-parity artifact differs from the earlier seal byte hash, but its adapted Top-5 and frozen Top-5 exactly match the downstream anchor and native E5 source on all 1,398 queries",
        },
        "runtime_patches": {
            "classification": "NUMERICAL_COMPATIBILITY_ONLY",
            "authorized": "eager FP32 batch 2",
            "scientific_contract_changed": False,
            "failed_variants": ["FP16 raw-score parity", "FP32 batch 4 numeric gate", "FP32 SDPA numeric gate"],
        },
        "failures": sorted(set(failures)),
    }
    input_path = output / "VIMRC_ANSWERABILITY_FOLD0_INPUT_MANIFEST.json"
    write_json(input_path, input_manifest)

    top5_hash = hashlib.sha256(("\n".join(qid_top5_rows) + "\n").encode("utf-8")).hexdigest()
    qid_hash = hashlib.sha256(("\n".join(sorted(pool, key=int)) + "\n").encode("utf-8")).hexdigest()
    lock = {
        "schema_version": "dsc2026.research_v2.vimrc_prediction_lock.v1",
        "status": "LOCKED_KILLED_INTERFACE",
        "experiment": "VIMRC_ANSWERABILITY_FOLD0",
        "queries": len(predictions),
        "predictions_path": str(paths["predictions"]),
        "predictions_sha256": observed["predictions"],
        "qid_order_sha256": qid_hash,
        "canonical_qid_top5_sha256": top5_hash,
        "verdict": "KILL",
        "anti_rescue": "No QA model, prompt, span length, evidence, aggregation, fusion, threshold, or inference grid.",
    }
    lock_path = output / "VIMRC_ANSWERABILITY_FOLD0_PREDICTION_LOCK.json"
    write_json(lock_path, lock)

    forensic = {
        "schema_version": "dsc2026.research_v2.vimrc_causal_forensic.v1",
        "status": "COMPLETE_KILL_FAMILY_CLOSED",
        "sealed_gate": gate,
        "verdict": "KILL",
        "metrics": report["metrics"],
        "recomputed": recomputed,
        "metric_checks": metric_checks,
        "headroom": {
            "candidate_ceiling": recomputed["candidate_ceiling"],
            "candidate_minus_qa_top5": recomputed["candidate_ceiling"] - recomputed["qa_recall_at_5"],
            "qa_depth50_minus_qa_top5": report["metrics"]["recall_depth"]["50"] - report["metrics"]["recall_at_5"],
            "clean_union_minus_anchor": recomputed["clean_expert_union"] - recomputed["anchor_recall_at_5"],
            "clean_plus_qa_increment": recomputed["clean_union_delta"],
        },
        "causal_read": [
            "The fixed QA reader retains most candidate gold by depth 50 but cannot order the document boundary: its top-5 recall is 0.460956 while its depth-50 recall is 0.968884.",
            "The answer-span versus no-answer margin is therefore not a calibrated proxy for parent relevance under this fixed full legal-document interface.",
            "Only +0.003100 recall lies beyond the complete clean-expert union, while the standalone safety condition fails by a wide margin.",
        ],
        "closed_scope": "This exact frozen Vietnamese extractive-QA answerability interface; no QA model or inference grid is authorized.",
    }
    forensic_path = output / "VIMRC_ANSWERABILITY_FOLD0_CAUSAL_FORENSIC.json"
    write_json(forensic_path, forensic)

    closure = {
        "schema_version": "dsc2026.research_v2.vimrc_closure_audit.v1",
        "status": "PASS_CLOSED" if not failures else "FAIL_CLOSED",
        "failures": sorted(set(failures)),
        "score_database": {
            "integrity_check": db_integrity,
            "progress_rows": progress[0],
            "progress_distinct_qids": progress[1],
            "sequences": progress[2],
            "score_rows": score_stats[0],
            "score_distinct_qids": score_stats[1],
            "candidate_pairs": score_stats[2],
            "null_scores": score_stats[3],
            "expected_candidate_pairs": len(expected_pairs),
            "per_query_sequence_mismatches": per_query_mismatches,
        },
        "output_contract": {
            "queries": len(predictions),
            "prediction_contract_failures": sum(item.startswith("prediction_") for item in failures),
            "report_verdict": report["verdict"],
            "sealed_verdict": sealed_verdict,
        },
        "recomputed": recomputed,
        "artifacts": {
            "input_manifest": str(input_path),
            "prediction_lock": str(lock_path),
            "causal_forensic": str(forensic_path),
        },
    }
    closure_path = output / "VIMRC_ANSWERABILITY_FOLD0_CLOSURE_AUDIT.json"
    write_json(closure_path, closure)

    output_manifest = {
        "schema_version": "dsc2026.research_v2.vimrc_output_manifest.v1",
        "status": closure["status"],
        "verdict": "KILL",
        "files": {
            path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (paths["report"], paths["predictions"], input_path, lock_path, forensic_path, closure_path)
        },
        "reproduce": (
            'D:\\Study\\DSC2026\\dsc_env\\Scripts\\python.exe '
            'D:\\Study\\DSC2026\\sota\\src\\research_v2_open_rl\\close_vimrc_answerability_fold0.py'
        ),
    }
    output_path = output / "VIMRC_ANSWERABILITY_FOLD0_OUTPUT_MANIFEST.json"
    write_json(output_path, output_manifest)
    print(json.dumps({"status": closure["status"], "failures": sorted(set(failures)), "recomputed": recomputed}, indent=2))


if __name__ == "__main__":
    main()
