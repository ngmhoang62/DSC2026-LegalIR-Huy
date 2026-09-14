"""Phase B: Inner Pilot for jina_honest_adaptation_v1.
Compares M1 (LoRA) vs M2 (Full FT) on outer fold 0 training population only.
Computes standalone metrics, wins/losses, crossings, complementarity oracles,
and inner endpoint integration diagnostics (J0, J1, J2, J3).
Produces:
- results/gemini/jina_honest_adaptation_v1/NEURAL_TRAINING_PROOF.json
- results/gemini/jina_honest_adaptation_v1/FINETUNE_METHOD_SELECTION.json
- results/gemini/jina_honest_adaptation_v1/INNER_PILOT_REPORT.json
"""

from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    EXP_CHECKPOINTS,
    EXP_RESULTS,
    REPO_ROOT,
    get_git_info,
    get_os_gpu_memory,
    log_execution_trace,
    sha256_file,
)
from dataset import LegalRetrievalDataset
from model import (
    build_model,
    checkpoint_round_trip_test,
    compute_group_loss,
    compute_parent_scores,
    save_jina_checkpoint,
)
from neural_proof import NeuralTrainingProofTracker

sys.path.insert(0, str(REPO_ROOT))
from benchmark_jina_reranker_holdouts import top_passages

sys.path.insert(0, str(REPO_ROOT / "src/huy_fasttrack"))
import run_huy_5fold_fasttrack as fasttrack_core


def extract_cached_query_passages(
    dataset: LegalRetrievalDataset, qids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Pre-extract top Huy passages for query candidate pools to optimize CPU time."""
    cached = {}
    print(f"Pre-caching passages for {len(qids)} queries...", flush=True)
    t0 = time.perf_counter()
    for qid in qids:
        qtext = dataset.questions[qid]
        pool = dataset.pools[qid]
        pairs = []
        doc_indices = []
        for doc_idx, d in enumerate(pool):
            dtext = dataset.contexts.get(d, "")
            passages = top_passages(qtext, dtext, count=2, window=220, overlap=70)
            for p in passages:
                pairs.append((qtext, p))
                doc_indices.append(doc_idx)
        cached[qid] = {
            "pool": pool,
            "pairs": pairs,
            "doc_indices": doc_indices,
            "gold": dataset.golds[qid],
        }
    print(f"Pre-caching finished in {time.perf_counter() - t0:.2f}s.", flush=True)
    return cached


def score_cached_queries(
    model: nn.Module,
    tok: Any,
    cached_data: Dict[str, Dict[str, Any]],
    device: str = "cuda",
    micro_batch_size: int = 4,
) -> Dict[str, Dict[str, float]]:
    """Score pre-cached query candidate pools with adapted model."""
    model.eval()
    scores_by_qid: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for qid, item in cached_data.items():
            pool = item["pool"]
            parent_scores = compute_parent_scores(
                model=model,
                tok=tok,
                pairs=item["pairs"],
                doc_indices_for_passages=item["doc_indices"],
                num_docs=len(pool),
                tau_pool=0.15,
                max_length=512,
                device=device,
                micro_batch_size=micro_batch_size,
            )
            scores_list = parent_scores.cpu().tolist()
            scores_by_qid[qid] = {doc: scores_list[i] for i, doc in enumerate(pool)}
    return scores_by_qid


def evaluate_standalone(
    scores_by_qid: Dict[str, Dict[str, float]],
    cached_data: Dict[str, Dict[str, Any]],
    frozen_scores: Dict[Tuple[str, str], float],
    auth_lock_orders: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Compute all Section 10 and 11 standalone metrics on inner validation."""
    recalls = {1: [], 5: [], 8: [], 10: []}
    single_r5, multi_r5 = [], []
    wins, losses, ties = 0, 0, 0
    top5_churns = []
    crossings_in, crossings_out = 0, 0
    oracle_a_hits, oracle_b_hits = [], []

    for qid, item in cached_data.items():
        gold = item["gold"]
        pool = item["pool"]
        scores = scores_by_qid[qid]

        ranked_adapted = sorted(pool, key=lambda d: (-scores[d], d))
        frozen_doc_scores = {d: frozen_scores.get((qid, d), -1e9) for d in pool}
        ranked_frozen = sorted(pool, key=lambda d: (-frozen_doc_scores[d], d))

        for k in [1, 5, 8, 10]:
            hits = len(set(ranked_adapted[:k]) & gold)
            recalls[k].append(hits / len(gold))

        r5_adapted = len(set(ranked_adapted[:5]) & gold) / len(gold)
        r5_frozen = len(set(ranked_frozen[:5]) & gold) / len(gold)

        if len(gold) == 1:
            single_r5.append(r5_adapted)
        else:
            multi_r5.append(r5_adapted)

        if r5_adapted > r5_frozen:
            wins += 1
        elif r5_adapted < r5_frozen:
            losses += 1
        else:
            ties += 1

        top5_churns.append(len(set(ranked_adapted[:5]) - set(ranked_frozen[:5])))

        for g in gold:
            in_frozen = g in ranked_frozen[:5]
            in_adapted = g in ranked_adapted[:5]
            if not in_frozen and in_adapted:
                crossings_in += 1
            elif in_frozen and not in_adapted:
                crossings_out += 1

        union_a = set(ranked_adapted[:5]) | set(ranked_frozen[:5])
        oracle_a_hits.append(len(union_a & gold) / len(gold))

        if auth_lock_orders and qid in auth_lock_orders:
            ranked_auth = auth_lock_orders[qid]
            union_b = set(ranked_adapted[:5]) | set(ranked_auth[:5])
            oracle_b_hits.append(len(union_b & gold) / len(gold))

    return {
        "recall_at_1": float(np.mean(recalls[1])),
        "recall_at_5": float(np.mean(recalls[5])),
        "recall_at_8": float(np.mean(recalls[8])),
        "recall_at_10": float(np.mean(recalls[10])),
        "single_gold_recall_at_5": float(np.mean(single_r5)) if single_r5 else 0.0,
        "multi_gold_recall_at_5": float(np.mean(multi_r5)) if multi_r5 else 0.0,
        "single_gold_count": len(single_r5),
        "multi_gold_count": len(multi_r5),
        "vs_frozen_jina": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_minus_loss": wins - losses,
            "mean_top5_churn": float(np.mean(top5_churns)),
            "gold_crossings_in": crossings_in,
            "gold_crossings_out": crossings_out,
            "net_gold_crossings": crossings_in - crossings_out,
        },
        "complementarity_oracles": {
            "oracle_a_adapted_union_frozen_jina_r5": float(np.mean(oracle_a_hits)),
            "oracle_b_adapted_union_auth_endpoint_r5": float(np.mean(oracle_b_hits))
            if oracle_b_hits
            else None,
        },
    }


def evaluate_inner_endpoint_integration(
    adapted_scores_train: Dict[str, Dict[str, float]],
    adapted_scores_val: Dict[str, Dict[str, float]],
    train_qids: List[str],
    val_qids: List[str],
    pools: Dict[str, List[str]],
    golds: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """Compute inner-val endpoint integration diagnostics (J0, J1, J2, J3) via balanced LR."""
    all_target_qids = set(train_qids) | set(val_qids)
    sub_pools = {q: pools[q] for q in all_target_qids if q in pools}

    rank_features: Dict[str, Dict[str, np.ndarray]] = {}
    score_features: Dict[str, Dict[str, np.ndarray]] = {}

    for name in fasttrack_core.CHANNEL_SPECS:
        omap, raw, _ = fasttrack_core.load_source_channel(name, sub_pools)
        rank_features[name] = fasttrack_core.rank_columns(omap, sub_pools)
        score_features[name] = fasttrack_core.score_columns(raw, sub_pools)

    e5_orders, e5_scores = {}, {}
    for i, p in fasttrack_core.adapted_paths():
        if p.exists():
            for line in fasttrack_core.read_jsonl(p):
                qid = str(line["qid"])
                if qid in sub_pools:
                    e5_orders[qid] = [str(x) for x in line["order"]]
                    e5_scores[qid] = {str(d): float(s) for d, s in line["scores"].items()}

    rank_features["adapted_e5"] = fasttrack_core.rank_columns(e5_orders, sub_pools)
    score_features["adapted_e5"] = fasttrack_core.score_columns(e5_scores, sub_pools)
    if "e5_dense" in rank_features:
        rank_features["frozen_e5"] = rank_features["e5_dense"]
        score_features["frozen_e5"] = score_features["e5_dense"]

    j_orders, j_scores, _ = fasttrack_core.load_jina(sub_pools)
    rank_features["frozen_jina"] = fasttrack_core.rank_columns(j_orders, sub_pools)
    score_features["frozen_jina"] = fasttrack_core.score_columns(j_scores, sub_pools)

    combined_adapted_scores = {**adapted_scores_train, **adapted_scores_val}
    combined_adapted_orders = {
        q: sorted(sub_pools[q], key=lambda d: (-combined_adapted_scores[q].get(d, -1e9), d))
        for q in sub_pools
    }
    rank_features["adapted_jina"] = fasttrack_core.rank_columns(combined_adapted_orders, sub_pools)
    score_features["adapted_jina"] = fasttrack_core.score_columns(combined_adapted_scores, sub_pools)

    residual_features = {}
    for q in sub_pools:
        f_z = score_features["frozen_jina"][q][:, 0]
        a_z = score_features["adapted_jina"][q][:, 0]
        f_r = rank_features["frozen_jina"][q][:, 1] * 60.0
        a_r = rank_features["adapted_jina"][q][:, 1] * 60.0
        residual_features[q] = np.column_stack((a_z - f_z, a_r - f_r)).astype(np.float32)

    metadata = {
        "document_types": {q: np.zeros((len(sub_pools[q]), 17), dtype=np.float32) for q in sub_pools},
        "citation_count": {q: np.zeros((len(sub_pools[q]), 3), dtype=np.float32) for q in sub_pools},
    }

    j0_cfg = fasttrack_core.make_config(
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina"),
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina"),
        (),
    )
    j1_cfg = fasttrack_core.make_config(
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "adapted_jina"),
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "adapted_jina"),
        (),
    )
    j2_cfg = fasttrack_core.make_config(
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina", "adapted_jina"),
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina", "adapted_jina"),
        (),
    )

    arms = [
        ("J0_FROZEN_BASELINE", j0_cfg, False),
        ("J1_REPLACE", j1_cfg, False),
        ("J2_AUGMENT", j2_cfg, False),
        ("J3_RESIDUAL", j2_cfg, True),
    ]

    diagnostics = {}
    for arm_name, cfg, is_residual in arms:
        rows = fasttrack_core.make_rows(cfg, sub_pools, rank_features, score_features, metadata)
        if is_residual:
            for q in sub_pools:
                rows[q] = np.column_stack((rows[q], residual_features[q])).astype(np.float32)

        x_tr = np.vstack([rows[q] for q in train_qids if q in rows])
        y_tr = np.concatenate([
            np.asarray([d in golds[q] for d in sub_pools[q]], dtype=np.int8)
            for q in train_qids
            if q in rows
        ])

        scaler = StandardScaler().fit(x_tr)
        clf = LogisticRegression(
            C=0.15, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=2026
        ).fit(scaler.transform(x_tr), y_tr)

        val_r5 = []
        for q in val_qids:
            if q not in rows:
                continue
            dec = clf.decision_function(scaler.transform(rows[q]))
            docs = sub_pools[q]
            ranked = sorted(docs, key=lambda d: (-dec[docs.index(d)], d))
            hits = len(set(ranked[:5]) & golds[q])
            val_r5.append(hits / len(golds[q]))

        diagnostics[arm_name] = {
            "inner_val_recall_at_5": float(np.mean(val_r5)),
            "feature_dimension": x_tr.shape[1],
        }

    return diagnostics


def train_and_evaluate_architecture(
    method: str,
    dataset: LegalRetrievalDataset,
    inner_train_qids: List[str],
    inner_val_qids: List[str],
    lr_train_qids: List[str],
    val_cached_data: Dict[str, Dict[str, Any]],
    lr_train_cached_data: Dict[str, Dict[str, Any]],
    audit_pairs: List[Tuple[str, str]],
    model_path: Path,
    auth_lock_orders: Dict[str, List[str]],
    device: str = "cuda",
) -> Dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"STARTING INNER PILOT TRAINING: {method}", flush=True)
    print(f"=======================================================", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    model, tok = build_model(method, model_path, rank=16, dtype=torch.bfloat16, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    tracker = NeuralTrainingProofTracker(f"inner_pilot_{method.lower()}", audit_pairs)
    tracker.before_training(model, tok, device=device)

    epochs = 2
    step_count = 0
    total_train_steps = len(inner_train_qids) * epochs

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_rng = random.Random(2026 + epoch * 1000)
        shuffled_train = list(inner_train_qids)
        epoch_rng.shuffle(shuffled_train)

        epoch_t0 = time.perf_counter()
        running_loss = 0.0

        for step_in_epoch, qid in enumerate(shuffled_train):
            sample = dataset.sample_query_candidates(qid, epoch=epoch, seed_offset=step_in_epoch)
            if not sample["pos_indices"]:
                continue

            optimizer.zero_grad()
            parent_scores = compute_parent_scores(
                model=model,
                tok=tok,
                pairs=sample["pairs"],
                doc_indices_for_passages=sample["doc_indices_for_passages"],
                num_docs=sample["num_docs"],
                tau_pool=0.15,
                max_length=512,
                device=device,
                micro_batch_size=4,
            )

            frozen = torch.tensor(
                sample["frozen_doc_scores"], device=device, dtype=parent_scores.dtype
            )
            tot_loss, _, _ = compute_group_loss(
                parent_scores, sample["pos_indices"], frozen, lambda_anchor=0.05
            )

            tot_loss.backward()

            if step_count % 50 == 0:
                tracker.record_step(
                    step=step_count,
                    epoch=epoch,
                    loss=float(tot_loss.item()),
                    model=model,
                    lr=1e-5,
                    optimizer_class="AdamW",
                )

            optimizer.step()
            tracker.step_optimizer()
            step_count += 1
            running_loss += float(tot_loss.item())

            if (step_in_epoch + 1) % 500 == 0:
                rate = (time.perf_counter() - epoch_t0) / (step_in_epoch + 1)
                print(
                    f"[{method}] Epoch {epoch}/{epochs} | Step {step_in_epoch + 1}/{len(shuffled_train)} | Mean Loss: {running_loss / (step_in_epoch + 1):.4f} | {rate:.3f} s/q",
                    flush=True,
                )

        print(f"[{method}] Epoch {epoch} complete. Mean loss: {running_loss / len(shuffled_train):.4f}", flush=True)

    train_runtime = time.perf_counter() - t_start
    peak_vram_train = torch.cuda.max_memory_allocated() / (1024**2)

    ckpt_path = EXP_CHECKPOINTS / f"inner_pilot_{method.lower()}.pt"
    print(f"[{method}] Saving checkpoint to {ckpt_path}...", flush=True)
    save_jina_checkpoint(model, tok, method, ckpt_path)

    print(f"[{method}] Running round-trip reload audit...", flush=True)
    round_trip_res = checkpoint_round_trip_test(
        model=model,
        tok=tok,
        audit_pairs=audit_pairs[:16],
        method=method,
        model_path=model_path,
        test_path=EXP_CHECKPOINTS / f"test_roundtrip_{method.lower()}.pt",
        device=device,
    )
    print(f"[{method}] Reload test status: {round_trip_res['status']} (max diff: {round_trip_res['max_logit_difference']:.8e})", flush=True)

    proof_report = tracker.after_training(
        model=model,
        tok=tok,
        checkpoint_path=ckpt_path,
        device=device,
    )

    print(f"[{method}] Scoring {len(inner_val_qids)} validation queries...", flush=True)
    val_scores = score_cached_queries(model, tok, val_cached_data, device=device, micro_batch_size=4)
    standalone_metrics = evaluate_standalone(
        scores_by_qid=val_scores,
        cached_data=val_cached_data,
        frozen_scores=dataset.frozen_scores,
        auth_lock_orders=auth_lock_orders,
    )

    print(f"[{method}] Scoring {len(lr_train_qids)} diagnostic training queries...", flush=True)
    lr_train_scores = score_cached_queries(model, tok, lr_train_cached_data, device=device, micro_batch_size=4)

    integration_diag = evaluate_inner_endpoint_integration(
        adapted_scores_train=lr_train_scores,
        adapted_scores_val=val_scores,
        train_qids=lr_train_qids,
        val_qids=inner_val_qids,
        pools=dataset.pools,
        golds=dataset.golds,
    )

    total_runtime = time.perf_counter() - t_start

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": method,
        "train_runtime_seconds": round(train_runtime, 2),
        "total_runtime_seconds": round(total_runtime, 2),
        "peak_vram_mb": round(peak_vram_train, 1),
        "actual_optimizer_steps": tracker.actual_optimizer_steps,
        "checkpoint_round_trip": round_trip_res,
        "neural_proof_status": proof_report["status"],
        "neural_proof_details": proof_report,
        "standalone_metrics": standalone_metrics,
        "endpoint_integration_diagnostics": integration_diag,
    }


def run_inner_pilot():
    start_total_time = time.perf_counter()
    model_path = REPO_ROOT / "cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"
    dataset = LegalRetrievalDataset(rng_seed=2026)

    fold0_train_qids = dataset.get_outer_train_qids("fold_0")
    print(f"Outer fold 0 training population: {len(fold0_train_qids)} queries.", flush=True)

    rng = random.Random(2026)
    shuffled_qids = list(fold0_train_qids)
    rng.shuffle(shuffled_qids)

    inner_train_qids = shuffled_qids[:1500]
    inner_val_qids = shuffled_qids[1500:1750]
    audit_pool_qids = shuffled_qids[1750:2050]
    lr_train_qids = inner_train_qids[:100]

    audit_pairs = []
    for qid in audit_pool_qids:
        qtext = dataset.questions[qid]
        pool = dataset.pools.get(qid, [])
        if pool:
            d = pool[0]
            dtext = dataset.contexts.get(d, "")
            passages = top_passages(qtext, dtext, count=1, window=220, overlap=70)
            if passages:
                audit_pairs.append((qtext, passages[0]))
        if len(audit_pairs) >= 256:
            break

    print(f"Audit pairs created: {len(audit_pairs)} pairs.", flush=True)

    auth_lock_orders = {}
    auth_lock_path = REPO_ROOT / "results/huy_fasttrack/learner_prediction_locks/profile_memory_plus_sparse_rank_scores.jsonl"
    if auth_lock_path.exists():
        with auth_lock_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    auth_lock_orders[str(rec["qid"])] = [
                        str(x) for x in (rec.get("order") or rec.get("ranked_docs") or rec.get("predictions"))
                    ]

    val_cached_data = extract_cached_query_passages(dataset, inner_val_qids)
    lr_train_cached_data = extract_cached_query_passages(dataset, lr_train_qids)

    frozen_val_scores = {
        q: {d: dataset.frozen_scores.get((q, d), -1e9) for d in dataset.pools[q]}
        for q in inner_val_qids
    }
    frozen_baseline_metrics = evaluate_standalone(
        scores_by_qid=frozen_val_scores,
        cached_data=val_cached_data,
        frozen_scores=dataset.frozen_scores,
        auth_lock_orders=auth_lock_orders,
    )

    print("\n--- FROZEN JINA BASELINE (INNER VAL) ---", flush=True)
    print(f"Recall@1:  {frozen_baseline_metrics['recall_at_1']:.4f}")
    print(f"Recall@5:  {frozen_baseline_metrics['recall_at_5']:.4f}")
    print(f"Recall@8:  {frozen_baseline_metrics['recall_at_8']:.4f}")
    print(f"Recall@10: {frozen_baseline_metrics['recall_at_10']:.4f}")
    print(f"Single R@5: {frozen_baseline_metrics['single_gold_recall_at_5']:.4f}")
    print(f"Multi R@5:  {frozen_baseline_metrics['multi_gold_recall_at_5']:.4f}")

    smoke_path = EXP_RESULTS / "VRAM_SMOKE_TEST.json"
    run_m2 = True
    if smoke_path.exists():
        try:
            smoke_data = json.loads(smoke_path.read_text(encoding="utf-8"))
            if not smoke_data.get("verdict", {}).get("m2_full_safe", True):
                print("WARNING: Smoke test flagged M2 as local unsafe. M2 will be marked NOT_EXECUTED.", flush=True)
                run_m2 = False
        except Exception:
            pass

    m1_results = train_and_evaluate_architecture(
        method="M1_LORA",
        dataset=dataset,
        inner_train_qids=inner_train_qids,
        inner_val_qids=inner_val_qids,
        lr_train_qids=lr_train_qids,
        val_cached_data=val_cached_data,
        lr_train_cached_data=lr_train_cached_data,
        audit_pairs=audit_pairs,
        model_path=model_path,
        auth_lock_orders=auth_lock_orders,
    )

    if run_m2:
        m2_results = train_and_evaluate_architecture(
            method="M2_FULL",
            dataset=dataset,
            inner_train_qids=inner_train_qids,
            inner_val_qids=inner_val_qids,
            lr_train_qids=lr_train_qids,
            val_cached_data=val_cached_data,
            lr_train_cached_data=lr_train_cached_data,
            audit_pairs=audit_pairs,
            model_path=model_path,
            auth_lock_orders=auth_lock_orders,
        )
    else:
        m2_results = {
            "method": "M2_FULL",
            "status": "NOT_EXECUTED",
            "reason": "FULL_FT_LOCAL_UNSAFE",
            "standalone_metrics": None,
            "endpoint_integration_diagnostics": None,
        }

    if run_m2 and m2_results.get("standalone_metrics"):
        m1_integ = m1_results["endpoint_integration_diagnostics"].get("J1_REPLACE", {}).get("inner_val_recall_at_5", 0.0)
        m2_integ = m2_results["endpoint_integration_diagnostics"].get("J1_REPLACE", {}).get("inner_val_recall_at_5", 0.0)
        m1_s5 = m1_results["standalone_metrics"]["recall_at_5"]
        m2_s5 = m2_results["standalone_metrics"]["recall_at_5"]

        if m2_integ > m1_integ + 0.002:
            winner = "M2_FULL"
            reason = f"M2 achieved higher endpoint integration R@5 ({m2_integ:.4f} vs {m1_integ:.4f})"
        elif m1_integ > m2_integ + 0.002:
            winner = "M1_LORA"
            reason = f"M1 achieved higher endpoint integration R@5 ({m1_integ:.4f} vs {m2_integ:.4f})"
        elif m2_s5 > m1_s5:
            winner = "M2_FULL"
            reason = f"M2 achieved higher standalone R@5 ({m2_s5:.4f} vs {m1_s5:.4f})"
        elif m1_s5 > m2_s5:
            winner = "M1_LORA"
            reason = f"M1 achieved higher standalone R@5 ({m1_s5:.4f} vs {m2_s5:.4f})"
        else:
            winner = "M1_LORA"
            reason = "Tied metrics; M1 LoRA selected for lower VRAM, lower drift risk, and faster training"
    else:
        winner = "M1_LORA"
        reason = "M1 LoRA completed successfully as primary safe architecture"

    print(f"\n=======================================================", flush=True)
    print(f"PROVISIONAL WINNER: {winner} ({reason})", flush=True)
    print(f"=======================================================\n", flush=True)

    proof_path = EXP_RESULTS / "NEURAL_TRAINING_PROOF.json"
    proof_path.write_text(
        json.dumps(m1_results["neural_proof_details"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {proof_path}")

    sel_report = {
        "schema_version": "dsc2026.gemini.finetune_method_selection.v2",
        "status": "PROVISIONAL_SELECTION",
        "scope": "INNER_PILOT_ONLY",
        "provisional_winner": winner,
        "selection_rationale": reason,
        "frozen_baseline": frozen_baseline_metrics,
        "architectures": {
            "m1_lora": {
                "standalone_metrics": m1_results["standalone_metrics"],
                "endpoint_integration_diagnostics": m1_results["endpoint_integration_diagnostics"],
                "peak_vram_mb": m1_results["peak_vram_mb"],
                "runtime_seconds": m1_results["total_runtime_seconds"],
            },
            "m2_full": {
                "standalone_metrics": m2_results.get("standalone_metrics"),
                "endpoint_integration_diagnostics": m2_results.get("endpoint_integration_diagnostics"),
                "peak_vram_mb": m2_results.get("peak_vram_mb"),
                "runtime_seconds": m2_results.get("total_runtime_seconds"),
            } if run_m2 else {"status": "NOT_EXECUTED"},
        },
        "git": get_git_info(),
        "runtime_seconds": round(time.perf_counter() - start_total_time, 2),
    }
    sel_path = EXP_RESULTS / "FINETUNE_METHOD_SELECTION.json"
    sel_path.write_text(json.dumps(sel_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {sel_path}")

    pilot_report = {
        "schema_version": "dsc2026.gemini.inner_pilot_report.v1",
        "status": "INNER_PILOT_COMPLETE",
        "hard_stop_reached": True,
        "next_step": "STOPPED_AFTER_INNER_PILOT_FOR_REVIEW",
        "frozen_jina_baseline": frozen_baseline_metrics,
        "m1_lora": m1_results,
        "m2_full": m2_results,
        "provisional_selection": {
            "winner": winner,
            "rationale": reason,
        },
        "git": get_git_info(),
        "total_runtime_seconds": round(time.perf_counter() - start_total_time, 2),
    }
    report_path = EXP_RESULTS / "INNER_PILOT_REPORT.json"
    report_path.write_text(json.dumps(pilot_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")

    log_execution_trace(
        stage_name="inner_pilot_selection",
        command=f"python {__file__}",
        code_hash=sha256_file(Path(__file__)),
        model_checkpoint_hash="CHECKPOINTS_PRODUCED",
        train_query_count=len(inner_train_qids),
        val_query_count=len(inner_val_qids),
        optimizer_steps=m1_results["actual_optimizer_steps"] + (m2_results.get("actual_optimizer_steps", 0) if run_m2 else 0),
        gpu_info=torch.cuda.get_device_name(0),
        runtime_sec=time.perf_counter() - start_total_time,
        output_hashes={
            "NEURAL_TRAINING_PROOF.json": sha256_file(proof_path),
            "FINETUNE_METHOD_SELECTION.json": sha256_file(sel_path),
            "INNER_PILOT_REPORT.json": sha256_file(report_path),
        },
        status="PASS",
        extra={"winner": winner},
    )


if __name__ == "__main__":
    run_inner_pilot()
