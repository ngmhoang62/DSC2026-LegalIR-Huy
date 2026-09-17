"""Canonical 5-fold OOF admission model training and label-free action sealing.

Module for HUY_D1_AITEAM50_SOFT_ADMISSION_V1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import (
    CAL600_5FOLD_PATH,
    EXPECTED_5FOLD_SHA256,
    RESULTS_DIR,
    ROOT,
    sha256_file,
)


def train_oof_admission_model_and_seal_actions(
    all_ids: List[str],
    s_universe: Dict[str, List[str]],
    defenders: Dict[str, str],
    novel_map: Dict[str, List[str]],
    d1_top5: Dict[str, List[str]],
    features_per_query: Dict[str, np.ndarray],
    train_gold_labels: Dict[str, Set[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, Any], Dict[str, Any]]:
    """Trains 5-fold OOF admission models (using outer train gold only),
    scores held queries, executes held-query admission rule, and seals
    AITEAM50_SOFT_ADMISSION_ACTIONS_OOF.json before gold evaluation.
    """
    print("=== TRAINING 5-FOLD OOF ADMISSION MODEL & EXECUTING ADMISSION RULE ===", flush=True)
    t0 = time.perf_counter()

    # 1. Verify 5-fold split artifact
    split_sha = sha256_file(CAL600_5FOLD_PATH)
    if split_sha != EXPECTED_5FOLD_SHA256:
        raise RuntimeError(
            f"5-fold split SHA mismatch! {split_sha} != {EXPECTED_5FOLD_SHA256}"
        )

    split_doc = json.loads(CAL600_5FOLD_PATH.read_text(encoding="utf-8"))
    folds: Dict[str, List[str]] = split_doc["folds"]

    repaired_top5: Dict[str, List[str]] = {}
    action_records: Dict[str, Any] = {}
    oof_admission_scores: Dict[str, Dict[str, float]] = {}
    oof_admission_ranks: Dict[str, Dict[str, int]] = {}

    total_actions = 0

    for fold_name in sorted(folds.keys()):
        test_qids = folds[fold_name]
        train_qids = [q for fn, qlist in folds.items() if fn != fold_name for q in qlist]

        # Assemble training population
        X_train_list = []
        y_train_list = []
        weights_train_list = []

        for q in train_qids:
            docs_q = s_universe[q]
            n_q = len(docs_q)
            feat_q = features_per_query[q]
            gold_q = train_gold_labels[q]

            y_q = np.array([1 if d in gold_q else 0 for d in docs_q], dtype=np.int8)
            w_q = np.full(n_q, 1.0 / max(n_q, 1), dtype=np.float32)

            X_train_list.append(feat_q)
            y_train_list.append(y_q)
            weights_train_list.append(w_q)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        weights_train = np.concatenate(weights_train_list)

        # Fit fresh scaler and LogisticRegression per fold
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X_train), y_train, sample_weight=weights_train)

        # Evaluate held fold queries strictly label-free
        for q in test_qids:
            docs_q = s_universe[q]
            n_q = len(docs_q)
            feat_q = features_per_query[q]

            X_test = scaler.transform(feat_q)
            dec_scores = model.decision_function(X_test)

            # Rank by decision score descending, tie break by doc ID string ascending
            order = sorted(range(n_q), key=lambda i: (-dec_scores[i], docs_q[i]))

            adm_scores = {docs_q[i]: float(dec_scores[i]) for i in range(n_q)}
            adm_ranks = {docs_q[order[rank_0]]: rank_0 + 1 for rank_0 in range(n_q)}

            oof_admission_scores[q] = adm_scores
            oof_admission_ranks[q] = adm_ranks

            def_doc = defenders[q]
            def_rank = adm_ranks[def_doc]

            novel_docs = novel_map[q]
            if novel_docs:
                best_novel = min(novel_docs, key=lambda d: adm_ranks[d])
                best_novel_rank = adm_ranks[best_novel]
            else:
                best_novel = None
                best_novel_rank = None

            # Held-Query Admission Rule:
            # Fire iff:
            # 1. NOVEL(q) is non-empty
            # 2. ADMISSION_RANK(BEST_NOVEL) <= 5
            # 3. ADMISSION_RANK(DEFENDER) > 5
            action_fire = bool(
                novel_docs
                and best_novel is not None
                and best_novel_rank <= 5
                and def_rank > 5
            )

            d1_ranks = d1_top5[q]
            if action_fire:
                total_actions += 1
                new_cand_top5 = [d1_ranks[0], d1_ranks[1], d1_ranks[2], d1_ranks[3], best_novel]
            else:
                new_cand_top5 = list(d1_ranks[:5])

            repaired_top5[q] = new_cand_top5

            action_records[q] = {
                "qid": q,
                "fold": fold_name,
                "d1_defender": def_doc,
                "novel_candidates": novel_docs,
                "admission_scores": adm_scores,
                "admission_ranks": adm_ranks,
                "best_novel": best_novel,
                "defender_admission_rank": def_rank,
                "best_novel_admission_rank": best_novel_rank,
                "action_fire": action_fire,
                "candidate_top5": new_cand_top5,
            }

    print(f"OOF admission rule fired on {total_actions} / {len(all_ids)} queries", flush=True)

    # Seal action artifact before utility evaluation
    action_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.actions_oof.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "query_count": len(all_ids),
        "total_actions": total_actions,
        "action_rule": "NOVEL(q) != [] and ADMISSION_RANK(BEST_NOVEL) <= 5 and ADMISSION_RANK(DEFENDER) > 5",
        "model_spec": "LogisticRegression(C=0.15, class_weight='balanced', solver='liblinear', random_state=2026)",
        "sample_weighting": "query-balanced 1/n",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "actions": action_records,
    }

    action_path = RESULTS_DIR / "AITEAM50_SOFT_ADMISSION_ACTIONS_OOF.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    action_path.write_text(json.dumps(action_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    action_sha = sha256_file(action_path)
    print(f"Sealed actions artifact: {action_path} (SHA: {action_sha})", flush=True)

    action_meta = {
        "path": str(action_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": action_sha,
        "total_actions": total_actions,
    }

    return repaired_top5, action_records, action_meta
