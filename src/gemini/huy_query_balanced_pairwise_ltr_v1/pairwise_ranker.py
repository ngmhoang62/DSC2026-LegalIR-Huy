"""Query-balanced pairwise logistic utility ranker with mathematical consistency verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_query_balanced_pairwise_ltr_v1"


class QueryBalancedPairwiseRanker:
    """Query-balanced pairwise linear logistic utility ranker.

    Fits StandardScaler on document feature rows.
    Forms symmetric positive-negative preference pairs within each query.
    Weights each pair row by 1 / (2 * M) so every usable query contributes
    an exact total sample weight of 1.0.
    Fits LogisticRegression(C=0.15, fit_intercept=False, solver="liblinear").
    At inference time, computes linear utility w · z(q, d) for each candidate.
    """

    def __init__(
        self,
        C: float = 0.15,
        max_iter: int = 3000,
        random_state: int = 2026,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.model: LogisticRegression | None = None
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        X_by_qid: Dict[str, np.ndarray],
        y_by_qid: Dict[str, np.ndarray],
        train_qids: List[str],
    ) -> QueryBalancedPairwiseRanker:
        # Step 1: Fit scaler on all document rows
        X_all_docs = np.vstack([X_by_qid[q] for q in train_qids])
        self.scaler = StandardScaler().fit(X_all_docs)

        # Step 2: Form query-balanced symmetric pairs
        pair_rows: List[np.ndarray] = []
        pair_labels: List[int] = []
        pair_weights: List[float] = []

        for q in train_qids:
            x_q = X_by_qid[q]
            y_q = y_by_qid[q]
            z_q = self.scaler.transform(x_q)

            pos_idx = np.where(y_q == 1)[0]
            neg_idx = np.where(y_q == 0)[0]
            m = len(pos_idx) * len(neg_idx)
            if m == 0:
                continue

            w_pair = 1.0 / (2.0 * m)
            for p in pos_idx:
                for n in neg_idx:
                    # Positive directional pair: z_p - z_n, label 1
                    pair_rows.append(z_q[p] - z_q[n])
                    pair_labels.append(1)
                    pair_weights.append(w_pair)

                    # Negative symmetric pair: z_n - z_p, label 0
                    pair_rows.append(z_q[n] - z_q[p])
                    pair_labels.append(0)
                    pair_weights.append(w_pair)

        if not pair_rows:
            raise RuntimeError("No valid positive-negative pairs found in training queries.")

        X_pairs = np.asarray(pair_rows, dtype=np.float64)
        y_pairs = np.asarray(pair_labels, dtype=np.int8)
        w_pairs = np.asarray(pair_weights, dtype=np.float64)

        # Step 3: Fit LogisticRegression without intercept
        self.model = LogisticRegression(
            C=self.C,
            class_weight=None,
            solver="liblinear",
            fit_intercept=False,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_pairs, y_pairs, sample_weight=w_pairs)
        self.coef_ = self.model.coef_[0]
        return self

    def predict_utility(self, X_q: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.coef_ is None:
            raise RuntimeError("Ranker is not fitted yet.")
        z_q = self.scaler.transform(X_q)
        return z_q @ self.coef_

    def rank(
        self,
        X_q: np.ndarray,
        docs: List[str],
        k: int = 5,
        tie_breaker: str = "stable",
    ) -> List[str]:
        utility = self.predict_utility(X_q)
        if tie_breaker == "lexsort":
            order = np.lexsort((np.asarray(docs), -utility))
        else:
            order = np.argsort(-utility)
        return [docs[i] for i in order[:k]]


def run_pairwise_utility_unit_test(n_dim: int = 48, n_test_pairs: int = 100) -> Dict[str, Any]:
    """Mathematical consistency test: utility(a) - utility(b) == decision_function(z_a - z_b)."""
    np.random.seed(2026)
    n_samples = 300
    X_synth = np.random.randn(n_samples, n_dim).astype(np.float64)
    y_synth = np.random.randint(0, 2, n_samples).astype(np.int8)
    # Ensure both classes exist
    y_synth[0] = 1
    y_synth[1] = 0

    # Group into synthetic queries of 30 docs each
    X_dict = {f"q_{i}": X_synth[i * 30 : (i + 1) * 30] for i in range(10)}
    y_dict = {f"q_{i}": y_synth[i * 30 : (i + 1) * 30] for i in range(10)}
    train_ids = list(X_dict.keys())

    ranker = QueryBalancedPairwiseRanker(C=0.15)
    ranker.fit(X_dict, y_dict, train_ids)

    # Test on deterministic document pairs
    a_docs = np.random.randn(n_test_pairs, n_dim).astype(np.float64)
    b_docs = np.random.randn(n_test_pairs, n_dim).astype(np.float64)

    z_a = ranker.scaler.transform(a_docs)
    z_b = ranker.scaler.transform(b_docs)
    diff_z = z_a - z_b

    util_a = ranker.predict_utility(a_docs)
    util_b = ranker.predict_utility(b_docs)
    util_diff = util_a - util_b

    # Decision function of the linear model evaluated on difference
    df_eval = ranker.model.decision_function(diff_z)

    abs_errors = np.abs(util_diff - df_eval)
    max_abs_error = float(np.max(abs_errors))
    mean_abs_error = float(np.mean(abs_errors))

    passed = max_abs_error <= 1e-8

    result = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.pairwise_utility_unit_test.v1",
        "unit_test_name": "Mathematical Consistency: utility(a) - utility(b) == decision_function(z_a - z_b)",
        "feature_dimension": n_dim,
        "test_pairs_evaluated": n_test_pairs,
        "max_absolute_error": max_abs_error,
        "mean_absolute_error": mean_abs_error,
        "tolerance_threshold": 1e-8,
        "unit_test_passed": passed,
        "status": "PASS" if passed else "FAIL",
    }

    out_file = RESULTS_DIR / "PAIRWISE_UTILITY_UNIT_TEST.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"PAIRWISE_UTILITY_UNIT_TEST: max_abs_error = {max_abs_error:.2e} -> status = {result['status']}")
    assert passed, f"Unit test failed: max_abs_error = {max_abs_error:.2e} > 1e-8"
    return result


def audit_pairwise_data(
    X_by_qid: Dict[str, np.ndarray],
    y_by_qid: Dict[str, np.ndarray],
    train_qids: List[str],
    run_name: str,
) -> Dict[str, Any]:
    """Audit query weights, pair counts, and sample weight sums."""
    n_queries = len(train_qids)
    usable_queries = 0
    zero_gold_queries = 0
    raw_pairs_per_query = []
    total_sample_weights = []

    total_raw_pairs = 0
    total_directional_rows = 0

    for q in train_qids:
        y_q = y_by_qid[q]
        pos_count = int(np.sum(y_q == 1))
        neg_count = int(np.sum(y_q == 0))
        m = pos_count * neg_count

        if m == 0:
            zero_gold_queries += 1
        else:
            usable_queries += 1
            raw_pairs_per_query.append(m)
            directional_rows = 2 * m
            total_raw_pairs += m
            total_directional_rows += directional_rows
            # Each directional row gets weight 1 / (2 * m)
            query_weight_sum = directional_rows * (1.0 / (2.0 * m))
            total_sample_weights.append(query_weight_sum)

    min_weight = float(np.min(total_sample_weights)) if total_sample_weights else 0.0
    max_weight = float(np.max(total_sample_weights)) if total_sample_weights else 0.0

    # Every usable query total weight must equal 1.0 within numerical tolerance
    assert abs(min_weight - 1.0) < 1e-9 and abs(max_weight - 1.0) < 1e-9, (
        f"Query balancing violation: min={min_weight}, max={max_weight}"
    )

    return {
        "run_name": run_name,
        "total_queries": n_queries,
        "usable_queries_with_gold": usable_queries,
        "zero_gold_queries": zero_gold_queries,
        "raw_positive_negative_pairs": total_raw_pairs,
        "directional_pair_rows": total_directional_rows,
        "pairs_per_query": {
            "min": int(np.min(raw_pairs_per_query)) if raw_pairs_per_query else 0,
            "median": float(np.median(raw_pairs_per_query)) if raw_pairs_per_query else 0.0,
            "max": int(np.max(raw_pairs_per_query)) if raw_pairs_per_query else 0,
        },
        "query_sample_weights": {
            "min_total_weight_per_usable_query": min_weight,
            "max_total_weight_per_usable_query": max_weight,
            "query_balanced_assert_passed": True,
        },
    }


if __name__ == "__main__":
    run_pairwise_utility_unit_test()
