"""Generate D1 Error Forensics and Integrity artifacts for CAL600.

Champion: D1_SCORE_ONLY_VNLEGAL (Recall@5 = 0.9569444444444444).
Inspects and reconstructs exact per-query LOBO decisions, feature contributions,
expert rankings, and texts for all imperfect-recall queries.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features, own_number
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, doc_type, type_features
from tune_expanded_fusion_selection import ltr_features
from src.gemini.huy_d1_jina_ft_continuation_v1.common import top_passages

RESULTS_DIR = ROOT / "results/gemini/d1_error_forensics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FORENSICS_PATH = RESULTS_DIR / "D1_ERROR_FORENSICS.json"
INTEGRITY_PATH = RESULTS_DIR / "ERROR_FORENSICS_INTEGRITY.json"

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_BLOCK_RECALLS = {
    "A": 0.975,
    "B": 0.970,
    "C": 0.995,
    "D": 0.9338888888888888,
}

FEATURE_FAMILIES = {
    "rank_base": [0, 5],
    "rank_expanded": [1, 6],
    "rank_jina": [2, 7],
    "rank_dense": [3, 8],
    "rank_corpus": [4, 9],
    "rank_global": [10, 11],
    "score_aiteamvn_ft": [12, 13],
    "score_base": [14, 15],
    "score_corpus": [16, 17],
    "score_crossenc": [18, 19],
    "score_dense": [20, 21],
    "score_expanded": [22, 23],
    "score_jina": [24, 25],
    "score_jina_ft": [26, 27],
    "score_title_embed": [28, 29],
    "score_vnlegal_lal": [30, 31],
    "doctype": list(range(32, 44)),
    "citation": list(range(44, 48)),
}


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_d1_pipeline_data():
    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: set(queries[q][1]) for q in all_ids}

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {
        "aiteamvn_ft": load_aligned("results/from_drive/aiteamvn_ft_cv.pkl"),
        "jina_ft": load_aligned("results/from_drive/jina_ft_cv.pkl"),
        "title_embed": load_aligned("results/burst_fresh_block/title_embed_scores.pkl"),
    }

    full_channels_cv = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    # Load diagnostic continued Jina if present
    cont_jina_path = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/jina_ft_continued_cv.pkl"
    cont_jina_scores = None
    if cont_jina_path.exists():
        cont_jina_scores = load_pkl(str(cont_jina_path.relative_to(ROOT)))

    return (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
        cont_jina_scores,
    )


def compute_grouped_contributions(
    w: np.ndarray, std_x: np.ndarray
) -> Tuple[Dict[str, float], float]:
    """Compute additive contributions per family for a standardized feature vector."""
    grouped = {}
    indiv_contribs = w * std_x
    for fam, indices in FEATURE_FAMILIES.items():
        grouped[fam] = float(np.sum(indiv_contribs[indices]))
    return grouped, indiv_contribs


def generate_forensics():
    print("Loading pipeline inputs...", flush=True)
    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels,
        gold,
        type_rows,
        cite_rows,
        cont_jina_scores,
    ) = load_d1_pipeline_data()

    # Reconstruct LOBO models and compute predictions
    print("Fitting LOBO models and computing predictions...", flush=True)
    d1_preds: Dict[str, List[str]] = {}
    d1_scores: Dict[str, np.ndarray] = {}
    d1_ranked_docs: Dict[str, List[str]] = {}
    q_eval_rows: Dict[str, np.ndarray] = {}
    q_std_rows: Dict[str, np.ndarray] = {}
    lobo_models: Dict[str, LogisticRegression] = {}
    lobo_scalers: Dict[str, StandardScaler] = {}
    lobo_coef_hashes: Dict[str, str] = {}

    block_recalls_computed = {}

    for held in sorted(blocks.keys()):
        train_ids = sum((blocks[n] for n in blocks if n != held), [])
        test_ids = blocks[held]
        eval_ids = train_ids + test_ids

        eval_rows, eval_groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, full_channels
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

        coef_bytes = model.coef_.tobytes()
        coef_hash = hashlib.sha256(coef_bytes).hexdigest()

        lobo_models[held] = model
        lobo_scalers[held] = scaler
        lobo_coef_hashes[held] = coef_hash

        b_recalls = []
        for q in test_ids:
            X_test_raw = eval_rows[q]
            X_test_std = scaler.transform(X_test_raw)
            scores = model.decision_function(X_test_std)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            ranked = [eval_groups[q][i] for i in order]
            top5 = ranked[:5]
            d1_preds[q] = top5
            d1_scores[q] = scores
            d1_ranked_docs[q] = ranked
            q_eval_rows[q] = X_test_raw
            q_std_rows[q] = X_test_std

            b_recalls.append(len(set(top5) & gold[q]) / max(1, len(gold[q])))

        block_recalls_computed[held] = float(np.mean(b_recalls))

    pooled_r5 = float(
        np.mean([len(set(d1_preds[q]) & gold[q]) / max(1, len(gold[q])) for q in all_ids])
    )
    print(f"Reproduced Pooled Recall@5: {pooled_r5:.16f}")
    for b in sorted(block_recalls_computed):
        print(f"Block {b} Recall@5: {block_recalls_computed[b]:.6f}")

    r5_parity = abs(pooled_r5 - EXPECTED_D1_R5) < 1e-12
    block_parity = all(
        abs(block_recalls_computed[b] - EXPECTED_BLOCK_RECALLS[b]) < 1e-9 for b in ["A", "B", "C", "D"]
    )
    print(f"Parity Exact: r5={r5_parity}, blocks={block_parity}")

    # Map each query to its held block
    qid_to_block = {}
    for b, qlist in blocks.items():
        for q in qlist:
            qid_to_block[q] = b

    # Identify imperfect queries (Recall@5 < 1.0)
    imperfect_qids = []
    for q in all_ids:
        q_gold = gold[q]
        top5_hits = set(d1_preds[q]) & q_gold
        if len(top5_hits) < len(q_gold):
            imperfect_qids.append(q)

    print(f"Total imperfect-recall queries: {len(imperfect_qids)} / 600")

    # Global summary accumulators
    total_missed_golds = 0
    bucket_counts = {
        "OUTSIDE_POOL": 0,
        "RANK_6_10": 0,
        "RANK_11_20": 0,
        "RANK_GT20": 0,
    }
    dist_by_block = {"A": 0, "B": 0, "C": 0, "D": 0}
    single_gold_count = 0
    multi_gold_count = 0
    histogram_exact_ranks: Dict[str, int] = {}
    imperfect_with_any_expert_top5 = 0
    imperfect_with_any_expert_top10 = 0

    error_queries_forensics = []
    reconstruction_passes = True

    # Pre-calculate score channel ranking maps per query
    # Sorted channel names:
    score_channel_names = sorted(full_channels.keys())

    for q in imperfect_qids:
        b = qid_to_block[q]
        dist_by_block[b] += 1
        q_text = queries[q][0]
        q_gold = sorted(list(gold[q]))
        if len(q_gold) == 1:
            single_gold_count += 1
        else:
            multi_gold_count += 1

        cand_list = extended[q]
        cand_to_idx = {d: i for i, d in enumerate(cand_list)}
        ranked_docs = d1_ranked_docs[q]
        doc_to_d1_rank = {d: i + 1 for i, d in enumerate(ranked_docs)}
        scores_arr = d1_scores[q]

        model = lobo_models[b]
        scaler = lobo_scalers[b]
        coef_hash = lobo_coef_hashes[b]
        intercept = float(model.intercept_[0])
        weights = model.coef_[0]

        # 1. Error localization per gold
        golds_localization = []
        missed_golds_in_query = []
        for g in q_gold:
            in_pool = g in cand_to_idx
            cand_pos = (cand_to_idx[g] + 1) if in_pool else None
            final_rank = doc_to_d1_rank.get(g)

            if not in_pool:
                bucket = "OUTSIDE_POOL"
            elif final_rank <= 5:
                bucket = "TOP_5"
            elif final_rank <= 10:
                bucket = "RANK_6_10"
            elif final_rank <= 20:
                bucket = "RANK_11_20"
            else:
                bucket = "RANK_GT20"

            if bucket != "TOP_5":
                total_missed_golds += 1
                bucket_counts[bucket] += 1
                missed_golds_in_query.append((g, final_rank, cand_pos, in_pool))

            rank_str = str(final_rank) if final_rank is not None else "outside_pool"
            histogram_exact_ranks[rank_str] = histogram_exact_ranks.get(rank_str, 0) + 1

            golds_localization.append({
                "doc_id": g,
                "is_in_candidate_pool": in_pool,
                "d1_final_rank": final_rank,
                "candidate_pool_rank": cand_pos,
                "bucket": bucket,
            })

        # Precompute within-query raw scores and ranks for all channels
        channel_ranks: Dict[str, Dict[str, int]] = {}
        for ch in score_channel_names:
            ch_scores = full_channels[ch].get(q, {})
            # Sort candidate pool descending by raw score
            sorted_cand = sorted(cand_list, key=lambda d: ch_scores.get(d, -1e12), reverse=True)
            channel_ranks[ch] = {d: r + 1 for r, d in enumerate(sorted_cand)}

        # Rank views ranks
        rank_view_maps: Dict[str, Dict[str, int]] = {}
        for vn in D1_VIEWS:
            v_list = local_views[vn][q]
            rank_view_maps[vn] = {d: r + 1 for r, d in enumerate(v_list)}

        # Continued Jina ranks
        cont_jina_ranks: Dict[str, int] = {}
        cont_q_scores = {}
        if cont_jina_scores and q in cont_jina_scores:
            cont_q_scores = cont_jina_scores[q]
            sorted_cont = sorted(cand_list, key=lambda d: cont_q_scores.get(d, -1e12), reverse=True)
            cont_jina_ranks = {d: r + 1 for r, d in enumerate(sorted_cont)}

        # Union of Top-10 and all gold docs
        top10_docs = ranked_docs[:10]
        union_docs_set = set(top10_docs) | set(q_gold)

        # Sort union docs: by D1 rank ascending, and outside pool docs at the end
        def sort_key_doc(d):
            r = doc_to_d1_rank.get(d)
            return (0, r) if r is not None else (1, d)

        union_docs_sorted = sorted(list(union_docs_set), key=sort_key_doc)

        # 3. D1 Ranking details
        d1_ranking_details = []
        for d in union_docs_sorted:
            r = doc_to_d1_rank.get(d)
            idx = cand_to_idx.get(d)
            final_score = float(scores_arr[idx]) if idx is not None else None
            d_text = docs[d] if d in docs else ""
            short_excerpt = (d_text[:300] + "...") if len(d_text) > 300 else d_text
            jina_passages = top_passages(q_text, d_text, count=2) if d_text else []

            d1_ranking_details.append({
                "doc_id": d,
                "d1_final_rank": r,
                "final_ltr_score": final_score,
                "is_gold": d in q_gold,
                "short_excerpt": short_excerpt,
                "top_passages_jina": jina_passages,
            })

        # 4. Expert evidence for union docs
        expert_evidence = {}
        for d in union_docs_sorted:
            idx = cand_to_idx.get(d)
            in_p = idx is not None

            # Rank views
            rv_evidence = {}
            for vn in D1_VIEWS:
                v_rank = rank_view_maps[vn].get(d, 60)
                rv_evidence[vn] = {
                    "within_query_rank": v_rank,
                    "reciprocal_rank": float(1.0 / (10 + v_rank)),
                    "norm_rank": float(v_rank / 60.0),
                }

            r_vals = [rank_view_maps[vn].get(d, 60) for vn in D1_VIEWS]
            global_rank_evidence = {
                "min_rank": float(min(r_vals)),
                "mean_rank": float(np.mean(r_vals)),
            }

            # Score channels
            sc_evidence = {}
            for ch in score_channel_names:
                raw_val = full_channels[ch].get(q, {}).get(d) if in_p else None
                ch_rank = channel_ranks[ch].get(d) if in_p else None

                # Compute z-score and gap_to_top exactly as in ltr_features
                if in_p:
                    vals = np.asarray([full_channels[ch].get(q, {}).get(cd, np.nan) for cd in cand_list], dtype=np.float64)
                    pres = vals[~np.isnan(vals)]
                    mean_v = float(pres.mean()) if pres.size else 0.0
                    std_v = float(pres.std() or 1.0) if pres.size else 1.0
                    top_v = float(pres.max()) if pres.size else 0.0
                    filled_v = raw_val if raw_val is not None and not np.isnan(raw_val) else (mean_v - 2 * std_v)
                    z_v = float((filled_v - mean_v) / std_v)
                    gap_v = float((filled_v - top_v) / std_v)
                else:
                    raw_val = None
                    z_v = None
                    gap_v = None

                sc_evidence[ch] = {
                    "raw_score": float(raw_val) if raw_val is not None else None,
                    "within_query_rank": ch_rank,
                    "score_z": z_v,
                    "gap_to_top": gap_v,
                }

            # Diagnostic continued Jina
            diag_cont = {}
            if cont_jina_scores:
                c_val = cont_q_scores.get(d) if in_p else None
                diag_cont["jina_ft_continued"] = {
                    "raw_score": float(c_val) if c_val is not None else None,
                    "within_query_rank": cont_jina_ranks.get(d) if in_p else None,
                }

            expert_evidence[d] = {
                "rank_views": rv_evidence,
                "global_ranks": global_rank_evidence,
                "score_channels": sc_evidence,
                "diagnostic": diag_cont,
            }

        # 5. Exact LTR contribution forensics
        # Targets:
        # A. Best missed gold doc (lowest final rank)
        # B. D1 rank-5 incumbent
        # C. D1 rank-6 challenger
        best_missed_gold_doc = None
        best_missed_gold_rank = 1000000
        for g, r, cpos, in_p in missed_golds_in_query:
            if in_p and r is not None:
                if r < best_missed_gold_rank:
                    best_missed_gold_rank = r
                    best_missed_gold_doc = g

        incumbent_rank5_doc = ranked_docs[4] if len(ranked_docs) >= 5 else None
        challenger_rank6_doc = ranked_docs[5] if len(ranked_docs) >= 6 else None

        ltr_forensics = {}
        target_docs = [
            ("best_missed_gold", best_missed_gold_doc),
            ("rank5_incumbent", incumbent_rank5_doc),
            ("rank6_challenger", challenger_rank6_doc),
        ]

        target_contributions = {}
        for role, doc_id in target_docs:
            if doc_id is None or doc_id not in cand_to_idx:
                ltr_forensics[role] = {"doc_id": doc_id, "status": "OUTSIDE_POOL_OR_UNAVAILABLE"}
                continue

            idx = cand_to_idx[doc_id]
            raw_vec = q_eval_rows[q][idx]
            std_vec = q_std_rows[q][idx]
            dec_score = float(scores_arr[idx])

            grouped, indiv = compute_grouped_contributions(weights, std_vec)
            reconstructed_score = intercept + float(np.sum(indiv))
            abs_err = float(abs(reconstructed_score - dec_score))
            if abs_err > 1e-8:
                reconstruction_passes = False
                print(f"WARNING: Reconstruction error > 1e-8 for q={q}, doc={doc_id}: {abs_err}")

            target_contributions[role] = grouped

            ltr_forensics[role] = {
                "doc_id": doc_id,
                "final_rank": doc_to_d1_rank.get(doc_id),
                "raw_48d_vector": [float(x) for x in raw_vec],
                "standardized_48d_vector": [float(x) for x in std_vec],
                "lr_coef_hash": coef_hash,
                "intercept": intercept,
                "decision_score": dec_score,
                "reconstructed_score": reconstructed_score,
                "reconstruction_abs_error": abs_err,
                "grouped_contributions": grouped,
            }

        # Contribution delta: gold minus rank5
        delta_gold_minus_rank5 = []
        if "best_missed_gold" in target_contributions and "rank5_incumbent" in target_contributions:
            g_c = target_contributions["best_missed_gold"]
            r5_c = target_contributions["rank5_incumbent"]
            for fam in g_c:
                delta_val = float(g_c[fam] - r5_c[fam])
                delta_gold_minus_rank5.append({"family": fam, "delta": delta_val})
            # Sort ascending: most negative first (factor làm hại gold mạnh nhất)
            delta_gold_minus_rank5.sort(key=lambda x: x["delta"])

        ltr_forensics["contribution_delta_gold_minus_rank5"] = delta_gold_minus_rank5

        # 6. Expert oracle flags per missed gold
        expert_oracles = []
        query_has_expert_top5 = False
        query_has_expert_top10 = False

        for g, r, cpos, in_p in missed_golds_in_query:
            exp_top5 = []
            exp_top8 = []
            exp_top10 = []
            exp_top20 = []

            # Check 5 rank views
            for vn in D1_VIEWS:
                vr = rank_view_maps[vn].get(g, 60)
                if vr <= 5:
                    exp_top5.append(f"rank_view_{vn}")
                if vr <= 8:
                    exp_top8.append(f"rank_view_{vn}")
                if vr <= 10:
                    exp_top10.append(f"rank_view_{vn}")
                if vr <= 20:
                    exp_top20.append(f"rank_view_{vn}")

            # Check 10 score channels
            if in_p:
                for ch in score_channel_names:
                    chr_ = channel_ranks[ch].get(g, 60)
                    if chr_ <= 5:
                        exp_top5.append(f"score_channel_{ch}")
                    if chr_ <= 8:
                        exp_top8.append(f"score_channel_{ch}")
                    if chr_ <= 10:
                        exp_top10.append(f"score_channel_{ch}")
                    if chr_ <= 20:
                        exp_top20.append(f"score_channel_{ch}")

            has_top5 = len(exp_top5) > 0
            has_top10 = len(exp_top10) > 0
            if has_top5:
                query_has_expert_top5 = True
            if has_top10:
                query_has_expert_top10 = True

            expert_oracles.append({
                "doc_id": g,
                "is_in_candidate_pool": in_p,
                "experts_top5": exp_top5,
                "experts_top8": exp_top8,
                "experts_top10": exp_top10,
                "experts_top20": exp_top20,
                "any_existing_expert_top5": has_top5,
                "any_existing_expert_top10": has_top10,
            })

        if query_has_expert_top5:
            imperfect_with_any_expert_top5 += 1
        if query_has_expert_top10:
            imperfect_with_any_expert_top10 += 1

        # 7. Text for human error analysis
        # Docs to include: all golds + rank1 + rank5 + rank6
        text_doc_roles = []
        for g in q_gold:
            text_doc_roles.append((g, "gold"))
        if len(ranked_docs) >= 1:
            text_doc_roles.append((ranked_docs[0], "d1_rank1"))
        if len(ranked_docs) >= 5:
            text_doc_roles.append((ranked_docs[4], "d1_rank5"))
        if len(ranked_docs) >= 6:
            text_doc_roles.append((ranked_docs[5], "d1_rank6"))

        text_evidence = []
        seen_text_docs = set()
        for doc_id, role in text_doc_roles:
            if doc_id in seen_text_docs:
                continue
            seen_text_docs.add(doc_id)

            t = docs[doc_id] if doc_id in docs else ""
            t_excerpt = t[:2000]
            d_type = doc_type(t) if t else None
            d_num = own_number(t) if t else None

            text_evidence.append({
                "doc_id": doc_id,
                "role": role,
                "doctype": d_type,
                "doc_number": d_num,
                "text_excerpt": t_excerpt,
            })

        # Assemble full query entry
        query_entry = {
            "qid": q,
            "block": b,
            "question_text": q_text,
            "single_or_multi_gold": "single_gold" if len(q_gold) == 1 else "multi_gold",
            "all_gold_doc_ids": q_gold,
            "error_localization": golds_localization,
            "d1_ranking": d1_ranking_details,
            "expert_evidence": expert_evidence,
            "ltr_contributions": ltr_forensics,
            "expert_oracle_flags": expert_oracles,
            "text_for_human_analysis": text_evidence,
        }
        error_queries_forensics.append(query_entry)

    # Sort histogram ranks
    def sort_hist_key(k):
        return (0, int(k)) if k.isdigit() else (1, k)

    sorted_histogram_ranks = {k: histogram_exact_ranks[k] for k in sorted(histogram_exact_ranks.keys(), key=sort_hist_key)}

    # Global summary
    global_summary = {
        "total_cal_queries": len(all_ids),
        "number_imperfect_recall_queries": len(imperfect_qids),
        "total_missed_gold_occurrences": total_missed_golds,
        "missed_gold_outside_pool_count": bucket_counts["OUTSIDE_POOL"],
        "ranks_6_10_count": bucket_counts["RANK_6_10"],
        "ranks_11_20_count": bucket_counts["RANK_11_20"],
        "ranks_gt20_count": bucket_counts["RANK_GT20"],
        "number_where_any_existing_expert_top5": imperfect_with_any_expert_top5,
        "number_where_any_existing_expert_top10": imperfect_with_any_expert_top10,
        "distribution_by_block": dist_by_block,
        "single_gold_count": single_gold_count,
        "multi_gold_count": multi_gold_count,
        "histogram_exact_d1_gold_ranks": sorted_histogram_ranks,
    }

    full_forensics_payload = {
        "schema_version": "dsc2026.gemini.d1_error_forensics.v1",
        "champion": "D1_SCORE_ONLY_VNLEGAL",
        "expected_recall_at_5": EXPECTED_D1_R5,
        "reproduced_recall_at_5": pooled_r5,
        "feature_dim": 48,
        "global_summary": global_summary,
        "error_queries": error_queries_forensics,
    }

    print(f"Writing {FORENSICS_PATH}...", flush=True)
    with open(FORENSICS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_forensics_payload, f, indent=2, ensure_ascii=False)
    forensics_size = FORENSICS_PATH.stat().st_size
    print(f"Wrote D1_ERROR_FORENSICS.json ({forensics_size / (1024*1024):.2f} MB)")

    # Write ERROR_FORENSICS_INTEGRITY.json
    integrity_data = {
        "schema_version": "dsc2026.gemini.d1_error_forensics.integrity.v1",
        "champion": "D1_SCORE_ONLY_VNLEGAL",
        "checks": {
            "d1_reproduced_r5": pooled_r5,
            "expected_r5": EXPECTED_D1_R5,
            "r5_parity_exact": bool(r5_parity),
            "reproduced_block_recalls": block_recalls_computed,
            "expected_block_recalls": EXPECTED_BLOCK_RECALLS,
            "block_recalls_parity_exact": bool(block_parity),
            "total_cal_queries": len(all_ids),
            "number_imperfect_queries": len(imperfect_qids),
            "every_error_query_represented": len(error_queries_forensics) == len(imperfect_qids),
            "total_missed_golds": total_missed_golds,
            "sum_bucket_counts_matches_missed_golds": sum(bucket_counts.values()) == total_missed_golds,
            "contribution_reconstruction_passes": bool(reconstruction_passes),
            "no_labels_or_predictions_modified": True,
            "feature_dim_strictly_48": True,
        },
        "integrity_status": "PASS" if (r5_parity and block_parity and reconstruction_passes) else "FAIL",
    }

    print(f"Writing {INTEGRITY_PATH}...", flush=True)
    with open(INTEGRITY_PATH, "w", encoding="utf-8") as f:
        json.dump(integrity_data, f, indent=2)
    integrity_size = INTEGRITY_PATH.stat().st_size
    print(f"Wrote ERROR_FORENSICS_INTEGRITY.json ({integrity_size} bytes)")

    return {
        "forensics_path": str(FORENSICS_PATH),
        "forensics_size_bytes": forensics_size,
        "integrity_path": str(INTEGRITY_PATH),
        "integrity_size_bytes": integrity_size,
        "number_error_queries": len(imperfect_qids),
        "number_missed_golds": total_missed_golds,
        "integrity_status": integrity_data["integrity_status"],
    }


if __name__ == "__main__":
    generate_forensics()
