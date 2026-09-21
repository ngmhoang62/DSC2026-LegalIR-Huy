#!/usr/bin/env python
"""
Adaptive-K Precision Public Materializer V1
===========================================

CACHE-ONLY / FAIL-CLOSED public deployment for the frozen ROBUST_Z5 contract.

Safety properties:
  * Does not run any GPU model.
  * Refuses to continue if any required public cache is absent/incomplete.
  * Reconstructs the production D1 public ranker from existing caches.
  * Requires 1000/1000 exact ordered Top-5 parity with the current D1 champion.
  * Reads the frozen deployment contract produced by:
        run_adaptive_k_precision_finalize_v1.py
  * Applies exactly one action:
        prune D1 rank5 iff ROBUST_Z5 <= frozen final threshold
  * Never changes ranks 1-4 and never introduces a new document.
  * Produces variable-K (4 or 5) submission JSON + ZIP.

Run:
  python ../run_adaptive_k_precision_public_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SEED = 2026


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pickle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pickle.loads(path.read_bytes())


def load_score_obj(path: Path):
    obj = load_pickle(path)
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def robust_scale(vals) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if len(x) < 2:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    q25, q75 = np.percentile(x, [25, 75])
    iqrn = float(q75 - q25) / 1.349 if q75 > q25 else 0.0
    std = float(np.std(x))
    return max(mad, iqrn, std, 1e-6)


def robust_z5(ordered_docs: List[str], scoremap: Dict[str, float]) -> float:
    vals = np.asarray([scoremap[d] for d in ordered_docs], dtype=np.float64)
    if len(vals) < 5:
        raise RuntimeError("Need >=5 candidate scores for ROBUST_Z5")
    s5 = vals[4]
    med = float(np.median(vals))
    return float((s5 - med) / robust_scale(vals))


def require_complete_mapping(name: str, mapping: Dict[str, Any], qids: List[str]):
    missing = [q for q in qids if q not in mapping]
    if missing:
        raise RuntimeError(
            f"{name} cache incomplete: missing {len(missing)} qids; "
            f"first={missing[:10]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Frozen DEPLOYMENT_CONTRACT.json; defaults under results/manual.",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import (
        CORPUS_CAP,
        CORPUS_DEPTH,
        EXPANSION_CONFIG,
        RERANK_CONFIG,
        DocumentStore,
        raw_union,
        weighted_rrf,
    )
    from run_burst_multistage_submission import load_metadata
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    contract_path = args.contract
    if contract_path is None:
        contract_path = (
            root
            / "results/manual/huy_adaptive_k_precision_finalize_v1/"
            "DEPLOYMENT_CONTRACT.json"
        )
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    if contract.get("family") != "ROBUST_Z5":
        raise RuntimeError(f"Unexpected family: {contract.get('family')}")
    threshold = float(contract["final_threshold_all_cal600"])
    selected = contract.get("selected_oof", {})
    if int(selected.get("gold_removed", -1)) != 0:
        raise RuntimeError("Frozen contract did not have zero OOF gold removals")
    if abs(float(selected.get("delta_recall", 999.0))) > 1e-12:
        raise RuntimeError("Frozen contract changed OOF recall")

    print("[1/6] Frozen contract loaded", flush=True)
    print(
        f"  family=ROBUST_Z5 lambda={contract['global_lambda']} "
        f"threshold={threshold:.12f}",
        flush=True,
    )
    print(
        f"  OOF actions={selected.get('actions')} "
        f"gold_removed={selected.get('gold_removed')} "
        f"deltaP={selected.get('delta_macro_precision'):+.10f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # CAL600 full-training data
    # ------------------------------------------------------------------
    print("[2/6] Building full-CAL D1 training rows (CPU)...", flush=True)

    contexts_dir = (
        root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    docs = DocumentStore(sorted(contexts_dir.glob("context_*.json")))

    queries, blocks, cal_ids, extended, local_views, base_scores = (
        build_training_cap(
            root,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: set(map(str, queries[q][1])) for q in cal_ids}

    def load_pkl_rel(rel: str):
        obj = load_score_obj(root / rel)
        return obj

    def load_aligned(rel: str, floor=None):
        obj = load_pkl_rel(rel)
        fl = floor if floor is not None else min(
            v for q in obj for v in obj[q].values()
        )
        return {
            q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]}
            for q in cal_ids
        }

    vn_a = root / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"
    vn_b = root / "results/embedding_finetunc/vnlegal_lal_cv_scores.pkl"
    if vn_a.is_file():
        vnlegal_cv = load_score_obj(vn_a)
    elif vn_b.is_file():
        vnlegal_cv = load_score_obj(vn_b)
    else:
        raise FileNotFoundError("vnlegal_lal_cv_scores.pkl")

    full_channels_cv = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5),
        "aiteamvn_ft": load_aligned("results/from_drive/aiteamvn_ft_cv.pkl"),
        "jina_ft": load_aligned("results/from_drive/jina_ft_cv.pkl"),
        "title_embed": load_aligned("results/burst_fresh_block/title_embed_scores.pkl"),
    }

    type_table = build_type_table(root, docs, cal_ids, extended)
    type_rows = type_features(extended, type_table, queries, cal_ids)
    own, cited = build_citation_table(docs, cal_ids, extended)
    cite_rows = citation_features(extended, own, cited, cal_ids)

    cal_rows_base, cal_groups = ltr_features(
        local_views, D1_VIEWS, extended, cal_ids, full_channels_cv
    )
    cal_rows = {
        q: np.concatenate(
            [cal_rows_base[q], type_rows[q], cite_rows[q]], axis=1
        )
        for q in cal_ids
    }
    if cal_rows[cal_ids[0]].shape[1] != 48:
        raise RuntimeError(
            f"Expected D1 48D, got {cal_rows[cal_ids[0]].shape[1]}"
        )

    X = np.vstack([cal_rows[q] for q in cal_ids])
    y = np.concatenate(
        [[d in gold[q] for d in cal_groups[q]] for q in cal_ids]
    ).astype(np.int8)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    model.fit(scaler.transform(X), y)

    # ------------------------------------------------------------------
    # Public inputs — CACHE ONLY
    # ------------------------------------------------------------------
    print("[3/6] Loading PUBLIC caches only (GPU forbidden)...", flush=True)

    data_dir = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    paths_meta, doc_ids_meta, train_meta, public_meta = load_metadata(data_dir)
    public_ids = list(public_meta)
    valid_docs = set(map(str, doc_ids_meta))

    if len(public_ids) != 1000:
        raise RuntimeError(f"Expected 1000 public qids, got {len(public_ids)}")

    base_obj = load_pickle(root / "results/burst_gpu_threeview/cpu_top20.pkl")
    base_pub = base_obj["rankings"]
    require_complete_mapping("base_pub", base_pub, public_ids)

    # Retrieval cache: accept the same preferred sources as production.
    retrieval = None
    retrieval_source = None
    for rel in (
        "results/burst_multistage/public_retrieval.pkl",
        "results/burst_robust_fusion/public_retrieval.pkl",
        "results/burst_expanded_fusion/public_retrieval.pkl",
    ):
        p = root / rel
        if not p.is_file():
            continue
        obj = load_pickle(p)
        cache = obj.get("cache", {})
        if all(q in cache for q in public_ids):
            retrieval = cache
            retrieval_source = rel
            break
    if retrieval is None:
        raise RuntimeError(
            "No complete public retrieval cache found. "
            "Refusing to generate it while CE GPU training is running."
        )
    print(f"  retrieval={retrieval_source}", flush=True)

    raw_pub = {
        q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"])
        for q in public_ids
    }

    # Expansion cache
    expansion_path = root / "results/burst_expanded_fusion/expansion_scores.pkl"
    expansion_obj = load_pickle(expansion_path)
    if expansion_obj.get("config") != EXPANSION_CONFIG:
        raise RuntimeError("Expansion cache config mismatch; refusing GPU fallback")
    expansion_scores = expansion_obj.get("scores", {})
    require_complete_mapping("expansion_scores", expansion_scores, public_ids)

    # Ensure score coverage for exact raw union.
    bad = []
    for q in public_ids:
        miss = [d for d in raw_pub[q] if d not in expansion_scores[q]]
        if miss:
            bad.append((q, miss[:5]))
    if bad:
        raise RuntimeError(
            f"Expansion cache lacks required docs for {len(bad)} qids; "
            f"first={bad[:3]}. Refusing GPU fallback."
        )

    dense_rank_pub = {
        q: sorted(
            raw_pub[q],
            key=lambda d: (-expansion_scores[q][d], d),
        )
        for q in public_ids
    }
    expanded_pub = weighted_rrf(
        [raw_pub, dense_rank_pub],
        RERANK_CONFIG["expansion_weights"],
        RERANK_CONFIG["expansion_rrf_k"],
    )

    # Corpus cache
    corpus_path = (
        root
        / "results/burst_expanded_fusion"
        / f"corpus_rank_cap{CORPUS_CAP}.pkl"
    )
    corpus_obj = load_pickle(corpus_path)
    if int(corpus_obj.get("cap", -1)) != int(CORPUS_CAP):
        raise RuntimeError("Corpus cache cap mismatch")
    if int(corpus_obj.get("depth", -1)) < int(CORPUS_DEPTH):
        raise RuntimeError("Corpus cache depth insufficient")
    corpus_rank = corpus_obj["ranking"]
    corpus_score = corpus_obj["scores"]
    require_complete_mapping("corpus_rank", corpus_rank, public_ids)
    require_complete_mapping("corpus_score", corpus_score, public_ids)

    public_candidates = {
        q: list(
            dict.fromkeys(
                list(base_pub[q])
                + expanded_pub[q][: RERANK_CONFIG["expanded_depth"]]
                + corpus_rank[q][:CORPUS_DEPTH]
            )
        )
        for q in public_ids
    }

    # Rerank cache
    rerank_path = root / "results/burst_expanded_fusion/rerank_scores.pkl"
    rerank_obj = load_pickle(rerank_path)
    if rerank_obj.get("config") != RERANK_CONFIG:
        raise RuntimeError("Rerank cache config mismatch; refusing GPU fallback")

    for branch in ("jina", "dense"):
        require_complete_mapping(
            f"rerank_{branch}", rerank_obj[branch], public_ids
        )

    incomplete_pairs = []
    for q in public_ids:
        for d in public_candidates[q]:
            if (
                d not in rerank_obj["jina"][q]
                or d not in rerank_obj["dense"][q]
            ):
                incomplete_pairs.append((q, d))
                if len(incomplete_pairs) >= 10:
                    break
        if len(incomplete_pairs) >= 10:
            break
    if incomplete_pairs:
        raise RuntimeError(
            "Rerank cache incomplete for current public candidate set; "
            f"examples={incomplete_pairs}. Refusing GPU fallback."
        )

    # Other public score caches.
    vnlegal_pub = load_score_obj(
        root / "results/burst_userft_maxrecall/vnlegal_scores.pkl"
    )
    three_view_pub = load_pickle(
        root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl"
    )
    crossenc_pub = load_score_obj(
        root / "results/crossenc_fullpool/public_scores.pkl"
    )

    extra_pub_paths = {
        "aiteamvn_ft": "results/from_drive/aiteamvn_ft_public.pkl",
        "jina_ft": "results/from_drive/jina_ft_public.pkl",
        "title_embed": "results/burst_fresh_block/title_embed_public.pkl",
    }
    extra_pub_raw = {
        name: load_score_obj(root / rel)
        for name, rel in extra_pub_paths.items()
    }

    require_complete_mapping("vnlegal_pub", vnlegal_pub, public_ids)
    require_complete_mapping("three_view_pub", three_view_pub, public_ids)

    corpus_score_trim = {
        q: {
            d: corpus_score[q][d]
            for d in public_candidates[q]
            if d in corpus_score[q]
        }
        for q in public_ids
    }
    expansion_trim = {
        q: {
            d: expansion_scores[q][d]
            for d in public_candidates[q]
            if d in expansion_scores[q]
        }
        for q in public_ids
    }

    view_rank_pub = {
        "base": {q: list(base_pub[q]) for q in public_ids},
        "expanded": {
            q: expanded_pub[q][: RERANK_CONFIG["expanded_depth"]]
            for q in public_ids
        },
        "jina": {
            q: sorted(
                public_candidates[q],
                key=lambda d: (-rerank_obj["jina"][q][d], d),
            )
            for q in public_ids
        },
        "dense": {
            q: sorted(
                public_candidates[q],
                key=lambda d: (-rerank_obj["dense"][q][d], d),
            )
            for q in public_ids
        },
        "corpus": {
            q: sorted(
                [d for d in public_candidates[q] if d in corpus_score_trim[q]],
                key=lambda d: (-corpus_score_trim[q][d], d),
            )
            for q in public_ids
        },
    }

    floor_ce = min(
        min(v.values())
        for v in full_channels_cv["crossenc"].values()
        if v
    )
    public_scores = {
        "jina": rerank_obj["jina"],
        "dense": rerank_obj["dense"],
        "expansion": expansion_trim,
        "e5": {q: three_view_pub[q]["e5"] for q in public_ids},
        "corpus": {
            q: {
                d: corpus_score_trim[q].get(d, -1.0)
                for d in public_candidates[q]
            }
            for q in public_ids
        },
        "vnlegal_lal": vnlegal_pub,
        "crossenc": {
            q: {
                d: crossenc_pub.get(q, {}).get(d, floor_ce)
                for d in public_candidates[q]
            }
            for q in public_ids
        },
    }

    for name, raw in extra_pub_raw.items():
        floor = min(v for q in raw for v in raw[q].values())
        public_scores[name] = {
            q: {
                d: raw.get(q, {}).get(d, floor)
                for d in public_candidates[q]
            }
            for q in public_ids
        }

    # ------------------------------------------------------------------
    # Public D1 feature matrix + exact parity
    # ------------------------------------------------------------------
    print("[4/6] Scoring PUBLIC with full-CAL D1 (CPU)...", flush=True)

    public_queries_meta = {q: (public_meta[q], set()) for q in public_ids}
    public_types = build_type_table(
        root, docs, public_ids, public_candidates
    )
    public_type_rows = type_features(
        public_candidates, public_types, public_queries_meta, public_ids
    )
    public_own, public_cited = build_citation_table(
        docs, public_ids, public_candidates
    )
    public_cite_rows = citation_features(
        public_candidates, public_own, public_cited, public_ids
    )

    pub_rows_base, pub_groups = ltr_features(
        view_rank_pub,
        D1_VIEWS,
        public_candidates,
        public_ids,
        public_scores,
    )
    pub_rows = {
        q: np.concatenate(
            [pub_rows_base[q], public_type_rows[q], public_cite_rows[q]],
            axis=1,
        )
        for q in public_ids
    }

    champion_path = (
        root
        / "results/gemini/huy_vnlegal_rank_ablation_v1/"
        "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    )
    if not champion_path.is_file():
        raise FileNotFoundError(champion_path)
    champion_raw = json.loads(champion_path.read_text(encoding="utf-8"))
    champion = {
        str(q): (
            [str(d) for d in row["answer"]]
            if isinstance(row, dict)
            else [str(d) for d in row]
        )
        for q, row in champion_raw.items()
    }

    d1_ordered = {}
    d1_scoremaps = {}

    for q in public_ids:
        scores = model.decision_function(scaler.transform(pub_rows[q]))
        smap = {
            str(d): float(v)
            for d, v in zip(pub_groups[q], scores)
        }
        order = sorted(
            [str(d) for d in pub_groups[q]],
            key=lambda d: (-smap[d], d),
        )
        d1_ordered[q] = order
        d1_scoremaps[q] = smap

    mismatches = []
    for q in public_ids:
        top5 = [d for d in d1_ordered[q] if d in valid_docs][:5]
        if top5 != champion[q]:
            mismatches.append(
                {
                    "qid": q,
                    "reconstructed": top5,
                    "champion": champion[q],
                }
            )
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise RuntimeError(
            "PUBLIC D1 PARITY FAILED. "
            f"First mismatches={mismatches}"
        )
    print("  exact ordered D1 parity = 1000/1000 PASS", flush=True)

    # ------------------------------------------------------------------
    # Frozen adaptive-K action
    # ------------------------------------------------------------------
    print("[5/6] Applying frozen ROBUST_Z5 pruning...", flush=True)

    output = {}
    actions = []
    z_values = []

    for q in public_ids:
        ordered = d1_ordered[q]
        smap = d1_scoremaps[q]
        top5 = champion[q]

        # z5 is based on ALL D1 candidate scores, but the rank5 doc must agree
        # with the exact champion because parity is already enforced.
        if ordered[4] != top5[4]:
            raise RuntimeError(
                f"Internal rank5/champion disagreement qid={q}"
            )

        z5 = robust_z5(ordered, smap)
        z_values.append(z5)

        if z5 <= threshold:
            answer = top5[:4]
            actions.append(
                {
                    "qid": q,
                    "removed_doc": top5[4],
                    "robust_z5": z5,
                    "threshold": threshold,
                    "k_before": 5,
                    "k_after": 4,
                }
            )
        else:
            answer = top5

        if len(answer) not in (4, 5):
            raise RuntimeError(f"Unexpected K for qid={q}: {len(answer)}")
        if len(set(answer)) != len(answer):
            raise RuntimeError(f"Duplicate docs qid={q}")
        if any(d not in valid_docs for d in answer):
            raise RuntimeError(f"Noncanonical doc qid={q}")

        output[q] = {"answer": answer}

    out_dir = (
        root / "results/manual/huy_adaptive_k_precision_public_v1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "submission.json"
    zip_path = out_dir / "submission.zip"
    report_path = out_dir / "PUBLIC_ADAPTIVE_K_REPORT.json"

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.writestr("submission.json", json_path.read_bytes())

    # ZIP self-check
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.namelist() != ["submission.json"]:
            raise RuntimeError("ZIP layout invalid")
        payload = json.loads(zf.read("submission.json").decode("utf-8"))
    if set(payload) != set(public_ids):
        raise RuntimeError("Submission qid population mismatch")

    k_hist = Counter(len(output[q]["answer"]) for q in public_ids)
    prune_rate = len(actions) / len(public_ids)
    oof_rate = float(selected["actions"]) / 600.0
    rate_ratio = prune_rate / oof_rate if oof_rate > 0 else None

    report = {
        "schema": "manual.adaptive_k_precision_public_v1",
        "status": "READY_FOR_CODABENCH_FORMAT_CHECK",
        "source_champion_json": str(champion_path),
        "source_champion_sha256": sha256_file(champion_path),
        "deployment_contract": str(contract_path),
        "deployment_contract_sha256": sha256_file(contract_path),
        "family": "ROBUST_Z5",
        "threshold": threshold,
        "global_lambda": contract["global_lambda"],
        "public_d1_ordered_parity": {
            "matches": 1000,
            "total": 1000,
            "pass": True,
        },
        "public": {
            "queries": len(public_ids),
            "pruned_queries": len(actions),
            "prune_rate": prune_rate,
            "k_histogram": {str(k): int(v) for k, v in sorted(k_hist.items())},
            "mean_k": float(
                np.mean([len(output[q]["answer"]) for q in public_ids])
            ),
            "robust_z5": {
                "min": float(np.min(z_values)),
                "p05": float(np.quantile(z_values, 0.05)),
                "median": float(np.median(z_values)),
                "p95": float(np.quantile(z_values, 0.95)),
                "max": float(np.max(z_values)),
            },
        },
        "oof_reference": {
            "queries": 600,
            "actions": selected["actions"],
            "prune_rate": oof_rate,
            "gold_removed": selected["gold_removed"],
            "delta_recall": selected["delta_recall"],
            "delta_macro_precision": selected["delta_macro_precision"],
        },
        "distribution_shift_diagnostic": {
            "public_to_oof_prune_rate_ratio": rate_ratio,
            "warning": (
                bool(rate_ratio is not None and (rate_ratio < 0.5 or rate_ratio > 2.0))
            ),
        },
        "actions": actions,
        "files": {
            "submission_json": str(json_path),
            "submission_json_sha256": sha256_file(json_path),
            "submission_zip": str(zip_path),
            "submission_zip_sha256": sha256_file(zip_path),
        },
        "gpu_inference_performed": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[6/6] DONE", flush=True)
    print("=" * 96)
    print("PUBLIC D1 parity: 1000/1000 exact ordered PASS")
    print(
        f"Frozen threshold={threshold:.12f} | "
        f"public prunes={len(actions)}/1000 ({prune_rate:.1%})"
    )
    print(
        f"K histogram={dict(sorted(k_hist.items()))} | "
        f"meanK={report['public']['mean_k']:.4f}"
    )
    print(
        f"OOF prune rate={oof_rate:.1%} | "
        f"public/OOF ratio={rate_ratio:.3f}"
    )
    print(
        "Distribution shift warning="
        f"{report['distribution_shift_diagnostic']['warning']}"
    )
    print(f"JSON: {json_path}")
    print(f"ZIP : {zip_path}")
    print(f"Report: {report_path}")
    print("GPU inference performed: FALSE")
    print("=" * 96)


if __name__ == "__main__":
    main()
