#!/usr/bin/env python
"""
MATERIALIZE PRIVATE D1 + PRISM FROM EXISTING V14 CACHES
=======================================================

CPU-only. No neural inference. No private labels. No REL_L0.

Inputs:
  1) clean held-out CAL Prism scores (`best_channel_scores.pkl`)
  2) complete private Prism scores for the exact v14 candidate pool
  3) existing v14 private caches

Before accepting any Prism result, reconstruct ordinary private D1 and require
exact ordered Top5 parity 2080/2080 against D1_PRIVATE_V14_FAST.json.

Supported arms:
  prism_score
  prism_replace_jina_ft
  prism_replace_crossenc
  prism_score_rank

Output is separate and NEVER overwrites confirmed v14 artifacts.
"""

from __future__ import annotations
import argparse, hashlib, json, pickle, sys, zipfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
SEED = 2026


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pickle(path: Path):
    return pickle.loads(path.read_bytes())


def load_score_dict(path: Path):
    obj = load_pickle(path)
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        obj = obj["scores"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported score artifact: {type(obj)}")
    return {
        str(q): {str(d): float(s) for d, s in row.items()}
        for q, row in obj.items()
    }


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def zip_exact(json_path: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path, "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.namelist() != ["submission.json"]:
            raise RuntimeError("ZIP member contract failed")
        if zf.read("submission.json") != json_path.read_bytes():
            raise RuntimeError("ZIP byte parity failed")


def unwrap_scores(obj):
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def align(raw, ids, candidates, floor=None):
    if floor is None:
        vals = [v for q in raw.values() for v in q.values()]
        if not vals:
            raise RuntimeError("Empty score artifact")
        floor = float(min(vals))
    out = {
        q: {d: float(raw.get(q, {}).get(d, floor)) for d in candidates[q]}
        for q in ids
    }
    return out, float(floor)


def build_arm(name, full_channels, prism, local_views, candidates, ids):
    channels = dict(full_channels)
    views = dict(local_views)
    view_names = list(BASE_VIEWS)

    if name == "d1":
        pass
    elif name == "prism_score":
        channels["prism_ft"] = prism
    elif name == "prism_replace_jina_ft":
        channels.pop("jina_ft", None)
        channels["prism_ft"] = prism
    elif name == "prism_replace_crossenc":
        channels.pop("crossenc", None)
        channels["prism_ft"] = prism
    elif name == "prism_score_rank":
        channels["prism_ft"] = prism
        views["prism_ft"] = {
            q: sorted(candidates[q], key=lambda d: (-prism[q][d], d))
            for q in ids
        }
        view_names.append("prism_ft")
    else:
        raise ValueError(name)

    expected_dim = {
        "d1": 48,
        "prism_score": 50,
        "prism_replace_jina_ft": 48,
        "prism_replace_crossenc": 48,
        "prism_score_rank": 52,
    }[name]
    return channels, views, view_names, expected_dim


def train_full_cal(*, arm, all_ids, extended, local_views, full_channels,
                   prism_cal, type_rows, cite_rows, gold):
    from tune_expanded_fusion_selection import ltr_features

    channels, views, view_names, expected_dim = build_arm(
        arm, full_channels, prism_cal, local_views, extended, all_ids
    )
    rows0, groups = ltr_features(
        views, view_names, extended, all_ids, channels
    )
    rows = {
        q: np.concatenate([rows0[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }
    if rows[all_ids[0]].shape[1] != expected_dim:
        raise RuntimeError(
            f"{arm}: CAL dim {rows[all_ids[0]].shape[1]} != {expected_dim}"
        )

    X = np.vstack([rows[q] for q in all_ids])
    y = np.concatenate([
        [d in gold[q] for d in groups[q]]
        for q in all_ids
    ]).astype(np.int8)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=.15, class_weight="balanced", solver="liblinear",
        max_iter=3000, random_state=SEED,
    )
    model.fit(scaler.transform(X), y)
    return scaler, model, expected_dim


def reconstruct_private(root: Path, ids):
    from benchmark_dense_expansion_holdouts import raw_union
    from tune_burst_multistage_posterior import weighted_rrf
    from run_burst_expanded_fusion_submission import (
        EXPANSION_CONFIG, RERANK_CONFIG, CORPUS_CAP, CORPUS_DEPTH,
    )

    exact = root / "results/manual/huy_private_d1_rel_l0_exact_v1"
    cache = exact / "cache"
    stage5 = (
        root
        / "results/manual/huy_private_d1_rel_l0_approx_v1/"
        "cache/candidate_generation"
    )

    paths = {
        "retrieval": cache / "private_retrieval.pkl",
        "base": cache / "private_base_top20_historical_exact.pkl",
        "exp": stage5 / "expansion_scores.pkl",
        "corpus": stage5 / f"corpus_rank_cap{CORPUS_CAP}.pkl",
        "rr": stage5 / "rerank_scores.pkl",
        "e5": cache / "d1_e5_small_threeview_scores.pkl",
        "vn": cache / "vnlegal/vnlegal_scores_exact64_prefetch.pkl",
        "crossenc": cache / "crossenc_scores.pkl",
        "aift": cache / "aiteamvn_ft_scores.pkl",
        "jinaft": cache / "jina_ft_scores.pkl",
        "title": cache / "title_embed_scores.pkl",
    }
    for name, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(
                f"Required complete v14 cache missing [{name}]: {p}. "
                "This materializer never recomputes caches."
            )

    retrieval_obj = load_pickle(paths["retrieval"])
    retrieval = retrieval_obj.get("cache", retrieval_obj)
    base_obj = load_pickle(paths["base"])
    base = base_obj.get("rankings", base_obj)

    expansion_score_all = unwrap_scores(load_pickle(paths["exp"]))
    corpus_obj = load_pickle(paths["corpus"])
    corpus_rank = corpus_obj["ranking"]
    corpus_score = corpus_obj["scores"]
    rr = load_pickle(paths["rr"])

    if any(q not in retrieval or q not in base for q in ids):
        raise RuntimeError("Private retrieval/base population incomplete")

    raw = {
        q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"])
        for q in ids
    }
    for q in ids:
        if q not in expansion_score_all or any(
            d not in expansion_score_all[q] for d in raw[q]
        ):
            raise RuntimeError(
                f"Expansion cache incomplete q={q}; refusing inference fallback"
            )

    dense_rank = {
        q: sorted(
            raw[q],
            key=lambda d: (-expansion_score_all[q][d], d),
        )
        for q in ids
    }
    expanded_rank_all = weighted_rrf(
        [raw, dense_rank],
        RERANK_CONFIG["expansion_weights"],
        RERANK_CONFIG["expansion_rrf_k"],
    )
    expanded = {
        q: expanded_rank_all[q][:RERANK_CONFIG["expanded_depth"]]
        for q in ids
    }
    candidates = {
        q: list(dict.fromkeys(
            list(base[q])
            + expanded[q]
            + list(corpus_rank[q][:CORPUS_DEPTH])
        ))
        for q in ids
    }

    corpus_sparse = {
        q: {
            d: corpus_score[q][d]
            for d in candidates[q]
            if d in corpus_score[q]
        }
        for q in ids
    }
    expansion = {
        q: {
            d: expansion_score_all[q][d]
            for d in candidates[q]
            if d in expansion_score_all[q]
        }
        for q in ids
    }

    for fam in ("jina", "dense"):
        bad = [
            q for q in ids
            if q not in rr.get(fam, {})
            or any(d not in rr[fam][q] for d in candidates[q])
        ]
        if bad:
            raise RuntimeError(
                f"Stage5 rerank cache incomplete {fam}: {len(bad)} queries"
            )

    return {
        "exact_root": exact,
        "cache": cache,
        "base": base,
        "expanded": expanded,
        "candidates": candidates,
        "corpus_sparse": corpus_sparse,
        "expansion": expansion,
        "rr": rr,
        "e5": load_pickle(paths["e5"]),
        "vn": load_pickle(paths["vn"]),
        "crossenc": load_pickle(paths["crossenc"]),
        "aift": load_pickle(paths["aift"]),
        "jinaft": load_pickle(paths["jinaft"]),
        "title": load_pickle(paths["title"]),
    }


def private_feature_bundle(*, root, documents, ids, questions, priv,
                           prism_private, arm, floors):
    from tune_expanded_fusion_selection import ltr_features
    from tune_doctype_features import build_type_table, type_features
    from tune_citation_graph import build_citation_table, citation_features

    cand = priv["candidates"]

    view_rank = {
        "base": {q: list(priv["base"][q]) for q in ids},
        "expanded": {q: list(priv["expanded"][q]) for q in ids},
        "jina": {
            q: sorted(
                cand[q],
                key=lambda d: (-priv["rr"]["jina"][q][d], d),
            )
            for q in ids
        },
        "dense": {
            q: sorted(
                cand[q],
                key=lambda d: (-priv["rr"]["dense"][q][d], d),
            )
            for q in ids
        },
        "corpus": {
            q: sorted(
                (d for d in cand[q] if d in priv["corpus_sparse"][q]),
                key=lambda d: (-priv["corpus_sparse"][q][d], d),
            )
            for q in ids
        },
    }

    scores = {
        "jina": priv["rr"]["jina"],
        "dense": priv["rr"]["dense"],
        "expansion": priv["expansion"],
        "e5": priv["e5"],
        "corpus": {
            q: {
                d: priv["corpus_sparse"][q].get(d, -1.0)
                for d in cand[q]
            }
            for q in ids
        },
        "vnlegal_lal": priv["vn"],
        "crossenc": {
            q: {
                d: priv["crossenc"].get(q, {}).get(d, floors["crossenc"])
                for d in cand[q]
            }
            for q in ids
        },
        "aiteamvn_ft": {
            q: {
                d: priv["aift"].get(q, {}).get(d, floors["aiteamvn_ft"])
                for d in cand[q]
            }
            for q in ids
        },
        "jina_ft": {
            q: {
                d: priv["jinaft"].get(q, {}).get(d, floors["jina_ft"])
                for d in cand[q]
            }
            for q in ids
        },
        "title_embed": {
            q: {
                d: priv["title"].get(q, {}).get(d, floors["title_embed"])
                for d in cand[q]
            }
            for q in ids
        },
    }

    channels, views, view_names, expected_dim = build_arm(
        arm, scores, prism_private, view_rank, cand, ids
    )

    qmeta = {q: (questions[q], set()) for q in ids}
    types = build_type_table(root, documents, ids, cand)
    trows = type_features(cand, types, qmeta, ids)
    own, cited = build_citation_table(documents, ids, cand)
    crows = citation_features(cand, own, cited, ids)

    rows0, groups = ltr_features(
        views, view_names, cand, ids, channels
    )
    rows = {
        q: np.concatenate([rows0[q], trows[q], crows[q]], axis=1)
        for q in ids
    }
    if rows[ids[0]].shape[1] != expected_dim:
        raise RuntimeError(
            f"{arm}: private dim {rows[ids[0]].shape[1]} != {expected_dim}"
        )
    return rows, groups, expected_dim


def infer(rows, groups, ids, scaler, model):
    top5, decision = {}, {}
    for q in ids:
        s = model.decision_function(scaler.transform(rows[q]))
        order = np.argsort(-s)
        top5[q] = [groups[q][i] for i in order[:5]]
        decision[q] = {
            groups[q][i]: float(s[i])
            for i in range(len(groups[q]))
        }
    return top5, decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prism-heldout-scores", type=Path, required=True)
    ap.add_argument("--prism-private-scores", type=Path, required=True)
    ap.add_argument(
        "--arm",
        choices=(
            "prism_score",
            "prism_replace_jina_ft",
            "prism_replace_crossenc",
            "prism_score_rank",
        ),
        required=True,
    )
    ap.add_argument("--private-file", default="private-official.json")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    cal_prism_path = args.prism_heldout_scores.resolve()
    private_prism_path = args.prism_private_scores.resolve()
    for p in (cal_prism_path, private_prism_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    print("[1/6] Loading CAL600 D1 + clean Prism held-out scores...")
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    from run_burst_expanded_fusion_submission import DocumentStore

    (
        _queries, _blocks, all_ids, extended, local_views, full_channels,
        gold, _vnlegal, type_rows, cite_rows,
    ) = load_cal_inputs()

    raw_cal_prism = load_score_dict(cal_prism_path)
    prism_cal, prism_cal_floor = align(
        raw_cal_prism, all_ids, extended
    )

    floors = {
        "crossenc": float(min(
            v for row in full_channels["crossenc"].values() for v in row.values()
        )),
        "aiteamvn_ft": float(min(
            v for row in full_channels["aiteamvn_ft"].values() for v in row.values()
        )),
        "jina_ft": float(min(
            v for row in full_channels["jina_ft"].values() for v in row.values()
        )),
        "title_embed": float(min(
            v for row in full_channels["title_embed"].values() for v in row.values()
        )),
    }

    print("[2/6] Training baseline D1 + selected Prism arm on all CAL600...")
    d1_scaler, d1_model, d1_dim = train_full_cal(
        arm="d1", all_ids=all_ids, extended=extended,
        local_views=local_views, full_channels=full_channels,
        prism_cal=prism_cal, type_rows=type_rows,
        cite_rows=cite_rows, gold=gold,
    )
    prism_scaler, prism_model, prism_dim = train_full_cal(
        arm=args.arm, all_ids=all_ids, extended=extended,
        local_views=local_views, full_channels=full_channels,
        prism_cal=prism_cal, type_rows=type_rows,
        cite_rows=cite_rows, gold=gold,
    )

    print("[3/6] Loading private queries + reconstructing v14 caches only...")
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    private_path = data / args.private_file
    raw_private = json.loads(private_path.read_text(encoding="utf-8"))
    ids = [str(q) for q in raw_private]
    questions = {
        str(q): str(v["question"] if isinstance(v, dict) else v)
        for q, v in raw_private.items()
    }
    if len(ids) != 2080:
        raise RuntimeError(f"Expected 2080 private queries, got {len(ids)}")

    corpus_paths = sorted((data / "selected-contexts").glob("context_*.json"))
    documents = DocumentStore(corpus_paths)
    valid_docs = {
        p.stem[len("context_"):]
        for p in corpus_paths
    }
    priv = reconstruct_private(root, ids)

    print("[4/6] Validating complete Prism private score cache...")
    raw_private_prism = load_score_dict(private_prism_path)
    missing = []
    total_pairs = 0
    for q in ids:
        if q not in raw_private_prism:
            missing.append((q, None))
            continue
        for d in priv["candidates"][q]:
            total_pairs += 1
            if d not in raw_private_prism[q] and len(missing) < 20:
                missing.append((q, d))
    if missing:
        raise RuntimeError(
            "Private Prism cache incomplete. First missing samples: "
            f"{missing[:20]}"
        )
    prism_private, prism_private_floor = align(
        raw_private_prism, ids, priv["candidates"]
    )

    print("  hard parity: reconstructing ordinary private D1...")
    baseline_rows, baseline_groups, _ = private_feature_bundle(
        root=root, documents=documents, ids=ids, questions=questions,
        priv=priv, prism_private=prism_private, arm="d1", floors=floors,
    )
    baseline_top5, _ = infer(
        baseline_rows, baseline_groups, ids, d1_scaler, d1_model
    )

    v14_path = (
        root
        / "results/manual/huy_private_d1_rel_l0_exact_v1/"
        "D1_PRIVATE_V14_FAST.json"
    )
    if not v14_path.is_file():
        raise FileNotFoundError(v14_path)
    v14 = json.loads(v14_path.read_text(encoding="utf-8"))
    mismatch = [
        q for q in ids
        if baseline_top5[q] != [str(d) for d in v14[q]["answer"]]
    ]
    if mismatch:
        raise RuntimeError(
            f"BLOCKED: cache-only D1 parity {len(ids)-len(mismatch)}/{len(ids)}. "
            f"Sample mismatches={mismatch[:5]}"
        )
    print("  D1 PRIVATE PARITY: 2080/2080 PASS")

    print(f"[5/6] Inferring private arm={args.arm}...")
    prism_rows, prism_groups, _ = private_feature_bundle(
        root=root, documents=documents, ids=ids, questions=questions,
        priv=priv, prism_private=prism_private, arm=args.arm, floors=floors,
    )
    prism_top5, prism_decision = infer(
        prism_rows, prism_groups, ids, prism_scaler, prism_model
    )

    churn_q = [q for q in ids if prism_top5[q] != baseline_top5[q]]
    set_churn_q = [
        q for q in ids if set(prism_top5[q]) != set(baseline_top5[q])
    ]
    entering = sum(
        len(set(prism_top5[q]) - set(baseline_top5[q]))
        for q in ids
    )
    leaving = sum(
        len(set(baseline_top5[q]) - set(prism_top5[q]))
        for q in ids
    )
    print(
        f"  ordered churn={len(churn_q)}/{len(ids)} "
        f"set churn={len(set_churn_q)}/{len(ids)} "
        f"enter={entering} leave={leaving}"
    )

    print("[6/6] Packaging separate Prism submission...")
    submission = {
        q: {"answer": [str(d) for d in prism_top5[q]]}
        for q in ids
    }
    for q in ids:
        ans = submission[q]["answer"]
        if (
            len(ans) != 5
            or len(set(ans)) != 5
            or any(d not in valid_docs for d in ans)
        ):
            raise RuntimeError(f"Invalid output q={q}: {ans}")

    out = root / "results/manual/huy_private_prism_v1"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.arm.upper()
    json_path = out / f"D1_PRIVATE_{tag}.json"
    zip_path = out / f"D1_PRIVATE_{tag}.zip"
    dump(json_path, submission)
    zip_exact(json_path, zip_path)

    score_cache = out / f"D1_PRIVATE_{tag}_DECISION_SCORES.pkl"
    score_cache.write_bytes(pickle.dumps(
        {"top5": prism_top5, "decision_scores": prism_decision},
        protocol=5,
    ))

    report = {
        "schema": "manual.private_prism_d1_v1",
        "status": "READY_FOR_PRIVATE_SUBMISSION",
        "arm": args.arm,
        "private_queries": len(ids),
        "baseline_private_parity": {
            "matches": len(ids), "expected": len(ids),
            "v14_json": str(v14_path), "v14_sha256": sha256(v14_path),
        },
        "prism": {
            "heldout_scores": str(cal_prism_path),
            "heldout_sha256": sha256(cal_prism_path),
            "private_scores": str(private_prism_path),
            "private_sha256": sha256(private_prism_path),
            "private_pairs": total_pairs,
            "cal_floor": prism_cal_floor,
            "private_floor": prism_private_floor,
        },
        "feature_dims": {"d1": d1_dim, "prism_arm": prism_dim},
        "churn_vs_v14": {
            "ordered_top5_queries": len(churn_q),
            "set_top5_queries": len(set_churn_q),
            "entering_docs": entering, "leaving_docs": leaving,
            "ordered_churn_qids": churn_q,
            "set_churn_qids": set_churn_q,
        },
        "submission": {
            "json": str(json_path), "json_sha256": sha256(json_path),
            "zip": str(zip_path), "zip_sha256": sha256(zip_path),
        },
        "scientific_contract": {
            "private_labels_used": False,
            "rel_l0": False,
            "old_neural_caches_recomputed": False,
            "candidate_pool_changed": False,
            "only_new_signal": "Prism private reranker score channel",
        },
    }
    report_path = out / f"REPORT_{tag}.json"
    dump(report_path, report)

    print("=" * 105)
    print("PRIVATE PRISM ARM READY")
    print("ARM:", args.arm)
    print("ZIP:", zip_path)
    print("SHA256:", sha256(zip_path))
    print("REPORT:", report_path)
    print("=" * 105)


if __name__ == "__main__":
    main()
