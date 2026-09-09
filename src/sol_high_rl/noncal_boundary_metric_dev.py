"""F3: fixed non-CAL diagonal boundary metric and DEV-only CAL evaluation."""
from __future__ import annotations
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from forensic_world_model import metrics, prepare_contract  # noqa: E402
from full_corpus_title_retrieval import encode, load_model  # noqa: E402
from tune_burst_phrases import multi_rrf  # noqa: E402
from tune_expanded_fusion_selection import ltr_features  # noqa: E402

OUT = ROOT / "results/sol_high_rl"
CACHE = ROOT / "cache/sol_high_rl/aiteam_ft_full_corpus"
DEV_FOLDS = ("fold_0", "fold_1", "fold_2")
TRAIN_QV = CACHE / "noncal1050_query_vectors_max512.f32.npy"
SCORE_CACHE = CACHE / "cal600_noncal_boundary_diag_scores.pkl"
MODEL_CACHE = CACHE / "noncal_boundary_diag_model.npz"


def paired(candidate, baseline, queries, ids):
    cv = {q: len(set(candidate[q][:5]) & queries[q][1]) / len(queries[q][1]) for q in ids}
    bv = {q: len(set(baseline[q][:5]) & queries[q][1]) / len(queries[q][1]) for q in ids}
    return {"wins": sum(cv[q] > bv[q] for q in ids), "losses": sum(cv[q] < bv[q] for q in ids),
            "ties": sum(cv[q] == bv[q] for q in ids),
            "changed_top5_sets": sum(set(candidate[q][:5]) != set(baseline[q][:5]) for q in ids)}


def load_index():
    meta = json.loads((CACHE / "chunks_cap32.json").read_text(encoding="utf-8"))
    offsets, cursor = {}, 0
    for d, count in zip(meta["documents"], meta["counts"]):
        offsets[d] = (cursor, cursor + count); cursor += count
    vectors = np.memmap(CACHE / "chunks_cap32.f16", mode="r", dtype=np.float16, shape=(cursor, 1024))
    return offsets, vectors


def best_vector(qvec, doc, offsets, vectors, weighted_q=None):
    begin, end = offsets[doc]
    block = np.asarray(vectors[begin:end], dtype=np.float32)
    target = qvec if weighted_q is None else weighted_q
    return block[int(np.argmax(block @ target))]


def train_metric(offsets, vectors):
    if MODEL_CACHE.exists():
        z = np.load(MODEL_CACHE)
        return z["weight"], json.loads(str(z["diagnostics"].item()))
    raw = json.loads((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json").read_text(encoding="utf-8"))
    queries = {str(q): (x["question"], {str(d) for d in x["answer"]}) for q, x in raw.items() if x.get("answer")}
    feasible = json.loads((OUT / "BOUNDARY_SPECIALIST_FEASIBILITY.json").read_text(encoding="utf-8"))
    train_ids = [r["qid"] for r in feasible["rows"]]
    cache = pickle.loads((ROOT / "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl").read_bytes())["cache"]
    if TRAIN_QV.exists():
        qvectors = np.load(TRAIN_QV)
        if qvectors.shape != (len(train_ids), 1024): raise RuntimeError("non-CAL qvector cache mismatch")
    else:
        model, tok = load_model()
        qvectors = encode(model, tok, [queries[q][0] for q in train_ids], batch_size=48, max_length=512).astype(np.float32)
        np.save(TRAIN_QV, qvectors)
        del model, tok; gc.collect(); torch.cuda.empty_cache()
    diffs, sample_weights = [], []
    used_pairs = 0
    for qi, q in enumerate(train_ids):
        ranking = multi_rrf([[d for d, _ in source] for source in cache[q]], [.063, .357, .28, .30], 5)
        positives = [d for d in ranking if d in queries[q][1] and d in offsets]
        negatives = [d for d in ranking[3:20] if d not in queries[q][1] and d in offsets]
        if not positives or not negatives: continue
        qv = qvectors[qi]
        pv = {d: best_vector(qv, d, offsets, vectors) for d in positives}
        nv = {d: best_vector(qv, d, offsets, vectors) for d in negatives}
        w = 1.0 / (2 * len(positives) * len(negatives))
        for p in positives:
            for n in negatives:
                diff = qv * (pv[p] - nv[n])
                diffs.extend((diff, -diff)); sample_weights.extend((w, w)); used_pairs += 1
        if (qi + 1) % 200 == 0: print(f"metric examples {qi+1}/{len(train_ids)}", flush=True)
    x = np.asarray(diffs, dtype=np.float32)
    y = np.tile(np.asarray([1, 0], dtype=np.int8), used_pairs)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.03, solver="liblinear", max_iter=2000, random_state=2026)
    model.fit(scaler.transform(x), y, sample_weight=np.asarray(sample_weights, dtype=np.float64))
    weight = (model.coef_[0] / scaler.scale_).astype(np.float32)
    pred = model.predict(scaler.transform(x))
    diagnostics = {"noncal_queries": len(train_ids), "unordered_pairs": used_pairs,
                   "symmetric_rows": len(y), "in_sample_pair_accuracy_diagnostic_only": float(np.mean(pred == y)),
                   "weight_min": float(weight.min()), "weight_max": float(weight.max()),
                   "weight_l2": float(np.linalg.norm(weight)), "C": .03}
    np.savez(MODEL_CACHE, weight=weight, diagnostics=json.dumps(diagnostics))
    return weight, diagnostics


def score_cal(weight, offsets, vectors, candidates, all_ids):
    if SCORE_CACHE.exists(): return pickle.loads(SCORE_CACHE.read_bytes())
    qvectors = np.load(CACHE / "cal600_query_vectors_max512.f32.npy")
    result = {}
    for qi, q in enumerate(all_ids):
        weighted_q = qvectors[qi] * weight
        row = {}
        for d in candidates[q]:
            begin, end = offsets[d]
            row[d] = float(np.max(np.asarray(vectors[begin:end], dtype=np.float32) @ weighted_q))
        result[q] = row
        if (qi + 1) % 100 == 0: print(f"metric CAL scores {qi+1}/600", flush=True)
    SCORE_CACHE.write_bytes(pickle.dumps(result, protocol=5))
    return result


def main():
    started = time.perf_counter()
    split_bytes = (OUT / "CAL600_STRATIFIED_5FOLD_SEED42.json").read_bytes(); split = json.loads(split_bytes)
    base_report = json.loads((OUT / "CAL600_CANONICAL_BASELINE_REPORT.json").read_text(encoding="utf-8"))
    if hashlib.sha256(split_bytes).hexdigest() != base_report["protocol"]["split_sha256"]: raise RuntimeError("split mismatch")
    baseline = json.loads((OUT / "CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    queries, _, all_ids, candidates, views, scores, names, _, extras, _ = prepare_contract()
    cbytes = (json.dumps({q: candidates[q] for q in all_ids}, ensure_ascii=False, indent=2) + "\n").encode()
    if hashlib.sha256(cbytes).hexdigest() != base_report["candidate_contract_sha256"]: raise RuntimeError("candidate mismatch")
    offsets, vectors = load_index(); weight, training = train_metric(offsets, vectors)
    specialist = score_cal(weight, offsets, vectors, candidates, all_ids)
    new_views = dict(views); new_scores = dict(scores); new_names = list(names) + ["noncal_boundary_diag"]
    new_scores["noncal_boundary_diag"] = specialist
    new_views["noncal_boundary_diag"] = {q: sorted(candidates[q], key=lambda d: (-specialist[q][d], d)) for q in all_ids}
    rows, groups = ltr_features(new_views, new_names, candidates, all_ids, new_scores)
    for extra in extras:
        for q in all_ids: rows[q] = np.concatenate([rows[q], extra[q]], axis=1)
    pred = {}
    for fold in DEV_FOLDS:
        test = split["folds"][fold]; held = set(test); train = [q for q in all_ids if q not in held]
        x = np.vstack([rows[q] for q in train]); y = np.concatenate([[d in queries[q][1] for d in groups[q]] for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
        model.fit(scaler.transform(x), y)
        for q in test:
            s = model.predict_proba(scaler.transform(rows[q]))[:, 1]
            pred[q] = [groups[q][i] for i in np.argsort(-s, kind="stable")]
    dev_ids = sum((split["folds"][f] for f in DEV_FOLDS), [])
    bm = metrics(baseline, queries, dev_ids)[0]; cm = metrics(pred, queries, dev_ids)[0]
    multi = [q for q in dev_ids if len(queries[q][1]) > 1]
    bmulti = metrics(baseline, queries, multi)[0]; cmulti = metrics(pred, queries, multi)[0]
    folds = {}
    for f in DEV_FOLDS:
        ids = split["folds"][f]; b = metrics(baseline, queries, ids)[0]; c = metrics(pred, queries, ids)[0]
        folds[f] = {"baseline": b, "candidate": c, "delta": c["recall_at_5"] - b["recall_at_5"], "paired": paired(pred, baseline, queries, ids)}
    delta = cm["recall_at_5"] - bm["recall_at_5"]; pp = paired(pred, baseline, queries, dev_ids)
    md = cmulti["recall_at_5"] - bmulti["recall_at_5"]
    promote = delta >= .005 and sum(x["delta"] > 0 for x in folds.values()) >= 2 and pp["wins"] > pp["losses"] and md >= -.02
    standalone = {q: sorted(candidates[q], key=lambda d: (-specialist[q][d], d)) for q in dev_ids}
    report = {"status": "PROMOTE_TO_CONFIRMATION" if promote else "REJECT", "family": "F3_noncal_boundary_diagonal_metric",
              "protocol": {"development_folds": list(DEV_FOLDS), "confirmation_outcomes_inspected": False,
                           "noncal_training_overlap": 0, "split_sha256": base_report["protocol"]["split_sha256"],
                           "candidate_contract_sha256": base_report["candidate_contract_sha256"]},
              "training": training, "standalone_dev": metrics(standalone, queries, dev_ids)[0],
              "baseline_dev": bm, "candidate_dev": cm, "dev_delta": delta, "paired": pp, "folds": folds,
              "multi_gold": {"baseline": bmulti, "candidate": cmulti, "delta": md},
              "promotion_gate_passed": promote, "runtime_seconds": time.perf_counter() - started}
    (OUT / "F3_NONCAL_BOUNDARY_METRIC_DEV_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "F3_NONCAL_BOUNDARY_METRIC_DEV_PREDICTIONS.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "training": training, "standalone": report["standalone_dev"],
                      "delta": delta, "paired": pp, "fold_deltas": {f: x["delta"] for f, x in folds.items()},
                      "multi_delta": md, "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
