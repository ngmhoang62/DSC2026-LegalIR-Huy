#!/usr/bin/env python
"""
HUY PIPELINE AUDIT L — REGULARIZATION × SCORE-GEOMETRY INTERACTION V1
=====================================================================

CPU-only exact D1 LOBO.

Motivation:
Current per-channel score features are:
  z = (score - mean_q)/std_q
  topdist = (score - max_q)/std_q

Within a query, topdist = z - constant(query), so for a linear ranker the two
features are close to affine-redundant for ordering. Yet Z_ONLY @ C=.15 lost
0.00333 Recall. A plausible reason is that duplicating correlated score slopes
weakens effective L2 regularization.

Preregistered primary test:
  Z_ONLY C=.30  (2x baseline C=.15)

Diagnostics:
  Z_ONLY C=.20,.25,.35,.40
  FULL48 C=.12,.15,.18,.20

No public labels. The primary claim is about whether compact 38D can recover the
48D control after accounting for regularization.
"""

from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_R = 0.9569444444444444
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def loadp(root, rel):
    return pickle.loads((root / rel).read_bytes())


def align(raw, cand, ids, floor=None):
    if isinstance(raw, dict) and isinstance(raw.get("scores"), dict):
        raw = raw["scores"]
    if floor is None:
        vals = [v for q in raw.values() for v in q.values()]
        floor = min(vals) if vals else -1e9
    return {q: {d: float(raw.get(q, {}).get(d, floor)) for d in cand[q]} for q in ids}


def prepare(root):
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")
    ))
    queries, blocks, ids, cand, views, base_scores = build_training_cap(
        root, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    channels = {
        **base_scores,
        "vnlegal_lal": align(loadp(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"), cand, ids),
        "crossenc": align(loadp(root, "results/crossenc_fullpool/cv_scores.pkl"), cand, ids, -11.5),
        **{k: align(loadp(root, rel), cand, ids) for k, rel in EXTRA.items()},
    }

    tt = build_type_table(root, docs, ids, cand)
    tr = type_features(cand, tt, queries, ids)
    own, cited = build_citation_table(docs, ids, cand)
    cr = citation_features(cand, own, cited, ids)

    return queries, blocks, ids, cand, views, channels, gold, tr, cr


def make_rows(world, mode):
    queries, blocks, ids, cand, views, channels, gold, tr, cr = world
    rows, groups = {}, {}

    for q in ids:
        docs = cand[q]
        rms = [{d: i+1 for i, d in enumerate(views[name][q])} for name in D1_VIEWS]
        score_cols = []

        for name in sorted(channels):
            raw = channels[name].get(q, {})
            v = np.asarray([raw.get(d, np.nan) for d in docs], dtype=np.float64)
            p = v[~np.isnan(v)]
            mu = float(p.mean()) if p.size else 0.0
            sd = float(p.std()) if p.size else 1.0
            if sd <= 1e-12:
                sd = 1.0
            top = float(p.max()) if p.size else mu
            v = np.where(np.isnan(v), mu - 2*sd, v)
            z = (v - mu) / sd
            td = (v - top) / sd
            if mode == "FULL":
                score_cols.extend([z, td])
            elif mode == "Z_ONLY":
                score_cols.append(z)
            else:
                raise ValueError(mode)

        rr = []
        for i, d in enumerate(docs):
            r = [rm.get(d, 60) for rm in rms]
            x = [1/(10+x) for x in r] + [x/60 for x in r] + [float(min(r)), float(np.mean(r))]
            x += [float(col[i]) for col in score_cols]
            x += tr[q][i].tolist()
            x += cr[q][i].tolist()
            rr.append(x)
        rows[q] = np.asarray(rr, dtype=np.float32)
        groups[q] = docs

    return rows, groups


def run(world, mode, C):
    queries, blocks, ids, cand, views, channels, gold, tr, cr = world
    rows, groups = make_rows(world, mode)
    pred, perq = {}, {}

    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([[d in gold[q] for d in groups[q]] for q in train]).astype(np.int8)
        sc = StandardScaler().fit(X)
        m = LogisticRegression(
            C=C, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=2026
        )
        m.fit(sc.transform(X), y)
        for q in blocks[held]:
            s = m.decision_function(sc.transform(rows[q]))
            order = np.argsort(-s, kind="stable")
            pred[q] = [groups[q][i] for i in order[:5]]

    for q in ids:
        perq[q] = len(set(pred[q]) & gold[q]) / len(gold[q])

    return {
        "mode": mode, "C": C, "dim": int(rows[ids[0]].shape[1]),
        "recall": float(np.mean([perq[q] for q in ids])),
        "precision": float(np.mean([len(set(pred[q]) & gold[q])/5 for q in ids])),
        "blocks": {b: float(np.mean([perq[q] for q in blocks[b]])) for b in sorted(blocks)},
        "pred": pred, "perq": perq,
    }


def cmp(base, x, ids):
    d = np.asarray([x["perq"][q]-base["perq"][q] for q in ids])
    return {
        "dR": x["recall"]-base["recall"],
        "dP": x["precision"]-base["precision"],
        "W": int((d>1e-12).sum()), "L": int((d<-1e-12).sum()),
        "T": int((np.abs(d)<=1e-12).sum()),
        "blocks": {b: x["blocks"][b]-base["blocks"][b] for b in base["blocks"]},
        "exact_top5": int(sum(x["pred"][q]==base["pred"][q] for q in ids)),
    }


def slim(r):
    return {k:v for k,v in r.items() if k not in ("pred","perq")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    print("[1/3] Loading D1 world...", flush=True)
    w = prepare(root)
    ids = w[2]

    specs = [
        ("FULL", .15, "CONTROL"),
        ("FULL", .12, "FULL_C012"),
        ("FULL", .18, "FULL_C018"),
        ("FULL", .20, "FULL_C020"),
        ("Z_ONLY", .20, "Z_C020"),
        ("Z_ONLY", .25, "Z_C025"),
        ("Z_ONLY", .30, "Z_C030_PRIMARY"),
        ("Z_ONLY", .35, "Z_C035"),
        ("Z_ONLY", .40, "Z_C040"),
    ]

    print("[2/3] Running fixed regularization interaction grid...", flush=True)
    results = {}
    for mode, C, name in specs:
        r = run(w, mode, C)
        results[name] = r
        print(f"  {name:<16s} dim={r['dim']:2d} R={r['recall']:.10f} P={r['precision']:.10f}", flush=True)

    base = results["CONTROL"]
    if abs(base["recall"]-EXPECTED_R) > 1e-12 or base["dim"] != 48:
        raise RuntimeError(f"Control parity failed: {base['dim']}D R={base['recall']}")

    rows = []
    for name, r in results.items():
        if name == "CONTROL":
            continue
        c = cmp(base, r, ids)
        rows.append({"name": name, **slim(r), "comparison": c})

    primary = next(x for x in rows if x["name"]=="Z_C030_PRIMARY")
    pc = primary["comparison"]
    primary_pass = (
        pc["dR"] >= -1e-12
        and all(v >= -1e-12 for v in pc["blocks"].values())
        and (pc["W"] > pc["L"] or pc["exact_top5"] == len(ids))
    )

    report = {
        "schema":"manual.regularization_score_geometry_interaction_v1",
        "control":slim(base),
        "variants":rows,
        "primary_hypothesis":{
            "variant":"Z_C030_PRIMARY",
            "rationale":"2x C compensates approximately for removing paired affine-redundant score slope",
            "pass":bool(primary_pass),
        },
        "public_labels_used":False,
        "multiple_comparison_note":"neighbors are stability diagnostics; primary test was Z_ONLY C=.30",
    }

    out = root/"results/manual/huy_regularization_score_geometry_interaction_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out/"REPORT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[3/3] RESULT")
    print("="*112)
    print(f"CONTROL 48D C=.15 R={base['recall']:.10f} P={base['precision']:.10f}")
    for x in rows:
        c=x["comparison"]
        print(
            f"{x['name']:<16s} dim={x['dim']:2d} dR={c['dR']:+.10f} "
            f"dP={c['dP']:+.10f} W/L/T={c['W']}/{c['L']}/{c['T']} "
            f"blocks={c['blocks']}"
        )
    print("PRIMARY_PASS:", primary_pass)
    print("Report:", path)
    print("="*112)


if __name__ == "__main__":
    main()
