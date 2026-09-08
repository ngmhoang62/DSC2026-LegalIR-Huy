"""Isolated test submission: adds darklethelong/vnlegal-lal as an EXTRA score
channel (not a replacement) to the production 5-channel LTR fusion, matching
the configuration that passed CV validation (score_cv_vnlegal_lal.py):
Recall 0.9511->0.9528, F2 0.6123->0.6139, zero block regressions.

Does NOT touch the production results/burst_expanded_fusion/submission.zip --
writes to a separate output directory. Reuses production's cached
candidate-generation/rerank artifacts read-only (cache_dir), so no GPU work
is repeated for stages already computed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_dense_expansion_holdouts import raw_union
from benchmark_jina_reranker_holdouts import top_passages
from run_burst_expanded_fusion_submission import (CORPUS_CAP, CORPUS_DEPTH,
    DocumentStore, EXPANSION_CONFIG, RERANK_CONFIG, VIEWS,
    corpus_dense, dense_expansion, load_public_retrieval, rerank)
from run_burst_multistage_submission import load_metadata
from tune_burst_multistage_posterior import weighted_rrf
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_dense_fusion import build_training
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

MODEL_REPO = "darklethelong/vnlegal-lal"
MODEL_LOCAL = "models/vnlegal-lal"


def ensure_vnlegal_model(root: Path):
    path = root / MODEL_LOCAL
    if path.exists() and any(path.iterdir()):
        print(f"  {MODEL_REPO}: already present, skipping", flush=True)
        return
    from huggingface_hub import snapshot_download
    print(f"  {MODEL_REPO}: downloading...", flush=True)
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_REPO, local_dir=str(path), allow_patterns=[
        "*.json", "*.txt", "*.safetensors", "*.bin", "*.model", "tokenizer*",
        "vocab*", "merges*", "special_tokens_map*",
    ], ignore_patterns=["onnx/*", "openvino/*", "*.onnx", "*.sig"])
    print(f"  {MODEL_REPO}: done", flush=True)


def score_vnlegal(root, output, public, public_ids, candidates, documents, device):
    # Reverted to CLS pooling (the original approach) per explicit user
    # decision 2026-08-21: the "corrected" last-token pooling scored better
    # on CV (F2 0.6139->0.6253, same Recall) but regressed real Recall on
    # the actual CodaBench leaderboard -- a confirmed CV-vs-leaderboard
    # mismatch, see project_vnlegal_lal_success.md. CLS pooling's own cached
    # scores (vnlegal_scores.pkl) are reused as-is.
    path = output / "vnlegal_scores.pkl"
    saved = {}
    if path.exists():
        saved = pickle.loads(path.read_bytes())
    remaining = [q for q in public_ids
                 if any(d not in saved.get(q, {}) for d in candidates[q])]
    print(f"vnlegal-lal cache {len(public_ids)-len(remaining)}/{len(public_ids)}",
          flush=True)
    if remaining:
        tokenizer = AutoTokenizer.from_pretrained(root / MODEL_LOCAL)
        model = AutoModel.from_pretrained(
            root / MODEL_LOCAL, dtype=torch.float16, low_cpu_mem_usage=True
        ).eval().to(device)
        print(f"vnlegal-lal ready on {torch.cuda.get_device_name(0)}", flush=True)
        for i, q in enumerate(remaining, 1):
            text = public[q]
            qvec = encode_cls(model, tokenizer, [text], 1, 512)[0]
            owners, passages = [], []
            for d in candidates[q]:
                for p in top_passages(text, documents[d], count=2):
                    owners.append(d)
                    passages.append(p)
            ds = dict(saved.get(q, {}))
            if passages:
                pvec = encode_cls(model, tokenizer, passages, 32, 512)
                for d, s in zip(owners, pvec @ qvec):
                    ds[d] = max(ds.get(d, -1e9), float(s))
            saved[q] = ds
            if i % 50 == 0:
                path.write_bytes(pickle.dumps(saved, protocol=5))
                print(f"  {i}/{len(remaining)}", flush=True)
        path.write_bytes(pickle.dumps(saved, protocol=5))
        del model
        torch.cuda.empty_cache()
    return saved


def ltr_fusion_with_vnlegal(root, names, view_rank, candidates, public_ids,
                           public_scores, documents, public_queries,
                           vnlegal_public_scores, ltr_c=.15, drop=(),
                           crossenc=False, bge="", weight="",
                           extra_channels=()):
    holdout_vnlegal_scores = pickle.loads(
        (root / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl").read_bytes())

    queries, _, holdout_ids, holdout_candidates, holdout_views, training_scores = (
        build_training(root, depth=CORPUS_DEPTH, cap=CORPUS_CAP,
                       extended_scores_path=
                       "results/corpus_index/holdout_extended_scores_cap32.pkl"))

    names = names + ["vnlegal_lal"]
    training_scores = dict(training_scores)
    training_scores["vnlegal_lal"] = holdout_vnlegal_scores
    holdout_views = dict(holdout_views)
    holdout_views["vnlegal_lal"] = {
        q: sorted(holdout_candidates[q],
                 key=lambda d: (-holdout_vnlegal_scores.get(q, {}).get(d, -1e9), d))
        for q in holdout_ids}

    public_scores = dict(public_scores)
    public_scores["vnlegal_lal"] = vnlegal_public_scores

    # Optional 7th channel: the AITeamVN cross-encoder scored over the FULL
    # candidate pool. Enters as a score-only feature (no rank view), exactly
    # as validated on CV -- see score_cv_crossenc_channel.py.
    if crossenc:
        cv_ce = pickle.loads((root / "results/crossenc_fullpool/cv_scores.pkl")
                             .read_bytes())["scores"]
        pub_ce = pickle.loads((root / "results/crossenc_fullpool/public_scores.pkl")
                              .read_bytes())["scores"]
        floor = min(min(v.values()) for v in cv_ce.values() if v)
        miss_h = sum(1 for q in holdout_ids for d in holdout_candidates[q]
                     if d not in cv_ce.get(q, {}))
        miss_p = sum(1 for q in public_ids for d in candidates[q]
                     if d not in pub_ce.get(q, {}))
        print(f"crossenc channel: holdout {miss_h} / public {miss_p} slots "
              f"uncovered (filled with floor {floor:.3f})", flush=True)
        training_scores["crossenc"] = {
            q: {d: cv_ce.get(q, {}).get(d, floor) for d in holdout_candidates[q]}
            for q in holdout_ids}
        public_scores["crossenc"] = {
            q: {d: pub_ce.get(q, {}).get(d, floor) for d in candidates[q]}
            for q in public_ids}

    # Generic extra score channels: NAME=CV_PKL,PUBLIC_PKL. Missing slots are
    # filled with the channel's own minimum, matching the CV sweep exactly
    # (mean-filling was measured to lose 0.0025 of the title-embedding gain).
    for spec in extra_channels:
        name, _, paths = spec.partition("=")
        cv_rel, _, pub_rel = paths.partition(",")
        raw_cv = pickle.loads((root / cv_rel).read_bytes())
        raw_pub = pickle.loads((root / pub_rel).read_bytes())
        for obj in (raw_cv, raw_pub):
            pass
        if isinstance(raw_cv, dict) and isinstance(raw_cv.get("scores"), dict):
            raw_cv = raw_cv["scores"]
        if isinstance(raw_pub, dict) and isinstance(raw_pub.get("scores"), dict):
            raw_pub = raw_pub["scores"]
        fl = min(v for q in raw_cv for v in raw_cv[q].values())
        miss_h = sum(1 for q in holdout_ids for d in holdout_candidates[q]
                     if d not in raw_cv.get(q, {}))
        miss_p = sum(1 for q in public_ids for d in candidates[q]
                     if d not in raw_pub.get(q, {}))
        print(f"extra channel {name}: holdout {miss_h} / public {miss_p} slots "
              f"uncovered (filled with {fl:.4f})", flush=True)
        training_scores[name] = {
            q: {d: raw_cv.get(q, {}).get(d, fl) for d in holdout_candidates[q]}
            for q in holdout_ids}
        public_scores[name] = {
            q: {d: raw_pub.get(q, {}).get(d, fl) for d in candidates[q]}
            for q in public_ids}

    # Optional 8th channel: bge-reranker-v2-m3, the strongest single reranker
    # measured on this pool (0.8878 alone). Score-only, like crossenc.
    if bge:
        cv_path, pub_path = ("results/bge_rerank/cv_scores.pkl",
                             "results/bge_rerank/public_scores.pkl")
        if bge == "ft":
            cv_path = "results/bge_finetune/cv_scores_ft.pkl"
            pub_path = "results/bge_finetune/public_scores_ft.pkl"
        print(f"bge channel source: {cv_path}", flush=True)
        cv_bg = pickle.loads((root / cv_path).read_bytes())
        pub_bg = pickle.loads((root / pub_path).read_bytes())
        fl = min(min(v.values()) for v in cv_bg.values() if v)
        miss_p = sum(1 for q in public_ids for d in candidates[q]
                     if d not in pub_bg.get(q, {}))
        print(f"bge channel: public {miss_p} slots uncovered "
              f"(filled with floor {fl:.3f})", flush=True)
        training_scores["bge"] = {
            q: {d: cv_bg.get(q, {}).get(d, fl) for d in holdout_candidates[q]}
            for q in holdout_ids}
        public_scores["bge"] = {
            q: {d: pub_bg.get(q, {}).get(d, fl) for d in candidates[q]}
            for q in public_ids}
    view_rank = dict(view_rank)
    view_rank["vnlegal_lal"] = {
        q: sorted(candidates[q],
                 key=lambda d: (-vnlegal_public_scores.get(q, {}).get(d, -1e9), d))
        for q in public_ids}

    # Bo bot kenh de don gian hoa tang rerank. View "expanded" duoc nuoi boi
    # score channel ten "expansion", nen phai anh xa truoc khi loc score.
    if drop:
        score_key = {"expanded": "expansion"}
        names = [n for n in names if n not in drop]
        dropped_scores = {score_key.get(n, n) for n in drop}
        training_scores = {k: v for k, v in training_scores.items()
                           if k not in dropped_scores}
        public_scores = {k: v for k, v in public_scores.items()
                         if k not in dropped_scores}
        print(f"Da bo kenh: {sorted(drop)} -> con {names}", flush=True)

    rows, groups = ltr_features(holdout_views, names, holdout_candidates, holdout_ids,
                                training_scores)
    holdout_types = build_type_table(root, documents, holdout_ids, holdout_candidates)
    holdout_type_rows = type_features(holdout_candidates, holdout_types, queries,
                                      holdout_ids)
    holdout_own, holdout_cited = build_citation_table(documents, holdout_ids,
                                                       holdout_candidates)
    holdout_cite_rows = citation_features(holdout_candidates, holdout_own,
                                          holdout_cited, holdout_ids)
    for q in rows:
        rows[q] = np.concatenate([rows[q], holdout_type_rows[q],
                                  holdout_cite_rows[q]], axis=1)

    x = np.vstack([rows[q] for q in holdout_ids])
    y = np.concatenate([[d in queries[q][1] for d in groups[q]]
                        for q in holdout_ids]).astype(np.int8)
    scaler = StandardScaler().fit(x)
    # C=0.15: the original 5-channel system's default, kept unchanged here
    # per explicit user choice -- gives the best F2/precision (0.6139) of
    # all configs tried with vnlegal_lal added, at Recall=0.9511 (same as
    # the original 5-channel baseline; blocks b and d individually sit
    # slightly below their own baseline values, block c compensates above).
    model = LogisticRegression(C=ltr_c, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=2026)
    # per_query: every query contributes the same total mass regardless of how
    # many candidates it has. Measured on CV to recover the recall the 8th
    # channel otherwise costs (0.9503 -> 0.9536) while changing 305/600 answers.
    sample_weight = None
    if weight == "per_query":
        sample_weight = np.concatenate(
            [np.full(len(groups[q]), 1.0 / len(groups[q])) for q in holdout_ids])
        print(f"LTR sample weighting: per_query", flush=True)
    model.fit(scaler.transform(x), y, sample_weight=sample_weight)
    print(f"LTR trained on {len(holdout_ids)} queries, {len(y)} candidate rows "
          f"({int(y.sum())} positive), views={names}, C={ltr_c}", flush=True)

    public_rows, public_groups = ltr_features(view_rank, names, candidates, public_ids,
                                              public_scores)
    public_types = build_type_table(root, documents, public_ids, candidates)
    public_type_rows = type_features(candidates, public_types, public_queries,
                                     public_ids)
    public_own, public_cited = build_citation_table(documents, public_ids, candidates)
    public_cite_rows = citation_features(candidates, public_own, public_cited,
                                         public_ids)
    for q in public_rows:
        public_rows[q] = np.concatenate([public_rows[q], public_type_rows[q],
                                         public_cite_rows[q]], axis=1)

    fused = {}
    fused_proba = {}
    for q in public_ids:
        proba = model.predict_proba(scaler.transform(public_rows[q]))[:, 1]
        order = np.argsort(-proba)
        fused[q] = [public_groups[q][i] for i in order]
        fused_proba[q] = {public_groups[q][i]: float(proba[i]) for i in order}
    detail = {"model": f"LogisticRegression(C={ltr_c}, balanced)",
              "train_queries": len(holdout_ids),
              "features": "rank+score+doctype+citation+vnlegal_lal"
                          + ("+crossenc" if crossenc else "")
                          + (f"+bge_{bge}" if bge else "")
                          + "".join("+" + c.split("=")[0] for c in extra_channels),
              "coefficients": model.coef_[0].round(4).tolist()}
    return fused, fused_proba, detail


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=root / "DSC2026-LegalIR-main/v4_run/public_test_dataset")
    ap.add_argument("--db", type=Path,
                    default=root / "benchmarks/legalir_full_fts.sqlite")
    ap.add_argument("--cache-dir", type=Path,
                    default=root / "results/burst_expanded_fusion")
    ap.add_argument("--output-dir", type=Path,
                    default=root / "results/burst_vnlegal_extra_fusion")
    ap.add_argument("--workers", type=int, default=2)
    # Defaults reproduce the shipped submission byte-for-byte.
    ap.add_argument("--ltr-c", type=float, default=.15)
    ap.add_argument("--alpha", type=float, default=.15)
    ap.add_argument("--drop", default="",
                    help="ten kenh can bo, cach nhau bang dau phay")
    ap.add_argument("--crossenc", action="store_true",
                    help="them kenh cross-encoder full-pool (kenh thu 7)")
    ap.add_argument("--bge", default="", choices=["", "base", "ft"],
                    help="kenh bge-reranker-v2-m3 thu 8: base hoac fine-tuned")
    ap.add_argument("--weight", default="", choices=["", "per_query"],
                    help="trong so mau cho LTR")
    ap.add_argument("--extra-channel", action="append", default=[],
                    metavar="NAME=CV_PKL,PUBLIC_PKL",
                    help="them kenh diem tuy y, lap lai duoc")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths, doc_ids, train, public = load_metadata(args.data_dir)
    public_ids = list(public)
    valid = set(doc_ids)
    retrieval = load_public_retrieval(root, args, doc_ids, train, public, public_ids)

    base = pickle.loads(
        (root / "results/burst_gpu_threeview/cpu_top20.pkl").read_bytes())["rankings"]
    if any(q not in base for q in public_ids):
        raise RuntimeError("Missing CPU multistage candidates; run the three-view job")

    documents = DocumentStore(paths)
    print(f"Document store ready ({len(paths)} contexts, lazy)", flush=True)

    raw = {q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids}
    retrieval.clear()
    expansion_scores = dense_expansion(root, args.cache_dir, public, public_ids,
                                       raw, documents, device)

    dense_rank = {q: sorted(raw[q], key=lambda d: (-expansion_scores[q][d], d))
                  for q in public_ids}
    expanded = weighted_rrf([raw, dense_rank], RERANK_CONFIG["expansion_weights"],
                            RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank, corpus_score = corpus_dense(root, args.cache_dir, public,
                                             public_ids, device,
                                             cap=CORPUS_CAP, depth=CORPUS_DEPTH)
    candidates = {q: list(dict.fromkeys(
        list(base[q]) + expanded[q][:RERANK_CONFIG["expanded_depth"]] +
        corpus_rank[q][:CORPUS_DEPTH]))
        for q in public_ids}
    sizes = [len(candidates[q]) for q in public_ids]
    print(f"Candidates min={min(sizes)} mean={sum(sizes)/len(sizes):.1f} "
          f"max={max(sizes)}", flush=True)
    corpus_score = {q: {d: corpus_score[q][d] for d in candidates[q]
                        if d in corpus_score[q]} for q in public_ids}
    expansion_scores = {q: {d: expansion_scores[q][d] for d in candidates[q]
                            if d in expansion_scores[q]} for q in public_ids}
    expanded = {q: expanded[q][:RERANK_CONFIG["expanded_depth"]] for q in public_ids}

    scores = rerank(root, args.cache_dir, public, public_ids, candidates,
                    documents, device)

    print("\n=== ensure darklethelong/vnlegal-lal ===", flush=True)
    ensure_vnlegal_model(root)
    print("\n=== score public candidates with vnlegal-lal ===", flush=True)
    vnlegal_public_scores = score_vnlegal(root, args.output_dir, public, public_ids,
                                          candidates, documents, device)

    view_rank = {
        "base": {q: list(base[q]) for q in public_ids},
        "expanded": expanded,
        "jina": {q: sorted(candidates[q], key=lambda d: (-scores["jina"][q][d], d))
                 for q in public_ids},
        "dense": {q: sorted(candidates[q], key=lambda d: (-scores["dense"][q][d], d))
                  for q in public_ids},
        "corpus": {q: sorted((d for d in candidates[q] if d in corpus_score[q]),
                             key=lambda d: (-corpus_score[q][d], d))
                   for q in public_ids},
    }
    names = VIEWS
    three_view = pickle.loads(
        (root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl")
        .read_bytes())["scores"]
    public_scores = {
        "jina": scores["jina"], "dense": scores["dense"],
        "expansion": expansion_scores,
        "e5": {q: three_view[q]["e5"] for q in public_ids},
        "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in candidates[q]}
                   for q in public_ids},
    }
    public_queries = {q: (public[q], set()) for q in public_ids}
    fused, fused_proba, detail = ltr_fusion_with_vnlegal(
        root, names, view_rank, candidates, public_ids, public_scores, documents,
        public_queries, vnlegal_public_scores, ltr_c=args.ltr_c,
        drop={x.strip() for x in args.drop.split(',') if x.strip()},
        crossenc=args.crossenc, bge=args.bge, weight=args.weight,
        extra_channels=tuple(args.extra_channel))
    print(f"Fusion ltr: views={detail['features']}", flush=True)

    # alpha=0.15: the original 5-channel system's default, kept unchanged
    # per explicit user choice (highest F2/precision of the configs tried
    # with vnlegal_lal added: 0.6139, Recall=0.9511 -- same Recall as the
    # unmodified 5-channel baseline).
    THRESHOLD_ALPHA = args.alpha
    predictions = {}
    for q in public_ids:
        final = [d for d in fused[q] if d in valid][:5]
        for pool in (candidates[q], doc_ids):
            for doc in pool:
                if len(final) >= 5:
                    break
                if doc not in final and doc in valid:
                    final.append(doc)
        final = final[:5]
        if fused_proba is not None and final:
            p = [fused_proba[q].get(d, 0.0) for d in final]
            kept = [final[0]]
            for d, s in zip(final[1:], p[1:]):
                if s >= THRESHOLD_ALPHA * p[0]:
                    kept.append(d)
            final = kept
        predictions[q] = {"answer": final}

    if set(predictions) != set(public):
        raise RuntimeError("QID mismatch")
    for q, row in predictions.items():
        answers = row["answer"]
        if not (0 < len(answers) <= 5) or len(set(answers)) != len(answers) or any(
                d not in valid for d in answers):
            raise RuntimeError(f"Invalid prediction: {q}")

    out_json = args.output_dir / "submission.json"
    out_zip = args.output_dir / "submission.zip"
    out_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_json, arcname="submission.json")
    lengths = [len(row["answer"]) for row in predictions.values()]
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "expansion": EXPANSION_CONFIG, "rerank": RERANK_CONFIG,
        "fusion": "ltr", "views": detail["features"], "detail": detail,
        "queries": len(predictions),
        "documents_per_query": f"1-5 (dynamic threshold, alpha={THRESHOLD_ALPHA})",
        "mean_documents_per_query": sum(lengths) / len(lengths),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
