"""Title feature, take two: embedding similarity instead of raw token overlap.

Token-overlap on titles was noise-level (1 win / 0 losses out of 600).  Sibling
laws on adjacent topics often share generic legal vocabulary ("bao ve", "quyen",
"trach nhiem") that token overlap can't discriminate, but a sentence embedding
should place them at different distances from the question.  This scores the
question against each candidate's extracted title with the same Vietnamese
dense encoder already used elsewhere in the pipeline.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_dense_fusion import build_training
from tune_doctype_features import build_type_table, type_features
from tune_title_features import lobo, title_table


def main():
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, extended, local, scores = build_training(root, depth=20)
    names = ["base", "expanded", "jina", "dense", "corpus"]

    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    type_tab = build_type_table(root, docs, all_ids, extended)
    titles = title_table(docs, all_ids, extended)
    type_rows = type_features(extended, type_tab, queries, all_ids)

    cache_path = root / "results/burst_fresh_block/title_embed_scores.pkl"
    if cache_path.exists():
        title_sim = pickle.loads(cache_path.read_bytes())
        print(f"Loaded cached title similarity: {len(title_sim)} queries",
              flush=True)
    else:
        model_path = root / "models/AITeamVN_Vietnamese_Embedding"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(
            model_path, dtype=torch.float16).eval().to("cuda")
        print(f"Embedding titles on {torch.cuda.get_device_name(0)}", flush=True)

        doc_ids = list(titles)
        title_texts = [titles[d] or "khong co tieu de" for d in doc_ids]
        started = time.perf_counter()
        title_vecs = encode_cls(model, tokenizer, title_texts, 64, 128)
        print(f"Embedded {len(doc_ids)} titles in "
              f"{time.perf_counter()-started:.1f}s", flush=True)
        title_vec_map = {d: v for d, v in zip(doc_ids, title_vecs)}

        q_texts = [queries[q][0] for q in all_ids]
        q_vecs = encode_cls(model, tokenizer, q_texts, 32, 128)
        q_vec_map = {q: v for q, v in zip(all_ids, q_vecs)}
        del model
        torch.cuda.empty_cache()

        title_sim = {}
        for q in all_ids:
            qv = q_vec_map[q]
            title_sim[q] = {d: float(qv @ title_vec_map[d]) for d in extended[q]}
        cache_path.write_bytes(pickle.dumps(title_sim, protocol=5))
        print("Title similarity cached", flush=True)

    def title_sim_rows(candidates, ids):
        rows = {}
        for q in ids:
            sims = np.asarray([title_sim[q].get(d, 0.0) for d in candidates[q]],
                             dtype=np.float32)
            mean, std = sims.mean(), sims.std() or 1.0
            top = sims.max()
            z = (sims - mean) / std
            gap_to_top = (sims - top) / std
            has_title = np.asarray(
                [1.0 if titles.get(d) else 0.0 for d in candidates[q]],
                dtype=np.float32)
            rows[q] = np.stack([sims, z, gap_to_top, has_title], axis=1)
        return rows

    title_emb_rows = title_sim_rows(extended, all_ids)

    report = {}
    m0, f0 = lobo(names, local, extended, queries, blocks, scores)
    report["baseline_5view"] = m0
    print(f"baseline_5view          recall={m0['recall']:.4f}", flush=True)

    m1, f1 = lobo(names, local, extended, queries, blocks, scores, [type_rows])
    report["plus_doctype"] = m1
    print(f"plus_doctype             recall={m1['recall']:.4f}", flush=True)

    m2, f2 = lobo(names, local, extended, queries, blocks, scores,
                  [title_emb_rows])
    report["plus_title_embed"] = m2
    print(f"plus_title_embed         recall={m2['recall']:.4f}", flush=True)

    m3, f3 = lobo(names, local, extended, queries, blocks, scores,
                  [type_rows, title_emb_rows])
    report["plus_doctype_titleembed"] = m3
    print(f"plus_doctype_titleembed  recall={m3['recall']:.4f}", flush=True)

    for c in (.1, .3, 1.0):
        m, _ = lobo(names, local, extended, queries, blocks, scores,
                   [type_rows, title_emb_rows], c=c)
        report[f"combo_C{c}"] = m
        print(f"combo_C{c:<4}             recall={m['recall']:.4f} "
              f"precision={m['precision']:.4f}", flush=True)

    def hit(f, q):
        return len(set(f[q][:5]) & queries[q][1]) / len(queries[q][1])
    best_label = max(report, key=lambda k: report[k]["recall"])
    fmap = {"baseline_5view": f0, "plus_doctype": f1, "plus_title_embed": f2,
           "plus_doctype_titleembed": f3}
    print(f"\nBest: {best_label} recall={report[best_label]['recall']:.4f}",
          flush=True)
    if best_label in fmap:
        w = l = t = 0
        for q in all_ids:
            a, b = hit(f1, q), hit(fmap[best_label], q)
            if b > a:
                w += 1
            elif b < a:
                l += 1
            else:
                t += 1
        print(f"{best_label} vs plus_doctype: wins={w} losses={l} ties={t}",
              flush=True)

    path = root / "burst_title_embedding_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
