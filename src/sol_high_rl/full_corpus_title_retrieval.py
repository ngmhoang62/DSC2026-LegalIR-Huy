"""Round-1 full-corpus title retrieval with Huy's fine-tuned AITeamVN encoder."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, XLMRobertaConfig


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tune_corpus_cap32_fusion import build_training_cap  # noqa: E402
from tune_title_features import extract_title  # noqa: E402


OUT = ROOT / "results/sol_high_rl/title_retrieval_ft"
CACHE = ROOT / "cache/sol_high_rl/title_retrieval_ft"
MODEL_PATH = ROOT / "fine_tune/AITeamVN_Vietnamese_Embedding"
SLUG_ID_RE = re.compile(r"-\d+$")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def slug_title(link: str) -> str:
    if not link:
        return ""
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = SLUG_ID_RE.sub("", slug)
    return slug.replace("-", " ").strip()


@torch.inference_mode()
def encode(model, tokenizer, texts, batch_size=64, max_length=128):
    vectors = []
    for begin in range(0, len(texts), batch_size):
        batch = tokenizer(texts[begin:begin + batch_size], padding=True,
                          truncation=True, max_length=max_length,
                          return_tensors="pt")
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        cls = model(**batch).last_hidden_state[:, 0]
        vectors.append(F.normalize(cls.float(), p=2, dim=1).cpu().numpy())
        if (begin // batch_size + 1) % 25 == 0:
            print(f"encoded {min(begin + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(vectors)


def load_model():
    config = XLMRobertaConfig(
        vocab_size=250002, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096,
        max_position_embeddings=8194, type_vocab_size=1,
        pad_token_id=1, bos_token_id=0, eos_token_id=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(
        MODEL_PATH, config=config, dtype=torch.float16,
        low_cpu_mem_usage=True).eval().to("cuda")
    return model, tokenizer


def macro_oracle(queries, candidates, ids):
    return float(np.mean([
        len(set(candidates[q]) & queries[q][1]) / len(queries[q][1]) for q in ids
    ]))


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    queries, blocks, all_ids, current, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)

    title_meta_path = CACHE / "titles.json"
    if title_meta_path.exists():
        title_meta = json.loads(title_meta_path.read_text(encoding="utf-8"))
    else:
        title_meta = []
        paths = sorted((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
                       .glob("context_*.json"))
        for i, path in enumerate(paths, 1):
            row = json.loads(path.read_text(encoding="utf-8"))
            doc = str(row["id"])
            extracted = extract_title(row.get("passage") or "")
            fallback = slug_title(row.get("link") or "")
            title = extracted or fallback or "khong co tieu de"
            title_meta.append({"doc_id": doc, "title": title,
                               "source": "passage_subject" if extracted else
                                         "link_slug" if fallback else "placeholder"})
            if i % 2000 == 0:
                print(f"read titles {i}/{len(paths)}", flush=True)
        atomic_json(title_meta_path, title_meta)

    doc_ids = [row["doc_id"] for row in title_meta]
    title_texts = [row["title"] for row in title_meta]
    title_vec_path = CACHE / "title_vectors.f32.npy"
    query_vec_path = CACHE / "query_vectors.f32.npy"
    if title_vec_path.exists() and query_vec_path.exists():
        title_vectors = np.load(title_vec_path, mmap_mode="r")
        query_vectors = np.load(query_vec_path, mmap_mode="r")
    else:
        model, tokenizer = load_model()
        print(f"model ready on {torch.cuda.get_device_name(0)}", flush=True)
        title_vectors = encode(model, tokenizer, title_texts, batch_size=64, max_length=128)
        query_vectors = encode(model, tokenizer, [queries[q][0] for q in all_ids],
                               batch_size=64, max_length=128)
        np.save(title_vec_path, title_vectors.astype(np.float32))
        np.save(query_vec_path, query_vectors.astype(np.float32))
        del model
        torch.cuda.empty_cache()

    scores = np.asarray(query_vectors) @ np.asarray(title_vectors).T
    rankings = {}
    for row, q in enumerate(all_ids):
        part = np.argpartition(-scores[row], 20)[:20]
        order = sorted(part.tolist(), key=lambda i: (-float(scores[row, i]), doc_ids[i]))
        rankings[q] = [doc_ids[i] for i in order]
    atomic_json(OUT / "TITLE_TOP20_RANKINGS.json", rankings)

    all_gold = {(q, d) for q in all_ids for d in queries[q][1]}
    current_gold = {(q, d) for q, d in all_gold if d in current[q]}
    missing_gold = all_gold - current_gold
    report = {
        "status": "COMPLETE",
        "hypothesis": "full-corpus fine-tuned title embeddings add unique candidate information",
        "model": "fine_tune/AITeamVN_Vietnamese_Embedding",
        "contract": {"pooling": "CLS+L2", "query_prefix": "none",
                     "document_prefix": "none", "max_length": 128,
                     "title_policy": "passage subject declaration, else label-free URL slug"},
        "corpus_titles": len(doc_ids),
        "title_sources": {
            name: sum(row["source"] == name for row in title_meta)
            for name in ("passage_subject", "link_slug", "placeholder")
        },
        "current_candidate_oracle": macro_oracle(queries, current, all_ids),
        "depths": {},
        "runtime_seconds": time.perf_counter() - started,
        "max_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    for depth in (10, 20):
        source = {q: rankings[q][:depth] for q in all_ids}
        union = {q: list(dict.fromkeys(current[q] + source[q])) for q in all_ids}
        recovered = {(q, d) for q, d in missing_gold if d in source[q]}
        novel_counts = [sum(d not in current[q] for d in source[q]) for q in all_ids]
        per_block = {}
        for block, ids in blocks.items():
            miss_block = {(q, d) for q, d in missing_gold if q in set(ids)}
            per_block[block] = {
                "source_recall_at_depth": macro_oracle(queries, source, ids),
                "union_oracle": macro_oracle(queries, union, ids),
                "unique_missing_gold_occurrences_recovered": len(recovered & miss_block),
            }
        report["depths"][str(depth)] = {
            "source_recall": macro_oracle(queries, source, all_ids),
            "union_oracle": macro_oracle(queries, union, all_ids),
            "union_oracle_delta": macro_oracle(queries, union, all_ids) -
                                  macro_oracle(queries, current, all_ids),
            "unique_missing_gold_occurrences_recovered": len(recovered),
            "recovered": [{"qid": q, "doc_id": d} for q, d in sorted(recovered)],
            "mean_novel_documents": float(np.mean(novel_counts)),
            "total_novel_candidate_slots": int(sum(novel_counts)),
            "unique_gold_per_novel_slot": len(recovered) / max(sum(novel_counts), 1),
            "per_block": per_block,
        }
    atomic_json(OUT / "REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
