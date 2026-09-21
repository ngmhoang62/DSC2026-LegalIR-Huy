#!/usr/bin/env python
"""
CAL600 Adapted-E5 Full-Corpus Complementarity V1
=================================================

High-EV source diagnostic for DSC2026 Legal IR.

Scientific question:
    Does strict out-of-fold adapted VietLegal-E5 full-corpus retrieval recover
    CAL600 gold documents outside the exact D1 candidate pool, especially golds
    not already recovered by the existing AITeamVN-FT full-corpus source?

Mechanism:
  - Exact D1 CAL600 population/pool/gold from the existing loader.
  - Exact Research-V2 E5 chunk bank: 343,347 chunks -> 8,507 parents.
  - Exact EXP-112 query encoder / ParentBank top2_mean scoring.
  - For every CAL query, use ONLY the adapter whose V2 fold held that query out.
  - Full-corpus ranking top150, deterministic score DESC + doc_id ASC.
  - Seal E5 rankings BEFORE reading gold utility.
  - Then compare:
        current D1 candidate ceiling
        + adapted-E5 top20/top50
        + existing AITeamVN-FT top20/top50
        + union(E5, AITeam) top20/top50
    and enumerate outside-pool gold rescues / overlaps.

No CAL labels affect retrieval, adapter choice, depth, or ranking.

Run:
    python ../run_cal600_adapted_e5_full_corpus_v1.py \
      --repo-root /d/Study/DSC2026/sota \
      --batch-size 16

Resumability:
    Each fold is saved independently. Re-running skips complete fold files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import torch


TOPK_SAVE = 150
DEPTHS = (5, 10, 20, 50)
FOLDS = tuple(f"fold_{i}" for i in range(5))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidate_oracle(
    candidates: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    ids: List[str],
) -> float:
    return float(
        np.mean(
            [
                len(set(candidates[q]) & gold[q]) / len(gold[q])
                for q in ids
            ]
        )
    )


def standalone_recall(
    rankings: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    ids: List[str],
    depth: int,
) -> float:
    return float(
        np.mean(
            [
                len(set(rankings[q][:depth]) & gold[q]) / len(gold[q])
                for q in ids
            ]
        )
    )


def union_candidates(
    current: Dict[str, List[str]],
    *sources: Dict[str, List[str]],
    depth: int,
) -> Dict[str, List[str]]:
    out = {}
    for q in current:
        row = list(current[q])
        seen = set(row)
        for source in sources:
            for d in source[q][:depth]:
                if d not in seen:
                    row.append(d)
                    seen.add(d)
        out[q] = row
    return out


def lexical_rank(doc_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.lexsort((doc_ids, -scores))


def checkpoint_for(root: Path, fold: str) -> Path:
    idx = int(fold.rsplit("_", 1)[1])
    if idx == 0:
        return (
            root
            / "results/research_v2_e5_transfer/"
            "research_v2_e5_transfer_fold0/training/epoch-2.pt"
        )
    return (
        root
        / f"results/research_v2_e5_confirmation/"
        f"fold_{idx}/training/epoch-2.pt"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required. Verify CUDA PyTorch inside dsc_env_huy."
        )

    out = root / "results/manual/huy_cal600_adapted_e5_full_corpus_v1"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading exact D1 CAL600 state...", flush=True)
    from src.gemini.huy_d1_legal_section_evidence_v1.common import load_cal_data

    (
        _docs,
        queries,
        blocks,
        all_ids,
        current_pool,
        _local_views,
        _channels,
        gold,
        _type_rows,
        _cite_rows,
    ) = load_cal_data()

    all_ids = [str(q) for q in all_ids]
    current_pool = {str(q): [str(d) for d in ds] for q, ds in current_pool.items()}
    gold = {str(q): {str(d) for d in ds} for q, ds in gold.items()}
    queries = {str(q): v for q, v in queries.items()}
    blocks = {str(k).upper(): [str(q) for q in v] for k, v in blocks.items()}

    if len(all_ids) != 600:
        raise RuntimeError(f"Expected CAL600, got {len(all_ids)} queries")

    print("[2/7] Loading Research-V2 bank contract and fold map...", flush=True)
    from research_v2_e5_transfer import e5_transfer_runner as core

    bundle = root / "cache/research_v2_e5_confirmation/bundle-v1"
    data = core.TransferData(bundle)

    folds_path = bundle / "V2_FOLDS.json"
    folds_obj = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_map = {
        str(qid): fold
        for fold, qids in folds_obj["folds"].items()
        for qid in qids
    }
    missing_fold = [q for q in all_ids if q not in fold_map]
    if missing_fold:
        raise RuntimeError(
            f"{len(missing_fold)} CAL qids missing V2 fold assignment; "
            f"sample={missing_fold[:10]}"
        )

    checkpoint_hashes = {}
    for fold in FOLDS:
        ckpt = checkpoint_for(root, fold)
        if not ckpt.is_file():
            raise RuntimeError(f"Missing adapter checkpoint: {ckpt}")
        checkpoint_hashes[fold] = sha256_file(ckpt)

    contract = {
        "schema": "manual.cal600_adapted_e5_full_corpus_v1.contract",
        "cal_queries": len(all_ids),
        "v2_parents": len(data.doc_ids),
        "v2_chunks": len(data.chunk_ids),
        "parent_aggregation": "top2_mean (top1 for singleton)",
        "ranking": "score_desc_then_doc_id_asc",
        "topk_saved": TOPK_SAVE,
        "anti_leakage": "CAL qid uses adapter trained with its assigned V2 fold held out",
        "folds_sha256": sha256_file(folds_path),
        "bank_sha256": sha256_file(bundle / "embeddings.f16.npy"),
        "checkpoint_sha256": checkpoint_hashes,
        "batch_size_execution_only": args.batch_size,
    }
    write_json(out / "SOURCE_CONTRACT.json", contract)

    print(
        f"  bank={len(data.chunk_ids)} chunks / {len(data.doc_ids)} parents; "
        f"CAL fold distribution="
        + str({f: sum(fold_map[q] == f for q in all_ids) for f in FOLDS}),
        flush=True,
    )

    print("[3/7] Building exact full-corpus ParentBank on GPU...", flush=True)
    bank = core.ParentBank(data.vectors, data.parent)
    doc_ids = np.asarray(data.doc_ids, dtype=str)
    print(
        f"  ParentBank ready; CUDA allocated="
        f"{torch.cuda.memory_allocated()/2**20:.0f} MiB",
        flush=True,
    )

    print("[4/7] Scoring CAL600 full corpus with strict OOF adapted E5...", flush=True)
    fold_groups = {
        fold: [q for q in all_ids if fold_map[q] == fold]
        for fold in FOLDS
    }

    for fold in FOLDS:
        fold_dir = out / fold
        pred_path = fold_dir / "FULL_CORPUS_TOP150.jsonl"
        report_path = fold_dir / "REPORT.json"

        if pred_path.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("status") == "COMPLETE"
                and report.get("predictions_sha256") == sha256_file(pred_path)
                and report.get("checkpoint_sha256") == checkpoint_hashes[fold]
            ):
                print(f"  {fold}: valid completed cache -> SKIP", flush=True)
                continue
            raise RuntimeError(
                f"{fold}: existing artifacts fail provenance; remove fold dir manually"
            )

        qids = fold_groups[fold]
        ckpt = checkpoint_for(root, fold)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        print(
            f"  {fold}: {len(qids)} CAL queries; loading held-out adapter...",
            flush=True,
        )
        model = core.QueryEncoder(
            bundle / "vietlegal-e5",
            checkpoint_path=ckpt,
        )
        model.eval()

        rows = []
        try:
            for start in range(0, len(qids), args.batch_size):
                local_qids = qids[start : start + args.batch_size]
                texts = [queries[q][0] for q in local_qids]

                with torch.inference_mode():
                    qvecs = model(texts)
                    parent_scores, _ = bank.mine(qvecs)

                values = parent_scores.detach().cpu().numpy()

                for qid, score_row in zip(local_qids, values):
                    order_idx = lexical_rank(doc_ids, score_row)
                    top_idx = order_idx[:TOPK_SAVE]
                    rows.append(
                        {
                            "qid": qid,
                            "fold": fold,
                            "order_top150": doc_ids[top_idx].tolist(),
                            "scores_top150": (
                                score_row[top_idx].astype(float).tolist()
                            ),
                        }
                    )

                completed = min(start + len(local_qids), len(qids))
                if (
                    completed % max(args.batch_size * 5, 1) == 0
                    or completed == len(qids)
                ):
                    print(
                        f"    {fold}: {completed}/{len(qids)} | "
                        f"peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB",
                        flush=True,
                    )
        finally:
            peak = float(torch.cuda.max_memory_allocated() / 2**20)
            del model
            gc.collect()
            torch.cuda.empty_cache()

        if len(rows) != len(qids):
            raise RuntimeError(
                f"{fold}: scored {len(rows)} rows, expected {len(qids)}"
            )

        write_jsonl(pred_path, rows)
        fold_report = {
            "status": "COMPLETE",
            "fold": fold,
            "queries": len(rows),
            "checkpoint": str(ckpt),
            "checkpoint_sha256": checkpoint_hashes[fold],
            "runtime_seconds": time.perf_counter() - started,
            "peak_allocated_mib": peak,
            "predictions_sha256": sha256_file(pred_path),
        }
        write_json(report_path, fold_report)
        print(
            f"  {fold}: COMPLETE in {fold_report['runtime_seconds']/60:.1f} min",
            flush=True,
        )

    print("[5/7] Sealing adapted-E5 full-corpus rankings...", flush=True)
    e5_rankings: Dict[str, List[str]] = {}
    fold_reports = {}
    pred_hashes = {}

    for fold in FOLDS:
        fold_dir = out / fold
        pred_path = fold_dir / "FULL_CORPUS_TOP150.jsonl"
        report_path = fold_dir / "REPORT.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "COMPLETE"
            or report.get("predictions_sha256") != sha256_file(pred_path)
        ):
            raise RuntimeError(f"Invalid completed fold artifact: {fold}")
        fold_reports[fold] = report
        pred_hashes[fold] = sha256_file(pred_path)

        for row in read_jsonl(pred_path):
            q = str(row["qid"])
            if q in e5_rankings:
                raise RuntimeError(f"Duplicate qid across folds: {q}")
            e5_rankings[q] = [str(d) for d in row["order_top150"]]

    if set(e5_rankings) != set(all_ids):
        raise RuntimeError(
            f"E5 population mismatch: got={len(e5_rankings)} expected=600"
        )

    seal = {
        "schema": "manual.cal600_adapted_e5_full_corpus_v1.seal",
        "contract_sha256": sha256_file(out / "SOURCE_CONTRACT.json"),
        "fold_prediction_sha256": pred_hashes,
        "query_count": len(e5_rankings),
        "topk": TOPK_SAVE,
        "NOTE": "Gold utility is evaluated only after this source seal.",
    }
    write_json(out / "LABEL_FREE_SOURCE_SEAL.json", seal)
    write_json(out / "CAL600_ADAPTED_E5_FULL_CORPUS_TOP150.json", e5_rankings)

    print("[6/7] Loading sealed AITeamVN-FT full-corpus Top50...", flush=True)
    aiteam_path = root / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_TOP50.json"
    aiteam_report_path = root / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_REPORT.json"
    if not aiteam_path.is_file() or not aiteam_report_path.is_file():
        raise RuntimeError(
            "Missing existing AITeamVN-FT full-corpus artifacts:\n"
            f"  {aiteam_path}\n  {aiteam_report_path}"
        )
    aiteam = json.loads(aiteam_path.read_text(encoding="utf-8"))
    aiteam = {
        str(q): [str(d) for d in row]
        for q, row in aiteam.items()
    }
    if set(aiteam) != set(all_ids):
        raise RuntimeError(
            f"AITeam CAL population mismatch: {len(aiteam)} vs 600"
        )

    print("[7/7] Evaluating candidate-source complementarity...", flush=True)

    current_ceiling = candidate_oracle(current_pool, gold, all_ids)

    outside_pairs = {
        (q, d)
        for q in all_ids
        for d in gold[q]
        if d not in set(current_pool[q])
    }

    depth_reports = {}
    for depth in DEPTHS:
        e5_union = union_candidates(
            current_pool,
            e5_rankings,
            depth=depth,
        )
        ai_union = union_candidates(
            current_pool,
            aiteam,
            depth=depth,
        )
        both_union = union_candidates(
            current_pool,
            e5_rankings,
            aiteam,
            depth=depth,
        )

        e5_rescues = {
            (q, d)
            for q, d in outside_pairs
            if d in set(e5_rankings[q][:depth])
        }
        ai_rescues = {
            (q, d)
            for q, d in outside_pairs
            if d in set(aiteam[q][:depth])
        }
        both_rescues = e5_rescues | ai_rescues
        overlap = e5_rescues & ai_rescues

        e5_only = e5_rescues - ai_rescues
        ai_only = ai_rescues - e5_rescues

        depth_reports[str(depth)] = {
            "adapted_e5_standalone_recall": standalone_recall(
                e5_rankings, gold, all_ids, depth
            ),
            "aiteam_standalone_recall": standalone_recall(
                aiteam, gold, all_ids, min(depth, 50)
            ),
            "current_candidate_ceiling": current_ceiling,
            "current_plus_e5_ceiling": candidate_oracle(
                e5_union, gold, all_ids
            ),
            "current_plus_e5_delta": (
                candidate_oracle(e5_union, gold, all_ids)
                - current_ceiling
            ),
            "current_plus_aiteam_ceiling": candidate_oracle(
                ai_union, gold, all_ids
            ),
            "current_plus_aiteam_delta": (
                candidate_oracle(ai_union, gold, all_ids)
                - current_ceiling
            ),
            "current_plus_union_ceiling": candidate_oracle(
                both_union, gold, all_ids
            ),
            "current_plus_union_delta": (
                candidate_oracle(both_union, gold, all_ids)
                - current_ceiling
            ),
            "outside_pool_gold_occurrences_total": len(outside_pairs),
            "e5_rescues": len(e5_rescues),
            "aiteam_rescues": len(ai_rescues),
            "union_rescues": len(both_rescues),
            "overlap_rescues": len(overlap),
            "e5_only_rescues": len(e5_only),
            "aiteam_only_rescues": len(ai_only),
            "e5_only": [
                {"qid": q, "doc_id": d}
                for q, d in sorted(e5_only)
            ],
            "aiteam_only": [
                {"qid": q, "doc_id": d}
                for q, d in sorted(ai_only)
            ],
            "overlap": [
                {"qid": q, "doc_id": d}
                for q, d in sorted(overlap)
            ],
            "union_rescued": [
                {"qid": q, "doc_id": d}
                for q, d in sorted(both_rescues)
            ],
            "block_rescues": {
                b: {
                    "e5": sum(q in set(ids) for q, _ in e5_rescues),
                    "aiteam": sum(q in set(ids) for q, _ in ai_rescues),
                    "union": sum(q in set(ids) for q, _ in both_rescues),
                }
                for b, ids in blocks.items()
            },
            "mean_novel_e5_docs": float(
                np.mean(
                    [
                        sum(
                            d not in set(current_pool[q])
                            for d in e5_rankings[q][:depth]
                        )
                        for q in all_ids
                    ]
                )
            ),
            "mean_novel_aiteam_docs": float(
                np.mean(
                    [
                        sum(
                            d not in set(current_pool[q])
                            for d in aiteam[q][:depth]
                        )
                        for q in all_ids
                    ]
                )
            ),
        }

    d20 = depth_reports["20"]
    d50 = depth_reports["50"]

    source_gate = {
        "criteria": {
            "top20_union_ceiling_delta_gte": 0.002,
            "or_top50_union_ceiling_delta_gte": 0.004,
            "and_e5_has_unique_outside_pool_rescue": True,
        },
        "passed": bool(
            (
                d20["current_plus_union_delta"] >= 0.002
                or d50["current_plus_union_delta"] >= 0.004
            )
            and d50["e5_only_rescues"] >= 1
        ),
    }

    verdict = (
        "PROMOTE_MULTI_SOURCE_CANDIDATE_FRONTIER"
        if source_gate["passed"]
        else "KILL_DENSE_FULL_CORPUS_EXPANSION_FRONTIER"
    )

    report = {
        "schema": "manual.cal600_adapted_e5_full_corpus_v1.report",
        "source_seal": seal,
        "aiteam_top50_sha256": sha256_file(aiteam_path),
        "current_candidate_ceiling": current_ceiling,
        "outside_pool_gold_occurrences": len(outside_pairs),
        "depths": depth_reports,
        "source_gate": source_gate,
        "verdict": verdict,
        "fold_runtime_reports": fold_reports,
    }
    write_json(out / "FINAL_REPORT.json", report)

    print("=" * 88)
    print(f"Current D1 pool ceiling : {current_ceiling:.10f}")
    print(f"Outside-pool gold occ.  : {len(outside_pairs)}")
    for depth in (20, 50):
        d = depth_reports[str(depth)]
        print(
            f"Depth {depth:>2} | "
            f"E5 Δ={d['current_plus_e5_delta']:+.10f} "
            f"AI Δ={d['current_plus_aiteam_delta']:+.10f} "
            f"UNION Δ={d['current_plus_union_delta']:+.10f} | "
            f"rescues E5/AI/U={d['e5_rescues']}/"
            f"{d['aiteam_rescues']}/{d['union_rescues']} | "
            f"E5-only={d['e5_only_rescues']} overlap={d['overlap_rescues']}"
        )
    print(f"Verdict                : {verdict}")
    print(f"Report                 : {out / 'FINAL_REPORT.json'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
