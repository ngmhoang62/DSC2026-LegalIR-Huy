"""Dataset and negative sampling engine for jina_honest_adaptation_v1.
Implements Sections 5, 8, 9, 10, 11 evidence contract and curriculum sampling.
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from common import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src/huy_fasttrack"))
sys.path.insert(0, str(REPO_ROOT / "src/research_v2_forensic"))

import run_huy_5fold_fasttrack as core
from benchmark_jina_reranker_holdouts import top_passages


class LegalRetrievalDataset:
    def __init__(self, rng_seed: int = 2026):
        self.seed = rng_seed
        print("Loading baseline dataset via fasttrack core...", flush=True)
        folds, pools, questions, golds, e5_orders, e5_scores, dup, pred_hashes = core.load_inputs()
        self.folds = folds
        self.fold_for = {qid: f for f, qids in folds.items() for qid in qids}
        self.pools = pools
        self.questions = questions
        self.golds = golds
        self.duplicate_exclusions = dup

        # Load contexts
        contexts_path = (
            REPO_ROOT
            / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
        )
        print("Loading canonical contexts...", flush=True)
        self.contexts = {}
        with open(contexts_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self.contexts[str(rec["doc_id"])] = str(rec["passage"] or "")

        # Load H_LOCAL retrieval rankings
        self.h_local = {}
        burst_path = (
            REPO_ROOT
            / "results/gemini/huy_sparse_resurrection_v1/cache/BURST_V2_RETRIEVAL_RESULTS.jsonl"
        )
        if burst_path.exists():
            print("Loading H_LOCAL retrieval rankings...", flush=True)
            with open(burst_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.h_local[str(row["qid"])] = [
                            str(x[0]) for x in row.get("h_local_top100", [])
                        ]

        # Load cached frozen scores from sqlite
        db_path = REPO_ROOT / "cache/research_v2_forensic/evidence_ab_scores.sqlite"
        print("Loading frozen Jina parent scores...", flush=True)
        self.frozen_scores: Dict[Tuple[str, str], float] = {}
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        cur = conn.cursor()
        for qid, doc, score in cur.execute(
            "SELECT qid, doc_id, score FROM scores WHERE arm='lexical'"
        ):
            self.frozen_scores[(str(qid), str(doc))] = float(score)
        conn.close()
        print(f"Ready: {len(self.pools)} queries, {len(self.contexts)} contexts.", flush=True)

    def get_outer_train_qids(self, held_fold: str) -> List[str]:
        """Return outer-training queries excluding held_fold and duplicate-linked exclusions."""
        excl = set(self.duplicate_exclusions.get(held_fold, []))
        qids = [
            qid
            for qid, f in self.fold_for.items()
            if f != held_fold and qid not in excl and self.golds.get(qid)
        ]
        return sorted(qids, key=int)

    def sample_query_candidates(
        self, qid: str, epoch: int = 1, seed_offset: int = 0
    ) -> Dict[str, Any]:
        """Sample candidate group for query qid following Sections 8, 9, 10 curriculum."""
        qtext = self.questions[qid]
        gold = self.golds[qid]
        pool = self.pools[qid]

        # Positives: all canonical gold parents in pool
        positives = [d for d in pool if d in gold]

        rng = random.Random(self.seed + int(qid) + seed_offset * 100000 + epoch * 1000)

        # Semi-hard: ranks 6-20 non-gold (indices 5..19)
        semi_pool = [d for d in pool[5:20] if d not in gold]
        semi_hard = rng.sample(semi_pool, min(4, len(semi_pool)))

        # Medium: ranks 21-40 non-gold (indices 20..39)
        med_pool = [d for d in pool[20:40] if d not in gold]
        medium = rng.sample(med_pool, min(3, len(med_pool)))

        # Tail: remaining pool (indices 40..)
        tail_pool = [d for d in pool[40:] if d not in gold]
        tail = rng.sample(tail_pool, min(1, len(tail_pool)))

        selected_negatives = list(semi_hard) + list(medium) + list(tail)

        # Section 9: H_LOCAL diversity negative
        if qid in self.h_local:
            for d in self.h_local[qid]:
                if d in self.contexts and d not in gold and d not in selected_negatives:
                    selected_negatives.append(d)
                    break

        # Section 10: Epoch 2 add 1 hard negative from authoritative ranks 1-8
        if epoch >= 2:
            rank1_8_pool = [d for d in pool[:8] if d not in gold and d not in selected_negatives]
            if rank1_8_pool:
                selected_negatives.append(rank1_8_pool[0])

        all_docs = positives + selected_negatives
        pos_indices = list(range(len(positives)))

        # Extract Huy R0 passages
        pairs = []
        doc_indices_for_passages = []
        for doc_idx, d in enumerate(all_docs):
            dtext = self.contexts.get(d, "")
            passages = top_passages(qtext, dtext, count=2, window=220, overlap=70)
            for p in passages:
                pairs.append((qtext, p))
                doc_indices_for_passages.append(doc_idx)

        # Frozen parent scores for stability regularization
        frozen_doc_scores = [self.frozen_scores.get((qid, d), 0.5) for d in all_docs]

        return {
            "qid": qid,
            "qtext": qtext,
            "docs": all_docs,
            "pos_indices": pos_indices,
            "pairs": pairs,
            "doc_indices_for_passages": doc_indices_for_passages,
            "frozen_doc_scores": frozen_doc_scores,
            "num_docs": len(all_docs),
        }
