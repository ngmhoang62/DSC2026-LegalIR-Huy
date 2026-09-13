"""Post-gate causal anatomy for the killed frozen MonoT5 Fold-0 expert."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OUT = ROOT / "results/research_v2_open_rl"


def rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()


def canonical_answers():
    train=json.loads((WORKSPACE/"LegalIR/public_test_dataset/train.json").read_text(encoding="utf-8"))
    exc=json.loads((WORKSPACE/"LegalIR/cache/final_preprocessed_v2/exclusions.json").read_text(encoding="utf-8"))
    alias={str(x["doc_id"]):str(x["duplicate_retained_id"]) for x in exc if x.get("duplicate_retained_id")}
    empty={str(x["doc_id"]) for x in exc if "empty_passage" in x.get("reasons",[])}
    return {str(q):{alias.get(str(d),str(d)) for d in v["answer"]}-empty for q,v in train.items()}


def recall(docs, gold): return len(set(docs)&gold)/len(gold)
def bucket(rank): return "missing" if rank is None else "1-5" if rank<=5 else "6-10" if rank<=10 else "11-20" if rank<=20 else "21-50" if rank<=50 else "51+"


def main():
    pool={str(x["qid"]):set(map(str,x["doc_ids"])) for x in rows(ROOT/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl") if x["fold"]=="fold_0"}
    mono={str(x["qid"]):list(map(str,x["ranking"])) for x in rows(OUT/"MONOT5_FOLD0_PREDICTIONS.jsonl")}
    anchor={str(x["qid"]):list(map(str,x["fused_top5"])) for x in rows(ROOT/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl") if x["fold"]=="fold_0"}
    e5_row={str(x["qid"]):x for x in rows(ROOT/"results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl")}
    jina_v2={str(x["qid"]):list(map(str,x["top5"])) for x in rows(ROOT/"results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl") if str(x["qid"]) in pool}
    source_db=WORKSPACE/"LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"
    db=sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro",uri=True)
    native={}
    for q,source,payload in db.execute("SELECT q,source,payload FROM sources WHERE source IN ('lal','jina')"):
        q=str(q)
        if q not in pool: continue
        native[(q,str(source))]=[str(x["doc_id"]) for x in json.loads(payload) if str(x["doc_id"]) in pool[q]][:5]
    db.close()
    answers=canonical_answers(); ks=(5,10,20,50)
    sums={k:[] for k in ks}; clean=[]; expanded=[]; anchor_union=[]
    unique_occ=0; unique_queries=0; mono_rank_on_anchor_misses=Counter(); mono_rank_on_clean_misses=Counter()
    rescues=[]
    for q,ranking in mono.items():
        gold=answers[q]
        for k in ks: sums[k].append(recall(ranking[:k],gold))
        expert_sets=[set(e5_row[q]["ft_order"][:5]),set(e5_row[q]["base_order"][:5]),set(jina_v2[q]),set(native[(q,"lal")]),set(native[(q,"jina")])]
        clean_set=set().union(*expert_sets); mono_set=set(ranking[:5]); expanded_set=clean_set|mono_set
        clean.append(recall(clean_set,gold)); expanded.append(recall(expanded_set,gold)); anchor_union.append(recall(set(anchor[q])|mono_set,gold))
        unseen=(mono_set&gold)-clean_set
        unique_occ+=len(unseen); unique_queries+=bool(unseen)
        for d in gold-set(anchor[q]):
            rank=ranking.index(d)+1 if d in ranking else None; mono_rank_on_anchor_misses[bucket(rank)]+=1
        for d in gold-clean_set:
            rank=ranking.index(d)+1 if d in ranking else None; mono_rank_on_clean_misses[bucket(rank)]+=1
            if rank is not None and rank<=5: rescues.append({"qid":q,"gold_doc":d,"mono_rank":rank,"gold_count":len(gold)})
    report={
      "schema_version":"dsc2026.research_v2.monot5_incremental_anatomy.v1","status":"COMPLETE_CACHE_ONLY_AFTER_KILL",
      "standalone_recall":{f"recall@{k}":float(np.mean(v)) for k,v in sums.items()},
      "union_oracles":{"mono_plus_current_anchor_fold0":float(np.mean(anchor_union)),
                       "existing_clean_experts_fold0":float(np.mean(clean)),
                       "existing_clean_experts_plus_mono_fold0":float(np.mean(expanded)),
                       "mono_increment_over_existing_clean_experts":float(np.mean(expanded)-np.mean(clean))},
      "mono_unique_beyond_all_existing_clean_experts":{"gold_occurrences":unique_occ,"queries":unique_queries},
      "mono_rank_of_gold_missed_by_current_anchor":dict(mono_rank_on_anchor_misses),
      "mono_rank_of_gold_missed_by_all_existing_clean_experts":dict(mono_rank_on_clean_misses),
      "unique_rescues":rescues,
      "interpretation":"Frozen multilingual generative transfer is too weak standalone. Only incremental mass beyond the entire clean expert set can justify task adaptation; anchor-only union is insufficient.",
      "inputs_sha256":{"mono_predictions":sha(OUT/"MONOT5_FOLD0_PREDICTIONS.jsonl"),"source_db":sha(source_db),"pool":sha(ROOT/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl")}
    }
    (OUT/"MONOT5_INCREMENTAL_ANATOMY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
