# Research V2 clean scaffold

This namespace is intentionally small. It does not import a historical pipeline
wholesale and it never writes into `burst_userft_maxrecall`, current submissions,
model checkpoints, or existing caches.

## Chosen module boundaries

1. `data_contract`: 8,507 retained parent documents; audited duplicate aliases;
   7,000 split members and 6,991 evaluable canonical labels.
2. `candidate_sources`: one frozen dense retriever plus one sparse BM25 source.
   Candidate generation is label-free at scoring time. A task-fine-tuned
   bi-encoder, if tested, is a separate fixed-pool ranking expert.
3. `evidence_contract`: one query-conditioned selector, one representation, and
   max-dominant parent aggregation. Structural and lexical evidence are an A/B
   contract, not simultaneous feature families.
4. `reranker`: five fold-specific task-adapted cross-encoders trained with
   boundary hard negatives.
5. `fusion`: absent from the primary pipeline. A single low-capacity meta-ranker
   is allowed only after a `>=0.005` recoverable-signal gate and must use fully
   nested upstream scores. No query-memory, rules, graph propagation, or feature
   zoo.
6. `selection`: deterministic top five; a set-aware selector is allowed only
   after a preregistered multi-gold gate passes.

## Existing code referenced, not copied

- Huy lexical evidence selector: `benchmark_jina_reranker_holdouts.top_passages`.
- Huy production feature extraction reference: `tune_expanded_fusion_selection.ltr_features`.
- LegalIR preprocessing and structural evidence references:
  `src/structural_chunker_v3.py` and `cache/final_preprocessed_v2`.
- LegalIR canonical label policy reference:
  `src/exp030_legal_evidence_routing.canonical_answers`.

## Executed contract

Use identical candidate query-document pairs, identical base cross-encoder,
identical train pairs, and identical five folds. Compare only:

- A: Huy `top_passages(question, parent, count=2)`;
- B: the best single `structural_v3` chunk selected label-free within parent.

Aggregate passage scores with parent `max` in both arms. The experiment is a
diagnostic gate; do not tune passage counts or blend A and B before the result.

The full evidence-contract A/B is complete. Lexical evidence beat structural-v3
by `0.0688743` Recall@5 on the fixed candidate pool and is now immutable for
boundary training.

Executable files in this namespace:

- `build_v2_executable_baseline.py`: seal retrieval curves and the bounded pool;
- `run_evidence_contract_ab.py`: score/evaluate the two evidence contracts;
- `build_v2_boundary_groups.py`: materialize fold-safe positives and negatives;
- `jina_v2_boundary_train.py`: train/resume/score the Jina-v2 adapter;
- `stage_kaggle_boundary_bundle.py`: build the hash-locked private Kaggle input.

The final Kaggle bundle is
`cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v2`.
Remote pilot/full execution remains blocked only by absent Kaggle credentials on
this host; local forward/backward and exact resume parity pass at 512 tokens.
