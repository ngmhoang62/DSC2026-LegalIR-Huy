# Research V2 clean scaffold

This namespace is intentionally small. It does not import a historical pipeline
wholesale and it never writes into `burst_userft_maxrecall`, current submissions,
model checkpoints, or existing caches.

## Chosen module boundaries

1. `data_contract`: 8,507 retained parent documents; audited duplicate aliases;
   7,000 split members and 6,991 evaluable canonical labels.
2. `candidate_sources`: one dense task-adapted retriever plus one sparse BM25
   source. Candidate generation is label-free at scoring time.
3. `evidence_contract`: one query-conditioned selector, one representation, and
   max-dominant parent aggregation. Structural and lexical evidence are an A/B
   contract, not simultaneous feature families.
4. `reranker`: five fold-specific task-adapted cross-encoders trained with
   boundary hard negatives.
5. `fusion`: a single low-capacity OOF meta-ranker over source ranks/scores and
   reranker score. No query-memory, rules, graph propagation, or feature zoo.
6. `selection`: deterministic top five; a set-aware selector is allowed only
   after a preregistered multi-gold gate passes.

## Existing code referenced, not copied

- Huy lexical evidence selector: `benchmark_jina_reranker_holdouts.top_passages`.
- Huy production feature extraction reference: `tune_expanded_fusion_selection.ltr_features`.
- LegalIR preprocessing and structural evidence references:
  `src/structural_chunker_v3.py` and `cache/final_preprocessed_v2`.
- LegalIR canonical label policy reference:
  `src/exp030_legal_evidence_routing.canonical_answers`.

## First executable experiment

Use identical candidate query-document pairs, identical base cross-encoder,
identical train pairs, and identical five folds. Compare only:

- A: Huy `top_passages(question, parent, count=2)`;
- B: the best single `structural_v3` chunk selected label-free within parent.

Aggregate passage scores with parent `max` in both arms. The experiment is a
diagnostic gate; do not tune passage counts or blend A and B before the result.

`build_v2_folds.py` creates the immutable split and companion SHA-256 file.

