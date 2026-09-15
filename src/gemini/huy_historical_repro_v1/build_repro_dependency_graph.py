"""Build and save REPRO_DEPENDENCY_GRAPH.json for clean-room reproduction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import reproduce

OUTPUT_PATH = REPO_ROOT / "results/gemini/huy_historical_repro_v1/REPRO_DEPENDENCY_GRAPH.json"

# Detailed analysis for each required path in reproduce.py::REQUIRED
GRAPH_SPECS = {
    "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json": {
        "classification": "RAW_INPUT",
        "description": "Official public test set containing 1,000 legal queries",
        "upstream_inputs": [],
        "generator": None,
        "query_count": 1000,
        "downstream_consumers": [
            "run_vnlegal_extra_channel_submission.py",
            "score_public_models.py",
            "score_public_title_embed.py",
            "run_burst_expanded_fusion_submission.py",
            "run_burst_multistage_submission.py"
        ]
    },
    "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json": {
        "classification": "RAW_INPUT",
        "description": "Full training dataset containing 7,000 labeled queries and gold answers",
        "upstream_inputs": [],
        "generator": None,
        "query_count": 7000,
        "downstream_consumers": [
            "tune_burst_kernel_posterior.py",
            "tune_burst_large_ltr.py",
            "benchmark_burst_v4_full_sqlite.py",
            "run_vnlegal_extra_channel_submission.py",
            "tune_expanded_fusion_robust.py"
        ]
    },
    "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts": {
        "classification": "RAW_INPUT",
        "description": "Canonical legal corpus of 8,532 context documents (JSON files)",
        "upstream_inputs": [],
        "generator": None,
        "document_count": 8532,
        "downstream_consumers": [
            "run_burst_expanded_fusion_submission.py::DocumentStore",
            "build_corpus_dense_index.py",
            "score_cv_custom_encoder.py",
            "score_cv_jina_ft.py",
            "score_public_models.py",
            "score_public_title_embed.py",
            "tune_title_embedding.py",
            "tune_doctype_features.py",
            "tune_citation_graph.py"
        ]
    },
    "results/from_drive/aiteamvn_ft_cv.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "score_cv_custom_encoder.py",
        "exact_command": "python score_cv_custom_encoder.py --model-path models/from_drive/AITeamVN_Vietnamese_Embedding --out results/from_drive/aiteamvn_ft_cv.pkl",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/from_drive/AITeamVN_Vietnamese_Embedding/model.safetensors",
            "results/corpus_index/holdout_extended_scores_cap32.pkl"
        ],
        "model_used": "AITeamVN_Vietnamese_Embedding (fine-tuned bi-encoder, XLM-RoBERTa Large 1024x24)",
        "uses_labels": False,
        "query_population": "600 CV queries (blocks a, b, c, d)",
        "expected_output_shape": "dict(600, sample_doc_count ~36)",
        "downstream_consumers": [
            "evaluate_cv.py",
            "reproduce.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/from_drive/aiteamvn_ft_public.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "score_public_models.py",
        "exact_command": "python score_public_models.py --kind bi --model-path models/from_drive/AITeamVN_Vietnamese_Embedding --out results/from_drive/aiteamvn_ft_public.pkl",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/from_drive/AITeamVN_Vietnamese_Embedding/model.safetensors",
            "results/crossenc_fullpool/public_scores.pkl"
        ],
        "model_used": "AITeamVN_Vietnamese_Embedding (fine-tuned bi-encoder, XLM-RoBERTa Large 1024x24)",
        "uses_labels": False,
        "query_population": "1000 Public queries",
        "expected_output_shape": "dict(1000, sample_doc_count ~40)",
        "downstream_consumers": [
            "reproduce.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/from_drive/jina_ft_cv.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "score_cv_jina_ft.py",
        "exact_command": "python score_cv_jina_ft.py --weights models/from_drive/jina_finetuned/model.safetensors --out results/from_drive/jina_ft_cv.pkl",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/jina-reranker-v2-base-multilingual",
            "models/from_drive/jina_finetuned/model.safetensors",
            "results/corpus_index/holdout_extended_scores_cap32.pkl"
        ],
        "model_used": "jina-reranker-v2-base-multilingual overlaid with fine-tuned state_dict (XLM-RoBERTa Base 768x12)",
        "uses_labels": False,
        "query_population": "600 CV queries (blocks a, b, c, d)",
        "expected_output_shape": "dict(600, sample_doc_count ~36)",
        "downstream_consumers": [
            "evaluate_cv.py",
            "reproduce.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/from_drive/jina_ft_public.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "score_public_models.py",
        "exact_command": "python score_public_models.py --kind jina --weights models/from_drive/jina_finetuned/model.safetensors --out results/from_drive/jina_ft_public.pkl",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/jina-reranker-v2-base-multilingual",
            "models/from_drive/jina_finetuned/model.safetensors",
            "results/crossenc_fullpool/public_scores.pkl"
        ],
        "model_used": "jina-reranker-v2-base-multilingual overlaid with fine-tuned state_dict",
        "uses_labels": False,
        "query_population": "1000 Public queries",
        "expected_output_shape": "dict(1000, sample_doc_count ~40)",
        "downstream_consumers": [
            "reproduce.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/burst_fresh_block/title_embed_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "tune_title_embedding.py",
        "exact_command": "python tune_title_embedding.py",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/AITeamVN_Vietnamese_Embedding",
            "results/corpus_index/holdout_extended_scores_cap32.pkl"
        ],
        "model_used": "AITeamVN/Vietnamese_Embedding (base frozen bi-encoder, 128 max_length, CLS)",
        "uses_labels": False,
        "query_population": "600 CV queries",
        "expected_output_shape": "dict(600, sample_doc_count ~34)",
        "downstream_consumers": [
            "evaluate_cv.py",
            "reproduce.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/burst_fresh_block/title_embed_public.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "score_public_title_embed.py",
        "exact_command": "python score_public_title_embed.py",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/AITeamVN_Vietnamese_Embedding",
            "results/crossenc_fullpool/public_scores.pkl"
        ],
        "model_used": "AITeamVN/Vietnamese_Embedding (base frozen bi-encoder, 128 max_length, CLS)",
        "uses_labels": False,
        "query_population": "1000 Public queries",
        "expected_output_shape": "dict(1000, sample_doc_count ~40)",
        "downstream_consumers": [
            "reproduce.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/burst_userft_maxrecall/vnlegal_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "run_vnlegal_extra_channel_submission.py::score_vnlegal()",
        "exact_command": "python run_vnlegal_extra_channel_submission.py [with cached/fresh inputs]",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/vnlegal-lal"
        ],
        "model_used": "darklethelong/vnlegal-lal (base frozen bi-encoder, CLS pooling, 2 passages/doc)",
        "uses_labels": False,
        "query_population": "1000 Public queries",
        "expected_output_shape": "dict(1000, sample_doc_count ~40)",
        "downstream_consumers": [
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/burst_expanded_fusion/expansion_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "run_burst_expanded_fusion_submission.py::dense_expansion()",
        "exact_command": "python run_burst_expanded_fusion_submission.py --stage expansion",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/AITeamVN_Vietnamese_Embedding",
            "results/burst_robust_fusion/public_retrieval.pkl"
        ],
        "model_used": "AITeamVN/Vietnamese_Embedding (base frozen bi-encoder)",
        "uses_labels": False,
        "query_population": "1000 Public queries",
        "expected_output_shape": "dict(config, scores: dict(1000))",
        "downstream_consumers": [
            "run_burst_expanded_fusion_submission.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/burst_expanded_fusion/corpus_rank_cap32.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "run_burst_expanded_fusion_submission.py::corpus_dense()",
        "exact_command": "python run_burst_expanded_fusion_submission.py",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
            "models/AITeamVN_Vietnamese_Embedding",
            "results/corpus_index/chunks_cap32.f16",
            "results/corpus_index/chunks_cap32.json"
        ],
        "model_used": "AITeamVN/Vietnamese_Embedding",
        "uses_labels": False,
        "query_population": "1000 Public queries",
        "expected_output_shape": "dict(cap, depth, ranking: dict(1000), scores: dict(1000))",
        "downstream_consumers": [
            "run_burst_expanded_fusion_submission.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/corpus_index/holdout_dense_rank_cap32.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "benchmark_corpus_dense_recall.py",
        "exact_command": "python benchmark_corpus_dense_recall.py --cap 32",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "models/AITeamVN_Vietnamese_Embedding",
            "results/corpus_index/chunks_cap32.f16",
            "results/corpus_index/chunks_cap32.json"
        ],
        "model_used": "AITeamVN/Vietnamese_Embedding",
        "uses_labels": False,
        "query_population": "600 CV queries",
        "expected_output_shape": "dict(cap, ranking: dict(600), scores: dict(600))",
        "downstream_consumers": [
            "evaluate_cv.py",
            "tune_corpus_cap32_fusion.py",
            "tune_corpus_dense_fusion.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/dense_expansion/union50_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "RESOLVABLE_FRESH_INFERENCE",
        "generator_script": "benchmark_dense_expansion_holdouts.py",
        "exact_command": "python benchmark_dense_expansion_holdouts.py",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/AITeamVN_Vietnamese_Embedding",
            "results/burst_large_ltr/*_retrieval.pkl"
        ],
        "model_used": "AITeamVN/Vietnamese_Embedding",
        "uses_labels": False,
        "query_population": "300 queries (a, b, c holdouts)",
        "expected_output_shape": "dict(depth, scores: dict(300))",
        "downstream_consumers": [
            "benchmark_expanded_rerank_holdouts.py",
            "tune_expanded_fusion_robust.py",
            "tune_expanded_fusion_selection.py"
        ]
    },
    "results/aiteamvn_dense/holdout_scores_512.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "PARTIALLY_RESOLVABLE",
        "generator_script": "benchmark_aiteamvn_holdouts.py",
        "exact_command": "python benchmark_aiteamvn_holdouts.py",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
            "models/AITeamVN_Vietnamese_Embedding",
            "results/jina_reranker/holdout_scores_finetuned.pkl",
            "results/e5_dense/holdout_scores.pkl"
        ],
        "notes": "Script generates 300 queries (a,b,c); cached file has 900 queries. Requires upstream holdout_scores_finetuned.pkl and e5_dense/holdout_scores.pkl.",
        "downstream_consumers": [
            "benchmark_expanded_rerank_holdouts.py",
            "tune_expanded_fusion_robust.py"
        ]
    },
    "results/burst_robust_fusion/public_retrieval.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "PARTIALLY_RESOLVABLE",
        "generator_script": "run_burst_expanded_fusion_submission.py::load_public_retrieval()",
        "exact_command": "python run_burst_expanded_fusion_submission.py",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "benchmarks/legalir_full_fts.sqlite"
        ],
        "notes": "Generates sparse retrieval lists via SQLite FTS5 database (buildable via benchmark_burst_v4_full_sqlite.py).",
        "downstream_consumers": [
            "run_burst_expanded_fusion_submission.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "PARTIALLY_RESOLVABLE",
        "generator_script": "tune_burst_large_ltr.py",
        "exact_command": "python tune_burst_large_ltr.py",
        "upstream_inputs": [
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
            "benchmarks/legalir_full_fts.sqlite"
        ],
        "notes": "Generates retrieval cache for 1,150 training/validation queries via SQLite FTS5 index.",
        "downstream_consumers": [
            "benchmark_jina_reranker_holdouts.py",
            "tune_burst_empirical_bayes_ltr.py"
        ]
    },
    "results/e5_dense/holdout_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [
            "intfloat/multilingual-e5-large"
        ],
        "notes": "Historical cache contains 900 holdout query scores for E5 dense channel. No generator script was included in the repository.",
        "downstream_consumers": [
            "tune_corpus_cap32_fusion.py",
            "tune_corpus_dense_fusion.py",
            "tune_expanded_fusion_robust.py"
        ]
    },
    "results/burst_gpu_threeview/cpu_top20.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [],
        "notes": "Contains top-20 candidate lists for 1,000 public queries from a three-view CPU job. No script named threeview or three-view exists in repo.",
        "downstream_consumers": [
            "run_burst_expanded_fusion_submission.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [],
        "notes": "Contains public E5 model scores extracted by run_vnlegal_extra_channel_submission.py line 377. Generator script not in repo.",
        "downstream_consumers": [
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/crossenc_fullpool/cv_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [
            "models/AITeamVN_Vietnamese_Reranker"
        ],
        "notes": "Contains cross-encoder scores for 900 CV queries. Referenced in run_vnlegal_extra_channel_submission.py line 129 as 'see score_cv_crossenc_channel.py', but score_cv_crossenc_channel.py was never committed.",
        "downstream_consumers": [
            "evaluate_cv.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/crossenc_fullpool/public_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [
            "models/AITeamVN_Vietnamese_Reranker"
        ],
        "notes": "Contains cross-encoder scores for 1,000 public queries. Generator script omitted from repository.",
        "downstream_consumers": [
            "run_vnlegal_extra_channel_submission.py",
            "score_public_models.py",
            "score_public_title_embed.py"
        ]
    },
    "results/burst_large_ltr/fresh_1251_1350_retrieval.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [],
        "notes": "Block b sparse retrieval cache (100 queries). Generator script omitted.",
        "downstream_consumers": [
            "tune_burst_empirical_bayes_ltr.py",
            "tune_expanded_fusion_robust.py"
        ]
    },
    "results/burst_large_ltr/fresh_1351_1450_retrieval.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [],
        "notes": "Block c sparse retrieval cache (100 queries). Generator script omitted.",
        "downstream_consumers": [
            "tune_expanded_fusion_robust.py"
        ]
    },
    "results/burst_large_ltr/fresh_1451_1750_retrieval.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [],
        "notes": "Block d sparse retrieval cache (300 queries). Generator script omitted.",
        "downstream_consumers": [
            "tune_expanded_fusion_robust.py"
        ]
    },
    "results/jina_reranker/holdout_scores_finetuned.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "MISSING_WEIGHT_DEPENDENCY",
        "generator_script": "benchmark_jina_reranker_holdouts.py",
        "exact_command": "python benchmark_jina_reranker_holdouts.py --checkpoint results/jina_reranker/burst_pairwise_state.pt --cache-name holdout_scores_finetuned.pkl",
        "upstream_inputs": [
            "models/jina-reranker-v2-base-multilingual",
            "results/jina_reranker/burst_pairwise_state.pt"
        ],
        "notes": "Generator exists, but required fine-tuned checkpoint burst_pairwise_state.pt is missing from disk and repository history.",
        "downstream_consumers": [
            "benchmark_aiteamvn_holdouts.py",
            "benchmark_expanded_rerank_holdouts.py",
            "tune_expanded_fusion_robust.py"
        ]
    },
    "results/expanded_rerank/scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "MISSING_WEIGHT_DEPENDENCY",
        "generator_script": "benchmark_expanded_rerank_holdouts.py",
        "exact_command": "python benchmark_expanded_rerank_holdouts.py",
        "upstream_inputs": [
            "results/jina_reranker/burst_pairwise_state.pt",
            "models/AITeamVN_Vietnamese_Embedding"
        ],
        "notes": "Requires missing burst_pairwise_state.pt.",
        "downstream_consumers": [
            "tune_expanded_fusion_robust.py",
            "tune_expanded_fusion_selection.py"
        ]
    },
    "results/burst_expanded_fusion/rerank_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "MISSING_WEIGHT_DEPENDENCY",
        "generator_script": "run_burst_expanded_fusion_submission.py::rerank()",
        "exact_command": "python run_burst_expanded_fusion_submission.py",
        "upstream_inputs": [
            "results/jina_reranker/burst_pairwise_state.pt",
            "models/AITeamVN_Vietnamese_Embedding"
        ],
        "notes": "Requires missing burst_pairwise_state.pt.",
        "downstream_consumers": [
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/vietnamese_reranker/holdout_scores_512_finetuned.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [],
        "notes": "Legacy channel loaded by build_views line 44 but not included in final LTR NAMES list. Generator omitted.",
        "downstream_consumers": [
            "tune_expanded_fusion_robust.py"
        ]
    },
    "results/embedding_finetune/vnlegal_lal_cv_scores.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [
            "models/vnlegal-lal"
        ],
        "notes": "Original script score_cv_vnlegal_lal.py was omitted from repository (referenced in docstrings).",
        "downstream_consumers": [
            "evaluate_cv.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    },
    "results/corpus_index/holdout_extended_scores_cap32.pkl": {
        "classification": "GENERATED_ARTIFACT",
        "status": "UNRESOLVED_GENERATOR",
        "generator_script": None,
        "exact_command": None,
        "upstream_inputs": [],
        "notes": "Precomputed Jina and Dense scores over the extended candidate pool for cap32. Dedicated generation script omitted.",
        "downstream_consumers": [
            "evaluate_cv.py",
            "score_cv_custom_encoder.py",
            "score_cv_jina_ft.py",
            "run_vnlegal_extra_channel_submission.py"
        ]
    }
}


def main():
    print(f"Total items analyzed: {len(GRAPH_SPECS)}")
    raw_inputs = [k for k, v in GRAPH_SPECS.items() if v["classification"] == "RAW_INPUT"]
    resolvable = [k for k, v in GRAPH_SPECS.items() if v.get("status") == "RESOLVABLE_FRESH_INFERENCE"]
    partially_resolvable = [k for k, v in GRAPH_SPECS.items() if v.get("status") == "PARTIALLY_RESOLVABLE"]
    unresolved = [k for k, v in GRAPH_SPECS.items() if v.get("status") == "UNRESOLVED_GENERATOR"]
    missing_weights = [k for k, v in GRAPH_SPECS.items() if v.get("status") == "MISSING_WEIGHT_DEPENDENCY"]

    print(f"  RAW_INPUT: {len(raw_inputs)}")
    print(f"  RESOLVABLE_FRESH_INFERENCE: {len(resolvable)}")
    print(f"  PARTIALLY_RESOLVABLE: {len(partially_resolvable)}")
    print(f"  UNRESOLVED_GENERATOR: {len(unresolved)}")
    print(f"  MISSING_WEIGHT_DEPENDENCY: {len(missing_weights)}")

    report = {
        "total_required_paths": len(reproduce.REQUIRED),
        "raw_inputs": raw_inputs,
        "resolvable_fresh_inference": resolvable,
        "partially_resolvable": partially_resolvable,
        "unresolved_generators": unresolved,
        "missing_weight_dependencies": missing_weights,
        "dependency_specs": GRAPH_SPECS
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dependency graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
