# RAGAS Adapter for EvalHub

[RAGAS](https://github.com/explodinggradients/ragas) (Retrieval Augmented Generation Assessment) adapter for the EvalHub evaluation service.

## Overview

This adapter evaluates RAG pipeline quality using RAGAS metrics. It connects to OpenAI-compatible model endpoints for LLM completions and embeddings, then runs RAGAS evaluation on datasets provided via EvalHub's data pipeline.

## Available Metrics

| Metric | Description |
|--------|-------------|
| `answer_relevancy` | How relevant the answer is to the question |
| `answer_similarity` | Semantic similarity between answer and reference |
| `context_precision` | Precision of retrieved context for answering |
| `context_recall` | How much of the reference is covered by context |
| `context_entity_recall` | Entity-level recall from context |
| `faithfulness` | Whether the answer is grounded in the context |
| `nv_accuracy` | Answer accuracy (class-based, ragas v0.4+) |
| `nv_context_relevance` | Context relevance (class-based, ragas v0.4+) |
| `factual_correctness` | Factual correctness of the answer |
| `noise_sensitivity` | Sensitivity to noise in context |
| `nv_response_groundedness` | Response groundedness (class-based, ragas v0.4+) |

## Dataset Format

The adapter expects JSONL or JSON datasets with RAGAS column names:

```jsonl
{"user_input": "What is AI?", "response": "Artificial Intelligence is...", "retrieved_contexts": ["AI is a field of..."], "reference": "AI stands for Artificial Intelligence"}
```

Use `column_map` in `parameters` to rename columns from your dataset format.

## Configuration

Key `parameters` parameters:

- `metrics`: List of metric names to evaluate
- `embedding_model`: Model name for embeddings (defaults to LLM model)
- `embedding_url`: Separate endpoint for embeddings
- `max_tokens`: Max tokens for LLM completions
- `temperature`: Sampling temperature
- `column_map`: Map dataset columns to RAGAS names
- `data_path`: Explicit path to dataset file

## Local Development

```bash
pip install -r requirements.txt
EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
```

## Testing

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -m integration
```
