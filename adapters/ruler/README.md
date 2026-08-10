# RULER Adapter

## Overview

This adapter integrates [NVIDIA RULER](https://github.com/NVIDIA/RULER) (**What's the Real
Context Size of Your LLM?**) with EvalHub. RULER is a synthetic long-context benchmark that
evaluates LLMs across 13 tasks at configurable context lengths, providing a principled measure
of *effective* context utilisation rather than a simple token-count ceiling.

## Architecture

### Key Components

| File | Purpose |
|---|---|
| `main.py` | `RulerAdapter` — EvalHub `FrameworkAdapter` implementation |
| `provider.yaml` | EvalHub provider manifest (benchmarks, runtime, parameters) |
| `Containerfile` | Container image definition (UBI9 Python 3.12) |
| `requirements.txt` | Runtime Python dependencies |
| `requirements-test.txt` | Test-only dependencies |
| `scripts/` | Vendored NVIDIA RULER data-generation and evaluation scripts |
| `meta/job.json` | Sample `JobSpec` for local testing |

### Benchmark Tasks (13 total)

| Category | Benchmark IDs |
|---|---|
| Needle-in-a-Haystack | `niah-single-noise`, `niah-single-essay`, `niah-single-uuid`, `niah-multikey`, `niah-needle-bg`, `niah-multikey-uuid`, `niah-multivalue`, `niah-multiquery` |
| Variable Tracking | `variable-tracking` |
| Aggregation | `common-words-extraction`, `frequency-words-extraction` |
| Question Answering | `qa-squad`, `qa-hotpotqa` |

### Adapter Lifecycle

```
INITIALIZING → LOADING_DATA → RUNNING_EVALUATION → POST_PROCESSING → PERSISTING_ARTIFACTS
```

For each (task, context-length) pair:

1. **Data generation** — calls vendored `scripts/data/prepare.py` to produce a synthetic
   JSONL dataset with the target number of tokens.
2. **Inference** — sends each sample to an OpenAI-compatible endpoint and collects predictions.
3. **Evaluation** — applies per-task string-match metrics from `scripts/eval/synthetic/constants.py`.
4. **Aggregation** — averages per-context-length scores into a task-level and overall score.

## Supported Model Providers

Any **OpenAI-compatible inference endpoint** (vLLM, LiteLLM, Red Hat RHOAI ServingRuntime, etc.).
Set `model.url` in the JobSpec to the `/v1` endpoint and `model.name` to the model ID.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `benchmarks` | array | null | Override: run multiple benchmark IDs in one job |
| `context_lengths` | array | `[4096, 8192, 16384]` | Context sizes (tokens) to evaluate |
| `num_samples` | integer | 10 | Samples per (task × context length) |
| `tokenizer_path` | string | model.name | HF model ID or tiktoken model for data generation |
| `tokenizer_type` | string | `hf` | `hf` (HuggingFace) or `openai` (tiktoken) |
| `model_template` | string | `base` | Chat prompt template (see `scripts/data/template.py`) |
| `tokens_to_generate` | integer | null | Max generation tokens (defaults to per-task value) |
| `batch_size` | integer | 1 | Inference batch / log interval |
| `random_seed` | integer | 42 | Seed for reproducible data generation |

## Example Job Spec

```json
{
  "id": "ruler-niah-essay-001",
  "provider_id": "ruler",
  "benchmark_id": "niah-single-essay",
  "model": {
    "url": "http://vllm.svc.cluster.local:8000/v1",
    "name": "meta-llama/Meta-Llama-3-8B-Instruct"
  },
  "num_examples": 20,
  "parameters": {
    "context_lengths": [4096, 8192],
    "tokenizer_path": "meta-llama/Meta-Llama-3-8B-Instruct",
    "tokenizer_type": "hf",
    "model_template": "meta-llama3",
    "random_seed": 42
  },
  "callback_url": "https://evalhub.apps.example.com"
}
```

## Local Testing

```bash
# Set up virtual environment
cd adapters/ruler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-test.txt

# Run tests
.venv/bin/pytest tests/ -v

# Run a single test file
.venv/bin/pytest tests/test_task_config.py -v
```

## Building the Container Image

```bash
# From repo root
make image-ruler

# Push to a registry
make push-ruler REGISTRY=quay.io/your-org VERSION=v0.1.0
```

## Vendored Scripts

`scripts/` contains a curated subset of NVIDIA RULER scripts adapted for the EvalHub
adapter pattern. The original NeMo-based prediction pipeline (`scripts/pred/call_api.py`)
and NeMo-based evaluation pipeline (`scripts/eval/evaluate.py`) are not called by the
adapter — the adapter implements its own OpenAI-compatible inference loop and invokes
only the metric functions from `scripts/eval/synthetic/constants.py`.

## License

Apache 2.0 — see repository root `LICENSE`.
Vendored NVIDIA RULER scripts retain their original NVIDIA copyright.
