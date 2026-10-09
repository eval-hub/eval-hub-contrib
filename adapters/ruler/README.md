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
| `prepare_assets.py` | Prepare and verify required runtime data assets |
| `DATA_LICENSES.md` | Data sources, attribution and licenses |
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
| `data_gen_timeout_seconds` | integer | 600 | Timeout for the essay downloader and each task-generation subprocess; each word-list/QA download uses this value capped at 60 seconds |

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

## Data-generation assets

All external datasets use the same runtime preparation flow. Before any task
datasets are generated, `prepare_assets.py` resolves the inputs required by the
selected tasks, validates existing cache files, and downloads only missing or
invalid inputs. The image contains the preparation code and source URLs;
dataset payloads are excluded from its build context.

| Selected task | Required input |
|---|---|
| Common words extraction (CWE) | `english_words.json` |
| SQuAD QA | `squad.json` |
| HotpotQA QA | `hotpotqa.json` |
| NIAH or VT configured with an essay haystack | `PaulGrahamEssays.json` |
| FWE, default VT, noise/needle NIAH | No external dataset |

All inputs are stored in the job's temporary `data/assets/` directory, under
`/tmp` by default. Generators read this shared directory via `RULER_DATA_DIR`.
Inputs are prepared once and reused across tasks and context lengths within
the job. The cache is removed with the job's temporary data, so a new job
downloads its required inputs again. Directly invoking `_generate_task_data`
requires preparing the inputs first; `run_benchmark_job` handles this order.

The word list and QA JSON files are checked against fixed SHA-256 digests and
valid JSON before replacement. The NVIDIA word-list URL pins commit
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`; HotpotQA uses the fixed Hugging Face
revision in NVIDIA's official downloader. Essays use the vendored NVIDIA URL
list and must produce non-empty JSON text. Remote essay contents can change
between runs. A failed download or invalid completed input stops preparation
before inference; the vendored essay downloader still logs and continues after
individual URL failures, so its download counts should be checked.

Runtime network access to the required sources is needed for a cold cache.
Tokenizer resources and NLTK sentence-tokenization data are separate runtime
dependencies and may also require network access on first use.

Sources, attribution and licensing information for all four inputs are in
[DATA_LICENSES.md](DATA_LICENSES.md), installed at `/app/DATA_LICENSES.md`.

## License

Adapter code: Apache 2.0 — see repository root `LICENSE`.
Downloaded data retains its upstream licensing; see [DATA_LICENSES.md](DATA_LICENSES.md).
Vendored NVIDIA RULER scripts retain their original NVIDIA copyright.

## Model and tokenizer authentication

For EvalHub jobs, set `model.auth.secret_ref` to the existing model credential
Secret. The adapter uses the SDK's mounted `api-key` credential, including the
sidecar reference token, for model requests. Keep the configured proxy URL;
the sidecar resolves the reference to the actual model credential.

For Kubernetes jobs using the local sidecar, an API key is optional when the
sidecar supplies ServiceAccount authentication. The adapter recognizes this
route by the shared loopback origin of the model and callback URLs with
`EVALHUB_MODE=k8s`, and creates the client with a `local` placeholder. The sidecar replaces this
placeholder with its ServiceAccount token before forwarding. Direct model
endpoints still require credentials.

The Secret's `hf-token` is passed to the tokenizer precheck and inherited by
synthetic data-generation subprocesses as `HF_TOKEN` and
`HUGGING_FACE_HUB_TOKEN`. Tokens are not passed as CLI arguments.

For direct model endpoints, `MODEL_API_KEY` or `OPENAI_API_KEY` remains supported
when no SDK credential is available. Existing HF token environment variables
remain supported when no mounted `hf-token` is available.

## Per-example diagnostics

The result directory includes `samples.jsonl` alongside `summary.csv` and
`results.json`. When OCI export is configured, all three files are included in
the result artifact before temporary predictions are deleted.

Each JSONL row contains `job_id`, `benchmark_id`, `model_name`, `task_id`,
`context_length`, the original sample `index`, `input` (full prompt), `outputs`
(expected answers, including expected variable names for variable tracking), and
`pred` (model answer). It also includes `metric_name`, `scorer`, `score` (0–1),
and `reference_matches` (one case-insensitive substring-match flag per expected
answer, in the same order as `outputs`). Indices can repeat across tasks/context
lengths; use `(task_id, context_length, index)` to identify a sample.

Scores use the same upstream metric function as aggregate scoring:
`string_match_all` gives partial credit for the fraction of expected answers
found, while QA uses `string_match_part` and gives credit when any expected
answer is found. These are substring metrics, not exact answer comparisons.
The upstream scorer rounds percentages to two decimals, so averaging rounded
sample scores can differ slightly from the aggregate score. Diagnostics are
saved for successfully completed evaluations; they do not recover samples from
previous runs whose temporary prediction files were already deleted.
