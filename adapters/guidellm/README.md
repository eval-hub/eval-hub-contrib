# GuideLLM Adapter for eval-hub

This adapter integrates [GuideLLM](https://github.com/vllm-project/guidellm) with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

## Overview

GuideLLM is a performance benchmarking platform designed to evaluate language model inference servers under realistic production conditions. It provides:

- **Multiple execution profiles**: Sweep, throughput, concurrent, constant, poisson, synchronous
- **Comprehensive metrics**: Time to First Token (TTFT), Inter-Token Latency (ITL), end-to-end latency, throughput
- **Flexible data sources**: Synthetic data generation, HuggingFace datasets, local files
- **Rich reporting**: JSON, CSV, HTML, and YAML output formats with detailed visualizations

This adapter implements the `FrameworkAdapter` pattern from evalhub-sdk, enabling seamless integration with the eval-hub service.

## Architecture

The adapter follows the eval-hub framework adapter pattern with automatic configuration:

1. **Settings-based configuration**: Runtime settings loaded automatically from environment
2. **Automatic JobSpec loading**: Job configuration auto-loaded from mounted ConfigMap
3. **Callback-based communication**: Progress updates and artifacts sent to sidecar via callbacks
4. **Synchronous execution**: The entire job lifetime is defined by the `run_benchmark_job()` method
5. **OCI artifact persistence**: Results persisted as OCI artifacts via the sidecar
6. **Structured results**: Returns `JobResults` with standardised performance metrics
7. **CLI passthrough**: Benchmark `parameters` map directly to GuideLLM CLI flags (see [Command-line parameters](#command-line-parameters))

## Execution Profiles

GuideLLM supports multiple load patterns for different testing scenarios:

- **sweep**: Automatically explore different request rates to find safe operating ranges
- **throughput**: Maximum capacity testing to identify performance limits
- **concurrent**: Simulate parallel users with fixed concurrency level
- **constant**: Fixed requests per second for steady-state testing
- **poisson**: Randomized request rates following Poisson distribution
- **synchronous**: Sequential requests for baseline measurements

## Performance Metrics

The adapter collects and reports comprehensive performance metrics:

- **Time to First Token (TTFT)**: Latency until first token generation
- **Inter-Token Latency (ITL)**: Time between subsequent tokens
- **End-to-end latency**: Complete request processing time
- **Throughput**: Requests per second, tokens per second
- **Token counts**: Prompt tokens and generated tokens
- **Latency distributions**: Percentiles and statistical measures

## Supported Backends

- OpenAI-compatible endpoints (vLLM, Text Generation Inference, etc.)
- Any HTTP API following OpenAI's chat completions or completions format

## Usage

### Building the Container

```bash
make image-guidellm
```

### Running Locally

For local testing without Kubernetes:

```bash
# Set environment for local mode
export EVALHUB_MODE=local
export EVALHUB_JOB_SPEC_PATH=meta/job.json
export SERVICE_URL=http://localhost:8080  # Optional: if mock service is running

# Run the adapter
python main.py
```

### Benchmark Configuration

GuideLLM jobs are configured through the `parameters` object on each benchmark in an eval-hub evaluation request. See [Command-line parameters](#command-line-parameters) for how those keys map to the GuideLLM CLI.

#### Performance Sweep

```json
{
  "id": "job-123",
  "benchmark_id": "performance_sweep",
  "model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "url": "http://localhost:8000/v1"
  },
  "parameters": {
    "--profile": "sweep",
    "--max-seconds": 30,
    "--data": "prompt_tokens=256,output_tokens=128",
    "--detect-saturation": true
  },
  "callback_url": "http://localhost:8080"
}
```

#### Throughput Test

```json
{
  "id": "job-123",
  "benchmark_id": "max_throughput",
  "model": {
    "name": "gpt-3.5-turbo",
    "url": "http://localhost:8000/v1"
  },
  "parameters": {
    "--profile": "throughput",
    "--max-seconds": 60,
    "--data": "hf:abisee/cnn_dailymail",
    "--data-args": {"name": "3.0.0"},
    "--data-column-mapper": {"text_column": "article"},
    "--data-samples": 100
  },
  "callback_url": "http://localhost:8080"
}
```

#### Concurrent Users

```json
{
  "id": "job-123",
  "benchmark_id": "concurrent_users",
  "model": {
    "name": "llama-2-7b",
    "url": "http://localhost:8000/v1"
  },
  "parameters": {
    "--profile": "concurrent",
    "--rate": 10,
    "--max-requests": 100,
    "--data": "prompt_tokens=512,output_tokens=256",
    "--warmup": "10"
  },
  "callback_url": "http://localhost:8080"
}
```

## Command-line parameters

The adapter builds the GuideLLM subprocess command from the eval-hub job `parameters` object. **Keys must be exact GuideLLM CLI flag names**, including the leading `--` (for example `"--max-seconds"`, not `max_seconds`).

When you submit an evaluation through the eval-hub API, each benchmark entry includes a `parameters` map. The adapter reads that map and passes each entry through to `guidellm benchmark` with minimal transformation:

| Value type in `parameters` | How it becomes a CLI argument |
|----------------------------|-------------------------------|
| string, number | `--flag value` |
| `true` | `--flag` (boolean flag, no value) |
| `false` or `null` | omitted |
| object (dict) | `--flag '{"key": "value"}'` (JSON-serialized) |

Example API payload fragment:

```json
{
  "model": {
    "url": "http://vllm-server:8000/v1",
    "name": "meta-llama/Llama-3.1-8B-Instruct"
  },
  "benchmarks": [
    {
      "id": "quick",
      "provider_id": "guidellm",
      "parameters": {
        "--profile": "constant",
        "--rate": 5,
        "--max-requests": 20,
        "--max-seconds": 60,
        "--data": "prompt_tokens=256,output_tokens=128",
        "--request-type": "chat_completions",
        "--detect-saturation": false
      }
    }
  ]
}
```

This produces a command equivalent to:

```bash
guidellm benchmark \
  --target http://vllm-server:8000/v1 \
  --output-path /tmp/guidellm_results_... \
  --outputs json,csv,html,yaml \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --profile constant \
  --rate 5 \
  --max-requests 20 \
  --max-seconds 60 \
  --data prompt_tokens=256,output_tokens=128 \
  --request-type chat_completions \
  --processor gpt2
```

(`--detect-saturation` is omitted because the value is `false`.)

### Adapter defaults and precedence

The adapter supplies a small set of defaults when you do not include the corresponding flag in `parameters`. If you **do** include the flag, your value wins.

| Flag | Source when omitted | Overridable via `parameters` |
|------|---------------------|------------------------------|
| `--target` | `model.url` | Yes |
| `--model` | `model.name` (if set) | Yes |
| `--output-path` | Adapter-managed temp directory | Yes |
| `--outputs` | `json,csv,html,yaml` | Yes |
| `--profile` | `sweep` | Yes |
| `--data` | `prompt_tokens=256,output_tokens=128` | Yes |
| `--request-type` | `chat_completions` | Yes |
| `--processor` | `gpt2` when `--data` uses synthetic `prompt_tokens=…,output_tokens=…` format | Yes |

Merge order is: **adapter defaults, then `parameters`**. Any key present in `parameters` replaces the adapter default for that flag.

All other GuideLLM flags are caller-supplied only. Use the exact flag names from `guidellm benchmark --help` — there is no snake_case conversion or eval-hub alias layer (for example, pass `"--max-requests"` directly; `num_examples` is not mapped automatically).

### Authentication

If the job mounts model credentials, the adapter may merge an API key into `--backend-kwargs` at runtime. If you already pass `--backend-kwargs` in `parameters`, the adapter merges `api_key` into your object rather than replacing it.

## Supported Request Payload Parameters

The following are commonly used GuideLLM flags passed through `parameters`. For the full set, refer to [GuideLLM documentation](https://github.com/vllm-project/guidellm) and `guidellm benchmark --help`.

### Core Parameters

| Parameter | Description | Adapter default |
|-----------|-------------|-----------------|
| `--profile` | Execution profile (sweep, throughput, concurrent, constant, poisson, synchronous) | `sweep` |
| `--max-requests` | Maximum number of requests to send | None |
| `--max-seconds` | Maximum duration in seconds | None |
| `--max-errors` | Error threshold before stopping | None |
| `--rate` | Rate value (meaning varies by profile) | None |
| `--warmup` | Warmup period to exclude | None |
| `--cooldown` | Cooldown period to exclude | None |
| `--detect-saturation` | Enable over-saturation detection (`true` / `false`) | omitted (`false`) |

### Data Sources

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--data` | Data source specification | `"prompt_tokens=256,output_tokens=128"` |
| `--data-args` | Arguments for dataset loading (object) | `{"name": "3.0.0"}` |
| `--data-column-mapper` | Column mapping for datasets (object) | `{"text_column": "article"}` |
| `--data-samples` | Maximum number of samples to use | `100` |
| `--processor` | HuggingFace tokenizer for synthetic data | `"google/flan-t5-small"`, `"gpt2"` |

**Note on `--processor`:**
- Required by GuideLLM when using synthetic data (`--data: "prompt_tokens=X,output_tokens=Y"`)
- Must be a valid HuggingFace model ID
- Defaults to `"gpt2"` when omitted and synthetic data is in use
- Not needed for HuggingFace datasets, files, and other non-synthetic sources

### Request Configuration

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--request-type` | API endpoint type | `chat_completions`, `completions`, `audio_transcription`, `audio_translation` |
| `--backend-kwargs` | Backend-specific options (object) | `{"validate_backend": false}` for servers without `/health` |

## Example Payloads

### Quick Performance Test with Limited Requests

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "sweep",
      "provider_id": "guidellm",
      "parameters": {
        "--profile": "sweep",
        "--max-requests": 100,
        "--max-seconds": 300,
        "--data": "prompt_tokens=256,output_tokens=128",
        "--request-type": "chat_completions",
        "--detect-saturation": true,
        "--processor": "google/flan-t5-small"
      }
    }
  ],
  "timeout_minutes": 30,
  "retry_attempts": 1
}
```

### Throughput Test with Dataset Limiting

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "throughput",
      "provider_id": "guidellm",
      "parameters": {
        "--profile": "throughput",
        "--max-requests": 500,
        "--max-seconds": 600,
        "--data": "prompt_tokens=512,output_tokens=256",
        "--request-type": "chat_completions",
        "--detect-saturation": true,
        "--warmup": "10",
        "--cooldown": "10",
        "--processor": "google/flan-t5-small"
      }
    }
  ],
  "timeout_minutes": 60,
  "retry_attempts": 1
}
```

### Concurrent Load Test

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "concurrent",
      "provider_id": "guidellm",
      "parameters": {
        "--profile": "concurrent",
        "--rate": 10,
        "--max-requests": 200,
        "--data": "prompt_tokens=512,output_tokens=256",
        "--warmup": 20,
        "--processor": "google/flan-t5-small"
      }
    }
  ],
  "timeout_minutes": 45,
  "retry_attempts": 1
}
```

### Ollama / servers without `/health`

```json
{
  "model": {
    "url": "http://localhost:11434/v1",
    "name": "llama3"
  },
  "benchmarks": [
    {
      "id": "local",
      "provider_id": "guidellm",
      "parameters": {
        "--profile": "synchronous",
        "--max-requests": 2,
        "--data": "prompt_tokens=30,output_tokens=10",
        "--request-type": "chat_completions",
        "--backend-kwargs": {"validate_backend": false}
      }
    }
  ]
}
```

### Using HuggingFace Dataset

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "throughput",
      "provider_id": "guidellm",
      "parameters": {
        "--profile": "throughput",
        "--max-requests": 100,
        "--max-seconds": 300,
        "--data": "hf:abisee/cnn_dailymail",
        "--data-args": {"name": "3.0.0"},
        "--data-column-mapper": {"text_column": "article"},
        "--data-samples": 100,
        "--request-type": "chat_completions"
      }
    }
  ],
  "timeout_minutes": 60,
  "retry_attempts": 1
}
```

## Output Artifacts

GuideLLM generates multiple output formats automatically persisted as OCI artifacts:

- **benchmarks.json**: Complete authoritative record with all metrics and sample requests
- **benchmarks.csv**: Tabular view for spreadsheets and BI tools
- **benchmarks.html**: Visual summary with latency distributions and interactive charts
- **benchmarks.yaml**: Human-readable alternative to JSON format
