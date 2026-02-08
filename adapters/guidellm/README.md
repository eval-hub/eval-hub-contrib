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

#### Performance Sweep

```json
{
  "id": "job-123",
  "benchmark_id": "performance_sweep",
  "model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "url": "http://localhost:8000/v1"
  },
  "benchmark_config": {
    "profile": "sweep",
    "max_seconds": 30,
    "data": "prompt_tokens=256,output_tokens=128",
    "detect_saturation": true
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
  "benchmark_config": {
    "profile": "throughput",
    "max_seconds": 60,
    "data": "hf:abisee/cnn_dailymail",
    "data_args": {"name": "3.0.0"},
    "data_column_mapper": {"text_column": "article"},
    "data_samples": 100
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
  "benchmark_config": {
    "profile": "concurrent",
    "rate": 10,
    "max_requests": 100,
    "data": "prompt_tokens=512,output_tokens=256",
    "warmup": "10%"
  },
  "callback_url": "http://localhost:8080"
}
```

## Supported Request Payload Parameters

The following parameters can be specified in the `benchmark_config` section or as top-level job parameters when submitting evaluation jobs.

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `profile` | Execution profile (sweep, throughput, concurrent, constant, poisson, synchronous) | `sweep` |
| `num_examples` | Limit evaluation to N requests (used as fallback for `max_requests`) | None |
| `max_requests` | Maximum number of requests to send (overrides `num_examples` if specified) | None |
| `max_seconds` | Maximum duration in seconds | None |
| `max_errors` | Error threshold before stopping | None |
| `rate` | Rate value (meaning varies by profile) | Profile-dependent |
| `warmup` | Warmup period to exclude (percentage or absolute, e.g., "10%" or "10") | None |
| `cooldown` | Cooldown period to exclude (percentage or absolute, e.g., "10%" or "10") | None |
| `detect_saturation` | Enable over-saturation detection | `false` |

**Note on `num_examples` vs `max_requests`:**
- `num_examples` is a common parameter across all EvalHub providers for limiting evaluations
- For GuideLLM, if `max_requests` is not specified in `benchmark_config`, the adapter automatically uses `num_examples` as the limit
- If both are specified, `max_requests` takes precedence
- This provides a consistent interface while supporting provider-specific parameters

### Data Sources

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data` | Data source specification | `"prompt_tokens=256,output_tokens=128"` |
| `data_args` | Arguments for dataset loading | `{"name": "3.0.0"}` |
| `data_column_mapper` | Column mapping for datasets | `{"text_column": "article"}` |
| `data_samples` | Maximum number of samples to use | `100` |
| `processor` | Tokenizer/processor for synthetic data | `"gpt2"` |

### Request Configuration

| Parameter | Description | Options |
|-----------|-------------|---------|
| `request_type` | API endpoint type | `chat_completions`, `completions`, `audio_transcription`, `audio_translation` |

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
        "profile": "sweep",
        "num_examples": 100,
        "max_seconds": 300,
        "data": "prompt_tokens=256,output_tokens=128",
        "request_type": "chat_completions",
        "detect_saturation": true
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
        "profile": "throughput",
        "num_examples": 500,
        "max_seconds": 600,
        "data": "prompt_tokens=512,output_tokens=256",
        "request_type": "chat_completions",
        "detect_saturation": true,
        "warmup": "10%",
        "cooldown": "10%"
      }
    }
  ],
  "timeout_minutes": 60,
  "retry_attempts": 1
}
```

### Using max_requests (Provider-Specific)

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
        "profile": "concurrent",
        "rate": 10,
        "max_requests": 200,
        "data": "prompt_tokens=512,output_tokens=256",
        "warmup": 20
      }
    }
  ],
  "timeout_minutes": 45,
  "retry_attempts": 1
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
        "profile": "throughput",
        "num_examples": 100,
        "max_seconds": 300,
        "data": "hf:abisee/cnn_dailymail",
        "data_args": {"name": "3.0.0"},
        "data_column_mapper": {"text_column": "article"},
        "data_samples": 100,
        "request_type": "chat_completions"
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
