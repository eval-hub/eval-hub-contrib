# GuideLLM Adapter for eval-hub

This adapter integrates [GuideLLM](https://github.com/vllm-project/guidellm) with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

## Overview

GuideLLM is a performance benchmarking platform designed to evaluate language model inference servers under realistic production conditions. The adapter exposes two selectable benchmarks:

- **Quick performance** (`quick_perf_test`): a short, bounded sweep for a fast performance snapshot.
- **Full performance** (`comprehensive_perf_test`): every GuideLLM traffic profile, run in sequence and aggregated into one result.

The adapter also provides comprehensive metrics (TTFT, ITL, end-to-end latency, throughput), flexible data sources (synthetic, HuggingFace, local files), and rich reporting (JSON, CSV, HTML, YAML).

This adapter implements the `FrameworkAdapter` pattern from evalhub-sdk, enabling seamless integration with the eval-hub service.

## Architecture

The adapter follows the eval-hub framework adapter pattern with automatic configuration:

1. **Settings-based configuration**: Runtime settings loaded automatically from environment
2. **Automatic JobSpec loading**: Job configuration auto-loaded from mounted ConfigMap
3. **Callback-based communication**: Progress updates and artifacts sent to sidecar via callbacks
4. **Synchronous execution**: The entire job lifetime is defined by the `run_benchmark_job()` method
5. **OCI artifact persistence**: Results persisted as OCI artifacts via the sidecar
6. **Structured results**: Returns `JobResults` with standardised performance metrics

## Benchmarks

| Benchmark id | Name | What it runs | Default bounds |
|--------------|------|--------------|----------------|
| `quick_perf_test` | Quick performance | One `sweep` profile | `max_seconds=30`, `max_requests=100` |
| `comprehensive_perf_test` | Full performance | `sweep`, `throughput`, `concurrent`, `constant`, `poisson`, sequentially | `max_seconds=60`, `max_requests=300` per profile |

Both presets default to synthetic data (`prompt_tokens=256,output_tokens=128`)
and `chat_completions` requests, and disable saturation detection.

### Overriding preset defaults

Preset values only fill parameters the job does not set, so any value in the
submitted `parameters` wins. In addition:

- `num_examples` takes precedence over the preset `max_requests` bound, as it
  does everywhere else in the adapter.
- `profiles: ["sweep", "poisson"]` on Full performance limits the suite to
  those profiles.
- `profile: "constant"` runs just that profile, for both benchmarks.
- Unknown benchmark ids keep the historical single-profile behaviour and
  default to `sweep`.
- Full performance sets a default `rate` for the profiles that need one
  (`concurrent=10`, `constant=5`, `poisson=5`) unless the job supplies a rate.

Supported GuideLLM profiles are `sweep`, `throughput`, `concurrent`,
`constant`, `poisson`, `synchronous`, and `async`. They differ in how load is
applied:

- **sweep**: Explore different request rates to find safe operating ranges
- **throughput**: Maximum capacity testing to identify performance limits
- **concurrent**: Simulate parallel users at a fixed concurrency level
- **constant**: Fixed requests per second for steady-state testing
- **poisson**: Randomized request rates following a Poisson distribution
- **synchronous**: Sequential requests for baseline measurements

### Result reporting for Full performance

Each profile writes its own output files under a profile subdirectory (for
example `sweep/benchmarks.json`). The job reports:

- one row per profile and metric, named `<profile>_<metric>` (for example
  `poisson_requests_per_second`), with the profile in the row metadata;
- the plain metric names (`requests_per_second`, `mean_ttft_ms`, ...) holding
  the average across the profiles that produced the metric, so the provider's
  primary score and pass criteria keep working;
- `evaluation_metadata.profiles` and `evaluation_metadata.profile_metrics` with
  the per-profile values used for the aggregate.

Quick performance reports the plain metric names for its single profile.

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

#### Quick performance

The preset supplies the sweep profile and its bounds, so a minimal benchmark
entry is enough:

```json
{
  "id": "job-123",
  "benchmark_id": "quick_perf_test",
  "model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "url": "http://localhost:8000/v1"
  },
  "parameters": {},
  "callback_url": "http://localhost:8080"
}
```

#### Full performance

```json
{
  "id": "job-123",
  "benchmark_id": "comprehensive_perf_test",
  "model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "url": "http://localhost:8000/v1"
  },
  "parameters": {
    "max_seconds": 120,
    "data": "prompt_tokens=512,output_tokens=256"
  },
  "callback_url": "http://localhost:8080"
}
```

#### Limiting Full performance to selected profiles

```json
{
  "id": "job-123",
  "benchmark_id": "comprehensive_perf_test",
  "model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "url": "http://localhost:8000/v1"
  },
  "parameters": {
    "profiles": ["sweep", "poisson"]
  },
  "callback_url": "http://localhost:8080"
}
```

## Supported Request Payload Parameters

The following parameters can be specified in the `parameters` section when submitting evaluation jobs.

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `profile` | Run a single execution profile (sweep, throughput, concurrent, constant, poisson, synchronous, async) | Preset-dependent |
| `profiles` | Run several execution profiles as a suite | Full preset: all five traffic profiles |
| `num_examples` | Limit evaluation to N requests (takes precedence over preset `max_requests`) | None |
| `max_requests` | Maximum number of requests to send per profile (overrides `num_examples` if specified) | Preset-dependent |
| `max_seconds` | Maximum duration in seconds per profile | Preset-dependent |
| `max_errors` | Error threshold before stopping | None |
| `rate` | Rate value (meaning varies by profile) | Profile/preset-dependent |
| `warmup` | Warmup period to exclude (percentage or absolute, e.g., "10%" or "10") | None |
| `cooldown` | Cooldown period to exclude (percentage or absolute, e.g., "10%" or "10") | None |
| `detect_saturation` | Enable over-saturation detection | `false` |

**Note on `num_examples` vs `max_requests`:**
- `num_examples` is a common parameter across all EvalHub providers for limiting evaluations
- For GuideLLM, if `max_requests` is not specified in `parameters`, the adapter automatically uses `num_examples` as the limit
- If both are specified, `max_requests` takes precedence
- This provides a consistent interface while supporting provider-specific parameters

### Data Sources

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data` | Data source specification | `"prompt_tokens=256,output_tokens=128"` |
| `data_args` | Arguments for dataset loading | `{"name": "3.0.0"}` |
| `data_column_mapper` | Column mapping for datasets | `{"text_column": "article"}` |
| `data_samples` | Maximum number of samples to use | `100` |
| `processor` | HuggingFace tokenizer for synthetic data generation (required for `prompt_tokens=X,output_tokens=Y` format) | `"google/flan-t5-small"`, `"gpt2"` |

**Note on `processor`:**
- Required when using synthetic data format (`data: "prompt_tokens=X,output_tokens=Y"`)
- Must be a valid HuggingFace model ID (e.g., `"google/flan-t5-small"`, `"meta-llama/Llama-3.1-8B-Instruct"`)
- Defaults to `"gpt2"` if not specified (widely available, small)
- Not needed when using actual datasets (HuggingFace datasets, files, etc.)

### Request Configuration

| Parameter | Description | Options |
|-----------|-------------|---------|
| `request_type` | API endpoint type | `chat_completions`, `completions`, `audio_transcription`, `audio_translation` |

## Example Payloads

### Quick performance with a request limit

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "quick_perf_test",
      "provider_id": "guidellm",
      "parameters": {
        "num_examples": 100,
        "processor": "google/flan-t5-small"
      }
    }
  ],
  "timeout_minutes": 30,
  "retry_attempts": 1
}
```

### Full performance with custom data

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "comprehensive_perf_test",
      "provider_id": "guidellm",
      "parameters": {
        "max_seconds": 120,
        "data": "prompt_tokens=512,output_tokens=256",
        "processor": "google/flan-t5-small"
      }
    }
  ],
  "timeout_minutes": 60,
  "retry_attempts": 1
}
```

### Full performance against a HuggingFace dataset

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "comprehensive_perf_test",
      "provider_id": "guidellm",
      "parameters": {
        "profiles": ["sweep", "throughput"],
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

Single-profile runs write these files at the results root. Multi-profile runs
write them under one subdirectory per profile (for example
`sweep/benchmarks.json`), and every file is persisted and reported to MLflow
with its relative path.
