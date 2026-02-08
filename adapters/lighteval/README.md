# LightEval Adapter for eval-hub

This adapter integrates [LightEval](https://github.com/huggingface/lighteval) with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

## Overview

LightEval is a lightweight evaluation framework for language models that supports:

- **Multiple model providers**: Transformers, vLLM, OpenAI, Anthropic, custom endpoints
- **Wide range of benchmarks**: HellaSwag, ARC, MMLU, TruthfulQA, GSM8K, and many more
- **Few-shot evaluation**: Configurable number of few-shot examples
- **Efficient evaluation**: Optimised for speed and resource usage

This adapter implements the `FrameworkAdapter` pattern from evalhub-sdk, enabling seamless integration with the eval-hub service.

## Architecture

The adapter follows the eval-hub framework adapter pattern with automatic configuration:

1. **Settings-based configuration**: Runtime settings loaded automatically from environment
2. **Automatic JobSpec loading**: Job configuration auto-loaded from mounted ConfigMap
3. **Callback-based communication**: Progress updates and artifacts sent to sidecar via callbacks
4. **Synchronous execution**: The entire job lifetime is defined by the `run_benchmark_job()` method
5. **OCI artifact persistence**: Results persisted as OCI artifacts via the sidecar
6. **Structured results**: Returns `JobResults` with standardised metrics

```
┌─────────────────────┐
│   ConfigMap         │
│   (JobSpec)         │
└──────┬──────────────┘
       │ mounted at /etc/eval-job/spec.json
       ▼
┌─────────────────────────────────────────┐
│   Kubernetes Job Pod                    │
│  ┌────────────────┐  ┌───────────────┐ │
│  │ Adapter        │  │ Sidecar       │ │
│  │ Container      │──│ Container     │ │
│  │                │  │               │ │
│  │ - Load JobSpec │  │ - Receive     │ │
│  │ - Run LightEval│  │   status      │ │
│  │ - Report via   │  │ - Persist     │ │
│  │   callbacks    │  │   artifacts   │ │
│  └────────────────┘  └───────────────┘ │
└─────────────────────────────────────────┘
```

## Supported Benchmarks

The adapter supports all LightEval tasks, organised by category:

### Commonsense Reasoning
- HellaSwag
- WinoGrande
- OpenBookQA
- ARC Easy

### Scientific Reasoning
- ARC Easy
- ARC Challenge

### Physical Commonsense
- PIQA

### Truthfulness
- TruthfulQA (multiple choice)
- TruthfulQA (generation)

### Mathematics
- GSM8K
- MATH (various subcategories)

### Knowledge
- MMLU
- TriviaQA

### Language Understanding
- GLUE benchmarks

## Supported Model Providers

- **transformers**: HuggingFace Transformers models
- **vllm**: vLLM inference engine
- **openai**: OpenAI API
- **anthropic**: Anthropic API
- **endpoint**: Custom OpenAI-compatible endpoints
- **litellm**: LiteLLM proxy

## Supported Request Payload Parameters

The following parameters can be specified in the `benchmark_config` section or as top-level job parameters when submitting evaluation jobs:

### Model Configuration

- **`provider`** (string, default: `"endpoint"`): Model provider type
  - `"endpoint"`: OpenAI-compatible API endpoint (uses LiteLLM)
  - `"openai"`: OpenAI API
  - `"anthropic"`: Anthropic API
  - `"transformers"`: HuggingFace Transformers (local)
  - `"vllm"`: vLLM inference server
  - `"litellm"`: LiteLLM proxy

### Evaluation Configuration

- **`num_examples`** (integer, optional): Limit evaluation to N samples per task
  - Useful for quick testing or resource-constrained environments
  - Example: `100` evaluates only 100 samples per task
  - If not specified, evaluates the full dataset
  - Passed via `--max-samples` to LightEval CLI

- **`num_few_shot`** (integer, default: `0`): Number of few-shot examples to include
  - `0`: Zero-shot evaluation
  - `1-5`: Few-shot evaluation with N examples
  - Higher values may improve accuracy but increase latency

- **`batch_size`** (integer, default: `1`): Batch size for evaluation
  - Higher values may improve throughput
  - Limited by available memory

- **`tasks`** (array of strings, optional): Specific tasks to run
  - If not specified, runs all tasks in the benchmark category
  - Example: `["hellaswag", "winogrande", "piqa"]`

### Provider-Specific Parameters

#### Transformers Provider

- **`device`** (string, optional): Device to use for inference
  - `"cuda"`: GPU acceleration
  - `"cpu"`: CPU-only inference
  - `"cuda:0"`, `"cuda:1"`: Specific GPU device

#### Endpoint/API Providers

- **`parameters`** (object, optional): Additional parameters passed to the model API
  - Example: `{"temperature": 0.7, "top_p": 0.9, "max_tokens": 100}`
  - These are appended to the model arguments for LiteLLM

### Example Payloads

#### Quick Test with Limited Samples

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "commonsense_reasoning",
      "provider_id": "lighteval",
      "parameters": {
        "provider": "endpoint",
        "num_few_shot": 0,
        "num_examples": 100
      }
    }
  ],
  "timeout_minutes": 60,
  "retry_attempts": 1
}
```

#### Few-Shot Evaluation

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "mmlu",
      "provider_id": "lighteval",
      "parameters": {
        "provider": "endpoint",
        "num_few_shot": 5,
        "num_examples": 500,
        "batch_size": 4
      }
    }
  ],
  "timeout_minutes": 120,
  "retry_attempts": 1
}
```

#### Specific Tasks with Custom Parameters

```json
{
  "model": {
    "url": "http://vllm-server.evalhub.svc.cluster.local:8000",
    "name": "tinyllama"
  },
  "benchmarks": [
    {
      "id": "custom_suite",
      "provider_id": "lighteval",
      "parameters": {
        "provider": "endpoint",
        "tasks": ["hellaswag", "winogrande", "arc:easy"],
        "num_few_shot": 3,
        "num_examples": 200,
        "parameters": {
          "temperature": 0.0,
          "top_p": 1.0
        }
      }
    }
  ],
  "timeout_minutes": 90,
  "retry_attempts": 1
}
```

## Usage

### Building the Container

```bash
docker build -t lighteval-adapter:latest .
```

### Running Locally

For local testing without Kubernetes, run `main.py` directly with environment variables:

```bash
# Set environment for local mode
export EVALHUB_MODE=local
export EVALHUB_JOB_SPEC_PATH=meta/job.json
export REGISTRY_URL=localhost:5000
export REGISTRY_INSECURE=true
export SERVICE_URL=http://localhost:8080  # Optional: if mock service is running

# Run the adapter
python main.py
```

Alternatively, for container testing with a custom job spec:

```bash
# Create a test job spec
cat > /tmp/job-spec.json <<EOF
{
  "id": "test-job-123",
  "benchmark_id": "hellaswag",
  "model": {
    "name": "gpt2",
    "url": "http://localhost:8000/v1"
  },
  "num_examples": 10,
  "benchmark_config": {
    "provider": "endpoint",
    "num_few_shot": 0,
    "random_seed": 42,
    "batch_size": 1,
    "parameters": {
      "temperature": 0.0,
      "max_tokens": 100
    }
  },
  "callback_url": "http://localhost:8080"
}
EOF

# Run the adapter container
docker run \
  -e EVALHUB_MODE=k8s \
  -e REGISTRY_URL=localhost:5000 \
  -e REGISTRY_INSECURE=true \
  -v /tmp/job-spec.json:/meta/job.json \
  lighteval-adapter:latest
```

### Benchmark Configuration

#### Single Task

```json
{
  "id": "job-123",
  "benchmark_id": "hellaswag",
  "model": {
    "name": "gpt2",
    "url": "http://localhost:8000/v1"
  },
  "benchmark_config": {
    "provider": "endpoint",
    "num_few_shot": 0
  },
  "callback_url": "http://localhost:8080"
}
```

#### Multiple Tasks

```json
{
  "id": "job-123",
  "benchmark_id": "commonsense_reasoning",
  "model": {
    "name": "gpt2",
    "url": "http://localhost:8000/v1"
  },
  "benchmark_config": {
    "provider": "endpoint",
    "tasks": ["hellaswag", "winogrande", "piqa"],
    "num_few_shot": 5
  },
  "callback_url": "http://localhost:8080"
}
```

#### Category-based

```json
{
  "id": "job-123",
  "benchmark_id": "commonsense_reasoning",
  "model": {
    "name": "gpt2",
    "url": "http://localhost:8000/v1"
  },
  "benchmark_config": {
    "provider": "endpoint"
  },
  "callback_url": "http://localhost:8080"
}
```

This will run all tasks in the "commonsense_reasoning" category.
