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
  "job_id": "test-job-123",
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
  }
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
  "job_id": "job-123",
  "benchmark_id": "hellaswag",
  "model": {
    "name": "gpt2",
    "url": "http://localhost:8000/v1"
  },
  "benchmark_config": {
    "provider": "endpoint",
    "num_few_shot": 0
  }
}
```

#### Multiple Tasks

```json
{
  "job_id": "job-123",
  "benchmark_id": "commonsense_reasoning",
  "model": {
    "name": "gpt2",
    "url": "http://localhost:8000/v1"
  },
  "benchmark_config": {
    "provider": "endpoint",
    "tasks": ["hellaswag", "winogrande", "piqa"],
    "num_few_shot": 5
  }
}
```

#### Category-based

```json
{
  "job_id": "job-123",
  "benchmark_id": "commonsense_reasoning",
  "model": {
    "name": "gpt2",
    "url": "http://localhost:8000/v1"
  },
  "benchmark_config": {
    "provider": "endpoint"
  }
}
```

This will run all tasks in the "commonsense_reasoning" category.
