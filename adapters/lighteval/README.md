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

- **`parameters`** (object, optional): Additional model configuration parameters passed through to LightEval's `model_args` string. Each key-value pair is appended as `key=value` to the model arguments used by LightEval's LiteLLM model config. This only applies to API-based providers (`endpoint`, `openai`, `anthropic`, `litellm`).

For the full list of supported fields, see the [LiteLLMModelConfig reference](https://huggingface.co/docs/lighteval/v0.13.0/en/package_reference/models#lighteval.models.endpoints.litellm_model.LiteLLMModelConfig). Examples:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | string | None | System prompt prepended to all requests |
| `concurrent_requests` | int | 10 | Max parallel API requests |
| `max_model_length` | int | None | Max context length |
| `generation_parameters` | string or object | None | Generation settings. Accepts brace notation string `"{temperature:0.1,max_new_tokens:512}"` or a YAML/JSON object (converted to brace notation automatically) |

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

#### CLI with Job Config File

To run an evaluation using `evalhub eval run --config job.yaml`, create a YAML config file with model arguments specified under `parameters.parameters`:

```yaml
# job.yaml
model:
  name: openai/meta-llama/Llama-3.2-3B-Instruct
  url: http://vllm-server:8000/v1
benchmarks:
  - id: math
    provider_id: lighteval
    parameters:
      provider: endpoint
      num_examples: 5
      num_few_shot: 3
      parameters:
        # Object format (recommended):
        generation_parameters:
          temperature: 0.1
          max_new_tokens: 512
          stop_tokens:
          - "\n"
          - "###"
        concurrent_requests: 7
        system_prompt: "You are a helpful math tutor."
```

The same configuration can also be expressed with `generation_parameters` as a pre-formatted brace notation string:

```yaml
      parameters:
        generation_parameters: "{temperature:0.1,max_new_tokens:512,stop_tokens:[\"\\n\", \"###\"]}"
        concurrent_requests: 7
        system_prompt: "You are a helpful math tutor."
```

```bash
evalhub eval run --config job.yaml
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

## LightEval Results and Eval Card Fields

### How the Adapter Processes LightEval Output

LightEval writes a `results_*.json` file to the output directory. The adapter reads this file and extracts:

- **Metrics** from `results` -- each `task|N` key maps to metric name/value pairs
- **Task configuration** from `config_tasks` -- dataset repos, subsets, few-shot counts
- **Generation parameters** from `config_general.model_config.generation_parameters`
- **Sample count** from `config_general.max_samples`

### Sample LightEval Results

#### Single-task benchmark (`benchmark_id=gsm8k`, zero-shot)

```json
{
  "config_general": {
    "lighteval_sha": "?",
    "num_fewshot_seeds": 1,
    "max_samples": 5,
    "model_config": {
      "model_name": "openai/llama3.2:3b-instruct-q4_K_M",
      "generation_parameters": {
        "temperature": 0,
        "max_new_tokens": null,
        "top_p": null,
        "seed": null
      }
    }
  },
  "results": {
    "gsm8k|0": {
      "extractive_match": 0.8,
      "extractive_match_stderr": 0.02
    },
    "all": {
      "extractive_match": 0.8,
      "extractive_match_stderr": 0.02
    }
  },
  "config_tasks": {
    "gsm8k|0": {
      "name": "gsm8k",
      "hf_repo": "openai/gsm8k",
      "hf_subset": "main",
      "hf_revision": null,
      "num_fewshots": 0
    }
  }
}
```

Corresponding `additional_info` produced by the adapter (`num_fewshots == 0` → `zero_shot`):

```json
"additional_info": {
  "dataset": [
    {
      "hf_repo": "openai/gsm8k",
      "hf_subset": "main",
      "sha": "740312add88f781978c0658806c59bc2815b9866"
    }
  ],
  "generation_parameters": {
    "temperature": 0
  },
  "zero_shot": 0.8
}
```

#### Multi-task benchmark (`benchmark_id=math`, 3-shot)

When `benchmark_id` maps to multiple tasks (e.g. `math` expands to `gsm8k`, `math:algebra`,
`math:counting_and_probability`), the results contain one entry per task plus an `all` aggregate:

```json
{
  "config_general": {
    "max_samples": 5,
    "model_config": {
      "model_name": "openai/llama3.2:3b-instruct-q4_K_M",
      "generation_parameters": {
        "temperature": 0.1,
        "max_new_tokens": 512,
        "top_p": null,
        "seed": null
      }
    }
  },
  "results": {
    "gsm8k|3": {
      "extractive_match": 1.0,
      "extractive_match_stderr": 0.0
    },
    "math:algebra|3": {
      "maj@n:n=4&...": 0.0,
      "maj@n:n=4&..._stderr": 0.0
    },
    "math:counting_and_probability|3": {
      "maj@n:n=4&...": 0.0,
      "maj@n:n=4&..._stderr": 0.0
    },
    "math:_average|3": {
      "maj@n:n=4&...": 0.0,
      "maj@n:n=4&..._stderr": 0.0
    },
    "all": {
      "extractive_match": 1.0,
      "extractive_match_stderr": 0.0,
      "maj@n:n=4&...": 0.0,
      "maj@n:n=4&..._stderr": 0.0
    }
  },
  "config_tasks": {
    "gsm8k|3": {
      "name": "gsm8k",
      "hf_repo": "openai/gsm8k",
      "hf_subset": "main",
      "num_fewshots": 3
    },
    "math:algebra|3": {
      "name": "math:algebra",
      "hf_repo": "DigitalLearningGmbH/MATH-lighteval",
      "hf_subset": "algebra",
      "num_fewshots": 3
    },
    "math:counting_and_probability|3": {
      "name": "math:counting_and_probability",
      "hf_repo": "DigitalLearningGmbH/MATH-lighteval",
      "hf_subset": "counting_and_probability",
      "num_fewshots": 3
    }
  }
}
```

Corresponding `additional_info` produced by the adapter (`num_fewshots == 3` → `alt_prompting`):

```json
"additional_info": {
  "alt_prompting": 0,
  "alt_prompting_description": "3-Shot",
  "dataset": [
    {
      "hf_repo": "openai/gsm8k",
      "hf_subset": "main",
      "sha": "740312add88f781978c0658806c59bc2815b9866"
    },
    {
      "hf_repo": "DigitalLearningGmbH/MATH-lighteval",
      "hf_subset": "algebra",
      "sha": "0530c78699ea5e8eb5530600900e1f328b48acad"
    },
    {
      "hf_repo": "DigitalLearningGmbH/MATH-lighteval",
      "hf_subset": "counting_and_probability",
      "sha": "0530c78699ea5e8eb5530600900e1f328b48acad"
    }
  ],
  "generation_parameters": {
    "max_new_tokens": 512,
    "temperature": 0.1
  }
}
```

### Fields the Adapter Extracts

| LightEval field | Adapter output | Description |
|---|---|---|
| `results.<task\|N>.<metric>` | `EvaluationResult.metric_name` | Normalised to `task.metric` (strips `\|N` suffix, simplifies metric names like `pass@k:k=1` to `pass@1`) |
| `results.<task\|N>.<metric>_stderr` | `EvaluationResult.confidence_interval` | Converted to a 95% CI: `value +/- 1.96 * stderr` |
| `config_general.max_samples` | `JobResults.num_examples_evaluated` | Number of samples evaluated per task |
| `config_tasks.<task>.hf_repo` | `additional_info.dataset[].hf_repo` | HuggingFace dataset repository |
| `config_tasks.<task>.hf_subset` | `additional_info.dataset[].hf_subset` | Dataset subset (e.g. `main`, `algebra`) |
| `config_tasks.<task>.num_fewshots` | `additional_info.zero_shot` or `additional_info.alt_prompting` | Determines prompting strategy (see below) |
| `config_general.model_config.generation_parameters` | `additional_info.generation_parameters` | Non-null generation parameters (temperature, max_new_tokens, etc.) |

### `additional_info` Structure

The `additional_info` dict is attached to `JobResults` and written into `results.json`. Its fields
are populated from LightEval's `config_tasks` and `config_general` sections.

#### `dataset`

An array of dataset references, one per HuggingFace repo + subset combination. The adapter resolves
the current commit SHA from HuggingFace Hub when available:

```json
"dataset": [
  {"hf_repo": "openai/gsm8k", "hf_subset": "main", "sha": "abc123..."},
  {"hf_repo": "DigitalLearningGmbH/MATH-lighteval", "hf_subset": "algebra", "sha": "def456..."}
]
```

If the SHA lookup fails (network error, private repo), the entry is included without a `sha` field.

#### `zero_shot` and `alt_prompting` (mutually exclusive)

These fields are **mutually exclusive** -- only one set is present in the output, never both.

The adapter reads `num_fewshots` from `config_tasks.<task>.num_fewshots`, which is set by LightEval
based on the `|N` suffix appended to each task in the CLI invocation (e.g. `gsm8k|3` means 3-shot).

- **Zero-shot** (`num_fewshots == 0`): only `zero_shot` is present, set to the overall score.

  ```json
  {"dataset": [...], "zero_shot": 0.78}
  ```

- **Few-shot** (`num_fewshots > 0`): only `alt_prompting` and `alt_prompting_description` are
  present. `alt_prompting` holds the overall score; `alt_prompting_description` is a human-readable
  label like `"3-Shot"` or `"5-Shot"`.

  ```json
  {"dataset": [...], "alt_prompting": 0.78, "alt_prompting_description": "3-Shot"}
  ```

**How N-shot information is collected:** The adapter passes `num_few_shot` from the job parameters
to LightEval as a per-task suffix (e.g. `gsm8k|3`). After evaluation, LightEval records the actual
fewshot count in `config_tasks.<task>.num_fewshots`. The adapter reads this value back from the
results rather than relying on the input parameter, so the recorded value reflects what LightEval
actually used.

#### `generation_parameters`

Non-null generation parameters extracted from the model config. Only parameters with actual values
are included; null-valued parameters are omitted:

```json
"generation_parameters": {"temperature": 0.1, "max_new_tokens": 512, "seed": 42}
```

If all generation parameters are null, this key is omitted entirely.

### Limitations: CoT and Multi-Turn Detection

Chain-of-Thought (CoT) and multi-turn evaluation strategies **cannot be reliably detected** from
LightEval results alone. LightEval does not expose structured metadata indicating whether a task
used CoT prompting or multi-turn interaction -- these are embedded in the task's prompt function
and are not surfaced in the results JSON.

<!-- Enhancement: CoT and multi-turn could be supported as explicit user-provided parameters
     (e.g. `"prompting_strategy": "cot"` or `"multi_turn": true` in the job parameters).
     The adapter would pass these through to additional_info without needing to infer them
     from the framework output. -->
