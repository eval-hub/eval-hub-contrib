# MTEB Adapter for eval-hub

This adapter integrates [MTEB (Massive Text Embedding Benchmark)](https://github.com/embeddings-benchmark/mteb) with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

## Overview

MTEB is a comprehensive benchmark for evaluating text embedding models across diverse tasks:

- **Semantic Textual Similarity (STS)**: Measure semantic similarity between sentence pairs
- **Retrieval**: Evaluate document retrieval capabilities
- **Classification**: Text classification tasks
- **Clustering**: Document clustering evaluation
- **Reranking**: Passage reranking tasks
- **Bitext Mining**: Parallel sentence mining
- **Pair Classification**: Sentence pair classification

This adapter implements the `FrameworkAdapter` pattern from evalhub-sdk, enabling seamless integration with the eval-hub service for embedding model evaluation.

## Architecture

The adapter follows the eval-hub framework adapter pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MTEBAdapter                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   JobSpec    │───▶│   Validate   │───▶│  Build CLI   │                   │
│  │   (input)    │    │   Config     │    │   Command    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                           │
│                                                 ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Report     │◀───│   Execute    │◀───│   MTEB CLI   │                   │
│  │   Progress   │    │  Subprocess  │    │  `mteb run`  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                                               │
│         ▼                   ▼                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Callbacks   │    │ Parse Results│───▶│  Normalize   │                   │
│  │  (sidecar)   │    │    Files     │    │ to JobResults│                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                           │
│                                                 ▼                           │
│                                          ┌──────────────┐                   │
│                                          │ OCI Artifact │                   │
│                                          │  Persistence │                   │
│                                          └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `MTEBAdapter` | Main adapter class extending `FrameworkAdapter` |
| `run_benchmark_job()` | Orchestrates the complete evaluation lifecycle |
| `_validate_config()` | Validates JobSpec fields and benchmark_config |
| `_resolve_tasks()` | Resolves benchmark_id to task list (presets or explicit) |
| `_build_mteb_command()` | Constructs MTEB CLI command from configuration |
| `_run_mteb_subprocess()` | Executes MTEB CLI with timeout and error handling |
| `_parse_results()` | Reads JSON results from MTEB output directory |
| `_extract_evaluation_results()` | Converts MTEB results to `EvaluationResult` objects |
| `_compute_overall_score()` | Aggregates task scores into single metric |
| `_save_detailed_results()` | Persists results for OCI artifact creation |

## Supported Benchmarks

### Benchmark Presets

The adapter provides convenient benchmark presets that expand to curated task lists:

| benchmark_id | Tasks |
|--------------|-------|
| `mteb_sts` | STS12, STS13, STS14, STS15, STS16, STS17, STSBenchmark, SICK-R |
| `mteb_retrieval` | NFCorpus, SciFact, ArguAna, TRECCOVID, Touche2020 |
| `mteb_classification` | AmazonReviewsClassification, Banking77Classification, EmotionClassification |
| `mteb_clustering` | ArxivClusteringP2P, ArxivClusteringS2S, BiorxivClusteringP2P |
| `mteb_reranking` | AskUbuntuDupQuestions, MindSmallReranking, SciDocsRR |

### Supported Task Types

You can also filter tasks by type using the `task_types` parameter:

- `BitextMining`
- `Classification`
- `Clustering`
- `InstructionRetrieval`
- `MultilabelClassification`
- `PairClassification`
- `Reranking`
- `Retrieval`
- `STS`
- `Summarization`

## Configuration

### JobSpec Structure

```json
{
  "id": "mteb-job-abc123",
  "benchmark_id": "mteb_sts",
  "model": {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "url": null
  },
  "num_examples": null,
  "benchmark_config": {
    "tasks": ["STS12", "STS13", "STS14"],
    "task_types": null,
    "task_categories": null,
    "languages": ["eng"],
    "batch_size": 32,
    "device": null,
    "verbosity": 2,
    "co2_tracker": false,
    "overwrite_results": true
  },
  "callback_url": "http://localhost:8080",
  "timeout_seconds": 7200
}
```

### benchmark_config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `list[str]` | `null` | Explicit list of MTEB task names |
| `task_types` | `list[str]` | `null` | Filter by task type (e.g., `["STS", "Retrieval"]`) |
| `task_categories` | `list[str]` | `null` | Filter by category (e.g., `["s2s", "p2p"]`) - maps to `--categories` |
| `languages` | `list[str]` | `["eng"]` | Language codes to include |
| `batch_size` | `int` | `32` | Batch size for encoding |
| `device` | `str` | `null` | Device override (`cuda`, `cpu`, `mps`, `cuda:0`) |
| `verbosity` | `int` | `2` | MTEB verbosity level (0-3) |
| `co2_tracker` | `bool` | `false` | Enable CO2 emissions tracking |
| `overwrite_results` | `bool` | `true` | Overwrite existing results |

## Usage

### Building the Container

```bash
# Using podman
podman build -t mteb-adapter:latest -f Containerfile .

# Using docker
docker build -t mteb-adapter:latest -f Containerfile .
```

### Running Locally

For local testing without Kubernetes:

```bash
# Set environment for local mode
export EVALHUB_MODE=local
export EVALHUB_JOB_SPEC_PATH=meta/job.json
export LOG_LEVEL=DEBUG

# Run the adapter directly
python main.py
```

### Running with Container

```bash
# Create a test job spec (or use meta/job.json)
cat > /tmp/job.json <<EOF
{
  "id": "test-mteb-001",
  "benchmark_id": "mteb_sts",
  "model": {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "url": null
  },
  "benchmark_config": {
    "tasks": ["STS12"],
    "batch_size": 32
  },
  "callback_url": "http://localhost:8080",
  "timeout_seconds": 3600
}
EOF

# Run the container
podman run --rm \
  -v /tmp/job.json:/meta/job.json:ro \
  -e EVALHUB_MODE=k8s \
  -e LOG_LEVEL=INFO \
  mteb-adapter:latest
```

### Example Configurations

#### STS Benchmark Suite

```json
{
  "id": "mteb-sts-full",
  "benchmark_id": "mteb_sts",
  "model": {
    "name": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "benchmark_config": {
    "batch_size": 64,
    "languages": ["eng"]
  },
  "callback_url": "http://localhost:8080",
  "timeout_seconds": 3600
}
```

#### Single Task Evaluation

```json
{
  "id": "mteb-banking77",
  "benchmark_id": "Banking77Classification",
  "model": {
    "name": "BAAI/bge-large-en-v1.5"
  },
  "benchmark_config": {
    "batch_size": 32,
    "device": "cuda"
  },
  "callback_url": "http://localhost:8080",
  "timeout_seconds": 1800
}
```

#### Task Type Filter

```json
{
  "id": "mteb-retrieval-all",
  "benchmark_id": "retrieval_evaluation",
  "model": {
    "name": "intfloat/e5-large-v2"
  },
  "benchmark_config": {
    "task_types": ["Retrieval"],
    "languages": ["eng"],
    "batch_size": 16,
    "device": "cuda:0"
  },
  "callback_url": "http://localhost:8080",
  "timeout_seconds": 14400
}
```

#### Multi-Task Custom Evaluation

```json
{
  "id": "mteb-custom-suite",
  "benchmark_id": "custom_benchmark",
  "model": {
    "name": "intfloat/multilingual-e5-large"
  },
  "benchmark_config": {
    "tasks": [
      "STS12",
      "STS13",
      "Banking77Classification",
      "ArxivClusteringP2P",
      "AskUbuntuDupQuestions"
    ],
    "batch_size": 16,
    "device": "cuda:0",
    "co2_tracker": true
  },
  "callback_url": "http://localhost:8080",
  "timeout_seconds": 7200
}
```

## Output Artifacts

The adapter persists the following files as OCI artifacts:

| File | Description |
|------|-------------|
| `mteb_results.json` | Raw MTEB output for all evaluated tasks |
| `results.json` | Structured results in eval-hub format |
| `summary.txt` | Human-readable evaluation summary |

### Structured Results Format

```json
{
  "job_id": "mteb-sts-001",
  "benchmark_id": "mteb_sts",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "framework": "mteb",
  "framework_version": "1.14.0",
  "overall_score": 0.823,
  "num_tasks": 8,
  "results": [
    {
      "metric_name": "STS12.test.main_score",
      "metric_value": 0.847,
      "metric_type": "float",
      "metadata": {
        "task": "STS12",
        "split": "test",
        "languages": ["eng-Latn"]
      }
    }
  ]
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EVALHUB_MODE` | Execution mode (`k8s` or `local`) | `local` |
| `EVALHUB_JOB_SPEC_PATH` | Path to job specification JSON | `/meta/job.json` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `REGISTRY_URL` | OCI registry URL for artifact storage | - |
| `REGISTRY_USERNAME` | Registry authentication username | - |
| `REGISTRY_PASSWORD` | Registry authentication password | - |
| `REGISTRY_INSECURE` | Allow insecure HTTP registry | `false` |

## Supported Models

The adapter works with any model supported by MTEB, including:

### Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### BGE Models
- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-base-en-v1.5`
- `BAAI/bge-large-en-v1.5`

### E5 Models
- `intfloat/e5-small-v2`
- `intfloat/e5-base-v2`
- `intfloat/e5-large-v2`
- `intfloat/multilingual-e5-large`

### Other Models
- Any model compatible with Sentence Transformers
- Custom models with `encode()` method

## References

- [MTEB Documentation](https://embeddings-benchmark.github.io/mteb/)
- [MTEB GitHub Repository](https://github.com/embeddings-benchmark/mteb)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [eval-hub-sdk Documentation](https://github.com/eval-hub/eval-hub-sdk)
- [Sentence Transformers](https://www.sbert.net/)
