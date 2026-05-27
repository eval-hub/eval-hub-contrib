# DeepEval Adapter for eval-hub

This directory is the **eval-hub community adapter** for **[DeepEval](https://github.com/confident-ai/deepeval)**, an open-source LLM evaluation framework. DeepEval provides a suite of metrics for evaluating LLM outputs, including faithfulness, answer relevancy, hallucination detection, factual correctness, and summarization quality. It uses an LLM-as-judge approach where a separate model scores the outputs.

**What this adapter does:** It plugs DeepEval's metrics into **evalhub-sdk**'s `FrameworkAdapter` contract. Eval-hub supplies a **JobSpec** (from a mounted job file in Kubernetes or `EVALHUB_JOB_SPEC_PATH` locally). The adapter loads test data from CSV, JSONL, or JSON files, constructs DeepEval `LLMTestCase` objects, runs the appropriate metric, maps results into `JobResults` / `EvaluationResult` metrics, and reports progress to the eval-hub sidecar.

## Available Benchmarks

| Benchmark ID | Category | Description | Metrics |
|---|---|---|---|
| `deepeval-faithfulness` | rag-evaluation | Tests if output is faithful to provided context | `faithfulness_score`, `claims_count`, `supported_claims_count` |
| `deepeval-relevancy` | rag-evaluation | Tests if output is relevant to the input query | `relevancy_score` |
| `deepeval-hallucination` | safety | Detects hallucinated content not grounded in context | `hallucination_score`, `hallucination_detected` |
| `deepeval-correctness` | accuracy | Tests factual correctness against expected output | `correctness_score` |
| `deepeval-summarization` | nlp | Tests summarization quality (alignment + coverage) | `summarization_score`, `alignment_score`, `coverage_score` |

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | Create a virtualenv: `python3 -m venv .venv && pip install -r requirements.txt` |
| **DeepEval** | Installed via `requirements.txt` (`deepeval>=2.0.0`) |
| **Judge model API key** | An OpenAI or Anthropic API key for the judge model (set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) |
| **Test dataset** | CSV, JSONL, or JSON files with required columns per benchmark (see below) |

### Required Dataset Columns

| Benchmark | Required Columns |
|---|---|
| `deepeval-faithfulness` | `input`, `actual_output`, `retrieval_context` |
| `deepeval-relevancy` | `input`, `actual_output` |
| `deepeval-hallucination` | `input`, `actual_output`, `context` |
| `deepeval-correctness` | `input`, `actual_output`, `expected_output` |
| `deepeval-summarization` | `input`, `actual_output` |

## Local Testing

1. **Python env:**

```bash
cd adapters/deepeval
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

2. **Prepare test data** in a directory (e.g. `test_data/data.csv`) with the columns your benchmark requires.

3. **Create a JobSpec** (or use `meta/job.json` as a starting point):

```json
{
  "id": "deepeval-local-001",
  "provider_id": "deepeval",
  "benchmark_id": "deepeval-faithfulness",
  "benchmark_index": 0,
  "model": {
    "url": "https://api.openai.com/v1",
    "name": "gpt-4o"
  },
  "parameters": {
    "eval_model_name": "gpt-4o",
    "threshold": 0.5,
    "dataset_format": "csv",
    "data_dir": "test_data"
  },
  "callback_url": "http://localhost:8080"
}
```

4. **Run:**

```bash
export EVALHUB_MODE=local
export EVALHUB_JOB_SPEC_PATH=meta/job.json
export OPENAI_API_KEY=your-key-here
python main.py
```

## Image Build

```bash
# From adapters/deepeval directory
podman build -f Containerfile -t quay.io/evalhub/community-deepeval:latest .

# From repo root
podman build -f adapters/deepeval/Containerfile -t quay.io/evalhub/community-deepeval:latest adapters/deepeval
```

From the repo root via Makefile:

```bash
make image-deepeval
make push-deepeval REGISTRY=quay.io/your-org VERSION=v1.0.0
```

## Parameters Reference

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `eval_model_name` | string | Yes | - | Judge model name (e.g. `gpt-4o`, `claude-sonnet-4-20250514`) |
| `threshold` | float | No | `0.5` | Minimum pass threshold for metric scores |
| `dataset_format` | string | No | `csv` | Input dataset format: `csv`, `jsonl`, or `json` |
| `data_dir` | string | No | - | Path to dataset directory (overridden by `/test_data` or `/data` mounts) |

## Layout

| Path | Purpose |
|---|---|
| `main.py` | Adapter implementation and CLI entrypoint |
| `Containerfile` | OCI image (UBI Python, `EVALHUB_MODE=k8s`) |
| `requirements.txt` | Python dependencies |
| `requirements-test.txt` | Test dependencies |
| `meta/job.json` | Example JobSpec (faithfulness benchmark) |
| `provider.yaml` | Provider + benchmark definition (eval-hub style) |
| `tests/` | pytest suite |

## References

- [DeepEval](https://github.com/confident-ai/deepeval)
- [DeepEval Documentation](https://docs.confident-ai.com)
- [eval-hub](https://github.com/eval-hub/eval-hub) / [eval-hub-sdk](https://github.com/eval-hub/eval-hub-sdk)
