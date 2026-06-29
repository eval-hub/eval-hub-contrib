# DeepEval Adapter for eval-hub

This directory is the **eval-hub community adapter** for **[DeepEval](https://github.com/confident-ai/deepeval)**, an open-source LLM evaluation framework. DeepEval provides a suite of metrics for evaluating LLM outputs, including faithfulness, answer relevancy, hallucination detection, factual correctness, and summarization quality. It uses an LLM-as-judge approach where a separate model scores the outputs.

**What this adapter does:** It plugs DeepEval's metrics into **evalhub-sdk**'s `FrameworkAdapter` contract. Eval-hub supplies a **JobSpec** (from a mounted job file in Kubernetes or `EVALHUB_JOB_SPEC_PATH` locally). The adapter loads test data from CSV, JSONL, or JSON files, constructs DeepEval test case objects (single-turn `LLMTestCase` or multi-turn `ConversationalTestCase`), runs the appropriate metric, maps results into `JobResults` / `EvaluationResult` metrics, and reports progress to the eval-hub sidecar.

## Available Benchmarks

### Single-Turn Benchmarks

| Benchmark ID | Category | Description | Metrics |
|---|---|---|---|
| `faithfulness` | rag-evaluation | Tests if output is faithful to provided context | `faithfulness_score`, `claims_count`, `supported_claims_count` |
| `relevancy` | rag-evaluation | Tests if output is relevant to the input query | `relevancy_score` |
| `hallucination` | safety | Detects hallucinated content not grounded in context | `hallucination_score`, `hallucination_detected` |
| `correctness` | accuracy | Tests factual correctness against expected output | `correctness_score` |
| `summarization` | nlp | Tests summarization quality | `summarization_score` |

### Multi-Turn Benchmarks

| Benchmark ID | Category | Description | Metrics |
|---|---|---|---|
| `conversation-completeness` | multi-turn | Tests if a chatbot addresses all user needs across a conversation | `conversation_completeness_score` |
| `role-adherence` | multi-turn | Tests if a chatbot stays in its assigned persona throughout a conversation | `role_adherence_score` |
| `knowledge-retention` | multi-turn | Tests if a chatbot retains information disclosed by the user in earlier turns | `knowledge_retention_score` |

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | Create a virtualenv: `python3 -m venv .venv && pip install -r requirements.txt` |
| **DeepEval** | Installed via `requirements.txt` (`deepeval>=2.0.0`) |
| **Judge model API key** | An OpenAI or Anthropic API key for the judge model (set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) |
| **Test dataset** | CSV, JSONL, or JSON files with required columns per benchmark (see below) |

### Required Dataset Columns — Single-Turn

| Benchmark | Required Columns |
|---|---|
| `faithfulness` | `input`, `actual_output`, `retrieval_context` |
| `relevancy` | `input`, `actual_output` |
| `hallucination` | `input`, `actual_output`, `context` |
| `correctness` | `input`, `actual_output`, `expected_output` |
| `summarization` | `input`, `actual_output` |

### Required Dataset Columns — Multi-Turn

Multi-turn benchmarks use `ConversationalTestCase` and require a `turns` field. **JSONL or JSON are the recommended formats** — using CSV requires `turns` to be a JSON-encoded string.

| Benchmark | Required Columns | Optional Columns |
|---|---|---|
| `conversation-completeness` | `turns` | `chatbot_role`, `scenario`, `expected_outcome` |
| `role-adherence` | `turns`, `chatbot_role` | `scenario` |
| `knowledge-retention` | `turns` | `chatbot_role`, `scenario` |

**JSONL format (recommended):**

```jsonl
{"turns": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Click 'Forgot password' on the login page."}]}
{"turns": [{"role": "user", "content": "My account is locked."}, {"role": "assistant", "content": "I can help you unlock it."}]}
```

**JSONL with `chatbot_role` (required for `role-adherence`):**

```jsonl
{"turns": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help?"}], "chatbot_role": "friendly customer support agent"}
```

**CSV format (turns must be a JSON-encoded string):**

```csv
turns,chatbot_role
"[{""role"": ""user"", ""content"": ""Hello""}, {""role"": ""assistant"", ""content"": ""Hi!""}]",support agent
```

Each turn must have `role` (`"user"` or `"assistant"`) and `content` (string). Optional per-turn fields such as `retrieval_context` and `tools_called` are supported by DeepEval but not required by these benchmarks.

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
  "benchmark_id": "faithfulness",
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
| `eval_model_name` | string | No | model.name | Judge model name (e.g. `microsoft-phi-4`, `gpt-4o`). Defaults to the evaluated model. |
| `eval_model_url` | string | No | model.url | Judge model base URL. Defaults to the evaluated model's URL. |
| `threshold` | float | No | `0.5` | Minimum pass threshold for metric scores |
| `dataset_format` | string | No | `csv` | Input dataset format: `csv`, `jsonl`, or `json`. Use `jsonl` for multi-turn benchmarks. |
| `data_dir` | string | No | - | Path to dataset directory (overridden by `/test_data` or `/data` mounts) |
| `chatbot_role` | string | No | - | Chatbot persona for Role Adherence (e.g. `"helpful support agent"`). Can also be set per-record in the dataset. |
| `per_attempt_timeout_seconds` | float | No | `300.0` | Per-attempt timeout for each LLM judge call. Default accommodates reasoning models (e.g. DeepSeek-R1, Phi-4) with long chain-of-thought output. Reduce for faster non-reasoning judges. |
| `retry_max_attempts` | int | No | `2` | Total LLM call attempts per metric (first attempt + retries). |
| `retry_cap_seconds` | float | No | `5.0` | Maximum backoff (seconds) between retry attempts. |
| `max_concurrent` | int | No | `1` | Maximum test cases evaluated concurrently. Lower values reduce burst load on the judge endpoint. |
| `throttle_value` | float | No | `0` | Seconds to wait between test case evaluations. Use with `max_concurrent` to pace a rate-limited endpoint. |

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
