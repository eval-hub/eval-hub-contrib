# RAGAS Adapter for EvalHub

[RAGAS](https://github.com/explodinggradients/ragas) (Retrieval Augmented Generation Assessment) adapter for the EvalHub evaluation service.

## Overview

This adapter evaluates RAG pipeline quality using RAGAS metrics. It connects to OpenAI-compatible model endpoints for LLM completions and embeddings, then runs RAGAS evaluation on datasets provided via EvalHub's data pipeline.

## Available Metrics

| Metric | Description |
|--------|-------------|
| `answer_relevancy` | How relevant the answer is to the question |
| `answer_similarity` | Semantic similarity between answer and reference |
| `context_precision` | Precision of retrieved context for answering |
| `context_recall` | How much of the reference is covered by context |
| `context_entity_recall` | Entity-level recall from context |
| `faithfulness` | Whether the answer is grounded in the context |
| `nv_accuracy` | Answer accuracy (class-based, ragas v0.4+) |
| `nv_context_relevance` | Context relevance (class-based, ragas v0.4+) |
| `factual_correctness` | Factual correctness of the answer |
| `noise_sensitivity` | Sensitivity to noise in context |
| `nv_response_groundedness` | Response groundedness (class-based, ragas v0.4+) |

## Dataset Format

The adapter expects JSONL or JSON datasets with RAGAS column names:

```jsonl
{"user_input": "What is AI?", "response": "Artificial Intelligence is...", "retrieved_contexts": ["AI is a field of..."], "reference": "AI stands for Artificial Intelligence"}
```

Use `column_map` in `parameters` to rename columns from your dataset format.

## Configuration

Key `parameters` parameters:

- `metrics`: List of metric names to evaluate
- `embedding_model`: Model name for embeddings (defaults to LLM model)
- `embedding_url`: Standalone endpoint override; Kubernetes uses the `embedding_url` Secret entry
- `judge_model`: Optional model used for LLM-based RAGAS metrics (defaults to the evaluated model)
- `judge_url`: Standalone endpoint override; Kubernetes uses the `judge_url` Secret entry
- `judge_api_key`: Standalone key override or explicit sidecar reference; Kubernetes normally reads `judge_api-key` from the projected Secret
- `max_tokens`: Max tokens for LLM completions
- `temperature`: Sampling temperature
- `column_map`: Map dataset columns to RAGAS names
- `data_path`: Explicit path to dataset file

### Kubernetes judge and embedding connections

Both RAG suites include metrics that call `/v1/embeddings`. A chat-only model
needs a separate embedding service. LLM-based metrics also need a judge that
supports structured JSON; set `judge_model` when that should differ from the
evaluated model.

Put connection URLs and credentials in the Secret referenced by
`model.auth.secret_ref`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ragas-model-connections
type: Opaque
stringData:
  api-key: "<target-api-key>"
  judge_api-key: "<judge-api-key>"
  judge_url: https://judge.example/v1
  embedding_api-key: "<embedding-api-key>"
  embedding_url: http://granite-embedding:8080/v1
```

Use the actual URLs and credentials from your deployment. An unauthenticated
embedding service can use a nonempty dummy `embedding_api-key` if it ignores
the Authorization header. Each role-specific URL requires its matching
`<role>_api-key` entry, even when the role and target share the same credential.

Select model names in the job's benchmark parameters:

```yaml
model:
  name: evaluated-chat-model
  url: https://target.example/v1
  auth:
    secret_ref: ragas-model-connections
benchmarks:
  - id: ragas_rag_default
    provider_id: ragas
    parameters:
      judge_model: deployed-judge-model
      embedding_model: granite-embedding
```

Supply the dataset as usual. EvalHub projects the Secret's URLs as localhost
sidecar URLs and its keys as reference tokens. The adapter reads those projected
entries automatically and sends `judge_api-key:ref` for judge requests and
`embedding_api-key:ref` for embedding requests. The sidecar selects the real
upstream and injects the real credential. Equal localhost URLs still represent
independent connections because the reference token controls routing.

With no role-specific Secret entries, that role reuses the primary model
connection (`api-key:ref`). A role-specific key without a URL selects a separate
credential on the primary endpoint. Projected URL paths are preserved.

In Kubernetes, parameter URL overrides must use the sidecar origin and key
overrides must be reference tokens. Put real upstream URLs and keys in the
Secret. `OPENAI_API_KEY` does not replace a projected role credential.

### Standalone endpoint overrides

Standalone/local runs retain `judge_url`, `judge_api_key`, and `embedding_url`
parameter overrides. A different endpoint uses its explicit key or the process's
`OPENAI_API_KEY`, falling back to a dummy key for unauthenticated endpoints.
It does not borrow the evaluated model's credential. Local model-auth directories
can also contain the role-specific URL/key files described above.

## Local Development

```bash
pip install -r requirements.txt
EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
```

## Testing

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -m integration
```

### Internal HTTPS endpoints

The adapter adds the mounted OpenShift service CA at
`/etc/pki/ca-trust/source/anchors/service-ca.crt` to the trust configuration of
its OpenAI clients, including embedding and judge clients. The existing HTTPX
trust configuration (`SSL_CERT_FILE`, `SSL_CERT_DIR`, or the default CA bundle)
is preserved. Certificate and hostname verification remain enabled.
