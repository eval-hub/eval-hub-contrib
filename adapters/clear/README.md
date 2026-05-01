# IBM CLEAR Adapter for eval-hub

This directory is an **eval-hub community adapter** for **[IBM CLEAR](https://github.com/IBM/CLEAR)** (Comprehensive LLM Error Analysis and Reporting). CLEAR runs an **agentic**, step-by-step pipeline over **JSON traces** (for example MLflow-style agent traces). It uses an LLM-as-judge to find recurring failure patterns and writes a structured report, mainly **`clear_results.json`**.

**What this adapter does:** It plugs that pipeline into **evalhub-sdk**’s `FrameworkAdapter` contract. Eval-hub supplies a **JobSpec** (from a mounted job file in Kubernetes or `EVALHUB_JOB_SPEC_PATH` locally). The adapter resolves where traces live, runs CLEAR, reads **`clear_results.json`**, maps CLEAR’s statistics into **`JobResults`** / **`EvaluationResult`** metrics, reports progress to the eval-hub sidecar, and optionally pushes artifacts to **MLflow** or an **OCI** bundle when the job requests it.

**Typical flow:**

1. **Input traces** — Use **`parameters.mlflow_traces_experiment_name`** / **`mlflow_traces_experiment_id`** to pull traces from MLflow, or set **`parameters.data_dir`** / **`traces_input_dir`** to a directory of `*.json` traces, or rely on mounted **`/test_data`** / **`/data`** when the platform stages files into the pod.
2. **Configuration** — Job parameters drive CLEAR (`eval_model_name`, `provider`, `inference_backend`, frameworks, etc.); `model.url` is used as the OpenAI-compatible endpoint when using the default LiteLLM-backed path.
3. **Execution** — CLEAR prepares trace data, runs the step-by-step agentic pipeline, then the adapter locates **`clear_results.json`** (standard layout under `step_by_step/clear_results/…` or a few fallbacks).
4. **Output** — Metrics (interactions, issues, agent scores, etc.) are returned to eval-hub; intermediate CLEAR directories can be trimmed while keeping a final **`clear_results.json`** under the run output.

| Field | Value |
|--------|--------|
| Provider id | `ibm-clear` |
| Benchmark id | `agentic-evaluation` |

**How to read this README**

| If you want to… | Start here |
|-------------------|------------|
| Run the adapter on your machine with trace files | [Local run](#local-run) |
| Call the eval-hub HTTP API | [Submit via eval-hub API](#submit-via-eval-hub-api-deployment) |
| Choose **`litellm`** vs **`endpoint`** and where API keys live | [`inference_backend` and credentials](#parametersinference_backend-and-credentials) |

## Local run

1. **Python env** — From `adapters/clear`: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`), then `pip install -r requirements.txt`.
2. **Traces** — Set **`parameters.mlflow_traces_experiment_name`** (or **`mlflow_traces_experiment_id`**) and **`mlflow_tracking_uri`** / **`MLFLOW_TRACKING_URI`** to pull traces from MLflow. Alternatively use **`parameters.data_dir`** / **`traces_input_dir`** for a folder of **`*.json`** files, or mounted **`/test_data`** / **`/data`**.
3. **Model** — Set **`model.url`** to your OpenAI-compatible endpoint. For **`inference_backend: litellm`**, export **`OPENAI_API_KEY`** in the shell (or use **`endpoint`** and configure CLEAR per the section below).
4. **Run** — `export EVALHUB_MODE=local` and `export EVALHUB_JOB_SPEC_PATH=meta/job.json`, then `python main.py`.

Replace placeholders in **`meta/job.json`** (`model.url`, namespaces, MLflow experiment names, etc.) for your environment.

**`meta/job.json` and model credentials:** This file uses **`inference_backend`: `litellm`**. On **Kubernetes**, keep **`model.auth.secret_ref`** set to the name of a Secret that stores the model API token under the key **`api-key`** (eval-hub mounts it; the adapter calls **`resolve_model_credentials()`** and sets **`OPENAI_API_KEY`**). For **local** runs with **`litellm`**, you may **delete the whole `model.auth` object** and export **`OPENAI_API_KEY`** instead—do **not** put the raw key in **`parameters`**. If you switch **`inference_backend`** to **`endpoint`**, remove **`model.auth`**; CLEAR uses **`model.url`** (or **`parameters.inference_url`**) as the HTTP inference endpoint and this adapter does not inject an OpenAI API key.

**Sample JobSpec — traces from MLflow + `endpoint` + MLflow results (placeholders only):** Save the JSON below to a file (e.g. **`meta/my-local-job.json`**) and point **`EVALHUB_JOB_SPEC_PATH`** at it. Replace **`model.url`**, **`eval_model_name`**, **`mlflow_tracking_uri`**, **`mlflow_traces_experiment_name`** (where traces are ingested), and **`mlflow_results_experiment_name`** (where CLEAR logs outputs).

```json
{
  "id": "clear-local-001",
  "provider_id": "ibm-clear",
  "benchmark_id": "agentic-evaluation",
  "benchmark_index": 0,
  "model": {
    "url": "https://your-inference-endpoint.example.com/v1",
    "name": "your-model-name"
  },
  "parameters": {
    "mlflow_tracking_uri": "https://your-mlflow-host/",
    "mlflow_traces_experiment_name": "your-traces-experiment",
    "mlflow_results_experiment_name": "your-clear-results-experiment",
    "eval_model_name": "openai/your-model-name",
    "provider": "openai",
    "agent_framework": "langgraph",
    "observability_framework": "mlflow",
    "inference_backend": "endpoint"
  },
  "callback_url": "http://localhost:8080"
}
```

```bash
cd adapters/clear
export EVALHUB_MODE=local
export EVALHUB_JOB_SPEC_PATH=meta/my-local-job.json
export MLFLOW_TRACKING_URI='https://your-mlflow-tracking.example.com/'
# If your MLflow requires auth: export MLFLOW_TRACKING_TOKEN='...'
python main.py
```

With **`inference_backend`** **`endpoint`**, you do not need **`OPENAI_API_KEY`** for this adapter. If MLflow upload fails (TLS, auth, or client mismatch), set the **`MLFLOW_*`** variables your server documents, or **`EVALHUB_MLFLOW_BACKEND=upstream`** with **`pip install mlflow-skinny`**.

## Submit via eval-hub API (deployment)

Use your deployed eval-hub **base URL** (dummy example: `https://evalhub.example.com`). The evalhub-sdk targets **`POST /api/v1/evaluations/jobs`**. The body below mirrors **`meta/job.json`** (model URL, parameters, **`litellm`**, **`model.auth`**) — your server may accept this superset; if submission fails, drop fields until it matches your eval-hub version’s schema.

**MLflow:** Set **`parameters.mlflow_results_experiment_name`** (or legacy **`parameters.mlflow_experiment_name`**) on the benchmark job so the adapter knows where to log **`clear_results.json`** and the metrics summary; see [MLflow](#mlflow).

**`inference_backend`:** With **`endpoint`**, this adapter does not require an OpenAI API key. With **`litellm`**, store the key in a **Kubernetes Secret** and reference it via **`model.auth.secret_ref`** (do not put the key in JSON parameters).

Example (dummy host, token, and secret names):

```bash
curl -sS -X POST 'https://evalhub.example.com/api/v1/evaluations/jobs' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJub3QtYS1yZWFsLXRva2Vu' \
  -H 'X-Tenant: your-namespace' \
  -d '{
  "experiment_name": "clear-agentic-eval-example",
  "model": {
    "url": "http://127.0.0.1:8000/v1",
    "name": "example-model",
    "auth": {
      "secret_ref": "my-openai-api-key-secret"
    }
  },
  "benchmarks": [
    {
      "id": "agentic-evaluation",
      "provider_id": "ibm-clear",
      "parameters": {
        "mlflow_traces_experiment_name": "your-mlflow-traces-experiment",
        "mlflow_results_experiment_name": "clear-agentic-eval-example",
        "eval_model_name": "openai/example-model",
        "provider": "openai",
        "agent_framework": "langgraph",
        "observability_framework": "mlflow",
        "inference_backend": "litellm"
      }
    }
  ]
}'
```

When **`inference_backend`** is **`endpoint`**, omit **`model.auth`** (no OpenAI key). When it is **`litellm`**, keep **`model.auth.secret_ref`** pointing at the Secret that holds the **`api-key`** value the adapter resolves.

## Image

```bash
cd adapters/clear
podman build -f Containerfile -t quay.io/evalhub/community-ibm-clear:latest .
```

From repo root: `podman build -f eval-hub-contrib/adapters/clear/Containerfile -t quay.io/evalhub/community-ibm-clear:latest eval-hub-contrib/adapters/clear`

Bump the CLEAR archive URL in `requirements.txt` when you adopt a newer IBM/CLEAR revision. Pin the container image tag in your eval-hub provider definition when you want immutable pulls.

## `parameters.inference_backend` and credentials

- **`litellm`** (default): The adapter sets **`OPENAI_BASE_URL`** from **`model.url`**. For the **OpenAI**-style provider path, IBM CLEAR’s LiteLLM integration expects **`OPENAI_API_KEY`** to be set (see `llm_client.py` in IBM/CLEAR). **Provide the key** via:
  - **`model.auth.secret_ref`** pointing at a Kubernetes Secret whose mounted key **`api-key`** holds the token (recommended), and/or
  - **`OPENAI_API_KEY`** in the container environment (e.g. local runs).
  Do **not** put the raw key in **`parameters`** (it can end up in a ConfigMap). If no key is set, CLEAR may fail at runtime with a missing `OPENAI_API_KEY` error depending on provider/model.

- **`endpoint`**: The adapter **does not** set **`OPENAI_BASE_URL`** or **`OPENAI_API_KEY`** for LiteLLM. IBM CLEAR’s pipeline expects **`endpoint_url`** for this backend; the adapter sets **`endpoint_url`** and **`inference_url`** from **`model.url`**, or from **`parameters.endpoint_url`** / **`parameters.inference_url`** if present. Use this when you are not using the LiteLLM/OpenAI-env wiring and do not need an OpenAI API key in this adapter.

## Model API key (Kubernetes, `litellm` + OpenAI-style usage)

Prefer **`model.auth.secret_ref`** on the job so credentials stay in a **Kubernetes Secret**, not in the job ConfigMap. Eval-hub mounts that secret (SDK path **`/var/run/secrets/model/api-key`**); the adapter uses **`evalhub.adapter.auth.resolve_model_credentials()`** and sets **`OPENAI_API_KEY`** when that key is present. For local development you can set **`OPENAI_API_KEY`** in the shell instead.

## MLflow

Configure **`MLFLOW_TRACKING_URI`** (and any other MLflow env) on the runtime. On the job, set **`parameters.mlflow_results_experiment_name`** (or **`parameters.mlflow_experiment_name`**). When set, the adapter logs **`clear_results.json`** and a **`metrics_summary.json`** alongside the standard eval-hub MLflow flow.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Adapter implementation and CLI entrypoint |
| `Containerfile` | OCI image (UBI Python, `EVALHUB_MODE=k8s`) |
| `requirements.txt` | Python dependencies |
| `meta/job.json` | Example JobSpec (`litellm`, **`model.auth.secret_ref`** placeholder, MLflow parameters) |
| `provider.yaml` | Provider + benchmark definition (eval-hub style) |

## References

- [IBM CLEAR](https://github.com/IBM/CLEAR)
- [eval-hub](https://github.com/eval-hub/eval-hub) / [eval-hub-sdk](https://github.com/eval-hub/eval-hub-sdk)
