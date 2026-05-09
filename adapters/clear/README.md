# IBM CLEAR Adapter for eval-hub

This directory is the **eval-hub community adapter** for **[IBM CLEAR](https://github.com/IBM/CLEAR)** (**Comprehensive LLM Error Analysis and Reporting**). CLEAR is meant for **agentic** workloads: you give it **execution traces** (what the agent and LLMs actually did), and it runs a pipeline that uses a **separate judge LLM** to classify problems, compare behavior across many runs, and summarize **where** failures cluster—not just that a single request failed. That makes it easier to reason about **patterns** (repeatable issue types, weak spots in the workflow) instead of reading traces one by one.

**Scope of this integration:** CLEAR’s agentic path here is aligned with **LangGraph**: your application is modeled as a **graph**, each **node** is a step or role (planner, tools, sub-agents, etc.), and traces record LLM calls **under** those nodes. The adapter assumes you can produce traces in that shape—typically **JSON** exported from runs that use **MLflow-compatible autologging**, as described in [`examples/docs/02-agent-traces.md`](examples/docs/02-agent-traces.md). That pairing matters because CLEAR’s reporting is built around **per-node** scoring and issue tables: you get dashboards that say how each **component** of the graph behaved across many traces, not only a single global score.

**Artifacts:** CLEAR writes **`clear_results.json`** (structured statistics and per-agent sections that this adapter maps into eval-hub **metrics**) and a **static HTML** summary (**`clear_results.html`**) for quick human review. Within Eval Hub, those artifacts can ride along with the job (and optionally **MLflow** or **OCI** uploads when configured), so evaluation stays repeatable and reviewable alongside the rest of your hub workflows.

**What this adapter does:** It plugs that pipeline into **evalhub-sdk**’s `FrameworkAdapter` contract. Eval-hub supplies a **JobSpec** (from a mounted job file in Kubernetes or `EVALHUB_JOB_SPEC_PATH` locally). The adapter resolves where traces live, runs CLEAR, reads **`clear_results.json`**, preserves the HTML dashboard for MLflow/OCI when applicable, maps CLEAR’s statistics into **`JobResults`** / **`EvaluationResult`** metrics, reports progress to the eval-hub sidecar, and optionally pushes artifacts to **MLflow** or an **OCI** bundle when the job requests it.

**Typical flow:**

1. **Input traces** — Prefer `/test_data` or `/data` when Eval Hub has staged data (e.g. from S3 `test_data_ref`), or set `parameters.data_dir` / `traces_input_dir` to a directory of `*.json` traces.
2. **Configuration** — Job parameters drive CLEAR (`eval_model_name`, `provider`, `inference_backend`, frameworks, etc.); `model.url` is used as the OpenAI-compatible endpoint when using the default LiteLLM-backed path.
3. **Execution** — CLEAR prepares trace data, runs the step-by-step agentic pipeline, writes **`clear_results.json`** and generates **dashboard HTML** (for example under `step_by_step/`); the adapter locates **`clear_results.json`** (standard layout under `step_by_step/clear_results/…` or a few fallbacks) and **copies key HTML** (e.g. **`clear_results.html`**) to the run root when needed for artifacts.
4. **Output** — Metrics (interactions, issues, agent scores, etc.) are returned to eval-hub; intermediate CLEAR directories can be trimmed while keeping a final **`clear_results.json`** and the preserved **HTML** report. Optional **`parameters.clear_dashboard_theme`** controls Red Hat styling on that HTML (see [`examples/docs/06-dashboard-theme.md`](examples/docs/06-dashboard-theme.md)).

| Field | Value |
|--------|--------|
| Provider id | `ibm-clear` |
| Benchmark ids | `agentic-evaluation` (default), `agentic-evaluation-custom-criteria`, `agentic-evaluation-predefined-issues` — see [`provider.yaml`](provider.yaml) and [`examples/docs/05-benchmarks-and-parameters.md`](examples/docs/05-benchmarks-and-parameters.md). |

**Hands-on tutorials** (local run, benchmarks, deployed Hub, MLflow env, theme) live under **`examples/`** — start at [`examples/README.md`](examples/README.md) (includes **first-time path**, sample **`input-traces/`**, and links to **`output/local/`** HTML/JSON snapshots). The sections below are **reference**; detail there avoids duplicating long step-by-step prose here.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.13+** | Create a virtualenv from `adapters/clear`: `python3 -m venv .venv && pip install -r requirements.txt` |
| **IBM CLEAR** | Installed from Git by `requirements.txt` (currently **`2.0.0-rc.2`**); PyPI wheels omit the agentic pipeline |
| **OpenAI-compatible inference endpoint** | Used as the **judge model** (`model.url`, e.g. an Ollama, vLLM, or OpenAI-API-compatible server). Supply `OPENAI_API_KEY` only when your endpoint requires it |
| **Agent traces** | One or more `*.json` trace files from a **LangGraph + MLflow-autologged** agent run — see [`examples/docs/02-agent-traces.md`](examples/docs/02-agent-traces.md) |
| **Eval Hub** *(deployed only)* | A running Eval Hub instance with the `ibm-clear` provider registered and traces staged from S3 — see [`examples/docs/04-deployed-eval-hub.md`](examples/docs/04-deployed-eval-hub.md) |

**How to read this README**

| If you want to… | Start here |
|-------------------|------------|
| **Tutorial** (CLEAR × Eval Hub concepts, local run, benchmarks, theme) | [`examples/README.md`](examples/README.md) |
| Run the adapter on your machine with trace files | [Local run](#local-run) |
| Understand S3 staging on eval-hub / Kubernetes | [Traces from S3](#traces-from-s3-deployed-eval-hub) |
| Call the eval-hub HTTP API | [Submit via eval-hub API](#submit-via-eval-hub-api-deployment) · more context in [`examples/docs/04-deployed-eval-hub.md`](examples/docs/04-deployed-eval-hub.md) |
| **`litellm`** and credentials (optional API key) | [`inference_backend` and credentials](#parametersinference_backend-and-credentials) |

## Traces from S3 (deployed eval-hub)

Eval-hub usually stages traces from object storage into the pod (`/test_data`, `/data`, etc.). This adapter scans for **`*.json`** files—same idea as local **`data_dir`**, but files arrive via staging. **Checklist and deployment-focused notes:** [`examples/docs/04-deployed-eval-hub.md`](examples/docs/04-deployed-eval-hub.md). **`meta/job.json`** (`test_data_ref.s3`) remains a shape reference; exact fields depend on your eval-hub version.

## Local run

**Full walkthrough:** [`examples/docs/03-local-run.md`](examples/docs/03-local-run.md).

Summary:

1. **Python env** — From `adapters/clear`: create a venv and `pip install -r requirements.txt`.
2. **Traces** — Point **`parameters.data_dir`** at a folder of CLEAR-compatible **`*.json`** traces.
3. **Model / `litellm`** — Set **`model.url`** (OpenAI-compatible **`/v1`** base). Use **`parameters.inference_backend`: `litellm`**. **API key:** Many **local** OpenAI-compatible servers (for example some **Ollama** setups) do **not** require **`OPENAI_API_KEY`**; set it **when your endpoint requires it**. On **Kubernetes / OpenShift**, prefer **`model.auth.secret_ref`** so the token lives in a **Secret** (mounted as **`api-key`**); do **not** put raw keys in **`parameters`**.
4. **Run** — `EVALHUB_MODE=local`, `EVALHUB_JOB_SPEC_PATH` pointing at your job JSON, then **`python main.py`**.

**Sample JobSpec — local traces + `litellm` + optional MLflow (placeholders):**

```json
{
  "id": "clear-local-001",
  "provider_id": "ibm-clear",
  "benchmark_id": "agentic-evaluation",
  "benchmark_index": 0,
  "experiment_name": "my-mlflow-experiment",
  "model": {
    "url": "https://your-inference-endpoint.example.com/v1",
    "name": "your-model-name"
  },
  "parameters": {
    "data_dir": "input-trace",
    "eval_model_name": "openai/your-model-name",
    "provider": "openai",
    "agent_framework": "langgraph",
    "observability_framework": "mlflow",
    "inference_backend": "litellm"
  },
  "callback_url": "http://localhost:8080"
}
```

```bash
cd adapters/clear
export EVALHUB_MODE=local
export EVALHUB_JOB_SPEC_PATH=meta/my-local-job.json
export MLFLOW_TRACKING_URI='https://your-mlflow-tracking.example.com/'
python main.py
```

MLflow troubleshooting and env vars: see [**MLflow**](#mlflow) below and [`examples/docs/03-local-run.md`](examples/docs/03-local-run.md).

## Submit via eval-hub API (deployment)

Use your deployed eval-hub **base URL** (dummy example: `https://evalhub.example.com`). The evalhub-sdk targets **`POST /api/v1/evaluations/jobs`**. The body below mirrors **`meta/job.json`** (same model URL, parameters, MLflow experiment name, S3 ref, and `litellm` + **`model.auth`**) — your server may accept this superset; if submission fails, drop fields until it matches your eval-hub version’s schema.

**MLflow:** **`experiment_name`** (e.g. `"clear-agentic-eval-example"` in the JSON below) is the **MLflow experiment** where this run’s results—including artifacts such as **`clear_results.json`**, **`metrics_summary.json`**, and HTML when configured—are written. You can use **`parameters.mlflow_experiment_name`** instead; see [MLflow](#mlflow) and [`examples/docs/03-local-run.md`](examples/docs/03-local-run.md).

**`inference_backend`:** Use **`litellm`**. Supply **`OPENAI_API_KEY`** only when your **`model.url`** endpoint requires it—on the cluster, prefer **`model.auth.secret_ref`** to a Secret (do not put raw keys in **`parameters`**).

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
        "data_dir": "input-trace",
        "eval_model_name": "openai/example-model",
        "provider": "openai",
        "agent_framework": "langgraph",
        "observability_framework": "mlflow",
        "inference_backend": "litellm"
      }
    }
  ],
  "test_data_ref": {
    "s3": {
      "bucket": "clear-traces",
      "path": "traces/",
      "secret_ref": {
        "name": "clear-traces-bucket",
        "namespace": "your-namespace"
      }
    }
  }
}'
```

With **`litellm`**, use **`model.auth.secret_ref`** when the worker needs an API key from a Secret; omit or adjust if your gateway does not require one.

## Image

```bash
cd adapters/clear
podman build -f Containerfile -t quay.io/evalhub/community-ibm-clear:latest .
```

From repo root: `podman build -f eval-hub-contrib/adapters/clear/Containerfile -t quay.io/evalhub/community-ibm-clear:latest eval-hub-contrib/adapters/clear`

Bump the CLEAR archive URL in `requirements.txt` when you adopt a newer IBM/CLEAR revision. Pin the container image tag in your eval-hub provider definition when you want immutable pulls.

## `parameters.inference_backend` and credentials

- **`litellm`** (default): The adapter sets **`OPENAI_BASE_URL`** from **`model.url`**. IBM CLEAR’s LiteLLM path uses **`OPENAI_API_KEY`** when your upstream requires it—**omit** it for many **local** OpenAI-compatible servers; set it when connecting to providers that enforce auth. **On Kubernetes / OpenShift**, prefer **`model.auth.secret_ref`** (Secret key **`api-key`**); for **local** runs you can export **`OPENAI_API_KEY`** or rely on no key if your server allows it. Never put raw keys in **`parameters`**.

- **`endpoint`** (legacy): Still accepted in code for backward compatibility but **deprecated** upstream—prefer **`litellm`** for new jobs. Details remain in **`main.py`** for maintainers.

## Model API key (Kubernetes, `litellm` + OpenAI-style usage)

Prefer **`model.auth.secret_ref`** on the job so credentials stay in a **Kubernetes Secret**, not in the job ConfigMap. Eval-hub mounts that secret (SDK path **`/var/run/secrets/model/api-key`**); the adapter uses **`evalhub.adapter.auth.resolve_model_credentials()`** and sets **`OPENAI_API_KEY`** when that key is present. For local development you can set **`OPENAI_API_KEY`** in the shell instead.

## MLflow

Configure **`MLFLOW_TRACKING_URI`** (and any other MLflow env) on the runtime. On the job, set **`experiment_name`** or **`parameters.mlflow_experiment_name`**. When set, the adapter uploads **`clear_results.json`**, **`metrics_summary.json`**, and HTML artifacts when present. **Local runs use the same rules** as cluster runs: upload is skipped if there is no experiment name or **`MLFLOW_TRACKING_URI`** is unset. Longer notes: [`examples/docs/03-local-run.md`](examples/docs/03-local-run.md).

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Adapter implementation and CLI entrypoint |
| `Containerfile` | OCI image (UBI Python, `EVALHUB_MODE=k8s`) |
| `requirements.txt` | Python dependencies |
| `meta/job.json` | Example JobSpec (`litellm`, S3 ref, **`model.auth.secret_ref`** placeholder) |
| `provider.yaml` | Provider + benchmark definition (eval-hub style) |
| `examples/README.md` | Tutorial index (`examples/docs/*.md`, notebook, benchmark samples) |

## References

- [IBM CLEAR](https://github.com/IBM/CLEAR)
- [eval-hub](https://github.com/eval-hub/eval-hub) / [eval-hub-sdk](https://github.com/eval-hub/eval-hub-sdk)
