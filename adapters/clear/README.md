# IBM CLEAR Adapter for eval-hub

This directory is an **eval-hub community adapter** for **[IBM CLEAR](https://github.com/IBM/CLEAR)** (Comprehensive LLM Error Analysis and Reporting). CLEAR runs an **agentic**, step-by-step pipeline over **JSON traces** (for example MLflow-style agent traces). It uses an LLM-as-judge to find recurring failure patterns and writes a structured report, mainly **`clear_results.json`**.

**What this adapter does:** It plugs that pipeline into **evalhub-sdk**’s `FrameworkAdapter` contract. Eval-hub supplies a **JobSpec** (from a mounted job file in Kubernetes or `EVALHUB_JOB_SPEC_PATH` locally). The adapter resolves where traces live, runs CLEAR, reads **`clear_results.json`**, maps CLEAR’s statistics into **`JobResults`** / **`EvaluationResult`** metrics, reports progress to the eval-hub sidecar, and optionally pushes artifacts to **MLflow** or an **OCI** bundle when the job requests it.

**Typical flow:**

1. **Input traces** — Prefer `/test_data` or `/data` when Eval Hub has staged data (e.g. from S3 `test_data_ref`), or set `parameters.data_dir` / `traces_input_dir` to a directory of `*.json` traces.
2. **Configuration** — Job parameters drive CLEAR (`eval_model_name`, `provider`, `inference_backend`, frameworks, etc.); `model.url` is used as the OpenAI-compatible endpoint when using the default LiteLLM-backed path.
3. **Execution** — CLEAR prepares trace data, runs the step-by-step agentic pipeline, then the adapter locates **`clear_results.json`** (standard layout under `step_by_step/clear_results/…` or a few fallbacks).
4. **Output** — Metrics (interactions, issues, agent scores, etc.) are returned to eval-hub; intermediate CLEAR directories can be trimmed while keeping a final **`clear_results.json`** under the run output.

| Field | Value |
|--------|--------|
| Provider id | `ibm-clear` |
| Benchmark id | `agentic-evaluation` |

## Dependencies

`requirements.txt` pins **eval-hub-sdk[adapter]**, **IBM CLEAR** from a **GitHub archive** (PyPI `clear-eval` wheels do not ship the full agentic stack), plus **pandas** and **pydantic**.

## Image

```bash
cd adapters/clear
podman build -f Containerfile -t quay.io/evalhub/community-ibm-clear:latest .
```

From repo root: `podman build -f eval-hub-contrib/adapters/clear/Containerfile -t quay.io/evalhub/community-ibm-clear:latest eval-hub-contrib/adapters/clear`

Bump the CLEAR archive URL in `requirements.txt` when you adopt a newer IBM/CLEAR revision. Pin the container image tag in your eval-hub provider definition when you want immutable pulls.

## Local run

```bash
cd adapters/clear
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json
python main.py
```

The example **`meta/job.json`** uses placeholders for `model.url`, S3 **`test_data_ref`**, and **`secret_ref`**—replace those for your environment. If you use **`inference_backend: litellm`**, set **`OPENAI_API_KEY`** in the environment when your judge endpoint requires it.

## `parameters.inference_backend`

- **`litellm`** (default): sets **`OPENAI_BASE_URL`** from **`model.url`** so CLEAR’s LiteLLM calls target your OpenAI-compatible gateway.
- **`endpoint`**: does not set **`OPENAI_*`** env vars; CLEAR uses **`parameters.inference_url`** or **`model.url`** as the inference endpoint.

## MLflow

Configure **`MLFLOW_TRACKING_URI`** (and any other MLflow env) on the runtime. On the job, set top-level **`experiment.name`** or **`parameters.mlflow_experiment_name`**. When set, the adapter logs **`clear_results.json`** and a **`metrics_summary.json`** alongside the standard eval-hub MLflow flow.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Adapter implementation and CLI entrypoint |
| `Containerfile` | OCI image (UBI Python, `EVALHUB_MODE=k8s`) |
| `requirements.txt` | Python dependencies |
| `meta/job.json` | Example JobSpec for local or reference |

## References

- [IBM CLEAR](https://github.com/IBM/CLEAR)
- [eval-hub](https://github.com/eval-hub/eval-hub) / [eval-hub-sdk](https://github.com/eval-hub/eval-hub-sdk)
