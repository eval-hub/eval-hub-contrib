# IBM CLEAR Adapter for eval-hub

Integrates **[IBM CLEAR](https://github.com/IBM/CLEAR)** with eval-hub via **evalhub-sdk** `FrameworkAdapter`: agentic pipeline on trace JSON (e.g. MLflow-style), `JobResults`, optional MLflow logging, optional OCI export.

**Input**: `/test_data` or `/data` (Eval Hub may stage S3 `test_data_ref` first), or `parameters.data_dir` / `traces_input_dir`. **Output**: metrics from `clear_results.json`; MLflow when `experiment.name` or `parameters.mlflow_experiment_name` is set.

| Field | Value |
|--------|--------|
| Provider id | `ibm-clear` |
| Benchmark id | `agentic-evaluation` |

## Dependencies

`requirements.txt`: **eval-hub-sdk[adapter]**, **IBM CLEAR** (GitHub archive; PyPI wheels lack full agentic), **pandas**, **pydantic**.

## Image

```bash
cd adapters/clear
podman build -f Containerfile -t quay.io/evalhub/community-ibm-clear:latest .
```

From repo root: `podman build -f eval-hub-contrib/adapters/clear/Containerfile -t quay.io/evalhub/community-ibm-clear:latest eval-hub-contrib/adapters/clear`

Bump the CLEAR URL in `requirements.txt` when updating IBM/CLEAR. Pin the image tag in your eval-hub provider definition when you want immutable pulls.

## Local run

```bash
cd adapters/clear
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json
python main.py
```

`meta/job.json` uses placeholders for `model.url`, S3 `test_data_ref`, and `secret_ref`—replace for your cluster. For `inference_backend: litellm`, set `OPENAI_API_KEY` if your judge needs it.

## `parameters.inference_backend`

- **`litellm`** (default): sets `OPENAI_BASE_URL` from `model.url`; optional `parameters.openai_api_key` (do not commit secrets).
- **`endpoint`**: uses `parameters.inference_url` or `model.url`; does not set `OPENAI_*` for LiteLLM.

## MLflow

Set `MLFLOW_TRACKING_URI` on the runtime; set `experiment.name` or `parameters.mlflow_experiment_name` on the job. Uploads `clear_results.json` and `metrics_summary.json` when enabled.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Adapter |
| `Containerfile` | Image |
| `requirements.txt` | Deps |
| `meta/job.json` | Example JobSpec |

## References

- [IBM CLEAR](https://github.com/IBM/CLEAR)
- [eval-hub](https://github.com/eval-hub/eval-hub) / [eval-hub-sdk](https://github.com/eval-hub/eval-hub-sdk)
