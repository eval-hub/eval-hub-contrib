# Example CLEAR jobs (three benchmarks)

These JSON files use the **same JobSpec shape as `meta/job.json`**: model **auth** secret ref (for cluster jobs), **`parameters.mlflow_traces_experiment_name`** (MLflow experiment that holds traces to ingest), **`parameters.mlflow_results_experiment_name`** (MLflow experiment for CLEAR outputs), optional **`mlflow_tracking_uri`**, and a **`callback_url`** for the sidecar when using Eval Hub.

| File | `benchmark_id` | Purpose |
|------|----------------|---------|
| `01-agentic-evaluation.json` | `agentic-evaluation` | **Default:** standard agent-mode judge + clustering. |
| `02-custom-criteria.json` | `agentic-evaluation-custom-criteria` | Adds **`parameters.evaluation_criteria`** (name → description). |
| `03-predefined-issues.json` | `agentic-evaluation-predefined-issues` | Adds **`parameters.predefined_issues`** (list of strings). |

Replace **placeholders** (`your-model-api-key-secret`, `callback_url`, `model.url`, `your-mlflow-traces-experiment`, `parameters.mlflow_results_experiment_name`, `mlflow_tracking_uri`, etc.) with your environment’s values before submitting to Eval Hub.

For **local-only** iteration, set `EVALHUB_MODE=local` and point **`parameters.mlflow_traces_experiment_name`** at an experiment on your tracking server (see **`meta/job.local-mlflow-traces.json`**), or use a **private** job copy with **`parameters.data_dir`** / pod mounts per **`main.py`** if you are not using MLflow trace ingest.

---

### Snippet — custom criteria (only the parts that differ from `01`)

```json
{
  "benchmark_id": "agentic-evaluation-custom-criteria",
  "parameters": {
    "evaluation_criteria": {
      "reasoning_clarity": "Agent provides clear step-by-step reasoning where applicable.",
      "tool_selection": "Agent selects appropriate tools for the task context.",
      "safety": "Response avoids harmful or policy-violating guidance."
    }
  }
}
```

Full file: **`02-custom-criteria.json`**.

---

### Snippet — predefined issues

```json
{
  "benchmark_id": "agentic-evaluation-predefined-issues",
  "parameters": {
    "predefined_issues": [
      "Incomplete reasoning — jumps to conclusions",
      "Incorrect tool selection for the task",
      "Ignored user constraint or policy"
    ]
  }
}
```

Full file: **`03-predefined-issues.json`**.
