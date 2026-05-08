# Example CLEAR jobs (three benchmarks)

**Tutorial:** step-by-step docs live in **[`../README.md`](../README.md)** (folder **`examples/`**) — overview, traces, local run, benchmarks, dashboard theme. Benchmark semantics (criteria, predefined issues, theme): **[`../docs/05-benchmarks-and-parameters.md`](../docs/05-benchmarks-and-parameters.md)** and **[`../docs/06-dashboard-theme.md`](../docs/06-dashboard-theme.md)**.

These JSON files use the **same JobSpec shape as `meta/job.json`**: deployed **Eval Hub**, **S3-backed traces** via `test_data_ref`, model **auth** secret ref, `parameters.data_dir` pointing at where traces appear **in the job pod** (here `input-trace`, same as the default sample), and a real **`callback_url`** for the sidecar.

| File | `benchmark_id` | Purpose |
|------|----------------|---------|
| `01-agentic-evaluation.json` | `agentic-evaluation` | **Default:** standard agent-mode judge + clustering. |
| `02-custom-criteria.json` | `agentic-evaluation-custom-criteria` | Adds **`parameters.evaluation_criteria`** (name → description). |
| `03-predefined-issues.json` | `agentic-evaluation-predefined-issues` | Adds **`parameters.predefined_issues`** (list of strings). |

Replace **placeholders** (`your-model-api-key-secret`, S3 bucket/path/secret, `callback_url`, `model.url`, `experiment_name`, etc.) with your environment’s values before submitting to Eval Hub.

These samples **omit** **`parameters.clear_dashboard_theme`** for brevity; omitting it uses the adapter’s **default** branded HTML. To force CLEAR’s stock HTML or to document the field explicitly, see **`meta/job.json`** and [dashboard theme doc](../docs/06-dashboard-theme.md).

For **local-only** iteration (no sidecar / no MLflow), you can still set `EVALHUB_MODE=local` and ensure trace JSONs exist under **`parameters.data_dir`** on disk (or under `/test_data` / `/data` per adapter rules); adjust paths in a **private** copy of the JSON if needed.

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
