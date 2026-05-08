# Run the CLEAR adapter locally

This is the same entrypoint the container uses: **`python main.py`**, with a JobSpec file whose path you pass in the environment.

## 1. Environment

From **`adapters/clear`** (same directory as **`main.py`**):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

IBM CLEAR is installed from **Git** per **`requirements.txt`** (PyPI wheels may omit agentic pieces).

## 2. Traces

Put one or more CLEAR-compatible **`*.json`** trace files in a directory. A common layout is **`input-trace/`** beside **`main.py`**; this repo also ships **example traces** under **`examples/input-traces/`** (see [02-agent-traces.md](02-agent-traces.md)). In your job JSON, set **`parameters.data_dir`** to that folder’s path (relative to where you run **`python main.py`**, or absolute). The adapter collects **`*.json`** under that directory.

## 3. Job specification

Copy or start from **`meta/job.json`**. Minimum expectations include:

- **`benchmark_id`**
- **`parameters.data_dir`** — path to your trace folder
- **`parameters.eval_model_name`**, **`parameters.provider`**
- **`parameters.inference_backend`**: **`"litellm"`**
- **`model.url`** — OpenAI-compatible API base (often **`…/v1`**)

For **local** runs with **`litellm`**, you often **delete `model.auth`** from the JSON when no Kubernetes Secret exists, and set **`OPENAI_API_KEY`** in the shell **only if** your endpoint requires it. Many **local** servers (for example some **Ollama** setups) do not require a key.

## 4. Run

```bash
export EVALHUB_MODE=local
export EVALHUB_JOB_SPEC_PATH=meta/job.json   # or your edited copy
python main.py
```

**MLflow upload (optional)** — set an experiment and tracking URI if you want artifacts in MLflow:

```bash
export MLFLOW_TRACKING_URI='https://your-mlflow-server.example/'
# experiment_name in job JSON, or parameters.mlflow_experiment_name
```

Without **`MLFLOW_TRACKING_URI`** / experiment configuration, the adapter skips MLflow upload as documented in the adapter README.

## 5. Outputs

After a successful run you should find **`clear_results.json`** and dashboard **HTML** (often **`clear_results.html`**) under the run output directory (often **`output/`** beside **`main.py`**). Optional styling: **`parameters.clear_dashboard_theme`** — [06-dashboard-theme.md](06-dashboard-theme.md).

**Committed tutorial snapshot** (same layout under **`examples/output/local/`** after you run from the notebook or point **`results_dir`** there): [clear_results.html](../output/local/clear_results.html), [clear_results.json](../output/local/clear_results.json). Open the HTML in a browser to preview the dashboard.

For **what the cards, graph, and issue tables mean**, see **§ HTML dashboard → How to read the HTML dashboard** in [07-results-schema-notes.md](07-results-schema-notes.md).

## Next

- Deployed cluster: [04-deployed-eval-hub.md](04-deployed-eval-hub.md)  
- Benchmarks: [05-benchmarks-and-parameters.md](05-benchmarks-and-parameters.md)  
