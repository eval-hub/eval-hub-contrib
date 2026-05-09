# IBM CLEAR adapter: examples and tutorials

Use the **markdown walkthrough** below, or jump straight to **[`clear_evalhub_example.ipynb`](clear_evalhub_example.ipynb)** (Part A: local run, Part B: deployed Hub).

## First-time path

1. **[`docs/01-overview.md`](docs/01-overview.md)** — what CLEAR, Eval Hub, and this adapter do (one short read).
2. **[`docs/02-agent-traces.md`](docs/02-agent-traces.md)** — what goes in **`input-traces/`** and MLflow shape expectations.
3. **[`docs/03-local-run.md`](docs/03-local-run.md)** — venv, job JSON, **`python main.py`**, outputs.

**Sample outputs** (open the HTML in a browser after clone):

| File | What it shows |
|------|---------------|
| [`output/local/clear_results.html`](output/local/clear_results.html) | Dashboard from a **small tutorial run** (2 traces, small local judge). How to interpret cards, graph, and issue tables: [07-results-schema-notes.md § HTML dashboard](docs/07-results-schema-notes.md#how-to-read-the-html-dashboard). |
| [`output/local/clear_results.json`](output/local/clear_results.json) | Structured results and stats for metrics mapping. |
| [`example_output/clear_results_example.html`](example_output/clear_results_example.html) | Dashboard from a **production-scale run**: 20 traces, 6-node workflow, 233 LLM calls, GPT-5 judge. Shows the depth of analysis CLEAR can surface over real data with a strong judge — see [07-results-schema-notes.md § Example report](docs/07-results-schema-notes.md#example-report-what-a-strong-judge-reveals). |

**Sample inputs:** [`input-traces/`](input-traces/) — see **Samples** in [`docs/02-agent-traces.md`](docs/02-agent-traces.md).

## Tutorial (read in order)

| Doc | Topic |
|-----|--------|
| [docs/01-overview.md](docs/01-overview.md) | IBM CLEAR, Eval Hub, and the **ibm-clear** adapter |
| [docs/02-agent-traces.md](docs/02-agent-traces.md) | What agent traces are and links to upstream trace format docs |
| [docs/03-local-run.md](docs/03-local-run.md) | Install packages and run **`main.py`** locally with trace JSON |
| [docs/04-deployed-eval-hub.md](docs/04-deployed-eval-hub.md) | Running jobs on a deployed Eval Hub (high level) |
| [docs/05-benchmarks-and-parameters.md](docs/05-benchmarks-and-parameters.md) | Benchmarks, **`evaluation_criteria`**, **`predefined_issues`** |
| [docs/06-dashboard-theme.md](docs/06-dashboard-theme.md) | **`clear_dashboard_theme`** (Red Hat HTML vs stock CLEAR) |
| [docs/07-results-schema-notes.md](docs/07-results-schema-notes.md) | **`clear_results.json`**, HTML dashboard, **how to read the dashboard**, CLEAR version notes |

## Other files here

| Path | Purpose |
|------|---------|
| [benchmark-jobs/](benchmark-jobs/) | Sample JobSpec JSON for the three benchmarks |
| [clear_evalhub_example.ipynb](clear_evalhub_example.ipynb) | Jupyter: local **`main.py`** + optional **`curl`** / Python SDK for Eval Hub |
| [env.example](env.example) | Copy to **`.env`** — do not commit **`.env`** |

Upstream IBM CLEAR: [github.com/IBM/CLEAR](https://github.com/IBM/CLEAR)
