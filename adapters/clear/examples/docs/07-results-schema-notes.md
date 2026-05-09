# CLEAR outputs (`clear_results.json`, HTML) and versions

IBM CLEAR produces **two** main artifacts: a **JSON** report and a **static HTML** dashboard. This page explains how **`clear_results.json`** maps into Eval Hub, how the **HTML** report relates to optional adapter styling, and what can change when you **upgrade CLEAR**.

**Example files** from a small tutorial run: [`examples/output/local/clear_results.json`](../output/local/clear_results.json), [`examples/output/local/clear_results.html`](../output/local/clear_results.html) (open locally in a browser).

**Richer example** (larger model, more traces, complex workflow): [`examples/example_output/clear_results_example.html`](../example_output/clear_results_example.html) — see [§ Example report: what a strong judge reveals](#example-report-what-a-strong-judge-reveals) below.

## `clear_results.json` (structured results)

CLEAR writes **`clear_results.json`**. This adapter reads it and maps its contents into Eval Hub **metrics**. The exact JSON layout follows the **CLEAR revision** pinned in **`requirements.txt`** (currently **`2.0.0-rc.2`**).

When you upgrade CLEAR, IBM may rename or nest fields (for example **per agent** sections). Compare your git pin with [IBM/CLEAR](https://github.com/IBM/CLEAR) release notes and with sample outputs from your target version before you treat the JSON shape as stable.

### Mapping into Eval Hub metrics

The adapter reads **`metadata.statistics`**, **`agents`**, and related sections using the logic in **`main.py`** for the pinned CLEAR version. If you automate downstream analysis on this JSON, either **pin CLEAR** or branch your parsers when IBM publishes a schema version you can rely on.

### Issue-to-LLM-call mapping in the full JSON

Beyond the statistics the adapter maps into Eval Hub metrics, the full **`clear_results.json`** also records a mapping of each **discovered issue back to the specific LLM calls** (spans) that triggered it. This lets you drill down from a clustered issue label to the actual model inputs and outputs behind each occurrence—without having to search through raw traces manually. Future versions of the HTML dashboard are expected to surface these span-level examples directly in the report UI.

## HTML dashboard

CLEAR generates the **static** dashboard files (for example **`clear_results.html`**). That HTML is a **first class** output alongside the JSON, not a secondary afterthought.

The adapter **preserves** the dashboard at the run root (alongside `clear_results.json`) so it survives cleanup of intermediate directories. It may also **restyle** that HTML for artifacts (for example MLflow or OCI upload). Optional **`clear_dashboard_theme`** controls branding on the HTML **without** changing **`clear_results.json`**; see [06-dashboard-theme.md](06-dashboard-theme.md).

### How to read the HTML dashboard

The HTML dashboard summarizes CLEAR results for an **agentic workflow**.

- **Top summary cards** — Typically include how many **workflow nodes** (agents / graph roles) were evaluated, how many **traces** were analyzed, and how many **LLM calls** were scored overall. Exact labels follow the CLEAR version that generated the report.
- **Workflow graph** — Shows how work **moves between nodes** in the workflow. **Larger nodes** were invoked more often; **thicker edges** indicate **more frequent** transitions between nodes.
- **Per-node sections** — After the workflow-level view, the dashboard breaks analysis down **by node**. Each section reflects CLEAR's scoring and issue discovery for that node's calls.
- **Issues table** — Lists **which problems** CLEAR associated with that node, **how often** each issue appeared, and **severity**. One evaluated LLM call can match **more than one** issue, so issue **frequencies do not need to sum to 100%**.
- **No issues** — Rows or spans labeled as having **no issues** are calls that CLEAR did **not** match to any catalogued issue for that node.

Upstream CLEAR may tweak layout and labels between releases; if something looks off, compare with **`clear_results.json`** and the CLEAR version in **`requirements.txt`**.

## Example report: what a strong judge reveals

The tutorial run in this repo uses **two traces** and a **small local model** as the judge—enough to verify the pipeline end to end, but not representative of production depth.

**[`examples/example_output/clear_results_example.html`](../example_output/clear_results_example.html)** shows what CLEAR surfaces with more realistic inputs: **20 traces** through a **6-node agentic workflow** (planner, classifier, analyst, researcher, reviewer, writer), **233 LLM calls** evaluated by **GPT-5** as the judge. Each node gets its own score and issue breakdown; recurring problems cluster into named issues with frequency and severity scores—giving a much clearer picture of where the workflow degrades than a single pass-fail score would.

This is the kind of output you would expect from a production-scale CLEAR run. Use the tutorial run to get started quickly; once you have a larger trace set and a strong judge model, run locally with `parameters.data_dir` pointing at your traces (see [03-local-run.md](03-local-run.md)) or submit a job to a deployed Eval Hub with traces staged from S3 (see [04-deployed-eval-hub.md](04-deployed-eval-hub.md)).

## Try it in a notebook

Use **[`clear_evalhub_example.ipynb`](../clear_evalhub_example.ipynb)** in Jupyter: **Part A** is a **local** adapter run; **Part B** is **listing providers, submitting a job, and waiting** on a **deployed** Eval Hub. Configure **`examples/.env`** from **`env.example`** first.
