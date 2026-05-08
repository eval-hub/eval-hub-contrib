# CLEAR outputs (`clear_results.json`, HTML) and versions

IBM CLEAR produces **two** main artifacts: a **JSON** report and a **static HTML** dashboard. This page explains how **`clear_results.json`** maps into Eval Hub, how the **HTML** report relates to optional adapter styling, and what can change when you **upgrade CLEAR**.

**Example files** from a small tutorial run: [`examples/output/local/clear_results.json`](../output/local/clear_results.json), [`examples/output/local/clear_results.html`](../output/local/clear_results.html) (open locally in a browser).

## `clear_results.json` (structured results)

CLEAR writes **`clear_results.json`**. This adapter reads it and maps its contents into Eval Hub **metrics**. The exact JSON layout follows the **CLEAR revision** pinned in **`requirements.txt`**.

When you upgrade CLEAR, IBM may rename or nest fields (for example **per agent** sections). Compare your git pin with [IBM/CLEAR](https://github.com/IBM/CLEAR) release notes and with sample outputs from your target version before you treat the JSON shape as stable.

### Mapping into Eval Hub metrics

The adapter reads **`metadata.statistics`**, **`agents`**, and related sections using the logic in **`main.py`** for the pinned CLEAR version. If you automate downstream analysis on this JSON, either **pin CLEAR** or branch your parsers when IBM publishes a schema version you can rely on.

## HTML dashboard

CLEAR generates the **static** dashboard files (for example **`clear_results.html`**). That HTML is a **first class** output alongside the JSON, not a secondary afterthought.

The adapter may **copy** or **restyle** that HTML for artifacts (for example MLflow or OCI upload). Optional **`clear_dashboard_theme`** controls branding on the HTML **without** changing **`clear_results.json`**; see [06-dashboard-theme.md](06-dashboard-theme.md).

## Try it in a notebook

Use **[`clear_evalhub_example.ipynb`](../clear_evalhub_example.ipynb)** in Jupyter: **Part A** is a **local** adapter run; **Part B** is **listing providers, submitting a job, and waiting** on a **deployed** Eval Hub. Configure **`examples/.env`** from **`env.example`** first.
