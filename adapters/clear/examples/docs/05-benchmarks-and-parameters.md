# Benchmarks, evaluation criteria, and predefined issues

The **ibm-clear** provider exposes multiple **`benchmark_id`** values. Job JSON under **`parameters`** must satisfy the benchmark contract (see [`provider.yaml`](../../provider.yaml)).

**JobSpec samples** for each benchmark live under **[`benchmark-jobs/`](../benchmark-jobs/)**; that folder’s **[`README.md`](../benchmark-jobs/README.md)** summarizes files and placeholders.

## Default agentic evaluation

**`benchmark_id`:** **`agentic-evaluation`**

Uses CLEAR’s **standard agent-mode** rubric (discovery and clustering as implemented upstream). No extra **`evaluation_criteria`** or **`predefined_issues`** object is required.

Sample: [`benchmark-jobs/01-agentic-evaluation.json`](../benchmark-jobs/01-agentic-evaluation.json)

## Custom criteria

**`benchmark_id`:** **`agentic-evaluation-custom-criteria`**

Set **`parameters.evaluation_criteria`** to a **flat** JSON object:

```json
"criterion_name": "Natural-language description of what good looks like."
```

- Names are **free-form**. They do **not** have to match CLEAR’s built-in default rubric labels; if you want to compare with IBM’s defaults, see **`default_agentic_eval_criteria_dict`** in IBM/CLEAR — [`evaluation_criteria.py`](https://github.com/IBM/CLEAR/blob/de85d3dfa7dc61f82150929fb54f55824f5a2a8d/src/clear_eval/pipeline/evaluation_criteria.py#L87) (for example **Correctness**, **Completeness**, …).
- The map **fully replaces** CLEAR’s default criteria for that run — it does **not** merge. If you omit “Correctness”, that criterion is **not** evaluated unless another row in your map covers that intent.
- Treat **custom** criteria as your **explicit** contract; upstream default lists can change over time.

The judge is guided by these descriptions but may still surface other issues depending on CLEAR’s prompts.

Sample: [`benchmark-jobs/02-custom-criteria.json`](../benchmark-jobs/02-custom-criteria.json)

## Predefined issues

**`benchmark_id`:** **`agentic-evaluation-predefined-issues`**

Use this mode when you **already know which classes of problems** you care about (for example “wrong tool”, “ignored safety constraint”) and you want CLEAR to **focus on those**, instead of letting the pipeline **discover** issue themes open-ended.

Set **`parameters.predefined_issues`** to a **non-empty** array of short strings — **free-form**, but each item should be **short**, **relatively general**, **one topic each**, not a long narrative or a single concrete failure instance.

CLEAR maps traces against this list rather than relying only on open-ended clustering for what counts as an issue.

Sample: [`benchmark-jobs/03-predefined-issues.json`](../benchmark-jobs/03-predefined-issues.json)

## Inference backend

Set **`parameters.inference_backend`** to **`"litellm"`**. Configure **`model.url`** and credentials per [03-local-run.md](03-local-run.md) and the adapter **`README.md`**.

## Dashboard theme (optional)

**`parameters.clear_dashboard_theme`** controls Red Hat vs stock CLEAR HTML. You **do not** need to add it to every sample JSON: omit it to use the adapter default, or set it when you want stock CLEAR styling—see [06-dashboard-theme.md](06-dashboard-theme.md). The canonical sample **`meta/job.json`** includes the field as an example.

## Next

[06-dashboard-theme.md](06-dashboard-theme.md) — HTML styling  
