# Dashboard HTML theme (`clear_dashboard_theme`)

CLEAR generates a **static HTML** report (for example **`clear_results.html`**). After CLEAR writes it, the adapter may **rewrite** that HTML file for **Red Hat**–aligned branding (CSS / title). **`clear_results.json`** is **not** modified by theming.

## Job parameter

Set **`parameters.clear_dashboard_theme`** on the JobSpec (documented in [`provider.yaml`](../../provider.yaml)).

| Intent | Typical values |
|--------|----------------|
| **Red Hat–styled** dashboard (adapter default when omitted) | Omit the key, or use **`red_hat`** / **`redhat`** |
| **CLEAR’s original** dashboard HTML (no adapter restyle) | **`clear`**, **`default`**, **`original`**, **`ibm`**, **`none`**, **`false`**, **`0`**, **`off`** |

Example snippet:

```json
"parameters": {
  "clear_dashboard_theme": "clear"
}
```

**`clear_results.json`** is unchanged. Only the **HTML files** the adapter processes for the run are restyled (or left as CLEAR produced them when you opt out).

You do **not** need to add this key to every **`benchmark-jobs/*.json`** file—omit it unless you want the sample to show the knob; **`meta/job.json`** already demonstrates **`clear_dashboard_theme`** for copy-paste.

## Next

[07-results-schema-notes.md](07-results-schema-notes.md): CLEAR outputs (JSON, HTML) and versions  
