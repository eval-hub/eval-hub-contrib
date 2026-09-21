# promptfoo Adapter

Wraps [promptfoo](https://github.com/promptfoo/promptfoo) (MIT license), an open-source
LLM testing tool, exposing two benchmarks against an EvalHub-provided model endpoint:

- **`promptfoo-eval`** — assertion-based prompt/model regression testing
- **`promptfoo-redteam`** — promptfoo's red-team plugin catalog (170+ plugins as of
  promptfoo 0.123.1), mapped to the OWASP LLM Top 10 plus industry-vertical packs
  (financial, medical, insurance, telecom, real estate) that complement rather than
  duplicate Garak's probe set — see "Garak overlap" below.

Every completed job persists promptfoo's own native `eval.json` (via
`promptfoo export eval <id> -o eval.json`, verified byte-identical to the `-o` flag on
`eval`/`redteam run` themselves) through three independent paths, so results can always
be reopened in promptfoo's own viewer via `promptfoo import`:

1. Always embedded in `JobResults.additional_info["promptfoo_eval_json"]` (size-gated,
   see `PROMPTFOO_EVAL_JSON_MAX_BYTES` in `main.py`)
2. Attached as an MLflow artifact when `job_spec.experiment_name` is set
3. Attached as an OCI artifact when `config.exports.oci` is set

## Verified operational constraints (promptfoo 0.123.1, checked 2026-09-21)

These were confirmed by actually running promptfoo, not read from documentation alone:

- **Red-team generation requires `PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=1`.**
  Without it, `redteam generate` / `redteam run` block on an interactive
  email-verification prompt against promptfoo's cloud service for *every* plugin, even
  fully deterministic ones — which a headless k8s Job cannot satisfy. This adapter
  always sets that variable; it is not configurable because there is no non-interactive
  alternative. With it set, generation runs locally against `generation_provider` (or
  promptfoo's own default model, which needs `OPENAI_API_KEY`) — no promptfoo.app
  account or network egress to promptfoo's cloud is required.
- **promptfoo returns exit code 0 even when test cases fail or error.** Pass/fail
  outcome must be read from `eval.json`'s `results.stats`, not the CLI return code. A
  non-zero return code means the CLI itself could not run (bad config, crash).
- **Custom OpenAI-compatible endpoints** use `providers`/`targets` entries shaped
  `{id: "openai:chat:<model>", config: {apiBaseUrl, apiKey}}` — verified against a real
  unreachable-endpoint run (promptfoo retried 4x then reported `errors: 1`, not a config
  parse failure).
- **`export eval <id> -o eval.json` output is byte-identical** to using `-o` directly on
  `eval`/`redteam run`. The adapter always goes through the explicit export step for
  uniformity across both benchmarks.
- **Redteam result rows carry `metadata.pluginId` and `metadata.severity`** — used for
  the per-plugin pass-rate breakdown in `additional_info`.

## Garak overlap (verified, not assumed)

Compared promptfoo's real plugin catalog (`promptfoo redteam plugins`, 170+ entries) against
Garak 0.17.0's real probe catalog (42 top-level probe modules). They overlap on jailbreak /
prompt-injection, encoding/obfuscation attacks, and harmful-content/toxicity categories.
promptfoo has no equivalent gap on the Garak side worth calling out separately, but promptfoo
covers real estate Garak does not:

- RAG-specific attacks (`rag-poisoning`, `rag-document-exfiltration`, `rag-source-attribution`)
- API/access-control attacks in the OWASP API Top 10 style (`bola`, `bfla`, `rbac`, `debug-access`)
- MCP-specific attacks (`mcp`)
- Industry-vertical compliance packs (`financial:*`, `medical:*`, `insurance:*`, `telecom:*`,
  `realestate:*`, `pharmacy:*`, `ecommerce:*`, `coding-agent:*`)

This is why the two are complementary in `provider.yaml`'s `agent.complements`, not a
duplicate-adapter situation.

## Metrics

| Metric | Type | Description |
|---|---|---|
| `pass_rate` | float | `successes / (successes + failures + errors)`. Omitted when `n_evaluated` is 0. |
| `n_evaluated` | int | Total test cases run |
| `n_passed` | int | Test cases that passed all assertions |
| `n_failed` | int | Test cases with a failed assertion |
| `n_errors` | int | Test cases where the provider call itself failed (not an assertion failure) |

`overall_score` is set to `pass_rate`. `promptfoo-redteam` additionally reports
`pass_rate_by_plugin` and `severity_by_plugin` in `additional_info` (best-effort; empty
if promptfoo's result metadata shape changes).

## Parameters

See `provider.yaml` for the full annotated list. Key ones:

| Parameter | Benchmark | Description |
|---|---|---|
| `prompts` / `tests` | promptfoo-eval | Generate a config from these instead of `config_yaml` |
| `config_yaml` | promptfoo-eval | Pass through an existing promptfoo project config verbatim (providers are always overwritten with the EvalHub model endpoint) |
| `plugins` | promptfoo-redteam | Plugin IDs to run; defaults to an OWASP LLM Top 10-mapped subset |
| `generation_provider` | promptfoo-redteam | Provider used to generate adversarial content; unset falls back to promptfoo's default (requires `OPENAI_API_KEY`) |

## Local testing

```sh
cd adapters/promptfoo
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-test.txt
.venv/bin/pytest tests/test_adapter.py -v
```

Tests monkeypatch the promptfoo CLI subprocess boundary (`_run_promptfoo_cli`) with
canned `eval.json` fixtures matching the real shape captured from a live promptfoo
0.123.1 run — no `promptfoo` binary or network access needed to run the suite.

To exercise the real CLI locally (requires Node.js and a reachable model endpoint):

```sh
npm install -g promptfoo@0.123.1
EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json .venv/bin/python main.py
```
