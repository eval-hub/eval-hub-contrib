# promptfoo Adapter

Wraps [promptfoo](https://github.com/promptfoo/promptfoo) (MIT license), an open-source
LLM testing tool, exposing two benchmarks against an EvalHub-provided model endpoint:

- **`promptfoo-eval`** — assertion-based prompt/model regression testing
- **`promptfoo-redteam`** — promptfoo's red-team plugin catalog (170+ plugins as of
  promptfoo 0.123.1), mapped to the OWASP LLM Top 10 plus industry-vertical packs
  (financial, medical, insurance, telecom, real estate) that complement rather than
  duplicate Garak's probe set — see "Garak overlap" below.

Every completed job persists promptfoo's own native `eval.json` (written directly via
`promptfoo eval -o eval.json` — verified byte-identical to `promptfoo export eval <id>
-o eval.json`, see "Verified operational constraints" below for why the direct `-o` form
is used instead) through three independent paths, so results can always be reopened in
promptfoo's own viewer via `promptfoo import`:

1. Always embedded in `JobResults.additional_info["promptfoo_eval_json"]` (size-gated,
   see `PROMPTFOO_EVAL_JSON_MAX_BYTES` in `main.py`)
2. Attached as an MLflow artifact when `job_spec.experiment_name` is set — verified
   end-to-end against a running RHOAI MLflow deployment (see [Working with
   model registries and
   MLflow](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/latest/html/working_with_mlflow/index)):
   run created, metrics and EvalCard/EnvironmentCard params logged, `eval.json` PUT as a
   real artifact and independently re-fetched via the MLflow API to confirm it matches
   the run byte-for-byte. See "MLflow: verified working, with required RBAC" below. Built
   from `evaluation_metadata`, not from path 1 — deliberately independent of the
   `additional_info` size gate, so a large `eval.json` still reaches MLflow.
3. Attached as an OCI artifact when `config.exports.oci` is set — only `eval.json`
   itself, not the whole working directory, which also holds `promptfooconfig.yaml`
   (embeds the target model's plaintext `apiKey`; exporting the whole directory would
   leak it into the OCI artifact)

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
- **promptfoo suppresses its decorated stdout results table (including the `(ID:
  eval-...)` line) when stdout is not a TTY**, as in any container — confirmed by a real
  failure on a live OpenShift cluster: `promptfoo eval` exited 0 with fully empty stdout.
  The adapter never parses CLI stdout for an eval ID; it always writes `eval.json`
  directly via `-o` on the final `eval` step (verified byte-identical to `promptfoo
  export eval <id> -o eval.json`) and reads `evalId` back out of the JSON itself.
  `promptfoo-redteam` runs `redteam generate -w` (writes generated tests back into the
  same config file) followed by a plain `eval -o eval.json` against it — never `redteam
  run`, which has no flag for writing full eval.json results.
- **Redteam result rows carry `metadata.pluginId` and `metadata.severity`** — used for
  the per-plugin pass-rate breakdown in `additional_info`.
- **Red-team GRADING uses its own separate default model, independent of
  `--provider`.** `redteam generate --provider X` only controls attack generation. The
  subsequent `eval` step's grading defaults to a hardcoded model name unrelated to `X`
  and, if unreachable, every graded test silently reports `pass=false` against a 404 —
  not a genuine vulnerability finding, a broken grading pipeline. The adapter passes
  `generation_provider` as `eval --grader` too so both steps route to the same
  operator-configured model. Confirmed live against a real model endpoint: red-team
  results were 100% "failed" until this fix — the grader was 404ing against a default
  model name (`gpt-5.5-2026-04-23` as of promptfoo 0.123.1) that doesn't exist on a
  typical self-hosted deployment.
- **Per-provider `config.timeoutMs` is NOT consumed by the OpenAI provider family** —
  verified directly against promptfoo 0.123.1's source, not just docs. Its request path
  reads only the global `REQUEST_TIMEOUT_MS` environment variable
  (`getRequestTimeoutMs()` in `src/providers/shared.ts`). The per-test timeout that
  actually applies is the top-level `evaluateOptions.timeoutMs`, read at the evaluator
  level (`context.options.timeoutMs` in `src/evaluator.ts`) — that's what `main.py`'s
  `request_timeout` parameter is wired to. `max_concurrency` is passed both ways: as
  promptfoo's own `-j` CLI flag (documented flag precedence: "Command-line flags -
  Override all other settings") and as `evaluateOptions.maxConcurrency`, for
  `config_yaml`-passthrough consistency.

## MLflow: verified working, with required RBAC

`callbacks.mlflow.save()` was exercised end-to-end against a running RHOAI MLflow
deployment (deployed via the `mlflow.opendatahub.io/v1` `MLflow` custom resource — see
[Working with model registries and
MLflow](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/latest/html/working_with_mlflow/index)),
not just code-reviewed. Three things had to be right, none of which are adapter code
changes — this is deployment/RBAC guidance for anyone running this adapter in
`EVALHUB_MODE=local` (a bare k8s Job, not behind a full EvalHub CR/server):

1. **`MLFLOW_TRACKING_URI` needs the tracking-server path prefix reported by the
   MLflow CR itself.** Read it from `status.address.url` rather than assuming a bare
   `https://<mlflow-service>:8443` — RHOAI's MLflow deployment serves its API under a
   path prefix (e.g. `.../mlflow`), and hitting the bare host without it produces a
   confusing generic 404 (not a clean "not found" JSON error), which is easy to misread
   as a workspace-provisioning problem when it's actually just a wrong base URL.
2. **`MLFLOW_WORKSPACE=<namespace>` is required and just needs to name a real k8s
   namespace** — no separate workspace-provisioning step, CR, or namespace label was
   needed once the URL was correct. RHOAI's MLflow uses namespace-based multi-tenancy;
   see the linked MLflow-on-RHOAI docs for the full model. (This is a different code
   path from MLflow result-commit as mediated by a full EvalHub CR/server — this
   adapter's `EVALHUB_MODE=local` path calls MLflow directly via the SDK's own
   `MlflowClient` and never goes through an EvalHub server at all.)
3. **The calling identity needs RBAC on `mlflow.kubeflow.org/{experiments,runs}`** (a
   pseudo-resource RHOAI's MLflow auth layer checks via SubjectAccessReview). A
   ServiceAccount with only `experiments` access can look up/create an experiment but
   gets a 403 on `runs/create` — both resources need `get/list/create` (`runs` also
   needs `update`, used when finalizing a run). Minimal Role:
   ```yaml
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: mlflow-experiment-access
   rules:
   - apiGroups: [mlflow.kubeflow.org]
     resources: [experiments, runs]
     verbs: [get, list, create, update]
   ```
   bound to the Job's ServiceAccount via a matching RoleBinding.

With all three in place: `promptfoo eval` → metrics + EvalCard/EnvironmentCard params
logged → `eval.json` PUT as a real MLflow artifact → independently re-fetched via
`GET .../mlflow-artifacts/artifacts/...` and confirmed byte-identical to the run.
`MLFLOW_TRACKING_TOKEN_PATH=/var/run/secrets/kubernetes.io/serviceaccount/token` (the
pod's own projected ServiceAccount token) was sufficient auth; a cluster with a
self-signed service certificate additionally needs
`MLFLOW_TRACKING_INSECURE_TLS=true` (or a proper CA bundle in production).

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
