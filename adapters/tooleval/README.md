# ToolEval Adapter

Evaluates a model’s tool-use sequencing against a cache-only
[StableToolBench](https://github.com/THUNLP-MT/StableToolBench) `/virtual` API
(ToolBench G1/G2/G3).

The evaluation Job image is `quay.io/evalhub/community-tooleval:latest`. The
tool server (`quay.io/evalhub/community-toolbench-server:0.3.0`) is deployed
separately by the cluster admin — see `tool-server/README.md` and
`tool-server/deploy.yaml`. Cache misses do not call RapidAPI or OpenAI.

MUT calls go through the EvalHub sidecar (`model.url` + `api-key:ref`). Judge
calls use `judge_api-key:ref` and `judge_url` from the same model secret when
present.

## Benchmarks

| ID | Mode | Default `max_steps` |
|----|------|---------------------|
| `single-tool` | G1 — one tool call | 1 |
| `multi-tool` | G2 — several tools | 3 |
| `multi-step` | G3 — iterative observe loop | 5 |

Per task the MUT plans `call` or `finish`, the adapter POSTs `/virtual`, then
repeats until `finish` or `max_steps`. `probe_models` (default true) fail-fast
probes MUT and judge before tasks.

## Metrics

| Metric | Description |
|--------|-------------|
| `pass_rate` | Judge Solved=1 / Unsure=0.5 / Unsolved=0 (or structural match if `enable_judge` is false) |
| `win_rate` | Judge WIN=1 / LOSE=0 vs `reference_calls` |

`overall_score` is `pass_rate`. Trajectories are saved to MLflow
(`trajectories.json`, `summary.json`). Set `exports.oci` on the job to also
export those files as an OCI artifact.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_server_url` | string | _(required)_ | Tool server base URL, e.g. `http://toolbench-server.my-tenant.svc:8080` |
| `num_tasks` | integer | `5` | Number of tasks to run |
| `max_steps` | integer | _(per benchmark)_ | Max agent steps; forced to 1 for `single-tool` |
| `instruction` | string | _(per benchmark)_ | User instruction for the MUT |
| `enable_judge` | boolean | `true` | When false, score by structural match vs `reference_calls` |
| `judge_model` | string | `config.model.name` | Judge model id |
| `probe_models` | boolean | `true` | Fail-fast probe of MUT (and judge when enabled) |
| `debug_io` | boolean | `false` | Log truncated MUT / `/virtual` / judge payloads (`TOOLEVAL_DEBUG_IO=1`). Does not log API keys. |

Full list: `provider.yaml`.

### Model auth Secret

| Key | Purpose |
|-----|---------|
| `api-key` / `*_api-key` | MUT via sidecar |
| `judge_api-key` + `judge_url` | Judge via sidecar |

The sidecar must be able to reach MUT and judge hosts. Ingress for the tool
server is in `tool-server/deploy.yaml`.

## How the three benchmarks work

All three talk to the **fixture** tool-server (echo + uppercase). Use a
`FETCH_FULL_CACHE=1` image only if you also change `instruction` /
`reference_calls` to tools that exist in that cache.

Shared loop for every task:

1. Probe MUT (and judge if enabled).
2. `GET /tools` on `tool_server_url`.
3. Ask the MUT for a JSON action: `call` or `finish`.
4. On `call`, `POST /virtual` (cache-only). Feed the observation back.
5. Repeat until `finish` or `max_steps`.
6. Score: judge `pass_rate` / `win_rate`, or structural match if
   `enable_judge` is false.

| ID | What is different |
|----|-------------------|
| `single-tool` | `max_steps` forced to **1**. Catalog is filtered to **echo**. Success = that one `/virtual` call matches `tool_input`. |
| `multi-tool` | Up to **3** steps. MUT may call **echo then uppercase** (order-insensitive vs `reference_calls`). Should `finish` when both are done. |
| `multi-step` | Up to **5** steps. Same two tools, but the prompt tells the MUT to **observe** each `/virtual` result before the next call. |

Secret (same for all): `api-key` (MUT). Optional `judge_api-key` + `judge_url`
if the judge is a different endpoint.

## REST examples (create job)

These are `POST /api/v1/evaluations/jobs` bodies (not adapter JobSpecs).
Replace `tool_server_url`, `model.url`, and `secret_ref`. `debug_io: true`
prints truncated MUT / `/virtual` / judge payloads in the job log.

- `examples/rest-single-tool.json`
- `examples/rest-multi-tool.json`
- `examples/rest-multi-step.json`

```sh
curl -sS -X POST "$EVALHUB/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -H "X-User: $USER" -H "X-Tenant: $NS" \
  --data-binary @adapters/tooleval/examples/rest-single-tool.json
```

## Adapter JobSpec (local / mounted `meta/job.json`)

These JSON files are **adapter JobSpecs**: what EvalHub mounts for the job and
what you pass to `EVALHUB_JOB_SPEC_PATH` locally. They are **not** the REST
create-job body.

```json
{
  "id": "tooleval-single-tool-example",
  "provider_id": "tooleval",
  "benchmark_id": "single-tool",
  "benchmark_index": 0,
  "model": {
    "name": "my-model",
    "url": "http://localhost:8080",
    "auth": {
      "secret_ref": "my-model-auth"
    }
  },
  "parameters": {
    "tool_server_url": "http://toolbench-server.my-tenant.svc:8080",
    "num_tasks": 3,
    "tool_name": "echo",
    "api_name": "echo_message",
    "tool_input": {"message": "hello"},
    "enable_judge": true
  },
  "callback_url": "http://localhost:8080"
}
```

Also: `examples/job-single-tool.json`, `examples/job-multi-tool.json`, `examples/job-multi-step.json`.

## Building images

```sh
make image-tooleval
make image-toolbench-server          # -> .../community-toolbench-server:0.3.0
make image-toolbench-server FETCH_FULL_CACHE=1  # -> ...:0.3.0-full
kubectl -n <tenant> apply -f adapters/tooleval/tool-server/deploy.yaml
```

## Running tests locally

```sh
make test-tooleval
```

Or:

```sh
cd adapters/tooleval
uv venv
uv pip install -r requirements.txt -r requirements-test.txt
uv run pytest tests/ -v
```
