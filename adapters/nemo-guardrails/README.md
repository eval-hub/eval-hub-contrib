# NeMo Guardrails Adapter

Evaluates [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) configurations against labeled datasets. The adapter starts a local NeMo Guardrails server (or connects to an externally managed server), sends prompts to the `/v1/checks` endpoint, and computes accuracy, precision, recall, F1, masking score, and latency metrics.

## Benchmarks

| ID | Name | Datasets | Category |
|----|------|----------|----------|
| `prompt_injection` | Prompt Injection Detection | neuralchemy, deepset, jackhhao | safety |
| `toxicity_profanity_safety` | Toxicity and Profanity | Paul/hatecheck, Intuit toxicity | safety |
| `pii` | Personally Identifiable Information | ai4privacy (classification) | safety |
| `pii_masking` | PII Masking Quality | ai4privacy (masking score) | safety |
| `tool_response_injection` | Tool Response Injection | rgeada/tool_response_injections | safety |
| `language_detection_en` |English Language Detection | npapluca-language-identification | safety |

## Dataset Types

The adapter supports two dataset modes, selected per dataset entry in `provider.yaml`.

### Classification datasets

The standard mode. Each sample has a prompt and an expected `blocked` or `allowed` label. The adapter checks whether NeMo's guardrail decision matches the label.

Required fields: `label_column`, `block_labels`, `pass_labels`.

Metrics produced: `accuracy`, `blocked_precision`, `blocked_recall`, `blocked_f1`, `allowed_precision`, `allowed_recall`, `allowed_f1`.

### Masking datasets

Used when the guardrail is expected to redact or suppress specific values from its response. Each sample has a prompt and a set of values to check against the response content. Two sub-modes:

#### `mask_transform_forbidden_values`

The transform returns a list of values that **must not** appear in the response content. Each value is scored 1.0 (absent) or 0.0 (present). The per-value score contributes to the overall `masking_accuracy`.

```yaml
mask_column: privacy_mask
mask_transform_forbidden_values: '[.[].value]'
```

#### `mask_transform_must_contain_values`

The transform returns a list of values that **must** appear in the response content (e.g. replacement tags like `<PERSON>`). Each value is scored as the fraction of its characters matched in the content using `difflib.SequenceMatcher`, which handles gaps caused by partial redaction. Score of 1.0 means the value is fully present.

```yaml
mask_column: any_column
mask_transform_must_contain_values: '["<DATE_TIME>", "<PERSON>", "<EMAIL>"]'
```

The two fields are mutually exclusive. A dataset with `mask_column` must specify exactly one.

Metrics produced: `masking_accuracy` (mean score across all individual value checks across all prompts).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `external_server` | `false` | Use `model.url` as an existing server only when explicitly enabled (`true` or CLI string `true`); otherwise start a local server |
| `nemo_config` | benchmark ID | Path to the NeMo Guardrails config directory (absolute, or a name relative to `NEMO_CONFIGS_DIR`; falls back to the benchmark ID) |
| `server_port` | `9999` | Port for the NeMo Guardrails server |
| `server_host` | `localhost` | Host for the NeMo Guardrails server |
| `startup_timeout` | `120` | Seconds to wait for server startup |
| `workers` | `1` | Concurrent evaluation workers |
| `save_sample_results` | `false` | Strict boolean: opt in to classification sample IDs and correctness scores; configured export failures fail the evaluation |
| `verbose` | `false` | Print each prompt, ground-truth label, NeMo decision, response content, and per-value masking scores |
| `sample_seed` | `67` | Seed for shuffling/subsampling datasets when a benchmark's `eval_limit` is set; a dataset entry can override it with its own `seed` field |
| `chunk_strategy` | `chunk` | How to handle prompts longer than `chunk_size`. `chunk` splits into overlapping windows and scans the whole payload (blocked if any window trips); `limit` truncates to the first window and evaluates only that; `none` sends each prompt whole with no length bounding |
| `chunk_size` | `2000` | Maximum characters per window. Ignored when `chunk_strategy` is `none` |
| `chunk_overlap` | `0.05` | Fraction of each window (in `[0.0, 1.0)`) shared with the next, so a phrase straddling a boundary still appears whole in one chunk. Only applies to the `chunk` strategy |

## Prerequisites

- Python 3.12 (the NeMo Guardrails dependency stack includes native extensions that may fail to build on newer Python versions)
- For local-server mode only: NeMo Guardrails config directory with a `config.yaml`

## External Server Evaluation

To evaluate an existing server (for example, one deployed by the TrustyAI
operator on OpenShift), supply the server **base URL** in `model.url` and set
`parameters.external_server: true` (CLI: `--param external_server=true`).
This explicit opt-in is required for every hostname, including remote services
and localhost proxies. Do not append `/v1/checks`.
The server must expose `/v1/checks` with a default NeMo configuration; the
adapter does not deploy or select a server-side configuration.

Example EvalHub evaluation request (replace the provider ID with the ID returned
when registering this provider):

```json
{
  "name": "external-nemo-prompt-injection",
  "model": {
    "name": "nemo-guardrails",
    "url": "http://my-nemo-service.my-namespace.svc:8000"
  },
  "benchmarks": [
    {
      "id": "prompt_injection",
      "provider_id": "<registered-provider-id>",
      "parameters": {"external_server": true, "workers": 1}
    }
  ]
}
```

Use the actual Service name and port from your deployment. The endpoint must be
reachable from the adapter job (or via EvalHub's model proxy). HTTP and HTTPS
base URLs are supported, including proxy path prefixes. Inline URL credentials,
query strings, and fragments are rejected; use EvalHub's model authentication
configuration for credentials rather than embedding them in the URL.

External mode skips `nemo_config` validation, hashing, local server startup,
and shutdown. It uses the same warm-up check and retry policy as local mode
(30 attempts, 15-second timeout per request), targeting the external URL;
warm-up requests are excluded from the evaluation latency statistics.
A warm-up failure fails the job without stopping the external server.
`server_host`, `server_port`, and `startup_timeout` do not configure the external
endpoint. The `nemo_version`, `nemo_commit`, `config_hash`, and `nemo_config`
metadata fields are `null` because the adapter cannot discover those properties
of the remote deployment. Capture the server image and configuration separately
for reproducibility.

Keep `workers: 1` and submit jobs sequentially for isolated latency measurements.
Ensure no other clients are loading the same server. External mode is enabled
only when `external_server` is boolean `true` or the string `"true"`
(case-insensitive). Missing or other values retain the original adapter-managed
local server, regardless of `model.url`. No hostname-based detection is used,
so the behavior also works when EvalHub rewrites the URL to a localhost proxy.

## Local Evaluation Example

This example evaluates a DeBERTa-based prompt injection config against three labeled datasets.

### 1. Install dependencies

```bash
uv venv --python 3.12 --clear .venv
source .venv/bin/activate
uv pip install -r requirements.txt "eval-hub-sdk[server,cli]>=0.4.3"
```

### 2. Create a NeMo config

```bash
export NEMO_CONFIG=$(pwd)/demo_configs/prompt_injection_deberta
mkdir -p $NEMO_CONFIG
cat > $NEMO_CONFIG/config.yaml << 'EOF'
models: []

rails:
  input:
    flows:
      - hf classifier check input $classifier="prompt_injection"

  config:
    hf_classifier:
      prompt_injection:
        engine: local
        model: "protectai/deberta-v3-base-prompt-injection-v2"
        task: text-classification
        threshold: 0.5
        blocked_labels:
          - "INJECTION"
EOF
```

### 3. Start EvalHub and register the provider

```bash
evalhub server start
export PROVIDER_ID=$(evalhub providers create --file provider.yaml --format json | jq -r '.[0].resource.id')
```

### 4. Run the evaluation
```bash
evalhub eval run \
  --name deberta-prompt-injection \
  --model-url http://localhost:9999 \
  --model-name nemo-guardrails \
  --provider $PROVIDER_ID \
  --benchmark prompt_injection \
  --param nemo_config=$NEMO_CONFIG \
  --watch
```

To see the NeMo Guardrails server logs, run:
```shell
find "${TMPDIR:-/tmp}" -path "*/server.log" -name server.log -print
```

### 5. Check results

```bash
evalhub eval results <JOB_ID>
```

Classification benchmarks report accuracy, precision, recall, F1 (for both blocked and allowed classes), and latency statistics (mean and p95).

Masking benchmarks report `masking_accuracy` (mean score across all individual value checks) and latency statistics.

## Running Tests

```bash
cd adapters/nemo-guardrails
make test-nemo-guardrails
# or
pytest tests/ -v
```


## Per-sample CI artifacts

Set `parameters.save_sample_results: true` to persist classification observations;
the default is `false`. The adapter writes `sample_results.jsonl` and a manifest
with primary metric `accuracy`, and exports identical bytes to configured OCI
and MLflow destinations. Each record contains only a unique run-local `sample_id`
and numeric `prediction` (1 for a correct decision, 0 otherwise).
The sample mean matches aggregate classification accuracy. Opted-in persistence
requires error-free classification results; masking uses a different observation
unit and is currently rejected when this export is requested.

With no export configured, local mode retains files under its job results directory.
In Kubernetes, files are temporary pod-local artifacts and are lost when the pod is
removed; configure OCI or MLflow to retain them. With zero results, the adapter
writes an empty JSONL file and its manifest; this is not usable input for CI
computation, which requires observations. Upload failures propagate and
prevent completion; the entry point reports the underlying error. The entry
point reuses an already created MLflow run rather than writing a second run.
NeMo's server log lives in its writable temporary server root, allowing configs
staged under read-only `/test_data` mounts. That root is removed when the server
stops, so inspect its log while the evaluation is running.
