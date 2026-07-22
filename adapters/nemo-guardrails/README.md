# NeMo Guardrails Adapter

Evaluates [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) configurations against classification datasets. The adapter starts a local NeMo Guardrails server, sends prompts to the `/v1/guardrail/checks` endpoint, and computes accuracy, precision, recall, F1, and latency metrics.

## Benchmarks

| ID | Name | Datasets | Category |
|----|------|----------|----------|
| `prompt_injection` | Prompt Injection Detection | neuralchemy, deepset, jackhhao | safety |
| `toxicity` | Toxicity and Profanity | Paul/hatecheck, Intuit toxicity | safety |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nemo_config` | benchmark ID | Absolute path to the NeMo Guardrails config directory |
| `server_port` | `9999` | Port for the NeMo Guardrails server |
| `server_host` | `localhost` | Host for the NeMo Guardrails server |
| `startup_timeout` | `120` | Seconds to wait for server startup |
| `workers` | `1` | Concurrent evaluation workers |
| `verbose` | `false` | Print each prompt and decision |

## Prerequisites

- Python 3.12+
- NeMo Guardrails config directory with a `config.yml`
[.venv](.venv)
## Local Evaluation Example

This example evaluates a DeBERTa-based prompt injection config against three labeled datasets.

### 1. Install dependencies

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt "eval-hub-sdk[server,cli]>=0.4.3"
```

### 2. Create a NeMo config

```bash
export NEMO_CONFIG=$(pwd)/demo_configs/prompt_injection_deberta
mkdir -p $NEMO_CONFIG
cat > $NEMO_CONFIG/config.yml << 'EOF'
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
tail -f $NEMO_CONFIG/../server.log
```

### 5. Check results

```bash
evalhub eval results <JOB_ID>
```

The results include accuracy, precision, recall, F1 (for both blocked and allowed classes), and latency statistics (mean and p95).

## Running Tests

```bash
cd adapters/nemo-guardrails
make test-nemo-guardrails
# or
pytest tests/ -v
```
