# IFBench Adapter

Evaluates a model endpoint against [IFBench](https://github.com/allenai/IFBench)
(AllenAI, Apache 2.0) — 58 out-of-domain verifiable instruction constraints.

## Metrics

| Metric | Type | Description |
|---|---|---|
| `accuracy` | float | Prompt-level accuracy for configured scoring_mode (default: loose) |
| `prompt_level_strict` | float | Prompt-level strict accuracy |
| `prompt_level_loose` | float | Prompt-level loose accuracy (paper default) |
| `inst_level_strict` | float | Instruction-level strict accuracy |
| `inst_level_loose` | float | Instruction-level loose accuracy |
| `n_evaluated` | int | Number of prompts evaluated |

`overall_score` equals `accuracy`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scoring_mode` | string | `loose` | Primary scoring mode (`loose` or `strict`) |
| `num_examples` | integer | _(full set)_ | Cap prompts evaluated |
| `max_concurrent` | integer | `4` | Concurrent API calls |
| `max_tokens` | integer | `2048` | Max generation tokens per prompt |
| `temperature` | number | `0.0` | Sampling temperature |
| `request_timeout` | integer | `120` | Per-request API timeout (seconds) |

## Example job

```json
{
  "id": "ifbench-test-001",
  "provider_id": "ifbench",
  "benchmark_id": "ifbench",
  "benchmark_index": 0,
  "model": {
    "url": "http://vllm-svc:8080/v1",
    "name": "meta-llama/llama-3.1-8b-instruct"
  },
  "parameters": {
    "num_examples": 50,
    "max_concurrent": 8,
    "scoring_mode": "loose",
    "temperature": 0.0
  },
  "callback_url": "http://evalhub-sidecar:8081"
}
```

## Running tests locally

```sh
cd adapters/ifbench
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-test.txt
.venv/bin/pytest tests/ -v
```

Standalone runs (no EvalHub server) should set `IFBENCH_LOCAL_ONLY=1` so the adapter
does not POST to `callback_url`. Do **not** use `EVALHUB_MODE=local` for that —
EvalHub local runtime sets `EVALHUB_MODE=local` and still expects sidecar reports.

## References

- Paper: https://arxiv.org/abs/2507.02833
- Upstream: https://github.com/allenai/IFBench
- eval-hub-contrib issue: https://github.com/eval-hub/eval-hub-contrib/issues/131
