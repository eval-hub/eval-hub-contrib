# WildGuard Safety Benchmark Adapter

Evaluates a model endpoint against the [WildGuard](https://arxiv.org/abs/2406.18495)
safety benchmark (`allenai/wildguard`, MIT license). Given a prompt+response pair the
adapter sends the WildGuard instruction template to the model, parses the response as
`safe` or `unsafe`, and compares against the ground-truth label.

## Metrics

| Metric | Type | Description |
|---|---|---|
| `accuracy` | float | Fraction of examples correctly classified |
| `safe_recall` | float | Recall on the safe class |
| `unsafe_recall` | float | Recall on the unsafe class |
| `n_evaluated` | int | Total examples evaluated |
| `n_safe_correct` | int | Safe examples classified correctly |
| `n_unsafe_correct` | int | Unsafe examples classified correctly |

`overall_score` is set to `accuracy`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `split` | string | `test` | HuggingFace dataset split |
| `num_examples` | integer | _(full split)_ | Cap the number of examples evaluated |
| `max_concurrent` | integer | `4` | Concurrent API calls to the model endpoint |

## Example job

```json
{
  "id": "wildguard-test-001",
  "provider_id": "wildguard",
  "benchmark_id": "wildguard-safety",
  "benchmark_index": 0,
  "model": {
    "url": "http://vllm-svc:8080/v1",
    "name": "meta-llama/llama-3.1-8b-instruct"
  },
  "parameters": {
    "split": "test",
    "num_examples": 100,
    "max_concurrent": 8
  },
  "callback_url": "http://evalhub-sidecar:8081"
}
```

## Running tests locally

```sh
cd adapters/wildguard
uv venv
uv pip install -r requirements.txt -r requirements-test.txt
uv run pytest tests/ -v
```
