# DataBench adapter

Evaluates English table question answering through an OpenAI-compatible chat endpoint.
Uses the official [`databench-eval` 4.0.1](https://github.com/jorses/databench_eval)
comparison function for booleans, numbers, categories, and lists. The primary metric
is `accuracy` (fraction of correctly answered questions on **0–1**, higher is better).
A proposed smoke threshold of `0.25` means 25%, not a score of 25; it is not a
production quality guarantee and is not applied until the registration/test policy is agreed.

## Benchmarks

| ID | Table | Gold answer |
|---|---|---|
| `databench_lite` | `sample.parquet` | `sample_answer` |

Only DataBench Lite is currently exposed by this provider adapter. The current EvalHub
sandbox constraints prohibit generated-code execution, so the upstream code-based
full-table DataBench evaluation path is not supported.

The adapter uses [`cardiffnlp/databench`](https://huggingface.co/datasets/cardiffnlp/databench),
English `qa` config, `train` split, pinned to revision
`e75d53add267d2f9cfa32efd65ad77f0807adfad`. This split is the published question
inventory, not a training operation. Dataset and table downloads require network
access on first use; `HF_HOME` may point to a populated cache for offline runs.

All columns and rows of the selected table are serialized as CSV in the prompt.
No gold `columns_used` annotation is used. The model returns an answer directly;
this adapter does not run generated Python or reproduce the upstream code-generation
baseline. Full-table prompts can exceed an endpoint's context window; the adapter
never silently samples or truncates a full table. Use a suitable long-context endpoint,
select Lite explicitly, or filter `table_id` for a smoke test. `max_prompt_chars` is
only a character-size guard, not a model-specific token budget.

Parameters are documented in `provider.yaml`. `num_examples` must be positive;
omit it for all questions. Questions are selected in source order without shuffling.
Generation is sequential with a default 120-second timeout and 2,048 output tokens.
API failures or missing/empty/whitespace-only answers fail the job; they are not silently counted as wrong
answers. Output truncation is retained as `finish_reason` in diagnostics.

## Authentication and artifacts

Uses SDK model credentials from `model.auth.secret_ref`, then `MODEL_API_KEY` or
`OPENAI_API_KEY`. An unauthenticated in-cluster endpoint uses a dummy API key.
Optional Hugging Face credentials use secret key `hf-token` or `HF_TOKEN`.
The standard EvalHub sidecar supplies model routing/TLS in cluster jobs.

Each run writes `samples.jsonl` (index, question, table ID, expected answer, model
answer, official checker result, finish reason) and `results.json` into a unique
`/tmp/databench-results-*` directory. Configure `exports.oci` to retain these files
outside the ephemeral job Pod. Aggregate results are reported through SDK callbacks.
No raw credential values are stored in artifacts.

## Development

```sh
make test-databench
make image-databench
```

The tests include a local OpenAI-compatible HTTP server and real upstream scoring,
without requiring model credentials. `meta/job.json` is an illustrative Lite smoke
request; replace its model endpoint and supply the EvalHub sidecar callback for a
real run. The image tag in `provider.yaml` is a build destination, not a claim that
an image has already been published. EvalHub/Operator registration is a follow-up.

## License and attribution

The [dataset card at the pinned revision](https://huggingface.co/datasets/cardiffnlp/databench/blob/e75d53add267d2f9cfa32efd65ad77f0807adfad/README.md)
declares MIT. The official evaluator is also MIT, copyright 2024 Jorge Osés Grijalba;
its installed package is used without vendoring its implementation. See
[upstream license](https://github.com/jorses/databench_eval/blob/main/LICENSE).

Dataset authors: Jorge Osés Grijalba, Luis Alfonso Ureña-López, Eugenio Martínez
Cámara, and Jose Camacho-Collados. Reference: *Question Answering over Tabular Data
with DataBench: A Large-Scale Empirical Evaluation of LLMs*, LREC-COLING 2024.
[Paper](https://aclanthology.org/2024.lrec-main.1179/).

Numeric comparison, category/date handling, boolean aliases and order-independent
list matching are delegated to the pinned official comparison function. No extra
answer extraction, code-block removal, case folding or custom grader is added.
Invalid nonempty answers are handled by that comparator; unsupported source answer
types fail rather than receive an invented score. CSV serialization preserves all
columns, uses pandas CSV quoting, and serializes missing cells as empty CSV fields.

A `table_id` filter or `num_examples` limit is reported as `evaluation_scope=subset`
with the selection parameters in result metadata. A Lite run is explicitly labelled
`table_variant=sample`, even when all Lite questions are evaluated. AgenticDataBench
and agent/tool/code-execution workflows are out of scope.
