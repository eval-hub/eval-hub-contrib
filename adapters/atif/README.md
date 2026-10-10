# ATIF EvalHub Adapter

The ATIF adapter evaluates pre-recorded [Agent Trajectory Interchange Format
(ATIF)](https://github.com/agent-trajectory/atif) trajectories with an
OpenAI-compatible language-model judge. It is an EvalHub provider that loads
ATIF JSON files, derives evaluation criteria, scores trajectory steps, and
publishes aggregate scores and failure-categorization metadata.

This adapter supports local files and mounted directories. It provides
auto-scoring, adapter-local custom rubrics, and a file-backed
benchmark/reference flow. Detailed training-selection results are stored in
MLflow rather than returned through the EvalHub API.

## How it works

For each EvalHub job, the adapter:

1. Reads the job parameters and validates the configured thresholds.
2. Discovers one JSON file or recursively discovers JSON files in a directory.
3. Parses each file with the ATIF models from `eval-hub-sdk`.
4. Validates the ATIF schema version, duplicate trajectory IDs, file size,
   collection size, and step limits.
5. Loads a named local reference rubric, loads a validated custom rubric, or
   asks the runtime-sidecar judge to derive criteria from the first trajectory.
6. Scores every step in every trajectory, including embedded subagent
   trajectories, concurrently within the configured judge limit.
7. Requests a failure category for steps whose score is below
   `failure_threshold`.
8. Aggregates step scores into a trajectory score and then an overall score.
9. Optionally marks trajectories as training-eligible when their score meets
   `training_threshold`.
10. Uploads detailed training-selection results to MLflow when configured.
11. Returns compact EvalHub metrics and the MLflow run ID.

In Kubernetes, the adapter calls the judge through the runtime sidecar at:

```text
http://localhost:8080/v1/chat/completions
```

The judge receives JSON in the OpenAI chat message content. `JobSpec.model.url`
and `JobSpec.model.name` are the authoritative judge configuration. The sidecar
routes the local proxy request to that URL and resolves `model.auth.secret_ref`.
The adapter sends only the reference token (`Bearer api-key:ref`) in Kubernetes;
it never sends a raw mounted credential. Set `EVALHUB_JUDGE_PROXY_URL` only when
the sidecar uses a non-default local address.

For local or disconnected contract tests, the adapter uses `model.url` directly
and accepts an OpenAI-compatible `/v1/chat/completions` endpoint. If the URL is
HTTPS and the resolved local credential directory contains `ca_cert`, that CA
file is used for TLS verification. In Kubernetes, TLS termination and upstream
certificate handling belong to the sidecar. No external network access is
required when `model.url` points to an on-cluster service.

## Input format

`trajectory_path` may point to:

- a single `.json` ATIF trajectory file; or
- a directory, in which case all `*.json` files are discovered recursively in
  deterministic path order.

The adapter uses the SDK's `Trajectory` model for validation. It currently
accepts the schema versions exposed by the installed SDK, including
`ATIF-v1.0` through `ATIF-v1.8` in the validated environment.

The canonical ATIF field is `schema_version`. For compatibility with the
adapter ticket contract, `atif_schema_version` is also accepted as an alias;
if both fields are present, they must agree.

Each trajectory must contain valid ATIF data. Duplicate non-null
`trajectory_id` values across files are rejected. Empty collections, malformed
JSON, unsupported schema versions, oversized files, and trajectories exceeding
the step limit fail the job during loading.

The default path is `/test_data/trajectory.json`, which is suitable for a
mounted EvalHub test-data volume. For local execution, provide a path relative
to the adapter's working directory or an absolute path.

## Parameters

Parameters are supplied through the EvalHub job's generic `parameters` object.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `trajectory_path` | string | `/test_data/trajectory.json` | Single ATIF JSON file or directory containing ATIF JSON files. |
| `scoring_mode` | string | `auto` | `auto` derives criteria; `benchmark` uses a built-in benchmark rubric; `reference` uses a local registry; `custom` uses `provider_params.rubric`. |
| `benchmark_name` | string | unset | Built-in benchmark name required by `benchmark` mode. |
| `reference_registry_path` | string | unset | JSON registry path required for `reference` mode. |
| `reference_rubric` | string | `default` | Named rubric selected from the registry. |
| `provider_params.rubric` | object/string | unset | Canonical inline rubric object or YAML/JSON document required for `custom` mode. Maximum size is 16 KiB for document strings. |
| `custom_rubric` | object/string | unset | Backward-compatible alias for `provider_params.rubric`. |
| `custom_rubric_path` | string | unset | Backward-compatible mounted YAML or JSON rubric path. Maximum size is 16 KiB. |
| `concurrency_limit` | integer | `10` | Maximum number of top-level trajectories scored concurrently. Values below 1 are clamped to one for scoring. |
| `max_file_bytes` | integer | `10485760` | Maximum size of an individual input file in bytes. Must be positive. |
| `max_trajectory_files` | integer | `10000` | Maximum number of discovered JSON files. Must be positive. |
| `max_steps_per_trajectory` | integer | `500` | Maximum number of steps in each trajectory node, including nested subagent nodes. The complete trajectory tree is additionally limited by `max_total_steps`. |
| `max_subagent_depth` | integer | `3` | Maximum embedded-subagent depth below a root trajectory. At the limit, child trajectories are scored without further descent. |
| `max_total_steps` | integer | `10000` | Maximum number of steps in one complete trajectory tree. |
| `subagent_aggregation` | string | `flat` | `flat` averages every scored trajectory, `hierarchical` averages each parent with its descendants, and `separate` reports only root scores in the overall aggregate while retaining nested results. |
| `failure_threshold` | float | `0.5` | Scores strictly below this value are treated as detectable failures and sent for categorization. Must be in `[0, 1]`. |
| `completion_threshold` | float | `0.5` | Scores at or above this value mark a trajectory as passed. Must be in `[0, 1]`. |
| `training_threshold` | float | unset | Optional score threshold for training eligibility. A trajectory is eligible when its aggregate score is greater than or equal to this value. Must be in `[0, 1]`. |
| `judge_timeout_seconds` | float | `30` | Per-request HTTP timeout. Timeout failures are retried up to `judge_max_attempts` times. |
| `judge_max_attempts` | integer | `3` | Maximum HTTP attempts per judge request, including the first attempt. 429, 5xx, and transport/timeout failures are retried. |
| `judge_initial_backoff_seconds` | float | `0.5` | Initial exponential backoff for retryable judge failures. |
| `max_judge_requests` | integer | unset | Optional job-wide cap on HTTP attempts, including retries. Exceeding it fails the current trajectory. |
| `partial_result_policy` | string | `fail_fast` | `fail_fast` aborts on the first scoring failure; `skip_failed_trajectory` retains failure metadata and aggregates successful trajectories only. |

The adapter rejects an explicitly supplied invalid threshold before loading
input. If `training_threshold` is omitted, `training_eligible` is `null` for
each trajectory and the training manifest is empty.

### Example parameters

```json
{
  "trajectory_path": "/test_data/atif",
  "concurrency_limit": 4,
  "max_file_bytes": 10485760,
  "max_trajectory_files": 1000,
  "max_steps_per_trajectory": 500,
  "failure_threshold": 0.5,
  "completion_threshold": 0.5,
  "training_threshold": 0.8
}
```

## Scoring behavior

### Custom scoring

Set `scoring_mode` to `custom` and provide `provider_params.rubric`. The rubric
may be an object or a YAML/JSON document string. A rubric contains named criteria with descriptions and
positive weights:

```json
{
  "name": "answer_quality",
  "aggregation": "weighted_mean",
  "criteria": [
    {"name": "correctness", "description": "Matches the expected result", "weight": 2},
    {"name": "clarity", "description": "Is concise and understandable", "weight": 1}
  ]
}
```

For example, the same rubric can be supplied as YAML:

```yaml
name: answer_quality
aggregation: weighted_mean
criteria:
  - name: correctness
    description: Matches the expected result
    weight: 2
  - name: clarity
    description: Is concise and understandable
    weight: 1
```

The older `custom_rubric` and `custom_rubric_path` parameters remain accepted.
Only one rubric source may be supplied. Rubrics are parsed and fully validated
when the job starts, before trajectory loading or judge calls.

The supported aggregation modes are `weighted_mean`, `mean`, and `minimum`.
The rubric must contain 1–32 unique criteria, each with a non-empty name and
description. Criterion names and descriptions are limited to 2,000 characters;
the complete rubric is limited to 16 KiB.

For every step, the judge must return exactly one score in `[0, 1]` for every
criterion:

```json
{"scores": {"correctness": 0.9, "clarity": 0.8}}
```

The adapter validates every score and computes the aggregate locally. Missing,
extra, malformed, non-finite, or out-of-range criterion scores fail the job
closed. The rubric and trajectory are sent as structured JSON data, and the
judge is explicitly instructed to treat them as data rather than executable
instructions.

### Benchmark/reference scoring

Set `scoring_mode` to `benchmark` and provide a registered `benchmark_name` to
use the built-in benchmark rubric:

```json
{
  "scoring_mode": "benchmark",
  "benchmark_name": "swe-bench-lite"
}
```

The initial built-in registry contains `swe-bench-lite`, `terminal-bench`, and
`humaneval`. It supplies stable scoring criteria for an existing ATIF
trajectory; it does not download data, execute the benchmark, or rerun the
agent. The registry version and selected benchmark are included in result
metadata.

For user-managed rubrics, set `scoring_mode` to `reference` and provide
`reference_registry_path` and a `reference_rubric`. The registry is adapter-local, so it can be mounted with
benchmark data or baked into the adapter image without a server change. Its
minimum shape is:

```json
{
  "version": 1,
  "rubrics": {
    "answer_quality": {
      "criteria": [
        {"name": "correctness", "description": "Matches the expected result", "weight": 1.0}
      ],
      "references": {
        "trajectory-id": {"answer": "Expected answer"},
        "default": {"answer": "Fallback reference"}
      }
    }
  }
}
```

Criteria names must be unique, and weights must be finite positive numbers.
Each rubric must contain at least one criterion and one reference fixture. The
adapter selects a fixture using `extra.reference_id`, then `trajectory_id`,
then the optional `default` fixture. Missing fixtures fail the job rather than
silently falling back to auto scoring. The selected rubric and reference are
included in each judge request; score validation, failure categorization,
training eligibility, and aggregate metrics remain the same as auto scoring.

### Criteria derivation

The adapter sends the first trajectory's `extra.task_instruction` to the
judge and asks for concise JSON criteria. If that response is not valid JSON,
the adapter uses a fallback criterion named `default` with weight `1.0`.

### Step scoring

Each top-level step is sent to the judge with the derived criteria. The judge
must return JSON containing a numeric `score` in the inclusive range `[0, 1]`.
Scores must be finite; booleans, missing values, non-numeric values, malformed
JSON, and out-of-range values are rejected.

A genuine score of `0.0` is valid. Invalid score responses raise
`JudgeResponseError` and fail the evaluation rather than being silently
converted to zero.

The trajectory score is the arithmetic mean of its top-level step scores. An
empty `steps` list currently produces a score of `0.0`; this behavior remains
part of the proof-of-concept contract and should be distinguished from a judge
protocol failure.

### Failure categorization

When a step receives a valid score below `failure_threshold`, it is a
detectable agent-behavior failure and the adapter makes a second judge request
for a category. A score at or above the threshold is not sent for
categorization. Supported categories are:

- `tool_selection_failure`
- `context_loss`
- `policy_boundary_violation`
- `reasoning_failure`
- `none`

Each categorized step includes `category`, a `[0, 1]` `confidence`, a
`rationale`, and `categorization_status: categorized`. Invalid or incomplete
categorization responses are retained as `uncategorized` with the raw response
in the step metadata and `categorization_status: uncategorized`. If the
categorization judge is unavailable after retries, the step is retained as
`uncategorized` with `categorization_status: judge_error`; transport details
are logged but are not copied into result metadata. Categorization rate
is calculated over trajectories containing detectable failures: a trajectory
counts as fully categorized only when all of its detectable failures receive a
valid category response, including `none` when the judge finds no actionable
failure.

The fixed acceptance corpus is in `tests/fixtures/failure_corpus/`. The
repeatable offline acceptance test
`tests/test_failure_categorization_acceptance.py` runs it through the adapter
with two deterministic judge profiles representing an open model
(`llama-3.1-8b-instruct`) and a proprietary model (`gpt-4.1`). Both profiles
must achieve at least 90% categorization coverage and category accuracy. Run:

```sh
PYTHONPATH=. pytest -q tests/test_failure_categorization_acceptance.py
```

### Nested subagent scoring

Embedded `subagent_trajectories` are scored recursively and retained under
their parent in `evaluation_metadata["atif_trajectories"]`. Each trajectory has
its own `score`; parents also expose `aggregate_score`. Parent aggregates use
the trajectory step counts as weights, including recursively aggregated child
scores. Nested trajectory IDs must be unique across the complete input
collection. The depth and total-step limits prevent unbounded work. Repeated
ancestor agent identities are logged as circular delegations and scored in
isolation without further descent.

The default `flat` mode includes root and nested scores in the overall mean.
`hierarchical` rolls each child aggregate into its parent before calculating
the root-level mean. `separate` keeps nested scores available as diagnostics
but calculates the overall score from roots only. Failure and categorization
diagnostics include all scored trajectories.

### Retries and errors

HTTP 429 and 5xx responses, timeouts, and transport failures are retried up to
`judge_max_attempts` with exponential backoff. Other HTTP errors, invalid score
responses, and input validation errors fail the job under the default
`fail_fast` policy. With `skip_failed_trajectory`, failed trajectories are
retained in `atif_trajectories` with `status=failed`, error type/message, and
are excluded from aggregate metrics; successful trajectories still complete.
The request cap counts every HTTP attempt, including retries. A transport
failure during separate categorization is isolated to that step and reported
as `judge_error`.

## Results

### Judge observability

When EvalHub OpenTelemetry metrics are enabled, the adapter records the
following job-labelled metrics through the SDK telemetry provider:

| Metric | Type | Meaning |
| --- | --- | --- |
| `atif.judge.call.latency` | Histogram (ms) | Latency of every judge HTTP attempt. |
| `atif.judge.call.count` | Counter | Total judge HTTP attempts, including retries. |
| `atif.judge.call.success` | Counter | Attempts returning a non-error HTTP response. |
| `atif.judge.call.errors` | Counter | HTTP, timeout, and transport-error attempts. |
| `atif.judge.token.count` | Counter | Prompt and completion tokens from response `usage`. |

All metrics include `evalhub.job_id` and `atif.scoring_mode`; error points also
include a bounded `error.type` and HTTP response status where available. The
result metadata includes the corresponding per-job counts and token totals.
After the first trajectory completes, the adapter logs an estimated remaining
time based on the first trajectory's elapsed scoring time.

The adapter returns the following `EvaluationResult` metrics:

| Metric | Meaning |
| --- | --- |
| `atif_overall_score` | Mean score across all loaded trajectories. |
| `atif_trajectory_count` | Number of scored trajectories. |
| `atif_failure_categorization_rate` | Fraction of trajectories with detectable failures for which every detectable failure was categorized; `1.0` when no detectable failures exist. |

The EvalHub `overall_score` is set to `atif_overall_score`, and
`num_examples_evaluated` is the number of loaded trajectories.

## Evaluation metadata

`evaluation_metadata` includes:

```text
atif_trajectories
atif_trajectory_metadata
atif_scoring_mode
atif_failure_threshold
atif_detectable_failure_count
atif_categorized_failure_count
atif_uncategorized_failure_count
atif_categorization_judge_error_count
atif_detectable_failure_trajectory_count
atif_categorized_failure_trajectory_count
atif_failure_categorization_rate
atif_training_threshold
atif_training_manifest
```

Each entry in `atif_trajectories` contains the trajectory ID, local source
path, aggregate score, completion threshold, `passed` status, step count, per-step results, detectable-failure count,
categorized-failure count, uncategorized-failure count,
categorization-judge-error count, and `training_eligible`. Each step result
contains its trajectory ID, zero-based step index, step ID, and score. When a
step contains tool calls, it also contains `tool_name` (the first callable) and
`tool_names` (all distinct callables). Detectable failures additionally contain
`category`, `confidence`, `rationale`, and `categorization_status`; malformed
responses also contain `raw_judge_response`.

The detailed result is JSON serializable through the SDK `JobResults` model
(`model_dump(mode="json")`). A trajectory passes when its aggregate score is
greater than or equal to `completion_threshold`; this status is independent of
failure categorization and training eligibility.

When `training_threshold` is set, trajectories are classified using
`aggregate_score >= training_threshold`. When MLflow is configured, the adapter
uploads two JSON artifacts in the same run:

- `atif/training/eligible_trajectories.json`
- `atif/training/ineligible_trajectories.json`

The artifacts contain detailed trajectory and step results, eligibility,
threshold, counts, and source paths. For S3-backed test data, source paths are
reported as their original `s3://` URIs; local and PVC inputs retain their
filesystem paths. Failed trajectories are included in the ineligible artifact.
The EvalHub completion event contains only aggregate
metrics and the MLflow run ID; it does not contain the detailed trajectory tree
or training manifest. MLflow also records the threshold and cohort counts as
searchable run parameters. Without a threshold, the adapter preserves the
single `atif/evaluation_results.json` artifact behavior.

`atif_trajectory_metadata` contains the parsed schema version, trajectory and
session identity, task instruction, agent name/version, model, tool-definition
count, and the original steps including tool calls, observations, and reasoning
traces. The same identity is emitted in the typed SDK Environment Card under
`env_card.custom`.

## Provider and container

The provider ID is `atif`, and the default benchmark ID is
`atif_trajectory_default`. The provider manifest is in [`provider.yaml`](provider.yaml).
The adapter image is built from [`Containerfile`](Containerfile) and runs
`python main.py` as a non-root user.

The provider's benchmark pass criterion is an `atif_overall_score` of at least
`0.5`. This benchmark criterion is separate from `failure_threshold` and
`training_threshold`:

- `failure_threshold` controls failure categorization.
- `training_threshold` controls training eligibility metadata.
- The benchmark pass criterion controls the benchmark's pass/fail evaluation.

## Local development and tests

From this adapter directory, install the runtime and test requirements, then
run the tests:

```sh
pip install -r requirements.txt -r requirements-test.txt
PYTHONPATH=. pytest -q tests
```

The test suite covers ATIF parsing and limits, supported schema versions,
duplicate IDs, malformed input, score validation, genuine zero scores,
failure categorization, 429 retries, training eligibility, and invalid
training thresholds. Benchmark-mode tests cover built-in rubric lookup,
dispatch, unknown names, missing names, validation, and reproducibility;
reference-mode tests cover rubric selection, reference injection into judge
prompts, and missing-fixture failures.
Proxy contract tests additionally cover Kubernetes credential indirection,
configured model routing, and local HTTPS endpoints with optional CA bundles.

## Current limitations

- Input is local or mounted filesystem data; S3 discovery and download are
  owned by a downstream EvalHub input contract.
- Benchmark rubrics are currently adapter-local and built in; server-managed
  benchmark catalogs and Harbor verifier execution are not implemented.
- Nested `subagent_trajectories` are scored recursively with configurable depth,
  total-step, duplicate-ID, and aggregation controls.
- Partial results are supported through `partial_result_policy`, but resume and
  checkpointing are not implemented.
- Per-step judge diagnostics are limited; full structured redacted diagnostics
  and typed result fields require additional work.
- Training eligibility is exposed through the typed SDK trajectory summaries and
  detailed MLflow artifacts; the completion API intentionally carries only the
  MLflow run ID and aggregate metrics.
- Production image promotion, dependency onboarding, and release management
  are outside this adapter directory.
