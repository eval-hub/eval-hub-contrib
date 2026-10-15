# ATIF EvalHub adapter

Scores **already-recorded** [ATIF](https://github.com/agent-trajectory/atif)
trajectories with an OpenAI-compatible judge. It never reruns an agent, executes
trajectory tools, or substitutes for a benchmark's verifier.

## Execution contract

The SDK entry point is `run_benchmark_job(config, callbacks) -> JobResults`.
`config.model` is the judge. **An `experiment_name` and the SDK MLflow callback
are required**; missing configuration fails before judge work. A failed upload
or a save returning no run ID fails the job rather than dropping detailed scores.
The sample [`meta/job.json`](meta/job.json) contains an experiment name; configure
the MLflow backend as documented by the SDK before a real run.

1. Discover JSON files under the mounted input path.
2. Identify and validate ATIF candidates, recording accepted/ignored/invalid files.
3. Resolve per-task criteria and score interactions with bounded context.
4. Score embedded descendants up to the configured depth; retain explicit failure
   and unscored records, and aggregate successful scores.
5. Persist versioned details, source-addressable training selections, and provenance.
6. Return compact metrics and an artifact index through the SDK's existing generic
   `artifacts` callback field, together with the MLflow run ID and Environment Card.

This adapter does **not** add the server/SDK `atif_trajectory` input discriminator
or implement report UI browsing/filtering. Generic S3/PVC/Git mounting belongs to
EvalHub. The artifact index enables a consumer to retrieve MLflow details; server
storage, tenant-authorized retrieval, and UI support still need end-to-end testing.

## Inputs and provenance

`trajectory_path` defaults to `/test_data`. It accepts one JSON file or a directory
recursively scanned in deterministic order. `trajectory.json` and `*.atif.json` are
explicit candidates; other JSON objects with ATIF-like fields (`schema_version`,
`atif_schema_version`, `agent`, or `steps`) are also candidates. Unrelated JSON in
collections is ignored. Malformed JSON is recorded as invalid rather than guessed
to be unrelated. A single explicitly supplied file is always validated.

The default `invalid_input_policy=skip_invalid` skips invalid candidates and records
per-file outcomes. `fail_fast` aborts instead. No usable input always fails. Limits
on total collection bytes and accepted steps are fatal, independently of the
invalid-input policy. Discovery and per-file size are also bounded. Symlink paths
and non-regular files are rejected; rubric files follow the same policy.

Schema versions are validated by the pinned SDK (`ATIF-v1.0`–`ATIF-v1.8`). This is
not certification of every producer/version combination. `schema_version` must be
explicit, including in embedded nodes. `atif_schema_version` is accepted as a
root alias only if it agrees with the canonical field. Sequential step IDs,
embedded references, and trajectory identity uniqueness are SDK-validated.
Duplicate non-null IDs across accepted trees are invalid.

Producer trajectory IDs are preserved. ID-less roots receive a stable
source-qualified SHA-256 evaluation ID. Each node records the original producer
ID, input SHA-256, local path, source URI, and a JSON Pointer selector (empty for a
root; e.g. `/subagent_trajectories/0` for an embedded child).

When `TEST_DATA_S3_BUCKET`/`TEST_DATA_S3_KEY` are available and files are under
`/test_data`, source URIs are reconstructed using the generic init's layout,
including its single-object basename rule. Otherwise they remain local paths.
**This fallback is not a substitute for an init-produced per-object provenance
manifest.** Ambiguous object/prefix layouts still require the cross-repository
source-manifest contract. Local paths are not automatically downloadable URLs.

## Judge routing and authentication

- **Kubernetes:** call the invocation's `model.url`; the server rewrites it to the
  runtime sidecar and preserves the provider path. `EVALHUB_JUDGE_PROXY_URL` remains
  an explicit override for unusual deployments. With `model.auth` configured, send only
  `Bearer api-key:ref`. The adapter does not read cluster credential files. The
  sidecar owns upstream authentication and TLS.
- **Local:** use the invocation's `model.url` directly. `model.auth.secret_ref` must
  be an absolute `file:///` directory URL containing `api-key` and optionally
  `ca_cert`. Send the actual local key; verify HTTPS using that CA or system trust.
  Do not put credentials in the job spec, input, image, or source control. A
  different on-disk JobSpec is never used to discover credentials.

One asynchronous HTTP client is reused for the scoring phase. A fixed worker set
bounds root concurrency; each tree's steps and children are scored sequentially.
HTTP 429/5xx and transport/timeouts are retried with exponential backoff. Other
HTTP failures and invalid score responses are not retried. Request budgets count
attempts, including derivation, categorization, and retries. The scoring deadline
cancels outstanding workers and closes the client.

## Scoring modes

### Auto

Task resolution prefers non-empty `extra.task_instruction`, then the first
textual `source=user` frame (including text blocks). This handles the checked-in
real Harbor Terminus fixture. It does not infer a task from arbitrary tool output.
If neither exists, auto mode fails that node; use explicit metadata or an explicit
rubric. In multi-task user conversations the first user frame is only the initial
task; subsequent messages remain history.

Criteria are derived once **per distinct task within a job**, keyed by task and
prompt-template hash. Each derivation must have 1–32 uniquely named criteria with
positive finite weights. For compatibility, a missing description uses the
criterion name; invalid JSON or invalid criteria never cause a generic fallback.
Exact normalized criteria, criteria/task hashes, prompt version, benchmark
registry version when applicable, and judge model/settings are persisted.

### Benchmark and reference

`scoring_mode=benchmark` requires `benchmark_name`:
`swebench-verified`, `terminal-bench`, or `humanevalfix`. These are built-in assessment
rubrics, **not benchmark execution or verifier-equivalent correctness claims**.
Seed names match the benchmark names in Harbor's public
[benchmark registry](https://github.com/harbor-framework/harbor/blob/main/registry.json).
Domain-owner approval and real-judge quality measurement remain release gates.

`scoring_mode=reference` requires a local `reference_registry_path`:

```json
{"rubrics": {"default": {
  "criteria": [{"name": "correctness", "description": "Matches the reference", "weight": 1}],
  "references": {"default": {"answer": "Expected answer"}}
}}}
```

`reference_rubric` selects the rubric. Reference selection uses
`extra.reference_id`, then `trajectory_id`, then the optional `default` reference.
Missing references fail explicitly.

### Custom

`scoring_mode=custom` requires `provider_params.rubric`, an object or safe YAML/JSON
document. Legacy `custom_rubric` and `custom_rubric_path` also work. Supply exactly
one source. All forms are limited to 16 KiB; file reads are bounded before parsing.

```yaml
name: answer_quality
aggregation: weighted_mean
criteria:
  - name: correctness
    description: Addresses the requested task
    weight: 2
  - name: clarity
    description: Explains the result clearly
    weight: 1
```

Criteria have unique non-empty names/descriptions (up to 2,000 characters each)
and positive finite weights. The judge returns `{"scores": {"correctness": 0.9,
"clarity": 0.8}}`; the adapter validates the exact criterion set and computes
`weighted_mean`, `mean`, or `minimum` locally.

### Context, scores, and aggregation

Every scoring/categorization request carries the task, tool definitions, current
step, recent prior interactions/observations, and available parent delegation
evidence. A dedicated system instruction treats all trajectory/rubric/reference
content as untrusted evidence. This reduces instruction confusion; it is **not a
proof of prompt-injection immunity**.

Context projection omits top-level step metrics, token arrays, and arbitrary
producer extras. Image/audio references are replaced by explicit omitted markers;
this is text-based evaluation, not multimodal judging. Half of `max_context_bytes` is allocated to context/current-step
evidence. Oldest history is dropped first and the omitted count is retained per
scored step. Oversized required task/current-step/delegation evidence or a complete
request over the limit fails explicitly; it is not silently clipped and scored.

For backward compatibility, **all recorded frames (including user/system frames)
are scored** and participate in denominators. This choice is recorded as
`atif_scored_frames=all`; changing to agent-only scoring requires a versioned
contract change. Genuine zero scores are valid; malformed, boolean, non-finite,
missing, or out-of-range scores are protocol failures, not low-quality scores.

A node's own score is its mean step score. Its subtree aggregate is weighted by
scored step counts: `(own_mean * own_steps + sum(child_aggregate * child_steps)) /
(own_steps + sum(child_steps))`. Root depth is zero. Nodes beyond
`max_subagent_depth` remain explicitly unscored; a child that fails or is omitted
makes every ancestor `complete=false`. The all-failed-root case always fails the
job.
`partial_result_policy=skip_failed_trajectory` otherwise retains failed nodes and
allows successful roots to contribute. Repeated agent names are not cycles.

Collection `atif_overall_score` depends on `subagent_aggregation`:

- `flat`: mean of each scored node's own mean (not a global step-weighted mean).
- `hierarchical`: mean of root subtree aggregates.
- `separate`: mean of root own scores; descendants remain in detail artifacts.

### Failure categorization

Steps below `failure_threshold` receive one of `tool_selection_failure`,
`context_loss`, `policy_boundary_violation`, `reasoning_failure`, or `none`.
Invalid responses are `uncategorized`; transport failures have
`categorization_status=judge_error`. Raw malformed responses and raw exception
messages are not persisted in score results. Responses are capped at 64 KiB and
retained rationales at 2,000 characters. Valid judge rationales may contain
sensitive evidence and still require normal artifact access/retention controls.

`atif_failure_categorization_rate` measures the fraction of low-scoring trajectories
for which **every** low-scoring step receives an actionable category. `none` is a
valid abstention, not actionable coverage. The no-detected-failures convention is
1.0. This proxy metric does not establish ground-truth detection or category
accuracy. `tests/test_failure_categorization_contract.py` is a mocked request
contract test, **not** the strategy's two-real-judge ≥90% acceptance demonstration.

## Results, training, and retrieval

Detailed result schema version **2** adds explicit completeness, node relationships,
source selectors, criteria provenance, and loading diagnostics. Without a training
threshold, retrieve `atif/evaluation_results.json` from the returned MLflow run.
With a threshold, retrieve:

- `atif/report.json` — job metadata, Environment Card, summaries, and cohort paths;
- `atif/training/eligible_trajectories.json`;
- `atif/training/ineligible_trajectories.json`.

Cohorts are **flat and disjoint**: records contain parent/child IDs, not embedded
copies of nodes in the other cohort. Eligibility requires complete subtree scoring
and `aggregate_score >= training_threshold`. A node passes only if complete and
above `completion_threshold`. No threshold means eligibility is unspecified.
These are score-based selections, not claims about privacy, licensing, or training
suitability.

Manifest entries contain `source_uri`, `trajectory_id`, `selector`, and `sha256`.
Selections are **node-level**, not permission to export entire source files or
embedded subtrees. An eligible root may contain an ineligible descendant. A
consumer must resolve the selector, exclude separately selected descendants from
the node export, and verify the hash. `source_paths` is informational only.

The completion event's `artifacts["evalhub.atif_report"]` carries the schema version,
MLflow run ID, artifact paths, scored/eligible counts, completeness, and selection
unit. Details/manifests stay out of the completion event. The adapter passes full
summaries to SDK persistence before compaction so searchable MLflow eligible
counts include embedded nodes correctly. A generic index is not a finished UI.

The Environment Card summarizes common schema, agent/version, model, and tool
counts. Mixed collections use null for non-common fields and mark `heterogeneous`.
Agent version is never mislabeled as model version. Per-node metadata/provenance
is retained in artifacts; duplicate full raw trajectories are not stored by default.

## Parameters

All options live in `JobSpec.parameters`; see [`provider.yaml`](provider.yaml).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `trajectory_path` | `/test_data` | File or mounted collection directory |
| `scoring_mode` | `auto` | `auto`, `benchmark`, `reference`, `custom` |
| `benchmark_name` | unset | Required in benchmark mode |
| `reference_registry_path` / `reference_rubric` | unset / `default` | Reference configuration |
| `provider_params.rubric` | unset | Custom object or YAML/JSON; legacy aliases above |
| `concurrency_limit` | 10 | Positive maximum concurrent root workers |
| `max_file_bytes` | 10485760 | Per-file byte limit |
| `max_trajectory_files` | 10000 | Discovery file limit |
| `max_collection_bytes` | 67108864 | Collection-wide byte budget |
| `max_steps_per_trajectory` | 500 | Per-node step limit |
| `max_total_steps` | 10000 | Collection-wide accepted-tree step budget |
| `invalid_input_policy` | `skip_invalid` | Or `fail_fast` |
| `max_subagent_depth` | 3 | Root is depth 0 |
| `subagent_aggregation` | `flat` | Or `hierarchical`, `separate` |
| `partial_result_policy` | `fail_fast` | Or `skip_failed_trajectory` |
| `failure_threshold` / `completion_threshold` | 0.5 / 0.5 | Inclusive completion, strict failure threshold |
| `training_threshold` | unset | Optional complete-node selection threshold |
| `max_context_bytes` | 65536 | Complete serialized judge request bound (minimum 1024) |
| `judge_timeout_seconds` | 30 | Per-attempt HTTP timeout |
| `judge_max_attempts` | 3 | 1–10 attempts per request |
| `judge_initial_backoff_seconds` | 0.5 | Nonnegative initial exponential backoff |
| `max_judge_requests` | unset | Optional job-wide HTTP attempt budget |
| `job_timeout_seconds` | 3600 | Deadline for scoring, including derivation/retries |

## Observability, build, and tests

OTel exports `atif.judge.call.{latency,count,success,errors}`,
`atif.judge.protocol.errors`, and input/output `atif.judge.token.count`.
HTTP success and valid protocol responses are deliberately separate signals.
The old root-count ETA was removed because it ignored nested work/concurrency.

From the repository root:

```sh
make test-atif
make smoke-atif REGISTRY=localhost VERSION=dev  # build + non-root, no-network smoke
make lock-atif                                 # regenerate Python 3.12 dependency pins
```

The container includes every application module and tests installed-image imports,
JobSpec parsing, scoring, and artifact construction without host mounts. The smoke
judge/backend are mocked; this is not a live inference or upload test. PR CI runs
unit/contract tests and the container smoke. Python runtime dependencies are pinned
in `requirements.lock`; the SDK uses an immutable public HTTPS Git commit rather
than an SSH development branch. **That integration commit is not an approved SDK
release.** The UBI base tag/build toolchain also need release-managed provenance.

Inputs and results are retained in memory under collection budgets, not streamed
or sharded. Large-collection memory/throughput sizing, real-judge categorization,
MLflow upload/download and tenant tests, disconnected sidecar tests, report UX,
server/SDK input/provenance agreement, and release/dependency promotion remain
separate acceptance gates. Do not mark the full strategy delivered based solely
on these mocked tests.
