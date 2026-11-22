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

A directory containing a top-level `manifest.jsonl` uses it as the authoritative
input list instead of recursive discovery. This matches NeMo Gym's
`gym eval export atif` output: each JSONL row has `trajectory_path`,
`_ng_task_index`, `_ng_rollout_index`, and optional `expected_sha256`. Paths are
relative to the manifest directory; absolute paths, traversal, symlinks, duplicate
paths and repeated task/rollout identities are rejected. Row order determines
evaluation order, and unlisted JSON files are not loaded. The manifest has its own
4 MiB bound and is not charged to trajectory file or collection-byte accounting.
When `expected_sha256` is present it is compared with SHA-256 of the exact
trajectory file bytes (the Gym exporter hashes the UTF-8 JSON bytes it writes,
including the final newline); absent digests are recorded as unverified. NeMo Gym
writes no digest inside the trajectory document, so there is no in-file value to
leave unverified. Manifest task/rollout identifiers and verification status are
carried in persisted provenance. Without `manifest.jsonl`, legacy recursive discovery
is unchanged.

The ATIF model stays the contract: a root field it does not declare is rejected as
`invalid_schema`, and producer metadata belongs in the declared `extra` object, which
is copied to `provenance.producer_extra`. So that a rejected file is still
diagnosable, the undeclared root fields are captured from the raw payload before
validation and recorded under `producer_undeclared_root` on that file's entry in
`atif_input_outcomes`, which the `atif/evaluation_results.json` report artifact
carries. The copy is capped at 4 KiB per file; past the cap only the key names are
kept, with `_preserved: false`.

Manifest rows follow `invalid_input_policy`. `skip_invalid` records the rejected
row as an invalid input (`ATIF manifest contains no entries` included) and keeps
the remaining entries, so a partially corrupt export scores what it can and
reports its counts; `fail_fast` aborts naming the row. A digest mismatch is not an
invalid input: it is governed by `partial_result_policy`, failing the job with the
offending file named under `fail_fast`, or withholding that entry's score and
incrementing `atif_failed_trajectory_count` under `skip_failed_trajectory`.
A row that repeats a path or a task/rollout identity is one the producer wrote
twice, so it follows the same policy: `skip_invalid` keeps the first and counts the
second, `fail_fast` names it. Unreadable manifests, unsafe paths (absolute,
traversal, symlink, outside the input root, non-regular) and exceeded entry budgets
stay fatal under both policies, because a manifest whose file list cannot be trusted
no longer describes the export. Manifest-driven discovery accepts an empty result
instead of failing - with `skip_invalid` because nothing usable survived, and under
either policy when every entry was withheld for a digest mismatch, since a producer's
wrong digest is not malformed input.

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
root; e.g. `/subagent_trajectories/0` for an embedded child). For manifest-backed
Gym input, the report also records the manifest's task/rollout indices, expected
hash and whether that hash was verified. This is distinct from the SHA-256 over
raw input bytes that the adapter records for every input.

Producer provenance is preserved through ATIF's declared `extra` object, copied
verbatim into each node's `provenance.producer_extra` in the persisted report
(`atif/evaluation_results.json`, indexed by `evalhub.atif_report`). The SDK rejects
undeclared root fields by design, so a producer field must live in `extra` to be
retained; such files are recorded as invalid input (`invalid_schema`) instead of
being silently dropped. There is no canonical-form digest re-derivation: NeMo Gym
declares `expected_sha256` in its manifest over the exact bytes it writes, which is
what the adapter verifies.

When `TEST_DATA_S3_BUCKET`/`TEST_DATA_S3_KEY` are available and files are under
`/test_data`, source URIs are reconstructed using the generic init's layout,
including its single-object basename rule. Otherwise they remain local paths.
**This fallback is not a substitute for an init-produced per-object provenance
manifest.** Ambiguous object/prefix layouts still require the cross-repository
source-manifest contract. Local paths are not automatically downloadable URLs.

## Judge routing and authentication

All auxiliary model credentials and endpoints come from `model.auth.secret_ref`.
Use `<role>_api-key` and optional `<role>_url` entries in that Secret; model IDs
remain benchmark parameters. When a role has no entries, it reuses `model.url`
and the primary `api-key` from the same Secret. A role URL requires its matching
role key, even when the real key equals the primary key. Kubernetes adapters see
only projected references and localhost URLs; the sidecar selects the upstream
and injects its credential. Local jobs use the SDK's `file:///` auth directory.

Inline role URLs/keys are ignored. Role credentials and endpoints are not read
from environment variables. See [model-role authentication](../../docs/model-role-auth.md)
for the Secret layout, supported roles, migration, and sidecar limitations.

ATIF uses `judge_api-key`/`judge_url` when present, with the invocation's model
name. `EVALHUB_JUDGE_PROXY_URL` is ignored. Local auth-directory reads remain
bounded and reject symlinks. The sidecar owns cluster TLS and upstream credentials.

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
makes every ancestor `complete=false`.
`partial_result_policy=skip_failed_trajectory` retains failed nodes and allows
successful roots to contribute. Repeated agent names are not cycles.

Collection `atif_overall_score` depends on `subagent_aggregation`:

- `flat`: mean of each scored node's own mean (not a global step-weighted mean).
- `hierarchical`: mean of root subtree aggregates.
- `separate`: mean of root own scores; descendants remain in detail artifacts.

### Unscored reporting

A job that scored nothing still completes. It publishes `atif_trajectory_count`
(and `atif_manifest_entry_count` when a manifest drove discovery), plus the
`atif_complete`, failed/unscored and invalid-input counts in the report metadata,
but **no `atif_overall_score`** and no `atif_failure_categorization_rate`, so an
unscored job cannot be mistaken for a passing one. This covers every root beyond
`max_subagent_depth` and every judge call failing under
`skip_failed_trajectory`. A genuine zero keeps `atif_overall_score=0.0` with
`atif_trajectory_count > 0`, and `atif_complete` is only ever true when at least
one trajectory scored and no input was left unscored, invalid or withheld for a
digest mismatch. Downstream, an absent score means the
platform test result is nil and `EvaluationThresholdViolated` does not fire; the
adapter emits one warning per unscored job.

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

A genuine export is worth generating once when a producer field changes: NeMo Gym
writes one ATIF document per rollout plus `manifest.jsonl`, whose `expected_sha256`
covers exactly the bytes it wrote (`json.dumps(indent=2)` plus a newline) - what this
adapter hashes. The k8s-tests `adapters-atif` section generates that shape (documents,
rows carrying `_ng_task_index` and `_ng_rollout_index`, digests) and runs it through a
real job, so the manifest and provenance paths are exercised without a Gym checkout. A
live capture needs a Gym environment and a model deployment; it is not part of this
repository's tests.

The container includes every application module and tests installed-image imports,
JobSpec parsing, scoring, and artifact construction without host mounts. The smoke
judge/backend are mocked; this is not a live inference or upload test. PR CI runs
unit/contract tests and the container smoke. Python runtime dependencies are pinned
in `requirements.lock`; the SDK is pinned to the approved `eval-hub-sdk~=1.1.0`
release, which carries the ATIF v1.0-v1.8 models, the declared-`extra` provenance
policy and typed trajectory score summaries. The UBI base tag/build toolchain also
need release-managed provenance.

Inputs and results are retained in memory under collection budgets, not streamed
or sharded. Large-collection memory/throughput sizing, real-judge categorization,
MLflow upload/download and tenant tests, disconnected sidecar tests, report UX,
server/SDK input/provenance agreement, and release/dependency promotion remain
separate acceptance gates. Do not mark the full strategy delivered based solely
on these mocked tests.
