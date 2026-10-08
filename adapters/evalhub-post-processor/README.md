# EvalHub post-processing runtime

Runs post-processing operations on stored evaluation results. It does not call a model.
The initial operation computes prediction-powered confidence intervals with
[`ppi_py`](https://github.com/aangelopoulos/ppi_py), using `ppi_mean_ci` with
`lam=1` (original PPI). Output is sent to the post-processing job's `/events`
endpoint under `benchmark_status_event.additional_info`.

The provider/benchmark identifiers are `evalhub-internal` and
`evaluation-post-processor`, matching eval-hub's post-processing API mapping. The provider is marked internal-only and is omitted from the public provider API.
This adapter uses the SDK's `FrameworkAdapter` and `JobSpec`, with its own
callbacks for metric-free events and strict delivery failure handling. A model
URL is optional for this adapter. Other adapters and SDK parsing are unchanged.

## Job configuration

The mounted `/meta/job.json` remains a single job object. Its `parameters` holds
an object mapping operation names to configurations. See
[meta/job.json](meta/job.json) for a full job.

```json
{
  "operations": {
    "confidence_interval": {
      "num_parallel_threads": 3,
      "results_data_ref": {"eval_job": {"id": "completed-job-uuid"}},
      "calibration_data_ref": [
        {
          "pvc": {"claim_name": "calibration-data"},
          "data_config": {
            "format": "jsonl",
            "columns": {
              "sample_id": "example_id",
              "label": "human_score",
              "prediction": "judge_score",
              "benchmark_id": "benchmark_id",
              "provider_id": "provider_id"
            }
          }
        }
      ],
      "significance_level": 0.05
    }
  },
  "operation_order": ["confidence_interval"]
}
```

- `operation_order` is optional. When present, it must list every operation name
  exactly once, in the required execution order. When omitted, relative operation
  order is unspecified. A failure stops processing, so subsequent operations do
  not run.
- Unknown operation names fail validation.
- `significance_level` is **alpha**, as defined in eval-hub's OpenAPI schema:
  `0.05` requests 95% coverage; `0.7` requests 30% coverage. Numeric strings are
  also accepted by this runtime for compatibility with the proposed job format.
- `num_parallel_threads` defaults to 1 and is limited to 1–64. It parallelizes
  source benchmarks inside the CI operation. Result order matches
  `results.benchmarks[]`. The API's existing nested setting at
  `results_data_ref.eval_job.num_parallel_threads` is also accepted.
- `calibration_data_ref` is a nonempty array, following the API schema; a single
  reference object is accepted as a shorthand. Each entry can include an optional
  `data_config` object describing that dataset.
- Existing server-generated flat parameters (`results_data_ref`,
  `calibration_data_ref`, `significance_level`) are normalized to one CI operation.

## Data and statistical contract

The initial operation estimates a **mean of per-example numeric scores**. Both
the calibration predictions and evaluation predictions must be scores from the
same prediction procedure, in the same units as the trusted labels. Calibration
examples must be representative of the evaluation population for PPI's coverage
assumptions to apply. This operation does not score raw text or generate labels.

An aggregate metric alone is insufficient. Nonlinear aggregates such as F1,
correlation, and percentiles require an operation implementing their estimand;
passing a numeric primary metric does not turn it into a mean. An artifact
declaring an `estimand` other than `mean` is rejected. The caller is responsible
for supplying mean-score data when no estimand is declared.

Canonical evaluation JSON:

```json
{
  "primary_score": {"metric": "accuracy"},
  "estimand": "mean",
  "samples": [
    {"sample_id": "e1", "prediction": 0.2},
    {"sample_id": "e2", "prediction": 0.4},
    {"sample_id": "e3", "prediction": 0.8},
    {"sample_id": "e4", "prediction": 0.9}
  ]
}
```

Canonical calibration JSONL for `benchmark-a` from `provider-a`:

```jsonl
{"sample_id":"c1","benchmark_id":"benchmark-a","provider_id":"provider-a","label":1,"prediction":0.8}
{"sample_id":"c2","benchmark_id":"benchmark-a","provider_id":"provider-a","label":0,"prediction":0.2}
{"sample_id":"c3","benchmark_id":"benchmark-a","provider_id":"provider-a","label":1,"prediction":0.7}
```

For separate calibration examples, both `label` and `prediction` are required.
If a calibration example's `sample_id` exists in the evaluation data, its
prediction may be joined by ID. Such examples are removed from the unlabeled
evaluation array. Duplicate IDs, disagreeing joined predictions, missing pairs,
booleans, nonnumeric/nonfinite scores, and fewer than two paired calibration or
two remaining unlabeled examples fail with an error. Two is a validation minimum,
not a recommendation for statistical adequacy. PPI bounds are returned without
clipping, so they can extend beyond the nominal score range.

### Caller-defined formats

Each `calibration_data_ref` entry holds its own optional `data_config`, alongside
the source (`pvc`, `s3`, `git`, or `hf`). Omitting it uses automatic file-format
detection and the canonical column names.

The adapter reads `calibration_data_ref[].data_config` and `results_data_ref.data_config`:

| Field | Meaning |
|---|---|
| `format` | `auto` (default), `json`, `jsonl`, `csv`, or `parquet`; `.ndjson` files are also recognized |
| `path` | Relative file or directory within the downloaded artifact to read |
| `columns` | Maps canonical roles to source column names; dotted paths can access nested JSON fields |
| `selection` | Constant benchmark/metric identity for a file that has no identity columns |
| `value_mappings` | Explicit string-category-to-number maps for `prediction` or `label`; every present category must be declared |

The standalone Eval Hub API exposes result mappings alongside the result source:

```json
{
  "results_data_ref": {
    "eval_job": {"id": "completed-job-uuid"},
    "data_config": {
      "format": "json",
      "columns": {"sample_id": "id", "prediction": "scores.telemath_scorer.value"},
      "value_mappings": {"prediction": {"C": 1, "I": 0}}
    }
  }
}
```

Omitting result `data_config` preserves automatic detection and canonical fields.
Value mappings are generic: the runtime has no framework-specific score rules.
Mappings apply after field lookup, including dotted JSON paths. Their outputs
must be finite JSON numbers; unknown categories fail rather than defaulting to
zero. The previous operation-level `results_data_config` is retained for direct
adapter JobSpecs only; using both locations is rejected. Calibration fields
accepted by the HTTP API follow its separate `CalibrationDataConfig` schema.

Column roles are `sample_id`, `prediction`, `label`, `metric`, `benchmark_id`,
and `provider_id`. JSON can be an array, a `samples`/`records`
array inside an object, or one sample object. Without a mapping, `doc_id` is an
alias for `sample_id`; predictions can also be stored at the metric's own key,
`metrics[metric]`, or `scores[metric]`. A score object `{"value": 0.8}` is accepted.

For a job with multiple benchmarks, calibration must carry both `benchmark_id`
and `provider_id` in each row or its file's `selection`. This pair matches
`results.benchmarks[].id` and `results.benchmarks[].provider_id` in the source job,
so the same benchmark ID can be distinguished across providers. Optional `metric`
further selects rows. For a single source benchmark, identity fields are optional.
If a source job runs the same benchmark/provider pair more than once, the matching
calibration rows are used for each run's primary metric.

For example, a reference to a calibration file with no identity columns can use:

```json
{
  "pvc": {"claim_name": "calibration-data"},
  "data_config": {
    "format": "jsonl",
    "selection": {
      "benchmark_id": "benchmark-a",
      "provider_id": "provider-a",
      "metric": "accuracy"
    }
  }
}
```

The mounted JobSpec and `/events` payload retain the API-required
`benchmark_index` field for tracking individual benchmark executions.

For multiple calibration references, each entry's `data_config` describes that
entry's format, columns, and selection. For heterogeneous evaluation artifacts,
`results_data_ref.data_config` can be an array whose entries each have a `selection`
matching each source benchmark's ID/provider pair, with exactly one configuration
selected per source result, for example:

```json
[
  {"selection":{"benchmark_id":"benchmark-a","provider_id":"provider-a"},"format":"jsonl","columns":{"prediction":"acc"}},
  {"selection":{"benchmark_id":"benchmark-a","provider_id":"provider-b"},"format":"csv","columns":{"sample_id":"id","prediction":"score"}}
]
```

For `eval_job` results the primary metric comes from stored benchmark test
metadata, the original explicit benchmark configuration, or provider metadata
fetched through the sidecar. Its stored value must be numeric. A direct external
results reference must declare `primary_score.metric` in its JSON object or a
`manifest.json` next to its sample files. Aggregate-only artifacts are rejected.

## Sources and the sidecar

`callback_url` must point to the sidecar. **All evaluation metadata, provider
metadata, MLflow, OCI, and event requests go through that URL.** The adapter
never connects to MLflow or an OCI registry directly and does not load their
credentials. HTTP redirects fail instead of bypassing the sidecar.

| Reference | Required source fields | Access |
|---|---|---|
| `eval_job` | `id` | Sidecar GET `/api/v1/evaluations/jobs/{id}`; parent and every benchmark must be completed |
| `mlflow` | `run_id`, `artifact_path` (empty for all) | Sidecar MLflow run metadata, paginated artifact listing, and artifact download APIs |
| `oci` | `coordinates` (`oci_host`, `oci_repository`, `oci_tag`) or a `digest` alongside host/repository; `artifact_path` | Sidecar `/v2/{repository}/manifests/...` and `/blobs/...`; digest/size verification and safe tar extraction |
| `s3` | `bucket`, `key`, optional `secret_ref` | Download an object or prefix with projected credentials or the normal AWS credential chain |
| `pvc` | `claim_name`, optional `sub_path` | Copy from a declared read-only mount |
| `git` | HTTP(S) `url`, `ref`, optional `sub_path`, `secret_ref` | Fetch the revision and copy data; no repository code execution |
| `hf` | `repo_id`, optional `revision`, `sub_path`, `secret_ref` | Download a Hugging Face dataset snapshot |

Calibration supports the API's `s3`, `pvc`, `git`, and `hf` references. Each
reference must contain exactly one source. Data is materialized in temporary
directories and removed when job execution finishes or fails.

For source jobs, `results.benchmarks[]` is authoritative for both collections
and explicit submissions. MLflow uses `experiment` plus each `mlflow_run_id`;
`artifacts.mlflow.artifact_path` is honored and `logs_path` is an artifact hint,
never a local path to open. MLflow artifacts must be served through the sidecar's
artifact proxy. `MLFLOW_TRACKING_URI` and `MLFLOW_TRACKING_TOKEN` are not used;
the optional workspace header and workspace artifact URI are supported.

If OCI export metadata is also available, it is tried when MLflow cannot produce
usable per-example data. OCI uses each benchmark's `artifacts.oci_reference` or
`oci_digest` together with `exports.oci`; the job-level tag is never guessed as
the benchmark's tag.

### Runtime integration requirements

The contrib adapter cannot configure an already running sidecar. The eval-hub
runtime must arrange these before launching the post-processing pod:

1. Configure the MLflow proxy for the source tracking service. Configure the OCI
   proxy using **the post-processing job's** `exports.oci.coordinates` and its
   registry credential mount. The current sidecar supports one registry/repository
   per pod, so source coordinates must match. Merely placing OCI coordinates in
   `parameters.operations` does not enable that proxy. A mismatch fails before
   any OCI request. Registry redirects must be resolved/served by the sidecar.
2. Mount referenced PVCs read-only and set
   `EVALHUB_POST_PROCESSOR_PVC_MOUNTS` to a JSON object, for example
   `{"calibration-data":"/mnt/calibration"}`. A claim name is never interpreted
   as a host filesystem path.
3. Project named S3/git/Hugging Face Secrets under
   `${EVALHUB_POST_PROCESSOR_SECRET_ROOT:-/var/run/secrets/post-processor}/<secret_ref>/`.
   S3 uses `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`,
   `AWS_S3_ENDPOINT`, and optionally `AWS_SESSION_TOKEN`. Git uses `username`
   and `password`; Hugging Face uses `token`. OCI's `k8s.connection` is for the
   runtime/sidecar to resolve; this adapter never reads that registry Secret.
4. Preserve `operations` and the data-format configurations in the benchmark
   parameters, including `results_data_ref.data_config` and each calibration
   reference's `data_config`. The Eval Hub standalone API carries these into
   the generated execution job.

The Eval Hub runtime mounts calibration PVCs for its internal post-processing
job using the dedicated PVC mapping above. Other source types need their own
credential/download setup; accepting a data reference does not provision that
setup automatically.

## Events

The first event reports `running`. Success reports `completed` with outputs under
operation names. Source-job results preserve each benchmark's identity:

```json
{
  "benchmark_status_event": {
    "id": "evaluation-post-processor",
    "provider_id": "evalhub-internal",
    "benchmark_index": 0,
    "status": "completed",
    "additional_info": {
      "confidence_interval": {
        "benchmarks": [
          {"id":"benchmark-a","provider_id":"provider-a","benchmark_index":0,
           "confidence_interval":{"lower":0.75,"upper":0.85}}
        ]
      }
    },
    "started_at": "2026-01-12T10:45:32Z",
    "completed_at": "2026-01-12T10:47:12Z",
    "duration_seconds": 100
  }
}
```

A direct external reference has a single `confidence_interval` under the
operation, without a `benchmarks` array. Failures report `failed`, stop later
operations, and exit nonzero. Event delivery failures also exit nonzero.
The adapter sends `duration_seconds`; eval-hub's current typed event schema may
discard that field, so server schema support is needed to persist it. Started
and completed timestamps are supported.

## Build, test, tag, and push

From the repository root:

```sh
make test-evalhub-post-processor
make image-evalhub-post-processor REGISTRY=quay.io/your-org VERSION=dev
make tag-evalhub-post-processor SOURCE_IMAGE=quay.io/your-org/evalhub-post-processor:dev REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-evalhub-post-processor REGISTRY=quay.io/your-org VERSION=v1.0.0
```

Set `BUILD_TOOL=docker` if desired. The image is named `evalhub-post-processor`
without a `community-` prefix. Aggregate make targets and both image publishing
workflows include it. CI runs the tests, style checks, and an image build.
Tests use the real SDK and PPI implementation, with HTTP/storage boundaries
substituted by fixtures; they do not require remote services or credentials.

For local execution set `EVALHUB_MODE=local`,
`EVALHUB_JOB_SPEC_PATH=/path/to/job.json`, the PVC/Secret mounts as needed, and
run `python main.py` in this directory with dependencies installed. The
`callback_url` must still be a working sidecar. No model key is needed.

## Adding operations

Register a function in `post_processor/operations.py` with
`@register("operation_name")`. It receives `Context` and its configuration and
returns a JSON-compatible object. The runner handles ordering, temporary
directories, failure status, and aggregation under the registered name.
