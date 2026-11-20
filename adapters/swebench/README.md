# SWE-bench Adapter

Evaluates LLM-generated code patches against the [SWE-bench](https://www.swebench.com/) benchmark using Kubernetes Jobs.

## How It Works

The adapter orchestrates evaluation by creating K8s Jobs for each SWE-bench instance:

1. Receives a predictions JSON file (standard SWE-bench format)
2. Creates one K8s Job per instance using pre-built SWE-bench container images
3. Each Job applies the model's patch, runs the test suite, and exits
4. The adapter grades results using SWE-bench's grading infrastructure
5. Reports `resolve_rate` (% of instances where the patch passes all tests) as the primary metric

## Prerequisites

- **Kubernetes cluster** with permissions to create/delete Jobs
- **Pre-built SWE-bench images** in a container registry accessible from the cluster
  - Default: `docker.io/swebench/sweb.eval.x86_64.<instance_id>:latest`
  - Configurable via `k8s_registry` parameter for private registries or the OpenShift/K8s internal registry

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions_path` | string | `gold` | Path to predictions file. Use `"gold"` for ground-truth patches, or a file path like `/test_data/predictions.jsonl` when using `test_data_ref.s3` |
| `k8s_registry` | string | `docker.io/swebench` | Container registry where SWE-bench images are pulled from |
| `instance_image_tag` | string | `latest` | Image tag for SWE-bench instance images. Use versioned tags (e.g. `v1`) to avoid caching issues on OpenShift |
| `max_workers` | integer | `10` | Maximum concurrent evaluation Jobs |
| `timeout_per_instance` | integer | `1800` | Per-instance timeout in seconds |
| `split` | string | `test` | Dataset split (`test` or `dev`) |
| `instance_ids` | array | `null` | Specific instance IDs to evaluate (all if null) |
| `save_sample_results` | boolean | `false` | Set `true` to persist completed-instance samples and their manifest; persistence failures then fail the job |

## Supported Benchmarks

| Benchmark ID | Dataset | Instances |
|-------------|---------|-----------|
| `swebench_verified` | SWE-bench Verified | 500 |
| `swebench_lite` | SWE-bench Lite | 300 |
| `swebench_full` | SWE-bench Full | 2294 |

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `resolve_rate` | accuracy | Fraction of instances where the patch resolves the issue (primary metric) |
| `resolved_count` | count | Number of resolved instances |
| `completed_count` | count | Number of instances that completed evaluation |
| `total_instances` | count | Total instances evaluated |
| `error_count` | count | Instances that errored during evaluation |

## Predictions Format

The adapter accepts the standard SWE-bench predictions format (JSON or JSONL):

```json
{
  "django__django-11099": {
    "instance_id": "django__django-11099",
    "model_patch": "diff --git a/...",
    "model_name_or_path": "my-model"
  }
}
```

## Using S3 Predictions

To evaluate predictions stored in S3 (or S3-compatible storage like MinIO),
use `test_data_ref.s3` on the benchmark config. The eval-hub platform
downloads the file to `/test_data/` via an init container before the
adapter starts.

Set `predictions_path` to the local path where the file will land:

```yaml
benchmarks:
  - id: swebench_verified
    provider_id: <provider-id>
    parameters:
      predictions_path: /test_data/predictions.jsonl
      max_workers: 8
    test_data_ref:
      s3:
        bucket: my-bucket
        key: runs/my-model/predictions.jsonl
        secret_ref: my-s3-credentials
```

The `secret_ref` references a K8s Secret containing S3 credentials
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally
`AWS_ENDPOINT_URL` for S3-compatible stores).

## RBAC Requirements

Unlike other eval-hub adapters which run evaluations inside a single pod,
the SWE-bench adapter creates **one K8s Job per evaluation instance** (each
instance needs its own isolated container with a different codebase and
dependencies).  This means the adapter pod's service account needs
permissions to manage Jobs -- a requirement unique to this adapter.

```yaml
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list"]
```

### OpenShift SCC Requirement

SWE-bench images have root-owned files.  The adapter uses an init
container that runs as root to fix permissions, then the eval container
runs as UID 1001.  On OpenShift the service account needs the `anyuid`
SCC for the init container.

The included `rbac.yaml` grants Job/pod access and the `anyuid` SCC to the
existing `evalhub-evalhub-opendatahub-job` service account; it does not create
a service account. The adapter must run as this account, or you must update both
binding subjects to match its account, which child Jobs inherit. Replace
`NAMESPACE` with your tenant namespace and apply (requires cluster-admin for
the cluster-scoped SCC binding):

```bash
sed 's/NAMESPACE/my-namespace/g' rbac.yaml | oc apply -n my-namespace -f -
```

With `KUBECONFIG` configured, the adapter reads its own Pod through the Kubernetes
API and uses that Pod's service account for child Jobs. This requires the `pods/get`
permission already included above; discovery failure stops the run rather than
silently selecting the namespace default. Without a kubeconfig, legacy token
discovery is retained, falling back to the namespace default when unavailable. Job parameters cannot select
a service account. Child pods
disable service-account token automounting for both init and evaluation containers.

## Local Development

```bash
# Run tests
cd adapters/swebench
pip install -r requirements.txt -r requirements-test.txt
python -m pytest tests/ -v

# Build container
make image-swebench

# Test with gold predictions (requires K8s cluster)
EVALHUB_JOB_SPEC_PATH=adapters/swebench/meta/job.json \
  python adapters/swebench/main.py
```

The `Test SWE-bench` GitHub Actions workflow runs the full unit and child-pod
security suite on SWE-bench pull requests and pushes to `main`. These tests mock
cluster execution and require no cluster credentials.

## Per-instance results

Per-instance persistence is opt-in: set `parameters.save_sample_results: true`
to write completed instances to `sample_results.jsonl`, with a stable instance
ID as `sample_id` and a numeric `resolve_rate` observation. Omitted or `false`
disables sample/manifest output; only boolean values are accepted.
When at least one instance completes, their mean matches the aggregate resolve
rate. Incomplete infrastructure runs remain in diagnostic artifacts and are
excluded from scored observations. A model patch that fails to apply is an
unresolved observation (0), included in both the aggregate denominator and sample
file. If none complete, the aggregate falls back to
0.0, but the sample file is empty and cannot produce a confidence interval.
If no prediction IDs match the selected dataset, the adapter returns 0.0 and
writes fresh empty `summary.json`/`results.json`, skipping OCI/MLflow exporters.
With sample persistence enabled it also writes an empty sample file and manifest;
otherwise stale sample/manifest files are removed. Unrelated files are preserved.
When enabled, configured OCI and MLflow exports include the same JSONL bytes
and primary-metric manifest. Sample/manifest files from a reused results
directory are removed when persistence is disabled. Invalid flags fail before
loading data or starting Jobs. With no remote export configured, opted-in
samples remain on the local filesystem.

SWE-bench's Kubernetes client honors `KUBECONFIG` when no in-cluster token is
mounted. Child Jobs inherit the adapter pod namespace selected by eval-hub for
the request tenant (`X-Tenant`); job parameters cannot override it. Outside a pod,
the namespace falls back to `default`. With `KUBECONFIG`, the account is read from the adapter Pod using its hostname
and namespace; discovery errors fail the run. Without it, legacy pod-token
discovery and the namespace-default fallback remain.
Job parameters cannot select the service account. Child pods never automatically
mount the account token.
All such credentials must be provisioned by the cluster, not copied from a laptop.
When sample persistence is enabled, local write errors and configured OCI/MLflow
export errors fail the job. A configured export returning no artifact/run
reference also fails. The failure event carries the underlying error in
`error_message`, exposed through the evaluation API at
`status.benchmarks[].error_message`; the process exits nonzero without reporting
completion. Full MLflow error details require [SDK PR #204](https://github.com/eval-hub/eval-hub-sdk/pull/204)
to be released and included in the image. When sample persistence is
disabled, diagnostic OCI export failures retain their prior warning-only
behavior and leave the OCI artifact reference unset.
`PERSISTING_ARTIFACTS` is reported only for configured OCI exports, as required
by the repository lifecycle; local writes and MLflow-only exports skip that phase.

### MLflow failure-detail dependency

Until [SDK PR #204](https://github.com/eval-hub/eval-hub-sdk/pull/204) is released
and included in the image, the SDK can report a generic MLflow failure before the
adapter reports the backend cause. Evaluation still fails, but full backend error
detail in the evaluation API is not guaranteed. A separate follow-up PR will
update the SDK dependency; this PR keeps the existing version constraint.
