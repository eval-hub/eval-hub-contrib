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
| `k8s_namespace` | string | current namespace | Kubernetes namespace for evaluation Jobs (auto-detected if not set) |
| `instance_image_tag` | string | `latest` | Image tag for SWE-bench instance images. Use versioned tags (e.g. `v1`) to avoid caching issues on OpenShift |
| `max_workers` | integer | `10` | Maximum concurrent evaluation Jobs |
| `timeout_per_instance` | integer | `1800` | Per-instance timeout in seconds |
| `split` | string | `test` | Dataset split (`test` or `dev`) |
| `instance_ids` | array | `null` | Specific instance IDs to evaluate (all if null) |

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
      k8s_namespace: my-namespace
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

The included `rbac.yaml` creates a `swebench-eval` service account
with all necessary RBAC and SCC bindings.  Replace `NAMESPACE` with
your tenant namespace and apply:

```bash
sed 's/NAMESPACE/my-namespace/g' rbac.yaml | oc apply -f -
```

Set `service_account: swebench-eval` in the adapter parameters so the
eval Jobs use this service account.

## Local Development

```bash
# Run tests
cd adapters/swebench
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -v

# Build container
make image-swebench

# Test with gold predictions (requires K8s cluster)
EVALHUB_JOB_SPEC_PATH=adapters/swebench/meta/job.json \
  python adapters/swebench/main.py
```
