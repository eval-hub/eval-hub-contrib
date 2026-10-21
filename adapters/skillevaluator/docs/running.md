# Running the SkillEvaluator adapter

You need an EvalHub server with a Kubernetes job runtime, the `evalhub` CLI pointed at it, and a registry the
cluster can pull from.

```sh
pip install "eval-hub-sdk[cli]~=1.0.6"
evalhub config set base_url https://evalhub.example.com
evalhub config set token "$EVALHUB_TOKEN"
```

## 1. Build and publish the image

```sh
make image-skillevaluator REGISTRY=registry.example.com/myorg VERSION=0.1.0
make push-skillevaluator  REGISTRY=registry.example.com/myorg VERSION=0.1.0
```

`make help` lists the related targets (`clean-skillevaluator`, `build-and-push-skillevaluator`).

## 2. Register the provider

Copy `provider.yaml`, set `runtime.k8s.image` to the image you pushed (a digest-pinned reference is best), and
register it:

```sh
evalhub providers create --file provider.yaml
evalhub providers list          # note the provider id
```

If your server rejects `image_pull_policy: always` for provider registration, use `if_not_present`.

## 3. Describe the job

[`examples/skill-eval.yaml`](../examples/skill-eval.yaml) evaluates one skill from a git repository with all
three benchmarks. `model` is required by the job schema but unused. Use the provider id returned above (or
`skillevaluator` if your server keeps the manifest id).

```yaml
name: skill-gate
model: {url: http://unused/v1, name: unused}
benchmarks:
  - id: skill-quality
    provider_id: skillevaluator
    parameters: {min_score: 80}
    test_data_ref:
      git:
        url: https://github.com/example-org/agent-skills.git
        ref: v1.2.0                      # branch, tag or full commit SHA
        sub_path: skills/changelog-formatter
```

`test_data_ref` is set per benchmark and takes exactly one source. Instead of `git`, use one of:

```yaml
# Hugging Face Hub
test_data_ref:
  hf: {repo_id: example-org/agent-skills, revision: main, sub_path: skills/changelog-formatter}

# a PersistentVolumeClaim already in the job namespace
test_data_ref:
  pvc: {claim_name: skills, sub_path: skills/changelog-formatter}

# S3-compatible storage
test_data_ref:
  s3: {bucket: my-bucket, key: skills/changelog-formatter, secret_ref: my-s3-connection}
```

Private repositories use `secret_ref` (a Kubernetes Secret in the job namespace), never credentials in the job
file. If one source holds several skills, leave out `sub_path` and set `parameters.skill_path` instead.

## 4. Submit and read the result

```sh
evalhub eval run --config examples/skill-eval.yaml --wait --timeout 900
evalhub eval status <job-id> --format json
```

Without `--wait` the command prints the job id and returns. Illustrative metrics for a healthy skill:

| Benchmark | Key metrics |
|---|---|
| `skill-quality` | `quality_score` 0.91, `quality_grade` A, `meets_min_score` true |
| `skill-security` | `gate_score` 1.0, `scan_complete` true, `critical_count` 0 |
| `skill-pii` | `gate_score` 1.0, `finding_count` 0 |

A failed benchmark carries a message code: `scan_incomplete` (see [incomplete scans](incomplete-scans.md)), a
missing or ambiguous skill (set `sub_path` or `skill_path`), or an unexpected upstream report shape.

## Parameters

| Name | Default | Description |
|---|---|---|
| `skill_path` | auto | Skill directory under `/test_data`; required only if more than one skill is staged |
| `min_score` | `70` | `skill-quality`: bar for the `meets_min_score` metric (0-100) |
| `on_incomplete` | `fail` | `fail` or `report`, see [incomplete scans](incomplete-scans.md) |
| `timeout_seconds` | `300` | Per-check timeout (10-3600) |
| `submitter` | none | `skill-pii`: enables the personal home-path check |
| `mlflow_experiment_name` | none | Fallback MLflow experiment name |

The upstream content digest of the evaluated tree is recorded as `evaluation_metadata.skill_digest`.
