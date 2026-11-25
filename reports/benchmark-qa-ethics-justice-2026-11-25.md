# Benchmark QA report: `ethics_justice`

## Summary

**Result:** PASS for benchmark loading, API execution, and disconnected egress controls. The `1,000`-sample run returned `acc=0.501`; this is at the task's 0.5 random-choice baseline and does not establish useful model quality.

**Provider / benchmark:** contrib `lm-eval / ethics_justice`; EvalHub catalog `lm_evaluation_harness / ethics_justice`. The live adapter runs used a temporary user provider (`8250f76e-9672-4947-9431-9bdb4e9c064c`) to execute the newly built contrib image through the EvalHub API.

**Run date:** 2026-11-25 UTC

The upstream task resolves in pinned `lm_eval==0.4.13`, and the connected API run loaded the dataset and tokenizer. The disconnected test initially exposed an OpenShift DNS allowlist mismatch and then a read-only Hugging Face cache lock. Both issues were corrected. The final image completed disconnected runs with the S3-staged dataset/tokenizer bundle, and a live probe from the adapter container confirmed that only configured cluster destinations were reachable.

The source audit found `ethics_justice` absent from both benchmark catalogs before these changes.

## Source and implementation

| Item | Value |
| --- | --- |
| Upstream task | [ETHICS task README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_ethics/README.md) and [`justice.yaml`](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/hendrycks_ethics/justice.yaml) |
| Dataset repo / subset / split | `EleutherAI/hendrycks_ethics / justice / test`; 2,704 test examples |
| Tokenizer | Connected: `google/flan-t5-small`; disconnected: `/test_data/tokenizer` |
| Framework and dependencies | `lm_eval==0.4.13`, `datasets==3.6.0`; task validation passed |
| Image | `quay.io/rh-ee-nbs/nbs-dev:ethics-justice-offlinefix-20261125`, `linux/amd64`, runtime digest `sha256:89b1822834668f95114c858119581f3ac50e84e7880f7164766ff6b79d92a6c6` |
| Contrib branch | `codex/benchmark-onboarding-ethics-justice` |
| EvalHub branch | `codex/benchmark-onboarding-ethics-justice` |

The provider catalogs declare binary `acc`, `num_few_shot: 0`, and a 0.5 random-choice baseline. The adapter task mapping uses loglikelihood, which requires an OpenAI-compatible completions endpoint with echo and logprobs support. `trust_remote_code=true` is needed by the upstream dataset loader.

## Parameter coverage

| Parameter | Values exercised | Result / applicability |
| --- | --- | --- |
| `num_examples` | API runs: 10, 100, 1,000; feature fixture: 1 | Limits were accepted and honored |
| `num_fewshot` | `0` | Zero-shot result completed |
| `trust_remote_code` | `true` | Upstream dataset loader succeeded |
| `api_style` | `completions` | Required for loglikelihood; sidecar served the completions endpoint |
| `tokenizer_backend` | `huggingface` | Tokenizer loaded in both modes |
| `tokenizer` | Connected model ID; disconnected `/test_data/tokenizer` | Both paths loaded |
| `hf_offline` | Connected `false`; disconnected `true` | Disconnected subprocess had `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| `hf_datasets_cache` | Disconnected `/test_data/hf-cache` | Copied into a writable cache before dataset loading |
| `test_data_ref` | S3 `mlpipeline/offline/fvt/benchmarks/lm_evaluation_harness/ethics_justice/` | Init completed after downloading 53 files (~81 MiB) |
| Model endpoint/auth | In-cluster vLLM completions URL; existing `vllm-llama-secret` reference | API run completed; Secret data was not read or copied |
| Runtime controls | `batch_size=1`, `num_concurrent=1`, `timeout_http=120` | Accepted by the adapter |
| Not applicable | `gen_kwargs`, `custom_tasks_path`, `allow_code_execution`, adapter `dataset_source=s3` | Built-in loglikelihood task; no generation, custom task, or generated-code execution. S3 is used by EvalHub's asset staging reference, while lm-eval reads the staged HF cache. |

## Scenario results

| Scenario | Mode | API result | Evidence |
| --- | --- | --- | --- |
| Connected smoke | Connected | `completed`; 10 samples; `acc=0.5` | Job `15346282-efaf-49ca-a074-5196b5199347`; run used the pre-cache-fix image, and the changed cache-copy path is only activated for offline runs |
| Initial policy attempt | Disconnected | Init failed before adapter start; API remained pending | Job `38442aa7-e52f-4097-a036-d367d92a7884`; DNS to MinIO timed out |
| Initial staged-cache attempt | Disconnected | Adapter failed before scoring | Job `2947488a-810b-4e59-bb76-71a39484f732`; Datasets could not create a lock file in read-only `/test_data/hf-cache` |
| Offline fix smoke | Disconnected | `completed`; 10 samples; `acc=0.5` | Job `77134cfd-46ca-4d04-bb5b-a7201e41d41c`; writable cache copy confirmed in adapter log |
| Offline repeated run | Disconnected | `completed`; 100 samples; `acc=0.51` | Job `8ce751fc-922d-4d89-b77a-4729076474d4` |
| Offline egress run | Disconnected | `completed`; 1,000 samples; `acc=0.501` | Job `0e6e04cd-e5fd-43d6-8fd7-8f44db1fe904`; adapter pod image digest and live egress probes recorded below |

The final init container reported `download complete files=53 mb=81` and `eval-runtime-init completed`. The adapter log confirmed it prepared the staged Hugging Face cache and evaluated 1,000 examples. The score is effectively at chance for this binary task, so these runs validate the pipeline and assets rather than model capability.

## Cluster resources provisioned

| Namespace | Resource | Purpose / source | Cleanup |
| --- | --- | --- | --- |
| `sagar` | PVC `ethics-justice-onboarding-20261125`, 5 Gi | Staging workspace; reached `Bound` | Removed after evidence was saved |
| `sagar` | Existing Secret reference `vllm-llama-secret` | Model endpoint authentication | Reused; secret contents were not inspected |
| `sagar` | Existing Secret reference `inspect-test-data-s3` | Staging bundle from the in-cluster S3-compatible service | Reused; secret contents were not inspected |
| `sagar` | Temporary provider `8250f76e-9672-4947-9431-9bdb4e9c064c` | Runs the contrib adapter image through the API | Deleted after evidence was saved |
| `sagar` | NetworkPolicy `ethics-justice-onboarding-egress` | Provider/benchmark-scoped disconnected allowlist | Deleted after the run |
| `sagar` | Staging/upload and egress-probe Jobs and ConfigMaps | Dataset staging and connectivity checks | Removed after the run |
| `sagar` | EvalHub per-execution Jobs, ConfigMaps, and Ingress-only NetworkPolicies | API-created runtime resources for the test jobs | Removed; temporary API job records were hard-deleted after saving results here |
| Shared test bucket | ETHICS Justice fixture, 53 files (~81 MiB) | EvalHub disconnected FVT prefix listed above | Retain for the disconnected FVT |

## Egress controls

The provider-scoped `Egress` policy selected the evaluation pod using `app=evalhub`, `component=evaluation-job`, provider ID `8250f76e-9672-4947-9431-9bdb4e9c064c`, and benchmark ID `ethics_justice`. It allowed cluster DNS (including the OpenShift DNS service IP and target port), EvalHub on TCP 8443, the configured model service on TCP 8000, and the MinIO fixture source on TCP 9000. No public CIDRs or Hugging Face destinations were allowed.

EvalHub also created a per-execution NetworkPolicy for the same pod. Its policy type was `Ingress` only, so it did not add any egress destinations.

From the live adapter container, TCP probes reported EvalHub, model, and MinIO `ALLOWED`; `huggingface.co` and `1.1.1.1` `BLOCKED`. A separate read of the live `lm_eval` subprocess environment found all three offline flags set to `1`. The adapter's model calls used the pod-local sidecar endpoint.

**Boundary:** Kubernetes NetworkPolicy is pod-scoped. The adapter and proxy sidecar share the pod network namespace, so the adapter can also reach the cluster destinations allowed for the sidecar. The test proves that the pod cannot reach Hugging Face or public Internet destinations; it does not provide per-container isolation.

## Failure analysis

| Test | Symptom | Evidence | Cause and confidence | Fix / impact |
| --- | --- | --- | --- | --- |
| First disconnected attempt | Init container timed out resolving `minio.minio.svc.cluster.local` | Job `38442aa7-e52f-4097-a036-d367d92a7884`; DNS Service uses port 53 with DNS pod target port 5353 | NetworkPolicy allowed service port 53 only; high confidence | Added the cluster DNS service IP and pod target port 5353 for UDP/TCP. Follow-up egress probe passed. |
| Second disconnected attempt | Dataset cache lock creation failed with `Read-only file system` | Job `2947488a-810b-4e59-bb76-71a39484f732`; adapter traceback pointed to a `.lock` file under `/test_data/hf-cache` | EvalHub mounts the staged bundle read-only, while Datasets writes locks into its cache; high confidence | Adapter copies the staged cache into its writable work directory before launching lm-eval. Unit tests cover the copy and missing-cache failures; 10/100/1,000-example disconnected reruns passed. |
| Score interpretation | `acc` ranged from 0.5 to 0.51 | Completed API results; task's binary random baseline is 0.5 | Expected for the smoke model and sample sizes; no evidence of a loader or scoring defect | Keep this as a pipeline PASS only. Do not interpret as evidence of model quality. |

## Validation

- `lm_eval validate --tasks ethics_justice`: passed (`lm_eval==0.4.13`).
- `make test-lm-eval`: 1,170 passed, 24 skipped; Ruff passed. Existing deprecation warnings were emitted.
- EvalHub `make fmt lint`: passed; golangci-lint reported 0 issues.
- `go test ./tests/features -run '^TestBenchmarkProviderFixtureCounts$'`: passed; the new Jsonnet request fixture compiles for connected and disconnected modes.
- Skill repository `make lint`: passed, including `skillsaw` on `benchmark-onboarding`.
- Skill repository tests: 50 passed.
- `git diff --check`: passed in all changed repositories.
