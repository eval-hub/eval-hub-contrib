# PyRIT Adapter

Wraps [PyRIT](https://github.com/Azure/PyRIT) (MIT license), Microsoft's Python
Risk Identification Tool for generative AI, exposing a headless `pyrit_scan`
benchmark against an EvalHub-provided OpenAI-compatible model endpoint.

- **`pyrit-scan`** — run a named PyRIT scan scenario via `pyrit_scan run`

**Status (RHAIRFE-3602):** scaffold + CI/image wiring. `PyritAdapter.run_benchmark_job`
is not implemented yet — keep the PR Draft until the CLI bridge lands. EvalHub
catalog / operator ConfigMap sync is **RHAIRFE-3603** (this `provider.yaml` is
the source contract, not auto-mounted on contrib merge alone).

## Metric semantics

`overall_score` / `pass_rate` = resisted objectives / `n_evaluated`
(= `1 - attack_success_rate`). **Lower pass_rate ⇒ more vulnerable.**

## Layout

```text
adapters/pyrit/
  main.py                 # PyritAdapter (FrameworkAdapter)
  provider.yaml           # EvalHub provider / benchmark catalog
  Containerfile           # UBI9 python-311 + unixODBC + pyrit==1.1.0
  requirements.txt        # SDK + pyyaml (unit tests)
  requirements-pyrit.txt  # pyrit runtime pin (image only)
  requirements-test.txt
  meta/job.json
  tests/
```

## Local development

```bash
make test-pyrit
make image-pyrit REGISTRY=quay.io/evalhub
# Local aarch64 build (2026-11-12): ~2.84 GB virtual size after unixODBC + pyrit==1.1.0
podman image inspect quay.io/evalhub/community-pyrit:latest --format '{{.Size}}'
make push-pyrit REGISTRY=quay.io/evalhub
```

`build-images.yml` maps adapter dir `pyrit` → `quay.io/evalhub/community-pyrit`
by default (no special `IMAGE_MAP` entry required).

**Image notes (William review):** `pyrit==1.1.0` pulls azure-*, pyodbc, av, scipy, datasets, etc.
The Containerfile installs `unixODBC` (pyodbc) and a temporary Rust toolchain
(`base2048` has no wheel on some arches), then removes build-only packages.
`constraints.txt` pins the transitive tree from a successful image freeze.
`pyrit_scan --help` reports that backend commands expect a running PyRIT
server (`start-server` / `--server-url`) — the adapter CLI bridge must account
for that when implementing `run_benchmark_job`.

Local smoke (after implementation):

```bash
EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
```

## Secrets

`attacker_provider_api_key` and `scorer_provider_api_key` are job parameters
(same pattern as promptfoo `generation_provider_api_key`). Implementation must
never embed them in `additional_info`, MLflow params, or OCI artifacts — see
`SECRET_PARAM_NAMES` / `test_secret_param_names` in the adapter tests.

## Image

`quay.io/evalhub/community-pyrit:latest`
