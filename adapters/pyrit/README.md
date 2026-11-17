# PyRIT Adapter

Wraps [PyRIT](https://github.com/Azure/PyRIT) (MIT license), Microsoft's Python
Risk Identification Tool for generative AI, exposing a headless `pyrit_scan`
benchmark against an EvalHub-provided OpenAI-compatible model endpoint.

- **`pyrit-scan`** — run a named PyRIT scenario via
  `pyrit_scan run <scenario> --target openai_chat --start-server`

**Status (RHAIRFE-3602):** CLI bridge implemented (`run_benchmark_job` →
`pyrit_scan` + results fetch + `pass_rate` metrics). EvalHub catalog /
operator ConfigMap sync remains **RHAIRFE-3603** (this `provider.yaml` is the
source contract, not auto-mounted on contrib merge alone).

## Metric semantics

`overall_score` / `pass_rate` = resisted / `n_evaluated` among **scored**
probes only (`success` \| `failure`). `error` / `undetermined` are excluded
so a broken endpoint cannot look like a perfect pass. If nothing was scored,
`pass_rate` is omitted (`None`), never `1.0`.
**Lower pass_rate ⇒ more vulnerable.**

## How a job runs

1. Write job-local `.env` (`OPENAI_CHAT_*`, optional `ADVERSARIAL_CHAT_*` /
   `OBJECTIVE_SCORER_CHAT_*`) and `.pyrit_conf` (in-memory memory,
   `auth_mode: none`).
2. `pyrit_scan run <parameters.scenario> --target openai_chat --start-server`.
3. Parse scenario result id from CLI output, `GET` detailed results from the
   local backend, then `pyrit_scan stop-server`.
4. Map attack outcomes → `pass_rate` / `n_evaluated` / `attack_success_rate`.

Required job parameter: `parameters.scenario` (e.g. `foundry.red_team_agent`,
`airt.leakage`, `garak.encoding`). Optional: `max_objectives` (→
`--max-dataset-size`), `techniques` (concrete technique names only — not
aggregates like `easy`), `scan_timeout`, attacker/scorer provider overrides.

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
The adapter always passes `--start-server --auth-mode none` so the job does
not need a pre-running backend or interactive auth.

Local smoke (needs a reachable OpenAI-compatible endpoint in `meta/job.json`):

```bash
EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json \
  OPENAI_API_KEY=... python main.py
```

## Secrets

`attacker_provider_api_key` and `scorer_provider_api_key` are job parameters
(same pattern as promptfoo `generation_provider_api_key`). Implementation must
never embed them in `additional_info`, MLflow params, or OCI artifacts — see
`SECRET_PARAM_NAMES` / `test_secret_param_names` in the adapter tests.

## Image

`quay.io/evalhub/community-pyrit:latest`
