# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

eval-hub-contrib is a collection of community-contributed evaluation framework adapters for the eval-hub service. Each adapter under `adapters/` wraps an external evaluation framework (LightEval, GuideLLM, MTEB, IBM CLEAR) and implements the `FrameworkAdapter` pattern from the `eval-hub-sdk`.

## Build and test commands

Container images are built with `podman` by default (override with `BUILD_TOOL=docker`):

```sh
make image-lighteval          # build one adapter
make images                   # build all adapters
make push-lighteval REGISTRY=quay.io/your-org VERSION=v1.0.0
```

Each adapter has its own Python venv, dependencies, and pytest suite:

```sh
make test-lighteval           # run one adapter's tests
make test-clear               # run one adapter's tests
make tests                    # run all adapter tests
```

To run a single test file or test case manually:

```sh
cd adapters/lighteval
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-test.txt
.venv/bin/pytest tests/test_adapter.py -v             # all tests in file
.venv/bin/pytest tests/test_adapter.py::test_lighteval_happy_path -v  # single test
```

## Commit conventions

This project enforces [Conventional Commits](https://www.conventionalcommits.org) via commitizen. A pre-commit hook validates commit messages (install with `pre-commit install --hook-type commit-msg`). CI will also reject non-conforming messages on PRs to `main`.

Format: `<type>(<scope>): <subject>`

Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`. Scope is typically the adapter name (e.g., `lighteval`, `guidellm`, `clear`, `mteb`).

## Architecture

### Adapter structure

Every adapter lives in `adapters/<name>/` with a consistent layout:

- `main.py` -- single-file adapter implementing `FrameworkAdapter.run_benchmark_job(config, callbacks) -> JobResults`
- `provider.yaml` -- declares the adapter to eval-hub (id, runtime resources, benchmarks, parameters). CI validates this file on new-adapter PRs.
- `meta/job.json` -- sample `JobSpec` used by tests
- `tests/` -- pytest suite; `conftest.py` sets up fixtures, `test_adapter.py` tests the happy-path plumbing by monkeypatching the framework execution (not the eval-hub-sdk layer)
- `Containerfile`, `requirements.txt`, `requirements-test.txt`

### FrameworkAdapter contract (from eval-hub-sdk)

Each adapter subclass:

1. Receives a `JobSpec` (benchmark id, model config, parameters) and `JobCallbacks`
2. Reports progress through a fixed lifecycle of phases: `INITIALIZING` -> `LOADING_DATA` -> `RUNNING_EVALUATION` -> `POST_PROCESSING` -> `PERSISTING_ARTIFACTS`
3. Invokes the underlying framework (typically via subprocess CLI)
4. Extracts metrics into `EvaluationResult` objects and computes an `overall_score`
5. Optionally persists detailed results as OCI artifacts via `callbacks.create_oci_artifact()`
6. Returns `JobResults`

### Adding a new adapter

1. Create `adapters/<name>/` with the files above
2. Add `provider.yaml` (see `adapters/mteb/provider.yaml` for an annotated example)
3. Add build/push/test targets to the root `Makefile`
4. Add a CI workflow at `.github/workflows/test-<name>.yml`
