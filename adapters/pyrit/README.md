# PyRIT Adapter

Wraps [PyRIT](https://github.com/Azure/PyRIT) (MIT license), Microsoft's Python
Risk Identification Tool for generative AI, exposing a headless `pyrit_scan`
benchmark against an EvalHub-provided OpenAI-compatible model endpoint.

- **`pyrit-scan`** — run a named PyRIT scan scenario via `pyrit_scan run`

Scaffold for **RHAIRFE-3602**. Implementation (CLI bridge, metrics mapping,
OCI/MLflow artifact handling) lands in follow-up commits in this adapter
directory.

## Layout

```text
adapters/pyrit/
  main.py              # PyritAdapter (FrameworkAdapter)
  provider.yaml        # EvalHub provider / benchmark catalog
  Containerfile        # UBI9 python-311 image
  requirements.txt
  requirements-test.txt
  meta/job.json        # Local / unit-test job fixture
  tests/
    conftest.py
    test_adapter.py
```

## Local development

```bash
# Unit tests (no live LLM)
make test-pyrit

# Image build / push (once Makefile targets are wired)
make image-pyrit
make push-pyrit
```

Local smoke (after implementation):

```bash
EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
```

## Image

`quay.io/evalhub/community-pyrit:latest`
