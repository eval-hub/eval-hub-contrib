# Contributing

This repository contains community-contributed adapters that integrate evaluation frameworks with [eval-hub](https://github.com/eval-hub/eval-hub). Each adapter implements the `FrameworkAdapter` pattern from the evalhub-sdk.

## Adding an adapter

1. Create a new directory under `adapters/` named after the framework (e.g. `adapters/my-framework/`)
2. Implement `main.py` using the `FrameworkAdapter` pattern from evalhub-sdk
3. Add a `Containerfile` and `requirements.txt`
4. Add build/push targets to the root `Makefile`
5. Document the adapter in its own `README.md`

## Building adapters

```sh
# Build a specific adapter
make image-lighteval
make image-guidellm

# Build all adapters
make images

# Push to a registry
make push-lighteval REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-guidellm REGISTRY=quay.io/your-org VERSION=v1.0.0
```

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org). All commit messages must follow the format:

```
<type>(<scope>): <subject>
```

**Common types:** `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

**Examples:**

```
feat(lighteval): add support for custom task configuration
fix(guidellm): handle connection timeout on benchmark run
docs: update adapter development guide
chore: bump dependencies
```

PRs targeting `main` will fail CI if any commit message does not follow this format.

If you have [pre-commit](https://pre-commit.com) installed, commit messages are also checked locally:

```sh
pre-commit install --hook-type commit-msg
```
