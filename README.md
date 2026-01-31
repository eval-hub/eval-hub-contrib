# eval-hub-contrib

Community-contributed evaluation framework adapters for eval-hub.

## Overview

This repository contains adapters that integrate various evaluation frameworks with the eval-hub service. Each adapter implements the `FrameworkAdapter` pattern from the evalhub-sdk, enabling seamless integration with the eval-hub evaluation service.

## Supported Frameworks

| Framework | Container Image | Local | Kubernetes | Notes |
|-----------|----------------|-------|------------|-------|
| [LightEval](https://github.com/huggingface/lighteval) | `quay.io/eval-hub/community-lighteval:latest` | ✗ | ✓ | Lightweight evaluation framework for language models |

## Building Adapters

```bash
# Build specific adapter
make image-lighteval

# Build all adapters
make images

# Push to registry
make push-lighteval REGISTRY=quay.io/your-org VERSION=v1.0.0
```

## License

See the [LICENSE](LICENSE) file for details.
