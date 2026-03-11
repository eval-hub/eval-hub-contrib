# eval-hub-contrib

Community-contributed evaluation framework adapters for eval-hub.

## Overview

This repository contains adapters that integrate various evaluation frameworks with the eval-hub service. Each adapter implements the `FrameworkAdapter` pattern from the evalhub-sdk, enabling seamless integration with the eval-hub evaluation service.

## Supported Frameworks

| Framework | Container Image | Local | Kubernetes | Notes |
|-----------|----------------|-------|------------|-------|
| [LightEval](https://github.com/huggingface/lighteval) | `quay.io/evalhub/community-lighteval:latest` | ✗ | ✓ | Lightweight evaluation framework for language models |
| [GuideLLM](https://github.com/vllm-project/guidellm) | `quay.io/evalhub/community-guidellm:latest` | ✗ | ✓ | Performance benchmarking platform for LLM inference servers |
| [MTEB](https://github.com/embeddings-benchmark/mteb) | `quay.io/evalhub/community-mteb:latest` | ✗ | ✓ | Massive Text Embedding Benchmark for embedding models |

## Building Adapters

```bash
# Build specific adapter
make image-lighteval
make image-guidellm

# Build all adapters
make images

# Push to registry
make push-lighteval REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-guidellm REGISTRY=quay.io/your-org VERSION=v1.0.0
```
## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding adapters.

## License

See the [LICENSE](LICENSE) file for details.
