# eval-hub-contrib

Community-contributed evaluation framework adapters for eval-hub.

## Overview

This repository contains adapters that integrate various evaluation frameworks with the eval-hub service. Each adapter implements the `FrameworkAdapter` pattern from the evalhub-sdk, enabling seamless integration with the eval-hub evaluation service.

## Supported Frameworks

| Framework | Container Image | Kubernetes | Notes |
|-----------|----------------|------------|-------|
| [LightEval](https://github.com/huggingface/lighteval) | `quay.io/evalhub/community-lighteval:latest` | ✓ | Lightweight evaluation framework for language models |
| [GuideLLM](https://github.com/vllm-project/guidellm) | `quay.io/evalhub/community-guidellm:latest` | ✓ | Performance benchmarking for LLM inference servers |
| [MTEB](https://github.com/embeddings-benchmark/mteb) | `quay.io/evalhub/community-mteb:latest` | ✓ | Massive Text Embedding Benchmark for embedding models |
| [IBM CLEAR](https://github.com/IBM/CLEAR) | `quay.io/evalhub/community-ibm-clear:latest` | ✓ | Agentic trace analysis (LLM-as-judge error reporting) |
| [Inspect AI](https://inspect.aisi.org.uk/) | `quay.io/evalhub/community-inspect:latest` | ✓ | UK AISI framework — Petri/Bloom alignment auditing and 75 inspect-evals benchmarks |

## Inspect AI Adapter

The Inspect AI adapter exposes alignment auditing and safety evaluation through the [Petri](https://meridianlabs-ai.github.io/inspect_petri/) and [Bloom](https://meridianlabs-ai.github.io/inspect_petri/extensions/petri-bloom.html) tools from Meridian Labs, as well as 35 curated benchmarks from the [inspect-evals](https://github.com/UKGovernmentBEIS/inspect_evals) community library.

**75 benchmarks** across three categories:

- **36 Petri alignment audits** — covers all 40 built-in seed tag categories including sycophancy, deception, alignment faking, jailbreak, harmful cooperation, self-preservation, power seeking, oversight subversion, and more.
- **2 Bloom behavioral suites** — automated scenario generation from high-level behavior descriptions.
- **36 inspect-evals** — safety (AgentHarm, WMDP, StrongREJECT, MASK), scheming (agentic misalignment, GDM self-proliferation, GDM stealth), cybersecurity (Cybench, CyberSecEval), coding (HumanEval, SWE-bench), math (GSM8K, MATH, AIME), knowledge (MMLU, GPQA), and agent capabilities (GAIA, TheAgentCompany).

**Model configuration** — no provider prefixes required in job specs. The adapter detects the correct API from environment variables:

| Environment variable | API used |
|---|---|
| `OPENAI_BASE_URL` + `OPENAI_API_KEY` | OpenAI-compatible (vLLM, OpenRouter) |
| `OLLAMA_BASE_URL` or port 11434 | Ollama native |
| `ANTHROPIC_API_KEY` | Anthropic Messages API |

See [adapters/inspect/README.md](adapters/inspect/README.md) for full documentation, deployment examples, and benchmark catalog.

## Building Adapters

```bash
# Build specific adapter
make image-lighteval
make image-guidellm
make image-inspect

# Build all adapters
make images

# Run adapter tests
make test-inspect
make tests

# Push to registry
make push-inspect REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-lighteval REGISTRY=quay.io/your-org VERSION=v1.0.0
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding adapters.

## License

See the [LICENSE](LICENSE) file for details.
