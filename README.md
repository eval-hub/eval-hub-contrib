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
| Framework | Container Image | Local | Kubernetes | Notes |
|-----------|----------------|-------|------------|-------|
| [LightEval](https://github.com/huggingface/lighteval) | `quay.io/evalhub/community-lighteval:latest` | ✗ | ✓ | Lightweight evaluation framework for language models |
| [GuideLLM](https://github.com/vllm-project/guidellm) | `quay.io/evalhub/community-guidellm:latest` | ✗ | ✓ | Performance benchmarking platform for LLM inference servers |
| [MTEB](https://github.com/embeddings-benchmark/mteb) | `quay.io/evalhub/community-mteb:latest` | ✗ | ✓ | Massive Text Embedding Benchmark for embedding models |
| [IBM CLEAR](https://github.com/IBM/CLEAR) | `quay.io/evalhub/community-ibm-clear:latest` | ✓ | ✓ | Agentic trace analysis (LLM-as-judge error reporting) |
| [RAGAS](https://github.com/explodinggradients/ragas) | `quay.io/evalhub/community-ragas:latest` | ✗ | ✓ | RAG pipeline quality evaluation (faithfulness, relevancy, context precision/recall, and more) |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | `quay.io/evalhub/community-swebench:latest` | ✗ | ✓ | Software engineering benchmark for code patch evaluation |

## JobPhase Lifecycle

Every adapter must report progress through the `JobPhase` lifecycle via `callbacks.report_status()`. The server validates phases against a fixed set, so adapters must emit them in order and use only the values listed below.

### Phases

1. **`INITIALIZING`** — Validate configuration, resolve credentials, set up temporary directories. Emit at the start of `run_benchmark_job`.
2. **`LOADING_DATA`** — Load datasets, download test data, prepare inputs. Emit before any data I/O.
3. **`RUNNING_EVALUATION`** — Execute the framework (subprocess, API call, etc.). Emit before the main workload begins.
4. **`POST_PROCESSING`** — Parse results, extract metrics, compute scores. Emit after the framework finishes.
5. **`PERSISTING_ARTIFACTS`** — Create OCI artifacts from result files. Emit **only when OCI exports are configured** (`config.exports.oci`). Skip this phase entirely when there is nothing to persist.
6. **`COMPLETED`** — **Do not emit manually.** This phase is sent automatically by `callbacks.report_results()`.

### Status update format

Only `status` and `phase` are forwarded to the server. Other fields (`progress`, `message`, `current_step`, etc.) are silently dropped by the SDK.

```python
# Success path — emit for each phase
callbacks.report_status(
    JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
)

# Failure path — use error_message (ErrorInfo is deprecated)
callbacks.report_status(
    JobStatusUpdate(
        status=JobStatus.FAILED,
        error_message=MessageInfo(message=str(e), message_code="evaluation_error"),
    )
)
```

### PERSISTING_ARTIFACTS gating

The `PERSISTING_ARTIFACTS` phase must only be reported when OCI exports are configured. When no exports are configured, skip both the phase and the OCI call:

```python
oci_artifact = None
oci_exports = config.exports.oci if config.exports else None
if oci_exports is not None and output_files:
    callbacks.report_status(
        JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS)
    )
    oci_artifact = callbacks.create_oci_artifact(
        OCIArtifactSpec(files_path=results_dir, coordinates=oci_exports.coordinates)
    )
```

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
