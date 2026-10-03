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
| [Inspect AI](https://inspect.aisi.org.uk/) | `quay.io/evalhub/community-inspect:latest` | ✓ | UK AISI framework — 75 benchmarks total: 36 Petri alignment audits, 2 Bloom suites, and 37 inspect-evals benchmarks |
| [RAGAS](https://github.com/explodinggradients/ragas) | `quay.io/evalhub/community-ragas:latest` | ✓ | RAG pipeline quality evaluation (faithfulness, relevancy, context precision/recall, and more) |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | `quay.io/evalhub/community-swebench:latest` | ✓ | Software engineering benchmark for code patch evaluation |
| [DeepEval](https://github.com/confident-ai/deepeval) | `quay.io/evalhub/community-deepeval:latest` | ✓ | LLM-as-judge evaluation: faithfulness, relevancy, hallucination, correctness, summarization, and multi-turn conversation metrics |
| [RULER](https://github.com/NVIDIA/RULER) | `quay.io/evalhub/community-ruler:latest` | ✓ | NVIDIA RULER long-context benchmark — 13 synthetic tasks across needle-in-a-haystack, variable tracking, aggregation, and QA at configurable context lengths |
| [WildGuard](https://arxiv.org/abs/2406.18495) | `quay.io/evalhub/community-wildguard:latest` | ✓ | AllenAI safety classification benchmark — evaluates a model's ability to classify prompt+response pairs as safe or unsafe, reporting accuracy and per-class recall |
| [IFBench](https://arxiv.org/abs/2507.02833) | `quay.io/evalhub/community-ifbench:latest` | ✓ | AllenAI precise instruction-following benchmark — 58 OOD verifiable constraints with programmatic scoring (prompt-level loose accuracy) |
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | `quay.io/eval-hub/community-nemo-guardrails:latest` | ✓ | Safety rail evaluation — prompt injection and toxicity detection benchmarks |
| [FollowBench](https://github.com/YJiangcm/FollowBench) | `quay.io/evalhub/community-followbench:latest` | ✓ | Multi-level instruction-following benchmark with HSR, SSR, and CSL metrics |
| [promptfoo](https://github.com/promptfoo/promptfoo) | `quay.io/evalhub/community-promptfoo:latest` | ✓ | Assertion-based prompt regression testing (`promptfoo-eval`) and adversarial red-team scanning (`promptfoo-redteam`) — 170+ OWASP LLM Top 10 plugins plus industry-vertical compliance packs (financial, medical, insurance, telecom, and more) |

## Inspect AI Adapter

The Inspect AI adapter exposes alignment auditing and safety evaluation through the [Petri](https://meridianlabs-ai.github.io/inspect_petri/) and [Bloom](https://meridianlabs-ai.github.io/inspect_petri/extensions/petri-bloom.html) tools from Meridian Labs, as well as curated benchmarks from the [inspect-evals](https://github.com/UKGovernmentBEIS/inspect_evals) community library.

**75 benchmarks** across three categories:

- **36 Petri alignment audits** — covers all 40 built-in seed tag categories including sycophancy, deception, alignment faking, jailbreak, harmful cooperation, self-preservation, power seeking, oversight subversion, and more.
- **2 Bloom behavioral suites** — automated scenario generation from high-level behavior descriptions.
- **37 inspect-evals** — safety (AgentHarm, WMDP, StrongREJECT, MASK), scheming (agentic misalignment, GDM self-proliferation, GDM stealth), cybersecurity (Cybench, CyberSecEval), coding (HumanEval, SWE-bench), math (GSM8K, MATH, AIME), knowledge (MMLU, GPQA), and agent capabilities (GAIA, TheAgentCompany).

**Model configuration** — no provider prefixes required in job specs. The adapter detects the correct API from environment variables:

| Environment variable | API used |
|---|---|
| `OPENAI_BASE_URL` + `OPENAI_API_KEY` | OpenAI-compatible (vLLM, OpenRouter) |
| `OLLAMA_BASE_URL` or port 11434 | Ollama native |
| `ANTHROPIC_API_KEY` | Anthropic Messages API |

See [adapters/inspect/README.md](adapters/inspect/README.md) for full documentation, deployment examples, and benchmark catalog.

## DeepEval Adapter

The DeepEval adapter integrates [DeepEval](https://github.com/confident-ai/deepeval) into eval-hub using an LLM-as-judge approach. A separate judge model scores outputs against configurable thresholds. Test data is loaded from CSV, JSONL, or JSON files and mapped to either single-turn or multi-turn DeepEval test cases.

**8 benchmarks** across two categories:

- **5 single-turn** — faithfulness (retrieval grounding), answer relevancy, hallucination detection, factual correctness, and summarization quality.
- **3 multi-turn** — conversation completeness (all user needs are addressed), role adherence (chatbot stays in persona), and knowledge retention (chatbot recalls user-disclosed information across turns).

**Judge model configuration** — the adapter accepts an independent judge model separate from the evaluated model:

| Parameter | Description |
|---|---|
| `eval_model_name` | Judge model name (defaults to the evaluated model) |
| `eval_model_url` | OpenAI-compatible base URL for the judge endpoint |
| `threshold` | Minimum pass score (default `0.5`) |
| `dataset_format` | Input format: `csv`, `jsonl`, or `json` (default `csv`) |

See [adapters/deepeval/README.md](adapters/deepeval/README.md) for full documentation, dataset column requirements, and multi-turn conversation format.

## RULER Adapter

The RULER adapter integrates [NVIDIA RULER](https://github.com/NVIDIA/RULER) (**What's the Real Context Size of Your LLM?**) into eval-hub. RULER is a synthetic long-context benchmark that evaluates LLMs across 13 tasks at configurable context lengths, measuring *effective* context utilisation rather than a simple token-count ceiling.

**13 benchmarks** across four categories:

- **8 Needle-in-a-Haystack** — single needle with noise/essay/UUID haystacks, multi-key, multi-value, multi-query, and needle-background variants.
- **1 Variable Tracking** — track chains of variable assignments and return the final value.
- **2 Aggregation** — identify the most frequent words and the top coded words in a Zipf-distributed list.
- **2 Question Answering** — long-context QA with SQuAD and HotpotQA passages.

**Model configuration** — any OpenAI-compatible inference endpoint. Set `model.url` to the `/v1` endpoint and `model.name` to the model ID.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `context_lengths` | `[4096, 8192, 16384]` | Context sizes (tokens) to evaluate |
| `num_samples` | `10` | Samples per (task × context length) |
| `tokenizer_path` | `model.name` | HuggingFace model ID or tiktoken model for data generation |
| `tokenizer_type` | `hf` | `hf` (HuggingFace) or `openai` (tiktoken) |
| `model_template` | `base` | Chat prompt template (see `scripts/data/template.py`) |
| `random_seed` | `42` | Seed for reproducible data generation |

See [adapters/ruler/README.md](adapters/ruler/README.md) for full documentation, example job specs, and vendored script details.

## WildGuard Adapter

The WildGuard adapter integrates the [WildGuard](https://arxiv.org/abs/2406.18495) safety benchmark (`allenai/wildguard`, MIT licence) from AllenAI. For each prompt+response pair in the dataset, the adapter sends the WildGuard instruction template to the model, parses its natural-language output as `safe` or `unsafe` (outputs containing neither are treated as `unknown`), and compares against the ground-truth label. Unknown predictions are counted as incorrect when calculating accuracy.

**1 benchmark:**

- **`wildguard-safety`** — evaluates the model against the WildGuard test split, reporting accuracy and per-class recall.

**Metrics:**

| Metric | Description |
|---|---|
| `accuracy` | Fraction of examples correctly classified (`overall_score`) |
| `safe_recall` | Recall on genuinely safe responses |
| `unsafe_recall` | Recall on genuinely unsafe responses |

Score guide: 0.5 or below is near random chance; 0.85 or above is strong performance. Low `unsafe_recall` indicates under-refusal; low `safe_recall` indicates over-refusal.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `split` | `test` | HuggingFace dataset split |
| `num_examples` | _(full split)_ | Cap the number of examples (useful for smoke tests) |
| `max_concurrent` | `4` | Concurrent API calls to the model endpoint |
| `request_timeout` | `120` | Per-request timeout in seconds |

See [adapters/wildguard/README.md](adapters/wildguard/README.md) for full documentation and example job specs.

## IFBench Adapter

The IFBench adapter integrates [IFBench](https://github.com/allenai/IFBench) (AllenAI, Apache 2.0) — a benchmark of 58 out-of-domain verifiable instruction constraints. The adapter loads the bundled IFBench test set (299 prompts), generates completions via an OpenAI-compatible endpoint, and scores responses with programmatic constraint checkers from the upstream `ifbench` package.

**1 benchmark:**

- **`ifbench`** — evaluates prompt-level instruction-following accuracy (strict and loose), plus instruction-level scores.

**Metrics:**

| Metric | Description |
|---|---|
| `accuracy` | Prompt-level accuracy for the configured `scoring_mode` (`overall_score`) |
| `prompt_level_strict` | Prompt-level strict accuracy |
| `prompt_level_loose` | Prompt-level loose accuracy (paper default) |
| `inst_level_strict` | Instruction-level strict accuracy |
| `inst_level_loose` | Instruction-level loose accuracy |

Score guide: GPT-4o scores ~34% on this benchmark; pass threshold in curated collections is typically `0.05` (5%).

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `scoring_mode` | `loose` | Primary scoring mode (`loose` or `strict`) |
| `num_examples` | _(full set)_ | Cap the number of prompts (useful for smoke tests) |
| `max_concurrent` | `4` | Concurrent API calls to the model endpoint |
| `max_tokens` | `2048` | Max generation tokens per prompt |
| `temperature` | `0.0` | Sampling temperature (paper uses 0) |
| `request_timeout` | `120` | Per-request timeout in seconds |

See [adapters/ifbench/README.md](adapters/ifbench/README.md) for full documentation and example job specs.

## promptfoo Adapter

The promptfoo adapter integrates [promptfoo](https://github.com/promptfoo/promptfoo) (MIT license) into eval-hub, exposing two evaluation families against any OpenAI-compatible model endpoint.

**2 benchmarks:**

- **`promptfoo-eval`** — assertion-based prompt/model regression testing. Loads prompts and assertions from CSV, JSONL, or JSON; scores each test case against configurable assertion types (`contains`, `llm-rubric`, `javascript`, and more) and reports overall pass rate.
- **`promptfoo-redteam`** — adversarial red-team scanning using promptfoo's plugin catalog (170+ plugins as of promptfoo 0.123.1). Plugins are mapped to the OWASP LLM Top 10 and extend into RAG-specific attacks (`rag-poisoning`, `rag-document-exfiltration`, `rag-source-attribution`), API/access-control attacks (`bola`, `bfla`, `rbac`, `debug-access`), MCP-specific attacks (`mcp`), and industry-vertical compliance packs (financial, medical, insurance, telecom, real estate, pharmacy, e-commerce, coding-agent) that complement rather than duplicate Garak's probe set.

**Three model roles for red-team runs** — the target model (under test), a generation model (produces adversarial inputs), and a grading model (scores whether each attack succeeded) can each use a different endpoint with a different credential.

> **Required: configure a grading model.** Without an explicit `generation_provider`, the grader defaults to a promptfoo-hosted model that is unreachable on a self-hosted or air-gapped deployment. When unreachable, every red-team result silently reports as `pass=false` regardless of the target model's actual behavior — the grading pipeline is broken, not the target. This is a verified behavior, confirmed on a live OpenShift cluster (September 2026): results were 100% "failed" until `generation_provider` was passed as the grader endpoint. Pass the `generation_provider` parameter to route both attack generation and result grading to the same operator-configured model.

**Native `eval.json` always retained** — every completed job persists promptfoo's own `eval.json` through three independent paths so results can be reopened in promptfoo's own viewer (`promptfoo import`) or fed into CI tooling:

1. Always embedded in `JobResults.additional_info["promptfoo_eval_json"]` (size-gated at `PROMPTFOO_EVAL_JSON_MAX_BYTES`)
2. Attached as an MLflow artifact when `experiment_name` is set — verified end-to-end against a running RHOAI MLflow deployment; see [adapters/promptfoo/README.md](adapters/promptfoo/README.md) for required RBAC
3. Attached as an OCI artifact when `config.exports.oci` is set

**Metrics:**

| Metric | Description |
|---|---|
| `pass_rate` | `successes / (successes + failures + errors)` (`overall_score`) |
| `n_evaluated` | Total test cases run |
| `n_passed` | Test cases where all assertions passed |
| `n_failed` | Test cases with a failed assertion |
| `n_errors` | Test cases where the provider call itself failed (not an assertion failure) |

`promptfoo-redteam` additionally reports `pass_rate_by_plugin` and `severity_by_plugin` in `additional_info`.

**Key parameters:**

| Parameter | Benchmark | Default | Description |
|---|---|---|---|
| `plugins` | promptfoo-redteam | OWASP LLM Top 10 subset | Plugin IDs or category aliases to run; use `promptfoo redteam plugins` to list all 170+ |
| `redteam_purpose` | promptfoo-redteam | _(none)_ | Plain-language description of what the application does; used to generate contextually relevant attacks rather than generic ones |
| `num_tests` | promptfoo-redteam | `5` | Number of test cases per plugin |
| `generation_provider` | promptfoo-redteam | promptfoo default (unreachable on self-hosted) | OpenAI-compatible URL for the model that generates adversarial content **and** grades results — set this to avoid silent all-failed results |
| `prompts` / `tests` | promptfoo-eval | _(none)_ | Generate a promptfoo config from these instead of `config_yaml` |
| `config_yaml` | promptfoo-eval | _(none)_ | Pass an existing promptfoo project config verbatim (providers are always overwritten with the EvalHub model endpoint) |
| `max_concurrency` | both | `4` | Maximum concurrent API calls |
| `request_timeout` | both | `120` | Per-request timeout in seconds |
| `experiment_name` | both | _(none)_ | MLflow experiment name; when set, results and `eval.json` are logged as an MLflow run |

See [adapters/promptfoo/README.md](adapters/promptfoo/README.md) for full documentation, verified operational constraints, MLflow RBAC requirements, and Garak overlap analysis.

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
# Build a specific adapter image
make image-lighteval
make image-guidellm
make image-mteb
make image-inspect
make image-deepeval
make image-ragas
make image-swebench
make image-ruler
make image-nemo-guardrails
make image-promptfoo
make image-followbench

# Build all adapter images
make images

# Run adapter tests
make test-lighteval
make test-guidellm
make test-mteb
make test-clear
make test-inspect
make test-deepeval
make test-ragas
make test-swebench
make test-ruler
make test-nemo-guardrails
make test-promptfoo
make tests

# Push to registry
make push-lighteval REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-guidellm REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-mteb REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-inspect REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-deepeval REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-ragas REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-swebench REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-ruler REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-nemo-guardrails REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-promptfoo REGISTRY=quay.io/your-org VERSION=v1.0.0
```

## Publishing a Versioned Framework Image

Merges to `main` automatically publish a `latest` image for adapters changed by the merge. Deployments should use a versioned image tag instead of `latest`.
To publish a versioned image for a framework:

1. Confirm the required adapter change has merged to `main` and that its `main` image build succeeded.
2. Create a `v*` release branch from the intended `main` commit, for example:

   ```bash
   git fetch origin
   git switch main
   git pull --ff-only origin main
   git switch -c v0.5.3
   git push origin v0.5.3
   ```

3. In GitHub Actions, open **Build and push adapter images**. In **Use workflow from**, click **Run workflow** and enter the adapter directory name (for example, `ragas`) in the `adapter` field, and select the release branch from above (for example, `v0.5.3`).
The workflow must run from the release branch for the published tag to be `v0.5.3` rather than `latest`.
4. Wait for both the build and push steps to complete, then verify the resulting `quay.io/evalhub/community-<adapter>:vX.Y.Z` tag in Quay.
5. Update the consuming deployment configuration, such as the TrustyAI Operator image parameter, to the versioned tag (or its immutable digest), and validate an evaluation job using that image.

The workflow detects changed adapters for normal `main` pushes. A newly created release branch normally has no adapter-file diff of its own, so the explicit manual workflow run and adapter selection in step 3 are required to publish the versioned tag.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding adapters.

## License

See the [LICENSE](LICENSE) file for details.
