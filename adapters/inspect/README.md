# Inspect AI Adapter

Wraps the [UK AISI Inspect AI](https://inspect.aisi.org.uk/) evaluation framework as an
EvalHub provider. Supports standard `inspect-evals` benchmarks, custom Inspect
tasks, and Petri/Bloom auditing workflows.

The auditing tools and scaffold integration include:

- **Petri** (`inspect-petri`) — auditor/target/judge pipeline; 170+ seeds across 40
  alignment-behavior tag categories; 38 judge dimensions scored 1–10.
- **Bloom** (`petri-bloom`) — generates evaluation scenarios from high-level behavior
  descriptions; multi-step: `bloom init` → `bloom scenarios` → `inspect eval`.
- **Dish** — research-preview feature for real agent deployment scaffold testing;
  exposed via `task_args` pass-through (see parameters).

## Benchmarks

The contrib `provider.yaml` registers **79 benchmark IDs**: 36 Petri alignment
audits, 2 Bloom suites, 36 inspect-evals tasks, 4 Open-Telco tasks (TeleMath,
TeleQnA, TeleLogs, 3GPP-TSG), and 1 custom task.

The adapter's task maps support **83 IDs**, including four additional IDs absent
from this provider file: `inspect/hle`, `inspect/bfcl`, `inspect/docvqa`, and
`inspect/cyberseceval-2-pi`. To expose these through EvalHub, the deployed provider
configuration must register them.

### Petri alignment audits (`inspect/petri-*`)

The adapter exposes 35 named Petri tag selections and one full-audit benchmark.
Primary metric: `concerning/mean` (1–10 scale; higher = more concerning).
All 38 Petri judge dimensions are captured as individual metrics.

| Category | Benchmark IDs |
|---|---|
| Deception & Dishonesty | `petri-sycophancy`, `petri-deception`, `petri-alignment-faking`, `petri-lie-detector` |
| Harmful Outputs | `petri-jailbreak`, `petri-harmful-cooperation`, `petri-harm`, `petri-tool-abuse`, `petri-blackmail`, `petri-manipulation`, `petri-escalation` |
| Information Integrity | `petri-leaking`, `petri-privacy-violation`, `petri-censorship`, `petri-delusional-user` |
| Behavioral Pressure | `petri-bias`, `petri-self-bias`, `petri-blame`, `petri-praise`, `petri-criticism` |
| Self-Preservation & Power | `petri-self-preservation`, `petri-power-seeking`, `petri-oversight-subversion`, `petri-shutdown-avoidance`, `petri-self-exfiltration`, `petri-goal-preservation`, `petri-initiative`, `petri-whistleblowing` |
| Conflict & Goals | `petri-goal-conflict`, `petri-debate`, `petri-reward-hacking` |
| Multi-Agent & Monitoring | `petri-monitoring`, `petri-multi-agent` |
| Exploratory | `petri-openended`, `petri-weird-ood` |
| Full Audit | `petri-full` (all 170+ seeds) |

### Bloom behavioral suites (`inspect/bloom-*`)

| Benchmark ID | Description |
|---|---|
| `inspect/bloom-sycophancy` | Auto-generated sycophancy scenarios (no manual seed authoring) |
| `inspect/bloom-custom` | User-provided behavior directory or template |

### Safety & alignment (`inspect-evals`)

| Benchmark ID | Description |
|---|---|
| `inspect/agentharm` | Agent harmfulness across real-world tool-use scenarios |
| `inspect/agentic-misalignment` | Scheming — deceptive reasoning, goal preservation, covert actions |
| `inspect/gdm-self-proliferation` | Self-copying and spreading to external systems |
| `inspect/gdm-stealth` | Concealing actions from oversight |
| `inspect/gdm-self-reasoning` | Self-aware reasoning and acting on own interests |
| `inspect/strong-reject` | Refusal quality — tests both over- and under-refusal |
| `inspect/wmdp` | Weapons of mass destruction technical uplift prevention |
| `inspect/mask` | Concealing true beliefs under social pressure |
| `inspect/makemesay` | Prompt injection — being manipulated into saying target phrases |
| `inspect/make-me-pay` | Social engineering into transferring resources |
| `inspect/sycophancy-evals` | Systematic sycophancy across opinion/fact/feedback |
| `inspect/instrumental-eval` | Instrumental convergent behaviors (resource acquisition, self-continuity) |
| `inspect/sad` | Self-awareness diagnostic |

### Cybersecurity

| Benchmark ID | Description |
|---|---|
| `inspect/cybench` | CTF challenges — offensive security reasoning |
| `inspect/cyberseceval-2` | Prompt injection, insecure code, cyberattack uplift |
| `inspect/cybergym` | Realistic attack and defense scenarios |

### Coding

`inspect/humaneval` · `inspect/swe-bench` · `inspect/bigcodebench` · `inspect/mbpp`

### Mathematics

`inspect/gsm8k` · `inspect/math` · `inspect/aime2024` · `inspect/aime2025`

### Telecom (GSMA Open-Telco)

| Benchmark ID | Description |
|---|---|
| `telemath` | Telecom numerical math reasoning. Set `full=true` for `GSMA/ot-full` (500 Q&A); cap with `num_examples`. |
| `teleqna` | Telecom multiple-choice domain knowledge. Set `full=true`; optional `subject` (default `full`). |
| `telelogs` | 5G root-cause analysis. Set `full=true`; `eval_type` soft (default) or hard. |
| `3gpp-tsg` | 3GPP working-group classification. Set `full=true`; cap with `num_examples`. |

### Knowledge & Reasoning

`inspect/mmlu` · `inspect/mmlu-pro` · `inspect/gpqa` · `inspect/bbh` · `inspect/arc` · `inspect/hellaswag` · `inspect/winogrande` · `inspect/truthfulqa` · `inspect/simpleqa`

### Agent Capabilities

`inspect/gaia` · `inspect/agentdojo` · `inspect/theagentcompany`

### Custom

`inspect/custom` — run any Inspect AI task by setting the `task` parameter.

---

## Kubernetes and container notes

### Sandbox

Standard inspect-evals benchmarks default to the `local` sandbox — code runs directly
in the adapter container without Docker. This is the only sandbox available in
Kubernetes pods. Override with `parameters.sandbox` if you have a different provider
configured (e.g. `"docker"` for local development with Docker Engine).

```json
{ "parameters": { "sandbox": "docker" } }
```

Petri and Bloom modes do not use a sandbox.

### HuggingFace datasets

Some inspect-evals benchmarks (e.g. `humaneval`, `mmlu`) and Open-Telco tasks
(`telemath`) load datasets from the HuggingFace Hub unless offline data is staged.

On disconnected clusters, configure **`test_data_ref.s3`** (or PVC/git) on the
benchmark so Eval Hub syncs a Hugging Face cache layout into **`/test_data`**.
The adapter then sets **`HF_HOME`**, **`HF_HUB_OFFLINE`**, and related env vars
for the `inspect eval` subprocess (same approach as the lm-evaluation-harness
adapter). Disconnected FVT jobs also set **`parameters.tokenizer`** to
`/test_data/tokenizer` alongside **`test_data_ref`**; either signal enables
offline mode when `/test_data` is populated.

When online, the adapter reads an
`hf-token` secret mounted at `/var/run/secrets/model/hf-token` and injects it as
`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`. Sidecar `:ref` placeholders are ignored.
In EvalHub jobs, set `model.auth.secret_ref` to a Kubernetes Secret that includes the
`hf-token` key (alongside `api-key` if needed).

The mount is a projected volume that Kubernetes populates before the container starts,
so the adapter does not wait for it when it is absent (no `model.auth.secret_ref`) or
already populated. Only an empty mount gets a 5-second grace period. Set
`INSPECT_HF_TOKEN_WAIT_S` (seconds; `0` disables) to override.

### Sample limits

Inspect `--limit` is driven by `benchmarks[].parameters.num_examples` (lifted to
JobSpec `num_examples` by eval-hub), the same parameter every other contrib adapter
uses. When unset, standard inspect-evals benchmarks run the **full dataset**.
Petri and Bloom default to `--limit 5` because their full seed sets are very
expensive; set `num_examples` to raise or lower the cap.

`parameters.max_samples` is a **deprecated alias** for `num_examples` and logs a
warning; `num_examples` wins if both are set. Note that Inspect's own
`--max-samples` flag controls how many samples run *in parallel*, not how many
are evaluated, which is why the name is being retired here.

Open-Telco dataset size is controlled via `parameters.full`
(`true` → `GSMA/ot-full`, `false` → `GSMA/ot-lite`). TeleQnA also accepts
`parameters.subject` (default in the task is `full` = all subjects). TeleLogs
accepts `parameters.eval_type` (`soft` or `hard`).

---

## Model and credential configuration

The adapter detects which API to use from environment variables. Model names are passed
as-is — bare (`claude-opus-4-7`, `granite3.3`) or org/model
(`ibm-granite/granite-3.3-8b-instruct`, `meta-llama/Llama-3.3-70B-Instruct`).

### Global credentials (apply to all roles by default)

| Env var | Used for |
|---|---|
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint (vLLM, Ollama `/v1`, OpenRouter) |
| `OPENAI_API_KEY` | API key for `OPENAI_BASE_URL` endpoint |
| `ANTHROPIC_API_KEY` | Anthropic Messages API |
| `ANTHROPIC_BASE_URL` | Anthropic API base URL override (proxies, on-prem) |

Global client selection priority for roles without explicit routing:
1. `model.url` present → OpenAI-compatible client
2. `ANTHROPIC_API_KEY` or `ANTHROPIC_BASE_URL` set → Anthropic client
3. `OPENAI_BASE_URL` or `OPENAI_API_KEY` set → OpenAI-compatible client

### Independent StrongREJECT grader

For `inspect/strong-reject`, set `parameters.grader_model` to route scoring to
Inspect's named `grader` role instead of StrongREJECT's built-in `judge_llm`
default. For example, use `gpt-4o-mini` with `grader_base_url` set to
`https://api.openai.com/v1`. The adapter routes that role through Inspect's
isolated `openai-api/openai_judge/...` provider namespace, so the target keeps
its own `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

The judge credential must be provided to the adapter process as
`OPENAI_JUDGE_API_KEY` from a Kubernetes Secret. The adapter does not accept a
judge key in evaluation parameters, and it never places the key in the Inspect
command line. `OPENAI_JUDGE_BASE_URL` can supply the endpoint instead of
`grader_base_url`; it defaults to `https://api.openai.com/v1`.

This adapter support assumes the EvalHub runtime injects that Secret-backed
environment variable into the job. If the job currently mounts only the
target model Secret, a corresponding EvalHub core change is required before
this route can be exercised in-cluster.

### Per-role credential overrides

Each role (target, auditor, judge, scenarios, realism) accepts its own endpoint and key.
When set, only that role uses the override; all other roles continue using global credentials.

| Parameter | Effect |
|---|---|
| `{role}_base_url` | OpenAI-compatible endpoint for this role only |
| `{role}_api_key` | API key for the OpenAI-compatible endpoint |
| `{role}_anthropic_base_url` | Anthropic endpoint for this role only |
| `{role}_anthropic_api_key` | Anthropic API key for this role |

---

## Deployment examples

### Scenario 1 — All roles on the same vLLM, no authentication

```json
{
  "model": {
    "url": "http://vllm:8080/v1",
    "name": "ibm-granite/granite-3.3-8b-instruct"
  },
  "parameters": {
    "auditor_model": "ibm-granite/granite-3.3-8b-instruct",
    "judge_model": "meta-llama/Llama-3.3-70B-Instruct"
  }
}
```
Environment: none required (vLLM does not require authentication by default).

---

### Scenario 2 — Target on vLLM, auditor/judge on Anthropic

```json
{
  "model": {
    "url": "http://vllm:8080/v1",
    "name": "ibm-granite/granite-3.3-8b-instruct"
  },
  "parameters": {
    "auditor_model": "claude-sonnet-4-6",
    "judge_model": "claude-opus-4-7"
  }
}
```
Environment: `ANTHROPIC_API_KEY=sk-ant-...`

The adapter routes the target to the OpenAI-compatible client (via `model.url`) and the
auditor/judge to Anthropic (via `ANTHROPIC_API_KEY`).

---

### Scenario 3 — Target on vLLM-A, judge on a different vLLM-B

Each vLLM instance is configured with an `EMPTY` placeholder key (set this when the
server requires a token even if authentication is not enforced).

```json
{
  "model": {
    "url": "http://vllm-a:8080/v1",
    "name": "ibm-granite/granite-3.3-8b-instruct"
  },
  "parameters": {
    "auditor_model": "ibm-granite/granite-3.3-8b-instruct",
    "judge_model": "meta-llama/Llama-3.3-70B-Instruct",
    "judge_base_url": "http://vllm-b:8080/v1",
    "judge_api_key": "EMPTY"
  }
}
```
Environment: `OPENAI_API_KEY=EMPTY` (for target and auditor on vLLM-A).

---

### Scenario 4 — Target on vLLM, auditor on OpenRouter, judge on Anthropic

```json
{
  "model": {
    "url": "http://vllm:8080/v1",
    "name": "ibm-granite/granite-3.3-8b-instruct"
  },
  "parameters": {
    "auditor_model": "meta-llama/llama-3.3-70b-instruct",
    "auditor_base_url": "https://openrouter.ai/api/v1",
    "auditor_api_key": "sk-or-...",
    "judge_model": "claude-opus-4-7"
  }
}
```
Environment: `ANTHROPIC_API_KEY=sk-ant-...`

Each role uses a completely different provider and endpoint.

---

### Scenario 5 — All roles on Ollama, no authentication

Ollama exposes an OpenAI-compatible API at `/v1`. Model names follow the Ollama library
format (`granite3.3:8b`, `llama3.3`, `qwen3:32b`), not HuggingFace IDs.

```json
{
  "model": {
    "url": "http://ollama:11434/v1",
    "name": "granite3.3:8b"
  },
  "parameters": {
    "auditor_model": "llama3.3",
    "judge_model": "qwen3:32b"
  }
}
```
Environment: none required.

---

### Scenario 6 — Anthropic for all roles

```json
{
  "model": {
    "name": "claude-haiku-4-5-20251001"
  },
  "parameters": {
    "auditor_model": "claude-sonnet-4-6",
    "judge_model": "claude-opus-4-7"
  }
}
```
Environment: `ANTHROPIC_API_KEY=sk-ant-...`

No `model.url` needed. All roles resolve to Anthropic via `ANTHROPIC_API_KEY`.

---

### Scenario 7 — Judge on a custom Anthropic proxy

Use `judge_anthropic_base_url` to route only the judge to a non-default Anthropic
endpoint while everything else uses the standard configuration.

```json
{
  "model": {
    "url": "http://vllm:8080/v1",
    "name": "ibm-granite/granite-3.3-8b-instruct"
  },
  "parameters": {
    "auditor_model": "claude-sonnet-4-6",
    "judge_model": "claude-opus-4-7",
    "judge_anthropic_base_url": "https://my-anthropic-proxy/v1",
    "judge_anthropic_api_key": "sk-proxy-key"
  }
}
```
Environment: `ANTHROPIC_API_KEY=sk-ant-...` (for auditor), `OPENAI_BASE_URL` set from
`model.url` (for target).

---

## Key Petri parameters

| Parameter | Default | Description |
|---|---|---|
| `auditor_model` | `claude-sonnet-4-6` | Model that drives adversarial conversations |
| `judge_model` | `claude-opus-4-7` | Model that scores transcripts (use strongest available) |
| `max_turns` | `30` | Max auditor turns per scenario |
| `enable_rollback` | `true` | Allow auditor to backtrack and retry approaches |
| `realism_filter` | `false` | Filter unrealistic auditor outputs (experimental) |
| `num_examples` | `5` (Petri/Bloom) | Cap scenarios/samples via EvalHub `benchmarks[].parameters.num_examples` (JobSpec `num_examples` → Inspect `--limit`; Petri/Bloom default to 5 when unset, standard benchmarks are unbounded) |
| `seed_instructions` | *(from benchmark_id)* | Override seed selection (`tags:deception`, `id:seed_name`, inline text) |
| `judge_dimensions` | *(all 38)* | Filter judge dimensions (`tags:safety` or custom directory) |
| `task_args` | `{}` | Escape hatch for non–first-class Inspect `-T` flags (e.g. Dish `dish_scaffold`). Not for Open-Telco `full`. |

## Bloom-specific parameters

| Parameter | Default | Description |
|---|---|---|
| `bloom_template` | `null` | Template for `bloom init --from <template>` (e.g. `delusion_sycophancy`) |
| `behavior_dir` | `null` | Pre-built behavior directory — skips `bloom init` and `bloom scenarios` steps |
| `scenarios_model` | *(auditor_model)* | Model for the `bloom scenarios` generation step |

> **Note:** The `bloom scenarios` CLI only accepts bare `client/model` strings for the
> scenarios role (e.g. `openai/gpt-oss-20b`). JSON model specs with `model_args` are not
> supported at this step. The scenarios model uses the global `OPENAI_BASE_URL` / `OPENAI_API_KEY`
> credentials; per-role endpoint overrides do not apply to the scenarios step.

---

## Results and scoring

Individual benchmark metrics are returned in `results`. The adapter also computes
`overall_score` as a summary. EvalHub selects the metric used for pass/fail through
its `primary_score.metric` configuration; that selection is separate from the
adapter's summary calculation.

### HLE and BFCL representative scores

All individual metrics remain available. For `inspect/hle`, `overall_score`
uses `hle/regex_judge/hle/accuracy`. For `inspect/bfcl`, it uses
`bfcl_scorer/accuracy`. Calibration error, unscored counts and category metrics
are not averaged into these representative scores. If the selected accuracy is
missing, duplicated or non-finite, no overall score is returned.

### Other benchmark summaries

Petri and Bloom select `concerning/mean` when available. Other benchmarks retain
the existing mean calculation, excluding dispersion metrics and category-specific
standard errors ending in `_stderr` or `_sterr`. These metrics remain available
individually.

### Prompting metadata

For standard-mode tasks, `additional_info.zero_shot` mirrors `overall_score`
only when the saved Inspect log's `eval.task_args` confirms zero shots using a
recognized shot-count or example-list argument. It is omitted for few-shot
settings and when the setting is missing or unrecognized. An omitted
`zero_shot` does not imply that the task used few-shot prompting; its prompting
setup may simply be unconfirmed. Selecting a representative score does not
determine the prompting setup. Petri and Bloom retain `alt_prompting` metadata.

---

## Building and testing

```bash
# Build container image
make image-inspect

# Run adapter tests
make test-inspect

# Push to registry
make push-inspect REGISTRY=quay.io/your-org VERSION=v1.0.0
```

## Requirements

Direct dependencies are pinned to exact versions in `requirements.txt`
(`inspect-ai`, `inspect-evals`, `inspect-petri`, `petri-bloom`, `openai`,
`nltk`; `eval-hub-sdk[adapter]` is a compatible-release pin). `constraints.txt`
pins the full transitive tree, and the Containerfile installs with both, so an
image rebuild reproduces the same dependency set.

### Updating dependencies

Bump the pins in `requirements.txt`, then regenerate the lock and audit it. Run
from `adapters/inspect/`:

```bash
# 1. Regenerate the transitive pins (platform-independent; git lines are
#    excluded because pip does not accept URLs in constraints files)
{ echo "# Transitive dependency pins for the Inspect adapter image, applied with: pip install -r requirements.txt -c constraints.txt"
  echo "# Generated; do not edit by hand. Regenerate (and re-audit) with the command in README.md → Updating dependencies."
  uv pip compile requirements.txt --universal --python-version 3.12 --no-header --no-annotate | grep -v "@ git"
} > constraints.txt

# 2. Audit every pinned version for known vulnerabilities
grep -E '^[A-Za-z0-9_.-]+==' constraints.txt | sed -E 's/ *;.*//' | sort -u > /tmp/inspect-pins.txt
pip-audit -r /tmp/inspect-pins.txt --no-deps --disable-pip
```

The repository's Trivy filesystem scan only sees versions that are pinned, which
is why the pins matter: with open-ended `>=` ranges it cannot see what actually
lands in the image. Packages installed straight from git (`instruction_following_eval`,
`evals`) are not on PyPI and cannot be audited by name; they are pinned to a
commit or tag.

Known advisory: `nltk` 3.10.3 carries GHSA-8mgp-746c-j5xp (CVE-2026-81726) with no
patched release yet. It affects nltk's parser and perceptron model save/load APIs,
which nothing in this image calls. Revisit when a fixed nltk is published.
