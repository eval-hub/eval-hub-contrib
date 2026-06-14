# Inspect AI Adapter

Wraps the [UK AISI Inspect AI](https://inspect.aisi.org.uk/) evaluation framework as an
EvalHub provider, exposing three Meridian Labs alignment-auditing tools:

- **Petri** (`inspect-petri`) — auditor/target/judge pipeline; 170+ seeds across 40
  alignment-behavior tag categories; 38 judge dimensions scored 1–10.
- **Bloom** (`petri-bloom`) — generates evaluation scenarios from high-level behavior
  descriptions; multi-step: `bloom init` → `bloom scenarios` → `inspect eval`.
- **Dish** — research-preview feature for real agent deployment scaffold testing;
  exposed via `task_args` pass-through (see parameters).

Standard inspect-evals benchmarks (gsm8k, mmlu, humaneval, swe-bench, custom tasks)
are also supported.

---

## Benchmarks

**75 benchmarks total** — 36 Petri alignment audits, 2 Bloom suites, 37 inspect-evals + custom.

### Petri alignment audits (`inspect/petri-*`)

All 40 built-in Petri seed tag categories are exposed as individual named benchmarks.
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

### Knowledge & Reasoning

`inspect/mmlu` · `inspect/mmlu-pro` · `inspect/gpqa` · `inspect/bbh` · `inspect/arc` · `inspect/hellaswag` · `inspect/winogrande` · `inspect/truthfulqa` · `inspect/simpleqa`

### Agent Capabilities

`inspect/gaia` · `inspect/agentdojo` · `inspect/theagentcompany`

### Custom

`inspect/custom` — run any Inspect AI task by setting the `task` parameter.

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

Client selection priority per role (no per-role override):
1. `model.url` present → OpenAI-compatible client
2. `ANTHROPIC_API_KEY` or `ANTHROPIC_BASE_URL` set → Anthropic client
3. `OPENAI_BASE_URL` or `OPENAI_API_KEY` set → OpenAI-compatible client

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
    "judge_model": "meta-llama/Llama-3.3-70B-Instruct",
    "max_samples": 5
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
    "judge_model": "claude-opus-4-7",
    "max_samples": 5
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
    "judge_api_key": "EMPTY",
    "max_samples": 5
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
    "judge_model": "claude-opus-4-7",
    "max_samples": 5
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
    "judge_model": "qwen3:32b",
    "max_samples": 5
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
    "judge_model": "claude-opus-4-7",
    "max_samples": 5
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
    "judge_anthropic_api_key": "sk-proxy-key",
    "max_samples": 5
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
| `max_samples` | `null` | Limit scenarios per run — recommended for cost control |
| `seed_instructions` | *(from benchmark_id)* | Override seed selection (`tags:deception`, `id:seed_name`, inline text) |
| `judge_dimensions` | *(all 38)* | Filter judge dimensions (`tags:safety` or custom directory) |
| `task_args` | `{}` | Pass-through to `inspect eval` (`{"dish_scaffold": "claude-code"}` enables Dish) |

## Bloom-specific parameters

| Parameter | Default | Description |
|---|---|---|
| `bloom_template` | `null` | Template for `bloom init --from <template>` (e.g. `delusion_sycophancy`) |
| `behavior_dir` | `null` | Pre-built behavior directory — skips `bloom init` and `bloom scenarios` steps |
| `scenarios_model` | *(auditor_model)* | Model for the `bloom scenarios` generation step |

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

- `inspect-ai >= 0.3.40`
- `inspect-evals >= 0.1.0`
- `inspect-petri >= 3.0.0`
- `petri-bloom >= 0.1.0`
- `eval-hub-sdk[adapter] >= 0.1.7`
