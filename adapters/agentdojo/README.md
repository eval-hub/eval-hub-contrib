# AgentDojo Adapter

Eval-hub adapter for [AgentDojo](https://github.com/ethz-spylab/agentdojo) — a dynamic environment to evaluate prompt injection attacks and defenses for LLM agents.

## Overview

AgentDojo benchmarks LLM agents across simulated tool-use environments (workspace, travel, banking, slack) and measures:

- **Utility** — Does the agent complete the legitimate user task?
- **Security** — Does the agent resist prompt injection attacks hidden in the environment?

The adapter runs `python -m agentdojo.scripts.benchmark` as a subprocess, parses the per-task JSON results, and converts them into structured `EvaluationResult` metrics.

## Architecture

```
┌─────────────────────────────────────────┐
│              eval-hub                   │
│                                         │
│  JobSpec ──► AgentDojoAdapter           │
│               │                         │
│               ├─ validate config        │
│               ├─ run benchmark (CLI)    │
│               ├─ parse result JSONs     │
│               ├─ extract metrics        │
│               └─ persist OCI artifacts  │
│                                         │
│  JobResults ◄── utility + security      │
└─────────────────────────────────────────┘
```

## Supported Benchmarks

### Suites

| Suite       | Description                          |
|-------------|--------------------------------------|
| `workspace` | Email, calendar, cloud drive tasks   |
| `travel`    | Travel booking and planning tasks    |
| `banking`   | Financial transaction tasks          |
| `slack`     | Slack-like messaging tasks           |

### Attacks

| Attack                                | Description                                        |
|---------------------------------------|----------------------------------------------------|
| `direct`                              | Directly instructs the model to perform injection  |
| `ignore_previous`                     | "Ignore previous instructions" style               |
| `system_prompt`                       | Mimics system prompt to override behavior          |
| `tool_knowledge`                      | Uses knowledge of available tools                  |
| `important_instructions`              | Sophisticated system prompt impersonation          |
| `important_instructions_no_user_name` | Same without user name                             |
| `injecagent`                          | InjecAgent-style attack                            |
| `dos`                                 | Denial-of-service (prevents task completion)       |

### Defenses

| Defense                          | Description                                     |
|----------------------------------|-------------------------------------------------|
| `tool_filter`                    | Filters tools to only relevant ones (OpenAI)    |
| `transformers_pi_detector`       | DeBERTa-based prompt injection detector         |
| `spotlighting_with_delimiting`   | Wraps tool outputs with delimiters              |
| `repeat_user_prompt`             | Re-injects user prompt in execution loop        |

## Supported Request Payload Parameters

| Parameter            | Type           | Default    | Description                                    |
|----------------------|----------------|------------|------------------------------------------------|
| `provider_type`      | `string`       | `"openai-compatible"` | AgentDojo provider type (e.g. `openai-compatible`, `local`, `vllm_parsed`) |
| `attack`             | `string\|null` | `null`     | Attack to use (see table above)                |
| `defense`            | `string\|null` | `null`     | Defense to use (see table above)               |
| `benchmark_version`  | `string`       | `"v1.2.2"` | AgentDojo benchmark version                    |
| `suites`             | `string[]`     | all suites | Which suites to run                            |
| `user_tasks`         | `string[]`     | all        | Specific user tasks (e.g., `["user_task_0"]`)  |
| `injection_tasks`    | `string[]`     | all        | Specific injection tasks                       |
| `system_message`     | `string\|null` | `null`     | Custom system message                          |
| `system_message_name`| `string\|null` | `null`     | Named system message from defaults             |
| `modules_to_load`    | `string[]`     | `[]`       | Custom modules to register attacks/defenses    |
| `timeout_seconds`    | `integer`      | `7200`     | Timeout for benchmark execution                |

## Usage

### Building the Container

```bash
cd adapters/agentdojo
podman build -t agentdojo-adapter:latest -f Containerfile .
```

### Running Locally

```bash
# Set required API keys
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...

# Run with example job spec
EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
```

### Running with Container

```bash
podman run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e EVALHUB_JOB_SPEC_PATH=/meta/job.json \
  -v $(pwd)/meta:/meta:ro \
  agentdojo-adapter:latest
```

## Example Configurations

### Utility-only evaluation (no attack)

```json
{
  "id": "agentdojo-utility-001",
  "benchmark_id": "agentdojo",
  "model": {
    "url": null,
    "name": "gpt-4o-2024-05-13"
  },
  "parameters": {
    "benchmark_version": "v1.2.2"
  },
  "callback_url": "http://localhost:8080"
}
```

### Single suite with attack

```json
{
  "id": "agentdojo-workspace-attack-001",
  "benchmark_id": "workspace",
  "model": {
    "url": null,
    "name": "claude-3-5-sonnet-20241022"
  },
  "parameters": {
    "attack": "important_instructions",
    "benchmark_version": "v1.2.2"
  },
  "callback_url": "http://localhost:8080"
}
```

### Attack + defense evaluation

```json
{
  "id": "agentdojo-defense-eval-001",
  "benchmark_id": "agentdojo",
  "model": {
    "url": null,
    "name": "gpt-4o-2024-05-13"
  },
  "parameters": {
    "attack": "tool_knowledge",
    "defense": "spotlighting_with_delimiting",
    "suites": ["workspace", "banking"],
    "benchmark_version": "v1.2.2"
  },
  "callback_url": "http://localhost:8080"
}
```

### Local model via vLLM endpoint

```json
{
  "id": "agentdojo-local-001",
  "benchmark_id": "workspace",
  "model": {
    "url": "http://localhost:8000/v1",
    "name": "meta-llama/Llama-3.1-8B-Instruct"
  },
  "parameters": {
    "provider_type": "local",
    "attack": "direct",
    "benchmark_version": "v1.2.2"
  },
  "callback_url": "http://localhost:8080"
}
```

## Output Artifacts

The adapter produces the following files in the OCI artifact:

| File                  | Description                                                |
|-----------------------|------------------------------------------------------------|
| `task_results.json`   | Per-task utility/security results (without message traces) |
| `results.json`        | Structured EvaluationResult metrics                        |
| `summary.txt`         | Human-readable summary with aggregate scores               |
| `raw_logs.tar.gz`     | Full AgentDojo trace logs (including message histories)    |

## Environment Variables

| Variable             | Required | Description                              |
|----------------------|----------|------------------------------------------|
| `OPENAI_API_KEY`     | *        | OpenAI API key (for OpenAI models)       |
| `ANTHROPIC_API_KEY`  | *        | Anthropic API key (for Claude models)    |
| `TOGETHER_API_KEY`   | *        | Together API key (for Together models)   |
| `GCP_PROJECT`        | *        | GCP project (for Gemini models)          |
| `GCP_LOCATION`       | *        | GCP location (for Gemini models)         |
| `EVALHUB_MODE`       | No       | `"k8s"` or `"local"` (default: local)   |
| `EVALHUB_JOB_SPEC_PATH` | No   | Override job spec path                   |
| `LOG_LEVEL`          | No       | DEBUG, INFO, WARNING, ERROR              |

\* Required depending on which model provider is used.

## Submitting Jobs via the EvalHub API

Submit evaluation jobs via the EvalHub REST API. See the
[EvalHub docs](https://eval-hub.github.io/) for installation and deployment.

### Workspace suite with prompt injection attack

```bash
TOKEN=$(oc whoami -t)
EVALHUB_URL=$(oc get route evalhub -n eval-hub -o jsonpath='{.spec.host}')

curl -sk -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant: eval-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "agentdojo-workspace-attack",
    "model": {
      "url": "https://your-openai-compatible-endpoint/v1",
      "name": "your-model-id",
      "auth": {
        "secret_ref": "agentdojo-api-keys"
      }
    },
    "benchmarks": [
      {
        "id": "workspace",
        "provider_id": "agentdojo",
        "parameters": {
          "attack": "important_instructions"
        }
      }
    ]
  }' \
  "https://$EVALHUB_URL/api/v1/evaluations/jobs"
```

### All suites, no attack (utility only)

```bash
curl -sk -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant: eval-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "agentdojo-full-utility",
    "model": {
      "url": "https://your-openai-compatible-endpoint/v1",
      "name": "your-model-id"
    },
    "benchmarks": [
      {
        "id": "agentdojo",
        "provider_id": "agentdojo"
      }
    ]
  }' \
  "https://$EVALHUB_URL/api/v1/evaluations/jobs"
```

### Attack + defense evaluation

```bash
curl -sk -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant: eval-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "agentdojo-attack-defense",
    "model": {
      "url": "https://your-openai-compatible-endpoint/v1",
      "name": "your-model-id",
      "auth": {
        "secret_ref": "agentdojo-api-keys"
      }
    },
    "benchmarks": [
      {
        "id": "workspace",
        "provider_id": "agentdojo",
        "parameters": {
          "attack": "tool_knowledge",
          "defense": "spotlighting_with_delimiting"
        }
      }
    ]
  }' \
  "https://$EVALHUB_URL/api/v1/evaluations/jobs"
```

### Check job status

```bash
JOB_ID="<job-id-from-response>"
curl -sk -H "Authorization: Bearer $TOKEN" -H "X-Tenant: eval-hub" \
  "https://$EVALHUB_URL/api/v1/evaluations/jobs/$JOB_ID"
```

The `model.auth.secret_ref` should reference a k8s Secret in the same
namespace containing keys like `OPENAI_COMPATIBLE_API_KEY`. The adapter
reads these via the SDK's `read_model_auth_key()` helper.

## References

- [AgentDojo Paper](https://arxiv.org/abs/2406.13352)
- [AgentDojo GitHub](https://github.com/ethz-spylab/agentdojo)
- [AgentDojo Documentation](https://agentdojo.spylab.ai/)
- [Results Explorer](https://agentdojo.spylab.ai/results/)
