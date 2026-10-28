# Model-role authentication

Additional model connections belong in the job's `model.auth.secret_ref`, alongside
the primary model credential. Model IDs remain benchmark parameters. Adapters
do not use former inline role URL/key parameters or auxiliary runtime environment
variables to select connections. Legacy connection parameters are ignored, whether
or not role entries exist in the Secret. Their presence alone never fails a job,
and their values never override or fill in missing Secret values. This applies to
local and Kubernetes jobs. Existing target-only adapter authentication is outside
this change.

| Adapter | Secret role prefixes | Model selection |
| --- | --- | --- |
| Inspect Petri/Bloom | `auditor`, `judge`, `scenarios`, `realism` | `<role>_model` |
| Inspect HLE/CyberSecEval | `judge` | `judge_model` |
| Inspect StrongREJECT | `judge` | `grader_model` |
| Inspect named roles | the role name, including `grader` | `model_roles: {role: model_id}` |
| DeepEval | `judge` | `eval_model_name` |
| FollowBench, ToolEval | `judge` | `judge_model`, default `model.name` |
| CLEAR | `judge` | `eval_model_name`, `provider` |
| ATIF | `judge` | `model.name` |
| Promptfoo generation | `generation_provider` | `generation_provider`, default target provider ID |
| Promptfoo grading | `judge` | `judge_model`, default generation provider ID |
| Ragas (existing #243) | `judge`, `embedding` | `judge_model`, `embedding_model` |

Ragas retains #243's existing standalone override behavior; this extension changes
the other adapters listed above. Dataset, Hugging Face, tool-server, MLflow, and
artifact-storage credentials are separate concerns.

## Secret and request example

The following values are placeholders, not credentials. The primary upstream URL
is still the API request's `model.url`.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: evaluation-model-auth
type: Opaque
stringData:
  api-key: <target-credential>
  judge_api-key: <judge-credential>
  judge_url: https://judge.example/v1
  auditor_api-key: <auditor-credential>
  auditor_url: https://auditor.example/v1
  scenarios_api-key: <scenario-credential>
  scenarios_url: https://scenario.example/v1
  realism_api-key: <realism-credential>
  realism_url: https://realism.example/v1
  generation_provider_api-key: <generation-credential>
  generation_provider_url: https://generation.example/v1
```

```json
{
  "name": "petri-with-independent-models",
  "model": {
    "name": "target-model",
    "url": "https://target.example/v1",
    "auth": {"secret_ref": "evaluation-model-auth"}
  },
  "benchmarks": [{
    "provider_id": "inspect",
    "id": "inspect/petri-sycophancy",
    "parameters": {"auditor_model": "auditor-model", "judge_model": "judge-model", "num_examples": 1}
  }]
}
```

With neither role entry, the role uses the primary URL and the Secret's `api-key`.
A role key without its URL uses the primary URL with that different credential.
A role URL without its key is an error. To reuse a real credential on a different
URL, copy it into the role's key; do not rely on URL equality to choose credentials.
Kubernetes projects the keys as `<role>_api-key:ref` and rewrites URLs to the
sidecar, preserving their paths. Equal localhost URLs therefore still select
different upstreams through their reference tokens. Start a new job after changing
the Secret, since the sidecar caches its credentials at startup.

Local jobs use an absolute `file:///` auth-directory reference with the same key
filenames. There is no fallback to `FOLLOWBENCH_JUDGE_API_KEY`,
`TOOLEVAL_JUDGE_API_KEY`, `OPENAI_JUDGE_API_KEY`, `GENERATION_PROVIDER_API_KEY`,
or `EVALHUB_JUDGE_PROXY_URL`. Framework subprocess environment values are outputs
of Secret resolution, not additional configuration inputs.

## Sidecar support and deferred changes

The sidecar code is unchanged by this work. Its model proxy is a catch-all after
EvalHub, MLflow, and OCI routing. It resolves arbitrary `*_api-key:ref` Bearer
tokens, chooses the matching `*_url` host, replaces the reference with the real
credential, and preserves the incoming path and query. No role-specific path
registration is needed for OpenAI-compatible chat, completions, embeddings,
models, or Responses API calls. The upstream must implement the requested API.

Inspect's pinned `inspect-ai==0.3.278` Ollama provider subclasses its
OpenAI-compatible provider and accepts an explicit API key. It uses `/v1` APIs
with a Bearer credential, so it fits the existing proxy contract. This does not
claim support for every third-party native Ollama client.

The following sidecar work requires a separate decision; none is implemented here:

1. **Anthropic API-key authentication.** Recognize reference tokens carried in
   `x-api-key`, resolve the role and upstream, and replace that header with the
   real key. Today only `Authorization: Bearer` drives reference resolution;
   an `x-api-key` reference would otherwise pass through unresolved and select
   the primary upstream. Avoid adding a conflicting Bearer header to an
   API-key-authenticated Anthropic request. Define precedence if both headers
   are supplied, and reject missing references without forwarding.
2. **Protocol-specific regression coverage.** Exercise `/v1/messages`,
   `/v1/messages/count_tokens`, streaming, and header isolation against distinct
   authenticated upstreams before enabling native Anthropic role routing.
   Paths already reach the model catch-all; the missing piece is authentication,
   not a `/v1/messages` route. Add the corresponding tests with the deferred fix.
3. **CLEAR's other native providers.** RITS uses `RITS_API_KEY` authentication,
   and Watsonx exchanges an API key with IBM IAM before model calls. Supporting
   these requires an explicit upstream authentication mode and sidecar-owned
   credential/token handling; sending a Bearer reference as a provider API key
   is insufficient. CLEAR currently requires `provider: openai`, including for
   its legacy HTTP backend. An OpenAI-compatible gateway remains usable.

Native `anthropic/` role selections currently fail explicitly in Inspect instead
of sending an unresolved credential. Use an OpenAI-compatible gateway for those
models until the deferred support is available. Authenticated native providers
with other conventions (for example Azure `api-key`, Gemini `x-goog-api-key`, or
AWS signing) need their own design before enabling them through this contract;
this change does not advertise them as working routes.

Primary sources: [Inspect provider configuration](https://inspect.aisi.org.uk/providers.html),
[Claude token-count API](https://platform.claude.com/docs/en/api/http/messages/count_tokens).
The pinned Inspect package source, rather than the current documentation alone,
was checked for role parsing and Ollama's request transport.

## Migration

Move inline URLs/keys (including DeepEval's `eval_model_url`, CLEAR's
`endpoint_url`/`inference_url`, Inspect's role `*_base_url`/`*_api_key`, and
Promptfoo's generation connection parameters) to the role entries above.
Old requests may retain connection parameters: adapters ignore them and use only
the Secret connection or the primary fallback described above. Connection fields
in framework model-option dictionaries are removed before invoking the framework;
unrelated generation settings are retained. Invalid Secret configurations, such as
a role URL without its matching role key, still fail with an actionable error.
Remove runtime auxiliary credential/URL variables and keep only model selection
in benchmark parameters. Inspect also accepts legacy model-role dictionaries with
a `model` name, ignoring their connection settings. Remove credentials from
request parameters and fixture generators even though they are ignored, since
submitted parameters may still be persisted in job metadata.
