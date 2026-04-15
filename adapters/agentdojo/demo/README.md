# AgentDojo on EvalHub - OpenShift Demo

End-to-end deployment of the AgentDojo prompt injection benchmark adapter
on OpenShift with EvalHub deployed directly (no operator required).

## Overview

This demo deploys:

1. **EvalHub service** - evaluation orchestration service with REST API
2. **AgentDojo adapter** - container that runs AgentDojo benchmarks as EvalHub jobs

The adapter evaluates LLM agents on **utility** (task completion) and
**security** (resistance to prompt injection) across four simulated
environments: workspace, slack, banking, and travel.

## Prerequisites

- OpenShift cluster with `oc` CLI logged in (cluster-admin)
- `podman` (or `docker`) for building the adapter image
- An OpenAI-compatible model endpoint (e.g. vLLM, LiteLLM proxy)

## Quick Start

All commands assume you are in this directory (`adapters/agentdojo/demo/`).

### 1. Create the namespace

```bash
oc apply -f 00-namespace.yaml
```

### 2. Create the EvalHub configuration

```bash
oc apply -f 01-evalhub-config.yaml
```

This creates a ConfigMap with the EvalHub service configuration
(SQLite in-memory database, auth disabled for testing). The config
includes sidecar settings that tell job pods how to call back to
EvalHub via the in-cluster service URL.

### 3. Register the AgentDojo provider

```bash
oc apply -f 02-agentdojo-provider.yaml
```

This creates a ConfigMap with the provider definition that EvalHub uses
to discover and run AgentDojo benchmarks.

### 4. Deploy EvalHub and configure RBAC

```bash
oc apply -f 03-evalhub.yaml
oc apply -f 03a-rbac.yaml
```

This creates the EvalHub Deployment, Service, and Route. The provider
ConfigMap is mounted via a projected volume (the same mechanism the
TrustyAI Service Operator uses). The RBAC binding grants the default
ServiceAccount permissions to create ConfigMaps, Pods, and Secrets
needed for evaluation job runs.

Wait for it to become ready:

```bash
oc rollout status deployment/evalhub -n eval-hub --timeout=120s
```

### 5. Expose the internal registry and build the adapter image

The adapter image must be available in the OpenShift internal registry.
First, expose the registry route:

```bash
oc patch configs.imageregistry.operator.openshift.io/cluster \
  --type=merge -p '{"spec":{"defaultRoute":true}}'
```

Then log in and build/push using the repo Makefile:

```bash
REGISTRY_HOST=$(oc get route default-route -n openshift-image-registry \
  -o jsonpath='{.spec.host}')

podman login "$REGISTRY_HOST" \
  -u "$(oc whoami)" -p "$(oc whoami -t)" --tls-verify=false

# From the repo root (eval-hub-contrib/):
make image-agentdojo REGISTRY="$REGISTRY_HOST/eval-hub"
podman push "$REGISTRY_HOST/eval-hub/community-agentdojo:latest" \
  --tls-verify=false
```

Verify the image stream was created:

```bash
oc get is -n eval-hub
# NAME                  TAGS     UPDATED
# community-agentdojo   latest   ...
```

### 6. Create API key secret

If your model endpoint requires authentication:

```bash
# Create a .env file with your endpoint details:
#   LITELLM_API_KEY=sk-your-key
#   LITELLM_API_URL=https://your-endpoint/v1

set -a && source .env && set +a
envsubst < 05-secret.yaml.tpl | oc apply -f -
```

## Verify the deployment

```bash
# Check the pod is running
oc get pods -n eval-hub -l app=evalhub

# Check health
EVALHUB_URL=$(oc get route evalhub -n eval-hub -o jsonpath='{.spec.host}')
curl -sk "https://$EVALHUB_URL/api/v1/health"

# Check agentdojo provider is registered
curl -sk "https://$EVALHUB_URL/api/v1/evaluations/providers" \
  -H "X-Tenant: eval-hub"
```

## Submit a smoke test

Run a minimal evaluation (single task, utility only, no attack):

```bash
EVALHUB_URL=$(oc get route evalhub -n eval-hub -o jsonpath='{.spec.host}')

curl -sk -X POST \
  -H "X-Tenant: eval-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "agentdojo-smoke-test",
    "model": {
      "url": "https://your-endpoint/v1",
      "name": "openai-compatible",
      "auth": {
        "secret_ref": "agentdojo-api-keys"
      }
    },
    "benchmarks": [
      {
        "id": "workspace",
        "provider_id": "agentdojo",
        "parameters": {
          "model_id": "your-model-id",
          "user_tasks": ["user_task_0"],
          "benchmark_version": "v1.2.2"
        }
      }
    ]
  }' \
  "https://$EVALHUB_URL/api/v1/evaluations/jobs"
```

Check job status:

```bash
JOB_ID="<job-id-from-response>"
curl -sk -H "X-Tenant: eval-hub" \
  "https://$EVALHUB_URL/api/v1/evaluations/jobs/$JOB_ID"
```

## Adding more providers

To add additional providers, create a ConfigMap and add it to the
projected volume in `03-evalhub.yaml`:

```yaml
volumes:
- name: evalhub-providers
  projected:
    sources:
    - configMap:
        name: evalhub-provider-agentdojo
    - configMap:
        name: evalhub-provider-lighteval
```

## File Reference

| File | Description |
|------|-------------|
| `00-namespace.yaml` | Creates the `eval-hub` namespace |
| `01-evalhub-config.yaml` | EvalHub service configuration (SQLite, auth disabled, sidecar config) |
| `02-agentdojo-provider.yaml` | AgentDojo provider ConfigMap for EvalHub |
| `03-evalhub.yaml` | EvalHub Deployment, Service, and Route |
| `03a-rbac.yaml` | RBAC for the default ServiceAccount to manage job resources |
| `04-build-and-push-adapter.sh` | Builds and pushes the adapter container image (standalone script) |
| `05-secret.yaml.tpl` | Secret template for model endpoint credentials (uses `envsubst`) |

## Operator-managed deployment

For production use, EvalHub can be managed by the
[TrustyAI Service Operator](https://github.com/trustyai-explainability/trustyai-service-operator)
via an `EvalHub` custom resource. The operator handles provider ConfigMap
discovery, RBAC, TLS, sidecar configuration, and lifecycle management
automatically. See the
[EvalHub documentation](https://github.com/eval-hub/eval-hub) for details.

## AgentDojo Benchmarks

| Suite | Description |
|-------|-------------|
| `workspace` | Office productivity tasks (email, calendar, cloud drive) |
| `slack` | Slack-like messaging tasks |
| `banking` | Financial transaction tasks |
| `travel` | Travel booking and planning tasks |
| `agentdojo` | All suites combined |

Each suite measures **utility** (does the agent complete the task?) and
**security** (does the agent resist prompt injection attacks?).

## Cleanup

```bash
oc delete deployment evalhub -n eval-hub
oc delete service evalhub -n eval-hub
oc delete route evalhub -n eval-hub
oc delete configmap evalhub-config evalhub-provider-agentdojo -n eval-hub
oc delete secret agentdojo-api-keys -n eval-hub
oc delete rolebinding evalhub-admin -n eval-hub
oc delete namespace eval-hub
```
