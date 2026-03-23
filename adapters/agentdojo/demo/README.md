# AgentDojo on EvalHub - OpenShift Demo

End-to-end deployment of the AgentDojo prompt injection benchmark adapter
on OpenShift, managed by the TrustyAI Service Operator via an EvalHub
custom resource.

## Overview

This demo deploys:

1. **TrustyAI Service Operator** - manages the EvalHub lifecycle (does not require ODH)
2. **EvalHub service** - evaluation orchestration service with REST API
3. **AgentDojo adapter** - container that runs AgentDojo benchmarks as EvalHub jobs

The adapter evaluates LLM agents on **utility** (task completion) and
**security** (resistance to prompt injection) across four simulated
environments: workspace, slack, banking, and travel.

## Prerequisites

- OpenShift cluster with `oc` CLI logged in
- `podman` (or `docker`) for building the adapter image
- `go` >= 1.22 (for building kustomize during operator install)
- A clone of the [trustyai-service-operator](https://github.com/trustyai-explainability/trustyai-service-operator)
- An OpenAI-compatible model endpoint (e.g. vLLM, LiteLLM proxy)

## Quick Start

All commands assume you are in this directory (`adapters/agentdojo/demo/`).

### 1. Create the namespace

```bash
oc apply -f 00-namespace.yaml
```

### 2. Deploy the TrustyAI operator

```bash
chmod +x 01-operator-crds.sh
OPERATOR_REPO=/path/to/trustyai-service-operator ./01-operator-crds.sh
```

This installs the CRDs (including EvalHub) and deploys the operator
controller into the `eval-hub` namespace.

### 3. Build and push the AgentDojo adapter image

```bash
chmod +x 04-build-and-push-adapter.sh
./04-build-and-push-adapter.sh
```

This builds the adapter container from `adapters/agentdojo/` and pushes
it to the OpenShift internal registry.

### 4. Register the AgentDojo provider

```bash
oc apply -f 02-agentdojo-provider.yaml
```

This creates a ConfigMap with the provider definition that EvalHub uses
to discover and run AgentDojo benchmarks.

### 5. Create the EvalHub instance

```bash
oc apply -f 03-evalhub.yaml
```

Wait for it to become ready:

```bash
oc get evalhub -n eval-hub -w
# NAME      PHASE   READY   AGE
# evalhub   Ready   True    30s
```

### 6. (Optional) Create API key secret

If your model endpoint requires authentication, create a secret from
your environment variables:

```bash
# Create a .env file with your endpoint details:
#   LITELLM_API_KEY=sk-your-key
#   LITELLM_API_URL=https://your-endpoint/v1

set -a && source .env && set +a
envsubst < 05-secret.yaml.tpl | oc apply -f -
```

## Verify the deployment

```bash
# Check all components are running
oc get pods -n eval-hub

# Check EvalHub is ready with agentdojo provider
oc get evalhub -n eval-hub

# Check health via the route
TOKEN=$(oc whoami -t)
EVALHUB_URL=$(oc get route evalhub -n eval-hub -o jsonpath='{.spec.host}')
curl -sk -H "Authorization: Bearer $TOKEN" "https://$EVALHUB_URL/api/v1/health"
```

## File Reference

| File | Description |
|------|-------------|
| `00-namespace.yaml` | Creates the `eval-hub` namespace |
| `01-operator-crds.sh` | Installs CRDs and deploys the TrustyAI operator |
| `02-agentdojo-provider.yaml` | AgentDojo provider ConfigMap for EvalHub |
| `03-evalhub.yaml` | EvalHub custom resource |
| `04-build-and-push-adapter.sh` | Builds and pushes the adapter container image |
| `05-secret.yaml.tpl` | Secret template for model endpoint credentials (uses `envsubst`) |

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
oc delete evalhub evalhub -n eval-hub
oc delete configmap evalhub-provider-agentdojo -n eval-hub
# To remove the operator:
OPERATOR_REPO=/path/to/trustyai-service-operator NAMESPACE=eval-hub \
  $OPERATOR_REPO/bin/kustomize build $OPERATOR_REPO/config/base 2>/dev/null \
  | sed 's/namespace: system/namespace: eval-hub/g' \
  | oc delete -f -
oc delete namespace eval-hub
```
