#!/bin/bash
# Install TrustyAI Service Operator CRDs and deploy the operator.
#
# Prerequisites:
#   - oc CLI logged into the target cluster
#   - go >= 1.22 installed (for kustomize build)
#   - A clone of the trustyai-service-operator repo
#
# Usage:
#   OPERATOR_REPO=/path/to/trustyai-service-operator ./01-operator-crds.sh

set -euo pipefail

OPERATOR_REPO="${OPERATOR_REPO:?Set OPERATOR_REPO to the path of your trustyai-service-operator clone}"
NAMESPACE="${NAMESPACE:-eval-hub}"

echo "==> Installing CRDs from $OPERATOR_REPO"
cd "$OPERATOR_REPO"

# Ensure kustomize v5 is available
if ! ./bin/kustomize version 2>/dev/null | grep -q 'v5'; then
  echo "    Installing kustomize v5..."
  GOBIN="$(pwd)/bin" go install sigs.k8s.io/kustomize/kustomize/v5@latest
fi

# Apply CRDs
./bin/kustomize build config/crd | oc apply -f -

# Deploy the operator (fix hardcoded namespace: system -> target namespace)
echo "==> Deploying operator to namespace $NAMESPACE"
cd config/manager && ../../bin/kustomize edit set image controller=quay.io/trustyai/trustyai-service-operator:latest && cd ../..
./bin/kustomize build config/base 2>/dev/null \
  | sed "s/namespace: system/namespace: $NAMESPACE/g" \
  | oc apply -n "$NAMESPACE" -f -

echo "==> Waiting for operator pod to be ready..."
oc wait --for=condition=available deployment/trustyai-service-operator-controller-manager \
  -n "$NAMESPACE" --timeout=120s

echo "==> Operator deployed successfully"
