#!/bin/bash
# Build the AgentDojo adapter image and push it to the OpenShift internal registry.
#
# Prerequisites:
#   - oc CLI logged into the target cluster
#   - podman or docker installed
#   - The eval-hub namespace must already exist
#
# Usage:
#   ./04-build-and-push-adapter.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADAPTER_DIR="$SCRIPT_DIR/.."
NAMESPACE="${NAMESPACE:-eval-hub}"
BUILD_TOOL="${BUILD_TOOL:-podman}"

# Discover the external registry route
REGISTRY_HOST=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}' 2>/dev/null)
if [ -z "$REGISTRY_HOST" ]; then
  echo "ERROR: Could not find the OpenShift internal registry route."
  echo "Ensure the image registry has an external route exposed."
  exit 1
fi

IMAGE="$REGISTRY_HOST/$NAMESPACE/agentdojo-adapter:latest"

echo "==> Logging into registry $REGISTRY_HOST"
$BUILD_TOOL login "$REGISTRY_HOST" -u "$(oc whoami)" -p "$(oc whoami -t)" --tls-verify=false

echo "==> Building image: $IMAGE"
cd "$ADAPTER_DIR"
$BUILD_TOOL build --platform linux/amd64 -t "$IMAGE" -f Containerfile .

echo "==> Pushing image: $IMAGE"
$BUILD_TOOL push "$IMAGE" --tls-verify=false

echo "==> Done. Image available at:"
echo "    External: $IMAGE"
echo "    Internal: image-registry.openshift-image-registry.svc:5000/$NAMESPACE/agentdojo-adapter:latest"
