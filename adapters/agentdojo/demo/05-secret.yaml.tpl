apiVersion: v1
kind: Secret
metadata:
  name: agentdojo-api-keys
  namespace: eval-hub
type: Opaque
stringData:
  OPENAI_COMPATIBLE_API_KEY: "${LITELLM_API_KEY}"
  LITELLM_API_KEY: "${LITELLM_API_KEY}"
  OPENAI_COMPATIBLE_BASE_URL: "${LITELLM_API_URL}"
