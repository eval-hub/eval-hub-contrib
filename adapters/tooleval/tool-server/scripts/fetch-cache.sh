#!/usr/bin/env bash
# Download StableToolBench tools + response cache into ./data/cache (not committed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/data/cache}"
TMP="${TMPDIR:-/tmp}/stb-cache-$$"
mkdir -p "$TMP" "$DEST"

URL="${STABLETOOLBENCH_CACHE_URL:-https://huggingface.co/datasets/stabletoolbench/Cache/resolve/main/server_cache.zip}"
echo "Downloading StableToolBench cache from $URL ..."
curl -fL --retry 3 -o "$TMP/server_cache.zip" "$URL"
unzip -q -o "$TMP/server_cache.zip" -d "$TMP/extracted"

# Zip layout varies; locate tools/ and tool_response_cache/
tools="$(find "$TMP/extracted" -type d -name tools | head -1)"
cache="$(find "$TMP/extracted" -type d -name tool_response_cache | head -1)"
if [[ -z "$tools" || -z "$cache" ]]; then
  echo "Could not find tools/ or tool_response_cache/ in archive" >&2
  find "$TMP/extracted" -maxdepth 3 -type d >&2 || true
  exit 1
fi

rm -rf "$DEST/tools" "$DEST/tool_response_cache"
mkdir -p "$DEST"
cp -a "$tools" "$DEST/tools"
cp -a "$cache" "$DEST/tool_response_cache"
echo "Installed tools + cache under $DEST"
du -sh "$DEST/tools" "$DEST/tool_response_cache"
rm -rf "$TMP"
