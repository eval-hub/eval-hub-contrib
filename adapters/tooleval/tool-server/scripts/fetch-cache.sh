#!/usr/bin/env bash
# Download StableToolBench tools + response cache into ./data/cache (not committed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/data/cache}"
TMP="${TMPDIR:-/tmp}/stb-cache-$$"
mkdir -p "$TMP" "$DEST"

# Pinned HF dataset revision (see PIN.txt). Override URL only with a matching SHA-256.
DEFAULT_REV="bd6dcef8100632b113c9c277a1293ed79cb90865"
DEFAULT_URL="https://huggingface.co/datasets/stabletoolbench/Cache/resolve/${DEFAULT_REV}/server_cache.zip"
# Git LFS OID for server_cache.zip (SHA-256 of archive contents).
DEFAULT_SHA256="a3ed72052c2dc961d8e9b71860374d90ded6d475ca6f6449c78fb248cb82a5b2"

URL="${STABLETOOLBENCH_CACHE_URL:-$DEFAULT_URL}"
EXPECTED_SHA256="${STABLETOOLBENCH_CACHE_SHA256:-$DEFAULT_SHA256}"

if [[ -z "$EXPECTED_SHA256" ]]; then
  echo "STABLETOOLBENCH_CACHE_SHA256 is required when overriding the cache URL" >&2
  exit 1
fi
if [[ "$URL" != "$DEFAULT_URL" && -z "${STABLETOOLBENCH_CACHE_SHA256:-}" ]]; then
  echo "When STABLETOOLBENCH_CACHE_URL is set, also set STABLETOOLBENCH_CACHE_SHA256" >&2
  exit 1
fi

echo "Downloading StableToolBench cache from $URL ..."
curl -fL --retry 3 -o "$TMP/server_cache.zip" "$URL"

actual="$(sha256sum "$TMP/server_cache.zip" | awk '{print $1}')"
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
  echo "SHA-256 mismatch for server_cache.zip" >&2
  echo "  expected: $EXPECTED_SHA256" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi
echo "Verified SHA-256 $EXPECTED_SHA256"

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
