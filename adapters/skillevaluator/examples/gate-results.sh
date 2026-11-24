#!/usr/bin/env bash
# Gate a CI job on the result of `evalhub eval run --format json`.
#
# Usage: gate-results.sh [result.json]      (reads stdin when no file is given)
#
# Exits 0 only when every benchmark completed and every gate passed. Exits 1 and prints one line per
# problem otherwise. It checks per-benchmark status because `evalhub eval run --wait` exits non-zero
# only when the whole job is FAILED: a job where some benchmarks completed and others failed ends as
# PARTIALLY_FAILED and the CLI still exits 0.
#
# ALLOW_INCOMPLETE=1 downgrades "scan incomplete, not scanned" (on_incomplete=report) to a warning. Use it
# only if a missing scan is acceptable for the pipeline; the default treats it as a failure.
#
# Only job ids, benchmark ids, message codes and metric names are printed: never finding text.
# Requires jq.
set -euo pipefail

allow_incomplete="${ALLOW_INCOMPLETE:-0}"

# Read once so stdin works, and refuse empty or non-JSON input: jq prints nothing for an empty document,
# which must never be mistaken for "no problems".
doc="$(cat "${1:-/dev/stdin}")"
if ! printf '%s' "$doc" | jq -e 'type == "array" or type == "object"' >/dev/null 2>&1; then
  echo "Skill gate FAILED: no job result to check (empty or invalid JSON)." >&2
  exit 1
fi

problems="$(jq -r --arg allow "$allow_incomplete" '
  (if type == "array" then .[0] else . end) as $job
  | ( ($job.status.benchmarks // [])[]
      | select(.status != "completed")
      | "\(.id): benchmark \(.status)"
        + (if .error_message then " (\(.error_message.message_code // "error"))" else "" end) ),
    ( ($job.results.benchmarks // [])[]
      | .id as $id | (.metrics // {}) as $m
      | ( if $m.scan_complete == false then
            (if $allow == "1" then "WARN \($id): scan incomplete, not scanned" else "\($id): scan incomplete, not scanned" end)
          else empty end ),
        ( if ($m.gate_score // 1) < 1 then "\($id): gate failed" else empty end ),
        ( if $m.meets_min_score == false then "\($id): quality below min_score" else empty end ) ),
    ( if (($job.results.benchmarks // []) | length) == 0 then "no benchmark results" else empty end )
' <<<"$doc")"

warnings="$(printf '%s\n' "$problems" | grep '^WARN ' || true)"
failures="$(printf '%s\n' "$problems" | grep -v '^WARN ' | grep -v '^$' || true)"

if [ -n "$warnings" ]; then
  echo "$warnings" | sed 's/^WARN /  - warning: /' >&2
fi
if [ -n "$failures" ]; then
  echo "Skill gate FAILED:" >&2
  echo "$failures" | sed 's/^/  - /' >&2
  exit 1
fi
echo "Skill gate passed."
