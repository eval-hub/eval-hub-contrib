# Using the adapter in CI pipelines

Run the three checks from a pipeline with the `evalhub` CLI, then gate the build on the result. The CLI talks to
an EvalHub server that must already have the provider registered ([Running](running.md)).

## What the pipeline needs

- **The CLI:** `pip install "eval-hub-sdk[cli]~=1.0.6"` (the SDK needs Python 3.11 or later).
- **Server and credentials**, from the environment so nothing lands in the job file or logs:

  | Variable | Purpose |
  |---|---|
  | `EVALHUB_BASE_URL` | EvalHub server URL |
  | `EVALHUB_TOKEN` | API token; store it as a CI secret and never echo it |
  | `EVALHUB_PROFILE` | Optional: a named profile instead of the two above |

- **`jq`**, for [`examples/gate-results.sh`](../examples/gate-results.sh).
- **Reachability:** the cluster, not the CI runner, fetches the skill. A `git` source must be cloneable from the
  cluster. For a private repository create a Secret in the job namespace and pass its name as `secret_ref`
  (`--test-data-git-secret`).

## Exit codes: do not rely on them alone

`evalhub eval run --wait` exits **1 only when the whole job ends `FAILED`**. A job where some benchmarks complete
and others fail ends `PARTIALLY_FAILED`, and the CLI still exits 0. With the default `on_incomplete=fail`, one
`scan_incomplete` among three benchmarks is exactly that case. On a `FAILED` job the CLI exits before printing any
JSON.

So gate on the per-benchmark results. `gate-results.sh` reads the JSON and exits 1 unless every benchmark
completed, every `gate_score` is 1.0, `meets_min_score` is true and no scan is incomplete. It fails closed on
empty or invalid input, and prints only benchmark ids, message codes and metric names, never finding text.

## GitHub Actions

Copy `gate-results.sh` into your skills repository (here `ci/gate-results.sh`). The job evaluates the commit under
test, so the repository must be reachable from the cluster.

```yaml
name: skill-gate
on:
  pull_request:
    paths: ["skills/**"]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "eval-hub-sdk[cli]~=1.0.6"
      - name: Evaluate skill
        env:
          EVALHUB_BASE_URL: ${{ vars.EVALHUB_BASE_URL }}
          EVALHUB_TOKEN: ${{ secrets.EVALHUB_TOKEN }}
          SKILL_PATH: skills/changelog-formatter
        run: |
          set +e
          evalhub eval run \
            --name "skill-gate-${GITHUB_SHA::8}" \
            --model-url http://unused/v1 --model-name unused \
            --provider skillevaluator \
            -b skill-quality -b skill-security -b skill-pii \
            --test-data-git-url "${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}.git" \
            --test-data-git-ref "${GITHUB_SHA}" \
            --test-data-git-sub-path "${SKILL_PATH}" \
            --param min_score=80 --param on_incomplete=fail \
            --wait --timeout 900 --format json > result.json 2> run.log
          rc=$?
          cat run.log
          if [ "$rc" -ne 0 ]; then
            job="$(sed -n 's/^Job submitted: //p' run.log | head -n1)"
            echo "evalhub exited ${rc} (job ${job:-unknown})" >&2
            [ -n "$job" ] && evalhub eval status "$job" --format json > result.json
          fi
          set -e
          ./ci/gate-results.sh result.json
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: skill-gate-result
          path: result.json
```

Notes:

- The flags apply the same source and parameters to every `-b` benchmark. `min_score` is only used by
  `skill-quality`, and `on_incomplete` by the two scans. For different settings per benchmark, use a job file
  ([`examples/skill-eval.yaml`](../examples/skill-eval.yaml)) instead of flags.
- `--param min_score=80` is sent as the integer 80; the CLI converts numeric and boolean values.
- `--timeout` bounds the wait. If the job does not finish in time the command errors and the step fails.
- `--format json` sends the progress lines ("Job submitted: ...") to stderr, so `result.json` holds only JSON.
- An unfinished or unreachable server leaves `result.json` empty. The gate treats that as a failure.

## Any other CI system

The steps are the same. A shell fragment you can adapt (GitLab CI, Jenkins, Tekton):

```sh
pip install "eval-hub-sdk[cli]~=1.0.6"      # EVALHUB_BASE_URL and EVALHUB_TOKEN come from CI secrets
set +e
evalhub eval run --config ci/skill-eval.yaml --wait --timeout 900 --format json > result.json 2> run.log
rc=$?
set -e
cat run.log
./ci/gate-results.sh result.json             # exit 1 fails the pipeline
```

If `evalhub eval run` exited non-zero before printing JSON, `result.json` is empty and the gate fails. To see
why, fetch the job with `evalhub eval status <job-id> --format json` (the id is in `run.log`).

## Tolerating incomplete scans

Many real skills make SkillSpector report a partial scan ([incomplete scans](incomplete-scans.md)). If you accept
that in CI, pass `on_incomplete=report` and tell the gate to treat "not scanned" as a warning:

```sh
evalhub eval run ... --param on_incomplete=report --wait --format json > result.json
ALLOW_INCOMPLETE=1 ./ci/gate-results.sh result.json
```

A missing scan then shows as `warning: <benchmark>: scan incomplete, not scanned` while real failures still fail
the build. Leave `ALLOW_INCOMPLETE` unset to keep an unscanned skill from passing.

## Verification status

`gate-results.sh` is covered by this adapter's tests against synthetic job documents, including the
`PARTIALLY_FAILED` and empty-input cases. The workflow above uses the CLI's flags and the SDK's job model as of
`eval-hub-sdk` 1.0.6; it has not been run end to end from a hosted CI runner against a live EvalHub server.
