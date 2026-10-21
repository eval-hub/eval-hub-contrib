# Incomplete scans and `on_incomplete`

A check is **incomplete** when SkillEvaluator reports that a scan could not fully run. Two causes occur:

1. **A scanner is missing or cannot run**, for example the `skillspector` binary. This points at a broken image,
   not at the skill.
2. **SkillSpector only partly analysed the skill.** This is common for real skills whose `SKILL.md` refers to
   files outside the skill directory (for example backtick references to `VERSION` or `dependencies.yaml`).
   SkillSpector then reports partial analysis, and SkillEvaluator drops the findings and reports zero counts.

In practice incomplete results come from `skill-security`. `skill-pii` is a regex-only scan.

## Choosing the behaviour

`parameters.on_incomplete` is set per benchmark:

| Value | Job | Metrics |
|---|---|---|
| `fail` (default) | Fails with message code `scan_incomplete`, the scans involved and a bounded reason | none |
| `report` | Completes | only `scan_complete=false`; `overall_score=null` and `outcome=incomplete` in the metadata |

An incomplete scan never emits a `gate_score` or severity counts: zeros would read as "no findings". Reasons are
upstream's own status text; anything that quotes the skill is dropped.

## Passing `report`

In a job file:

```yaml
benchmarks:
  - id: skill-security
    provider_id: skillevaluator
    parameters:
      on_incomplete: report          # fail (default) | report
    test_data_ref:
      git: {url: https://github.com/example-org/agent-skills.git, ref: v1.2.0, sub_path: skills/changelog-formatter}
```

or with flags (applies to every `-b` benchmark in the command):

```sh
evalhub eval run --name skill-gate --model-url http://unused/v1 --model-name unused \
  --provider skillevaluator -b skill-security \
  --test-data-git-url https://github.com/example-org/agent-skills.git --test-data-git-ref v1.2.0 \
  --test-data-git-sub-path skills/changelog-formatter \
  --param on_incomplete=report --wait
```

Any other value is rejected before the CLI runs: `parameters.on_incomplete must be one of ['fail', 'report']`.

## Using `report` safely

With `report`, a downstream consumer must treat a missing `gate_score` or `scan_complete=false` as
"not scanned", never as a pass. [`examples/gate-results.sh`](../examples/gate-results.sh) does this and fails the
pipeline on it unless you set `ALLOW_INCOMPLETE=1`; see [CI pipelines](ci-pipelines.md).

## What a pass does not mean

SkillSpector static analysis scored **0% malicious recall** on
[MaliciousSkillBench](https://arxiv.org/abs/2608.19901). A passing `skill-security` result means the static
rules found nothing, not that the skill is benign. Use it as one gate, alongside a malicious-skill benchmark.
