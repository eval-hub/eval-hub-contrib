# Development

## Layout

| File | Role |
|---|---|
| `main.py` | `SkillEvaluatorAdapter`: validates parameters, runs the CLI, reports status, builds the cards |
| `_runner.py` | Runs the CLI by subprocess with an allow-listed environment; never logs its output |
| `_results.py` | Pure mapping from the CLI's JSON report to metrics and an outcome |
| `_skill_source.py` | Finds the skill under `/test_data` and keeps it inside that directory |
| `examples/` | Example job file and the CI gate script |

## Implementation notes

- SkillEvaluator documents a CLI and a versioned JSON report, not a Python API, so the adapter drives the CLI and
  parses the report. An unexpected report shape fails the job loudly instead of producing a plausible score.
- SkillEvaluator and SkillSpector are not published on PyPI. Each is installed from a pinned release commit into
  its own virtualenv in the image (`requirements-skillevaluator.txt`, `requirements-skillspector.txt`), so
  neither dependency stack constrains the adapter's. `git` exists only in the build stage.
- Tier 1 `publication_status` is ignored: upstream reports `incomplete` whenever Tier 2 or Tier 3 did not run,
  which is the normal case here.
- The EvalHub `target_type` enum has no skill/artifact value; the provider declares `agent` and receives the
  skill through `test_data_ref`.

## Tests

```sh
make test-skillevaluator        # offline: stubbed CLI replays golden reports (tests/fixtures/reports)
make test-skillevaluator-live   # real pinned CLIs; needs git and network; guards the report contract
```

The gate script test needs `jq` and is skipped without it. The golden reports are real CLI output with finding
text redacted and paths normalised. The malicious-looking skill used by the live tests is generated into a temp
directory at test time and never committed.

CI also enforces lint (`ruff==0.16.9`) and docstring coverage of at least 80% (`interrogate`, tests included).

## Upgrading SkillEvaluator or SkillSpector

1. Change the commit in `requirements-skillevaluator.txt` / `requirements-skillspector.txt` (and the SkillSpector
   version SkillEvaluator documents).
2. Run `make test-skillevaluator-live`. A schema change fails here first.
3. If the report shape changed, regenerate `tests/fixtures/reports/*.json` from real output (redact `message`,
   `line_content`, `suggestion` and `[CATEGORY-SEVERITY]` lines; normalise paths) and update `_results.py`.
4. Re-run the container checks in `.github/workflows/test-skillevaluator.yml`.

## Roadmap

- **LLM stages**: `rubric-eval` (judge on any OpenAI-compatible endpoint) and Tier 2 deduplication (needs an
  embeddings endpoint).
- **Tier 3 live evaluation** (`skillevaluator-live`, separate provider): with/without-skill agent runs, five
  dimensions, Skill Lift, pass@k. Needs a sandbox that works in a pod (upstream's default is Docker).
- **Parse SkillSpector coverage**: read its coverage and reason codes so harmless partial scans can pass with a
  warning while real tool failures still fail.
- **Code integrity checks** (Semgrep, Gitleaks, Bandit) as a separate benchmark.
- An evaluation collection pairing this adapter with a malicious-skill benchmark.
