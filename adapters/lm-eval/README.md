# LM Evaluation Harness adapter

## Local checks

`make test-lm-eval` installs the pinned framework and test dependencies, applies
the reviewed task compatibility fixes, then runs Ruff and the adapter tests.
For an existing virtual environment, run these in the adapter directory:

```sh
python scripts/apply_harness_fixes.py
python -m pytest tests/ -m "not local"
```

The Containerfile and registry CI job apply the same fixes to the Harness used
by the `lm_eval` CLI. They are scoped to version 0.4.13 and exact source hashes;
a dependency change fails explicitly until the fixes are reviewed or removed.
Remove the compatibility patches when the consumed upstream release includes
the corresponding behavior. No runtime package files are rewritten.

## Task compatibility fixes

- **tinyTruthfulQA (RHAI-51139, ODH Harness PRs #248/#249):** runs with other than
  exactly 100 completed anchor outcomes omit the standard IRT score. The adapter
  returns the executed sample count and `additional_info.aggregation` with
  `status: skipped`, `required_examples: 100`, and the reason. No zero score is
  substituted, and no scored EvalCard entry is emitted for an unscored run.
  Exactly 100 outcomes retain the original estimator. This does not establish
  that arbitrary reordered samples are valid anchors; the standard task and
  its predefined ordering must still be used. Other tiny tasks keep their
  aggregation behavior. Group means omit an unscored child's value, weight
  and stderr together; a group with no scored children remains unscored. The
  group mean covers the scored children only, while execution sample metadata
  still counts all completed samples. Full tiny-benchmark scoring still needs
  its existing optional `tinyBenchmarks` dependency; a partial execution check
  does not.
- **CAREQA (RHOAIENG-85393, ODH Harness PR #251):** importing the task for
  `careqa_open_perplexity` does not initialize generation metrics. Generated
  answers still load BLEU, ROUGE, BERTScore and BLEURT on first use and report
  the actual metric that failed to load. Their scoring definitions are unchanged.

These preserve fixes previously made in the ODH Harness adapter/task library
when using this standalone adapter. Job-level remote-code and unsafe-code
settings are outside this change.
