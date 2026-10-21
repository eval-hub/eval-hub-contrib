# Security model

Skills are untrusted input.

- Nothing in the skill is executed by these benchmarks; all checks are static analysis.
- The skill path is validated to stay inside `/test_data` (no `..`, no symlink escapes) and symlinked or hidden
  directories are not traversed.
- The `skillevaluator` CLI runs by `subprocess` with a fixed argument list (no shell) and an allow-listed
  environment: job credentials and service-account variables are not forwarded.
- Finding text and matched lines quote skill content, so they are never put in metrics, logs or failure messages.
  Only counts and the scanner's own rule names are exposed; the full report is kept as the OCI artifact when
  `exports.oci` is set.
- No network access is needed: `security-scan` runs with `--no-llm`, and Semgrep (not used by these benchmarks) is
  configured upstream with metrics off. Deny egress to the job pod.
- The image runs read-only as a non-root user, including with an arbitrary UID.
- Credentials for private skill sources are referenced through `secret_ref`, never placed in the job request.

## What a pass does not mean

SkillSpector static analysis scored **0% malicious recall** on
[MaliciousSkillBench](https://arxiv.org/abs/2608.19901). A passing `skill-security` result means the static rules
found nothing, not that the skill is benign. Use it as one gate, alongside a malicious-skill benchmark.
