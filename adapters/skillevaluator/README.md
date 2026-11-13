# SkillEvaluator adapter

Static evaluation of **agent skills** (a directory containing `SKILL.md`, per the
[Agent Skills specification](https://agentskills.io/)) with
[NVIDIA SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator)
([docs](https://docs.nvidia.com/skills/skillevaluator)), Apache-2.0 and Experimental upstream.

This adapter covers **Tier 1 static checks**. They are deterministic: no model is called and no skill content is
executed. LLM-backed checks (rubric, Tier 2 deduplication) and live with/without-skill agent evaluation (Tier 3)
are planned follow-ups, see the [roadmap](docs/development.md#roadmap).

## Benchmarks

| Benchmark | Upstream command | What it reports |
|---|---|---|
| `skill-quality` | `quality-check` | `quality_score` (0-100 mapped to 0..1), `quality_grade` (A-F), four dimension scores (correctness 35%, discoverability 25%, reliability 25%, efficiency 15%), `meets_min_score`, `issue_count` |
| `skill-security` | `security-scan --no-llm` | SkillSpector static analysis: severity counts and a pass/fail `gate_score` |
| `skill-pii` | `pii-scan` | Personal paths, emails, phone numbers, SSNs, credentials, keys, JWTs, webhooks, public IPs, card numbers, wallet addresses: severity counts and `gate_score` |

`gate_score` is `1.0` when the scan completed with no blocking findings and `0.0` when it completed and failed
the gate. The verdict is upstream's (`overall_status`), not re-implemented here. A scan that cannot finish is
never reported as zero findings: see [incomplete scans](docs/incomplete-scans.md).

A pass is not proof a skill is benign: SkillSpector static analysis scored 0% malicious recall on
MaliciousSkillBench. See the [security model](docs/security-model.md).

## Supplying the skill

The skill is staged read-only at `/test_data` through `test_data_ref`. The adapter targets RHOAI 3.6 and later,
where `git`, `hf`, `s3` and `pvc` are all supported, and never fetches anything itself. Exactly one skill is
expected; if the staged material holds several, set `parameters.skill_path` (relative to `/test_data`).

## Quick start

```sh
make image-skillevaluator push-skillevaluator REGISTRY=registry.example.com/myorg VERSION=0.1.0
evalhub providers create --file adapters/skillevaluator/provider.yaml
evalhub eval run --config adapters/skillevaluator/examples/skill-eval.yaml --wait
```

## Documentation

| Topic | Page |
|---|---|
| Build, register, job files, sources, results, parameters | [docs/running.md](docs/running.md) |
| CI pipelines with the `evalhub` CLI, gating on results | [docs/ci-pipelines.md](docs/ci-pipelines.md) |
| `on_incomplete` and partial scans | [docs/incomplete-scans.md](docs/incomplete-scans.md) |
| Threat model and what a pass means | [docs/security-model.md](docs/security-model.md) |
| Layout, tests, upgrading, roadmap | [docs/development.md](docs/development.md) |
