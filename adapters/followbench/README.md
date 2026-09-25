# FollowBench Adapter

FollowBench evaluates whether language models follow multiple constraints in
complex instructions.

It covers:

- Content constraints
- Situation constraints
- Style constraints
- Format constraints
- Mixed constraints
- Example constraints

## Upstream source

- Repository: https://github.com/YJiangcm/FollowBench
- Paper: https://arxiv.org/abs/2310.20410
- Pinned revision: `6278f4c1377b4eafab737267b8b21acd52ea0e52`
- License: Apache License 2.0
- License file: `FOLLOWBENCH-LICENSE`

FollowBench should be attributed to:

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li,
Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang.

## Metrics

| Metric | Scale | Description |
|---|---:|---|
| `hsr` | `0.0–1.0` | Hard Satisfaction Rate |
| `ssr` | `0.0–1.0` | Soft Satisfaction Rate |
| `csl` | `0.0–5.0` | Consistent Satisfaction Levels |
| `n_evaluated` | integer | Number of evaluated records |

`hsr` is the primary metric. Higher values are better.

HSR and SSR are normalized to the range `0.0–1.0`.
CSL preserves the five-level FollowBench scale.

## Evaluation flow

The adapter:

1. Loads the pinned FollowBench data.
2. Sends each FollowBench instruction to the evaluated model.
3. Applies deterministic rule-based checks where available.
4. Sends judge-based cases to a configurable external judge.
5. Calculates HSR, SSR, and CSL.
6. Reports status and results through the EvalHub callback contract.

The adapter does not use the upstream model-serving loop, `fschat`, or vLLM
evaluation loop.

## External judge

The following parameters are supported:

- `judge_model`
- `judge_url`
- `judge_api_key`
- `num_examples`
- `max_tokens`
- `temperature`
- `request_timeout`

Credentials must be injected through configuration or environment variables.
Credentials must not be committed to source code, job specifications, or
container images.

Judge runtime and cost depend on:

- Number of evaluated example groups
- Number of constraint levels
- Judge prompt length
- Judge completion length
- Judge model latency

Use a small `num_examples` value for local development and smoke tests.

## Local setup

Create the Python environment:

```bash
uv venv --python 3.12 .venv-followbench
source .venv-followbench/bin/activate
uv pip install \
  -r adapters/followbench/requirements.txt \
  -r adapters/followbench/requirements-test.txt
```

Run the adapter tests from the adapter directory:

```bash
cd adapters/followbench
python -m pytest tests
```

## Smoke-test expectations

The smallest complete smoke run uses one example group and evaluates levels
1 through 5. It may make up to five model-generation calls and five judge
calls; rule-based cases can reduce the number of judge calls. Runtime depends
on model and judge latency, prompt length, and generated-token limits.

For a real smoke run, use the approved EvalHub image/provider path with a
reachable OpenAI-compatible model endpoint and judge endpoint. The job
specification uses `num_examples: 1` and does not contain credentials. Inject
credentials through the runtime Secret or environment configuration.

The expected output contains `hsr`, `ssr`, `csl`, and `n_evaluated`. HSR and
SSR are fractions in `0.0–1.0`; CSL is the average number of consecutive
satisfied levels on the five-level scale. The proposed HSR smoke floor is
`0.20`, pending calibration against reference results.
