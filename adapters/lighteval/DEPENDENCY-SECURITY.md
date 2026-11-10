# LightEval adapter: dependency security notes

Scan date: 2026-10-26. Tool: `pip-audit` (PyPI/OSV advisories) on the universal lock
(`requirements.lock`), which covers every platform: 183 package versions, including
the Linux-only CUDA packages (`nvidia-*`, `cuda-*`, `triton`) that torch pulls into
the x86_64 and aarch64 images. Environment markers are stripped before auditing,
because `pip-audit` otherwise evaluates them for the machine it runs on and silently
skips the Linux-only packages. `lighteval` 0.13.0 is a normal PyPI release and is
audited with the rest.

Not covered: operating-system packages in the UBI9 base image (the Trivy workflow
covers those) and anything outside Python.

Re-run:

```sh
cd adapters/lighteval
uv pip compile requirements.txt --override requirements-overrides.txt \
    --universal --python-version 3.12 --no-header --no-annotate -o requirements.lock
grep -v '^#' requirements.lock | grep '==' | sed 's/ ;.*//' | sort -u > /tmp/audit-in.txt   # strip markers
pip-audit -r /tmp/audit-in.txt --no-deps --disable-pip
```

## Result

| | Vulnerable packages | Advisories |
|---|---|---|
| Before: `lighteval==0.13.0`, `litellm==1.84.0` | 3 (nltk, litellm, diskcache) | 44 |
| After: `lighteval==0.13.0`, `litellm==1.102.2`, `nltk==3.10.4` | 1 (diskcache) | 1 |

"Before" is the tree a fresh resolve of the previous pins gives. A resolve that keeps
older cached versions of transitive packages is worse, see
[Stale transitive versions](#stale-transitive-versions).

## Known vulnerabilities with no fix (shipped set)

### diskcache 5.6.3 — PYSEC-2026-2447 / CVE-2025-69872 / GHSA-w8v5-vhqr-4h9v

> DiskCache (python-diskcache) through 5.6.3 uses Python pickle for serialization by
> default. An attacker with write access to the cache directory can achieve arbitrary
> code execution when a victim application reads from the cache.

- **No fixed version.** 5.6.3 is the latest release and dates from 2023-08-31.
- **Why it is installed:** `litellm[caching]` requires `diskcache==5.6.3` exactly, and
  lighteval imports it at start-up: `lighteval/models/endpoints/litellm_model.py` runs
  `litellm.cache = Cache(type=DISK)` when it is imported. Dropping the `[caching]`
  extra therefore breaks lighteval's import.
- **Exposure in this adapter:** exploiting it needs write access to the cache
  directory, which litellm creates as `.litellm_cache` in the working directory.
  `lighteval_logprob_patch.py` sets `litellm.cache = None` right after lighteval's
  import, so nothing reads from that directory during an evaluation. lighteval's
  LLM-as-judge metrics (`metrics/utils/llm_as_judge.py`) switch the disk cache back on;
  none of the 28 advertised benchmarks use judge metrics.
- **Mitigation:** none upstream. Keep the pod filesystem private to the job (it is, in
  the standard deployment). Revisit when litellm or lighteval drops diskcache.

### Advisories that no longer apply, for reference

These affected the previous `nltk==3.9.1` pin (42 advisories) and are gone with
`nltk==3.10.4`. Of the 42, **2 have no fixed version in any OSV record** and 4 are
recorded inconsistently (one record names a fix, another none):

| Advisory | Aliases | Fixed in | Summary |
|---|---|---|---|
| PYSEC-2026-3955 | CVE-2026-12372, GHSA-2jhm-w3mp-jcwr | none recorded | SSRF in nltk 3.9.4 and develop |
| PYSEC-2026-4318 | CVE-2026-12259 | none recorded | `Downloader._download_package()` writes and may extract downloaded bytes before checking the SHA-256/MD5 checksum |
| PYSEC-2026-97 | CVE-2026-0846, GHSA-h8wq-7xc4-p3qx | 3.9.3 (one record: none) | arbitrary file read in `nltk.util.filestring()` (absolute paths, traversal) |
| PYSEC-2026-98 | CVE-2026-0847, GHSA-68j8-pq59-fqgm | 3.9.3 (one record: none) | path traversal in several `CorpusReader` classes |
| PYSEC-2026-597 | CVE-2026-12243, GHSA-m42h-3232-vpv3 | 3.10.0 (one record: none) | incomplete fix for a path traversal in `nltk.data` |
| PYSEC-2026-3740 | CVE-2026-81726, GHSA-8mgp-746c-j5xp | 3.10.3 (one record: none) | path traversal in model-artifact APIs; still present in 3.10.3 |

`nltk 3.10.3` (what lighteval `main` pins) still carries PYSEC-2026-3740 with no fix.
`nltk 3.10.4` has no advisory at scan time, which is why it is installed on top via
`requirements-overrides.txt`. If that file is dropped, expect that one advisory.

nltk 3.10.4 was checked against every nltk call lighteval makes (`word_tokenize`,
`TreebankWordTokenizer`, `edit_distance`, `sentence_bleu`, `scores.f_measure`, the
`punkt_tab` download): results are identical to 3.9.1.

## litellm: choosing a version

The adapter drives lighteval's LiteLLM backend, so the litellm version is ours to
choose (lighteval only asks for `litellm[caching]>=1.66.0`, as an optional extra).

The version must exist on the Red Hat package index, which `verify-requirements.yml`
queries. On 2026-10-27 that index offered litellm 1.102.2, 1.84.0, 1.75.8 and 1.74.15.
1.103.9, which this table first chose from PyPI alone, was not there, so the check
failed on `main`. 1.102.2 is the newest version the index has, and the one used now.

| litellm | Own advisories | Caps `filelock<4.0` | Repeated request (cache hit) without `orjson`+`soundfile` |
|---|---|---|---|
| 1.84.0 (previous pin) | PYSEC-2026-4066 (proxy only, see below) | no | works (tested) |
| 1.96.2, 1.97.0 | none | no | not tested |
| **1.102.2 (chosen, newest on the Red Hat index)** | none (pip-audit of the lock) | no (filelock resolves to 4.2.4) | not tested; the wrapper disables the cache, so a hit cannot occur |
| 1.103.9 (first pin; not on the Red Hat index) | none | no | **crashes (tested); disabled by the wrapper** |
| 1.104.7, 1.105.5 | none | **yes** | 1.105.5 crashed in a lighteval run (not isolated) |
| 1.106.4 (latest, 2026-10-25) | none | **yes** | **crashes (tested)**; wrapper fix also works (tested) |

**PYSEC-2026-4066 (litellm 1.84.0):** any authenticated *LiteLLM proxy* user could
redirect an outbound provider call and exfiltrate the operator's provider credentials.
It concerns the proxy server. This adapter uses litellm as a client library and never
runs the proxy, so it is not exploitable here. Fixed in 1.88.6, 1.89.7, 1.90.7,
1.91.5, 1.92.2, 1.93.2, 1.94.3, 1.95.1 and 1.96.2 and later.

### Caveat 1: a litellm cache hit crashes, and the benchmark is scored on empty output

lighteval enables litellm's disk cache and sends `caching=True` with every chat
request. On litellm 1.103.9 and 1.106.4 a **cache hit** takes a code path that imports
`litellm.proxy` modules:

```
litellm.completion -> caching_handler._sync_get_cache
  -> _should_defer_streaming_cache_hit_callbacks
  -> litellm/llms/anthropic/.../response_cache.py -> litellm/proxy/... 
  -> import orjson            # ModuleNotFoundError
  -> import soundfile         # ModuleNotFoundError once orjson is present
```

Neither package is installed by `litellm[caching]` (`orjson` is declared only under
litellm's `proxy` extra; `soundfile` not at all). A first request works, an identical
repeated request raises before any network call.

What lighteval does with that exception (`__call_api`): it logs a warning and retries
8 times with waits of 1, 2, 4, 8, 16, 32, 64, 64 s (**191 s per request**), then
**returns an empty response**. The task finishes with exit code 0 and scores blank
output as wrong answers. A generative benchmark therefore reports a low score instead
of failing. This is the worst property of the bug.

When it triggers: any repeated identical request, such as a re-run in the same working
directory (the disk cache persists there) or duplicated prompts inside a task. It does
not affect log-likelihood benchmarks, which bypass litellm in
`lighteval_logprob_patch.py`.

Fix in this adapter: the wrapper sets `litellm.cache = None` after importing lighteval,
so every request reaches the model and no proxy-only package is needed
(`test_litellm_response_cache_is_disabled_so_cache_hits_cannot_crash`; it fails
without the fix on 1.103.9). Installing `orjson` and `soundfile` also works but adds
two dependencies and leaves the path exposed to the next missing module.

### Caveat 2: litellm 1.104 and later pin filelock to the vulnerable 3.x line

`litellm>=1.104` declares `filelock>=3.16.1,<4.0`. The advisories below are fixed only
in filelock 4.2.1, so with 1.104 or later the resolver stays on filelock 3.32.x:

| Advisory | Fixed in | Summary |
|---|---|---|
| CVE-2026-112940 / GHSA-v4x8-3wr5-3wq8 | 4.2.1 | `ReadWriteLock` / `AsyncReadWriteLock` (SQLite-backed) hand SQLite an unsafe name |
| CVE-2026-112913 / GHSA-9vhv-jxww-xfm5 | 4.2.1 | Windows: same locks connect SQLite to an unsafe path |
| CVE-2026-112907 / GHSA-567q-5qvw-pf92 | 4.2.1 | Windows: soft locks (`SoftFileLock` and variants) |

Two of three are Windows-only and the adapter image is Linux; whether anything in the
image uses `ReadWriteLock` was not checked.

## Why lighteval stays on the 0.13.0 release

An earlier revision of this branch pinned `lighteval @ git+https://github.com/huggingface/lighteval.git@7fa19dc6f53a713ba8bb6541e7ed66dad2e24fd7`
(`0.13.1.dev0`, 2026-10-06), because `main` pins `nltk==3.10.3` while 0.13.0 pins
`nltk==3.9.1`. It was dropped for three reasons:

- **The requirements check rejects it.** `verify-requirements.yml` runs the action
  `ruivieira/basic-ops-tasks` in `python-index` mode on `requirements.txt`. Its code
  (`src/requirements.ts`) turns every VCS or direct-URL requirement into
  `status: 'error'`: "VCS requirements are not supported: only the configured index is
  queried". That job would fail for a `git+https` line even with the index set.
- **No security gain.** nltk is raised to 3.10.4 by `requirements-overrides.txt` either
  way, and the audit of the lock is identical for both lighteval versions: 183 package
  versions, only diskcache vulnerable.
- **`main` fixes none of what this adapter patches.** `LiteLLMClient.loglikelihood`
  still raises `NotImplementedError`, PIQA still loads a dataset script
  (`ybisk/piqa`), and the `truthfulqa:gen` prompt still reads `mc1_targets`.
  `lighteval_logprob_patch.py` and `custom_tasks.py` stay necessary.

A git pin would also need `git` and github.com access during the image build (the
`ubi9/python-311`, `-312` and `-314` images do ship git 2.52.0) and would pin an
unreleased commit.

The unit suite (115 tests) and the end-to-end CLI cases pass on both lighteval 0.13.0
and that `main` commit. Transitive packages are locked in `requirements.lock`, so the
audit above describes exactly what the image installs.

## Container image: pip cannot resolve the unpinned set; use the lock

Building the Containerfile with `pip install -r requirements.txt` on the
`ubi9/python-312` image (pip 24.2) **failed**:
`pip._vendor.resolvelib.resolvers.ResolutionTooDeep: 200000` after about 10 minutes of
backtracking through `aiobotocore`, `inspect-ai` and `transformers`. `uv` resolves the
same requirements in seconds.

It comes from the changed pins, not from the base image: `main`'s previous requirements
(`lighteval==0.13.0`, `litellm==1.84.0`) resolve on the `python-311` image in 2m43s, and
the git-pinned `lighteval@7fa19dc` (a variant tried earlier) with `litellm==1.84.0` resolves in 3m38s. `lighteval==0.13.0` with
`litellm==1.103.9` had not finished after about 15 minutes and was stopped, so
litellm 1.103.9 is the likely trigger. Only the 3.12 failure was observed; the unlocked
set was not built on 3.11 or 3.14.

Fix: `requirements.lock`, a fully pinned universal lock generated with `uv pip compile`
(command in its header), installed with `pip install --no-deps -r requirements.lock`.
`atif` already uses a `requirements.lock` the same way. `pip check` then reports one
conflict, the intended one (lighteval wants nltk 3.10.3, the lock has 3.10.4).

Regenerate the lock whenever `requirements.txt` or `requirements-overrides.txt`
changes, and re-run the audit above.

## Python version of the base image

Built from the lock on `linux/arm64`, then ran the end-to-end CLI cases (`arc:easy`,
`truthfulness`, `physical_commonsense`, `math`) inside the container:

| Base image | Python | Last rebuilt | Result |
|---|---|---|---|
| `ubi9/python-311` (previous base) | 3.11.13 | 2026-05-28 | builds, end-to-end passes (repo Containerfile, unmodified except for the lock) |
| **`ubi9/python-312` (chosen)** | 3.12.15 | 2026-10-24 | builds, end-to-end passes; Linux default start method is still `fork` |
| `ubi9/python-313` | n/a | n/a | no such image at `registry.access.redhat.com/ubi9/python-313` (HTTP 404) |
| `ubi9/python-314` | 3.14.8 | 2026-10-24 | builds, end-to-end passes for the four cases above; **LiveCodeBench would silently score 0**, see below |

All ship pip 24.2, git 2.52.0, UID 1001 and RHEL 9.8. The unit suite (115 tests) also
passes in local venvs on 3.12, 3.13 and 3.14. The same package versions resolve on
3.11 to 3.14.

### Python 3.14: `multiprocessing` defaults to `forkserver` on Linux

Through 3.13 Linux forks new processes; 3.14 starts them from a fork server, so the
target of a `Process` must be picklable. lighteval's LiveCodeBench scorer
(`lcb:codegeneration_v6`, `tasks/tasks/lcb/codegen_metrics.py`) starts a `Process` whose
target, `_temp_run`, is defined inside `check_correctness`. Tested directly:

| Start method | `check_correctness` on a trivial correct solution |
|---|---|
| `fork` (3.11 to 3.13 default on Linux) | correct |
| `forkserver` (3.14 default on Linux), `spawn` (macOS default) | `PicklingError: Can't pickle local object <function check_correctness.<locals>._temp_run>` |

`evaluate_generations_by_problem` wraps the call in `except Exception: pass`, so the
error is swallowed, every generation counts as failed and the benchmark reports
pass@1 = 0 with exit code 0. The same stdlib check in the UBI images: `python-312`
default `fork`, nested target works; `python-314` default `forkserver`, nested target
raises `PicklingError`.

LiveCodeBench was **not run end to end on any Python version** (the stub model returns
a letter, not code), so this is shown at the function level, not in a full job. Use
3.12 or earlier, or set the start method to `fork` in the wrapper and test LiveCodeBench
against a model that returns code, before moving to 3.14.

Not tested: `linux/amd64` and the other architectures (the build host is arm64; the
x86_64 tree differs in its torch CUDA packages), and `ubi10` images (the paths tried
returned 404).

The image is large: 14.4 GB on arm64, mostly torch's CUDA wheels. The previous image
was not measured for comparison.

## Stale transitive versions

Fresh resolves pick fixed versions of these. A build that reuses cached layers or an
older lock can resolve to the vulnerable ones, which is why a floor or lock file is
worth having if the Red Hat index allows it:

| Package | Vulnerable version seen | Advisories | Fixed in |
|---|---|---|---|
| aiohttp | 3.14.3 | 8 (CVE-2026-111063…111066, 111748…111750, GHSA-2g87-pp6c-29x5) | 3.14.4 |
| gitpython | 3.1.59 | 4 (PYSEC-2026-3982…3984, CVE-2026-114145) | 3.1.60 / 3.2.1 |
| filelock | 3.32.4 | 3 (see above) | 4.2.1 |
| certifi | 2026.7.22 | PYSEC-2026-4320 | 2026.10.18 |
| fsspec | 2025.9.0 | PYSEC-2026-4217 | 2026.6.0 |
| multidict | 6.7.1 | PYSEC-2026-4244 | 6.9.1 |
| opentelemetry-api | 1.44.0 | CVE-2026-114603 | 1.45.1 |
| setuptools | 84.0.0 | GHSA-xmj3-9gf7-h52w | 84.1.0 |
| accelerate | 1.14.0 | PYSEC-2026-3804 (no fix listed then) | not flagged at 1.15.0 |

## What was tested

Python 3.12, lighteval 0.13.0, litellm 1.103.9, nltk 3.10.4, eval-hub-sdk 1.0.6 (the original
run, before the pin moved to 1.102.2; the unit suite was rerun on 1.102.2, see below; the same
cases also passed with lighteval `7fa19dc` on Python 3.11 to 3.14):

- Unit tests, including the comparison of per-choice log-likelihoods with lighteval's
  `TransformersModel`: 115 passed.
- End-to-end through the real lighteval CLI against a local endpoint serving a tiny
  model: `arc:easy`, `glue:cola`, `winogrande`, `commonsense_reasoning`,
  `truthfulness`, `physical_commonsense`, `math`, `hellaswag` and a one-shot
  `arc:easy` with a system prompt. Scores come from a random-weight model and only
  show that the plumbing works.

After the pin moved to litellm 1.102.2 these were run (2026-10-28):

- The unit suite on the host: 129 passed, 1 skipped, 4 xfailed.
- Every entry in `requirements.txt` resolves on the Red Hat index that
  `verify-requirements.yml` uses, including `litellm[caching]==1.102.2`.
- `pip install -r requirements.lock` on `registry.access.redhat.com/ubi9/python-312`
  installed `litellm-1.102.2` and `nltk-3.10.4` with no errors. The image build then
  stopped because the build machine's disk was full, so no image was built from the new
  lock.
- A 3-problem `lcb:codegeneration_v6` job through the adapter in a Linux container with
  litellm 1.102.2 swapped into the earlier image, against a vLLM endpoint: it completed
  (exit 0, 1 of 3 problems passed, no OOM kill). On a macOS host the same generation
  requests also succeeded through litellm 1.102.2, but lighteval's code grader does not
  run on macOS, so that run gave no score.

The other end-to-end CLI cases above were not repeated.

Not tested: a real RHOAI cluster, Anthropic, `linux/amd64` builds, and an image built
from the new lock.

The CI check only looks at `requirements.txt`. A rough name-and-version match of all 180
pins in `requirements.lock` against the same index found 29 it could not match: 18
packages it does not list (the NVIDIA CUDA libraries, `cuda-bindings`, `cuda-pathfinder`
and `cuda-toolkit`, as expected on a CPU index) and 11 versions it lacks, for example
`transformers==5.21.0`, `starlette==1.10.0` and `inspect-ai==0.3.285`. Local-version
builds such as a `+cpu` torch can make that match miss, so treat it as a hint. The
`Containerfile` installs the lock from the default index, so a build from this repository
is not affected; a downstream build that installs the lock from the Red Hat index may be.
