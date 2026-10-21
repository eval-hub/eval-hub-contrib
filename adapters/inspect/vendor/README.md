# Vendored dependencies

Two Inspect adapter dependencies are not published on PyPI and have no AIPCC
curated-index ingestion path, so they cannot be pinned in any requirements
file that Hermeto resolves. They are vendored here as pristine source trees
and installed from the image build:

| Package | Upstream | Vendored at | Version | License |
|---|---|---|---|---|
| `evals` | https://github.com/gsma-labs/evals | `708f1c5498793536824eae9919ae003f13c910ea` (tag `v1.1.0`, identical tree) | 1.1.0 | MIT |
| `instruction_following_eval` | https://github.com/josejg/instruction_following_eval | `0c495b2f95155e8b10acb919ae283bfb4d5be6e2` | 0.1.0 | Apache-2.0 |

`evals` registers the GSMA Open-Telco Inspect tasks (`evals/telemath`,
`evals/teleqna`, `evals/telelogs`, `evals/three_gpp`, …) through its
`[project.entry-points.inspect_ai]` hook; `instruction_following_eval` is
imported by inspect-evals' ifeval task.

## Local modifications (deliberate, exhaustive)

The trees are byte-identical to the upstream archives except for:

- `evals/PKG-INFO` — **added**. The upstream source archive is not an sdist
  and carries no version metadata; built from a git-less directory,
  setuptools-scm falls back to `fallback_version = "0.0.0+unknown"`. The
  added two-line sdist stub (`Name: evals`, `Version: 1.1.0`) makes every
  build path resolve the deterministic, meaningful version `1.1.0`.
- `evals/uv.lock` — **removed**. Upstream's development-environment lockfile
  is not used when the package is installed, and the repo's Trivy filesystem
  scan would otherwise audit upstream's dev pins, which are unrelated to the
  adapter image.

`instruction_following_eval` carries a static `version = "0.1.0"` in its
pyproject.toml and needs no metadata stub.

Nothing else differs. The trees are bundled in this repository — the image
build never fetches them from the network. To independently verify the
bundled trees against upstream (optional, run on a connected workstation;
not part of any build):

```sh
curl -sL -o /tmp/evals.tar.gz \
  https://github.com/gsma-labs/evals/archive/708f1c5498793536824eae9919ae003f13c910ea.tar.gz
echo "644ec6f7eb32706a591d972321461e0e761b1e5b4f65120738383060d3c0ca5c  /tmp/evals.tar.gz" | sha256sum -c
mkdir -p /tmp/evals-verify && tar xzf /tmp/evals.tar.gz -C /tmp/evals-verify
diff -r /tmp/evals-verify/evals-708f1c5498793536824eae9919ae003f13c910ea evals

curl -sL -o /tmp/ifeval.tar.gz \
  https://github.com/josejg/instruction_following_eval/archive/0c495b2f95155e8b10acb919ae283bfb4d5be6e2.tar.gz
echo "ce25f3bd2c29cefea147096f2520494f47e59220107b262cc0f54907499cc9df  /tmp/ifeval.tar.gz" | sha256sum -c
mkdir -p /tmp/ifeval-verify && tar xzf /tmp/ifeval.tar.gz -C /tmp/ifeval-verify
diff -r /tmp/ifeval-verify/instruction_following_eval-0c495b2f95155e8b10acb919ae283bfb4d5be6e2 instruction_following_eval
```

## How they are installed

- Community build (`Containerfile`) and the dev/test flow (`make test-inspect`,
  `uv pip install -r requirements.txt`): referenced as relative paths
  (`./vendor/evals`, `./vendor/instruction_following_eval`) directly in
  `requirements.txt`, so their transitive dependencies resolve under the same
  `constraints.txt` pins as everything else.
- Hermetic Konflux build (`Dockerfile.konflux`): Hermeto only parses
  requirements files, which cannot carry local paths, so the vendored trees
  are installed by a separate `pip install --no-deps` step after
  `requirements-rhoai.txt`. `--no-deps` is correct because every transitive
  dependency of both packages is already pinned in `requirements-rhoai.txt`
  from the AIPCC curated index (the pins annotated `via evals` /
  `via instruction-following-eval` in that file) and installed by the
  preceding step; pip's PEP 517 build isolation finds the build tools
  (setuptools, setuptools-scm, wheel — see `requirements-rhoai-build.txt`)
  in Hermeto's prefetched store via the injected `PIP_FIND_LINKS`.

## Updating a vendored dependency

1. Pick the new upstream commit (full sha).
2. Replace the tree under `vendor/<name>/` with the extracted archive,
   re-apply the modifications listed above (drop `uv.lock`, refresh
   `PKG-INFO` version for `evals`).
3. Diff the declared `dependencies` in the vendored `pyproject.toml` against
   `requirements-rhoai.txt` / `constraints.txt`: every dependency must stay
   pinned, and new dependencies must be added to both pin files (for the
   downstream lockfile, from the AIPCC curated index).
4. Update the table (commit, version) and the verification sha256s above.
