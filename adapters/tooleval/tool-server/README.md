# StableToolBench virtual API (cache-only)

Vendored from [THUNLP-MT/StableToolBench](https://github.com/THUNLP-MT/StableToolBench)
(Apache-2.0). See `PIN.txt` for the pinned commit.

## Endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Liveness/readiness |
| GET | `/tools` | Discover tools under `tools_folder` (`category`, `tool_name`, `api_list`) |
| POST | `/virtual` | StableToolBench virtual tool call |

`config.yml` sets `cache_only: true`. Cache misses return `error=cache_miss`
and never call RapidAPI or OpenAI.

## Data

- `data/fixtures/` — tiny offline tools + one cached response (default image)
- Full cache: build with `FETCH_FULL_CACHE=1` (tag `:0.3.0-full`; downloads HF
  `stabletoolbench/Cache`) or mount tools/cache and set `TOOLS_FOLDER` /
  `CACHE_FOLDER`

## Local smoke

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
STABLETOOLBENCH_CONFIG=./config.yml PYTHONPATH=vendor .venv/bin/uvicorn main:app --port 18080
```
