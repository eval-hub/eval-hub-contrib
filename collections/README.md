# EvalHub Collections

Community-contributed evaluation collections for [eval-hub](https://github.com/eval-hub/eval-hub).

## Overview

This directory contains pre-defined evaluation collections that group related benchmarks with scoring weights, pass criteria, and metadata. Collections are consumed directly by the eval-hub service and can be referenced by `id` in API calls or SDK requests.

## Available Collections

| ID | Name | Category | Benchmarks | Pass Threshold |
|----|------|----------|------------|---------------|
| [`korean-comprehensive-eval`](korean-comprehensive-eval.yaml) | Korean Comprehensive Evaluation | `language-korean` | `kmmlu_direct_law`, `kobest_wic`, `arc_easy` | 0.50 |
| [`kmmlu-fewshot-comparison`](kmmlu-fewshot-comparison.yaml) | KMMLU Law — Few-Shot Comparison | `language-korean` | `kmmlu_direct_law` (5-shot) | 0.45 |

## Contributing

1. Create a YAML file in this directory following the schema in [eval-hub/config/collections/](https://github.com/eval-hub/eval-hub/tree/main/config/collections).
2. Add a row to the table in this README.
3. Open a pull request with a commit message following Conventional Commits:
   ```
   feat(collections): add <your-collection-id>
   ```
