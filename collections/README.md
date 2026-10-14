# EvalHub Collections

Community-contributed evaluation collections for [eval-hub](https://github.com/eval-hub/eval-hub).

## Overview

This directory contains pre-defined evaluation collections that group related benchmarks with scoring weights, pass criteria, and metadata. Collections are consumed directly by the eval-hub service and can be referenced by `id` in API calls or SDK requests.

## Available Collections

| ID | Name | Category | Benchmarks | Pass Threshold |
|----|------|----------|------------|---------------|
| [`korean-comprehensive-eval`](korean-comprehensive-eval.yaml) | Korean Comprehensive Evaluation | `language-korean` | `kmmlu_direct_law`, `kobest_wic`, `arc_easy` | 0.50 |
| [`kmmlu-fewshot-comparison`](kmmlu-fewshot-comparison.yaml) | KMMLU Law — Few-Shot Comparison | `language-korean` | `kmmlu_direct_law` (5-shot) | 0.45 |
| **Combined collections** | | | | |
| [`combined-coding`](combined-coding.yaml) | Combined Coding | `coding` | `inspect/humaneval`⚠️, `inspect/mbpp`⚠️, `inspect/swe-bench`⚠️, `inspect/bigcodebench`⚠️, `lcb:codegeneration_v6` | 0.45 |
| [`combined-knowledge`](combined-knowledge.yaml) | Combined Knowledge | `knowledge` | `inspect/truthfulqa`, `mmlu`, `triviaqa`, `openbookqa`, `glue:cola`, `glue:sst2`, `glue:mrpc` | 0.60 |
| [`combined-reasoning`](combined-reasoning.yaml) | Combined Reasoning | `reasoning` | `inspect/gsm8k`, `inspect/bbh`, `inspect/hellaswag`, `inspect/winogrande`, `aime24`, `aime25`, `math_500`, `math:algebra`, `math:counting_and_probability` | 0.45 |
| [`combined-safety-alignment`](combined-safety-alignment.yaml) | Combined Safety & Alignment | `safety` | 8× Petri⚠️, `inspect/truthfulqa`, `quick`, `owasp_llm_top10`, `avid_ethics`, `quality` | — |
| **Inspect AI collections** ⚠️ G9 | | | | |
| [`inspect-agent`](inspect-agent.yaml) | Inspect Agent Capabilities | `agent` | `inspect/gaia`, `inspect/agentdojo`, `inspect/theagentcompany` | 0.30 |
| [`inspect-alignment`](inspect-alignment.yaml) | Inspect Alignment Audit | `alignment` | 5× Petri (sycophancy, deception, alignment-faking, power-seeking, oversight-subversion) | ≤3.0 |
| [`inspect-coding`](inspect-coding.yaml) | Inspect Coding | `coding` | `inspect/humaneval`, `inspect/mbpp`, `inspect/swe-bench`, `inspect/bigcodebench` | 0.45 |
| [`inspect-cybersecurity`](inspect-cybersecurity.yaml) | Inspect Cybersecurity | `cybersecurity` | `inspect/cybench`, `inspect/cyberseceval-2`, `inspect/cybergym` | 0.30 |
| [`inspect-knowledge`](inspect-knowledge.yaml) | Inspect Knowledge | `knowledge` | `inspect/truthfulqa` | 0.50 |
| [`inspect-reasoning`](inspect-reasoning.yaml) | Inspect Reasoning | `reasoning` | `inspect/gsm8k`, `inspect/bbh`, `inspect/hellaswag`, `inspect/winogrande` | 0.55 |
| [`inspect-safety`](inspect-safety.yaml) | Inspect Safety | `safety` | 4× Petri (jailbreak, harmful-cooperation, self-preservation, power-seeking) | ≤3.0 |
| **Lighteval collections** | | | | |
| [`lighteval-code`](lighteval-code.yaml) | Lighteval Code | `code` | `lcb:codegeneration_v6` | 0.20 |
| [`lighteval-knowledge`](lighteval-knowledge.yaml) | Lighteval Knowledge | `knowledge` | `mmlu`, `triviaqa`, `openbookqa`, `knowledge` | 0.62 |
| [`lighteval-language-understanding`](lighteval-language-understanding.yaml) | Lighteval Language Understanding | `language_understanding` | `glue:cola`, `glue:sst2`, `glue:mrpc`, `language_understanding` | 0.70 |
| [`lighteval-math`](lighteval-math.yaml) | Lighteval Math | `math` | `gsm8k`, `math`, `math:algebra`, `math:counting_and_probability`, `math_500` | 0.55 |
| [`lighteval-reasoning`](lighteval-reasoning.yaml) | Lighteval Reasoning | `reasoning` | `aime24`, `aime25` | 0.10 |
| [`lighteval-safety`](lighteval-safety.yaml) | Lighteval Safety | `safety` | `truthfulqa:mc`, `truthfulqa:generation`, `truthfulness` | 0.50 |
| **Knowledge & Reasoning** | | | | |
| [`knowledge-reasoning-v1`](knowledge-reasoning-v1.yaml) | Knowledge & Reasoning v1 | `knowledge` | `mmlu_cot_llama`, `leaderboard_mmlu_pro`, `leaderboard_bbh`, `leaderboard_musr`, `leaderboard_gpqa`, `truthfulqa_mc1`, `inspect/simpleqa`⚠️, `inspect/winogrande`⚠️, `inspect/hellaswag`⚠️, `inspect/arc`⚠️ | 0.55 |
| **Other providers** | | | | |
| [`garak-red-team`](garak-red-team.yaml) | Garak Red-Team | `security` | `quick`, `owasp_llm_top10`, `avid_security`, `avid_ethics`, `avid_performance`, `quality`, `cwe` | ≤0.30 |
| [`guidellm-perf`](guidellm-perf.yaml) | GuideLLM Performance | `performance` | `quick_perf_test`, `sweep`, `concurrent` | 10.0 OTS |
| [`ragas-rag`](ragas-rag.yaml) | RAGAS RAG Evaluation | `rag_evaluation` | `ragas_rag_default`, `ragas_rag_full` | 0.50 |
| [`ruler-long-context`](ruler-long-context.yaml) | RULER Long-Context | `long_context` | 13× NIAH/VT/Aggregation/QA tasks | ≥7/13 |

## Contributing

1. Create a YAML file in this directory following the schema in [eval-hub/config/collections/](https://github.com/eval-hub/eval-hub/tree/main/config/collections).
2. Add a row to the table in this README.
3. Open a pull request with a commit message following Conventional Commits:
   ```
   feat(collections): add <your-collection-id>
   ```
