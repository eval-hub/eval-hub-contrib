# RULER data sources and licenses

This document covers all external datasets used by the RULER generators.
All four inputs are prepared at run time, before task generation, only when
the selected tasks need them. They share the job's temporary `data/assets/`
cache, under `/tmp` by default, passed to generators as `RULER_DATA_DIR`.
Dataset payloads are excluded from the image. This notice is installed at
`/app/DATA_LICENSES.md`.

| Data | Tasks using it | Download and storage |
|---|---|---|
| NVIDIA word list | Common words extraction (CWE) | Job data-preparation startup; temporary cache, excluded from image |
| SQuAD v2.0 development set | SQuAD QA | Job data-preparation startup; temporary cache, excluded from image |
| HotpotQA development distractor set | HotpotQA QA | Job data-preparation startup; temporary cache, excluded from image |
| Paul Graham essay corpus | NIAH with essay haystacks; VT when configured for essays | Job data-preparation startup; temporary cache, excluded from image |

## NVIDIA RULER word list

- File: `english_words.json` in the shared runtime cache.
- Source: [NVIDIA RULER](https://github.com/NVIDIA/RULER), commit
  `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.
- Download: [official Git LFS payload](https://media.githubusercontent.com/media/NVIDIA/RULER/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/json/english_words.json).
- Upstream repository license: [Apache License 2.0](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/LICENSE).
- SHA-256: `affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca`.

## Stanford Question Answering Dataset (SQuAD v2.0)

- File: `squad.json` in the shared runtime cache.
- Dataset: SQuAD v2.0 development set, provided by the
  [SQuAD project](https://rajpurkar.github.io/SQuAD-explorer/).
- Credit: Pranav Rajpurkar, Robin Jia and Percy Liang,
  [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822), ACL 2018.
- Download: [official dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json).
- License: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
  [Legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).
- Changes: the original download is stored as `squad.json`; its contents are
  unchanged. No endorsement by the dataset authors is implied.
- SHA-256: `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`.

## HotpotQA development distractor set

- File: `hotpotqa.json` in the shared runtime cache.
- Dataset: HotpotQA development set in the distractor setting, provided by
  the [HotpotQA project](https://hotpotqa.github.io/).
- Credit: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen,
  Ruslan Salakhutdinov and Christopher D. Manning, EMNLP 2018.
- Download: [revision-pinned mirror](https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json),
  as specified by [NVIDIA RULER's official download script](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/json/download_qa_dataset.sh).
- License: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
  [Legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).
- Changes: the mirror payload is stored as `hotpotqa.json`; its contents are
  unchanged. No endorsement by the dataset authors is implied.
- SHA-256: `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.

## Paul Graham essay haystack (runtime download)

- File: `PaulGrahamEssays.json` in the shared runtime cache.
- Corpus: essays by Paul Graham, using the vendored
  [NVIDIA RULER URL list](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt)
  and [downloader](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/json/download_paulgraham_essay.py).
- Sources: [Paul Graham's website](https://www.paulgraham.com/articles.html)
  and the essay files linked by NVIDIA from
  [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main/needlehaystack/PaulGrahamEssays).
  GitHub file links are fetched directly from `raw.githubusercontent.com`.
- Credit and usage information: Paul Graham. His
  [official FAQ](https://www.paulgraham.com/gfaq.html) expresses a preference
  for linking to essays instead of mirroring them. A general open
  redistribution license for the essay content has not been verified.
  The Apache 2.0 license of the NVIDIA downloader is not assigned to the essays.
- Packaging: the essay corpus is excluded from the image. Downloaded HTML
  is converted to text and concatenated with the downloaded text files using
  the vendored downloader before any task datasets are generated in a job
  that needs essays. The cache is reused within a job and removed with its
  temporary data.
- Integrity: remote essay contents can change; this runtime corpus has no
  fixed build-time checksum. Generated evaluation datasets retain the
  adapter's existing dataset-hash provenance.
