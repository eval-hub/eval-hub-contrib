"""Run inside the built image as a non-root user with network access disabled."""

import sys
from pathlib import Path

import nltk


def unexpected_download(*args, **kwargs):
    raise AssertionError("IFBench attempted to download NLTK data at runtime")


nltk.download = unexpected_download
sys.path.insert(0, "/app")

import main  # noqa: E402,F401 — exercise the actual adapter import path
from ifbench import data_path, instructions_util  # noqa: E402

from _evaluation import (  # noqa: E402
    InputExample,
    evaluate_instruction_following_strict,
    read_prompt_list,
)

assert instructions_util.split_into_sentences("Hello world. Goodbye world.") == [
    "Hello world.", "Goodbye world.",
]
assert instructions_util.count_stopwords("the cat and the dog") == 3
assert nltk.pos_tag(["cat"])[0][0] == "cat"
prompts = read_prompt_list(data_path())
assert prompts and all(example.prompt and example.instruction_id_list for example in prompts)

example = InputExample(
    key="container-smoke",
    instruction_id_list=["keywords:existence"],
    prompt="Include zebra.",
    kwargs=[{"keywords": ["zebra"]}],
)
assert evaluate_instruction_following_strict(
    example, {example.prompt: "The zebra is here."}
).follow_all_instructions
assert not evaluate_instruction_following_strict(
    example, {example.prompt: "The giraffe is here."}
).follow_all_instructions
assert Path(nltk.data.find("tokenizers/punkt_tab")).is_dir()
print(f"IFBench imports, {len(prompts)} bundled prompts, NLTK resources, and scoring passed offline")
