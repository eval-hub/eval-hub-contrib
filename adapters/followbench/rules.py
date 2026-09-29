"""FollowBench rule-based constraint checks."""

from __future__ import annotations

import ast
import re
import string


RULE_BASED_SOURCES = {
    "E2E",
    "WIKIEVENTS",
    "CONLL2003",
    "text_editing",
    "cnn_dailymail",
    "xsum",
    "samsum",
    "gigaword",
    "arxiv",
    "BBH_logical",
    "BBH_time",
    "self_made_space",
    "gsm_8k",
}


def contain_word(generation: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", generation) is not None


def count_sentences(generation: str) -> int:
    sentences = re.split(r"[.!?]", generation)
    return len([sentence for sentence in sentences if sentence.strip()])


def count_sentence_words_less(generation: str, limit: int) -> bool:
    sentences = re.split(r"[.!?]", generation)
    return all(len(sentence.split()) < limit for sentence in sentences if sentence)


def count_sentence_words_more(generation: str, limit: int) -> bool:
    sentences = re.split(r"[.!?]", generation)
    return all(len(sentence.split()) > limit for sentence in sentences if sentence)


def sentence_contains_word(generation: str, number: int, word: str) -> bool:
    sentences = [
        sentence for sentence in re.split(r"[.!?]", generation) if sentence.strip()
    ]

    if len(sentences) < number:
        return False

    return contain_word(sentences[number - 1], word)


def sentence_is_present_perfect(generation: str, number: int) -> bool:
    sentences = [
        sentence for sentence in re.split(r"[.!?]", generation) if sentence.strip()
    ]

    if len(sentences) < number:
        return False

    return re.search(r"\b(has|have)\s+\w+", sentences[number - 1]) is not None


def is_present_continuous(generation: str) -> bool:
    return re.search(r"\b(am|is|are)\s+\w+ing\b", generation) is not None


def paragraphs_start_with_number_marker(generation: str) -> bool:
    paragraphs = generation.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.lstrip()

        if paragraph and re.match(r"\d+\.", paragraph) is None:
            return False

    return True


def check_procrastination_variation(generation: str) -> bool:
    return re.search(r"procrast", generation, re.IGNORECASE) is None


def rule_evaluation_E2E(generation: str, target: str, level: int) -> bool:
    return generation == target


def rule_evaluation_WIKIEVENTS(
    generation: str,
    target: str,
    level: int,
) -> bool:
    generation_lines = generation.splitlines()
    target_lines = target.splitlines()
    return all(line in target_lines for line in generation_lines)


def rule_evaluation_CONLL2003(
    generation: str,
    target: str,
    level: int,
) -> bool:
    try:
        return generation in ast.literal_eval(target)
    except (SyntaxError, ValueError):
        return False


def rule_evaluation_text_editing(
    generation: str,
    target: str,
    level: int,
) -> bool:
    return generation == target


def rule_evaluation_cnn_dailymail(
    generation: str,
    target: str,
    level: int,
) -> bool:
    valid = count_sentences(generation) == 3
    valid = valid and count_sentence_words_less(generation, 15)

    if level >= 3:
        valid = valid and sentence_contains_word(generation, 1, "Potter")
        valid = valid and sentence_contains_word(generation, 2, "actor")
        valid = valid and sentence_contains_word(generation, 3, "films")

    if level >= 4:
        valid = valid and not sentence_contains_word(generation, 2, "lavish")

    if level >= 5:
        valid = valid and sentence_is_present_perfect(generation, 3)

    return valid


def rule_evaluation_xsum(
    generation: str,
    target: str,
    level: int,
) -> bool:
    valid = count_sentences(generation) == 1
    valid = valid and len(generation.split()) < 20

    if level >= 3:
        valid = valid and not contain_word(generation, "Newton Stewart")

    if level >= 4:
        valid = valid and is_present_continuous(generation)

    if level >= 5:
        valid = valid and contain_word(generation, "operation")

    return valid


def rule_evaluation_samsum(
    generation: str,
    target: str,
    level: int,
) -> bool:
    valid = count_sentences(generation) == 1
    valid = valid and len(generation.split()) < 15

    if level >= 3:
        valid = valid and contain_word(generation, "stuff")

    if level >= 4:
        valid = valid and check_procrastination_variation(generation)

    if level >= 5:
        valid = valid and sum(
            character in string.punctuation for character in generation
        ) == 1

    return valid


def rule_evaluation_gigaword(
    generation: str,
    target: str,
    level: int,
) -> bool:
    valid = len(generation.split()) == 8

    if level >= 2:
        valid = valid and not any(
            character in string.punctuation for character in generation
        )

    if level >= 3:
        valid = valid and generation.islower()

    if level >= 4:
        valid = valid and not contain_word(generation, "bus")

    if level >= 5:
        valid = valid and contain_word(generation, "in")

    return valid


def rule_evaluation_arxiv(
    generation: str,
    target: str,
    level: int,
) -> bool:
    valid = count_sentences(generation) == 1
    valid = valid and len(generation.split()) <= 20

    if level >= 3:
        valid = valid and generation.startswith("We")

    if level >= 4:
        valid = valid and contain_word(generation, "activations")

    if level >= 5:
        valid = valid and not contain_word(generation, "transformer")

    return valid


def rule_evaluation_BBH_logical(
    generation: str,
    target: str,
    level: int,
) -> bool:
    matches = re.findall(r"\([A-Z]\)", generation)
    return bool(matches) and matches[-1] == target


def rule_evaluation_BBH_time(
    generation: str,
    target: str,
    level: int,
) -> bool:
    matches = re.findall(r"\d{2}/\d{2}/\d{4}", generation)
    return bool(matches) and matches[-1] == target


def rule_evaluation_self_made_space(
    generation: str,
    target: str,
    level: int,
) -> bool:
    return target in generation


def rule_evaluation_gsm_8k(
    generation: str,
    target: str,
    level: int,
) -> bool:
    matches = re.findall(r"\$\d+", generation)
    return bool(matches) and matches[-1] == target


def rule_evaluation_format(
    generation: str,
    example_id: int,
    level: int,
) -> bool:
    if example_id == 22:
        paragraphs = generation.split("\n\n")

        if level <= 2:
            return len(paragraphs) == 3 and all(
                count_sentences(paragraph) < 3 for paragraph in paragraphs
            )

        valid = len(paragraphs) == 3
        valid = valid and all(
            count_sentences(paragraph) < 3 for paragraph in paragraphs
        )
        valid = valid and count_sentence_words_more(generation, 20)

        if level >= 4:
            valid = valid and paragraphs_start_with_number_marker(generation)

        if level >= 5:
            valid = valid and generation.endswith("Those are suggestions")

        return valid

    if example_id == 30:
        constraints = [
            "**" in generation,
            len(re.findall(r"\*\*[\d].+?:", generation)) == 5,
        ]

        sentences = re.split(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",
            generation,
        )

        keyword_sentences = [
            sentence
            for sentence in sentences
            if any(f"{number}." in sentence for number in range(1, 6))
        ]

        constraints.append(
            all(len(sentence.split(".")) < 3 for sentence in keyword_sentences)
        )
        constraints.append(
            all(10 <= len(sentence.split()) <= 15 for sentence in sentences)
        )
        constraints.append(
            all(
                not word.endswith("-ly")
                for sentence in sentences
                for word in sentence.split()
            )
        )

        return all(constraints[:level])

    return False


def check_match(target: str, generation: str) -> bool:
    if not generation.strip():
        return False

    pattern = target.replace("{{", "{").replace("}}", "}")
    pattern = pattern.replace("{answer}", ".*")
    pattern = re.escape(pattern).replace(r"\.\*", ".*")

    return re.fullmatch(pattern, generation) is not None


def evaluate_rule_constraint(
    source: str,
    generation: str,
    target: str,
    level: int,
    example_id: int,
) -> bool | None:
    """Evaluate a rule-based constraint, or return None if judge-based."""
    if source in RULE_BASED_SOURCES:
        evaluator = globals()[f"rule_evaluation_{source}"]
        return bool(evaluator(generation, target, level))

    if source == "format" and example_id in {22, 30}:
        return rule_evaluation_format(generation, example_id, level)

    return None


def evaluate_example_constraint(target: str, generation: str) -> bool:
    """Evaluate an example constraint using its answer template."""
    return check_match(target, generation)