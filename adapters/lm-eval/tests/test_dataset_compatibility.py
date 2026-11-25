"""Exercise real dataset loading and scoring without a model server or Hub access."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytestmark = pytest.mark.lmeval


@pytest.mark.parametrize("file_format", ["json", "csv", "parquet"])
@pytest.mark.parametrize("output_type", ["multiple_choice", "generate_until"])
def test_native_data_files_evaluate(
    file_format, output_type, tmp_path, monkeypatch, lmeval
):
    from datasets import Dataset
    from lm_eval.api.model import LM

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    class FixedAnswers(LM):
        def loglikelihood(self, requests, **kwargs):
            return [
                (0.0 if request.args[1].strip() == "yes" else -10.0, True)
                for request in requests
            ]

        def loglikelihood_rolling(self, requests, **kwargs):
            return [0.0 for _ in requests]

        def generate_until(self, requests, **kwargs):
            return ["yes" for _ in requests]

    data = Dataset.from_list([{"question": "Reply yes.", "answer": "yes", "label": 1}])
    path = tmp_path / f"data.{file_format}"
    getattr(data, f"to_{file_format}")(str(path))
    task_name = f"native_{file_format}_{output_type}"
    task = {
        "task": task_name,
        "dataset_path": file_format,
        "dataset_kwargs": {
            "data_files": {"test": str(path)},
            "cache_dir": str(tmp_path / "cache"),
        },
        "test_split": "test",
        "output_type": output_type,
        "doc_to_text": "{{question}}",
        "doc_to_target": "label" if output_type == "multiple_choice" else "answer",
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
        ],
    }
    if output_type == "multiple_choice":
        task["doc_to_choice"] = ["no", "yes"]
    else:
        task["process_results"] = lambda doc, answers: {
            "acc": float(answers[0] == doc["answer"])
        }
    result = lmeval.simple_evaluate(
        model=FixedAnswers(),
        tasks=[task],
        num_fewshot=0,
        limit=1,
        bootstrap_iters=0,
        log_samples=True,
    )
    assert result["results"][task_name]["acc,none"] == 1.0
    assert len(result["samples"][task_name]) == 1


@pytest.mark.parametrize("trust", [False, True])
def test_dataset_scripts_cannot_be_enabled(trust, tmp_path, lmeval):
    from datasets import load_dataset

    script = tmp_path / "script.py"
    script.write_text("raise AssertionError('dataset script must not execute')\n")
    with pytest.raises(RuntimeError, match="Dataset scripts are no longer supported"):
        load_dataset(str(script), trust_remote_code=trust)


@pytest.mark.parametrize("trust", [False, True])
def test_tokenizer_remote_code_option_is_separate(trust, tmp_path, monkeypatch, lmeval):
    from lm_eval.api.registry import get_model
    from lm_eval.config.evaluate_config import EvaluatorConfig
    from tokenizers import Tokenizer, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[UNK]": 0, "yes": 1, "[EOS]": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(tmp_path)
    config = EvaluatorConfig(
        model="local-completions",
        model_args={
            "model": "example/model",
            "base_url": "http://127.0.0.1:1/v1/completions",
            "tokenizer": str(tmp_path),
            "tokenizer_backend": "huggingface",
        },
        trust_remote_code=trust,
    )
    config._set_trust_remote_code()
    with patch.object(
        AutoTokenizer, "from_pretrained", wraps=AutoTokenizer.from_pretrained
    ) as load:
        get_model("local-completions").create_from_arg_obj(config.model_args)
    assert load.call_args.kwargs["trust_remote_code"] is trust


def test_catalog_retains_other_benchmarks():
    provider = yaml.safe_load((Path(__file__).parents[1] / "provider.yaml").read_text())
    identifiers = {benchmark["id"] for benchmark in provider["benchmarks"]}
    assert "cruxeval_output" in identifiers
    assert identifiers.isdisjoint(
        {
            "aclue_ancient_chinese_culture",
            "cmmlu_college_mathematics",
            "cmmlu_high_school_mathematics",
            "cmmlu_professional_medicine",
            "cmmlu_traditional_chinese_medicine",
            "ethics_cm",
            "qasper_freeform",
            "scrolls_qasper",
        }
    )
