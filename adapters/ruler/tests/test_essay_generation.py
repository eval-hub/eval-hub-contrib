"""Exercise the real essay generator with an offline Hugging Face tokenizer.

NLTK data must be provisioned as in the Containerfile (or make test-ruler).
No tokenizer, sentence splitting, generator, or subprocess calls are mocked.
"""
import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from main import RulerAdapter
from tokenizer import HFTokenizer


def test_real_essay_generation_within_token_budget(tmp_path):
    # Small deterministic prose fixture replaces the network-downloaded corpus,
    # retaining sentence boundaries and the essay repetition/insertion path.
    essay = (
        "An author started writing essays about programming. "
        "She described the lessons learned from building a small company. "
        "Her friends read each draft and suggested clearer explanations. "
        "The next essay explored how people develop new ideas. "
    ) * 100
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "PaulGrahamEssays.json").write_text(json.dumps({"text": essay}))

    # Use a real local byte-level HF tokenizer; no Hub access or model weights.
    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator([essay], trainers.BpeTrainer(
        vocab_size=512, special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ))
    tokenizer_path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]").save_pretrained(tokenizer_path)
    tokenizer = HFTokenizer(str(tokenizer_path))

    adapter = RulerAdapter.__new__(RulerAdapter)
    output = adapter._generate_task_data(
        "niah_single_2", 4096, tmp_path, str(tokenizer_path), "hf", "base", 2, 42,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row["input"], str) and row["input"]
        assert isinstance(row["answer_prefix"], str) and row["answer_prefix"]
        assert len(row["outputs"]) == 1
        assert isinstance(row["outputs"][0], str) and row["outputs"][0]
        assert row["outputs"][0] in row["input"]
        assert "An author started writing essays" in row["input"]
        length = len(tokenizer.text_to_tokens(row["input"] + row["answer_prefix"])) + 128
        assert length == row["length"] == row["length_w_model_temp"]
        assert 128 < length <= 4096
