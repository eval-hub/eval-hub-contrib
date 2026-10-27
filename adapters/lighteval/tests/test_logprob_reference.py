"""Compare the loglikelihood patch against lighteval's reference backend.

The patch scores multiple-choice docs through ``/v1/completions`` with
``echo=True``.  lighteval's own backends (``TransformersModel``, ``VLLMModel``)
score the same docs locally.  For the same model and prompt settings both must
return the same per-choice log-likelihoods, otherwise the benchmark scores of
the two paths are not comparable (RHOAIENG-84704, PR review of 2026-10-26).

The test needs no network: a tiny randomly initialised GPT-2 with a tokenizer
trained on a few sentences is saved to a temporary directory, loaded by the
reference backend, and also served by an in-process ``/v1/completions`` echo
endpoint that mimics vLLM (joint tokenisation of the prompt, ``text_offset`` per
echoed token, ``None`` for the first token).
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("lighteval")

from lighteval.tasks.requests import Doc  # noqa: E402

SYSTEM_PROMPT = "You are terse."

CORPUS = [
    "Question: What is the capital of France? Answer: Paris is the capital.",
    "Question: What is the capital of Italy? Answer: Rome is the capital.",
    "Instruction: choose the best answer. Answer: yes no maybe A B C D.",
] * 20


def _load_patch_module():
    patch_path = Path(__file__).parent.parent / "lighteval_logprob_patch.py"
    spec = importlib.util.spec_from_file_location("_logprob_patch_reference", patch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", params=[False, True], ids=["no-bos", "bos"])
def tiny_model_dir(request, tmp_path_factory):
    """A tiny GPT-2; with ``bos`` its tokenizer prepends a BOS token, as Llama and
    Mistral tokenizers do and vLLM's completions endpoint keeps by default."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    if request.param:
        from tokenizers.processors import TemplateProcessing

        tok.post_processor = TemplateProcessing(
            single="<|endoftext|> $A",
            special_tokens=[("<|endoftext|>", tok.token_to_id("<|endoftext|>"))],
        )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        bos_token="<|endoftext|>",
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(fast),
            n_layer=2,
            n_head=2,
            n_embd=32,
            n_positions=512,
            bos_token_id=0,
            eos_token_id=0,
        )
    )
    directory = tmp_path_factory.mktemp("tiny-gpt2")
    model.save_pretrained(directory)
    fast.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def echo_server(tiny_model_dir):
    """A vLLM-like ``/v1/completions`` endpoint backed by the tiny model."""
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    model = GPT2LMHeadModel.from_pretrained(tiny_model_dir, torch_dtype=torch.float32).eval()
    prompts: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompts.append(body["prompt"])
            # Like vLLM: special tokens are kept, every token is decoded on its own and
            # ``text_offset`` is the running length of those decoded strings. With a BOS
            # token that makes every offset larger than the character position in the
            # prompt by len(decoded BOS).
            # vLLM accepts text or token ids; ``add_special_tokens`` applies to text only.
            if isinstance(body["prompt"], list):
                token_ids = list(body["prompt"])
            else:
                token_ids = tokenizer(
                    body["prompt"], add_special_tokens=body.get("add_special_tokens", True)
                )["input_ids"]
            ids = torch.tensor([token_ids])
            with torch.no_grad():
                logp = torch.log_softmax(model(ids).logits[0], dim=-1)
            token_logprobs = [None] + [
                logp[i - 1, ids[0, i]].item() for i in range(1, ids.shape[1])
            ]
            pieces = [tokenizer.decode([int(t)]) for t in ids[0]]
            offsets, running = [], 0
            for piece in pieces:
                offsets.append(running)
                running += len(piece)
            choice = {
                "logprobs": {
                    "tokens": pieces,
                    "token_logprobs": token_logprobs,
                    "text_offset": offsets,
                }
            }
            if body.get("return_token_ids"):
                choice["prompt_token_ids"] = token_ids
            payload = {"choices": [choice]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", prompts
    server.shutdown()


@pytest.fixture(scope="module")
def reference_model(tiny_model_dir, tmp_path_factory):
    from lighteval.models.transformers.transformers_model import (
        TransformersModel,
        TransformersModelConfig,
    )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return TransformersModel(
        TransformersModelConfig(
            model_name=str(tiny_model_dir),
            batch_size=2,
            model_parallel=False,
            dtype="float32",
            device="cpu",
            system_prompt=SYSTEM_PROMPT,
            # lighteval caches results per (task_name, doc.id) on disk.  Keep the
            # cache out of the user's real one so stale entries cannot be served.
            cache_dir=str(tmp_path_factory.mktemp("lighteval-cache")),
        )
    )


@pytest.fixture
def patched_client(echo_server, monkeypatch, tmp_path):
    from lighteval.models.endpoints.litellm_model import (
        LiteLLMClient,
        LiteLLMModelConfig,
    )

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    base_url, _ = echo_server
    client = LiteLLMClient(
        LiteLLMModelConfig(
            model_name="openai/tiny",
            base_url=base_url,
            api_key="k",
            system_prompt=SYSTEM_PROMPT,
            cache_dir=str(tmp_path / "lighteval-cache"),
        )
    )
    mod = _load_patch_module()
    monkeypatch.setattr(LiteLLMClient, "loglikelihood", mod._loglikelihood_via_completions)
    return client


def _fewshot(query: str, choices: list[str], gold: int) -> Doc:
    return Doc(query=query, choices=choices, gold_index=gold)


QUESTION = "Question: What is the capital of France? Answer:"


def _docs() -> dict[str, Doc]:
    docs = _build_docs()
    for name, doc in docs.items():
        doc.id, doc.task_name = name, f"reference-{name}"
    return docs


def _build_docs() -> dict[str, Doc]:
    return {
        "bare-query": Doc(query=QUESTION, choices=[" Paris", " Rome"], gold_index=0),
        "context-ends-in-space": Doc(
            query=QUESTION + " ", choices=["Paris", "Rome", "yes"], gold_index=0
        ),
        "context-ends-in-newline": Doc(
            query="Question: What is the capital of Italy?\nAnswer:\n",
            choices=["Rome", "Paris"],
            gold_index=0,
        ),
        # A token spans the context/choice boundary.  The reference slices the joint
        # encoding at len(tokens(context)), so which tokens count as the answer depends
        # on how the tokenizer merges, not on where the characters are split.
        "token-merges-across-boundary": Doc(
            query="Question: Wh", choices=["at is the capital?", "o is it?"], gold_index=0
        ),
        # The joint encoding is as long as the context's own: the reference keeps the
        # last token as the answer instead of an empty continuation.
        "merge-keeps-token-count": Doc(
            query="Question: What is the capital of Italy? Answer: Rome is the ca",
            choices=["pital.", "rrot."],
            gold_index=0,
        ),
        "instruction-fewshot-system": Doc(
            query="Question: What is the capital of Italy? Answer:",
            choices=[" Rome", " Paris", " yes"],
            gold_index=0,
            instruction="Instruction: choose the best answer.\n",
            fewshot_samples=[
                _fewshot(
                    "Question: What is the capital of France? Answer:",
                    [" Paris", " Rome"],
                    0,
                ),
                _fewshot(
                    "Question: What is the capital of Italy? Answer:",
                    [" Rome", " Paris"],
                    0,
                ),
            ],
        ),
    }


_MERGE_DOCS = ("token-merges-across-boundary", "merge-keeps-token-count")


@pytest.mark.parametrize("name", list(_docs()))
def test_patch_matches_reference_backend(name, reference_model, patched_client):
    doc = _docs()[name]

    (ref,) = reference_model.loglikelihood([doc])
    (got,) = patched_client.loglikelihood([doc])

    assert got.logprobs == pytest.approx(ref.logprobs, abs=1e-3), name
    # Same preferred choice, so LoglikelihoodAcc gives the same verdict.
    assert max(range(len(got.logprobs)), key=got.logprobs.__getitem__) == max(
        range(len(ref.logprobs)), key=ref.logprobs.__getitem__
    )
    # Token-normalised metrics divide by these counts.  The reference pads each
    # choice's token list with -1 up to the longest continuation in the batch, so
    # only its non-negative entries are real scored tokens.
    assert [len(t) for t in got.output_tokens] == [
        sum(1 for token in t if token >= 0) for t in ref.output_tokens
    ]


def test_reference_comparison_is_sensitive_to_prompt_construction(
    reference_model, patched_client, echo_server
):
    """Guard against a vacuous comparison: the few-shot prefix must change scores.

    If it did not, ``test_patch_matches_reference_backend`` would also pass for a
    patch that scores ``doc.query`` alone.
    """
    full = _docs()["instruction-fewshot-system"]
    query_only = Doc(
        query=full.query,
        choices=full.choices,
        gold_index=0,
        id="query-only",
        task_name="reference-query-only",
    )

    (with_prompt,) = reference_model.loglikelihood([full])
    (without,) = reference_model.loglikelihood([query_only])
    assert with_prompt.logprobs != pytest.approx(without.logprobs, abs=1e-2)

    _, prompts = echo_server
    prompts.clear()
    patched_client.loglikelihood([full])
    sent = prompts[0]
    assert sent.startswith(SYSTEM_PROMPT)
    assert "Instruction: choose the best answer." in sent
    assert sent.count("Answer:") == 3  # two few-shot examples and the query


@pytest.mark.parametrize("name", _MERGE_DOCS)
def test_joint_tokenization_differs_where_a_token_merges_across_the_boundary(
    name, reference_model, patched_client, monkeypatch
):
    """The opt-out scores the jointly tokenized text, which is not what the transformers
    backend scores when a token merges across the boundary.  This keeps the comparison
    above from passing for the wrong reason: if joint scoring matched here too, the merge
    docs would not be testing separate tokenization at all."""
    monkeypatch.setenv("EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION", "joint")
    doc = _docs()[name]

    (ref,) = reference_model.loglikelihood([doc])
    (got,) = patched_client.loglikelihood([doc])

    assert got.logprobs != pytest.approx(ref.logprobs, abs=1e-3)
