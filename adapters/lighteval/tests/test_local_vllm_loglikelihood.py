"""Log-likelihood scoring against a real vLLM server, compared with the reference tokenization.

lighteval's ``TransformersModel`` tokenizes the question and each answer separately and joins
the token ids (``tok_encode_pair(..., pairwise=True)``).  A completions endpoint given text
tokenizes it as one string, which differs where a token merges across the boundary.  The
patch sends token ids instead; this test checks, against vLLM itself and not a mock, that

* the ids it sends are exactly the ones the reference would build with the model's own
  tokenizer, and
* the scores it gets back are close to a float32 forward pass over those ids (vLLM runs the
  model in bfloat16, so the numbers are close but not identical).

Requires an OpenAI-compatible vLLM server on http://localhost:11434 serving
``Qwen/Qwen2.5-0.5B-Instruct`` (the ``test-local-vllm-cpu`` CI job starts one).  Skipped on
servers that do not return ``prompt_token_ids`` (Ollama, older vLLM).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

BASE = "http://localhost:11434/v1"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

#: Largest tolerated difference between vLLM's bfloat16 score of a choice and a float32
#: forward pass over the same token ids, in nats.  The first CI run on vLLM 0.19.1 differed
#: by at most 0.27 across six choices.
SCORE_TOLERANCE = 0.5


def _vllm_with_token_ids() -> bool:
    try:
        resp = requests.post(
            f"{BASE}/completions",
            json={"model": MODEL, "prompt": "Hello", "max_tokens": 0, "echo": True,
                  "logprobs": 1, "return_token_ids": True},
            timeout=30,
        )
        return resp.ok and bool(resp.json()["choices"][0].get("prompt_token_ids"))
    except Exception:
        return False


def _load_patch_module():
    path = Path(__file__).parent.parent / "lighteval_logprob_patch.py"
    spec = importlib.util.spec_from_file_location("_logprob_patch_vllm_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.local
@pytest.mark.openai_endpoint
@pytest.mark.skipif(not _vllm_with_token_ids(), reason="no vLLM serving Qwen2.5-0.5B-Instruct with return_token_ids")
def test_pairwise_scores_match_the_reference_tokenization_on_a_real_vllm(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from lighteval.tasks.requests import Doc

    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.delenv("EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION", raising=False)
    mod = _load_patch_module()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    docs = [
        # A clean split: the usual shape of a multiple-choice prompt.
        Doc(query="Question: What is the capital of France?\nAnswer:", choices=[" Paris", " Rome"], gold_index=0),
        # The context ends in a newline that is moved to the front of each choice.
        Doc(query="Question: What is the capital of Italy?\nAnswer:\n", choices=["Rome", "Paris"], gold_index=0),
        # A token that merges across the boundary when the text is tokenized as one string.
        Doc(query="Compute 2+2=", choices=["4", "=4"], gold_index=0),
    ]

    sent: list[dict] = []
    real_post = mod._post_completion

    def recording_post(req):
        sent.append(json.loads(req.data.decode()))
        return real_post(req)

    monkeypatch.setattr(mod, "_post_completion", recording_post)
    client = SimpleNamespace(base_url=BASE, model=f"openai/{MODEL}", api_key="dummy")

    responses = mod._loglikelihood_via_completions(client, docs)

    merged_somewhere = False
    for doc, resp in zip(docs, responses):
        context = mod._build_context(None, doc)
        stripped = context.rstrip()
        whitespace = context[len(stripped):]
        context_ids = tokenizer.encode(stripped)
        for choice, got, positions in zip(doc.choices, resp.logprobs, resp.output_tokens):
            choice_ids = tokenizer.encode(whitespace + choice, add_special_tokens=False)
            expected_ids = context_ids + choice_ids
            joint_ids = tokenizer.encode(context + choice)
            merged_somewhere |= joint_ids != expected_ids

            # The ids the patch sent to score this choice are the reference's, exactly.
            assert expected_ids in [body["prompt"] for body in sent if isinstance(body["prompt"], list)], (
                f"{doc.query!r} + {choice!r}: expected token ids were never sent"
            )
            assert len(positions) == len(choice_ids)

            ids = torch.tensor([expected_ids])
            with torch.no_grad():
                logp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
            reference = sum(logp[i - 1, ids[0, i]].item() for i in range(len(context_ids), len(expected_ids)))
            print(f"{doc.query!r:70} + {choice!r:8} vllm={got:9.4f} fp32={reference:9.4f} diff={got - reference:+.4f}")
            assert got == pytest.approx(reference, abs=SCORE_TOLERANCE)

    assert merged_somewhere, "no doc merged across the boundary, so the comparison proves nothing about it"
