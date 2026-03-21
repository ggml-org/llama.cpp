"""
Test for sampler state save/restore with speculative checkpoints.

Bug: restore_checkpoint() restores KV and recurrent state but not the sampler.
Ghost tokens from rejected drafts remain in the sampler's prev ring buffer,
grammar state, and penalty chains. With grammar constraints, rejected
speculation advances the grammar past the rollback point, producing invalid output.

Requires Qwen3.5-0.8B (hybrid model).
"""

import json
import os
import pytest
from utils import ServerProcess

QWEN35_08B = os.environ.get(
    "QWEN35_08B_MODEL",
    os.path.expanduser("~/Models/Qwen3.5-0.8B-BF16.gguf"),
)

requires_qwen35_08b = pytest.mark.skipif(
    not os.path.exists(QWEN35_08B),
    reason="Requires local Qwen3.5-0.8B model",
)


@pytest.fixture(scope="module", autouse=True)
def do_something():
    """Override conftest's load_all — we use local models, not HF presets."""
    pass


WARMUP_PROMPTS = [
    "Write a quicksort implementation in Python with detailed comments.",
    "Write a mergesort implementation in Python with detailed comments.",
    "Write a heapsort implementation in Python with detailed comments.",
]


def _warmup_ngram(server, n_predict=128):
    """Send diverse prompts to build ngram data for speculative drafting."""
    for prompt in WARMUP_PROMPTS:
        server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "temperature": 0.0, "top_k": 1, "n_predict": n_predict,
        })


@requires_qwen35_08b
def test_grammar_output_valid_after_checkpoint_restore():
    """JSON grammar output must be valid after checkpoint restore.

    If grammar state is not restored with the checkpoint, rejected speculation
    advances the grammar past the rollback point, producing invalid JSON.
    Uses f16 V cache to isolate the sampler bug from V cache divergence.
    Uses --draft-p-min 0.99 to force rejections and exercise the restore path.
    """
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = QWEN35_08B
    server.n_ctx = 2048
    server.n_slots = 1
    server.n_gpu_layer = 99
    server.seed = 3407
    server.temperature = 0.0
    server.n_predict = 256
    server.draft_min = 4
    server.draft_max = 16
    server.ctk = "f16"
    server.ctv = "f16"
    server.fa = "on"
    server.reasoning = "off"
    server.extra_args = [
        "--spec-type", "ngram-mod",
        "--spec-use-checkpoints", "on",
        "--ctx-checkpoints", "4",
        "--draft-p-min", "0.99",
        "--no-cache-prompt",
    ]
    server.start(timeout_seconds=120)

    # prime ngram model so drafts are generated and rejected
    _warmup_ngram(server)

    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "age", "hobbies"],
    }

    res = server.make_request("POST", "/completion", data={
        "prompt": (
            "Generate a JSON object describing a person named Alice "
            "who is 30 years old and likes reading and hiking."
        ),
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "json_schema": json_schema,
    })
    assert res.status_code == 200

    content = res.body["content"]
    parsed = json.loads(content)
    assert "name" in parsed
    assert "age" in parsed
    assert "hobbies" in parsed
    assert isinstance(parsed["hobbies"], list)


@requires_qwen35_08b
def test_repetition_penalty_determinism_with_checkpoints():
    """Output with repeat_penalty must be identical with and without checkpoints.

    Without sampler state restore, ghost tokens from rejected drafts
    accumulate in the penalty ring buffer, altering output that uses
    repetition penalty. Compare checkpoint-enabled output against
    a non-checkpoint baseline — they should be identical.
    """
    base_config = dict(
        model_hf_repo=None,
        model_hf_file=None,
        model_file=QWEN35_08B,
        n_ctx=2048,
        n_slots=1,
        n_gpu_layer=99,
        seed=3407,
        temperature=0.0,
        n_predict=256,
        draft_min=4,
        draft_max=16,
        ctk="f16",
        ctv="f16",
        fa="on",
        reasoning="off",
    )

    prompt = "List 10 different animals. One per line. No numbering."
    request_data = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "repeat_penalty": 1.3,
    }

    # baseline: no checkpoints
    server_base = ServerProcess()
    for k, v in base_config.items():
        setattr(server_base, k, v)
    server_base.extra_args = [
        "--spec-type", "ngram-mod",
        "--draft-p-min", "0.99",
        "--no-cache-prompt",
    ]
    server_base.start(timeout_seconds=120)
    _warmup_ngram(server_base)
    res_base = server_base.make_request("POST", "/completion", data=request_data)
    assert res_base.status_code == 200
    output_base = res_base.body["content"]
    server_base.stop()

    # with checkpoints
    server_ckpt = ServerProcess()
    for k, v in base_config.items():
        setattr(server_ckpt, k, v)
    server_ckpt.extra_args = [
        "--spec-type", "ngram-mod",
        "--spec-use-checkpoints", "on",
        "--ctx-checkpoints", "4",
        "--draft-p-min", "0.99",
        "--no-cache-prompt",
    ]
    server_ckpt.start(timeout_seconds=120)
    _warmup_ngram(server_ckpt)
    res_ckpt = server_ckpt.make_request("POST", "/completion", data=request_data)
    assert res_ckpt.status_code == 200
    output_ckpt = res_ckpt.body["content"]

    assert output_base == output_ckpt, (
        f"Output differs with repeat_penalty when checkpoints are enabled.\n"
        f"Without checkpoints:\n{output_base[:200]}\n\n"
        f"With checkpoints:\n{output_ckpt[:200]}"
    )
