"""
Regression tests for attention KV cache leak during speculative checkpoint operations.

Bug: restore_checkpoint() uses PARTIAL_ONLY flag, which only restores recurrent state.
Attention KV entries from rejected draft tokens at positions P+1..P+N remain as orphans.
Each rejection cycle leaks ~N attention KV cells, filling KV cache within a few requests.

Two leak sites:
  1. restore_checkpoint callback (server-context.cpp) — partial acceptance rollback
  2. rewind() (speculative.cpp) — bonus token + unaccepted drafts after full rejection

Requires Qwen3.5 (hybrid model with recurrent+attention layers).
"""

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


def _create_checkpoint_server(n_ctx=1024):
    """Create server with checkpoints and small context to trigger KV exhaustion.

    Uses --draft-p-min 0.99 to force almost all drafts to be rejected,
    guaranteeing the checkpoint restore/rewind path is exercised regardless
    of ngram prediction quality.
    """
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = QWEN35_08B
    server.n_ctx = n_ctx
    server.n_slots = 1
    server.n_gpu_layer = 99
    server.seed = 3407
    server.temperature = 0.0
    server.draft_min = 4
    server.draft_max = 16
    server.ctk = "f16"
    server.ctv = "f16"
    server.fa = "on"
    server.extra_args = [
        "--spec-type", "ngram-mod",
        "--spec-use-checkpoints", "on",
        "--ctx-checkpoints", "4",
        "--draft-p-min", "0.99",
        "--no-cache-prompt",
    ]
    return server


@requires_qwen35_08b
def test_kv_cache_does_not_leak_across_requests():
    """Server must handle 20 sequential requests without KV exhaustion.

    With attention KV leak, each speculative rejection cycle leaks ~N KV cells
    (N = draft_max). At n_ctx=512, draft_max=16, and draft_p_min=0.99 (forcing
    near-100% rejection), KV fills within a few requests.
    With the fix, KV is properly cleaned after each checkpoint restore/rewind.
    """
    server = _create_checkpoint_server(n_ctx=512)
    server.start(timeout_seconds=120)

    n_requests = 20
    n_predict = 128

    for i in range(n_requests):
        res = server.make_request("POST", "/completion", data={
            "prompt": f"Write a sorting algorithm number {i} in Python.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": n_predict,
        })
        assert res.status_code == 200, (
            f"Request {i+1}/{n_requests} failed with status {res.status_code}. "
            "KV cache likely exhausted due to attention KV leak after checkpoint restore."
        )

    # server must still be alive
    assert server.process.poll() is None, (
        f"Server crashed with return code {server.process.returncode}"
    )


@requires_qwen35_08b
def test_f16_checkpoint_determinism():
    """10 identical requests must produce identical output with f16 V cache.

    Validates that checkpoint save/restore/cleanup produces bit-exact results.
    This does NOT test quantized V cache divergence (known limitation).
    Uses default draft-p-min (no forced rejection) since the test validates
    determinism under normal operation, not rejection handling.
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
    server.draft_min = 4
    server.draft_max = 16
    server.ctk = "f16"
    server.ctv = "f16"
    server.fa = "on"
    server.extra_args = [
        "--spec-type", "ngram-mod",
        "--spec-use-checkpoints", "on",
        "--ctx-checkpoints", "4",
        "--no-cache-prompt",
    ]
    server.start(timeout_seconds=120)

    # prime ngram model so all test requests use the same speculative path
    server.make_request("POST", "/completion", data={
        "prompt": "Write a quicksort implementation in C. Output only code.",
        "temperature": 0.0, "top_k": 1, "n_predict": 256,
    })

    outputs = []
    for i in range(10):
        res = server.make_request("POST", "/completion", data={
            "prompt": "Write a quicksort implementation in C. Output only code.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 256,
        })
        assert res.status_code == 200, f"Request {i+1} failed: {res.status_code}"
        outputs.append(res.body["content"])

    assert len(set(outputs)) == 1, (
        f"Output divergence: {len(set(outputs))} unique outputs across "
        f"10 identical requests (f16 V cache, Qwen3.5-0.8B)."
    )
