"""
Regression test for server crash on KV cache exhaustion during speculative decode.

Bug: When decode fails during speculative drafting (e.g. KV cache exhaustion
with concurrent requests), the server does not clear draft state before calling
sample_and_accept(). This leads to GGML_ASSERT(logits != nullptr) because the
failed batch has no logits for the drafted token positions.

Requires Qwen3.5-9B + 0.8B models.
"""

import os
import time
import pytest
import requests as req_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import ServerProcess

QWEN35_9B = os.environ.get(
    "QWEN35_9B_MODEL",
    os.path.expanduser("~/Models/Qwen3.5-9B-Q4_K_M.gguf"),
)
QWEN35_08B = os.environ.get(
    "QWEN35_08B_MODEL",
    os.path.expanduser("~/Models/Qwen3.5-0.8B-BF16.gguf"),
)

requires_qwen35 = pytest.mark.skipif(
    not os.path.exists(QWEN35_9B) or not os.path.exists(QWEN35_08B),
    reason="Requires local Qwen3.5-9B and 0.8B models",
)


@pytest.fixture(scope="module", autouse=True)
def do_something():
    """Override conftest's load_all — we use local models, not HF presets."""
    pass


def _make_chat_request(base_url, messages, max_tokens=400):
    """Make a chat request, returning (status, body) or (None, error_str)."""
    try:
        resp = req_lib.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "test",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=300,
        )
        return resp.status_code, resp.json()
    except Exception as e:
        return None, str(e)


@requires_qwen35
def test_server_survives_kv_exhaustion_with_ngram():
    """Server must not crash when KV cache fills during ngram speculation.

    Sends concurrent multi-turn requests to fill KV cache, triggering decode
    failure during speculative drafting. Without the fix, this crashes with
    GGML_ASSERT(logits != nullptr) in sampling.cpp.
    """
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = QWEN35_9B
    server.n_ctx = 4096
    server.n_slots = 4
    server.n_gpu_layer = 99
    server.seed = 3407
    server.temperature = 0.0
    server.jinja = True
    server.draft_max = 48
    server.extra_args = [
        "--spec-type", "ngram-mod",
        "--spec-use-checkpoints", "on",
        "--ctx-checkpoints", "4",
        "--no-warmup",
        "--chat-template-kwargs", '{"enable_thinking": false}',
    ]
    server.fa = None
    server.start(timeout_seconds=120)
    base_url = f"http://{server.server_host}:{server.server_port}"

    # phase 1: build ngram data
    for i in range(3):
        _make_chat_request(
            base_url,
            [{"role": "user", "content": f"Write quicksort in Python variant {i}. Full code."}],
            max_tokens=500,
        )
        if server.process.poll() is not None:
            break

    # phase 2: parallel multi-turn to fill KV cache
    if server.process.poll() is None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                futures.append(executor.submit(
                    _make_chat_request,
                    base_url,
                    [
                        {"role": "user", "content": f"Write quicksort variant {i}"},
                        {"role": "assistant", "content": "def quicksort(a,l=0,h=None):\n    if h is None: h=len(a)-1\n    if l<h: p=partition(a,l,h); quicksort(a,l,p-1); quicksort(a,p+1,h)"},
                        {"role": "user", "content": "Add mergesort heapsort bubblesort insertion selection. Full implementations."},
                    ],
                    max_tokens=1500,
                ))
            _ = [f.result() for f in as_completed(futures)]

    time.sleep(2)

    assert server.process.poll() is None, (
        f"Server crashed with return code {server.process.returncode}. "
        "Expected graceful error handling on KV exhaustion during speculative decode."
    )


@requires_qwen35
def test_server_survives_kv_exhaustion_with_draft_model():
    """Same crash test via draft model path instead of ngram."""
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = QWEN35_9B
    server.model_draft = QWEN35_08B
    server.n_ctx = 4096
    server.n_slots = 4
    server.n_gpu_layer = 99
    server.seed = 3407
    server.temperature = 0.0
    server.jinja = True
    server.draft_max = 24
    server.extra_args = [
        "--spec-use-checkpoints", "on",
        "--ctx-checkpoints", "4",
        "--no-warmup",
        "--chat-template-kwargs", '{"enable_thinking": false}',
    ]
    server.fa = None
    server.start(timeout_seconds=120)
    base_url = f"http://{server.server_host}:{server.server_port}"

    # build ngram data
    for i in range(3):
        _make_chat_request(
            base_url,
            [{"role": "user", "content": f"Write quicksort in Python variant {i}. Full code."}],
            max_tokens=500,
        )
        if server.process.poll() is not None:
            break

    # parallel multi-turn
    if server.process.poll() is None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                futures.append(executor.submit(
                    _make_chat_request,
                    base_url,
                    [
                        {"role": "user", "content": f"Write quicksort variant {i}"},
                        {"role": "assistant", "content": "def quicksort(a,l=0,h=None):\n    if h is None: h=len(a)-1"},
                        {"role": "user", "content": "Add mergesort heapsort bubblesort insertion selection. Full implementations."},
                    ],
                    max_tokens=1500,
                ))
            _ = [f.result() for f in as_completed(futures)]

    time.sleep(2)

    assert server.process.poll() is None, (
        f"Server crashed with return code {server.process.returncode}. "
        "Expected graceful error handling on KV exhaustion during speculative decode."
    )
