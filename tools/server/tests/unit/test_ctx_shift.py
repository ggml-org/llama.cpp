import pytest
from utils import *

server = ServerPreset.tinyllama2()


SHORT_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
""".strip()

LONG_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
""".strip()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 512
    server.n_slots = 2
    server.n_predict = 128


def test_ctx_shift_enabled():
    # the prompt is 226 tokens
    # the slot context is 512/2 = 256 tokens
    # 96 tokens are generated thanks to shifting the context when it gets full
    global server
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 96,
        "prompt": SHORT_TEXT,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 226
    assert res.body["timings"]["predicted_n"] == 96
    assert res.body["truncated"] is True


@pytest.mark.parametrize("n_predict,n_token_output,truncated", [
    (64, 64, False),
    (-1, 248, True), # 8 tokens prompt + 248 tokens generated = 256 tokens total
])
def test_ctx_shift_disabled_short_prompt(n_predict: int, n_token_output: int, truncated: bool):
    global server
    server.n_predict = -1
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": "Hi how are you",
    })
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == n_token_output
    assert res.body["truncated"] == truncated


def test_ctx_shift_disabled_long_prompt():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 64,
        "prompt": LONG_TEXT,
    })
    assert res.status_code != 200
    assert "error" in res.body
    assert "exceeds the available context size" in res.body["error"]["message"]

def test_ctx_shift_disabled_stream():
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/completions", data={
        "n_predict": 256,
        "prompt": "Once",
        "stream": True,
    })
    content = ""
    for data in res:
        choice = data["choices"][0]
        if choice["finish_reason"] == "length":
            assert len(content) > 0
        else:
            assert choice["finish_reason"] is None
            content += choice["text"]


@pytest.mark.parametrize("endpoint,n_predict_key", [
    ("/completion", "n_predict"),
    ("/v1/completions", "max_tokens"),
])
def test_n_predict_minus_2(endpoint: str, n_predict_key: str):
    """n_predict == -2: generate until context is full, then stop (no ctx shift)."""
    global server
    server.n_predict = -1  # global: unlimited, request-level -2 overrides
    server.enable_ctx_shift = True  # enabled, but -2 should stop instead of shifting
    server.start()
    res = server.make_request("POST", endpoint, data={
        n_predict_key: -2,
        "prompt": "Hi how are you",
    })
    assert res.status_code == 200
    # "Hi how are you" is 8 tokens, slot ctx = 512/2 = 256, expect ~248 generated
    if "timings" in res.body:
        n_predicted = res.body["timings"]["predicted_n"]
        assert res.body["truncated"] is True
        assert res.body["stopped_limit"] is True
    else:
        n_predicted = res.body["usage"]["completion_tokens"]
    assert n_predicted == 248, f"n_predict=-2 should fill context (expected 248), got {n_predicted}"


def test_n_predict_minus_2_global():
    """n_predict == -2 set globally (via server CLI) should also work."""
    global server
    server.n_predict = -2
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hi how are you",
    })
    assert res.status_code == 200
    n_predicted = res.body["timings"]["predicted_n"]
    assert res.body["truncated"] is True
    assert res.body["stopped_limit"] is True
    assert n_predicted == 248, f"global n_predict=-2 should fill context (expected 248), got {n_predicted}"
