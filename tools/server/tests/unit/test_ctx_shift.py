import pytest
from utils import *

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
@pytest.fixture
def server(server_factory):
    s = server_factory('tinyllama2')
    s.n_ctx = 512
    s.n_slots = 2
    s.n_predict = 128
    return s


def test_ctx_shift_enabled(server):
    # the prompt is 226 tokens
    # the slot context is 512/2 = 256 tokens
    # 96 tokens are generated thanks to shifting the context when it gets full
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
def test_ctx_shift_disabled_short_prompt(server, n_predict: int, n_token_output: int, truncated: bool):
    server.n_predict = -1
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": "Hi how are you",
    })
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == n_token_output
    assert res.body["truncated"] == truncated


def test_ctx_shift_disabled_long_prompt(server):
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 64,
        "prompt": LONG_TEXT,
    })
    assert res.status_code != 200
    assert "error" in res.body
    assert "exceeds the available context size" in res.body["error"]["message"]

def test_ctx_shift_disabled_stream(server):
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
