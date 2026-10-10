import pytest
from utils import *

server = ServerPreset.tinylfm2()

HEAD = " ".join(f"intro{i}" for i in range(20)) + ". "
MID  = " ".join(f"middle{i}" for i in range(40)) + ". "
TAIL = " ".join(f"word{i}" for i in range(80))


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylfm2()


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return len(res.body["tokens"])


def prompt_n(prompt: str) -> int:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "n_cache_reuse": 8,
    })
    assert res.status_code == 200
    return res.body["timings"]["prompt_n"]


def test_cache_reuse_hybrid_off():
    global server
    server.start()
    prompt_n(HEAD + MID + TAIL)
    # stock path: the recurrent state does not match the shifted cache, so the prompt is processed again
    assert prompt_n(HEAD + TAIL) > n_tokens(TAIL)


@pytest.mark.parametrize("n_tail", [1, 4])
def test_cache_reuse_hybrid_remove(n_tail: int):
    global server
    server.n_cache_reuse_hybrid = n_tail
    server.start()
    prompt_n(HEAD + MID + TAIL)
    # TAIL moves back, only its last n_tail tokens and the edit boundary are processed
    assert prompt_n(HEAD + TAIL) <= n_tail + 4


def test_cache_reuse_hybrid_insert():
    global server
    server.n_cache_reuse_hybrid = 4
    server.start()
    prompt_n(HEAD + TAIL)
    # TAIL moves forward over the new text
    n = prompt_n(HEAD + MID + TAIL)
    assert n <= n_tokens(MID) + 8
    assert n < n_tokens(TAIL)


def test_cache_reuse_hybrid_replace_twice():
    global server
    server.n_cache_reuse_hybrid = 4
    server.start()
    other1 = " ".join(f"first{i}" for i in range(10)) + ". "
    other2 = " ".join(f"second{i}" for i in range(10)) + ". "
    prompt_n(HEAD + MID + TAIL)
    prompt_n(HEAD + other1 + TAIL)
    # the slot holds the result of the previous reuse, which must be reusable again
    assert prompt_n(HEAD + other2 + TAIL) <= n_tokens(other2) + 8


def test_cache_reuse_hybrid_min_share():
    global server
    server.n_cache_reuse_hybrid = 4
    server.cache_reuse_hybrid_min_share = 1.0
    server.start()
    prompt_n(HEAD + MID + TAIL)
    # the reused chunks never cover the whole prompt, so the stock path runs
    assert prompt_n(HEAD + TAIL) > n_tokens(TAIL)


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_cache_reuse_hybrid_mrope():
    # qwen35: hybrid memory with interleaved M-RoPE
    global server
    server.model_hf_repo = "ggml-org/Qwen3.5-0.8B-GGUF:Q4_0"
    server.model_alias = "qwen3.5-0.8b"
    server.offline = False
    server.n_cache_reuse_hybrid = 4
    server.start()
    prompt_n(HEAD + MID + TAIL)
    assert prompt_n(HEAD + TAIL) <= 8
    assert prompt_n(HEAD + MID + TAIL) <= n_tokens(MID) + 8
