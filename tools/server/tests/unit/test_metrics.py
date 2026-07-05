import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_prompt_cache_metrics():
    global server
    server.server_metrics = True
    server.n_slots = 1
    server.start()

    req = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 8,
        "cache_prompt": True,
    }

    res = server.make_request("POST", "/completion", data=req)
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data=req)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0

    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert match_regex(r"llamacpp:prompt_tokens_cache_total\s+[1-9]", res.body)
    assert "llamacpp:prompt_cache_hit_ratio" in res.body
