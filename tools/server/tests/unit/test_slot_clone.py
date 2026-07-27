import pytest
from utils import *

server = ServerPreset.tinyllama2()


LONG_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""".strip()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0


def test_slot_clone_to():
    global server
    server.start()

    # Process a long prompt on slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 16,
    })
    assert res.status_code == 200
    n_prompt_full = res.body["timings"]["prompt_n"]
    assert n_prompt_full > 0  # all tokens are processed

    # Clone the KV cache of slot 0 into slot 1
    res = server.make_request("POST", "/slots/0?action=clone_to&target=1")
    assert res.status_code == 200
    assert res.body["id_slot"] == 0
    assert res.body["id_slot_target"] == 1
    assert res.body["n_cloned"] > 0
    assert "clone_ms" in res.body["timings"]

    # A short suffix on slot 1 should reuse the cloned prefix:
    # only the suffix tokens are processed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT + " The end.",
        "id_slot": 1,
        "cache_prompt": True,
        "n_predict": 16,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] < 16  # only the suffix is processed
    assert res.body["timings"]["prompt_n"] < n_prompt_full

    # The source slot must not be corrupted by the clone:
    # the same suffix on slot 0 hits the cache too
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT + " The end.",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 16,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] < 16


def test_slot_clone_to_errors():
    global server
    server.start()

    # Process a prompt on slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 8,
    })
    assert res.status_code == 200

    # Target slot out of range
    res = server.make_request("POST", "/slots/0?action=clone_to&target=5")
    assert res.status_code == 400
    assert res.body["error"]["type"] == "invalid_request_error"

    # Source and target must be different
    res = server.make_request("POST", "/slots/0?action=clone_to&target=0")
    assert res.status_code == 400
    assert res.body["error"]["type"] == "invalid_request_error"

    # Missing target
    res = server.make_request("POST", "/slots/0?action=clone_to")
    assert res.status_code == 400
    assert res.body["error"]["type"] == "invalid_request_error"


def test_slot_clone_to_empty_source():
    global server
    server.start()

    # Cloning from a slot that never processed a prompt must fail
    res = server.make_request("POST", "/slots/1?action=clone_to&target=0")
    assert res.status_code == 400
    assert res.body["error"]["type"] == "invalid_request_error"
