import os
import pytest
from utils import *

server = ServerPreset.stories15m_moe()


# override the module-level fixture that tries to download all models
@pytest.fixture(scope="module", autouse=True)
def do_something():
    pass

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"
MODEL_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # use local model file to avoid requiring HTTPS support in the server
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = get_model_path()
    server.offline = False


def get_model_path():
    return os.path.abspath(download_file(MODEL_FILE_URL))


def get_lora_path():
    return os.path.abspath(download_file(LORA_FILE_URL))


def test_lora_hotload_and_activate():
    """Load a lora at runtime, activate it, and verify it changes model behavior."""
    global server
    server.start()

    # no adapters loaded initially
    res = server.make_request("GET", "/lora-adapters")
    assert res.status_code == 200
    assert len(res.body) == 0

    # load lora at runtime
    res = server.make_request("POST", "/lora-adapters/load", data={
        "path": get_lora_path(),
        "scale": 1.0,
    })
    assert res.status_code == 200
    assert res.body["id"] == 0
    assert res.body["scale"] == 1.0

    # verify it appears in the adapter list
    res = server.make_request("GET", "/lora-adapters")
    assert res.status_code == 200
    assert len(res.body) == 1
    assert res.body[0]["id"] == 0

    # activate and generate, should produce shakespeare-like text
    res = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": 1.0}
    ])
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex("(eye|love|glass|sun)+", res.body["content"])


def test_lora_hotload_scale_zero():
    """Load a lora but keep scale at 0, model should behave normally."""
    global server
    server.start()

    res = server.make_request("POST", "/lora-adapters/load", data={
        "path": get_lora_path(),
        "scale": 0.0,
    })
    assert res.status_code == 200

    # with scale 0, should produce normal bedtime story text
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex("(little|girl|three|years|old)+", res.body["content"])


def test_lora_hotload_unload():
    """Load a lora, verify it works, unload it, verify it's gone."""
    global server
    server.start()

    res = server.make_request("POST", "/lora-adapters/load", data={
        "path": get_lora_path(),
    })
    assert res.status_code == 200
    lora_id = res.body["id"]

    # activate it
    res = server.make_request("POST", "/lora-adapters", data=[
        {"id": lora_id, "scale": 1.0}
    ])
    assert res.status_code == 200

    # unload it
    res = server.make_request("POST", "/lora-adapters/unload", data={
        "id": lora_id,
    })
    assert res.status_code == 200
    assert res.body["success"] == True

    # adapter list should show it as unloaded (ptr null, scale 0)
    res = server.make_request("GET", "/lora-adapters")
    assert res.status_code == 200
    assert res.body[0]["scale"] == 0.0

    # generation should revert to normal
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex("(little|girl|three|years|old)+", res.body["content"])


def test_lora_hotload_multiple():
    """Load the same lora twice, verify both get unique IDs."""
    global server
    server.start()

    res1 = server.make_request("POST", "/lora-adapters/load", data={
        "path": get_lora_path(),
    })
    assert res1.status_code == 200
    assert res1.body["id"] == 0

    res2 = server.make_request("POST", "/lora-adapters/load", data={
        "path": get_lora_path(),
    })
    assert res2.status_code == 200
    assert res2.body["id"] == 1

    # both should appear in the list
    res = server.make_request("GET", "/lora-adapters")
    assert res.status_code == 200
    assert len(res.body) == 2


def test_lora_hotload_invalid_path():
    """Loading a nonexistent file should return an error."""
    global server
    server.start()

    res = server.make_request("POST", "/lora-adapters/load", data={
        "path": "/nonexistent/path/fake.gguf",
    })
    assert res.status_code != 200


def test_lora_hotload_missing_path():
    """Missing path field should return 400."""
    global server
    server.start()

    res = server.make_request("POST", "/lora-adapters/load", data={
        "scale": 1.0,
    })
    assert res.status_code != 200


def test_lora_unload_invalid_id():
    """Unloading a nonexistent ID should return an error."""
    global server
    server.start()

    res = server.make_request("POST", "/lora-adapters/unload", data={
        "id": 999,
    })
    assert res.status_code != 200


def test_lora_hotload_per_request():
    """Load lora at runtime, then use per-request scale control."""
    global server
    server.n_slots = 2
    server.start()

    res = server.make_request("POST", "/lora-adapters/load", data={
        "path": get_lora_path(),
    })
    assert res.status_code == 200
    lora_id = res.body["id"]

    # parallel requests: one with lora, one without
    tasks = [
        (server.make_request, ("POST", "/completion", {
            "prompt": "Look in thy glass",
            "lora": [{"id": lora_id, "scale": 0.0}],
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False,
        })),
        (server.make_request, ("POST", "/completion", {
            "prompt": "Look in thy glass",
            "lora": [{"id": lora_id, "scale": 1.0}],
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False,
        })),
    ]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    # scale 0 should be bedtime story
    assert match_regex("(little|girl|three|years|old)+", results[0].body["content"])
    # scale 1 should be shakespeare
    assert match_regex("(eye|love|glass|sun)+", results[1].body["content"])
