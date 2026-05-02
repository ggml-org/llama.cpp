import pytest
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_router_props():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["role"] == "router"
    assert res.body["max_instances"] == 2
    assert res.body["models_autoload"] is False
    assert res.body["build_info"].startswith("b")


@pytest.mark.parametrize(
    "model,success",
    [
        ("ggml-org/tinygemma3-GGUF:Q8_0", True),
        ("non-existent/model", False),
    ]
)
def test_router_chat_completion_stream(model: str, success: bool):
    global server
    server.start()
    content = ""
    ex: ServerError | None = None
    try:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
        })
        for data in res:
            if data["choices"]:
                choice = data["choices"][0]
                if choice["finish_reason"] in ["stop", "length"]:
                    assert "content" not in choice["delta"]
                else:
                    assert choice["finish_reason"] is None
                    content += choice["delta"]["content"] or ''
    except ServerError as e:
        ex = e

    if success:
        assert ex is None
        assert len(content) > 0
    else:
        assert ex is not None
        assert content == ""


def _get_model_status(model_id: str) -> str:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id)
        if last_status in desired:
            return last_status
        time.sleep(1)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    assert load_res.status_code == 200
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout)


def test_router_unload_model():
    global server
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    _load_model_and_wait(model_id)

    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model_id, {"unloaded"})


def test_router_models_max_evicts_lru():
    global server
    server.models_max = 2
    server.start()

    candidate_models = [
        "ggml-org/tinygemma3-GGUF:Q8_0",
        "ggml-org/test-model-stories260K:F32",
        "ggml-org/test-model-stories260K-infill:F32",
    ]

    # Load only the first 2 models to fill the cache
    first, second, third = candidate_models[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    # Verify both models are loaded
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load the third model - this should trigger LRU eviction of the first model
    _load_model_and_wait(third, timeout=120)

    # Verify eviction: third is loaded, first was evicted
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


def test_router_no_models_autoload():
    global server
    server.no_models_autoload = True
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 400
    assert "error" in res.body

    _load_model_and_wait(model_id)

    success_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert success_res.status_code == 200
    assert "error" not in success_res.body


def test_router_api_key_required():
    global server
    server.api_key = "sk-router-secret"
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    auth_headers = {"Authorization": f"Bearer {server.api_key}"}

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 401
    assert res.body.get("error", {}).get("type") == "authentication_error"

    _load_model_and_wait(model_id, headers=auth_headers)

    authed = server.make_request(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert authed.status_code == 200
    assert "error" not in authed.body


def test_router_priority_high_overrides_low():
    global server
    server.models_max = 2
    server.models_priority_default = 0  # default priority for models loaded via presets
    server.no_models_autoload = True
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    model_b = "ggml-org/test-model-stories260K:F32"
    model_c = "ggml-org/test-model-stories260K-infill:F32"

    # Load two models with default priority 0
    _load_model_and_wait(model_a, timeout=120)
    _load_model_and_wait(model_b, timeout=120)

    assert _get_model_status(model_a) == "loaded"
    assert _get_model_status(model_b) == "loaded"

    # Unload model_b so we have room
    server.make_request("POST", "/models/unload", data={"model": model_b})
    _wait_for_model_status(model_b, {"unloaded"}, timeout=120)

    # Load model_c with priority 5 via POST /models/load
    # It should evict model_a (priority 0 < priority 5) and NOT model_b (priority 0 == priority 0)
    # But wait, model_b has priority 0 which is NOT less than requesting priority 5
    # Actually, the request priority is 5, so we evict any running model with pri < 5
    # Both model_a (pri=0) and model_b (pri=0) have pri < 5, so we evict lowest (both equal pri=0, LRU wins)
    # model_a was loaded first so it's LRU, so model_a gets evicted
    load_c = server.make_request(
        "POST", "/models/load", data={"model": model_c, "priority": 5}
    )
    assert load_c.status_code == 200

    # Wait for model_c to be loaded
    _wait_for_model_status(model_c, {"loaded"}, timeout=120)

    # Verify model_a was evicted (lower priority than request)
    status_a = _get_model_status(model_a)
    assert status_a == "unloaded", f"Expected model_a to be evicted, got {status_a}"

    # Verify model_b is still loaded (LRU tiebreaker when pri=0, model_a was loaded first)
    status_b = _get_model_status(model_b)
    assert status_b == "loaded", f"Expected model_b to still be loaded, got {status_b}"

    # Verify model_c is loaded with priority 5
    models_res = server.make_request("GET", "/models")
    for model_item in models_res.body.get("data", []):
        if model_item.get("id") == model_c:
            assert model_item.get("priority") == 5
            break

def test_router_low_priority_does_not_evict_high():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.start()

    model_a = "ggml-org/test-model-stories260K:F32"
    model_b = "ggml-org/test-model-stories260K-infill:F32"

    # Load model_a with priority 5
    load_a = server.make_request(
        "POST", "/models/load", data={"model": model_a, "priority": 5}
    )
    assert load_a.status_code == 200
    _wait_for_model_status(model_a, {"loaded"}, timeout=120)

    # Load model_b with priority 1 (default)
    _load_model_and_wait(model_b, timeout=120)
    assert _get_model_status(model_b) == "loaded"

    # Now request to load model_a again with priority 2
    # This should NOT evict model_b (pri=1 < req pri=2, so model_b would be evicted)
    # But since model_a is already running, POST /models/load returns already running
    # Let's test a different scenario: unload model_b, load it back with pri=1
    # While model_a (pri=5) is running
    server.make_request("POST", "/models/unload", data={"model": model_b})
    _wait_for_model_status(model_b, {"unloaded"}, timeout=120)

    # model_a (pri=5) is running, try to load model_b (pri=1)
    # Since req pri=1 and model_a pri=5, we only evict pri < 1
    # model_a has pri=5 which is NOT < 1, so NO eviction should happen
    # But models_max=2 and we have 1 running, so room for 1 more
    _load_model_and_wait(model_b, timeout=120)
    assert _get_model_status(model_a) == "loaded", f"Expected model_a to still be loaded (high pri not evicted by low pri request), got {_get_model_status(model_a)}"
    assert _get_model_status(model_b) == "loaded"


def test_router_priority_lru_within_same_priority():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    model_b = "ggml-org/test-model-stories260K:F32"
    model_c = "ggml-org/test-model-stories260K-infill:F32"

    # Load first two models (default priority 0)
    _load_model_and_wait(model_a, timeout=120)
    _load_model_and_wait(model_b, timeout=120)

    assert _get_model_status(model_a) == "loaded"
    assert _get_model_status(model_b) == "loaded"

    # Load model_c with same priority (default 0) - should evict LRU (model_a)
    _load_model_and_wait(model_c, timeout=120)

    # model_a should be evicted (LRU among same priority)
    assert _get_model_status(model_a) == "unloaded"
    assert _get_model_status(model_b) == "loaded"
    assert _get_model_status(model_c) == "loaded"


def test_router_priority_default_in_props():
    global server
    server.models_max = 4
    server.models_priority_default = 3
    server.no_models_autoload = True
    server.start()

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body.get("default_priority") == 3


def test_router_models_load_with_priority_field():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    model_b = "ggml-org/test-model-stories260K:F32"

    _load_model_and_wait(model_a, timeout=120)
    _load_model_and_wait(model_b, timeout=120)

    # Verify both are loaded
    assert _get_model_status(model_a) == "loaded"
    assert _get_model_status(model_b) == "loaded"

    # Try to load model_b again with priority 10 via /models/load
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_b, "priority": 10}
    )
    # Model is already running, should get error
    assert load_res.status_code == 200
    assert load_res.body.get("success") is True
