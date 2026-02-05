import pytest
import time
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


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


def _load_model_and_wait(model_id: str, timeout: int = 60) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}
    )
    assert load_res.status_code == 200
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout)


def test_router_stop_idle():
    """Test that idle model instances are fully terminated after stop_idle_seconds."""
    global server
    server.stop_idle_seconds = 2
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    # load a model
    _load_model_and_wait(model_id, timeout=120)
    assert _get_model_status(model_id) == "loaded"

    # wait for the idle watchdog to terminate it
    _wait_for_model_status(model_id, {"unloaded"}, timeout=10)

    # model should have been stopped
    assert _get_model_status(model_id) == "unloaded"


def test_router_stop_idle_respawn():
    """Test that a stopped idle model is re-spawned on next request (autoload)."""
    global server
    server.stop_idle_seconds = 2
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    # load model, wait for idle stop
    _load_model_and_wait(model_id, timeout=120)
    _wait_for_model_status(model_id, {"unloaded"}, timeout=10)

    # send a chat request - should trigger autoload and succeed
    res = server.make_request("POST", "/chat/completions", data={
        "model": model_id,
        "max_tokens": 4,
        "messages": [
            {"role": "user", "content": "hello"},
        ],
    })
    assert res.status_code == 200
    assert "error" not in res.body

    # model should be loaded again
    assert _get_model_status(model_id) == "loaded"
