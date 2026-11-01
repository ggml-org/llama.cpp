import json
import os
import pytest
import time
from utils import *

# the service: a router with a UNIX socket that 'llama serve' spawns models through

server: ServerProcess

MODEL = "ggml-org/test-model-stories260K:F32"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()
    port = server.server_port
    server.service_sock = f"tmp/service-{port}.sock"
    server.config_dir = f"tmp/config-{port}"


def _model(model_id: str) -> dict | None:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body["data"]:
        if item["id"] == model_id:
            return item
    return None


def _wait_for_status(model_id: str, desired: set[str], timeout: int = 60) -> str:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        item = _model(model_id)
        last = item["status"]["value"] if item else "missing"
        if last in desired:
            return last
        time.sleep(0.2)
    raise AssertionError(f"model {model_id} stayed in {last}, wanted {desired}")


def _spawn(args: list[str], **kwargs) -> ServerResponse:
    return server.make_sock_request("POST", "/service/spawn", data={
        "args": args,
        "cwd": os.getcwd(),
        **kwargs,
    })


def test_service_status_and_lock():
    global server
    server.start()
    res = server.make_sock_request("GET", "/service/status")
    assert res.status_code == 200
    assert res.body["pid"] == server.process.pid
    assert res.body["port"] == server.server_port
    assert res.body["build"].startswith("b")
    assert res.body["running"] == []

    with open(os.path.join(server.config_dir, "llama.cpp", "service.json")) as f:
        lock = json.load(f)
    assert lock["pid"] == server.process.pid
    assert lock["port"] == server.server_port
    assert lock["sock"] == server.service_sock
    assert "--service-sock" in lock["args"]

    # the service routes are not on the TCP port
    res = server.make_request("GET", "/service/status")
    assert res.status_code == 404


def test_service_socket_serves_router_api():
    global server
    server.start()
    res = server.make_sock_request("GET", "/models")
    assert res.status_code == 200
    assert any(item["id"] == MODEL for item in res.body["data"])


def test_service_spawn_with_args():
    global server
    server.start()
    res = _spawn(["-hf", MODEL, "-c", "256", "--alias", "tiny"])
    assert res.status_code == 200
    assert res.body["model"] == MODEL
    assert _wait_for_status(MODEL, {"loaded"}) == "loaded"

    item = _model(MODEL)
    assert item["source"] == "spawned"
    assert "tiny" in item["aliases"]
    args = item["status"]["args"]
    assert args[args.index("--ctx-size") + 1] == "256"

    # a request without a model goes to the spawned model
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert res.status_code == 200
    assert len(res.body["choices"][0]["message"]["content"]) > 0

    # and the alias routes to it
    res = server.make_request("POST", "/v1/chat/completions", data={
        "model": "tiny",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert res.status_code == 200

    res = server.make_sock_request("GET", "/service/status")
    assert res.body["running"] == [MODEL]


def test_service_spawn_reserved_args_are_stripped():
    global server
    server.start()
    res = _spawn(["-hf", MODEL, "--port", "1234", "--models-max", "99"])
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})
    args = _model(MODEL)["status"]["args"]
    assert args[args.index("--port") + 1] != "1234"
    assert "--models-max" not in args


def test_service_spawn_bad_args():
    global server
    server.start()
    res = _spawn(["-hf", MODEL, "--no-such-flag"])
    assert res.status_code == 400
    res = _spawn(["-c", "256"])
    assert res.status_code == 400
    assert "no model" in res.body["error"]["message"]
    assert _model(MODEL)["status"]["value"] == "unloaded"


def test_service_spawn_replace():
    global server
    server.start()
    res = _spawn(["-hf", MODEL, "-c", "256"])
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})

    res = _spawn(["-hf", MODEL, "-c", "512"])
    assert res.status_code == 400
    assert "already running" in res.body["error"]["message"]

    res = _spawn(["-hf", MODEL, "-c", "512"], replace=True)
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})
    args = _model(MODEL)["status"]["args"]
    assert args[args.index("--ctx-size") + 1] == "512"


def test_service_spawned_model_restores_cache_entry_on_unload():
    global server
    server.start()
    before = _model(MODEL)
    assert before["source"] == "cache"

    res = _spawn(["-hf", MODEL, "-c", "256"])
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})
    assert _model(MODEL)["source"] == "spawned"

    res = server.make_request("POST", "/models/unload", data={"model": MODEL})
    assert res.status_code == 200
    _wait_for_status(MODEL, {"unloaded"})
    after = _model(MODEL)
    assert after["source"] == "cache"
    args = after["status"]["args"]
    assert args[args.index("--ctx-size") + 1] == "1024"  # the router's own -c, not the spawn's

    # a request without a model has nowhere to go now
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert res.status_code == 400


def test_service_spawned_model_is_not_evicted():
    global server
    server.models_max = 1
    server.start()
    res = _spawn(["-hf", MODEL, "-c", "256"])
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})

    # another model cannot evict the spawned one
    res = server.make_request("POST", "/v1/chat/completions", data={
        "model": "ggml-org/tinygemma3-GGUF:Q8_0",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert res.status_code != 200
    assert _model(MODEL)["status"]["value"] == "loaded"


def test_service_spawn_survives_reload():
    global server
    server.start()
    res = _spawn(["-hf", MODEL, "-c", "256"])
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})

    res = server.make_request("GET", "/models?reload=1")
    assert res.status_code == 200
    item = _model(MODEL)
    assert item["source"] == "spawned"
    assert item["status"]["value"] == "loaded"


def test_service_logs():
    global server
    server.start()
    res = server.make_sock_request("GET", f"/service/logs?model={MODEL}")
    assert res.status_code == 404

    res = _spawn(["-hf", MODEL, "-c", "256"])
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})

    res = server.make_sock_request("GET", f"/service/logs?model={MODEL}")
    assert res.status_code == 200
    assert "model loaded" in res.body

    # the default model needs no name
    res = server.make_sock_request("GET", "/service/logs")
    assert res.status_code == 200
    assert "model loaded" in res.body


def test_service_stop():
    global server
    server.start()
    res = _spawn(["-hf", MODEL, "-c", "256"])
    assert res.status_code == 200
    _wait_for_status(MODEL, {"loaded"})

    res = server.make_sock_request("POST", "/service/stop", data={})
    assert res.status_code == 200
    server.process.wait(timeout=30)
    assert server.process.returncode == 0
    assert not os.path.exists(os.path.join(server.config_dir, "llama.cpp", "service.json"))
    assert not os.path.exists(server.service_sock)
