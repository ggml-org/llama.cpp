import threading
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


def _get_model_ids(is_reload: bool) -> set[str]:
    res = server.make_request("GET", "/models" + ("?reload=1" if is_reload else ""))
    assert res.status_code == 200
    return {item["id"] for item in res.body.get("data", [])}


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


def test_router_reload_models():
    """POST /models/reload re-reads the INI preset and updates the model list."""
    global server

    preset_path = os.path.join(TMP_DIR, "test_reload.ini")

    # Initial preset: two models
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-a]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "\n"
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
        )

    server.models_preset = preset_path
    server.start()

    ids = _get_model_ids(is_reload=False)
    assert "model-reload-a" in ids
    assert "model-reload-b" in ids

    # Updated preset: remove a, keep b unchanged, add c
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
            "\n"
            "[model-reload-c]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
        )

    try:
        ids = _get_model_ids(is_reload=True)
        assert "model-reload-a" not in ids, "removed model should no longer appear"
        assert "model-reload-b" in ids, "unchanged model should still appear"
        assert "model-reload-c" in ids, "newly added model should appear"
    finally:
        os.remove(preset_path)


def test_router_remote_preset():
    global server
    server.model_hf_repo = "ggml-org/test-preset-ci"
    server.model_hf_file = None
    server.offline = False
    server.start()

    # Should see preset models in GET /models
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    ids = {item["id"] for item in res.body.get("data", [])}
    assert "tinygemma3-preset" in ids
    assert "stories260K-test" in ids

    # Should be able to load a preset model
    model_id = "tinygemma3-preset"
    _load_model_and_wait(model_id)


MODEL_DOWNLOAD_ID = "ggml-org/test-model-router-download:F16"
MODEL_DOWNLOAD_TIMEOUT = 30


def _listen_sse(
    server: ServerProcess, collected: list, stop: threading.Event, ready: threading.Event | None = None
):
    """Collect /models/sse events into `collected` until `stop` is set.

    When `ready` is provided, it is set once the streaming response is open,
    i.e. the server has accepted the connection and registered us as a
    subscriber. Callers that trigger one-shot events (e.g. download_finished)
    must wait on `ready` before acting, otherwise the event can be broadcast
    before this client is subscribed and be lost.
    """
    url = f"http://{server.server_host}:{server.server_port}/models/sse"
    try:
        with requests.get(url, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT) as resp:
            if ready is not None:
                ready.set()
            for line_bytes in resp.iter_lines():
                if stop.is_set():
                    break
                line = line_bytes.decode("utf-8")
                if line.startswith("data: "):
                    collected.append(json.loads(line[6:]))
    except Exception:
        pass


def _wait_for_sse_event(collected: list, event_type: str, model: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e.get("event") == event_type and e.get("model") == model for e in collected):
            return True
        time.sleep(0.5)
    return False


def test_router_download_model():
    """Case 1: download a model, verify SSE events and GET /models."""
    global server
    server.start()

    # Ensure the model is not present before we start
    server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")

    sse_events: list = []
    stop = threading.Event()
    sse_ready = threading.Event()
    sse_thread = threading.Thread(
        target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
    )
    sse_thread.start()

    # wait for the SSE client to be subscribed before triggering the download,
    # otherwise the one-shot download_finished event can be broadcast before
    # this client is registered and be lost
    assert sse_ready.wait(10), "SSE client failed to connect"

    # Trigger the download
    res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
    assert res.status_code == 200
    assert res.body.get("success") is True

    # Wait for download_finished SSE event
    finished = _wait_for_sse_event(
        sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
    )
    stop.set()

    assert finished, "Never received download_finished SSE event"
    assert any(
        e.get("event") == "download_progress" and e.get("model") == MODEL_DOWNLOAD_ID
        for e in sse_events
    ), "No download_progress events received"

    # Model should now appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID in ids, f"{MODEL_DOWNLOAD_ID} not found in /models after download"


def test_router_delete_model():
    """Case 2: delete the downloaded model, verify it disappears from GET /models."""
    global server
    server.start()

    # Ensure the model exists (download it if needed)
    if MODEL_DOWNLOAD_ID not in _get_model_ids(is_reload=False):
        sse_events: list = []
        stop = threading.Event()
        sse_ready = threading.Event()
        threading.Thread(
            target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
        ).start()
        # subscribe before triggering the download so the one-shot
        # download_finished event is not lost (see test_router_download_model)
        assert sse_ready.wait(10), "SSE client failed to connect"
        res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
        assert res.status_code == 200
        finished = _wait_for_sse_event(
            sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
        )
        stop.set()
        assert finished, "Model did not finish downloading before delete test"

    # Delete the model
    del_res = server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")
    assert del_res.status_code == 200
    assert del_res.body.get("success") is True

    # Model should no longer appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID not in ids, f"{MODEL_DOWNLOAD_ID} still present after deletion"


def _get_model_entry(model_id: str) -> dict:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    return next(item for item in res.body["data"] if item["id"] == model_id)


def test_router_models_preset_sequential_load():
    """Legacy sequential_load is canonicalized and contained without spawning."""
    global server

    preset_path = os.path.join(TMP_DIR, "test_sequential.ini")

    with open(preset_path, "w") as f:
        f.write(
            "[model-seq]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "sequential_load = true\n"
            "device = CUDA0\n"
            "alias = seq-alias\n"
            "ctx-size = 128\n"
        )

    server.models_preset = preset_path
    server.start()

    try:
        entry = _get_model_entry("model-seq")
        args = entry["status"]["args"]
        assert args.count("--sequential") == 1
        assert "--sequential-load" not in args
        assert "--no-sequential" not in args
        assert args[args.index("--device") + 1] == "CUDA0"
        # Router CLI values keep their documented precedence over model INI.
        assert args[args.index("--ctx-size") + 1] == "1024"
        assert "seq-alias" in entry["aliases"]
        assert entry["status"]["value"] == "unloaded"
        props = server.make_request("GET", "/props")
        assert props.body["max_instances"] == 1
    finally:
        os.remove(preset_path)


@pytest.mark.parametrize("sequential_line", ["", "sequential-load = false\n"])
def test_router_sequential_false_or_absent_stays_disabled(sequential_line: str):
    """False and absent settings do not enable child or router sequential mode."""
    preset_path = os.path.join(TMP_DIR, "test_sequential_disabled.ini")
    with open(preset_path, "w") as f:
        f.write(
            "[model-ordinary]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            f"{sequential_line}"
            "alias = ordinary-alias\n"
            "ctx-size = 96\n"
        )
    server.models_preset = preset_path
    server.start()
    try:
        entry = _get_model_entry("model-ordinary")
        args = entry["status"]["args"]
        assert "--sequential" not in args
        assert args.count("--no-sequential") == (1 if sequential_line else 0)
        assert args[args.index("--ctx-size") + 1] == "1024"
        assert "ordinary-alias" in entry["aliases"]
        assert entry["status"]["value"] == "unloaded"
        props = server.make_request("GET", "/props")
        assert props.body["max_instances"] != 1
    finally:
        os.remove(preset_path)


@pytest.mark.parametrize(
    "cli_value,ini_value,expected",
    [(False, "true", "--no-sequential"), (True, "false", "--sequential")],
)
def test_router_sequential_cli_precedence(cli_value: bool, ini_value: str, expected: str):
    preset_path = os.path.join(TMP_DIR, "test_sequential_precedence.ini")
    with open(preset_path, "w") as f:
        f.write(
            "[model-seq]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            f"sequential-load = {ini_value}\n"
            "device = CUDA0\n"
        )
    server.models_preset = preset_path
    server.sequential = cli_value
    server.device = "CUDA1"
    server.start()
    try:
        args = _get_model_entry("model-seq")["status"]["args"]
        assert expected in args
        assert args[args.index("--device") + 1] == "CUDA1"
    finally:
        os.remove(preset_path)


@pytest.mark.parametrize("models_max", [0, 2])
def test_router_sequential_explicit_models_max_rejected(models_max: int):
    preset_path = os.path.join(TMP_DIR, "test_sequential_models_max.ini")
    log_path = os.path.join(TMP_DIR, "test_sequential_models_max.log")
    with open(preset_path, "w") as f:
        f.write("[model-seq]\nhf-repo = local/test\nsequential-load = true\ndevice = CUDA0\n")
    server.models_preset = preset_path
    server.models_max = models_max
    server.log_path = log_path
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=10)
    with open(log_path) as f:
        assert "sequential loading requires --models-max 1 in router mode" in f.read()
    os.remove(preset_path)
    os.remove(log_path)


@pytest.mark.parametrize("device", [None, "none", "CPU", "CUDA0,CUDA1", "Vulkan0"])
def test_router_sequential_invalid_device_rejected(device: str | None):
    preset_path = os.path.join(TMP_DIR, "test_sequential_device.ini")
    log_path = os.path.join(TMP_DIR, "test_sequential_device.log")
    device_line = "" if device is None else f"device = {device}\n"
    with open(preset_path, "w") as f:
        f.write(f"[model-seq]\nhf-repo = local/test\nsequential-load = true\n{device_line}")
    server.models_preset = preset_path
    server.log_path = log_path
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=10)
    with open(log_path) as f:
        assert "sequential loading requires exactly one explicit CUDA device in router mode" in f.read()
    os.remove(preset_path)
    os.remove(log_path)
