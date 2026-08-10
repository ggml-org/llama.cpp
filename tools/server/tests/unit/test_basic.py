import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


def test_server_props():
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == server.n_slots
    default_val = res.body["default_generation_settings"]
    assert server.n_ctx is not None and server.n_slots is not None
    assert default_val["n_ctx"] == server.n_ctx / server.n_slots
    assert default_val["params"]["seed"] == server.seed


def test_server_models():
    global server
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias


def test_server_slots():
    global server

    # without slots endpoint enabled, this should return error
    server.server_slots = False
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 501 # ERROR_TYPE_NOT_SUPPORTED
    assert "error" in res.body
    server.stop()

    # with slots endpoint enabled, this should return slots info
    server.server_slots = True
    server.n_slots = 2
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert len(res.body) == server.n_slots
    assert server.n_ctx is not None and server.n_slots is not None
    assert res.body[0]["n_ctx"] == server.n_ctx / server.n_slots
    assert "params" not in res.body[0]


def test_server_metrics_during_decode():
    global server
    server.server_metrics = True
    server.server_slots = True
    server.n_ctx = 2048
    server.n_batch = 2048
    server.n_slots = 1
    server.n_predict = 1
    server.n_threads = 1
    server.n_gpu_layer = 0
    server.start()

    res = server.make_request("GET", "/metrics")
    match = re.search(r"^llamacpp:n_decode_total (\d+)$", res.body, re.MULTILINE)
    assert match is not None
    n_decode_start = int(match.group(1))

    prompt = "Once upon a time " * 350
    with ThreadPoolExecutor(max_workers=1) as executor:
        completion = executor.submit(server.make_request, "POST", "/completion", {
            "prompt": prompt,
            "n_predict": 1,
        })

        saw_processing_during_decode = False
        while not completion.done():
            res = server.make_request("GET", "/metrics", timeout=5)
            assert res.status_code == 200
            if "llamacpp:requests_processing 1\n" in res.body:
                match = re.search(r"^llamacpp:n_decode_total (\d+)$", res.body, re.MULTILINE)
                assert match is not None
                if int(match.group(1)) == n_decode_start:
                    saw_processing_during_decode = True
                    break

        assert saw_processing_during_decode
        assert not completion.done()

        res = server.make_request("GET", "/slots", timeout=5)
        assert res.status_code == 200
        assert res.body[0]["is_processing"]
        assert not completion.done()

        assert completion.result().status_code == 200


def test_load_split_model():
    global server
    server.offline = False
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf"
    server.model_alias = "tinyllama-split"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl)+", res.body["content"])


def test_no_ui():
    global server
    # default: UI enabled
    server.start()
    url = f"http://{server.server_host}:{server.server_port}"
    res = requests.get(url)
    assert res.status_code == 200
    assert "<!doctype html>" in res.text
    server.stop()

    # with --no-ui, the UI should be disabled
    server.no_ui = True
    server.start()
    res = requests.get(url)
    assert res.status_code == 404


def test_server_model_aliases_and_tags():
    global server
    server.model_alias = "tinyllama-2,fim,code"
    server.model_tags = "chat,fim,small"
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    model = res.body["data"][0]
    # aliases field must contain all aliases
    assert set(model["aliases"]) == {"tinyllama-2", "fim", "code"}
    # tags field must contain all tags
    assert set(model["tags"]) == {"chat", "fim", "small"}
    # id is derived from first alias (alphabetical order from std::set)
    assert model["id"] == "code"
