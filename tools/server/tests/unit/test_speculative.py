import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def _build_server(server_factory):
    s = server_factory("stories15m_moe")
    # set default values
    s.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    s.draft_min = 4
    s.draft_max = 8
    s.fa = "off"
    return s


@pytest.fixture
def server(server_factory):
    return _build_server(server_factory)


def test_with_and_without_draft(server, server_factory):
    server.model_draft = None  # disable draft model
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    })
    assert res.status_code == 200
    content_no_draft = res.body["content"]
    server.stop()

    # create new server with draft model
    server = _build_server(server_factory)
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    })
    assert res.status_code == 200
    content_draft = res.body["content"]

    assert content_no_draft == content_draft


def test_different_draft_min_draft_max(server):
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.draft_min = draft_min
        server.draft_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_slot_ctx_not_exceeded(server):
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift(server):
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(server, n_slots: int, n_requests: int):
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])
