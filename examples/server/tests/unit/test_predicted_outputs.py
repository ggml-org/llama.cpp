import pytest
from utils import *


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.draft_max = 1024
    server.debug = True


def test_with_and_without_prediced_outputs():
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "I believe the meaning of life is"}],
        "temperature": 0.0,
        "top_k": 1,
    })
    assert res.status_code == 200
    assert res.body["usage"]["completion_tokens_details"]["accepted_prediction_tokens"] == 0
    content_no_pred = res.body["choices"][0]["message"]["content"]
    server.stop()

    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "I believe the meaning of life is"}],
        "temperature": 0.0,
        "top_k": 1,
        "prediction": {"content": '''"Here?" Annabyed.
"Okay, Annabyes!" Annabyed.
As Annagged, Annap came and said,'''}
    })
    assert res.status_code == 200
    assert res.body["usage"]["completion_tokens_details"]["accepted_prediction_tokens"] == 54
    content_pred = res.body["choices"][0]["message"]["content"]
    server.stop()

    assert content_no_pred == content_pred


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        res = server.make_request("POST", "/v1/chat/completions", data={
            "messages": [{"role": "user", "content": "I believe the meaning of life is"}],
            "temperature": 0.0,
            "top_k": 1,
            "prediction": {"content": " believe the meaning of life is"}
        })
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])
