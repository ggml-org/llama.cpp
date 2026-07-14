import pytest
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.gcp_compat = True


def test_gcp_predict_camel_case():
    global server
    server.start()
    res = server.make_request("POST", "/predict", data={
        "instances": [
            {
                "@requestFormat": "chatCompletions",
                "max_tokens": 8,
                "messages": [
                    {"role": "user", "content": "What is the meaning of life?"},
                ],
            }
        ],
    })
    assert res.status_code == 200
    assert "predictions" in res.body
    assert len(res.body["predictions"]) == 1
    prediction = res.body["predictions"][0]
    assert "choices" in prediction
    assert len(prediction["choices"]) == 1
    assert prediction["choices"][0]["message"]["role"] == "assistant"
    assert len(prediction["choices"][0]["message"]["content"]) > 0


def test_gcp_predict_multiple_instances():
    global server
    server.n_slots = 2
    server.start()
    res = server.make_request("POST", "/predict", data={
        "instances": [
            {
                "@requestFormat": "chatCompletions",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Say hello"}],
            },
            {
                "@requestFormat": "chatCompletions",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Say world"}],
            },
        ],
    })
    assert res.status_code == 200
    assert len(res.body["predictions"]) == 2
    for prediction in res.body["predictions"]:
        assert "choices" in prediction
        assert len(prediction["choices"][0]["message"]["content"]) > 0
