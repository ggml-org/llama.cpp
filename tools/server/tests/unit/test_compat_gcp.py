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
                "stream": False,
            },
            {
                "@requestFormat": "chatCompletions",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Say world"}],
                "stream": False,
            },
        ],
    })
    assert res.status_code == 200
    assert len(res.body["predictions"]) == 2
    for prediction in res.body["predictions"]:
        assert "choices" in prediction
        assert len(prediction["choices"][0]["message"]["content"]) > 0


def test_gcp_predict_openai_chat_completion():
    global server
    server.start()
    res = server.make_request("POST", "/predict", data={
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"},
        ],
    })
    assert res.status_code == 200
    assert "predictions" not in res.body
    assert len(res.body["choices"]) == 1
    assert res.body["choices"][0]["message"]["role"] == "assistant"
    assert len(res.body["choices"][0]["message"]["content"]) > 0


@pytest.mark.parametrize("vertex_request", [False, True])
def test_gcp_predict_stream_chat_completion(vertex_request: bool):
    global server
    server.start()
    payload = {
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"},
        ],
        "stream": True,
    }
    if vertex_request:
        payload["@requestFormat"] = "chatCompletions"
        payload = {"instances": [payload]}

    content = ""
    for data in server.make_stream_request("POST", "/predict", data=payload):
        if data["choices"]:
            content += data["choices"][0]["delta"].get("content") or ""
    assert len(content) > 0


@pytest.mark.parametrize("vertex_request", [False, True])
def test_gcp_predict_stream_response_format(vertex_request: bool):
    global server
    server.start()
    payload = {
        "@requestFormat": "responses",
        "input": "What is the meaning of life?",
        "max_output_tokens": 8,
        "stream": True,
    }
    if vertex_request:
        payload = {"instances": [payload]}

    events = server.make_stream_request("POST", "/predict", data=payload)
    assert any(event["type"] == "response.completed" for event in events)


def test_gcp_predict_rejects_multiple_streaming_instances():
    global server
    server.start()
    res = server.make_request("POST", "/predict", data={
        "instances": [
            {
                "@requestFormat": "chatCompletions",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
            {
                "@requestFormat": "chatCompletions",
                "messages": [{"role": "user", "content": "Say world"}],
                "stream": False,
            },
        ],
    })
    assert res.status_code == 400
    assert "streaming is only supported for a single instance" in res.body["error"]["message"]
