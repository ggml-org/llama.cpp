import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_ignore_eos_populates_logit_bias():
    """ignore_eos=true must add EOG logit biases to generation_settings."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "Once upon a time",
        "ignore_eos": True,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    # EOG token biases must be present with -inf bias
    logit_bias = res.body["generation_settings"]["logit_bias"]
    assert len(logit_bias) > 0
    for entry in logit_bias:
        assert entry["bias"] is None  # null in JSON represents -inf


def test_ignore_eos_false_no_logit_bias():
    """ignore_eos=false (default) must NOT add EOG logit biases."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "Once upon a time",
        "ignore_eos": False,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    logit_bias = res.body["generation_settings"]["logit_bias"]
    assert len(logit_bias) == 0


@pytest.mark.parametrize("constraint", [
    {"grammar": 'root ::= "ok"'},
    {"json_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
])
def test_ignore_eos_with_grammar_rejected(constraint):
    """ignore_eos is not supported together with a grammar"""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "Once upon a time",
        "ignore_eos": True,
        "temperature": 0.0,
        **constraint,
    })
    assert res.status_code == 400, res.body
    assert res.body["error"]["type"] == "invalid_request_error"
