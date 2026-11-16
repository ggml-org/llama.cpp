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


def test_ignore_eos_with_response_format_rejected():
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
        "ignore_eos": True,
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    })
    assert res.status_code == 400, res.body
    assert res.body["error"]["type"] == "invalid_request_error"


def test_eog_ban_with_grammar_rejected():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "Once upon a time",
        "grammar": 'root ::= "ok"',
        "logit_bias": [[2, False]],  # 2 is EOS for this model
    })
    assert res.status_code == 400, res.body
    assert res.body["error"]["type"] == "invalid_request_error"


def test_ignore_eos_with_tools_rejected():
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
        "ignore_eos": True,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}],
    })
    assert res.status_code == 400, res.body
    assert res.body["error"]["type"] == "invalid_request_error"
