#!/usr/bin/env python3
import pytest

from utils import ServerPreset, ServerProcess

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.jinja = True
    server.reasoning_budget = 1
    server.chat_template_file = "../../../models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja"
    server.reasoning_force_close_message = "!!!ABORT REASONING"
    server.start()
    yield
    server.stop()


def test_reasoning_budget_forces_close():
    """Ensure reasoning budget triggers forced close injection with end tag."""
    res = server.make_request("POST", "/v1/chat/completions", data={
        "model": server.model_alias or "test",
        "messages": [
            {"role": "user", "content": "Tell me a short story."},
        ],
        "max_tokens": 32,
    })

    assert res.status_code == 200
    body = res.body
    assert "choices" in body and body["choices"], "no choices returned"

    message = body["choices"][0]["message"]
    reasoning_content = message.get("reasoning_content", "")

    assert server.reasoning_force_close_message in reasoning_content, "reasoning force close message not found in reasoning content"

def test_reasoning_custom_budget():
    """Ensure reasoning budget triggers forced close injection with end tag."""
    res = server.make_request("POST", "/v1/chat/completions", data={
        "model": server.model_alias or "test",
        "messages": [
            {"role": "user", "content": "Tell me a short story."},
        ],
        "max_tokens": 32,
        "thinking_budget_tokens": 5
    })

    assert res.status_code == 200
    body = res.body
    assert "choices" in body and body["choices"], "no choices returned"

    message = body["choices"][0]["message"]
    reasoning_content = message.get("reasoning_content", "")
    
    reasoning_before_abort = reasoning_content.split(server.reasoning_force_close_message)[0]
    assert len(reasoning_before_abort.split()) > 1, "reasoning content too short before force close"

    assert server.reasoning_force_close_message in reasoning_content, "reasoning force close message not found in reasoning content"