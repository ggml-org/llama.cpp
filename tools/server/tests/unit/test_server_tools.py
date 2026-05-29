#!/usr/bin/env python
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_tools = ["question"]
    server.model_alias = "tinyllama-2-server-tools"
    server.server_port = 8082
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_batch = 512


def test_question_tool_pause_and_reply_roundtrip():
    global server
    server.start()

    response = server.make_request(
        "POST",
        "/tools",
        {
            "tool": "question",
            "params": {
                "questions": [
                    {
                        "question": "Pick one",
                        "header": "Choice",
                        "options": [
                            {"label": "A", "description": "Option A"},
                            {"label": "B", "description": "Option B"},
                        ],
                    }
                ]
            },
            "context": {
                "conversation_id": "conv-question-1",
                "tool_call_id": "call-question-1",
            },
        },
    )

    assert response.status_code == 200
    assert response.body["status"] == "awaiting_user"
    assert response.body["kind"] == "question"
    request_id = response.body["request_id"]

    pending = server.make_request("GET", "/tools/pending?conversation_id=conv-question-1")
    assert pending.status_code == 200
    assert len(pending.body) == 1
    assert pending.body[0]["request_id"] == request_id

    reply = server.make_request(
        "POST",
        "/tools/reply",
        {
            "request_id": request_id,
            "conversation_id": "conv-question-1",
            "answers": [["A"]],
        },
    )

    assert reply.status_code == 200
    assert reply.body["status"] == "completed"
    assert "Pick one" in reply.body["plain_text_response"]
    assert "A" in reply.body["plain_text_response"]

    pending_after = server.make_request("GET", "/tools/pending?conversation_id=conv-question-1")
    assert pending_after.status_code == 200
    assert pending_after.body == []


def test_question_tool_requires_conversation_context():
    global server
    server.start()

    response = server.make_request(
        "POST",
        "/tools",
        {
            "tool": "question",
            "params": {
                "questions": [
                    {
                        "question": "Need context",
                        "header": "Ctx",
                        "options": [],
                    }
                ]
            },
        },
    )

    assert response.status_code == 200
    assert response.body["error"] == "question requires context.conversation_id"

