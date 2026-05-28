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
    server.server_tools = ["question", "todowrite", "artifact_create", "artifact_edit"]
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


def test_todowrite_returns_json_text_snapshot():
    global server
    server.start()

    response = server.make_request(
        "POST",
        "/tools",
        {
            "tool": "todowrite",
            "params": {
                "todos": [
                    {"content": "Inspect files", "status": "completed"},
                    {"content": "Patch server", "status": "in_progress"},
                ]
            },
        },
    )

    assert response.status_code == 200
    assert response.body["status"] == "completed"
    assert '"content": "Inspect files"' in response.body["plain_text_response"]
    assert '"status": "in_progress"' in response.body["plain_text_response"]


def test_artifact_create_and_edit_roundtrip():
    global server
    server.start()

    create = server.make_request(
        "POST",
        "/tools",
        {
            "tool": "artifact_create",
            "params": {
                "name": "plan.md",
                "mime_type": "text/markdown",
                "content": "# Plan\n\nShip it"
            },
            "context": {
                "conversation_id": "conv-artifact-1",
                "tool_call_id": "call-artifact-1",
            },
        },
    )

    assert create.status_code == 200
    assert create.body["status"] == "completed"
    assert create.body["artifact_id"].startswith("artifact-")
    assert len(create.body["attachments"]) == 1
    attachment = create.body["attachments"][0]
    assert attachment["type"] == "TEXT"
    assert attachment["presentation"] == "artifact"
    assert attachment["artifactId"] == create.body["artifact_id"]
    assert attachment["content"] == "# Plan\n\nShip it"

    edit = server.make_request(
        "POST",
        "/tools",
        {
            "tool": "artifact_edit",
            "params": {
                "artifact_id": create.body["artifact_id"],
                "content": "# Plan\n\nShipped"
            },
            "context": {
                "conversation_id": "conv-artifact-1",
                "tool_call_id": "call-artifact-2",
            },
        },
    )

    assert edit.status_code == 200
    assert edit.body["status"] == "completed"
    assert edit.body["artifact_id"] == create.body["artifact_id"]
    assert edit.body["attachments"][0]["content"] == "# Plan\n\nShipped"


def test_artifact_tools_require_conversation_context_and_existing_id():
    global server
    server.start()

    create = server.make_request(
        "POST",
        "/tools",
        {
            "tool": "artifact_create",
            "params": {
                "name": "note.txt",
                "mime_type": "text/plain",
                "content": "hello"
            },
        },
    )

    assert create.status_code == 200
    assert create.body["error"] == "artifact_create requires context.conversation_id"

    edit = server.make_request(
        "POST",
        "/tools",
        {
            "tool": "artifact_edit",
            "params": {
                "artifact_id": "artifact-missing",
                "content": "hello"
            },
            "context": {
                "conversation_id": "conv-artifact-missing",
                "tool_call_id": "call-artifact-missing",
            },
        },
    )

    assert edit.status_code == 200
    assert edit.body["error"] == "Artifact not found: artifact-missing"
