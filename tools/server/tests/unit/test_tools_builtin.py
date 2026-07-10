import os

import pytest
from utils import *

server: ServerProcess

# project root, used as the search directory for grep_search/file_glob_search
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

# marker for the grep_search test to find in this file
GREP_MARKER = "llama_cpp_test_tools_builtin_marker_grep_search"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()
    server.server_tools = "all"


def call_tool(name: str, params: dict) -> dict:
    res = server.make_request("POST", "/tools", data={"tool": name, "params": params})
    assert res.status_code == 200, res.body
    assert "error" not in res.body, res.body
    return res.body


def call_tool_expect_error(name: str, params: dict) -> str:
    res = server.make_request("POST", "/tools", data={"tool": name, "params": params})
    assert res.status_code == 200, res.body
    assert "error" in res.body, res.body
    return res.body["error"]


def test_tools_builtin_grep_search():
    global server
    server.start()

    res = call_tool("grep_search", {
        "path": PROJECT_ROOT,
        "pattern": GREP_MARKER,
        "include": "test_tools_builtin.py",  # bare pattern -> matches basename at any depth
    })
    text = res["plain_text_response"]
    assert "test_tools_builtin.py" in text
    assert GREP_MARKER in text
    assert "Total matches: 1" in text


def test_tools_builtin_read_file():
    global server
    server.start()

    this_file = os.path.join(PROJECT_ROOT, "tools", "server", "tests", "unit", "test_tools_builtin.py")
    res = call_tool("read_file", {"path": this_file})
    text = res["plain_text_response"]
    assert GREP_MARKER in text
    assert "def test_tools_builtin_read_file" in text


def test_tools_builtin_write_then_edit_file():
    global server
    server.start()

    log_path = os.path.join(PROJECT_ROOT, "test.log")
    try:
        write_res = call_tool("write_file", {"path": log_path, "content": "line1\nline2\nline3\n"})
        assert write_res["result"] == "file written successfully"

        read_before = call_tool("read_file", {"path": log_path})
        assert read_before["plain_text_response"] == "line1\nline2\nline3\n"

        edit_res = call_tool("edit_file", {
            "path": log_path,
            "edits": [
                {"old_text": "line2", "new_text": "line2-edited"},
                {"old_text": "line3\n", "new_text": "line3\nline4\n"},
            ],
        })
        assert edit_res["result"] == "file edited successfully"
        assert edit_res["edits_applied"] == 2

        read_after = call_tool("read_file", {"path": log_path})
        assert read_after["plain_text_response"] == "line1\nline2-edited\nline3\nline4\n"
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)


def test_tools_builtin_edit_file_rejects_non_unique_old_text():
    global server
    server.start()

    log_path = os.path.join(PROJECT_ROOT, "test.log")
    try:
        call_tool("write_file", {"path": log_path, "content": "dup\ndup\n"})
        err = call_tool_expect_error("edit_file", {
            "path": log_path,
            "edits": [{"old_text": "dup", "new_text": "changed"}],
        })
        assert "unique" in err
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)


def test_tools_builtin_edit_file_rejects_overlapping_edits():
    global server
    server.start()

    log_path = os.path.join(PROJECT_ROOT, "test.log")
    try:
        call_tool("write_file", {"path": log_path, "content": "line1\nline2\n"})
        err = call_tool_expect_error("edit_file", {
            "path": log_path,
            "edits": [
                {"old_text": "line1\nline2", "new_text": "a"},
                {"old_text": "line2", "new_text": "b"},
            ],
        })
        assert "overlap" in err
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)
