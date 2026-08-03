import os
import shutil
import subprocess

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


def call_tool(name: str, params: dict, headers: dict | None = None) -> dict:
    res = server.make_request("POST", "/tools", data={"tool": name, "params": params}, headers=headers)
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


def test_tools_builtin_exec_shell_command_stream():
    global server
    server.start()

    events = list(server.make_stream_request("POST", "/tools", data={
        "tool": "exec_shell_command",
        "params": {"command": "echo hello"},
        "stream": True,
    }))

    assert len(events) >= 2
    assert events[-1]["done"] is True
    assert not events[-1].get("error")
    chunks = "".join(e["chunk"] for e in events[:-1])
    assert "hello" in chunks
    assert "[exit code: 0]" in chunks


def test_tools_builtin_cwd_header():
    global server
    server.start()

    cwd_dir = os.path.join(PROJECT_ROOT, "tools", "server", "tests", "unit")
    headers = {"x-tool-cwd": cwd_dir}

    res = call_tool("read_file", {"path": "test_tools_builtin.py"}, headers=headers)
    assert GREP_MARKER in res["plain_text_response"]

    # exec_shell_command should also run with that directory as its working directory:
    # writing to a relative filename must land inside cwd_dir
    marker_name = "llama_cpp_test_tools_builtin_cwd_marker.txt"
    marker_path = os.path.join(cwd_dir, marker_name)
    try:
        command = f"echo hello > {marker_name}"
        call_tool("exec_shell_command", {"command": command}, headers=headers)
        assert os.path.exists(marker_path)
    finally:
        if os.path.exists(marker_path):
            os.remove(marker_path)


def _docker_unavailable_reason() -> str | None:
    """None if docker can be used to run a container, otherwise the reason it can't."""
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        return "docker is not installed"
    try:
        subprocess.run([docker_bin, "info"], capture_output=True, timeout=5, check=True)
    except Exception as e:
        return f"docker daemon is not usable: {e}"
    return None


@pytest.fixture
def docker_container():
    reason = _docker_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)  # ty: ignore[too-many-positional-arguments, invalid-argument-type]

    proc = subprocess.run(
        ["docker", "run", "-d", "--rm", "busybox", "sleep", "300"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"failed to start docker container: {proc.stderr.strip()}")  # ty: ignore[too-many-positional-arguments, invalid-argument-type]

    container_id = proc.stdout.strip()
    try:
        yield container_id
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


def test_tools_builtin_docker_header(docker_container: str):
    global server
    server.start()

    headers = {"x-tool-docker": docker_container, "x-tool-cwd": "/tmp"}

    write_res = call_tool("write_file", {"path": "test.log", "content": "hello docker\n"}, headers=headers)
    assert write_res["result"] == "file written successfully"

    read_res = call_tool("read_file", {"path": "test.log"}, headers=headers)
    assert read_res["plain_text_response"] == "hello docker\n"

    exec_res = call_tool("exec_shell_command", {"command": "cat test.log"}, headers=headers)
    assert "hello docker" in exec_res["plain_text_response"]


def test_tools_builtin_docker_runtime_cleans_up_spawned_container():
    reason = _docker_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)  # ty: ignore[too-many-positional-arguments, invalid-argument-type]

    global server
    server.server_tools_runtime = "docker:busybox"
    server.start()

    # exec_shell_command runs inside the container spawned for --tools-runtime; docker sets
    # the container's hostname to its own short id, so this also tells us which one to check
    res = call_tool("exec_shell_command", {"command": "hostname"})
    container_id = res["plain_text_response"].splitlines()[0].strip()
    assert len(container_id) >= 8, res

    running = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
        capture_output=True, text=True,
    )
    assert running.returncode == 0 and running.stdout.strip() == "true", running.stderr

    server.stop()

    # a clean server shutdown must stop and remove the container it spawned (it runs with --rm),
    # not leave it behind as an abandoned child
    leftover = subprocess.run(["docker", "inspect", container_id], capture_output=True, text=True)
    assert leftover.returncode != 0, f"container {container_id} was not cleaned up after server exit"


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
