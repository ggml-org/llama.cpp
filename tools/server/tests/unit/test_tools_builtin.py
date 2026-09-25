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

# image the container runtime tests run their shell in
CONTAINER_IMAGE = "busybox"


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


def test_tools_builtin_write_then_edit_file(tmp_path):
    global server
    server.start()

    log_path = str(tmp_path / "test.log")
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


def test_tools_builtin_edit_file_rejects_non_unique_old_text(tmp_path):
    global server
    server.start()

    log_path = str(tmp_path / "test.log")
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


def _container_engine_unavailable_reason(engine: str) -> str | None:
    """None if `engine` can run the image these tests use, otherwise the reason it can't."""
    engine_bin = shutil.which(engine)
    if engine_bin is None:
        return f"{engine} is not installed"
    try:
        # a daemon that answers `info` still cannot run a linux image when it serves windows
        # containers, so probe the image itself, which also pulls it before the tests
        subprocess.run([engine_bin, "run", "--rm", CONTAINER_IMAGE, "true"], capture_output=True, timeout=60, check=True)
    except Exception as e:
        return f"{engine} cannot run {CONTAINER_IMAGE}: {e}"
    return None


@pytest.fixture(params=["docker", "podman"])
def container_engine(request):
    engine = request.param
    reason = _container_engine_unavailable_reason(engine)
    if reason is not None:
        pytest.skip(reason)  # ty: ignore[too-many-positional-arguments, invalid-argument-type]
    return engine


@pytest.fixture
def container_id(container_engine: str):
    proc = subprocess.run(
        [container_engine, "run", "-d", "--rm", CONTAINER_IMAGE, "sleep", "300"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"failed to start {container_engine} container: {proc.stderr.strip()}")  # ty: ignore[too-many-positional-arguments, invalid-argument-type]

    cid = proc.stdout.strip()
    try:
        yield cid
    finally:
        subprocess.run([container_engine, "rm", "-f", cid], capture_output=True)


def test_tools_builtin_runtime_header(container_engine: str, container_id: str):
    global server
    server.start()

    headers = {"x-tool-runtime": f"{container_engine}-container:{container_id}", "x-tool-cwd": "/tmp"}

    write_res = call_tool("write_file", {"path": "test.log", "content": "hello container\n"}, headers=headers)
    assert write_res["result"] == "file written successfully"

    read_res = call_tool("read_file", {"path": "test.log"}, headers=headers)
    assert read_res["plain_text_response"] == "hello container\n"

    exec_res = call_tool("exec_shell_command", {"command": "cat test.log"}, headers=headers)
    assert "hello container" in exec_res["plain_text_response"]


def test_tools_builtin_runtime_header_unknown_scheme():
    global server
    server.start()

    # an unknown runtime must fail, never silently fall back to running on the host
    res = server.make_request("POST", "/tools",
                              data={"tool": "exec_shell_command", "params": {"command": "echo hi"}},
                              headers={"x-tool-runtime": "fake:does-not-exist"})
    assert res.status_code == 500, res.body
    assert "unknown tool runtime" in str(res.body)


def test_tools_builtin_runtime_header_rejects_ssh_option_injection():
    global server
    server.start()

    # ssh reads options from its argv, so a target starting with '-' must be rejected
    res = server.make_request("POST", "/tools",
                              data={"tool": "exec_shell_command", "params": {"command": "echo hi"}},
                              headers={"x-tool-runtime": "ssh:-oProxyCommand=touch /tmp/pwned"})
    assert res.status_code == 500, res.body
    assert "invalid ssh target" in str(res.body)


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_tools_builtin_runtime_header_rejects_container_option_injection(engine: str):
    global server
    server.start()

    # the container id lands on the `<engine> exec` command line, so an id that looks
    # like an option must be rejected
    res = server.make_request("POST", "/tools",
                              data={"tool": "exec_shell_command", "params": {"command": "echo hi"}},
                              headers={"x-tool-runtime": f"{engine}-container:--privileged"})
    assert res.status_code == 500, res.body
    assert "invalid container id" in str(res.body)


def test_tools_builtin_docker_runtime_cleans_up_spawned_container():
    # docker-only: this reads the container hostname to get the spawned id, which only docker
    # sets to the short id. podman is covered by the attach path above
    reason = _container_engine_unavailable_reason("docker")
    if reason is not None:
        pytest.skip(reason)  # ty: ignore[too-many-positional-arguments, invalid-argument-type]

    global server
    server.server_tools_runtime = f"docker:{CONTAINER_IMAGE}"
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


def test_tools_builtin_edit_file_rejects_overlapping_edits(tmp_path):
    global server
    server.start()

    log_path = str(tmp_path / "test.log")
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


def test_tools_builtin_file_glob_search_type_dir(tmp_path):
    global server
    server.start()

    (tmp_path / "project-alpha" / "src").mkdir(parents=True)
    (tmp_path / "project-alpha" / "README.md").write_text("alpha")
    (tmp_path / "project-alpha" / "src" / "main.cpp").write_text("int main() {}")
    (tmp_path / "project-beta").mkdir()
    (tmp_path / "project-beta" / "notes.txt").write_text("beta")

    res = call_tool("file_glob_search", {"path": str(tmp_path), "type": "dir"})
    text = res["plain_text_response"]
    assert "project-alpha/" in text
    assert "project-beta/" in text
    assert "project-alpha/src/" in text
    assert "README.md" not in text
    types = {e["path"]: e["type"] for e in res["entries"]}
    assert types["project-alpha"] == "dir"
    assert types["project-alpha/src"] == "dir"

    res_all = call_tool("file_glob_search", {"path": str(tmp_path), "type": "all", "include": "*proj*"})
    paths = [e["path"] for e in res_all["entries"]]
    assert "project-alpha" in paths
    assert "project-beta" in paths


def test_tools_builtin_file_glob_search_max_depth_and_limit(tmp_path):
    global server
    server.start()

    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "top.txt").write_text("top")
    (tmp_path / "a" / "mid.txt").write_text("mid")
    (tmp_path / "a" / "b" / "deep.txt").write_text("deep")

    res = call_tool("file_glob_search", {"path": str(tmp_path), "max_depth": 1})
    assert "top.txt" in res["plain_text_response"]
    assert "mid.txt" not in res["plain_text_response"]

    res = call_tool("file_glob_search", {"path": str(tmp_path), "max_depth": 2})
    assert "mid.txt" in res["plain_text_response"]
    assert "deep.txt" not in res["plain_text_response"]

    res = call_tool("file_glob_search", {"path": str(tmp_path), "limit": 1})
    assert len(res["entries"]) == 1
    assert "Total matches: 3" in res["plain_text_response"]


def test_tools_builtin_file_glob_search_junk_dirs(tmp_path):
    global server
    server.start()

    (tmp_path / "build" / "nested").mkdir(parents=True)
    (tmp_path / "build" / "artifact.txt").write_text("built")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cpp").write_text("int main() {}")

    # a junk directory stays selectable as a working directory
    res = call_tool("file_glob_search", {"path": str(tmp_path), "type": "dir", "max_depth": 1})
    assert "build" in [e["path"] for e in res["entries"]]

    # but it is never walked, so nothing inside it shows up
    res = call_tool("file_glob_search", {"path": str(tmp_path), "type": "all"})
    paths = [e["path"] for e in res["entries"]]
    assert "src/main.cpp" in paths
    assert "build/artifact.txt" not in paths
    assert "build/nested" not in paths


def test_tools_builtin_file_glob_search_rejects_invalid_type(tmp_path):
    global server
    server.start()

    err = call_tool_expect_error("file_glob_search", {"path": str(tmp_path), "type": "bogus"})
    assert "invalid type" in err


def test_tools_builtin_cwd_header_overrides_model_param(tmp_path):
    global server
    server.start()

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "marker.txt").write_text("marker")

    # a model-provided "cwd" in the params is overridden by the x-tool-cwd header
    res = call_tool("read_file", {"path": "marker.txt", "cwd": "/definitely/not/a/real/path"},
                    headers={"x-tool-cwd": str(workdir)})
    assert "marker" in res["plain_text_response"]


def test_tools_builtin_cwd_relative_paths(tmp_path):
    global server
    server.start()

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "rel.txt").write_text("relative-content")

    headers = {"x-tool-cwd": str(workdir)}

    # relative paths in file tools resolve against the header cwd
    res = call_tool("read_file", {"path": "rel.txt"}, headers=headers)
    assert "relative-content" in res["plain_text_response"]

    res = call_tool("write_file", {"path": "sub/out.txt", "content": "written"}, headers=headers)
    assert (workdir / "sub" / "out.txt").read_text() == "written"

    res = call_tool("file_glob_search", {"path": ".", "include": "*.txt"}, headers=headers)
    assert "rel.txt" in res["plain_text_response"]

    # absolute paths are unaffected by the cwd
    other = tmp_path / "other"
    other.mkdir()
    (other / "abs.txt").write_text("absolute-content")
    res = call_tool("read_file", {"path": str(other / "abs.txt")}, headers=headers)
    assert "absolute-content" in res["plain_text_response"]
