import json
import os
import shutil
import subprocess

import pytest
from utils import *

server: ServerProcess

# image the container runtime tests run their shell in
CONTAINER_IMAGE = "busybox"

# set to an existing .sif to skip the pull, e.g. on a machine without network access
APPTAINER_IMAGE_ENV = "LLAMA_TEST_APPTAINER_IMAGE"
APPTAINER_SOURCE    = "docker://" + CONTAINER_IMAGE
# instances the server spawns for --tools-runtime "apptainer:<image>" are named llama-tools-<random>
APPTAINER_OWNED_PREFIX = "llama-tools-"


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


def _apptainer_unavailable_reason(image: str) -> str | None:
    """None if apptainer can run `image`, otherwise the reason it can't."""
    apptainer_bin = shutil.which("apptainer")
    if apptainer_bin is None:
        return "apptainer is not installed"
    try:
        subprocess.run([apptainer_bin, "exec", "--containall", image, "true"],
                       capture_output=True, timeout=60, check=True)
    except Exception as e:
        return f"apptainer cannot run {image}: {e}"
    return None


@pytest.fixture(scope="session")
def apptainer_image(tmp_path_factory) -> str:
    apptainer_bin = shutil.which("apptainer")
    if apptainer_bin is None:
        pytest.skip("apptainer is not installed")  # ty: ignore[too-many-positional-arguments, invalid-argument-type]

    image = os.environ.get(APPTAINER_IMAGE_ENV)
    if not image:
        # pulled once per session: converting an image to a SIF is slow
        image = str(tmp_path_factory.mktemp("apptainer") / f"{CONTAINER_IMAGE}.sif")
        try:
            subprocess.run([apptainer_bin, "pull", image, APPTAINER_SOURCE],
                           capture_output=True, timeout=300, check=True)
        except Exception as e:
            pytest.skip(f"cannot pull {APPTAINER_SOURCE}: {e}")  # ty: ignore[too-many-positional-arguments, invalid-argument-type]

    reason = _apptainer_unavailable_reason(image)
    if reason is not None:
        pytest.skip(reason)  # ty: ignore[too-many-positional-arguments, invalid-argument-type]
    return image


def _owned_apptainer_instances() -> set[str]:
    """Names of the instances llama-server spawned for --tools-runtime (llama-tools-*)."""
    res = subprocess.run(["apptainer", "instance", "list", "--json"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    instances = json.loads(res.stdout).get("instances") or []
    return {i["instance"] for i in instances if i.get("instance", "").startswith(APPTAINER_OWNED_PREFIX)}


@pytest.fixture
def apptainer_instance(apptainer_image: str):
    # same flags the server uses for the instance it owns
    name = f"llama-test-{os.getpid()}"
    proc = subprocess.run(
        ["apptainer", "instance", "start", "--containall", "--writable-tmpfs", apptainer_image, name],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"failed to start apptainer instance: {proc.stderr.strip()}")  # ty: ignore[too-many-positional-arguments, invalid-argument-type]
    try:
        yield name
    finally:
        subprocess.run(["apptainer", "instance", "stop", name], capture_output=True)


def test_tools_builtin_apptainer_runtime_header(apptainer_instance: str):
    global server
    server.start()

    # /tmp is the writable in-memory overlay of the instance, and exists in every image
    headers = {"x-tool-runtime": f"apptainer-instance:{apptainer_instance}", "x-tool-cwd": "/tmp"}

    write_res = call_tool("write_file", {"path": "test.log", "content": "hello apptainer\n"}, headers=headers)
    assert write_res["result"] == "file written successfully"

    # state persists between calls, since every call is an `exec` into the same instance
    read_res = call_tool("read_file", {"path": "test.log"}, headers=headers)
    assert read_res["plain_text_response"] == "hello apptainer\n"

    exec_res = call_tool("exec_shell_command", {"command": "cat test.log"}, headers=headers)
    assert "hello apptainer" in exec_res["plain_text_response"]


@pytest.mark.parametrize("runtime,expected_error", [
    # the header must never make the server pull or start an image, whatever its form
    ("apptainer:docker://alpine",        "must name a running container or instance"),
    ("apptainer:alpine.sif",             "must name a running container or instance"),
    ("apptainer:--writable",             "must name a running container or instance"),
    # option injection: the name lands on the `apptainer exec` command line
    ("apptainer-instance:--ipc",         "invalid container id"),
    ("apptainer-instance:-o--privileged", "invalid container id"),
    ("apptainer-instance:",              "invalid container id"),
    ("apptainer-instance:a b",           "invalid container id"),
    ("apptainer-instance:../x",          "invalid container id"),
    # apptainer has instances, not containers
    ("apptainer-container:foo",          "unknown tool runtime"),
])
def test_tools_builtin_apptainer_runtime_header_rejected(runtime: str, expected_error: str):
    global server
    server.start()

    res = server.make_request("POST", "/tools",
                              data={"tool": "exec_shell_command", "params": {"command": "echo pwned"}},
                              headers={"x-tool-runtime": runtime})
    assert res.status_code == 500, res.body
    assert expected_error in str(res.body)
    assert "pwned" not in str(res.body)


def test_tools_builtin_apptainer_runtime_header_cannot_spawn(apptainer_image: str):
    global server
    before = _owned_apptainer_instances()
    server.start()

    # a valid, runnable image in the header must still be refused, and nothing may be started
    res = server.make_request("POST", "/tools",
                              data={"tool": "exec_shell_command", "params": {"command": "echo pwned"}},
                              headers={"x-tool-runtime": f"apptainer:{apptainer_image}"})
    assert res.status_code == 500, res.body
    assert "must name a running container or instance" in str(res.body)
    assert "pwned" not in str(res.body)
    assert _owned_apptainer_instances() == before


def test_tools_builtin_apptainer_runtime_isolation_and_cleanup(apptainer_image: str):
    global server
    # a file in the real $HOME of the server user, which plain `apptainer exec` would bind mount
    canary = os.path.join(os.path.expanduser("~"), f".llama-test-canary-{os.getpid()}")
    with open(canary, "w") as f:
        f.write("host-secret")

    before = _owned_apptainer_instances()
    try:
        server.server_tools_runtime = f"apptainer:{apptainer_image}"
        server.start()

        started = _owned_apptainer_instances() - before
        assert len(started) == 1, started

        # only reads: if isolation were broken, this test must not be the one to damage $HOME
        res = call_tool("exec_shell_command", {
            "command": f"cat {canary}; [ -e {canary} ] && echo VISIBLE || echo HIDDEN; pwd",
        })
        text = res["plain_text_response"]
        assert "host-secret" not in text, text
        assert "HIDDEN" in text and "VISIBLE" not in text, text
        # --containall does not mount the host cwd: without --pwd apptainer warns that it cannot chdir to it
        assert "WARNING" not in text, text
        assert "/tmp" in text.splitlines(), text

        server.stop()

        # a clean server shutdown must stop the instance it owns, not leave it behind
        leftover = _owned_apptainer_instances() & started
        assert not leftover, f"instance {leftover} was not cleaned up after server exit"
    finally:
        if os.path.exists(canary):
            os.remove(canary)


def test_tools_builtin_apptainer_runtime_respawns_dead_instance(apptainer_image: str):
    global server
    before = _owned_apptainer_instances()
    server.server_tools_runtime = f"apptainer:{apptainer_image}"
    server.start()

    started = _owned_apptainer_instances() - before
    assert len(started) == 1, started
    (dead,) = started

    # kill the instance from the outside: the next call must notice and start a new one
    subprocess.run(["apptainer", "instance", "stop", dead], capture_output=True, check=True)

    res = call_tool("exec_shell_command", {"command": "echo respawned"})
    assert "respawned" in res["plain_text_response"]

    now = _owned_apptainer_instances() - before
    assert len(now) == 1 and dead not in now, now

    server.stop()
    assert not (_owned_apptainer_instances() - before)
