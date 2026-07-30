#!/usr/bin/env python3
"""
Tests for the /filesystem/* endpoints (search, roots, git).

Invariants verified:
1. Endpoints are disabled (501) without --tools
2. /filesystem/roots returns the configured --browse-root as default
3. /filesystem/search ranks exact/prefix/substring matches and honors
   type, match, limit, max_depth, show_hidden and context path filters
4. Paths outside the configured browse roots are rejected (400)
5. /filesystem/git detects a repository and reports its branch; the
   not-a-repo case is a 200 with is_repo=false
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from utils import *

# browse-root tree built once per module:
#   <tmp>/project-alpha/   git repo on branch "main", with src/{main,util}.cpp
#   <tmp>/project-beta/    plain directory with notes.txt
#   <tmp>/.hidden/         hidden directory with secret.txt
#   <tmp>/README.md
ROOT: str


def _git_available() -> bool:
    return shutil.which("git") is not None


def _start_server(tools: bool = True, port: int = 8091, env: dict | None = None) -> ServerProcess:
    srv = ServerPreset.router()
    srv.no_ui = True
    srv.server_port = port
    if tools:
        srv.server_tools = "all"
        srv.browse_root = ROOT
    if env:
        srv.env_extra = env
    srv.start()
    return srv


@pytest.fixture(scope="module", autouse=True)
def setup_tree():
    global ROOT
    with tempfile.TemporaryDirectory(prefix="llama-fs-test-") as tmp:
        ROOT = os.path.realpath(tmp)

        os.makedirs(os.path.join(ROOT, "project-alpha", "src"))
        os.makedirs(os.path.join(ROOT, "project-beta"))
        os.makedirs(os.path.join(ROOT, ".hidden"))
        with open(os.path.join(ROOT, "README.md"), "w") as f:
            f.write("hello\n")
        with open(os.path.join(ROOT, "project-alpha", "src", "main.cpp"), "w") as f:
            f.write("int main;\n")
        with open(os.path.join(ROOT, "project-alpha", "src", "util.cpp"), "w") as f:
            f.write("int util;\n")
        with open(os.path.join(ROOT, "project-beta", "notes.txt"), "w") as f:
            f.write("notes\n")
        with open(os.path.join(ROOT, ".hidden", "secret.txt"), "w") as f:
            f.write("secret\n")

        if _git_available():
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=os.path.join(ROOT, "project-alpha"), check=True)

        yield


def _search(server: ServerProcess, **kwargs):
    return server.make_request("POST", "/v1/filesystem/search", data=kwargs)


def _git(server: ServerProcess, **kwargs):
    return server.make_request("POST", "/v1/filesystem/git", data=kwargs)


def test_roots_returns_configured_root():
    server = _start_server()
    res = server.make_request("GET", "/v1/filesystem/roots")
    assert res.status_code == 200
    roots = res.body["roots"]
    assert len(roots) == 1
    assert roots[0]["default"] is True
    assert os.path.realpath(roots[0]["path"]) == os.path.realpath(ROOT)


def test_search_substring_finds_entries():
    server = _start_server()
    res = _search(server, query="project")
    assert res.status_code == 200
    names = [r["name"] for r in res.body["results"]]
    assert "project-alpha" in names
    assert "project-beta" in names


def test_search_exact_ranks_before_prefix():
    with open(os.path.join(ROOT, "project"), "w") as f:
        f.write("x\n")
    try:
        server = _start_server()
        res = _search(server, query="project")
        assert res.status_code == 200
        names = [r["name"] for r in res.body["results"]]
        assert names[0] == "project"
    finally:
        os.remove(os.path.join(ROOT, "project"))


def test_search_type_filter():
    server = _start_server()
    res = _search(server, query="main", type="file")
    assert res.status_code == 200
    assert [r["name"] for r in res.body["results"]] == ["main.cpp"]
    assert res.body["results"][0]["type"] == "file"
    assert "size" in res.body["results"][0]

    res = _search(server, query="project", type="directory")
    assert res.status_code == 200
    assert all(r["type"] == "directory" for r in res.body["results"])
    assert len(res.body["results"]) == 2


def test_search_prefix_match_mode():
    server = _start_server()
    res = _search(server, query="proj", match="prefix")
    assert res.status_code == 200
    names = [r["name"] for r in res.body["results"]]
    assert "project-alpha" in names
    # substring-only matches are excluded in prefix mode
    assert "README.md" not in names


def test_search_pathlike_query():
    server = _start_server()
    res = _search(server, query="alpha/main")
    assert res.status_code == 200
    names = [r["name"] for r in res.body["results"]]
    assert "main.cpp" in names


def test_search_absolute_path_query():
    server = _start_server()
    # pasting a full path under the root matches like the relative form
    res = _search(
        server,
        query=os.path.join(os.path.realpath(server.browse_root), "project-alpha", "src", "main.cpp"),
    )
    assert res.status_code == 200
    names = [r["name"] for r in res.body["results"]]
    assert "main.cpp" in names


def _start_home_server(port: int = 8093) -> ServerProcess:
    # HOME pointed at the fixture tree so "~" queries resolve inside it
    return _start_server(port=port, env={"HOME": ROOT, "USERPROFILE": ROOT})


def test_search_tilde_expands_to_home():
    server = _start_home_server()
    res = _search(server, query="~")
    assert res.status_code == 200
    names = [r["name"] for r in res.body["results"]]
    assert "project-alpha" in names
    assert "project-beta" in names


def test_search_tilde_pathlike_query():
    server = _start_home_server()
    res = _search(server, query="~/alpha/src/main")
    assert res.status_code == 200
    names = [r["name"] for r in res.body["results"]]
    assert "main.cpp" in names


def test_search_tilde_without_slash_not_expanded():
    server = _start_home_server()
    res = _search(server, query="~alpha")
    assert res.status_code == 200
    assert res.body["results"] == []


def test_search_tilde_reroots_outside_context_path():
    server = _start_home_server()
    # scoped to project-alpha, but "~" must search from the browse root
    res = _search(
        server,
        query="~",
        path=os.path.join(os.path.realpath(server.browse_root), "project-alpha"),
    )
    assert res.status_code == 200
    names = [r["name"] for r in res.body["results"]]
    assert "project-beta" in names


def test_search_limit():
    server = _start_server()
    res = _search(server, query="project", limit=1)
    assert res.status_code == 200
    assert len(res.body["results"]) == 1

    res = _search(server, query="project", limit=201)
    assert res.status_code == 400


def test_search_hidden_excluded_by_default():
    server = _start_server()
    res = _search(server, query="secret")
    assert res.status_code == 200
    assert res.body["results"] == []

    res = _search(server, query="secret", show_hidden=True)
    assert res.status_code == 200
    assert [r["name"] for r in res.body["results"]] == ["secret.txt"]


def test_search_rejects_empty_query():
    server = _start_server()
    res = _search(server, query="")
    assert res.status_code == 400


def test_search_rejects_bad_type_and_match():
    server = _start_server()
    assert _search(server, query="x", type="bogus").status_code == 400
    assert _search(server, query="x", match="bogus").status_code == 400


def test_search_context_path_outside_roots_rejected():
    server = _start_server()
    res = _search(server, query="x", path="/etc" if os.name != "nt" else "C:\\Windows")
    assert res.status_code == 400


def test_search_inside_context_path():
    server = _start_server()
    res = _search(server, query="util", path=os.path.join(ROOT, "project-alpha", "src"))
    assert res.status_code == 200
    assert [r["name"] for r in res.body["results"]] == ["util.cpp"]


@pytest.mark.skipif(not _git_available(), reason="git not installed")
def test_git_detects_repo_and_branch():
    server = _start_server()
    res = _git(server, path=os.path.join(ROOT, "project-alpha", "src"))
    assert res.status_code == 200
    assert res.body["is_repo"] is True
    assert os.path.realpath(res.body["root"]) == os.path.realpath(os.path.join(ROOT, "project-alpha"))
    assert res.body["branch"] == "main"


def test_git_non_repo_is_200_with_is_repo_false():
    server = _start_server()
    res = _git(server, path=os.path.join(ROOT, "project-beta"))
    assert res.status_code == 200
    assert res.body["is_repo"] is False
    assert res.body["branch"] == ""


def test_git_path_outside_roots_rejected():
    server = _start_server()
    res = _git(server, path="/etc" if os.name != "nt" else "C:\\Windows")
    assert res.status_code == 400


def test_endpoints_disabled_without_tools():
    server = _start_server(tools=False, port=8092)
    res = server.make_request("GET", "/v1/filesystem/roots")
    assert res.status_code == 501
    assert res.body["error"]["type"] == "not_supported_error"

    assert _search(server, query="x").status_code == 501
    assert _git(server, path="/tmp").status_code == 501
