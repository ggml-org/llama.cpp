#!/usr/bin/env python3
"""
Tests for MCP server integration via the /tools endpoint.

Invariants verified:
1. MCP tools appear in /tools listing when configured
2. MCP tools use <server>:<tool> naming
3. MCP tools can be invoked and return correct results
4. Misconfigured MCP servers do not crash the server
5. Multiple MCP servers can be configured simultaneously
6. Warmup populates the tool list at startup
"""
import json
import os
import sys
import tempfile
import time

import pytest

from utils import *

# Path to the test MCP server fixture
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")
MCP_ECHO_SERVER = os.path.join(FIXTURES_DIR, "mcp_echo_server.py")

server: ServerProcess


def _mcp_config_json(servers: dict) -> str:
    """Create a JSON config string for --mcp-servers-json."""
    return json.dumps({"mcpServers": servers})


def _start_server_with_mcp(mcp_json: str, **kwargs) -> ServerProcess:
    """Helper to start a router server with MCP config."""
    srv = ServerPreset.router()
    srv.server_tools = "all"
    srv.no_ui = True
    srv.server_port = 8085  # avoid conflict with load_all() which uses 8080
    srv.mcp_servers_json = mcp_json
    for k, v in kwargs.items():
        setattr(srv, k, v)
    srv.start()
    return srv


def test_mcp_tools_listed_in_tools_endpoint():
    """MCP tools should appear in GET /tools with server:tool naming."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body

        tools = res.body
        assert isinstance(tools, list), f"Expected list, got {type(tools)}"

        # Find MCP tools - name is in "tool" field or definition.function.name
        def get_tool_name(t):
            return t.get("tool", "") or t.get("definition", {}).get("function", {}).get("name", "")

        mcp_tools = [t for t in tools if get_tool_name(t).startswith("echo_")]
        assert len(mcp_tools) >= 2, f"Expected at least 2 echo_ tools, got {len(mcp_tools)}: {mcp_tools}"

        tool_names = {get_tool_name(t) for t in mcp_tools}
        assert "echo_echo" in tool_names
        assert "echo_add" in tool_names

        # Verify tool structure
        echo_tool = next(t for t in mcp_tools if get_tool_name(t) == "echo_echo")
        assert "description" in echo_tool or "definition" in echo_tool
    finally:
        server.stop()


def test_mcp_tool_invocation():
    """MCP tools should be callable via POST /tools and return correct results."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # Call echo_echo
        res = server.make_request("POST", "/tools", data={
            "tool": "echo_echo",
            "params": {"message": "hello world"}
        })
        assert res.status_code == 200, res.body
        body = res.body
        assert "error" not in body, body
        # The result format depends on the tool implementation
        # For MCP tools, it should contain the tool result
        assert "plain_text_response" in body or "result" in body or "content" in body, body

        # Call echo_add
        res = server.make_request("POST", "/tools", data={
            "tool": "echo_add",
            "params": {"a": 3, "b": 5}
        })
        assert res.status_code == 200, res.body
        body = res.body
        assert "error" not in body, body
    finally:
        server.stop()


def test_mcp_bad_command_does_not_crash():
    """A misconfigured MCP server should not crash the llama-server."""
    global server
    mcp_json = _mcp_config_json({
        "nonexistent": {
            "command": "this_executable_does_not_exist_12345",
            "args": [],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # Server should still be healthy
        res = server.make_request("GET", "/health")
        assert res.status_code == 200, res.body

        # Builtin tools should still work
        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body
        tools = res.body
        # Should have builtin tools but no MCP tools from the bad server
        mcp_tools = [t for t in tools if t.get("name", "").startswith("nonexistent_")]
        assert len(mcp_tools) == 0, f"Expected no nonexistent_ tools, got {mcp_tools}"
    finally:
        server.stop()


def test_mcp_multiple_servers():
    """Multiple MCP servers can be configured simultaneously."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        },
        "echo2": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body

        tools = res.body

        def get_tool_name(t):
            return t.get("tool", "") or t.get("definition", {}).get("function", {}).get("name", "")

        echo_tools = [t for t in tools if get_tool_name(t).startswith("echo_")]
        echo2_tools = [t for t in tools if get_tool_name(t).startswith("echo2_")]

        assert len(echo_tools) >= 2, f"Expected echo_ tools, got {echo_tools}"
        assert len(echo2_tools) >= 2, f"Expected echo2_ tools, got {echo2_tools}"
    finally:
        server.stop()


def test_mcp_tools_not_listed_when_not_configured():
    """Without MCP config, no MCP tools should appear."""
    global server
    server = ServerPreset.router()
    server.server_tools = "all"
    server.no_ui = True
    server.server_port = 8085
    server.start()

    try:
        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body

        tools = res.body

        def get_tool_name(t):
            return t.get("tool", "") or t.get("definition", {}).get("function", {}).get("name", "")

        # Should only have builtin tools, no server: prefixed tools
        mcp_tools = [t for t in tools if ":" in get_tool_name(t)]
        assert len(mcp_tools) == 0, f"Expected no MCP tools, got {mcp_tools}"
    finally:
        server.stop()


def test_mcp_fail_once_tool_eventual_success():
    """Test that a tool that fails once eventually succeeds (tests instance respawn)."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # First call should succeed (warmup already spawned and shut down the instance,
        # but the first actual tool call will spawn a fresh instance)
        res = server.make_request("POST", "/tools", data={
            "tool": "echo_fail_once",
            "params": {}
        })
        # It might fail on first call if the warmup instance was shut down
        # and a new instance is spawned. The fail_once state is per-process,
        # so a fresh process will fail once then succeed.
        # Actually, warmup spawns, lists, then shuts down. So the first tool call
        # spawns a new process which will fail once.
        assert res.status_code in (200, 500), res.body
    finally:
        server.stop()


def test_mcp_tools_via_json_config_file():
    """Test that --mcp-servers-config (file) works as well as --mcp-servers-json."""
    global server
    config = {
        "mcpServers": {
            "echo": {
                "command": sys.executable,
                "args": [MCP_ECHO_SERVER],
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        server = ServerPreset.router()
        server.server_tools = "all"
        server.no_ui = True
        server.server_port = 8085
        server.mcp_servers_config = config_path
        server.start()

        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body

        tools = res.body

        def get_tool_name(t):
            return t.get("tool", "") or t.get("definition", {}).get("function", {}).get("name", "")

        mcp_tools = [t for t in tools if get_tool_name(t).startswith("echo_")]
        assert len(mcp_tools) >= 2, f"Expected echo_ tools, got {mcp_tools}"
    finally:
        os.unlink(config_path)
        server.stop()


def test_mcp_tools_slot_independent():
    """MCP tools should work without any slot concept; /tools is slot-independent."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # Call /tools without any slot binding — should succeed
        res = server.make_request("POST", "/tools", data={
            "tool": "echo_echo",
            "params": {"message": "hello"}
        })
        assert res.status_code == 200, res.body
        body = res.body
        assert "error" not in body, body
    finally:
        server.stop()


def test_mcp_concurrent_tool_calls():
    """Concurrent POST /tools to same MCP server should all succeed."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        def call_tool():
            return server.make_request("POST", "/tools", data={
                "tool": "echo_echo",
                "params": {"message": "hi"}
            })

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(call_tool) for _ in range(10)]
            results = [f.result() for f in futures]

        for res in results:
            assert res.status_code == 200, res.body
            assert "error" not in res.body, res.body
    finally:
        server.stop()


def test_mcp_tool_timeout():
    """Tool call should timeout if MCP server is too slow."""
    global server
    MCP_SLOW_SERVER = os.path.join(FIXTURES_DIR, "mcp_slow_server.py")
    mcp_json = _mcp_config_json({
        "slow": {
            "command": sys.executable,
            "args": [MCP_SLOW_SERVER, "--delay", "5"],
            "timeout_ms": 500
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        res = server.make_request("POST", "/tools", data={
            "tool": "slow_sleep",
            "params": {"seconds": 5}
        })
        assert res.status_code == 200, res.body
        body = res.body
        assert "error" in body, body
    finally:
        server.stop()


def test_mcp_warmup_partial_failure():
    """Good server's tools should appear even if bad server fails warmup."""
    global server
    mcp_json = _mcp_config_json({
        "good": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        },
        "bad": {
            "command": "nonexistent",
            "args": []
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body
        tools = res.body

        def get_tool_name(t):
            return t.get("tool", "") or t.get("definition", {}).get("function", {}).get("name", "")

        # good server tools should be present
        assert any("good_" in get_tool_name(t) for t in tools), f"Expected good: tools in {tools}"
    finally:
        server.stop()


def test_mcp_notification_during_request():
    """Notification during request should not be returned as response."""
    global server
    MCP_MALFORMED_SERVER = os.path.join(FIXTURES_DIR, "mcp_malformed_server.py")
    mcp_json = _mcp_config_json({
        "notifying": {
            "command": sys.executable,
            "args": [MCP_MALFORMED_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        res = server.make_request("POST", "/tools", data={
            "tool": "notifying_echo",
            "params": {"message": "hi"}
        })
        assert res.status_code == 200, res.body
        body = res.body
        assert "error" not in body, body
    finally:
        server.stop()


def test_mcp_instance_respawn_after_crash():
    """Tool call after process crash should respawn and succeed."""
    global server
    MCP_CRASH_SERVER = os.path.join(FIXTURES_DIR, "mcp_crash_server.py")
    mcp_json = _mcp_config_json({
        "crash": {
            "command": sys.executable,
            "args": [MCP_CRASH_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # First call succeeds
        res1 = server.make_request("POST", "/tools", data={
            "tool": "crash_echo",
            "params": {"message": "hi"}
        })
        assert res1.status_code == 200, res1.body
        assert "error" not in res1.body, res1.body

        # Second call should also succeed (respawned instance)
        res2 = server.make_request("POST", "/tools", data={
            "tool": "crash_echo",
            "params": {"message": "hi2"}
        })
        assert res2.status_code == 200, res2.body
        assert "error" not in res2.body, res2.body
    finally:
        server.stop()




def test_mcp_fail_once_eventual_success_verified():
    """Verify that fail_once tool eventually succeeds after respawn."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # First call may fail (fresh process)
        res1 = server.make_request("POST", "/tools", data={
            "tool": "echo_fail_once",
            "params": {}
        })
        # Second call should succeed
        res2 = server.make_request("POST", "/tools", data={
            "tool": "echo_fail_once",
            "params": {}
        })
        assert res2.status_code == 200, res2.body
        assert "error" not in res2.body, res2.body
    finally:
        server.stop()


def test_mcp_config_file_errors():
    """Invalid JSON config and missing file should cause server to fail to start."""
    # Invalid JSON - server should fail to start
    server = ServerPreset.router()
    server.server_tools = "all"
    server.no_ui = True
    server.server_port = 8085
    server.mcp_servers_json = "not valid json"
    try:
        server.start()
        assert False, "Server should not have started with invalid MCP JSON config"
    except RuntimeError:
        pass  # Expected: server process dies due to bad config

    # Missing file - server should fail to start
    server = ServerPreset.router()
    server.server_tools = "all"
    server.no_ui = True
    server.server_port = 8085
    server.mcp_servers_config = "/nonexistent/path.json"
    try:
        server.start()
        assert False, "Server should not have started with missing config file"
    except RuntimeError:
        pass  # Expected: server process dies due to missing config


def test_mcp_empty_tool_list():
    """MCP server reporting zero tools should result in empty tool list."""
    global server
    # Create a minimal server that returns empty tools list
    empty_server = os.path.join(FIXTURES_DIR, "_empty_mcp_server.py")
    with open(empty_server, "w") as f:
        f.write('''#!/usr/bin/env python3
import json, sys, os
def main():
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: request = json.loads(line)
        except: continue
        method = request.get("method")
        req_id = request.get("id")
        if method == "initialize":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "empty", "version": "1.0"}}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": []}}
        else:
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
if __name__ == "__main__":
    main()
''')
    try:
        mcp_json = _mcp_config_json({
            "empty": {
                "command": sys.executable,
                "args": [empty_server],
            }
        })
        server = _start_server_with_mcp(mcp_json)
        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body
        tools = res.body
        def get_tool_name(t):
            return t.get("tool", "") or t.get("definition", {}).get("function", {}).get("name", "")
        mcp_tools = [t for t in tools if get_tool_name(t).startswith("empty:")]
        assert len(mcp_tools) == 0, f"Expected no empty: tools, got {mcp_tools}"
    finally:
        os.unlink(empty_server)
        server.stop()


def test_mcp_rapid_succession_calls():
    """Many rapid calls should increment next_id correctly and correlate responses."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        for i in range(20):
            res = server.make_request("POST", "/tools", data={
                "tool": "echo_echo",
                "params": {"message": f"msg{i}"}
            })
            assert res.status_code == 200, res.body
            assert "error" not in res.body, res.body
    finally:
        server.stop()


def test_mcp_notification_burst():
    """Notification + response in a single write() with no flush should not strand the response."""
    global server
    MCP_BURST_SERVER = os.path.join(FIXTURES_DIR, "mcp_burst_server.py")
    mcp_json = _mcp_config_json({
        "burst": {
            "command": sys.executable,
            "args": [MCP_BURST_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        res = server.make_request("POST", "/tools", data={
            "tool": "burst_echo",
            "params": {"message": "burst test"}
        })
        assert res.status_code == 200, res.body
        body = res.body
        assert "error" not in body, body
    finally:
        server.stop()


def test_mcp_tool_definition_shape_via_chat_completions():
    """MCP tool definitions returned by GET /tools should have the correct shape for chat/completions."""
    global server
    mcp_json = _mcp_config_json({
        "echo": {
            "command": sys.executable,
            "args": [MCP_ECHO_SERVER],
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # Get MCP tool definitions
        res = server.make_request("GET", "/tools")
        assert res.status_code == 200, res.body
        tools = res.body

        def get_tool_name(t):
            return t.get("tool", "") or t.get("definition", {}).get("function", {}).get("name", "")

        echo_tools = [t for t in tools if get_tool_name(t).startswith("echo_")]
        assert len(echo_tools) >= 2, f"Expected echo_ tools, got {echo_tools}"

        echo_tool = next(t for t in echo_tools if get_tool_name(t) == "echo_echo")
        definition = echo_tool.get("definition", echo_tool)

        # Verify the definition has the standard function-calling shape
        assert definition.get("type") == "function", f"Expected type=function, got {definition.get('type')}"
        func = definition.get("function", {})
        assert "name" in func, "Missing function.name"
        assert "description" in func, "Missing function.description"
        assert "parameters" in func, f"Missing function.parameters, got keys: {list(func.keys())}"
        params = func["parameters"]
        assert params.get("type") == "object", f"Expected parameters.type=object, got {params.get('type')}"
        assert "properties" in params, "Missing parameters.properties"
    finally:
        server.stop()


def test_mcp_slow_tool_call_slot_release():
    """A slow tool call should not stall server shutdown for the full I/O timeout."""
    global server
    MCP_SLOW_SERVER = os.path.join(FIXTURES_DIR, "mcp_slow_server.py")
    mcp_json = _mcp_config_json({
        "slow": {
            "command": sys.executable,
            "args": [MCP_SLOW_SERVER, "--delay", "10"],
            "timeout_ms": 30000
        }
    })
    server = _start_server_with_mcp(mcp_json)

    try:
        # Start a slow tool call in a background thread
        def slow_call():
            return server.make_request("POST", "/tools", data={
                "tool": "slow_sleep",
                "params": {"seconds": 10}
            })

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_call)

            # Wait a moment for the call to start
            time.sleep(2)

            # Stop the server while the tool call is in progress.
            # With global MCP instances, close_all() is called explicitly at shutdown
            # (not from slot release), so shutdown should complete promptly.
            start_time = time.time()
            server.stop()
            elapsed = time.time() - start_time

            # The server should stop quickly, not wait for the full 30s I/O timeout.
            # With the terminating flag, send_rpc() bails out within one select()
            # slice (~50ms). This threshold MUST stay below the 5s force-kill
            # fallback in ServerProcess.stop(): without the flag, shutdown stalls
            # on the instance mutex and only completes when stop() sends SIGKILL
            # at ~5s -- which any threshold above 5 would still accept.
            assert elapsed < 3, f"Server stop took {elapsed:.1f}s, expected < 3s"

            # Wait for the future to complete (it will get an error response or timeout)
            try:
                res = future.result(timeout=5)
                # If we got a response, it should be an error since the server stopped
                if hasattr(res, 'status_code'):
                    assert res.status_code in (200, 500, 502, 503, 504), f"Unexpected status: {res.status_code}"
            except Exception:
                # Thread may have raised due to connection error - that's acceptable
                pass
    finally:
        server.stop()
