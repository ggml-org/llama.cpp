#!/usr/bin/env python3
"""
MCP server (NDJSON JSON-RPC over stdio) that spawns a long-lived grandchild which inherits
this process's stdin/stdout/stderr and keeps them open.

This reproduces the reader-teardown deadlock: killing the direct MCP child (SIGKILL, which is
all subprocess_terminate() does) does NOT close the stdout/stderr pipe write ends, because the
grandchild still holds them. A server that reads those pipes with a blocking read would then
wait forever for an EOF that never arrives, hanging teardown (both warmup shutdown at startup
and process shutdown). The polled, running-aware reader must exit regardless.
"""
import json
import os
import subprocess
import sys

# Spawn a grandchild that inherits our std handles (fds 0/1/2 = the MCP pipes) and lives well
# past any teardown in the tests. We do NOT redirect its stdio, so it keeps the pipe write ends
# open even after this process is killed.
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the input message",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to echo"}},
            "required": ["message"],
        },
    }
]


def handle_initialize(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "grandchild-test", "version": "1.0"},
        },
    }


def handle_tools_list(params, req_id):
    return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}


def handle_tools_call(params, req_id):
    if params.get("name") == "echo":
        message = params.get("arguments", {}).get("message", "")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": f"echo: {message}"}]},
        }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Unknown tool"}}


HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}


def main():
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})

        if req_id is None:
            continue  # notification, no response

        handler = HANDLERS.get(method)
        if handler:
            response = handler(params, req_id)
        else:
            response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
