#!/usr/bin/env python3
"""
Minimal MCP server that writes notification + response in a single write() with no flush.
This reproduces the buffering bug where read_message() can strand the response.
"""
import json
import sys
import os

TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the input message",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        }
    }
]

def handle_initialize(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "burst-test", "version": "1.0"}
        }
    }

def handle_tools_list(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"tools": TOOLS}
    }

def handle_tools_call(params, req_id):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "echo":
        message = arguments.get("message", "")
        notif = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progress": 50, "total": 100}
        }
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"echo: {message}"}]
            }
        }
        # Single os.write() call: both lines land in one pipe packet atomically.
        # This is the key difference from mcp_malformed_server.py which flushes between writes.
        data = (json.dumps(notif) + "\n" + json.dumps(response) + "\n").encode("utf-8")
        os.write(sys.stdout.fileno(), data)
        return None  # already written
    else:
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
        }
        return response

HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}

# notifications have no id and don't expect a response
NOTIFICATION_HANDLERS = {
    "notifications/initialized": lambda params, req_id: None,
}

def main():
    # Use line-buffered text mode for regular responses, but the burst write
    # uses os.write() directly to guarantee a single kernel write().
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

        # Check notification handlers first (no response expected)
        if not req_id and method in NOTIFICATION_HANDLERS:
            NOTIFICATION_HANDLERS[method](params, req_id)
            continue

        handler = HANDLERS.get(method)
        if handler:
            response = handler(params, req_id)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
