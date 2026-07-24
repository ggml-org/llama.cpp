#!/usr/bin/env python3
"""
MCP server that crashes after receiving a specific tool call.
"""
import json
import sys
import os

def handle_initialize(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "crash-test", "version": "1.0"}
        }
    }

def handle_tools_list(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo back the input message",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        }
                    }
                },
                {
                    "name": "crash",
                    "description": "Crash the server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    }

def handle_tools_call(params, req_id):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "echo":
        message = arguments.get("message", "")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"echo: {message}"}]
            }
        }
    elif tool_name == "crash":
        # Send a partial response then exit
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "crashing..."}]}}) + "\n")
        sys.stdout.flush()
        os._exit(1)
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
        }

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

        # JSON-RPC 2.0: a message without an id is a notification and must not receive a response
        if req_id is None:
            continue

        handler = HANDLERS.get(method)
        if handler:
            response = handler(params, req_id)
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
