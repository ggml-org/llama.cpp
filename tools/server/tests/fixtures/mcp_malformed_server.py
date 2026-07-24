#!/usr/bin/env python3
"""
MCP server that sends malformed responses and notifications during requests.
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
            "serverInfo": {"name": "malformed-test", "version": "1.0"}
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
                }
            ]
        }
    }

def handle_tools_call(params, req_id):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "echo":
        message = arguments.get("message", "")
        # Send a notification first (no id field)
        notif = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progress": 50, "total": 100}
        }
        sys.stdout.write(json.dumps(notif) + "\n")
        sys.stdout.flush()
        # Then send the actual response
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"echo: {message}"}]
            }
        }
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
            # Send malformed JSON response
            sys.stdout.write("THIS IS NOT JSON\n")
            sys.stdout.flush()
            continue

        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})

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
