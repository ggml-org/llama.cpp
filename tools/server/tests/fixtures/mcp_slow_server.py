#!/usr/bin/env python3
"""
MCP server that sleeps before responding, for timeout testing.
"""
import json
import sys
import os
import time
import argparse

TOOLS = [
    {
        "name": "sleep",
        "description": "Sleep for a given number of seconds",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Seconds to sleep"}
            },
            "required": ["seconds"]
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
            "serverInfo": {"name": "slow-test", "version": "1.0"}
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

    if tool_name == "sleep":
        seconds = arguments.get("seconds", 1)
        time.sleep(seconds)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"slept {seconds}s"}]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds for sleep tool")
    args = parser.parse_args()

    # Override the sleep duration
    global handle_tools_call
    def handle_tools_call(params, req_id):
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "sleep":
            seconds = arguments.get("seconds", args.delay)
            time.sleep(seconds)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"slept {seconds}s"}]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }

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
