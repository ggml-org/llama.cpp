import sys
import os
import json
import logging

logging.basicConfig(filename='/devel/tools/llama.cpp/mcp_dummy.log', level=logging.DEBUG)

def main():
    logging.info("Starting MCP Dummy Server")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            logging.info(f"Received: {line.strip()}")
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if "method" in req:
                method = req["method"]
                req_id = req.get("id")
                
                resp = {"jsonrpc": "2.0", "id": req_id}
                
                if method == "initialize":
                    resp["result"] = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "dummy", "version": "1.0"}
                    }
                elif method == "tools/list":
                    resp["result"] = {
                        "tools": [
                            {
                                "name": "get_weather",
                                "description": "Get weather for a location",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string"},
                                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                                    },
                                    "required": ["location"]
                                }
                            }
                        ]
                    }
                elif method == "tools/call":
                    params = req.get("params", {})
                    name = params.get("name")
                    args = params.get("arguments", {})
                    
                    logging.info(f"Tool call: {name} with {args}")
                    
                    content = [{"type": "text", "text": f"Weather in {args.get('location')} is 25C"}]
                    # For simplicity, return raw content or follow MCP spec?
                    # MCP spec: result: { content: [ {type: "text", text: "..."} ] }
                    # My mcp.hpp returns res["result"].
                    # My cli.cpp dumps res.dump().
                    # So passing full result object is fine.
                    resp["result"] = {
                        "content": content
                    }
                else:
                    # Ignore notifications or other methods
                    if req_id is not None:
                         resp["error"] = {"code": -32601, "message": "Method not found"}
                    else:
                        continue

                logging.info(f"Sending: {json.dumps(resp)}")
                if req_id is not None:
                    sys.stdout.write(json.dumps(resp) + "\n\n")
                    sys.stdout.flush()
        except Exception as e:
            logging.error(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
