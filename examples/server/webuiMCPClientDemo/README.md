# A Demo of the MCP Client using React and SSE
Still in progress but seems to work. 

# Instructions

Load the SSE servers: 
cd servers/python/simple-tool-prompt-resource
Load it with uv. 
```bash
uv run mcp-simple-tool-prompt-resource --transport sse --port 8000
```
cd servers/python/simple-tool-prompt-resource2
Load it with uv. 
```bash
uv run mcp-simple-tool-prompt-resource2 --transport sse --port 8001
```
The script assumes these are running at localhost. But you can edit the config and change it to any SSE server. 

Run the script. 

npm start


# Issues to consider: 
two tool names that match but are different servers, can prolly modify the tools_available to do mcpserver.name.toolname to the list tools. 

Haven't tested multiple tools being called, prolly should set max tool calls. 

Text completion streaming needs to be off to call tools. 

# Todo:

Outline process of using each MCP supported item. 

Tools:

Prompts:

Resources:


