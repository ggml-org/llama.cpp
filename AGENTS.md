# Agent Instructions

This repository is the llama.cpp project, but your focus should be **exclusively on the agent feature** located in `tools/agent/`.

## Scope

- Only work on files within `tools/agent/` and related documentation
- Do not explore or modify other parts of the llama.cpp codebase unless explicitly asked
- The agent is an agentic coding assistant built on top of llama.cpp

## Key Files

```
tools/agent/
├── agent.cpp          # Main entry point
├── agent-loop.cpp     # Core agent loop logic
├── agent-loop.h
├── permission.cpp     # Permission system for tool execution
├── permission.h
├── prompt.txt         # System prompt
├── tool-registry.cpp  # Tool registration
├── tool-registry.h
├── tools/             # Tool implementations
├── mcp/               # MCP server support
├── CMakeLists.txt
└── README.md
```

## Build

```bash
cmake -B build
cmake --build build --target llama-agent
```
