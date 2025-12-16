# llama-agent

An agentic coding assistant built on llama.cpp that can autonomously complete software engineering tasks.

## Overview

`llama-agent` is a CLI tool that transforms a local LLM into a powerful coding assistant. It uses tool calling to:

- Read and analyze source code
- Write new files and make edits
- Execute shell commands
- Navigate and explore codebases

The agent operates in an iterative loop: generating responses, executing tool calls, and continuing until the task is complete.

## Quick Start

```bash
# Build llama.cpp with agent support
cmake -B build
cmake --build build --target llama-agent

# Run with a model
./build/bin/llama-agent -m model.gguf

# Or download directly from Hugging Face
./build/bin/llama-agent -hf nvidia/Nemotron-3-Nano-30B-A3B-FP8-GGUF
```

## Recommended Models

Models with good tool-calling support work best:

| Model | Size | Notes |
|-------|------|-------|
| NVIDIA Nemotron-3-Nano-30B-A3B | ~26GB (Q5_K_M) | Excellent tool calling, uses Qwen3-coder XML format |
| Qwen3 Coder | Various | Native tool calling support |

## Usage

### Basic Commands

```bash
# Start interactive session
llama-agent -m model.gguf

# With larger context (recommended for complex tasks)
llama-agent -m model.gguf -c 32768

# Specify chat template if auto-detection fails
llama-agent -m model.gguf --chat-template qwen3-coder
```

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `bash` | Execute shell commands | `command`, `timeout?`, `description?` |
| `read` | Read file contents with line numbers | `file_path`, `offset?`, `limit?` |
| `write` | Create or overwrite a file | `file_path`, `content` |
| `edit` | Search and replace in a file | `file_path`, `old_string`, `new_string`, `replace_all?` |
| `glob` | Find files matching a pattern | `pattern`, `path?` |

### Permission System

The agent includes a permission system to ensure you stay in control:

**Default permissions:**
- `read`: Allowed (except sensitive files like `.env`, `*.key`, `*.pem`)
- `glob`: Allowed
- `bash`, `write`, `edit`: Requires confirmation

**Interactive prompts:**
```
┌─ PERMISSION: bash ──────────────────────────────────┐
│ Command: rm -rf ./build                             │
├─────────────────────────────────────────────────────┤
│ [y]es  [n]o  [a]lways  [N]ever                      │
└─────────────────────────────────────────────────────┘
```

Options:
- `y` - Allow this action once
- `n` - Deny this action
- `a` - Always allow this tool for this session
- `N` - Never allow this tool for this session

## Example Sessions

### Exploring a Codebase

```
> What files are in the src/ directory?

[Tool: glob] pattern="src/*"
Found 12 files:
  src/main.cpp
  src/parser.cpp
  src/lexer.cpp
  ...

> Show me the main function in main.cpp

[Tool: read] file_path="src/main.cpp"
   1│ #include <iostream>
   2│
   3│ int main(int argc, char** argv) {
   4│     // Entry point
   ...
```

### Making Edits

```
> Add a comment to the main function explaining what it does

[Tool: edit]
  file_path="src/main.cpp"
  old_string="int main(int argc, char** argv) {"
  new_string="// Entry point: parses arguments and initializes the application\nint main(int argc, char** argv) {"

Edit applied successfully.
```

### Running Commands

```
> Build the project and run the tests

[Tool: bash] command="cmake -B build && cmake --build build"
-- Configuring done
-- Generating done
-- Build finished

[Tool: bash] command="./build/bin/tests"
All 42 tests passed.
```

## MCP Server Support

llama-agent supports MCP (Model Context Protocol) servers, allowing you to extend the agent with external tools.

### Configuration

Create a `mcp.json` file in your working directory or `~/.config/llama-agent/mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-fs", "/allowed/path"],
      "enabled": true,
      "timeout": 30000
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | Yes | The command to run the MCP server |
| `args` | string[] | No | Arguments to pass to the command |
| `env` | object | No | Environment variables (supports `${VAR}` expansion) |
| `enabled` | boolean | No | Whether to start this server (default: true) |
| `timeout` | number | No | Tool call timeout in milliseconds (default: 60000) |

### MCP Tool Names

MCP tools are registered with qualified names: `mcp__<server>__<tool>`

For example, a `read_file` tool from a `filesystem` server becomes `mcp__filesystem__read_file`.

### Available MCP Servers

See the [Anthropic MCP servers](https://github.com/anthropics/anthropic-quickstarts/tree/main/mcp-servers) repository for available servers.

## Architecture

```
tools/agent/
├── agent.cpp           # Main entry point
├── agent-loop.cpp      # Core agent loop (generate → parse → execute)
├── tool-registry.cpp   # Tool registration and execution
├── permission.cpp      # Permission system
├── mcp/
│   ├── mcp-client.cpp      # MCP JSON-RPC client (stdio transport)
│   ├── mcp-server-manager.cpp  # Multi-server management
│   └── mcp-tool-wrapper.cpp    # MCP → tool_def adapter
└── tools/
    ├── tool-bash.cpp   # Shell command execution
    ├── tool-read.cpp   # File reading
    ├── tool-write.cpp  # File creation
    ├── tool-edit.cpp   # File editing
    └── tool-glob.cpp   # File pattern matching
```

## Building

The agent is built as part of the standard llama.cpp build:

```bash
cmake -B build
cmake --build build --target llama-agent
```

Or build everything:

```bash
cmake -B build
cmake --build build
```

## Limitations

- Tool calling quality depends heavily on the model
- Complex multi-step tasks may require larger context sizes
- Some models may not follow tool-calling formats correctly

## Safety

The agent includes several safety features:

- Permission prompts for destructive operations
- Blocking of sensitive file patterns (`.env`, `*.key`, etc.)
- Loop detection (prevents infinite tool call loops)
- Timeout for shell commands

Always review tool calls before approving, especially for `bash` and `write`/`edit` operations.
