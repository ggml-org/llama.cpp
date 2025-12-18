# llama-agent

An agentic coding assistant built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

<img width="2924" height="2168" alt="image" src="https://github.com/user-attachments/assets/00865829-532c-4294-b82b-6b025e1ea0f1" />

## What is it?

`llama-agent` transforms a local LLM into a coding assistant that can autonomously complete programming tasks. It uses tool calling to read files, write code, run commands, and navigate codebases.

## Quick Start

```bash
# Build
cmake -B build
cmake --build build --target llama-agent

# Run (downloads model automatically)
./build/bin/llama-agent -hf unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q5_K_M

# Or with a local model
./build/bin/llama-agent -m model.gguf
```

## Recommended Model

| Model | Command |
|-------|---------|
| Nemotron-3-Nano 30B | `-hf unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q5_K_M` |

## Available Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `read` | Read file contents with line numbers |
| `write` | Create or overwrite files |
| `edit` | Search and replace in files |
| `glob` | Find files matching a pattern |

## Usage Examples

```
> Find all TODO comments in src/

[Tool: bash] grep -r "TODO" src/
Found 5 TODO comments...

> Read the main.cpp file

[Tool: read] main.cpp
   1| #include <iostream>
   2| int main() {
   ...

> Fix the bug on line 42

[Tool: edit] main.cpp
Replaced "old code" with "fixed code"
```

## Commands

| Command | Description |
|---------|-------------|
| `/exit` | Exit the agent |
| `/clear` | Clear conversation history |
| `/tools` | List available tools |

## MCP Server Support

The agent supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers, allowing you to extend its capabilities with external tools.

### Configuration

Create an `mcp.json` file in your working directory or at `~/.config/llama-agent/mcp.json`:

```json
{
  "servers": {
    "gradio": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://tongyi-mai-z-image-turbo.hf.space/gradio_api/mcp/",
        "--transport",
        "streamable-http"
      ],
      "timeout": 120000
    }
  }
}
```

### Config Options

| Field | Description | Default |
|-------|-------------|---------|
| `command` | Executable to run (required) | - |
| `args` | Command line arguments | `[]` |
| `env` | Environment variables | `{}` |
| `timeout` | Tool call timeout in ms | `60000` |
| `enabled` | Enable/disable the server | `true` |

Environment variables in config values can use `${VAR_NAME}` syntax for substitution.

### Transport

Only **stdio** transport is supported natively. The agent spawns the server process and communicates via stdin/stdout using JSON-RPC 2.0.

For HTTP-based MCP servers (like Gradio endpoints), use a bridge such as `mcp-remote` as shown in the example above.

### Tool Naming

MCP tools are registered with qualified names: `mcp__<server>__<tool>`. For example, a `read_file` tool from a server named `filesystem` becomes `mcp__filesystem__read_file`.

Use `/tools` to see all available tools including MCP tools.

## Permission System

The agent asks for confirmation before:
- Running shell commands
- Writing or editing files
- Accessing files outside the working directory

When prompted: `y` (yes), `n` (no), `a` (always allow), `d` (deny always)

### Safety Features

- **Sensitive file blocking**: Automatically blocks access to `.env`, `*.key`, `*.pem`, credentials files
- **External directory warnings**: Prompts before accessing files outside the project
- **Dangerous command detection**: Warns for `rm -rf`, `sudo`, `curl|bash`, etc.
- **Doom-loop detection**: Detects and blocks repeated identical tool calls

### CLI Options

| Flag | Description |
|------|-------------|
| `--yolo` | Skip all permission prompts (dangerous!) |
| `--max-iterations N` | Max agent iterations (default: 50, max: 1000) |

### YOLO Mode

Skip all permission prompts:

```bash
./build/bin/llama-agent -m model.gguf --yolo
```

> [!CAUTION]
> **YOLO mode is extremely dangerous.** The agent will execute any command without confirmation, including destructive operations like `rm -rf`. This is especially risky with smaller models that have weaker instruction-following and may hallucinate unsafe commands. Only use this flag if you fully trust the model and understand the risks.

## License

MIT - see [LICENSE](LICENSE)
