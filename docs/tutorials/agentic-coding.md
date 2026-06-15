# Tutorial: Offline agentic coding with llama-server

This tutorial shows how to use llama-server as a fully offline, local AI coding assistant with tool calling capabilities.

## Overview

With llama-server's OpenAI-compatible API, you can use it as a drop-in replacement for cloud-based coding assistants in tools that support custom API endpoints (e.g. VS Code with Continue/Copilot extensions, Aider, terminal tools).

## Setup

### 1. Start the server

```sh
# Start with a coding-capable model and sufficient context
llama-server -m model.gguf -c 32768 --jinja --flash-attn
```

Key flags:
- `-c 32768` -- large context window for handling code files
- `--jinja` -- enable Jinja template processing for function calling
- `--flash-attn` -- faster prompt processing with flash attention

### 2. Verify the server

```sh
curl http://localhost:8080/health
# {"status": "ok"}
```

## Function calling / tool use

llama-server supports OpenAI-compatible function calling, allowing models to call external tools like file readers, shell commands, and code executors.

### Defining tools

Pass tool definitions in the API request:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "List files in the current directory"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "list_files",
          "description": "List files in a directory",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "description": "Directory path"}
            },
            "required": ["path"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read contents of a file",
          "parameters": {
            "type": "object",
            "properties": {
              "file_path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["file_path"]
          }
        }
      }
    ]
  }'
```

### Native tool format support

Some models have native tool-calling formats that llama-server recognizes automatically:

| Format | Models |
|--------|--------|
| Llama 3.x | Llama 3.1, 3.2, 3.3 |
| Hermes 2/3 Pro | Hermes, Nous Research models |
| Mistral Nemo | Mistral Nemo variants |
| Firefunction v2 | Fireworks AI models |
| Command R7B | Cohere Command R7B |

For models without native support, a generic tool-calling format is used (less token-efficient).

### Parallel tool calls

Enable parallel tool execution:

```json
{
  "parallel_tool_calls": true
}
```

## Structured output / JSON mode

### Basic JSON mode

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Return a JSON object with name and age"}],
    "response_format": {"type": "json_object"}
  }'
```

### JSON schema mode

Enforce a specific JSON structure:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Fix the bug in this code:\n\ndef add(a, b):\n    return a - b"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "code_fix",
        "schema": {
          "type": "object",
          "properties": {
            "file": {"type": "string"},
            "line": {"type": "integer"},
            "issue": {"type": "string"},
            "fix": {"type": "string"}
          },
          "required": ["file", "line", "issue", "fix"]
        }
      }
    }
  }'
```

## Connecting to coding tools

### Aider

Set the OpenAI API base to your local server:

```sh
aider --openai-api-base http://localhost:8080/v1 --openai-api-key no-key
```

### Continue (VS Code extension)

In your Continue config:

```json
{
  "models": [
    {
      "title": "Local llama.cpp",
      "provider": "openai",
      "model": "gpt-3.5-turbo",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "no-key"
    }
  ]
}
```

### Custom agent loop

Here is a minimal Python agent loop:

```python
import json
import requests

BASE = "http://localhost:8080/v1"

def call_model(messages, tools=None):
    body = {"model": "gpt-3.5-turbo", "messages": messages}
    if tools:
        body["tools"] = tools
    r = requests.post(f"{BASE}/chat/completions",
        json=body,
        headers={"Authorization": "Bearer no-key"})
    return r.json()["choices"][0]["message"]

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What files are in /tmp?"}]

for _ in range(10):  # max tool call iterations
    msg = call_model(messages, tools)
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            print(f"Tool call: {tc['function']['name']}({args})")
            # Execute tool and append result
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": f"Tool output placeholder for {args}"
            })
    else:
        print(msg["content"])
        break
```

## Performance optimization for coding

- Use a model with large context (32K+ tokens) for handling entire codebases
- Enable `--flash-attn` for faster prompt processing of large code files
- Use `--cache-prompt` to reuse the system prompt and common file content across requests
- Enable `--parallel` (`-np`) for handling multiple concurrent coding sessions
- Consider the `--cache-reuse` option for KV shifting when processing similar code snippets
- Use structured output (`json_schema`) rather than parsing free-form text for tools

## Speculative decoding

Boost generation speed with a draft model:

```sh
# Start with speculative decoding
llama-server -m model.gguf -md draft-model.gguf --jinja
```

See [speculative decoding docs](../speculative.md) for details.
