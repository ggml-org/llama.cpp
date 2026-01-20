# llama-agent

A coding agent that runs entirely inside [llama.cpp](https://github.com/ggml-org/llama.cpp): single binary, zero dependencies, native performance.

<img width="1536" height="641" alt="image" src="https://github.com/user-attachments/assets/494a5615-2c3a-4aee-ad49-2a89eb862f88" />

## Table of Contents

- [Quick Start](#quick-start)
- [Available Tools](#available-tools)
- [Commands](#commands)
- [Subagents](#subagents)
- [Skills](#skills)
- [AGENTS.md Support](#agentsmd-support)
- [MCP Server Support](#mcp-server-support)
- [Permission System](#permission-system)
- [HTTP API Server](#http-api-server)

## What is it?

`llama-agent` builds on llama.cpp's inference engine and adds an agentic tool-use loop on top. The result:

- **Single binary**: no Python, no Node.js, just download and run
- **Native speed**: tool calls in-process, no HTTP overhead
- **100% local**: offline, no API costs, your code stays on your machine
- **API server**: `llama-agent-server` exposes the agent via HTTP API with SSE streaming

## Quick Start

```bash
# Build CLI agent
cmake -B build
cmake --build build --target llama-agent

# Run (downloads model automatically)
./build/bin/llama-agent -hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M

# Or with a local model
./build/bin/llama-agent -m model.gguf
```

<details>
<summary><strong>Build & run the HTTP API server</strong></summary>

```bash
# Build with HTTP support
cmake -B build -DLLAMA_HTTPLIB=ON
cmake --build build --target llama-agent-server

# Run API server
./build/bin/llama-agent-server -hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M --port 8081
```

</details>

<details>
<summary><strong>Add to PATH for global access</strong></summary>

```bash
# Run from the llama.cpp directory after building
# For zsh:
echo "export PATH=\"\$PATH:$(pwd)/build/bin\"" >> ~/.zshrc
# For bash:
echo "export PATH=\"\$PATH:$(pwd)/build/bin\"" >> ~/.bashrc

# Open a new terminal, then run from anywhere
cd /path/to/your/project
llama-agent -hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M
```

</details>

<img width="1500" height="960" alt="image" src="https://github.com/user-attachments/assets/7f917819-50ab-447f-9504-6406b2670ad5" />

## Recommended Model

| Model | Command |
|-------|---------|
| GLM-4.7-Flash | `-hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M` |

<details>
<summary><strong>Optimized settings for GLM-4.7-Flash</strong></summary>

To reduce repetition and looping, use these sampling parameters ([recommended by Unsloth](https://unsloth.ai/docs/models/glm-4.7-flash#reducing-repetition-and-looping)):

```bash
llama-agent -hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_XL \
  --temp 0.2 --top-k 50 --top-p 0.95 --min-p 0.01 \
  --dry-multiplier 1.1 --ctx-size 16384
```

</details>

## Available Tools

The agent can use these tools to interact with your codebase and system.

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `read` | Read file contents with line numbers |
| `write` | Create or overwrite files |
| `edit` | Search and replace in files |
| `glob` | Find files matching a pattern |
| `task` | Spawn a subagent for complex tasks |

## Commands

Interactive commands available during a session. Type these directly in the chat.

| Command | Description |
|---------|-------------|
| `/exit` | Exit the agent |
| `/clear` | Clear conversation history |
| `/tools` | List available tools |
| `/skills` | List available skills |
| `/agents` | List discovered AGENTS.md files |

## Subagents

Subagents are specialized child agents that handle complex tasks independently, keeping the main conversation context clean and efficient.

| Flag | Description |
|------|-------------|
| `--subagents` | Enable subagents (disabled by default) |
| `--max-subagent-depth N` | Set max nesting depth (0-5, default: 1 when enabled) |
| `--no-subagents` | Explicitly disable subagents |

### Why Subagents?

Without subagents, every file read and search pollutes your main context. With subagents, only a summary enters main context. The detailed exploration is discarded afterward.

```
Main context:
└── task(explore) → "Found 3 TODO items in src/main.cpp:42,87,156" (50 tokens)
    └── Instead of ~5,700 tokens for all the exploration.
```

### Subagent Types

| Type | Purpose | Tools Available |
|------|---------|-----------------|
| `explore` | Search and understand code (read-only) | `glob`, `read`, `bash` (read-only) |
| `bash` | Execute shell commands | `bash` |
| `plan` | Design implementation approaches | `glob`, `read`, `bash` |
| `general` | General-purpose tasks | All tools |

<details>
<summary><strong>How subagents work (architecture)</strong></summary>

```
┌─────────────────┐
│   Main Agent    │  "Find where errors are handled"
└────────┬────────┘
         │ task(explore, "find error handling")
         ▼
┌─────────────────┐
│    Subagent     │  Does detailed exploration:
│    (explore)    │  - glob **/*.cpp
│                 │  - read 5 files
│                 │  - grep patterns
└────────┬────────┘
         │ Returns summary only
         ▼
┌─────────────────┐
│   Main Agent    │  Receives: "Errors handled in src/error.cpp:45
│                 │  via ErrorHandler class..."
└─────────────────┘
```

</details>

<details>
<summary><strong>Memory efficiency & parallel execution</strong></summary>

**Memory Efficiency**

Subagents share the model, so no additional VRAM is used:

| Resource | Main Agent | Subagent | Total |
|----------|------------|----------|-------|
| Model weights | ✓ | Shared | 1x |
| KV cache | ✓ | Shared via slots | 1x |
| Context window | Own | Own (discarded after) | Efficient |

**Parallel Execution**

Multiple subagents can run in the background simultaneously:

```
> Run tests and check for lint errors at the same time

[task-a1b2] ┌── ⚡ run-tests (bash)
[task-c3d4] ┌── ⚡ check-lint (bash)
[task-c3d4] │   └── done (1.8s)
[task-a1b2] │   └── done (2.1s)
```

**KV Cache Prefix Sharing**

Subagent prompts share a common prefix with the main agent, enabling automatic KV cache reuse:

```
Main agent prompt:    "You are llama-agent... [base] + [main agent instructions]"
Subagent prompt:      "You are llama-agent... [base] + # Subagent Mode: explore..."
                       ↑─────── shared prefix ──────↑
```

This reduces subagent startup latency and saves compute.

</details>

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

## Skills

Skills are reusable prompt modules that extend the agent's capabilities. They follow the [agentskills.io](https://agentskills.io) specification.

| Flag | Description |
|------|-------------|
| `--no-skills` | Disable skill discovery |
| `--skills-path PATH` | Add custom skills directory |

Skills are discovered from:
1. `./.llama-agent/skills/` - Project-local skills
2. `~/.llama-agent/skills/` - User-global skills
3. Custom paths via `--skills-path`

<details>
<summary><strong>Creating a skill</strong></summary>

Skills are directories containing a `SKILL.md` file with YAML frontmatter:

```bash
mkdir -p ~/.llama-agent/skills/code-review
cat > ~/.llama-agent/skills/code-review/SKILL.md << 'EOF'
---
name: code-review
description: Review code for bugs, security issues, and improvements. Use when asked to review code or a PR.
---

# Code Review Instructions

When reviewing code:
1. Run `git diff` to see changes
2. Read modified files for context
3. Check for bugs, security issues, style problems
4. Provide specific feedback with file:line references
EOF
```

**Skill Structure**

```
skill-name/
├── SKILL.md          # Required - YAML frontmatter + instructions
├── scripts/          # Optional - executable scripts
├── references/       # Optional - additional documentation
└── assets/           # Optional - templates, data files
```

**SKILL.md Format**

```yaml
---
name: skill-name          # Required: 1-64 chars, lowercase+numbers+hyphens
description: What and when # Required: 1-1024 chars, triggers activation
license: MIT              # Optional
compatibility: python3    # Optional: environment requirements
metadata:                 # Optional: custom key-value pairs
  author: someone
---

Markdown instructions for the agent...
```

**How Skills Work**

1. **Discovery**: At startup, the agent scans skill directories and loads metadata (name/description)
2. **Activation**: When your request matches a skill's description, the agent reads the full `SKILL.md`
3. **Execution**: The agent follows the skill's instructions, optionally running scripts from `scripts/`

This "progressive disclosure" keeps context lean: only activated skills consume tokens.

</details>

## AGENTS.md Support

The agent automatically discovers and loads [AGENTS.md](https://agents.md) files for project-specific guidance.

| Flag | Description |
|------|-------------|
| `--no-agents-md` | Disable AGENTS.md discovery |

Files are discovered from the working directory up to the git root, plus a global `~/.llama-agent/AGENTS.md`.

<details>
<summary><strong>Creating an AGENTS.md file</strong></summary>

Create an `AGENTS.md` file in your repository root:

```markdown
# Project Guidelines

## Build & Test
- Build: `cmake -B build && cmake --build build`
- Test: `ctest --test-dir build`

## Code Style
- Use 4-space indentation
- Follow Google C++ style guide

## PR Guidelines
- Include tests for new features
- Update documentation
```

**Search Locations (in precedence order)**

1. `./AGENTS.md` - Current working directory (highest precedence)
2. `../AGENTS.md`, `../../AGENTS.md`, ... - Parent directories up to git root
3. `~/.llama-agent/AGENTS.md` - Global user preferences (lowest precedence)

**Monorepo Support**

In monorepos, you can have nested `AGENTS.md` files:

```
repo/
├── AGENTS.md           # General project guidance
├── packages/
│   ├── frontend/
│   │   └── AGENTS.md   # Frontend-specific guidance (takes precedence)
│   └── backend/
│       └── AGENTS.md   # Backend-specific guidance
```

When working in `packages/frontend/`, both files are loaded with the frontend one taking precedence.

</details>

## MCP Server Support

The agent supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers, allowing you to extend its capabilities with external tools.

> **Note:** MCP servers using HTTPS (like HuggingFace) require SSL support. If you see `'https' scheme is not supported`, rebuild with:
> ```bash
> cmake -B build -DLLAMA_BUILD_LIBRESSL=ON
> cmake --build build -t llama-agent -j
> ```

Create an `mcp.json` file in your working directory or at `~/.llama-agent/mcp.json`:

```json
{
  "servers": {
    "gradio": {
      "command": "npx",
      "args": ["mcp-remote", "https://example.hf.space/gradio_api/mcp/", "--transport", "streamable-http"],
      "timeout": 120000
    }
  }
}
```

Use `/tools` to see all available tools including MCP tools.

<details>
<summary><strong>MCP configuration details</strong></summary>

**Config Options**

| Field | Description | Default |
|-------|-------------|---------|
| `command` | Executable to run (required) | - |
| `args` | Command line arguments | `[]` |
| `env` | Environment variables | `{}` |
| `timeout` | Tool call timeout in ms | `60000` |
| `enabled` | Enable/disable the server | `true` |

Config values support environment variable substitution using `${VAR_NAME}` syntax.

**Transport**

Only **stdio** transport is supported natively. The agent spawns the server process and communicates via stdin/stdout using JSON-RPC 2.0.

For HTTP-based MCP servers (like Gradio endpoints), use a bridge such as `mcp-remote`.

**Tool Naming**

MCP tools are registered with qualified names: `mcp__<server>__<tool>`. For example, a `read_file` tool from a server named `filesystem` becomes `mcp__filesystem__read_file`.

</details>

## Permission System

The agent asks for confirmation before:
- Running shell commands
- Writing or editing files
- Accessing files outside the working directory

When prompted: `y` (yes), `n` (no), `a` (always allow), `d` (deny always)

| Flag | Description |
|------|-------------|
| `--yolo` | Skip all permission prompts (dangerous!) |
| `--max-iterations N` | Max agent iterations (default: 50, max: 1000) |

### Safety Features

- **Sensitive file blocking**: Automatically blocks access to `.env`, `*.key`, `*.pem`, credentials files
- **External directory warnings**: Prompts before accessing files outside the project
- **Dangerous command detection**: Warns for `rm -rf`, `sudo`, `curl|bash`, etc.
- **Doom-loop detection**: Detects and blocks repeated identical tool calls

> [!CAUTION]
> **YOLO mode is extremely dangerous.** The agent will execute any command without confirmation, including destructive operations like `rm -rf`. This is especially risky with smaller models that have weaker instruction-following and may hallucinate unsafe commands. Only use this flag if you fully trust the model and understand the risks.

## HTTP API Server

`llama-agent-server` exposes the agent via HTTP API with Server-Sent Events (SSE) streaming.

```bash
# Build & run
cmake -B build -DLLAMA_HTTPLIB=ON
cmake --build build --target llama-agent-server
./build/bin/llama-agent-server -hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M --port 8081
```

### Basic Usage

```bash
# Create a session
curl -X POST http://localhost:8081/v1/agent/session \
  -H "Content-Type: application/json" \
  -d '{"yolo": true}'
# Returns: {"session_id": "sess_00000001"}

# Send a message (streaming response)
curl -N http://localhost:8081/v1/agent/session/sess_00000001/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "List files in the current directory"}'
```

<details>
<summary><strong>API endpoints reference</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/agent/session` | POST | Create a new session |
| `/v1/agent/session/:id` | GET | Get session info |
| `/v1/agent/session/:id/chat` | POST | Send message (SSE streaming) |
| `/v1/agent/session/:id/messages` | GET | Get conversation history |
| `/v1/agent/session/:id/permissions` | GET | Get pending permission requests |
| `/v1/agent/permission/:id` | POST | Respond to permission request |
| `/v1/agent/sessions` | GET | List all sessions |
| `/v1/agent/tools` | GET | List available tools |
| `/v1/agent/session/:id/stats` | GET | Get session token stats |

**Session Options**

- `yolo` (boolean): Skip permission prompts
- `max_iterations` (int): Max agent iterations (default: 50)
- `working_dir` (string): Working directory for tools

</details>

<details>
<summary><strong>SSE event types</strong></summary>

| Event | Description |
|-------|-------------|
| `iteration_start` | New agent iteration starting |
| `reasoning_delta` | Streaming model reasoning/thinking |
| `text_delta` | Streaming response text |
| `tool_start` | Tool execution beginning |
| `tool_result` | Tool execution completed |
| `permission_required` | Permission needed (non-yolo mode) |
| `permission_resolved` | Permission granted/denied |
| `completed` | Agent finished with stats |
| `error` | Error occurred |

**Example SSE Stream**

```
event: iteration_start
data: {"iteration":1,"max_iterations":50}

event: reasoning_delta
data: {"content":"Let me list the files..."}

event: tool_start
data: {"name":"bash","args":"{\"command\":\"ls\"}"}

event: tool_result
data: {"name":"bash","success":true,"output":"file1.txt\nfile2.cpp","duration_ms":45}

event: text_delta
data: {"content":"Here are the files:"}

event: completed
data: {"reason":"completed","stats":{"input_tokens":1500,"output_tokens":200}}
```

</details>

<details>
<summary><strong>Permission flow & session management</strong></summary>

**Permission Flow**

When `yolo: false`, dangerous operations require permission:

```
event: permission_required
data: {"request_id":"perm_abc123","tool":"bash","details":"rm -rf temp/","dangerous":true}
```

Respond via API:
```bash
curl -X POST http://localhost:8081/v1/agent/permission/perm_abc123 \
  -H "Content-Type: application/json" \
  -d '{"allow": true, "scope": "session"}'
```

Scopes: `once`, `session`, `always`

**Concurrent Sessions**

The server supports multiple concurrent sessions, each with its own conversation history and permission state.

```bash
# List all sessions
curl http://localhost:8081/v1/agent/sessions

# Delete a session
curl -X POST http://localhost:8081/v1/agent/session/sess_00000001/delete
```

</details>

## License

MIT - see [LICENSE](LICENSE)
