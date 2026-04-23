# llama-agent

llama-agent builds on llama.cpp's inference engine and adds an agentic tool-use loop on top.

* **Single binary, zero dependencies**: no Python, no Node.js, just download and run
* **Single process**: inference and agent loop in one process, no IPC overhead
* **Same model cache**: uses your existing llama.cpp models, no separate download or setup
* **Light harness**: one simple loop with a handful of built-in tools, optimized for small local models
* **100% local**: offline, no API costs, your code stays on your machine
* **No hidden telemetry**: zero tracking, zero phone-home, no usage events, no error reports sent anywhere
* **API server**: `llama-agent-server` exposes the agent via HTTP API with SSE streaming

<img width="1536" height="641" alt="image" src="https://github.com/user-attachments/assets/494a5615-2c3a-4aee-ad49-2a89eb862f88" />

> [!NOTE]
> ## New: Gemma 4 Vision
>
> [Gemma 4](https://blog.google/technology/developers/gemma-4/) is Google's latest open model family (Apache 2.0), built for agentic use with native tool calling and multimodal input. The **E4B variant** (4.5B effective params, ~5 GB quantized) runs comfortably on an 8 GB laptop and brings full vision capabilities to llama-agent. The model can read and analyze images, screenshots, diagrams, and documents.
>
> ```bash
> # make sure llama-agent is installed
> brew install gary149/llama-agent/llama-agent
>
> # launch with gemma 4 vision (~5 GB, runs on 8 GB machines)
> llama-agent -hf unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
>
> # if you have 16 GB+ RAM, use the bigger MoE variant instead
> llama-agent -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
> ```
>
> With vision enabled, the agent can process hundreds of images in a single session, classify animals by family, read text from screenshots, and analyze UI layouts. All locally, all with a 4B model.
>
> https://github.com/user-attachments/assets/16b62565-e3b1-4967-820f-750c0aec0f3a
>
> | Variant | Effective Params | GGUF Size | Vision | Best for |
> |---------|-----------------|-----------|--------|----------|
> | [E4B](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | 4.5B | ~5 GB | Yes | Laptops, on-device |
> | [26B-A4B](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | 3.8B active (MoE) | ~16 GB | Yes | 16 GB+ machines |
> | [31B](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | 30.7B | ~20 GB | Yes | 32 GB+ machines |

## Table of Contents

- [Quick Start](#quick-start)
- [Available Tools](#available-tools)
- [Commands](#commands)
- [Skills](#skills)
- [AGENTS.md Support](#agentsmd-support)
- [MCP Server Support](#mcp-server-support)
- [Permission System](#permission-system)
- [Session Persistence](#session-persistence)
- [Context Compaction](#context-compaction)
- [HTTP API Server](#http-api-server)

## Quick Start

```bash
# Install (macOS / Linux)
brew install gary149/llama-agent/llama-agent

# Run (downloads model automatically)
llama-agent -hf unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL
```

Or download pre-built binaries from [GitHub Releases](https://github.com/gary149/llama-agent/releases).

<details>
<summary><strong>Build from source</strong></summary>

```bash
# Build CLI agent
cmake -B build
cmake --build build --target llama-agent

# Run
./build/bin/llama-agent -hf unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL

# Or with a local model
./build/bin/llama-agent -m model.gguf
```

**Add to PATH for global access:**

```bash
# For zsh:
echo "export PATH=\"\$PATH:$(pwd)/build/bin\"" >> ~/.zshrc
# For bash:
echo "export PATH=\"\$PATH:$(pwd)/build/bin\"" >> ~/.bashrc
```

**Build the HTTP API server:**

```bash
cmake -B build -DLLAMA_HTTPLIB=ON
cmake --build build --target llama-agent-server
./build/bin/llama-agent-server -hf unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL --port 8081
```

</details>

<img width="1500" height="960" alt="image" src="https://github.com/user-attachments/assets/7f917819-50ab-447f-9504-6406b2670ad5" />

## Recommended Model

| Model | Command |
|-------|---------|
| GLM-4.7-Flash | `-hf unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL` |

<details>
<summary><strong>Optimized settings for GLM-4.7-Flash</strong></summary>

Use these parameters ([recommended by Unsloth](https://unsloth.ai/docs/models/glm-4.7-flash)):

```bash
llama-agent -hf unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL \
  --jinja --ctx-size 16384 --flash-attn on --fit on \
  --temp 0.7 --top-p 1.0 --min-p 0.01 --repeat-penalty 1.0
```

| Flag | Purpose |
|------|---------|
| `--flash-attn on` | Up to 1.48x speedup at batch size 1 ([PR #19092](https://github.com/ggml-org/llama.cpp/pull/19092)) |
| `--fit on` | Auto-optimizes GPU/CPU memory allocation |
| `--repeat-penalty 1.0` | Prevents output degradation (Unsloth recommendation) |

> **Note:** Flash attention has a known issue with KV quantization on very long prompts (~79k+ tokens). On Pascal GPUs (GTX 10xx), flash attention may reduce performance.

</details>

## Available Tools

The agent can use these tools to interact with your codebase and system.

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands (output keeps the tail — errors at the end are preserved) |
| `read` | Read file contents with line numbers (supports images with vision models) |
| `write` | Create or overwrite files |
| `edit` | Search and replace in files |
| `glob` | Find files matching a pattern |
| `update_plan` | Track and display task progress for multi-step operations |

## Commands

Interactive commands available during a session. Type these directly in the chat.

| Command | Description |
|---------|-------------|
| `/exit` | Exit the agent |
| `/clear` | Clear conversation history |
| `/tools` | List available tools |
| `/skills` | List available skills |
| `/agents` | List discovered AGENTS.md files |
| `/stats` | Show token usage and timing statistics |
| `/compact` | Manually compact conversation context |
| `!<cmd>` | Run a shell command and share the output with the LLM |
| `!!<cmd>` | Run a shell command without sharing output with the LLM |

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
2. `./.agents/skills/` - Project-local skills (alternative path)
3. `~/.llama-agent/skills/` - User-global skills
4. `~/.agents/skills/` - User-global skills (alternative path)
5. Custom paths via `--skills-path`

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

Use `/tools` to see all available tools including MCP tools. Use `--no-mcp` to skip MCP server loading entirely.

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
| `--max-iterations N` | Max agent iterations (default: unlimited) |

### Safety Features

- **Sensitive file blocking**: Automatically blocks access to `.env`, `*.key`, `*.pem`, credentials files
- **External directory warnings**: Prompts before accessing files outside the project
- **Dangerous command detection**: Warns for `rm -rf`, `sudo`, `curl|bash`, etc.
- **Doom-loop detection**: Detects and blocks repeated identical tool calls

> [!CAUTION]
> **YOLO mode is extremely dangerous.** The agent will execute any command without confirmation, including destructive operations like `rm -rf`. This is especially risky with smaller models that have weaker instruction-following and may hallucinate unsafe commands. Only use this flag if you fully trust the model and understand the risks.

## Session Persistence

Conversations are automatically saved to disk as append-only JSONL files, so you can resume where you left off.

Sessions are stored at `~/.llama-agent/sessions/` organized by working directory. Each run creates a new session file.

| Flag | Description |
|------|-------------|
| `--resume` | Resume the most recent session for the current directory |
| `--session <path>` | Use a specific session file (creates or resumes) |
| `--no-session` | Disable session persistence |

```bash
# Start a session (auto-saved)
llama-agent -hf model

# Resume where you left off
llama-agent -hf model --resume

# Works with piped input too
echo "hello" | llama-agent -hf model
echo "what did I say?" | llama-agent -hf model --resume

# Explicit session file
llama-agent -hf model --session ~/my-session.jsonl
```

The `/clear` command resets both the conversation and the session file.

## Context Compaction

Long conversations automatically trigger context compaction to stay within the model's context window. When the prompt approaches the context limit, the agent summarizes older messages using the model itself and replaces them with a structured summary. This allows arbitrarily long sessions without losing important context.

**How it works:**
1. After each completion, the agent checks whether prompt tokens exceed ~75% of the context window
2. If so, it finds a safe cut point at a turn boundary (never splitting a tool call from its result)
3. Older messages are serialized and sent to the model with a summarization prompt
4. The summary replaces the old messages, preserving goals, progress, key decisions, and next steps
5. If the context overflows entirely, the agent compacts and retries automatically

Compaction is enabled by default. The summary is iteratively updated on subsequent compactions, so context accumulates rather than being lost.

## HTTP API Server

`llama-agent-server` exposes the agent via HTTP API with Server-Sent Events (SSE) streaming.

```bash
# Build & run
cmake -B build -DLLAMA_HTTPLIB=ON
cmake --build build --target llama-agent-server
./build/bin/llama-agent-server -hf unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL --port 8081
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
- `max_iterations` (int): Max agent iterations (default: 0 = unlimited)
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
| `compaction_completed` | Context compaction finished |
| `completed` | Agent finished with stats |
| `error` | Error occurred |

**Example SSE Stream**

```
event: iteration_start
data: {"iteration":1,"max_iterations":0}

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

## Acknowledgments

Light harness inspired by [Pi](https://github.com/badlogic/pi-mono) by Mario Zechner.

## License

MIT - see [LICENSE](LICENSE)
