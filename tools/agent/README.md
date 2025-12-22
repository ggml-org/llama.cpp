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

## Installation

To use `llama-agent` from any directory, add it to your PATH:

```bash
# Run from the llama.cpp directory after building
echo "export PATH=\"\$PATH:$(pwd)/build/bin\"" >> ~/.zshrc

# Open a new terminal, then run from anywhere
cd /path/to/your/project
llama-agent -hf unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q5_K_M
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
| `/skills` | List available skills |
| `/agents` | List discovered AGENTS.md files |

## Skills

Skills are reusable prompt modules that extend the agent's capabilities. They follow the [agentskills.io](https://agentskills.io) specification.

### Creating a Skill

Skills are directories containing a `SKILL.md` file with YAML frontmatter:

```bash
mkdir -p ~/.config/llama-agent/skills/code-review
cat > ~/.config/llama-agent/skills/code-review/SKILL.md << 'EOF'
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

### Skill Structure

```
skill-name/
├── SKILL.md          # Required - YAML frontmatter + instructions
├── scripts/          # Optional - executable scripts
├── references/       # Optional - additional documentation
└── assets/           # Optional - templates, data files
```

### SKILL.md Format

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

### Search Paths

Skills are discovered from (in priority order):

1. `./.llama-agent/skills/` - Project-local skills
2. `~/.config/llama-agent/skills/` - User-global skills
3. Custom paths via `--skills-path`

### CLI Options

| Flag | Description |
|------|-------------|
| `--no-skills` | Disable skill discovery |
| `--skills-path PATH` | Add custom skills directory |

### How Skills Work

1. **Discovery**: At startup, the agent scans skill directories and loads metadata (name/description)
2. **Activation**: When your request matches a skill's description, the agent reads the full `SKILL.md`
3. **Execution**: The agent follows the skill's instructions, optionally running scripts from `scripts/`

This "progressive disclosure" keeps context lean - only activated skills consume tokens.

## AGENTS.md Support

The agent automatically discovers and loads [AGENTS.md](https://agents.md) files, which provide project-specific guidance for AI coding assistants.

### How It Works

1. **Discovery**: At startup, the agent searches from the working directory up to the git root for `AGENTS.md` files, plus a global file
2. **Precedence**: Files closer to the working directory take precedence (useful for monorepos). Global file has lowest precedence.
3. **Injection**: Content is loaded into the system prompt, giving the agent project-specific context

### Search Locations (in precedence order)

1. `./AGENTS.md` - Current working directory (highest precedence)
2. `../AGENTS.md`, `../../AGENTS.md`, ... - Parent directories up to git root
3. `~/.config/llama-agent/AGENTS.md` - Global user preferences (lowest precedence)

### Creating an AGENTS.md File

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

### Monorepo Support

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

When working in `packages/frontend/`, both `AGENTS.md` files are loaded, with the frontend one taking precedence.

### CLI Options

| Flag | Description |
|------|-------------|
| `--no-agents-md` | Disable AGENTS.md discovery |

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
| `--no-skills` | Disable skill discovery |
| `--skills-path PATH` | Add custom skills search path |
| `--no-agents-md` | Disable AGENTS.md discovery |

### YOLO Mode

Skip all permission prompts:

```bash
./build/bin/llama-agent -m model.gguf --yolo
```

> [!CAUTION]
> **YOLO mode is extremely dangerous.** The agent will execute any command without confirmation, including destructive operations like `rm -rf`. This is especially risky with smaller models that have weaker instruction-following and may hallucinate unsafe commands. Only use this flag if you fully trust the model and understand the risks.

## License

MIT - see [LICENSE](LICENSE)
