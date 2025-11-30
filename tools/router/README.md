# llama-router

A lightweight, cross-platform model orchestrator for llama.cpp that dynamically spawns and manages llama-server instances on demand.

## Overview

llama-router acts as an intelligent proxy that sits in front of your model collection. When a request arrives for a specific model, the router automatically starts the appropriate llama-server backend, waits for it to become ready, and forwards the request transparently. This enables a single API endpoint to serve multiple models without keeping them all loaded in memory simultaneously.

### Key Features

- **On-demand model loading**: Models are spawned only when requested, conserving GPU memory
- **Automatic discovery**: Scans the Hugging Face cache directory for available models
- **Hugging Face integration**: Download models directly via CLI with `-hf` flag
- **Collection import**: Recursively import local GGUF directories
- **Multimodal support**: Automatically detects and configures mmproj files for vision models
- **Model grouping**: Define groups to ensure only one model from a group runs at a time
- **OpenAI-compatible API**: Exposes `/v1/models` and `/v1/chat/completions` endpoints
- **Hot reload**: Admin endpoints for runtime configuration updates

### Cross-Platform Support

llama-router builds and runs on both Linux and Windows:

| Platform | Process Management | Path Handling |
|----------|-------------------|---------------|
| Linux/macOS | `fork()` + `execvp()` | `$HOME`, `/proc/self/exe` |
| Windows | `CreateProcess()` | `%USERPROFILE%`, `GetModuleFileName()` |

The router automatically detects the llama-server binary location relative to its own executable, falling back to PATH resolution if not found.

---

## Quick Start

### Building

llama-router is built as part of the llama.cpp server tools:

```bash
cmake -B build -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-router
```

The binary will be located at `build/bin/llama-router`.

### Running

Simply launch the router:

```bash
./llama-router
```

On first run, it will:
1. Create a default configuration at `~/.config/llama.cpp/router-config.json`
2. Scan the Hugging Face cache (`~/.cache/huggingface/hub/`) for existing models
3. Start listening on `127.0.0.1:8082`

---

## CLI Reference

### Download from Hugging Face

The `-hf` flag provides plug-and-play model downloading directly from Hugging Face:

```bash
# Download a model by repository (auto-selects best GGUF)
./llama-router -hf bartowski/Qwen3-8B-GGUF

# Specify a quantization variant
./llama-router -hf bartowski/Qwen3-8B-GGUF:Q4_K_M

# Or specify the exact filename
./llama-router -hf bartowski/Qwen3-8B-GGUF -hff Qwen3-8B-Q4_K_M.gguf
```

**CLI options for downloading:**

| Flag | Description |
|------|-------------|
| `-hf`, `-hfr`, `--hf-repo` | Hugging Face repository (format: `user/repo` or `user/repo:quant`) |
| `-hff`, `--hf-file` | Specific GGUF filename within the repository |

**Authentication:** Set the `HF_TOKEN` environment variable for private or gated repositories:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
./llama-router -hf meta-llama/Llama-3.1-8B-Instruct-GGUF
```

After download completes, the model is cached and will be automatically discovered on the next router startup.

### Import a Local Collection

Use `--import-dir` to recursively scan a directory and add all discovered GGUF models to your configuration:

```bash
# Import all models from a directory
./llama-router --import-dir ~/my-models/

# Import from multiple locations (run multiple times)
./llama-router --import-dir /mnt/nas/gguf-collection/
./llama-router --import-dir ~/experiments/fine-tuned/
```

The import process:
1. Recursively scans the directory for `.gguf` files
2. Excludes files containing "mmproj" in the filename (these are vision projectors)
3. Automatically pairs models with mmproj files found in the same directory
4. Adds new models with state `"manual"` to prevent accidental removal
5. Persists updated configuration to disk

**Multimodal detection:** If a model like `llava-v1.6-mistral-7b-Q4_K_M.gguf` is found alongside `mmproj-model-f16.gguf` in the same directory, the import will automatically configure the spawn command with `--mmproj`.

### Other CLI Options

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Display help message |
| `--config <path>` | Override config file location |

---

## Model States: `auto` vs `manual`

The router tracks each model's origin through a `state` field, which controls behavior during rescans:

### `auto` State

Models discovered automatically from the Hugging Face cache are marked as `"state": "auto"`. These models:

- Are added when first discovered in the cache
- Are **removed automatically** if the cached file disappears (e.g., cache cleanup)
- Are re-added if the file reappears

This enables seamless synchronization with `huggingface-cli` downloads and cache management.

### `manual` State

Models added via `--import-dir` or edited by hand in the config are marked as `"state": "manual"`. These models:

- Are **never automatically removed**, even if the file path becomes invalid
- Must be manually deleted from the configuration
- Survive rescans and configuration reloads

**Use cases for manual state:**
- Models on network storage that may be temporarily unavailable
- Fine-tuned models in development directories
- Models you want to persist regardless of file system changes

### Changing State

Edit the configuration file directly to change a model's state:

```json
{
  "name": "my-model.gguf",
  "path": "/path/to/my-model.gguf",
  "state": "manual"
}
```

Or set to `"auto"` if you want the router to manage its lifecycle.

---

## Configuration

### File Location

Default: `~/.config/llama.cpp/router-config.json`

Override with `--config`:

```bash
./llama-router --config /etc/llama-router/config.json
```

### Configuration Structure

```json
{
  "version": "1.0",
  "router": {
    "host": "127.0.0.1",
    "port": 8082,
    "base_port": 50000,
    "connection_timeout_s": 5,
    "read_timeout_s": 600,
    "admin_token": ""
  },
  "default_spawn": {
    "command": ["llama-server", "--ctx-size", "4096", "--n-gpu-layers", "99"],
    "proxy_endpoints": ["/v1/", "/health", "/slots", "/props"],
    "health_endpoint": "/health"
  },
  "models": [
    {
      "name": "Qwen3-8B-Q4_K_M.gguf",
      "path": "/home/user/.cache/huggingface/hub/models--bartowski--Qwen3-8B-GGUF/...",
      "state": "auto",
      "group": ""
    }
  ]
}
```

### Router Options

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `127.0.0.1` | Address to bind the router |
| `port` | `8082` | Port for incoming requests |
| `base_port` | `50000` | Starting port for spawned backends (increments per model) |
| `connection_timeout_s` | `5` | Upstream connection timeout |
| `read_timeout_s` | `600` | Upstream read timeout (long for streaming) |
| `admin_token` | `""` | Bearer token for admin endpoints (empty = no auth) |

### Default Spawn Configuration

The `default_spawn` block defines how llama-server instances are launched:

```json
{
  "command": ["llama-server", "--ctx-size", "4096", "--n-gpu-layers", "99"],
  "proxy_endpoints": ["/v1/", "/health", "/slots", "/props"],
  "health_endpoint": "/health"
}
```

The router automatically appends these arguments:
- `--model <path>` - The model file path
- `--port <port>` - Dynamically assigned port
- `--host 127.0.0.1` - Localhost binding for security

### Per-Model Spawn Override

Individual models can override the default spawn configuration:

```json
{
  "name": "llava-v1.6-mistral-7b-Q4_K_M.gguf",
  "path": "/path/to/model.gguf",
  "state": "manual",
  "spawn": {
    "command": [
      "llama-server",
      "--ctx-size", "8192",
      "--n-gpu-layers", "99",
      "--mmproj", "/path/to/mmproj-model-f16.gguf"
    ],
    "proxy_endpoints": ["/v1/", "/health"],
    "health_endpoint": "/health"
  }
}
```

### Model Groups

Groups ensure mutual exclusivity - when a model from a group is requested, any running model from a **different** group is stopped first:

```json
{
  "models": [
    {
      "name": "qwen3-8b-q4",
      "path": "/path/to/qwen3-8b-q4.gguf",
      "group": "8b-models"
    },
    {
      "name": "qwen3-8b-q8",
      "path": "/path/to/qwen3-8b-q8.gguf",
      "group": "8b-models"
    },
    {
      "name": "llama-70b-q4",
      "path": "/path/to/llama-70b-q4.gguf",
      "group": "70b-models"
    }
  ]
}
```

Behavior:
- Requesting `qwen3-8b-q4` while `qwen3-8b-q8` is running: **no restart** (same group)
- Requesting `llama-70b-q4` while `qwen3-8b-q4` is running: **stops qwen3, starts llama** (different group)

If no group is specified, the model name is used as a singleton group.

---

## API Endpoints

### OpenAI-Compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List all configured models |
| `/v1/chat/completions` | POST | Chat completion (model selected from request body) |

### Model-Specific Proxies

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/<model_name>/health` | GET | Health check for specific model |
| `/<model_name>/props` | GET | Model properties |
| `/<model_name>/slots` | GET | Slot information |

### Last-Spawned Shortcuts

These endpoints proxy to the most recently spawned model:

| Endpoint | Method |
|----------|--------|
| `/health` | GET |
| `/props` | GET |
| `/slots` | GET |

### Admin Endpoints

Protected by `admin_token` if configured:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/reload` | POST | Stop all running models |
| `/admin/rescan` | GET | Rescan cache and update config |

**Authentication:**

```bash
# Via Authorization header
curl -H "Authorization: Bearer <token>" http://localhost:8082/admin/rescan

# Via X-Admin-Token header
curl -H "X-Admin-Token: <token>" http://localhost:8082/admin/rescan
```

---

## Usage Examples

### Basic Chat Completion

```bash
curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Response

```bash
curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Write a haiku about AI"}],
    "stream": true
  }'
```

### List Available Models

```bash
curl http://localhost:8082/v1/models | jq '.data[].id'
```

### Check Model Health

```bash
# Specific model
curl http://localhost:8082/Qwen3-8B-Q4_K_M.gguf/health

# Last spawned model
curl http://localhost:8082/health
```

### Force Rescan After Downloads

```bash
# Download a new model
./llama-router -hf TheBloke/Mistral-7B-v0.1-GGUF:Q4_K_M

# Rescan without restarting
curl http://localhost:8082/admin/rescan
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       llama-router                          │
│                     (port 8082)                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Config    │  │   Scanner   │  │   Process Manager   │  │
│  │   Loader    │  │  (HF cache) │  │   (spawn/terminate) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                            │                                │
│  ┌─────────────────────────┴────────────────────────────┐   │
│  │                    HTTP Proxy                        │   │
│  │         (streaming support, header forwarding)       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ llama-server  │  │ llama-server  │  │ llama-server  │
│  (port 50000) │  │  (port 50001) │  │  (port 50002) │
│   Model A     │  │   Model B     │  │   Model C     │
└───────────────┘  └───────────────┘  └───────────────┘
```

### Request Flow

1. Client sends POST to `/v1/chat/completions` with `"model": "ModelA"`
2. Router checks if ModelA is already running
3. If not running, or if a conflicting group is active:
   - Terminate conflicting backends
   - Spawn new llama-server with assigned port
   - Poll `/health` until ready (10s timeout)
4. Forward request to backend, streaming response back to client
5. Backend remains running for subsequent requests

### Process Lifecycle

- **Spawn**: `fork()`/`CreateProcess()` with stdout/stderr capture
- **Health polling**: 200ms intervals, 10s timeout
- **Graceful shutdown**: SIGTERM → 1s wait → SIGKILL
- **Cleanup**: File descriptors closed, waitpid() called

---

## Signals and Shutdown

The router handles graceful shutdown on:
- `SIGINT` (Ctrl+C)
- `SIGTERM`

Shutdown sequence:
1. Stop accepting new connections
2. Terminate all managed llama-server processes
3. Wait for process cleanup
4. Exit

---

## Troubleshooting

### Model Not Found

```
{"error":"model not found"}
```

Check that the model name in your request matches exactly the `name` field in the configuration. Use `/v1/models` to list available names.

### Backend Not Ready

```
Backend for X did not become ready on port Y within 10000 ms
```

Possible causes:
- Model file corrupted or incompatible
- Insufficient GPU memory
- llama-server crashed during load

Check the router's stdout for llama-server output.

### Port Conflicts

If `base_port` conflicts with other services, change it in the configuration:

```json
{
  "router": {
    "base_port": 60000
  }
}
```

### Permission Denied on Config

Ensure the config directory exists and is writable:

```bash
mkdir -p ~/.config/llama.cpp
```

---

## Contributing

llama-router is part of the llama.cpp project. Contributions welcome via pull request.

## License

MIT License - See llama.cpp repository for details.
