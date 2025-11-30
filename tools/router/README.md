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
2. Scan the Hugging Face cache (`~/.cache/llama.cpp/`) for GGUF models
3. Add discovered models as `auto` state, inheriting `default_spawn` configuration
4. Start listening on `127.0.0.1:8082`

On every subsequent startup:
- Automatic rescan updates the model list (adds new, removes deleted cache files)
- All `auto` models inherit the current `default_spawn` settings
- `manual` models preserve their custom configurations

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

### Important: On-Demand Spawning

**All models spawn only when first requested via the API.** The router never starts backends at boot. The `auto`/`manual` state controls only rescan behavior:

- `auto`: Managed by cache scanner, inherits `default_spawn`
- `manual`: Protected from rescans, can have custom `spawn` configuration

### `auto` State

Models discovered automatically from the Hugging Face cache are marked as `"state": "auto"`. These models:

- Are added when first discovered in the cache
- Are **removed automatically** if the cached file disappears (e.g., cache cleanup)
- Are re-added if the file reappears
- **Inherit `default_spawn` configuration** - change `default_spawn` to optimize all `auto` models at once

This enables seamless synchronization with `huggingface-cli` downloads and cache management.

### `manual` State

Models added via `--import-dir` or edited by hand in the config are marked as `"state": "manual"`. These models:

- Are **never automatically removed**, even if the file path becomes invalid
- Must be manually deleted from the configuration
- Survive rescans and configuration reloads
- **Can have custom `spawn` configurations** that override `default_spawn`

**Use cases for manual state:**
- Models on network storage that may be temporarily unavailable
- Fine-tuned models in development directories
- Models you want to persist regardless of file system changes

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
      "path": "/home/user/.cache/llama.cpp/bartowski_Qwen3-8B-GGUF_Qwen3-8B-Q4_K_M.gguf",
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

### Optimizing for Your Hardware

The `default_spawn` is where you tune performance for your specific hardware. **All `auto` models inherit these settings**, so you can optimize once for your entire collection:

```json
{
  "default_spawn": {
    "command": [
      "llama-server",
      "-ngl", "999",
      "-ctk", "q8_0",
      "-ctv", "q8_0",
      "-fa", "on",
      "--mlock",
      "-np", "4",
      "-kvu",
      "--jinja"
    ],
    "proxy_endpoints": ["/v1/", "/health", "/slots", "/props"],
    "health_endpoint": "/health"
  }
}
```

**Common optimizations:**
- `-ngl 999`: Offload all layers to GPU
- `-ctk q8_0 -ctv q8_0`: Quantize KV cache to Q8 for lower VRAM usage
- `-fa on`: Enable Flash Attention
- `--mlock`: Lock model in RAM to prevent swapping
- `-np 4`: Process 4 prompts in parallel
- `-kvu`: Use single unified KV buffer for all sequences (also `--kv-unified`)
- `--jinja`: Enable Jinja template support

**Note:** The router automatically appends `--model`, `--port`, and `--host` - do not include these in your command.

Change `default_spawn`, reload the router, and all `auto` models instantly use the new configuration.

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

**Default Behavior: Single Model at a Time**

llama-router is designed for resource-constrained environments (small GPUs, consumer hardware). By default, **only ONE model runs at a time** - when you request a different model, the current one is stopped first. This ensures reliable operation on systems with limited VRAM.

To allow multiple models to run simultaneously, assign the **same group** to models that can coexist:

```json
{
  "models": [
    {
      "name": "qwen3-8b-q4",
      "path": "/path/to/qwen3-8b-q4.gguf",
      "group": "small-models"
    },
    {
      "name": "qwen3-8b-q8",
      "path": "/path/to/qwen3-8b-q8.gguf",
      "group": "small-models"
    },
    {
      "name": "llama-70b-q4",
      "path": "/path/to/llama-70b-q4.gguf",
      "group": "large-model"
    }
  ]
}
```

Behavior:
- Requesting `qwen3-8b-q4` while `qwen3-8b-q8` is running: **no restart** (same group)
- Requesting `llama-70b-q4` while `qwen3-8b-q4` is running: **stops qwen3, starts llama** (different group)

**Omitting the `group` field creates an exclusive singleton per model** - each model stops all others before starting.

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

## Documentation

For detailed technical documentation, design decisions, and contributing guidelines, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Contributing

llama-router is part of the llama.cpp project. Contributions welcome via pull request.

## License

MIT License - See llama.cpp repository for details.
