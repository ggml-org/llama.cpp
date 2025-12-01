# llama-router Architecture

Technical documentation for developers and contributors.

---

## Design Philosophy

llama-router follows KISS (Keep It Simple, Stupid) principles:

- **Minimal configuration**: Works out-of-box with HF cache scanning
- **Explicit persistence**: Config changes are written explicitly via admin endpoints, never hidden in business logic
- **Separation of concerns**: Core routing logic (`RouterApp`) has zero I/O, persistence handled by admin layer
- **Simple endpoint matching**: Prefix-based matching, no complex regex
- **Transparent proxy**: Headers and streaming forwarded as-is
- **On-demand only**: No models start at boot, everything spawns when first requested
- **Transparent operations**: Optional real-time notifications for swap feedback via SSE

### The auto + default_spawn Workflow

Models discovered from the HuggingFace cache are marked as `auto` and inherit the `default_spawn` configuration. This creates a powerful optimization pattern:

1. **Tune `default_spawn` once** with your preferred parameters (GPU layers, KV cache quantization, context size, etc.)
2. **All `auto` models automatically use these settings** - no per-model configuration needed
3. **Change `default_spawn` and reload** - all `auto` models instantly updated
4. **Customize individual models** by switching to `manual` state first to prevent rescan overwrites

This ensures consistent, optimized behavior across your entire model collection while allowing per-model overrides when needed. **Always set models to `manual` before customizing their spawn parameters** - otherwise your changes will be lost on the next rescan.

## Multi-Engine Support

llama-router is engine-agnostic. Any OpenAI-compatible inference backend can be orchestrated by configuring the appropriate spawn command and endpoints. The router simply:

1. Spawns the command specified in `spawn.command`
2. Polls `health_endpoint` until it returns HTTP 200 (customizable per backend)
3. Proxies requests matching `proxy_endpoints` to the running instance

This design allows you to mix llama.cpp, vLLM, Ollama, Text Generation Inference, or any custom backend in a single router configuration. Set models to `manual` state when using non-llama.cpp backends to prevent automatic cache rescans from removing them.

### Future: WebUI Administration (TODO)

The admin API endpoints (`/admin/reload`, `/admin/rescan`) are designed to support hot configuration and model management. A future WebUI will enable:

- **Live model downloads** from HuggingFace directly through the interface
- **Hot reconfiguration** of `default_spawn` and per-model settings without restart
- **Real-time monitoring** of running instances and resource usage
- **Interactive model management** (add, remove, customize spawn parameters)

This aligns with the project philosophy: **everything configurable at runtime, zero downtime required**. The current CLI and JSON-based workflow is production-ready; the WebUI will provide a more accessible interface to the same underlying admin API.

---

## Architecture

### System Architecture

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
   - Poll `/health` until ready (`ROUTER_BACKEND_READY_TIMEOUT_MS` timeout)
4. Forward request to backend, streaming response back to client
5. Backend remains running for subsequent requests

### Process Lifecycle

- **Spawn**: `fork()`/`CreateProcess()` with stdout/stderr capture
- **Health polling**: `ROUTER_BACKEND_HEALTH_POLL_MS` intervals, `ROUTER_BACKEND_READY_TIMEOUT_MS` timeout
- **Graceful shutdown**: SIGTERM → wait `ROUTER_PROCESS_SHUTDOWN_TIMEOUT_MS` → SIGKILL → poll every `ROUTER_PROCESS_POLL_INTERVAL_MS` until exit
- **Cleanup**: File descriptors closed, waitpid() called

---

## File Structure & Separation of Concerns

| Component | Files | Responsibility |
|-----------|-------|----------------|
| **Core** | `router-app.cpp/h` | Model lifecycle, spawn orchestration, group logic, progress notification emission (zero I/O except notifications) |
| **HTTP Endpoints** | `router-endpoints.cpp/h` | Public API routes (`/v1/models`, `/v1/chat/completions`) |
| **Admin** | `router-admin.cpp/h` | Admin routes with explicit config persistence |
| **Proxy** | `router-proxy.cpp/h` | HTTP forwarding, SSE streaming, header management |
| **Process** | `router-process.cpp/h` | Cross-platform subprocess spawning, I/O capture |
| **Config** | `router-config.cpp/h` | JSON load/write, rescan logic, `RescanResult` |
| **Scanner** | `router-scanner.cpp/h` | HF cache discovery, `--import-dir`, mmproj detection |
| **Main** | `router.cpp` | CLI parsing, server setup, signal handlers |
| **Utils** | `logging.cpp/h`, `router-constants.h` | Shared logging and constants |

**Design principles enforced:**
- `router-app`: Pure business logic, no filesystem I/O
- `router-admin`: Owns config persistence, explicit writes only
- `router-proxy`: Streaming & forwarding, value-captured lambdas to avoid use-after-free
- `router-process`: Platform abstraction, child processes never call parent logging functions

---

## Technical Notes

### Cross-Platform Process Management

The router handles subprocess spawning differently per platform:

**Linux/macOS:** Uses `fork()` + `execvp()` with careful attention to post-fork behavior. Child processes **must not** call logging functions that access parent singletons - they write directly to `STDERR_FILENO` instead to avoid use-after-fork crashes.

**Windows:** Uses `CreateProcess()` with separate process information structures and handle management.

### SSE Streaming Implementation

Server-Sent Events streaming required careful lifetime management to avoid use-after-free bugs:

1. **Capture by value**: Lambda captures must copy request data (headers, path, body), not reference stack variables that become invalid after the handler returns
2. **Explicit termination**: Call `sink.done()` followed by `return false` to signal httplib to close the connection properly - without this, streams deliver tokens correctly but never terminate

### PATH Binary Resolution

Spawn commands support both absolute/relative paths and PATH-based binaries:

- **Paths with separators**: `/usr/bin/llama-server`, `./llama-server`, `C:\llama\server.exe` - existence validated before spawn
- **PATH binaries**: `python`, `vllm`, `ollama`, `llama-server` - no validation, relies on shell PATH resolution

The router only validates file existence for commands containing `/` or `\\` path separators, allowing seamless use of system-installed binaries.

### Model-Scoped Route Stripping

Routes like `/<model>/health` are router-side aliases for convenience. Before proxying to the backend, the router strips the model prefix:

- User request: `GET /Qwen3-8B-Q4_K_M.gguf/health`
- Forwarded to backend: `GET /health`

Backends remain unaware of model-scoped routing - they expose standard endpoints like `/health`, `/v1/chat/completions`, etc.

### HTTP Header Management

The router strips `Content-Length` and `Transfer-Encoding` headers before forwarding requests. This is standard reverse-proxy behavior to handle chunked requests/responses properly and avoid conflicts when the proxy re-chunks data.

All other headers are forwarded transparently to preserve client context (authentication, user-agent, etc.).

### Real-Time Swap Notifications

The router implements an opt-in notification system for streaming swap progress to clients:

**Architecture:**
- `NotificationSink`: Function-based callback system in `router-config.h`
- `RouterApp::set/clear_notification_sink()`: Attach/detach sink before/after operations
- Progress emitted at 3 lifecycle points in `ensure_running()`:
  * After `terminate_process()` - unload notification
  * After `spawn_process()` - load notification
  * After `wait_for_backend_ready()` - ready notification

**Implementation:**
The proxy layer owns the full request lifecycle. For streaming requests with `notify_model_swap=true`:
1. Attach sink that enqueues formatted SSE chunks into the stream state
2. Call `ensure_running()` - notifications flow directly into the SSE queue
3. Clear sink before forwarding to backend (prevents backend logs in stream)

Messages use OpenAI-compatible `delta.reasoning_content` field, prefixed with `[llama-router]` to distinguish router operations from model reasoning.

**Design rationale:**
- Sink pattern allows clean separation: RouterApp emits events, proxy consumes them
- Notifications sent synchronously during operations = accurate timing perception
- Thread-safe via separate `notification_mutex` to avoid deadlock with main mutex
- Zero overhead when disabled (sink check + early return)

### Health Endpoint Purpose

The `health_endpoint` configuration field serves **spawn readiness polling only** - the router uses it to detect when a backend has finished loading and is ready to serve requests.

This is separate from user-facing health routes. Clients can still call `/<model>/health` or `/health` for their own monitoring needs. The backend must expose standard endpoints regardless of what `health_endpoint` is configured for polling.

### Multimodal Projector Priority

When importing collections with `--import-dir`, mmproj files are automatically detected with this search priority:

1. `*-bf16.gguf` (selected first)
2. `*-f16.gguf` (selected if BF16 not found)
3. `*-f32.gguf` (selected if neither BF16 nor F16 found)

All quantization variants of a model (Q4_K_M, Q5_K_M, Q6_K, etc.) found in the same directory share the same mmproj file.

**For manual models:** mmproj auto-detection applies only during initial import. You can edit `spawn.command` to remove `--mmproj` if unwanted - your changes persist across restarts. Only `auto` models get their spawn configuration regenerated on rescan.

### Manifest Robustness

The HF cache scanner gracefully handles missing or corrupted manifest files:

- If `~/.cache/llama.cpp/` doesn't exist, scanner returns empty mapping
- If individual manifest files are missing, they're silently skipped
- Models without manifest entries load successfully, just without mmproj auto-detection

**Cache structure example:**
```
~/.cache/llama.cpp/
├── bartowski_Qwen2.5-1.5B-Instruct-GGUF_Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
├── bartowski_Qwen2.5-1.5B-Instruct-GGUF_Qwen2.5-1.5B-Instruct-Q4_K_M.gguf.etag
├── manifest=bartowski=Qwen2.5-1.5B-Instruct-GGUF=latest.json
├── unsloth_Qwen3-VL-4B-Instruct-GGUF_Qwen3-VL-4B-Instruct-Q6_K.gguf
├── unsloth_Qwen3-VL-4B-Instruct-GGUF_Qwen3-VL-4B-Instruct-Q6_K.gguf.etag
├── unsloth_Qwen3-VL-4B-Instruct-GGUF_mmproj-F16.gguf
├── unsloth_Qwen3-VL-4B-Instruct-GGUF_mmproj-F16.gguf.etag
└── manifest=unsloth=Qwen3-VL-4B-Instruct-GGUF=Q6_K.json
```

Manifest files (`manifest=vendor=repo=quant.json`) contain metadata for mmproj auto-detection. The scanner uses underscore separators: `vendor_repo_filename.gguf`.

This ensures the router remains operational even with incomplete cache metadata.

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

## Contributing

llama-router is part of the llama.cpp project. Contributions welcome via pull request.

## License

MIT License - See llama.cpp repository for details.
