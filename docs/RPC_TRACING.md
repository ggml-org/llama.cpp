# RPC OpenTelemetry tracing

Opt-in instrumentation that emits an OpenTelemetry span for every client-side
`send_rpc_cmd` call. Useful for answering: *where is the time going inside a
multi-GPU / multi-host RPC pool?* — TCP queueing, kernel syscalls, transport
choice, peer compute, response-wait, or graph instantiation.

## Build

```bash
cmake -B build \
  -DGGML_RPC=ON \
  -DGGML_CUDA=ON \
  -DGGML_RPC_OPENTELEMETRY=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rpc-server llama-cli -j
```

Requires [opentelemetry-cpp](https://github.com/open-telemetry/opentelemetry-cpp)
(`find_package(opentelemetry-cpp CONFIG REQUIRED)`). On Debian/Ubuntu the
simplest path is to build it from source:

```bash
git clone https://github.com/open-telemetry/opentelemetry-cpp.git
cd opentelemetry-cpp
cmake -B build -DWITH_OTLP_GRPC=ON -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && sudo cmake --install build
```

## Run

```bash
GGML_RPC_OTLP_ENDPOINT=http://localhost:4317 \
GGML_RPC_OTLP_SERVICE=ggml-rpc-gpu0 \
  ./build/bin/rpc-server -H 127.0.0.1 -p 50052 -c
```

| Env var | Default | Purpose |
|---|---|---|
| `GGML_RPC_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP gRPC collector endpoint |
| `GGML_RPC_OTLP_SERVICE`  | `ggml-rpc` | `service.name` resource attribute |

Both the controller (the process calling `llama-cli --rpc ...`) and each
`rpc-server` worker emit spans. They share a trace ID through the OTLP
collector — the trace flame graph shows the full life of an RPC across both
sides.

## What gets emitted

Every call to `send_rpc_cmd` produces one of two spans:

| Span name | Attributes |
|---|---|
| `ggml.rpc.client.send`       | `rpc.cmd`, `rpc.input_bytes`, `rpc.has_response="false"` |
| `ggml.rpc.client.send_recv`  | `rpc.cmd`, `rpc.input_bytes`, `rpc.expected_output_bytes`, `rpc.has_response="true"` |

Failures (socket error, response-size mismatch, etc.) end the span with
`StatusCode::kError` and a short description.

Each span carries both:
- `rpc.cmd` — the integer enum value (stable across versions)
- `rpc.cmd_name` — the human-readable name (the more useful one for filtering)

| Name             | Value | Hot path? |
|---|---|---|
| ALLOC_BUFFER     | 0  |    |
| GET_ALIGNMENT    | 1  |    |
| GET_MAX_SIZE     | 2  |    |
| BUFFER_GET_BASE  | 3  |    |
| FREE_BUFFER      | 4  |    |
| BUFFER_CLEAR     | 5  |    |
| SET_TENSOR       | 6  | yes |
| SET_TENSOR_HASH  | 7  | yes |
| GET_TENSOR       | 8  | yes |
| COPY_TENSOR      | 9  |    |
| GRAPH_COMPUTE    | 10 | yes (the big one) |
| GET_DEVICE_MEMORY| 11 |    |
| INIT_TENSOR      | 12 |    |
| GET_ALLOC_SIZE   | 13 |    |
| HELLO            | 14 | once at handshake |
| DEVICE_COUNT     | 15 | once at init |
| GRAPH_RECOMPUTE  | 16 | yes |

Filter on `rpc.cmd_name = "GRAPH_COMPUTE"` to see only the heavy per-token
RPCs.

## Compatibility

- `GGML_RPC_OPENTELEMETRY=OFF` (default) makes every trace macro a no-op
  preprocessor expansion. Zero runtime cost, zero linker pull-in.
- Tracer initialisation is idempotent (`rpc_trace_init()` can be called
  repeatedly).
- Span lifecycle is thread-safe (per the OpenTelemetry C++ contract).
- Spans are batched and flushed in the background by the SDK; calling
  `rpc_trace_shutdown()` at process exit drains the queue.

## Why no `rpc_trace_init()` in `ggml-rpc.cpp` client code?

The header file is the public API; init/shutdown live in the *binary's* `main`.
Client-side code that uses `ggml-rpc` as a library (e.g. `llama-cli --rpc ...`)
should call `RPC_TRACE_INIT()` early in its own `main` if it wants spans. The
shipped `rpc-server` binary already does this.
