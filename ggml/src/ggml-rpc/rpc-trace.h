#pragma once

// OpenTelemetry tracing for ggml-rpc.
//
// Compile-time gated by GGML_RPC_OPENTELEMETRY. When the option is OFF, every
// macro below resolves to a no-op with zero runtime cost. When ON, each send /
// recv on the RPC hot path emits a span exported via OTLP/gRPC to an endpoint
// configured by the GGML_RPC_OTLP_ENDPOINT environment variable
// (default: http://localhost:4317).
//
// Design goals:
//   1. Zero runtime cost when disabled.
//   2. No exposure of OpenTelemetry headers to callers — implementation hidden
//      in rpc-trace.cpp via opaque span handles.
//   3. Safe to call before rpc_trace_init(): all functions become no-ops if
//      the tracer provider has not been created.
//   4. Thread-safe (OpenTelemetry C++ tracer is thread-safe by contract).

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Initialise the OTLP/gRPC exporter and tracer provider. Reads:
//   GGML_RPC_OTLP_ENDPOINT  (default "http://localhost:4317")
//   GGML_RPC_OTLP_SERVICE   (default "ggml-rpc")
// Safe to call multiple times — second and subsequent calls are no-ops.
// Returns 0 on success, nonzero on init failure (tracer stays disabled).
int  rpc_trace_init(void);

// Flush pending spans and shut down the tracer. Call on process exit.
void rpc_trace_shutdown(void);

// Opaque span handle. nullptr means "no span" (tracer disabled, init failed,
// or GGML_RPC_OPENTELEMETRY off).
typedef struct rpc_trace_span * rpc_trace_span_t;

// Start a span with the given name. Returns nullptr if tracing is disabled.
// Span attributes can be attached via rpc_trace_set_*. Must be matched with
// rpc_trace_span_end(); never freed manually.
rpc_trace_span_t rpc_trace_span_begin(const char * name);

void rpc_trace_set_int  (rpc_trace_span_t span, const char * key, int64_t value);
void rpc_trace_set_str  (rpc_trace_span_t span, const char * key, const char * value);
void rpc_trace_set_bytes(rpc_trace_span_t span, const char * key, size_t bytes);

// Mark a span as failed with a brief error message; auto-ends the span.
void rpc_trace_span_fail(rpc_trace_span_t span, const char * msg);

// End a successful span.
void rpc_trace_span_end (rpc_trace_span_t span);

#ifdef __cplusplus
}
#endif

// Convenience macros. These always call the real functions; the
// implementation is a no-op stub when GGML_RPC_OPENTELEMETRY is OFF.
#define RPC_TRACE_INIT()                (void) rpc_trace_init()
#define RPC_TRACE_SHUTDOWN()            rpc_trace_shutdown()
#define RPC_TRACE_BEGIN(name)           rpc_trace_span_begin(name)
#define RPC_TRACE_INT(s, k, v)          rpc_trace_set_int((s), (k), (v))
#define RPC_TRACE_STR(s, k, v)          rpc_trace_set_str((s), (k), (v))
#define RPC_TRACE_BYTES(s, k, v)        rpc_trace_set_bytes((s), (k), (v))
#define RPC_TRACE_FAIL(s, msg)          rpc_trace_span_fail((s), (msg))
#define RPC_TRACE_END(s)                rpc_trace_span_end((s))

#ifdef __cplusplus
// RAII span guard. Ends the span automatically on scope exit unless fail()
// is called first. Use in C++ code paths with early returns where a manual
// END is easy to miss.
class rpc_trace_scope {
public:
    explicit rpc_trace_scope(const char * name) : span_(rpc_trace_span_begin(name)) {}
    ~rpc_trace_scope() { end(); }
    rpc_trace_scope(const rpc_trace_scope &)            = delete;
    rpc_trace_scope & operator=(const rpc_trace_scope &) = delete;

    rpc_trace_span_t handle() const { return span_; }
    void set_int  (const char * k, long long v) { rpc_trace_set_int  (span_, k, v); }
    void set_str  (const char * k, const char * v) { rpc_trace_set_str  (span_, k, v); }
    void set_bytes(const char * k, unsigned long long v) { rpc_trace_set_bytes(span_, k, (unsigned long) v); }
    void fail(const char * msg) {
        if (!ended_) { rpc_trace_span_fail(span_, msg); ended_ = true; }
    }
    void end() {
        if (!ended_) { rpc_trace_span_end(span_); ended_ = true; }
    }
private:
    rpc_trace_span_t span_;
    bool ended_ = false;
};
#endif // __cplusplus

