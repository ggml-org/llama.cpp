// OpenTelemetry tracing implementation for ggml-rpc.
//
// Everything in this translation unit is gated on GGML_RPC_OPENTELEMETRY.
// When the option is OFF, this file compiles to an empty object (every symbol
// is provided as an inline no-op via the macros in rpc-trace.h) and need not
// be linked.

#include "rpc-trace.h"

#ifdef GGML_RPC_OPENTELEMETRY

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include <opentelemetry/exporters/otlp/otlp_grpc_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter_options.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/sdk/trace/batch_span_processor_factory.h>
#include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#include <opentelemetry/trace/default_span.h>
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/trace/scope.h>

namespace sdktrace = opentelemetry::sdk::trace;
namespace otlp     = opentelemetry::exporter::otlp;
namespace trace    = opentelemetry::trace;
namespace resource = opentelemetry::sdk::resource;

namespace {

std::atomic<bool>      g_initialised{false};
std::mutex             g_init_mutex;
std::shared_ptr<trace::TracerProvider> g_provider;

trace::Tracer * tracer_or_null() {
    if (!g_initialised.load(std::memory_order_acquire)) {
        return nullptr;
    }
    auto p = trace::Provider::GetTracerProvider();
    if (!p) {
        return nullptr;
    }
    static auto t = p->GetTracer("ggml-rpc", "0.1.0");
    return t.get();
}

const char * env_or(const char * key, const char * fallback) {
    const char * v = std::getenv(key);
    return (v && *v) ? v : fallback;
}

} // namespace

extern "C" int rpc_trace_init(void) {
    if (g_initialised.load(std::memory_order_acquire)) {
        return 0;
    }
    // Opt-in: tracing is OFF unless GGML_RPC_OTLP_ENDPOINT is explicitly set.
    // This keeps non-RPC tools (plain llama-cli, llama-bench, ...) from
    // spawning a BatchSpanProcessor thread and pointlessly attempting to
    // connect to a non-existent collector at every startup.
    const char * endpoint_env = std::getenv("GGML_RPC_OTLP_ENDPOINT");
    if (!endpoint_env || !*endpoint_env) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_initialised.load(std::memory_order_relaxed)) {
        return 0;
    }

    otlp::OtlpGrpcExporterOptions exporter_opts;
    exporter_opts.endpoint = endpoint_env;

    auto exporter  = otlp::OtlpGrpcExporterFactory::Create(exporter_opts);
    // High-volume RPC workloads emit thousands of spans/sec; the default
    // BSP queue (2048) and 5 s flush drop spans on the floor under load.
    // Use a larger queue and tighter flush so per-RPC traces are captured
    // faithfully. Overridable via env vars.
    sdktrace::BatchSpanProcessorOptions bsp_opts;
    bsp_opts.max_queue_size = 65536;
    bsp_opts.max_export_batch_size = 4096;
    bsp_opts.schedule_delay_millis = std::chrono::milliseconds(500);
    if (const char * v = std::getenv("GGML_RPC_OTLP_QUEUE")) {
        int parsed = std::atoi(v); if (parsed >= 1024) bsp_opts.max_queue_size = parsed;
    }
    auto processor = sdktrace::BatchSpanProcessorFactory::Create(std::move(exporter), bsp_opts);

    auto resource_attrs = resource::ResourceAttributes{
        {"service.name", std::string(env_or("GGML_RPC_OTLP_SERVICE", "ggml-rpc"))},
    };
    auto res = resource::Resource::Create(resource_attrs);

    g_provider = sdktrace::TracerProviderFactory::Create(std::move(processor), res);
    trace::Provider::SetTracerProvider(g_provider);

    g_initialised.store(true, std::memory_order_release);
    std::atexit([]() { rpc_trace_shutdown(); });
    return 0;
}

extern "C" void rpc_trace_shutdown(void) {
    if (!g_initialised.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if (auto sdk_provider = std::dynamic_pointer_cast<sdktrace::TracerProvider>(g_provider)) {
        sdk_provider->ForceFlush(std::chrono::milliseconds(2000));
        sdk_provider->Shutdown(std::chrono::milliseconds(2000));
    }
    std::shared_ptr<trace::TracerProvider> noop;
    trace::Provider::SetTracerProvider(noop);
    g_provider.reset();
}

// Opaque wrapper holding the OpenTelemetry shared_ptr<Span> directly.
// Lifetime: created in rpc_trace_span_begin, released in rpc_trace_span_end /
// _fail. The OpenTelemetry tracer/batch-processor handles span flushing on End().
struct rpc_trace_span {
    opentelemetry::nostd::shared_ptr<trace::Span> handle;
    explicit rpc_trace_span(opentelemetry::nostd::shared_ptr<trace::Span> h)
        : handle(std::move(h)) {}
};

extern "C" rpc_trace_span_t rpc_trace_span_begin(const char * name) {
    auto * t = tracer_or_null();
    if (!t) {
        return nullptr;
    }
    trace::StartSpanOptions opts;
    opts.kind = trace::SpanKind::kClient;
    auto span = t->StartSpan(name, opts);
    if (!span) {
        return nullptr;
    }
    return new rpc_trace_span(std::move(span));
}

extern "C" void rpc_trace_set_int(rpc_trace_span_t span, const char * key, int64_t value) {
    if (!span || !span->handle) return;
    span->handle->SetAttribute(key, value);
}

extern "C" void rpc_trace_set_str(rpc_trace_span_t span, const char * key, const char * value) {
    if (!span || !span->handle || !value) return;
    span->handle->SetAttribute(key, value);
}

extern "C" void rpc_trace_set_bytes(rpc_trace_span_t span, const char * key, size_t bytes) {
    if (!span || !span->handle) return;
    span->handle->SetAttribute(key, static_cast<int64_t>(bytes));
}

extern "C" void rpc_trace_span_fail(rpc_trace_span_t span, const char * msg) {
    if (!span || !span->handle) {
        delete span;
        return;
    }
    span->handle->SetStatus(trace::StatusCode::kError, msg ? msg : "");
    span->handle->End();
    delete span;
}

extern "C" void rpc_trace_span_end(rpc_trace_span_t span) {
    if (!span) return;
    if (span->handle) {
        span->handle->End();
    }
    delete span;
}



extern "C" int rpc_trace_span_get_ids(rpc_trace_span_t span,
                                      uint8_t trace_id[16],
                                      uint8_t span_id[8]) {
    // Zero-fill on every path so callers see a well-defined buffer even on
    // failure — server-side will treat all-zero trace_id as "no parent".
    for (int i = 0; i < 16; ++i) trace_id[i] = 0;
    for (int i = 0; i < 8;  ++i) span_id[i]  = 0;
    if (!span || !span->handle) {
        return 0;
    }
    auto ctx = span->handle->GetContext();
    if (!ctx.IsValid()) {
        return 0;
    }
    ctx.trace_id().CopyBytesTo(opentelemetry::nostd::span<uint8_t, 16>(trace_id, 16));
    ctx.span_id().CopyBytesTo (opentelemetry::nostd::span<uint8_t, 8> (span_id,  8));
    return 1;
}

extern "C" rpc_trace_span_t rpc_trace_span_begin_with_parent(const char * name,
                                                             const uint8_t trace_id[16],
                                                             const uint8_t parent_span_id[8]) {
    auto * t = tracer_or_null();
    if (!t) {
        return nullptr;
    }
    // All-zero trace_id means the client did not have an active span; fall
    // back to a normal root span on the server.
    bool nonzero = false;
    for (int i = 0; i < 16; ++i) { if (trace_id[i]) { nonzero = true; break; } }
    if (!nonzero) {
        trace::StartSpanOptions opts;
        opts.kind = trace::SpanKind::kServer;
        auto span = t->StartSpan(name, opts);
        if (!span) return nullptr;
        return new rpc_trace_span(std::move(span));
    }

    trace::TraceId    tid (opentelemetry::nostd::span<const uint8_t, 16>(trace_id, 16));
    trace::SpanId     pid (opentelemetry::nostd::span<const uint8_t, 8> (parent_span_id, 8));
    trace::TraceFlags flags(trace::TraceFlags::kIsSampled);
    trace::SpanContext parent_ctx(tid, pid, flags, /*is_remote=*/true);

    auto parent_span = opentelemetry::nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(parent_ctx));
    auto current = opentelemetry::context::RuntimeContext::GetCurrent();
    auto ctx_with_parent = trace::SetSpan(current, parent_span);

    trace::StartSpanOptions opts;
    opts.kind   = trace::SpanKind::kServer;
    opts.parent = ctx_with_parent;
    auto span = t->StartSpan(name, opts);
    if (!span) return nullptr;
    return new rpc_trace_span(std::move(span));
}

#else  // !GGML_RPC_OPENTELEMETRY -- no-op stubs so callers don't need the option.

#include <cstddef>
#include <cstdint>

extern "C" int  rpc_trace_init(void) { return 0; }
extern "C" void rpc_trace_shutdown(void) {}
extern "C" rpc_trace_span_t rpc_trace_span_begin(const char *) { return nullptr; }
extern "C" void rpc_trace_set_int  (rpc_trace_span_t, const char *, int64_t)     {}
extern "C" void rpc_trace_set_str  (rpc_trace_span_t, const char *, const char *) {}
extern "C" void rpc_trace_set_bytes(rpc_trace_span_t, const char *, size_t)       {}
extern "C" void rpc_trace_span_fail(rpc_trace_span_t, const char *)               {}
extern "C" void rpc_trace_span_end (rpc_trace_span_t)                             {}
extern "C" int  rpc_trace_span_get_ids(rpc_trace_span_t, uint8_t t[16], uint8_t s[8]) {
    for (int i = 0; i < 16; ++i) t[i] = 0;
    for (int i = 0; i < 8;  ++i) s[i] = 0;
    return 0;
}
extern "C" rpc_trace_span_t rpc_trace_span_begin_with_parent(const char *, const uint8_t[16], const uint8_t[8]) { return nullptr; }

#endif // GGML_RPC_OPENTELEMETRY
