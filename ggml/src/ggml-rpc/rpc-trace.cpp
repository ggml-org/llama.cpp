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
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_initialised.load(std::memory_order_relaxed)) {
        return 0;
    }

    otlp::OtlpGrpcExporterOptions exporter_opts;
    exporter_opts.endpoint = env_or("GGML_RPC_OTLP_ENDPOINT", "http://localhost:4317");

    auto exporter  = otlp::OtlpGrpcExporterFactory::Create(exporter_opts);
    auto processor = sdktrace::BatchSpanProcessorFactory::Create(std::move(exporter), {});

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

#endif // GGML_RPC_OPENTELEMETRY
