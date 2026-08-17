#include "server-telemetry.h"

#include "log.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

#ifdef LLAMA_SERVER_OPEN_TELEMETRY
#    include <opentelemetry/context/context.h>
#    include <opentelemetry/context/propagation/text_map_propagator.h>
#    include <opentelemetry/exporters/otlp/otlp_http_exporter_factory.h>
#    include <opentelemetry/exporters/otlp/otlp_http_exporter_options.h>
#    include <opentelemetry/sdk/common/disabled.h>
#    include <opentelemetry/sdk/resource/resource.h>
#    include <opentelemetry/sdk/trace/batch_span_processor_factory.h>
#    include <opentelemetry/sdk/trace/batch_span_processor_options.h>
#    include <opentelemetry/sdk/trace/sampler.h>
#    include <opentelemetry/sdk/trace/samplers/always_off_factory.h>
#    include <opentelemetry/sdk/trace/samplers/always_on_factory.h>
#    include <opentelemetry/sdk/trace/samplers/parent_factory.h>
#    include <opentelemetry/sdk/trace/samplers/trace_id_ratio_factory.h>
#    include <opentelemetry/sdk/trace/tracer_provider.h>
#    include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#    include <opentelemetry/trace/context.h>
#    include <opentelemetry/trace/propagation/http_trace_context.h>
#    include <opentelemetry/trace/span.h>
#    include <opentelemetry/trace/span_startoptions.h>
#    include <opentelemetry/trace/tracer.h>
#endif

namespace {

bool string_iequals(const std::string & lhs, const std::string & rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(lhs[i])) != std::tolower(static_cast<unsigned char>(rhs[i]))) {
            return false;
        }
    }
    return true;
}

std::string string_to_lower(std::string text) {
    for (char & c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

#ifdef LLAMA_SERVER_OPEN_TELEMETRY

namespace otel        = opentelemetry;
namespace otel_ctx    = opentelemetry::context;
namespace otel_otlp   = opentelemetry::exporter::otlp;
namespace otel_common = opentelemetry::sdk::common;
namespace otel_res    = opentelemetry::sdk::resource;
namespace otel_sdk    = opentelemetry::sdk::trace;
namespace otel_trace  = opentelemetry::trace;

std::mutex                                                        g_telemetry_mutex;
std::shared_ptr<otel_sdk::TracerProvider>                         g_provider;
otel::nostd::shared_ptr<otel_ctx::propagation::TextMapPropagator> g_propagator;
std::string                                                       g_service_version;
bool                                                              g_enabled = false;

class header_carrier final : public otel_ctx::propagation::TextMapCarrier {
  public:
    explicit header_carrier(const std::map<std::string, std::string> & headers) :
        read_headers(headers),
        write_headers(nullptr) {}

    explicit header_carrier(std::map<std::string, std::string> & headers) :
        read_headers(headers),
        write_headers(&headers) {}

    otel::nostd::string_view Get(otel::nostd::string_view key) const noexcept override {
        const std::string wanted(key.data(), key.size());
        for (const auto & entry : read_headers) {
            if (string_iequals(entry.first, wanted)) {
                return entry.second;
            }
        }
        return "";
    }

    void Set(otel::nostd::string_view key, otel::nostd::string_view value) noexcept override {
        if (write_headers == nullptr) {
            return;
        }
        const std::string name(key.data(), key.size());
        for (auto it = write_headers->begin(); it != write_headers->end();) {
            if (string_iequals(it->first, name)) {
                it = write_headers->erase(it);
            } else {
                ++it;
            }
        }
        write_headers->emplace(name, std::string(value.data(), value.size()));
    }

  private:
    const std::map<std::string, std::string> & read_headers;
    std::map<std::string, std::string> *       write_headers;
};

std::unique_ptr<otel_sdk::Sampler> make_sampler() {
    const char *      sampler_env = std::getenv("OTEL_TRACES_SAMPLER");
    const std::string sampler =
        sampler_env == nullptr || sampler_env[0] == '\0' ? "parentbased_always_on" : string_to_lower(sampler_env);

    if (sampler == "always_on") {
        return otel_sdk::AlwaysOnSamplerFactory::Create();
    }
    if (sampler == "always_off") {
        return otel_sdk::AlwaysOffSamplerFactory::Create();
    }

    double ratio = 1.0;
    if (sampler == "traceidratio" || sampler == "parentbased_traceidratio") {
        const char * arg = std::getenv("OTEL_TRACES_SAMPLER_ARG");
        if (arg != nullptr && arg[0] != '\0') {
            try {
                size_t parsed = 0;
                ratio         = std::stod(arg, &parsed);
                if (parsed != std::string(arg).size() || !std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
                    throw std::invalid_argument("ratio must be a finite number between 0.0 and 1.0");
                }
            } catch (const std::exception &) {
                LOG_WRN("invalid OTEL_TRACES_SAMPLER_ARG='%s', using 1.0\n", arg);
                ratio = 1.0;
            }
        }
    }

    if (sampler == "traceidratio") {
        return otel_sdk::TraceIdRatioBasedSamplerFactory::Create(ratio);
    }

    std::unique_ptr<otel_sdk::Sampler> root;
    if (sampler == "parentbased_always_off") {
        root = otel_sdk::AlwaysOffSamplerFactory::Create();
    } else if (sampler == "parentbased_traceidratio") {
        root = otel_sdk::TraceIdRatioBasedSamplerFactory::Create(ratio);
    } else {
        if (sampler != "parentbased_always_on") {
            LOG_WRN("unsupported OTEL_TRACES_SAMPLER='%s', using parentbased_always_on\n", sampler.c_str());
        }
        root = otel_sdk::AlwaysOnSamplerFactory::Create();
    }
    return otel_sdk::ParentBasedSamplerFactory::Create(std::shared_ptr<otel_sdk::Sampler>(std::move(root)));
}

struct telemetry_state {
    bool                                                              enabled = false;
    otel::nostd::shared_ptr<otel_trace::Tracer>                       tracer;
    otel::nostd::shared_ptr<otel_ctx::propagation::TextMapPropagator> propagator;
};

telemetry_state get_telemetry_state() {
    telemetry_state                           state;
    std::shared_ptr<otel_sdk::TracerProvider> provider;
    std::string                               version;
    {
        std::lock_guard<std::mutex> lock(g_telemetry_mutex);
        if (!g_enabled) {
            return state;
        }
        state.enabled    = true;
        state.propagator = g_propagator;
        provider         = g_provider;
        version          = g_service_version;
    }
    if (provider) {
        state.tracer = provider->GetTracer("llama.cpp.server", version);
    }
    return state;
}

#endif

}  // namespace

struct server_telemetry_span::Impl {
#ifdef LLAMA_SERVER_OPEN_TELEMETRY
    otel::nostd::shared_ptr<otel_trace::Span>                         span;
    otel::nostd::shared_ptr<otel_ctx::propagation::TextMapPropagator> propagator;
    otel_ctx::Context                                                 context;
    bool                                                              is_server = false;
#endif
    std::atomic<bool> ended{ false };
};

server_telemetry_span::server_telemetry_span(std::unique_ptr<Impl> impl) : pimpl(std::move(impl)) {}

server_telemetry_span::~server_telemetry_span() {
    end();
}

void server_telemetry_span::set_http_status(int status_code) {
#ifdef LLAMA_SERVER_OPEN_TELEMETRY
    if (!pimpl || pimpl->ended.load() || !pimpl->span) {
        return;
    }
    pimpl->span->SetAttribute("http.response.status_code", static_cast<int64_t>(status_code));
    if ((pimpl->is_server && status_code >= 500) || (!pimpl->is_server && status_code >= 400)) {
        pimpl->span->SetStatus(otel_trace::StatusCode::kError);
        pimpl->span->SetAttribute("error.type", std::to_string(status_code));
    }
#else
    (void) status_code;
#endif
}

void server_telemetry_span::set_error(const std::string & error_type) {
#ifdef LLAMA_SERVER_OPEN_TELEMETRY
    if (!pimpl || pimpl->ended.load() || !pimpl->span) {
        return;
    }
    pimpl->span->SetStatus(otel_trace::StatusCode::kError);
    pimpl->span->SetAttribute("error.type", error_type);
#else
    (void) error_type;
#endif
}

void server_telemetry_span::end() {
    if (!pimpl || pimpl->ended.exchange(true)) {
        return;
    }
#ifdef LLAMA_SERVER_OPEN_TELEMETRY
    if (pimpl->span) {
        pimpl->span->End();
    }
#endif
}

bool server_telemetry_init(bool enabled, const std::string & service_version, std::string & error) {
    if (!enabled) {
        return true;
    }

#ifndef LLAMA_SERVER_OPEN_TELEMETRY
    (void) service_version;
    error = "OpenTelemetry support is not compiled in; rebuild with -DLLAMA_SERVER_OPEN_TELEMETRY=ON";
    return false;
#else
    try {
        auto propagator = otel::nostd::shared_ptr<otel_ctx::propagation::TextMapPropagator>(
            new otel_trace::propagation::HttpTraceContext());

        if (otel_common::GetSdkDisabled()) {
            std::lock_guard<std::mutex> lock(g_telemetry_mutex);
            g_propagator      = std::move(propagator);
            g_service_version = service_version;
            g_enabled         = true;
            LOG_INF("OpenTelemetry SDK disabled by OTEL_SDK_DISABLED; trace-context propagation remains enabled\n");
            return true;
        }

        auto exporter = otel_otlp::OtlpHttpExporterFactory::Create(otel_otlp::OtlpHttpExporterOptions{});
        otel_sdk::BatchSpanProcessorOptions processor_options{};
        auto processor = otel_sdk::BatchSpanProcessorFactory::Create(std::move(exporter), processor_options);

        otel_res::ResourceAttributes attributes = {
            { "service.name",    std::string("llama-server") },
            { "service.version", service_version             },
        };
        auto                                      resource = otel_res::Resource::Create(attributes);
        auto                                      sampler  = make_sampler();
        std::shared_ptr<otel_sdk::TracerProvider> provider(
            otel_sdk::TracerProviderFactory::Create(std::move(processor), resource, std::move(sampler)));

        {
            std::lock_guard<std::mutex> lock(g_telemetry_mutex);
            g_provider        = provider;
            g_propagator      = std::move(propagator);
            g_service_version = service_version;
            g_enabled         = true;
        }

        LOG_INF("OpenTelemetry OTLP/HTTP tracing enabled\n");
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
#endif
}

void server_telemetry_shutdown() {
#ifdef LLAMA_SERVER_OPEN_TELEMETRY
    std::shared_ptr<otel_sdk::TracerProvider> provider;
    {
        std::lock_guard<std::mutex> lock(g_telemetry_mutex);
        if (!g_enabled) {
            return;
        }
        g_enabled    = false;
        provider     = std::move(g_provider);
        g_propagator = {};
        g_service_version.clear();
    }

    constexpr auto timeout = std::chrono::seconds(5);
    if (provider) {
        if (!provider->ForceFlush(timeout)) {
            LOG_WRN("OpenTelemetry trace flush timed out\n");
        }
        if (!provider->Shutdown(timeout)) {
            LOG_WRN("OpenTelemetry trace shutdown timed out\n");
        }
    }
#endif
}

server_telemetry_span_ptr server_telemetry_start_server_span(const std::string &                        method,
                                                             const std::string &                        route,
                                                             const std::string &                        path,
                                                             const std::string &                        scheme,
                                                             const std::string &                        server_address,
                                                             int                                        server_port,
                                                             const std::string &                        client_address,
                                                             const std::map<std::string, std::string> & headers) {
#ifndef LLAMA_SERVER_OPEN_TELEMETRY
    (void) method;
    (void) route;
    (void) path;
    (void) scheme;
    (void) server_address;
    (void) server_port;
    (void) client_address;
    (void) headers;
    return nullptr;
#else
    auto telemetry = get_telemetry_state();
    if (!telemetry.enabled) {
        return nullptr;
    }

    header_carrier    carrier(headers);
    otel_ctx::Context empty_context;
    auto              parent_context = telemetry.propagator->Extract(carrier, empty_context);

    auto impl        = std::make_unique<server_telemetry_span::Impl>();
    impl->propagator = telemetry.propagator;
    impl->context    = parent_context;
    impl->is_server  = true;

    if (!telemetry.tracer) {
        return server_telemetry_span_ptr(new server_telemetry_span(std::move(impl)));
    }

    otel_trace::StartSpanOptions options;
    options.kind   = otel_trace::SpanKind::kServer;
    options.parent = parent_context;

    auto span     = telemetry.tracer->StartSpan(method + " " + route,
                                                {
                                                { "http.request.method", method                            },
                                                { "http.route",          route                             },
                                                { "url.path",            path                              },
                                                { "url.scheme",          scheme                            },
                                                { "server.address",      server_address                    },
                                                { "server.port",         static_cast<int64_t>(server_port) },
                                                { "client.address",      client_address                    },
    },
                                                options);
    impl->context = otel_trace::SetSpan(parent_context, span);
    impl->span    = std::move(span);
    return server_telemetry_span_ptr(new server_telemetry_span(std::move(impl)));
#endif
}

server_telemetry_span_ptr server_telemetry_start_client_span(const std::string &               method,
                                                             const std::string &               scheme,
                                                             const std::string &               server_address,
                                                             int                               server_port,
                                                             const std::string &               path,
                                                             const server_telemetry_span_ptr & parent) {
#ifndef LLAMA_SERVER_OPEN_TELEMETRY
    (void) method;
    (void) scheme;
    (void) server_address;
    (void) server_port;
    (void) path;
    (void) parent;
    return nullptr;
#else
    auto telemetry = get_telemetry_state();
    if (!telemetry.enabled) {
        return nullptr;
    }

    otel_ctx::Context parent_context;
    if (parent && parent->pimpl) {
        parent_context = parent->pimpl->context;
    }

    auto impl        = std::make_unique<server_telemetry_span::Impl>();
    impl->propagator = telemetry.propagator;
    impl->context    = parent_context;
    impl->is_server  = false;

    if (!telemetry.tracer) {
        return server_telemetry_span_ptr(new server_telemetry_span(std::move(impl)));
    }

    otel_trace::StartSpanOptions options;
    options.kind   = otel_trace::SpanKind::kClient;
    options.parent = parent_context;

    const size_t      query_pos = path.find('?');
    const std::string url_path  = path.substr(0, query_pos);
    auto              span      = telemetry.tracer->StartSpan(method,
                                                              {
                                                { "http.request.method", method                            },
                                                { "url.path",            url_path                          },
                                                { "url.scheme",          scheme                            },
                                                { "server.address",      server_address                    },
                                                { "server.port",         static_cast<int64_t>(server_port) },
    },
                                                              options);
    impl->context               = otel_trace::SetSpan(parent_context, span);
    impl->span                  = std::move(span);
    return server_telemetry_span_ptr(new server_telemetry_span(std::move(impl)));
#endif
}

void server_telemetry_inject(std::map<std::string, std::string> & headers, const server_telemetry_span_ptr & span) {
#ifdef LLAMA_SERVER_OPEN_TELEMETRY
    if (!span || !span->pimpl || !span->pimpl->propagator || span->pimpl->ended.load()) {
        return;
    }

    header_carrier carrier(headers);
    span->pimpl->propagator->Inject(carrier, span->pimpl->context);
#else
    (void) headers;
    (void) span;
#endif
}
