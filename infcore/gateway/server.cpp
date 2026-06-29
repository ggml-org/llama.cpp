// infcore gateway — корпоративная лицензия.
#include "server.hpp"

#include <csignal>
#include <cstdio>
#include <sstream>

#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace infcore {

GatewayServer::GatewayServer(GatewayConfig cfg) : cfg_(std::move(cfg)) {
    for (const auto& m : cfg_.models) registry_.add(m);

    BackendSupervisor::Options opt;
    opt.llama_server_bin   = cfg_.llama_server_bin;
    opt.host               = cfg_.host;
    opt.port_range_start   = cfg_.port_range_start;
    opt.idle_timeout_ms    = cfg_.idle_timeout_ms;
    opt.startup_timeout_ms = cfg_.startup_timeout_ms;
    supervisor_ = std::make_unique<BackendSupervisor>(opt);
}

void GatewayServer::inc(const std::string& key) {
    std::lock_guard<std::mutex> lock(metrics_mu_);
    counters_[key].fetch_add(1, std::memory_order_relaxed);
}
long GatewayServer::get_counter(const std::string& key) {
    std::lock_guard<std::mutex> lock(metrics_mu_);
    return counters_[key].load(std::memory_order_relaxed);
}
std::string GatewayServer::render_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mu_);
    std::ostringstream os;
    for (auto& kv : counters_)
        os << "infcore_gateway_" << kv.first << " "
           << kv.second.load(std::memory_order_relaxed) << "\n";
    return os.str();
}

// --- helpers -----------------------------------------------------------------
namespace {

bool bearer_ok(const httplib::Request& req, const std::vector<std::string>& keys) {
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return false;
    const std::string& v = it->second;
    const std::string pfx = "Bearer ";
    if (v.size() <= pfx.size() || v.compare(0, pfx.size(), pfx) != 0) return false;
    std::string tok = v.substr(pfx.size());
    for (const auto& k : keys) if (k == tok) return true;
    return false;
}

void error_json(httplib::Response& res, int status, const std::string& type,
                const std::string& msg) {
    json e = {{"error", {{"type", type}, {"message", msg}, {"code", status}}}};
    res.status = status;
    res.set_content(e.dump(), "application/json");
}

// Штатная остановка по сигналу: на гибель от сигнала C++ деструкторы не вызываются,
// поэтому ловим SIGINT/SIGTERM и просим listen() вернуться -> отработает
// ~BackendSupervisor и погасит дочерние llama-server (без осиротевших процессов).
httplib::Server* g_active_srv = nullptr;
void on_term(int) { if (g_active_srv) g_active_srv->stop(); }

}  // namespace

// --- server ------------------------------------------------------------------
int GatewayServer::run() {
    httplib::Server svr;

    g_active_srv = &svr;
    std::signal(SIGINT, on_term);
    std::signal(SIGTERM, on_term);
    std::signal(SIGPIPE, SIG_IGN);   // оборванный клиент при SSE не должен ронять gateway

    // health (без авторизации)
    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        json h = {{"status", "ok"}, {"models", (int)cfg_.models.size()}};
        res.set_content(h.dump(), "application/json");
    });

    // pull-метрики (без авторизации; контур закрытый)
    svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(render_metrics(), "text/plain; version=0.0.4");
    });

    // /v1/models — OpenAI-совместимый список
    svr.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        if (!bearer_ok(req, cfg_.api_keys)) { inc("errors_total{type=\"unauthorized\"}");
            return error_json(res, 401, "invalid_request_error", "unauthorized"); }
        json data = json::array();
        for (const auto& m : registry_.list()) {
            if (!m.enabled) continue;
            data.push_back({{"id", m.logical_name}, {"object", "model"},
                            {"owned_by", "infcore"},
                            {"modality", modality_to_string(m.modality)}});
        }
        res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
    });

    // общий прокси на бэкенд llama-server (chat/completions, completions, embeddings)
    auto proxy = [this](const char* upstream_path, bool allow_stream) {
        return [this, upstream_path, allow_stream](const httplib::Request& req,
                                                   httplib::Response& res) {
            inc(std::string("requests_total{path=\"") + upstream_path + "\"}");
            if (!bearer_ok(req, cfg_.api_keys)) { inc("errors_total{type=\"unauthorized\"}");
                return error_json(res, 401, "invalid_request_error", "unauthorized"); }

            json body;
            try { body = json::parse(req.body); }
            catch (...) { inc("errors_total{type=\"bad_request\"}");
                return error_json(res, 400, "invalid_request_error", "невалидный JSON"); }

            const std::string model = body.value("model", std::string());
            ModelEntry e;
            if (model.empty() || !registry_.get(model, e)) { inc("errors_total{type=\"model_not_found\"}");
                return error_json(res, 404, "invalid_request_error", "model not found: " + model); }
            if (!e.enabled) { inc("errors_total{type=\"model_disabled\"}");
                return error_json(res, 409, "invalid_request_error", "model disabled: " + model); }

            // backend_url пуст -> модель управляемая: поднимаем процесс по требованию.
            std::string backend = e.backend_url;
            const bool managed = backend.empty();
            if (managed) {
                std::string serr;
                backend = supervisor_->ensure_ready(e, serr);
                if (backend.empty()) { inc("errors_total{type=\"backend_start_failed\"}");
                    return error_json(res, 502, "api_error", "не удалось поднять бэкенд: " + serr); }
                supervisor_->acquire(e.logical_name);
            }

            // подмена имени модели на имя бэкенда, если задано
            if (!e.upstream_model.empty()) body["model"] = e.upstream_model;
            const std::string out_body = body.dump();

            const bool stream = allow_stream && body.value("stream", false);

            httplib::Client cli(backend);
            cli.set_read_timeout(cfg_.request_timeout_ms / 1000, 0);
            cli.set_write_timeout(cfg_.request_timeout_ms / 1000, 0);
            httplib::Headers fwd = {{"Content-Type", "application/json"}};

            if (!stream) {
                auto up = cli.Post(upstream_path, fwd, out_body, "application/json");
                if (managed) supervisor_->release(e.logical_name);
                if (!up) { inc("errors_total{type=\"backend_unreachable\"}");
                    return error_json(res, 502, "api_error", "бэкенд недоступен: " + backend); }
                res.status = up->status;
                res.set_content(up->body, "application/json");
                return;
            }

            // SSE passthrough: тянем поток из бэкенда и пишем в sink целиком за один вызов.
            const std::string logical = e.logical_name;
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, backend, managed, logical, upstream_path, fwd, out_body](
                    size_t /*offset*/, httplib::DataSink& sink) {
                    httplib::Client up(backend);
                    up.set_read_timeout(cfg_.request_timeout_ms / 1000, 0);
                    auto r = up.Post(upstream_path, fwd, out_body, "application/json",
                                     [&sink](const char* data, size_t len) {
                                         return sink.write(data, len);
                                     });
                    if (!r) {
                        const char* msg =
                            "data: {\"error\":{\"message\":\"backend unreachable\"}}\n\n";
                        sink.write(msg, std::char_traits<char>::length(msg));
                    }
                    sink.done();
                    if (managed) supervisor_->release(logical);
                    return true;
                });
        };
    };

    svr.Post("/v1/chat/completions", proxy("/v1/chat/completions", true));
    svr.Post("/v1/completions",      proxy("/v1/completions", true));
    svr.Post("/v1/embeddings",       proxy("/v1/embeddings", false));

    std::printf("infcore gateway слушает http://%s:%d (моделей: %zu)\n",
                cfg_.host.c_str(), cfg_.port, cfg_.models.size());
    bool ok = svr.listen(cfg_.host, cfg_.port);
    g_active_srv = nullptr;
    if (!ok) {
        std::fprintf(stderr, "infcore: не удалось слушать %s:%d\n",
                     cfg_.host.c_str(), cfg_.port);
        return 1;
    }
    return 0;
}

}  // namespace infcore
