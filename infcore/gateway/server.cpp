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

    // RBAC: роли из конфига; principals -> ключи.
    rbac_.set_enabled(cfg_.rbac_enabled);
    bool has_admin_role = false;
    for (const auto& r : cfg_.roles) { rbac_.add_role(r); if (r.name == "admin") has_admin_role = true; }
    for (const auto& ap : cfg_.principals) authn_.add_key(ap.api_key, ap.principal);

    // Обратная совместимость: плоские api_keys = principal'ы с ролью admin.
    if (!cfg_.api_keys.empty()) {
        if (!has_admin_role) {
            Role admin; admin.name = "admin";
            admin.allow_models = {"*"}; admin.allow_endpoints = {"*"};
            rbac_.add_role(admin);
        }
        for (const auto& k : cfg_.api_keys) authn_.add_key(k, Principal{"legacy", "admin"});
    }

    if (cfg_.audit_sink == "file" && !audit_.open(cfg_.audit_path))
        std::fprintf(stderr, "infcore: не удалось открыть audit-журнал: %s\n", cfg_.audit_path.c_str());
}

void GatewayServer::audit_event(const Principal& pr, const std::string& client_ip,
                                const std::string& endpoint, const std::string& model,
                                const char* decision, const std::string& reason, int status) {
    AuditEvent ev;
    ev.subject = pr.subject;
    ev.role = pr.role;
    ev.endpoint = endpoint;
    ev.model = model;
    ev.client_ip = client_ip;
    ev.decision = decision;
    ev.reason = reason;
    ev.status = status;
    audit_.log(ev);
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

void error_json(httplib::Response& res, int status, const std::string& type,
                const std::string& msg) {
    json e = {{"error", {{"type", type}, {"message", msg}, {"code", status}}}};
    res.status = status;
    res.set_content(e.dump(), "application/json");
}

std::string auth_token(const httplib::Request& req) {
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return std::string();
    return parse_bearer(it->second);
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

    // /v1/models — OpenAI-совместимый список (только модели, разрешённые роли)
    svr.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        Principal pr;
        if (!authn_.verify(auth_token(req), pr)) { inc("errors_total{type=\"unauthorized\"}");
            audit_event({}, req.remote_addr, "/v1/models", "", "deny", "unauthorized", 401);
            return error_json(res, 401, "invalid_request_error", "unauthorized"); }
        json data = json::array();
        for (const auto& m : registry_.list()) {
            if (!m.enabled) continue;
            if (!rbac_.model_allowed(pr.role, m.logical_name)) continue;
            data.push_back({{"id", m.logical_name}, {"object", "model"},
                            {"owned_by", "infcore"},
                            {"modality", modality_to_string(m.modality)}});
        }
        audit_event(pr, req.remote_addr, "/v1/models", "", "allow", "", 200);
        res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
    });

    // общий прокси на бэкенд llama-server (chat/completions, completions, embeddings)
    auto proxy = [this](const char* upstream_path, bool allow_stream) {
        return [this, upstream_path, allow_stream](const httplib::Request& req,
                                                   httplib::Response& res) {
            inc(std::string("requests_total{path=\"") + upstream_path + "\"}");
            Principal pr;
            if (!authn_.verify(auth_token(req), pr)) { inc("errors_total{type=\"unauthorized\"}");
                audit_event({}, req.remote_addr, upstream_path, "", "deny", "unauthorized", 401);
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

            // RBAC: роль должна допускать и endpoint, и модель (default-deny).
            std::string reason;
            if (!rbac_.allow(pr.role, upstream_path, model, reason)) {
                inc("errors_total{type=\"forbidden\"}");
                audit_event(pr, req.remote_addr, upstream_path, model, "deny", reason, 403);
                return error_json(res, 403, "invalid_request_error", "forbidden: " + reason);
            }

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
                    audit_event(pr, req.remote_addr, upstream_path, model, "allow", "backend unreachable", 502);
                    return error_json(res, 502, "api_error", "бэкенд недоступен: " + backend); }
                res.status = up->status;
                res.set_content(up->body, "application/json");
                audit_event(pr, req.remote_addr, upstream_path, model, "allow", "", up->status);
                return;
            }

            // SSE passthrough: тянем поток из бэкенда и пишем в sink целиком за один вызов.
            const std::string logical = e.logical_name;
            const std::string client_ip = req.remote_addr;
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, backend, managed, logical, pr, model, client_ip, upstream_path, fwd, out_body](
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
                    audit_event(pr, client_ip, upstream_path, model, "allow",
                                r ? "" : "backend unreachable", 200);
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
