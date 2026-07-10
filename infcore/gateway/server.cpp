// infcore gateway — корпоративная лицензия.
#include "server.hpp"

#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace infcore {

GatewayServer::GatewayServer(GatewayConfig cfg) : cfg_(std::move(cfg)) {
    for (const auto& m : cfg_.models) registry_.add(m);

    BackendSupervisor::Options opt;
    opt.llama_server_bin   = cfg_.llama_server_bin;
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

// OpenAI-совместимая ошибка: {"error":{message,type,param,code}}.
// code - строковый машиночитаемый код (или null), как у OpenAI; не HTTP-статус.
void error_json(httplib::Response& res, int status, const std::string& type,
                const std::string& msg, const std::string& code = "") {
    json err = {{"type", type}, {"message", msg}, {"param", nullptr}};
    if (code.empty()) err["code"] = nullptr; else err["code"] = code;
    res.status = status;
    res.set_content(json{{"error", err}}.dump(), "application/json");
}

std::string auth_token(const httplib::Request& req) {
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return std::string();
    return parse_bearer(it->second);
}

// Разделяемое состояние SSE-прокси: фоновый поток тянет ответ бэкенда, а
// content-provider downstream отдаёт байты клиенту. Статус бэкенда становится
// известен (headers_ready) ДО коммита стрим-ответа - так мы можем вернуть
// обычную JSON-ошибку на не-2xx, не открывая text/event-stream.
struct StreamPump {
    std::mutex m;
    std::condition_variable cv;
    std::string buf;
    bool headers_ready = false;
    int  status = 0;
    bool done = false;
    bool ok = false;             // send() дошёл без транспортной ошибки
    bool consumer_gone = false;  // клиент отвалился -> прекращаем тянуть бэкенд
};

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
        std::string reason;
        if (!rbac_.allow(pr.role, "/v1/models", "", reason)) { inc("errors_total{type=\"forbidden\"}");
            audit_event(pr, req.remote_addr, "/v1/models", "", "deny", reason, 403);
            return error_json(res, 403, "invalid_request_error", "forbidden: " + reason); }
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

    // --- admin: управление моделями без перезапуска (доступ по RBAC: endpoint /admin/models) ---

    // GET /admin/models — полный список со статусом (включая выключенные).
    svr.Get("/admin/models", [this](const httplib::Request& req, httplib::Response& res) {
        Principal pr;
        if (!authn_.verify(auth_token(req), pr)) { inc("errors_total{type=\"unauthorized\"}");
            audit_event({}, req.remote_addr, "/admin/models", "", "deny", "unauthorized", 401);
            return error_json(res, 401, "invalid_request_error", "unauthorized"); }
        std::string reason;
        if (!rbac_.allow(pr.role, "/admin/models", "", reason)) { inc("errors_total{type=\"forbidden\"}");
            audit_event(pr, req.remote_addr, "/admin/models", "", "deny", reason, 403);
            return error_json(res, 403, "invalid_request_error", "forbidden: " + reason); }
        json data = json::array();
        for (const auto& m : registry_.list())
            data.push_back({{"id", m.logical_name}, {"modality", modality_to_string(m.modality)},
                            {"enabled", m.enabled}, {"managed", m.backend_url.empty()},
                            {"backend_url", m.backend_url}, {"arch", m.arch}});
        audit_event(pr, req.remote_addr, "/admin/models", "", "allow", "", 200);
        res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
    });

    // POST /admin/models/<name>/<enable|disable> — переключение в рантайме.
    svr.Post(R"(/admin/models/([^/]+)/(enable|disable))",
             [this](const httplib::Request& req, httplib::Response& res) {
        Principal pr;
        if (!authn_.verify(auth_token(req), pr)) { inc("errors_total{type=\"unauthorized\"}");
            audit_event({}, req.remote_addr, "/admin/models", "", "deny", "unauthorized", 401);
            return error_json(res, 401, "invalid_request_error", "unauthorized"); }
        std::string reason;
        if (!rbac_.allow(pr.role, "/admin/models", "", reason)) { inc("errors_total{type=\"forbidden\"}");
            audit_event(pr, req.remote_addr, "/admin/models", "", "deny", reason, 403);
            return error_json(res, 403, "invalid_request_error", "forbidden: " + reason); }
        const std::string name   = req.matches[1];
        const bool        enable = (req.matches[2] == "enable");
        if (!registry_.set_enabled(name, enable)) { inc("errors_total{type=\"model_not_found\"}");
            audit_event(pr, req.remote_addr, "/admin/models", name, "deny", "model not found", 404);
            return error_json(res, 404, "invalid_request_error", "model not found: " + name); }
        audit_event(pr, req.remote_addr, "/admin/models", name, "allow",
                    enable ? "enabled" : "disabled", 200);
        res.set_content(json{{"id", name}, {"enabled", enable}}.dump(), "application/json");
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

            const std::string client_ip = req.remote_addr;

            json body;
            try { body = json::parse(req.body); }
            catch (...) { inc("errors_total{type=\"bad_request\"}");
                audit_event(pr, client_ip, upstream_path, "", "deny", "invalid JSON", 400);
                return error_json(res, 400, "invalid_request_error", "невалидный JSON"); }

            const std::string model = body.value("model", std::string());
            ModelEntry e;
            if (model.empty() || !registry_.get(model, e)) { inc("errors_total{type=\"model_not_found\"}");
                audit_event(pr, client_ip, upstream_path, model, "deny", "model not found", 404);
                return error_json(res, 404, "invalid_request_error", "model not found: " + model, "model_not_found"); }
            if (!e.enabled) { inc("errors_total{type=\"model_disabled\"}");
                audit_event(pr, client_ip, upstream_path, model, "deny", "model disabled", 409);
                return error_json(res, 409, "invalid_request_error", "model disabled: " + model, "model_disabled"); }

            // RBAC: роль должна допускать и endpoint, и модель (default-deny).
            std::string reason;
            if (!rbac_.allow(pr.role, upstream_path, model, reason)) {
                inc("errors_total{type=\"forbidden\"}");
                audit_event(pr, client_ip, upstream_path, model, "deny", reason, 403);
                return error_json(res, 403, "invalid_request_error", "forbidden: " + reason);
            }

            // backend_url пуст -> модель управляемая: поднимаем процесс по требованию.
            std::string backend = e.backend_url;
            const bool managed = backend.empty();
            const std::string logical = e.logical_name;

            // RAII-токен активного запроса: release гарантирован даже при обрыве клиента
            // (иначе счётчик active утёк бы и reaper никогда не выгрузил бы бэкенд).
            std::shared_ptr<void> active_token;
            if (managed) {
                std::string serr;
                backend = supervisor_->ensure_ready(e, serr);
                if (backend.empty()) { inc("errors_total{type=\"backend_start_failed\"}");
                    audit_event(pr, client_ip, upstream_path, model, "error", "backend start failed", 502);
                    return error_json(res, 502, "api_error", "не удалось поднять бэкенд: " + serr); }
                supervisor_->acquire(logical);
                BackendSupervisor* sup = supervisor_.get();
                active_token = std::shared_ptr<void>(nullptr, [sup, logical](void*) { sup->release(logical); });
            }

            // подмена имени модели на имя бэкенда, если задано
            if (!e.upstream_model.empty()) body["model"] = e.upstream_model;
            const bool stream = allow_stream && body.value("stream", false);
            if (!allow_stream) body.erase("stream");  // embeddings: не транслируем stream бэкенду
            const std::string out_body = body.dump();

            const int t_sec  = cfg_.request_timeout_ms / 1000;
            const int t_usec = (cfg_.request_timeout_ms % 1000) * 1000;

            httplib::Headers fwd = {{"Content-Type", "application/json"}};
            if (managed) fwd.emplace("Authorization", "Bearer " + supervisor_->api_key());

            if (!stream) {
                httplib::Client cli(backend);
                cli.set_read_timeout(t_sec, t_usec);
                cli.set_write_timeout(t_sec, t_usec);
                auto up = cli.Post(upstream_path, fwd, out_body, "application/json");
                if (!up) { inc("errors_total{type=\"backend_unreachable\"}");
                    audit_event(pr, client_ip, upstream_path, model, "error", "backend unreachable", 502);
                    return error_json(res, 502, "api_error", "бэкенд недоступен: " + backend); }
                res.status = up->status;
                res.set_content(up->body, up->has_header("Content-Type")
                                              ? up->get_header_value("Content-Type")
                                              : "application/json");
                audit_event(pr, client_ip, upstream_path, model,
                            up->status < 400 ? "allow" : "error", "", up->status);
                return;
            }

            // Стриминг: фоновый поток тянет ответ бэкенда, статус узнаём до коммита stream-ответа.
            auto pump = std::make_shared<StreamPump>();
            std::thread([backend, upstream_path, fwd, out_body, t_sec, t_usec, pump] {
                httplib::Client cli(backend);
                cli.set_read_timeout(t_sec, t_usec);
                cli.set_write_timeout(t_sec, t_usec);
                httplib::Request rq;
                rq.method = "POST";
                rq.path = upstream_path;
                rq.headers = fwd;
                rq.body = out_body;
                rq.response_handler = [pump](const httplib::Response& rr) {
                    std::lock_guard<std::mutex> lk(pump->m);
                    pump->status = rr.status;
                    pump->headers_ready = true;
                    pump->cv.notify_all();
                    return true;
                };
                rq.content_receiver = [pump](const char* d, size_t n, size_t, size_t) {
                    std::lock_guard<std::mutex> lk(pump->m);
                    pump->buf.append(d, n);
                    pump->cv.notify_all();
                    return !pump->consumer_gone;
                };
                httplib::Response rr;
                httplib::Error er = httplib::Error::Success;
                bool sent = cli.send(rq, rr, er);
                std::lock_guard<std::mutex> lk(pump->m);
                pump->ok = sent && er == httplib::Error::Success;
                if (!pump->headers_ready) { pump->status = pump->ok ? rr.status : 0; pump->headers_ready = true; }
                pump->done = true;
                pump->cv.notify_all();
            }).detach();

            int ust;
            {
                std::unique_lock<std::mutex> lk(pump->m);
                pump->cv.wait(lk, [&] { return pump->headers_ready; });
                ust = pump->status;
            }

            // Ошибка ДО стрима (невалидные параметры / бэкенд недоступен) -> обычная JSON-ошибка,
            // а не SSE внутри 200.
            if (ust < 200 || ust >= 300) {
                std::string ebody; int st;
                {
                    std::unique_lock<std::mutex> lk(pump->m);
                    pump->cv.wait(lk, [&] { return pump->done; });
                    ebody = pump->buf;
                    st = pump->status ? pump->status : 502;
                }
                inc("errors_total{type=\"backend_error\"}");
                audit_event(pr, client_ip, upstream_path, model, "error",
                            "backend status " + std::to_string(st), st);
                res.status = st;
                if (!ebody.empty()) res.set_content(ebody, "application/json");
                else error_json(res, st, "api_error", "бэкенд вернул ошибку");
                return;  // active_token освобождается здесь
            }

            // 2xx: passthrough SSE. active_token живёт внутри provider -> release по завершении
            // стрима (в т.ч. при обрыве клиента).
            res.set_chunked_content_provider(
                "text/event-stream",
                [pump, active_token](size_t /*offset*/, httplib::DataSink& sink) {
                    std::unique_lock<std::mutex> lk(pump->m);
                    pump->cv.wait(lk, [&] { return !pump->buf.empty() || pump->done; });
                    if (!pump->buf.empty()) {
                        std::string chunk;
                        chunk.swap(pump->buf);
                        lk.unlock();
                        if (!sink.write(chunk.data(), chunk.size())) {
                            std::lock_guard<std::mutex> lk2(pump->m);
                            pump->consumer_gone = true;
                            return false;
                        }
                        return true;
                    }
                    const bool ok = pump->ok;
                    lk.unlock();
                    if (!ok) {  // поток бэкенда оборвался: синтетическая ошибка + корректное завершение
                        const char* ev =
                            "data: {\"error\":{\"message\":\"backend stream interrupted\","
                            "\"type\":\"api_error\",\"code\":null}}\n\ndata: [DONE]\n\n";
                        sink.write(ev, std::char_traits<char>::length(ev));
                    }
                    sink.done();
                    return false;
                },
                [this, pump, pr, client_ip, upstream_path, model](bool) {
                    audit_event(pr, client_ip, upstream_path, model, "allow",
                                pump->ok ? "" : "stream interrupted", 200);
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
