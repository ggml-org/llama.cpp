#include "router-endpoints.h"

#include "log.h"
#include "router-app.h"
#include "router-proxy.h"

#include <nlohmann/json.hpp>

#include <ctime>

using json = nlohmann::json;

static void handle_models(const RouterApp & app, httplib::Response & res) {
    json out;
    out["object"] = "list";
    out["data"]   = json::array();
    auto now       = static_cast<int>(time(nullptr));
    for (const auto & model : app.get_config().models) {
        out["data"].push_back({{"id", model.name}, {"object", "model"}, {"owned_by", "router"}, {"created", now}});
    }
    LOG_INF("Listing %zu models\n", out["data"].size());
    res.set_content(out.dump(), "application/json");
}

static bool parse_model_from_chat(const httplib::Request & req, std::string & model) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return false;
    }

    model = body.value("model", std::string());
    return !model.empty();
}

void register_routes(httplib::Server & server, RouterApp & app) {
    server.Get("/v1/models", [&app](const httplib::Request &, httplib::Response & res) { handle_models(app, res); });

    auto proxy_last_spawned = [&app](const httplib::Request & req, httplib::Response & res) {
        const std::string model = app.get_last_spawned_model();
        if (model.empty()) {
            LOG_WRN("No last spawned model available for %s\n", req.path.c_str());
            res.status = 503;
            res.set_content("no models running", "text/plain");
            return;
        }
        LOG_INF("Proxying %s to last spawned model %s\n", req.path.c_str(), model.c_str());
        const auto spawn_cfg = app.get_spawn_config(model);
        proxy_request(req, res, app, model, spawn_cfg.proxy_endpoints);
    };

    server.Get("/props", proxy_last_spawned);
    server.Get("/slots", proxy_last_spawned);
    server.Get("/health", proxy_last_spawned);

    server.Get(R"(^/(.+)/(health|props|slots)$)", [&app](const httplib::Request & req, httplib::Response & res) {
        auto model_it = req.matches.begin();
        ++model_it;
        std::string model_name = model_it != req.matches.end() ? model_it->str() : std::string();
        ++model_it;
        const std::string endpoint_suffix = model_it != req.matches.end() ? model_it->str() : std::string();
        LOG_INF("Proxying %s for model %s\n", req.path.c_str(), model_name.c_str());
        const auto spawn_cfg = app.get_spawn_config(model_name);
        const std::string corrected_path = "/" + endpoint_suffix;
        proxy_request(req, res, app, model_name, spawn_cfg.proxy_endpoints, corrected_path);
    });

    server.Post("/v1/chat/completions", [&app](const httplib::Request & req, httplib::Response & res) {
        std::string model;
        if (!parse_model_from_chat(req, model)) {
            LOG_WRN("Chat completion request missing model field\n");
            res.status = 400;
            res.set_content("{\"error\":\"invalid json or model missing\"}", "application/json");
            return;
        }

        LOG_INF("Proxying chat completion for model %s\n", model.c_str());
        const auto spawn_cfg = app.get_spawn_config(model);
        proxy_request(req, res, app, model, spawn_cfg.proxy_endpoints);
    });

    server.set_error_handler([](const httplib::Request &, httplib::Response & res) {
        res.status = 404;
        res.set_content("{\"error\":\"not found\"}", "application/json");
    });
}
