#include "router-endpoints.h"

#include "router-app.h"
#include "router-config.h"
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
            res.status = 503;
            res.set_content("no models running", "text/plain");
            return;
        }

        std::string error;
        if (!app.ensure_running(model, error)) {
            res.status = 503;
            res.set_content("no models running", "text/plain");
            return;
        }

        proxy_request(req, res, app.upstream_for(model));
    };

    server.Get("/props", proxy_last_spawned);
    server.Get("/slots", proxy_last_spawned);
    server.Get("/health", proxy_last_spawned);

    server.Get(R"(^/(.+)/(health|props|slots)$)", [&app](const httplib::Request & req, httplib::Response & res) {
        auto model_it = req.matches.begin();
        ++model_it;
        std::string model_name = model_it != req.matches.end() ? model_it->str() : std::string();
        std::string error;
        if (!app.ensure_running(model_name, error)) {
            res.status = 404;
            res.set_content("{\"error\":\"model unavailable\"}", "application/json");
            return;
        }
        proxy_request(req, res, app.upstream_for(model_name));
    });

    server.Post("/v1/chat/completions", [&app](const httplib::Request & req, httplib::Response & res) {
        std::string model;
        if (!parse_model_from_chat(req, model)) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid json or model missing\"}", "application/json");
            return;
        }

        std::string error;
        if (!app.ensure_running(model, error)) {
            res.status = 404;
            res.set_content("{\"error\":\"" + error + "\"}", "application/json");
            return;
        }

        proxy_request(req, res, app.upstream_for(model));
    });

    server.Post("/admin/reload", [&app](const httplib::Request &, httplib::Response & res) {
        app.stop_all();
        app.start_auto_models();
        res.set_content("{\"status\":\"reloaded\"}", "application/json");
    });

    server.set_error_handler([](const httplib::Request &, httplib::Response & res) {
        res.status = 404;
        res.set_content("{\"error\":\"not found\"}", "application/json");
    });
}
