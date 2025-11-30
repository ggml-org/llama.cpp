#include "router-admin.h"

#include "log.h"
#include "router-config.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static bool authorize_admin(const RouterConfig & cfg, const httplib::Request & req, httplib::Response & res) {
    if (cfg.router.admin_token.empty()) {
        return true;
    }

    const std::string bearer = "Bearer " + cfg.router.admin_token;
    const auto        auth   = req.get_header_value("Authorization");
    const auto        token  = req.get_header_value("X-Admin-Token");

    if (auth == bearer || token == cfg.router.admin_token) {
        return true;
    }

    res.status = 403;
    res.set_content("{\"error\":\"forbidden\"}", "application/json");
    LOG_WRN("Admin endpoint rejected unauthorized request from %s:%d\n", req.remote_addr.c_str(), req.remote_port);
    return false;
}

void register_admin_routes(httplib::Server & server, RouterApp & app, const std::string & config_path) {
    server.Post("/admin/reload", [&app](const httplib::Request & req, httplib::Response & res) {
        if (!authorize_admin(app.get_config(), req, res)) {
            return;
        }
        LOG_INF("Reloading router application: stopping managed models\n");
        app.stop_all();
        res.set_content("{\"status\":\"reloaded\"}", "application/json");
    });

    server.Get("/admin/rescan", [&app, config_path](const httplib::Request & req, httplib::Response & res) {
        if (!authorize_admin(app.get_config(), req, res)) {
            return;
        }

        const auto rescan_result = rescan_auto_models(app.get_config());
        LOG_INF("Admin rescan requested, found %zu new models (removed %zu)\n",
                rescan_result.added,
                rescan_result.removed);
        app.update_config(rescan_result.config);

        if (!config_path.empty() && (rescan_result.added > 0 || rescan_result.removed > 0)) {
            LOG_INF("Persisting updated configuration to %s\n", config_path.c_str());
            write_config_file(app.get_config(), config_path);
        }

        json out;
        out["status"]     = "rescanned";
        out["new_models"] = rescan_result.added;
        out["removed"]    = rescan_result.removed;
        res.set_content(out.dump(), "application/json");
    });
}
