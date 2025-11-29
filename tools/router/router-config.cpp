#include "router-config.h"

#include "common.h"
#include "log.h"
#include "router-scanner.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>

using json = nlohmann::json;

std::string get_default_config_path() {
    const char * home = std::getenv("HOME");
#if defined(_WIN32)
    if (home == nullptr) {
        home = std::getenv("USERPROFILE");
    }
#endif
    std::string base = home ? std::string(home) : std::string();
    if (!base.empty() && base.back() != DIRECTORY_SEPARATOR) {
        base.push_back(DIRECTORY_SEPARATOR);
    }
    return base + ".config" + DIRECTORY_SEPARATOR + "llama.cpp" + DIRECTORY_SEPARATOR + "router-config.json";
}

std::string expand_user_path(const std::string & path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        const char * home = std::getenv("HOME");
#if defined(_WIN32)
        if (home == nullptr) {
            home = std::getenv("USERPROFILE");
        }
#endif
        if (home != nullptr) {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

static void ensure_parent_directory(const std::string & path) {
    std::filesystem::path p(path);
    std::error_code      ec;
    auto parent = p.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent, ec)) {
        std::filesystem::create_directories(parent, ec);
    }
}

void write_config_file(const RouterConfig & cfg, const std::string & path) {
    json out;
    out["version"]       = cfg.version;
    out["default_spawn"] = cfg.default_spawn;
    out["router"]        = {{"host", cfg.router.host}, {"port", cfg.router.port}, {"base_port", cfg.router.base_port}, {"log_level", cfg.router.log_level}};

    out["models"] = json::array();
    for (const auto & m : cfg.models) {
        json obj;
        obj["name"]  = m.name;
        obj["path"]  = m.path;
        obj["state"] = m.state.empty() ? "manual" : m.state;
        if (!m.spawn.empty()) {
            obj["spawn"] = m.spawn;
        }
        out["models"].push_back(std::move(obj));
    }

    ensure_parent_directory(path);

    std::ofstream fout(path);
    if (!fout) {
        throw std::runtime_error("failed to write config file: " + path);
    }
    fout << out.dump(4) << std::endl;
}

RouterConfig generate_default_config(const std::string & path) {
    RouterConfig cfg;
    cfg.version       = "1.0";
    cfg.default_spawn = {"llama-server", "--ctx-size", "4096", "--n-gpu-layers", "99"};
    cfg.router        = RouterOptions{};
    cfg.models        = scan_default_models();

    write_config_file(cfg, path);
    LOG_INF("generated default config at %s\n", path.c_str());
    return cfg;
}

RouterConfig load_config(const std::string & path) {
    RouterConfig cfg;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return generate_default_config(path);
    }

    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("failed to open config file: " + path);
    }

    json data = json::parse(fin);
    if (data.contains("version")) {
        cfg.version = data["version"].get<std::string>();
    }
    if (data.contains("default_spawn")) {
        cfg.default_spawn = data["default_spawn"].get<std::vector<std::string>>();
    }
    if (data.contains("router")) {
        auto r = data["router"];
        if (r.contains("host")) cfg.router.host = r["host"].get<std::string>();
        if (r.contains("port")) cfg.router.port = r["port"].get<int>();
        if (r.contains("base_port")) cfg.router.base_port = r["base_port"].get<int>();
        if (r.contains("log_level")) cfg.router.log_level = r["log_level"].get<std::string>();
    }
    if (data.contains("models")) {
        for (const auto & m : data["models"]) {
            ModelConfig mc;
            mc.name  = m.value("name", "");
            mc.path  = m.value("path", "");
            mc.state = m.value("state", "manual");
            if (m.contains("spawn")) {
                mc.spawn = m["spawn"].get<std::vector<std::string>>();
            }
            cfg.models.push_back(std::move(mc));
        }
    }
    return cfg;
}
