#include "router-config.h"

#include "common.h"
#include "log.h"
#include "router-scanner.h"

#include <nlohmann/json.hpp>

#include <climits>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#else
#    include <unistd.h>
#endif

using json = nlohmann::json;

static std::string detect_llama_server_binary() {
#if defined(_WIN32)
    std::vector<char> buffer(MAX_PATH);
    DWORD             len = 0;
    while (true) {
        len = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (len == 0) {
            return std::string();
        }
        if (len < buffer.size()) {
            break;
        }
        buffer.resize(buffer.size() * 2);
    }

    std::filesystem::path path(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(len));
    return (path.parent_path() / "llama-server.exe").string();
#elif defined(__linux__)
    std::vector<char> buffer(1024);
    while (true) {
        ssize_t len = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
        if (len < 0) {
            return std::string();
        }
        if (static_cast<size_t>(len) < buffer.size() - 1) {
            buffer[len] = '\0';
            break;
        }
        buffer.resize(buffer.size() * 2);
    }

    std::filesystem::path path(buffer.data());
    return (path.parent_path() / "llama-server").string();
#elif defined(__APPLE__)
    std::vector<char> buffer(PATH_MAX);
    uint32_t          size = static_cast<uint32_t>(buffer.size());
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        buffer.resize(size);
        size = static_cast<uint32_t>(buffer.size());
        if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
            return std::string();
        }
    }

    std::filesystem::path path(buffer.data());
    return (path.parent_path() / "llama-server").string();
#else
    return std::string();
#endif
}

static SpawnConfig parse_spawn_config(const json & data) {
    SpawnConfig spawn;
    if (data.contains("command")) {
        spawn.command = data["command"].get<std::vector<std::string>>();
    }
    if (data.contains("proxy_endpoints")) {
        spawn.proxy_endpoints = data["proxy_endpoints"].get<std::vector<std::string>>();
    }
    if (data.contains("health_endpoint")) {
        spawn.health_endpoint = data["health_endpoint"].get<std::string>();
    }

    return spawn;
}

static json serialize_spawn_config(const SpawnConfig & spawn) {
    json obj;
    obj["command"] = spawn.command;
    obj["proxy_endpoints"] = spawn.proxy_endpoints;
    obj["health_endpoint"] = spawn.health_endpoint;
    return obj;
}

const SpawnConfig & get_default_spawn() {
    static const SpawnConfig spawn = [] {
        SpawnConfig default_spawn = {
            /*command          =*/ {"llama-server", "--ctx-size", "4096", "--n-gpu-layers", "99"},
            /*proxy_endpoints =*/ {"/v1/", "/health", "/slots", "/props"},
            /*health_endpoint =*/ "/health",
        };

        std::error_code ec;
        const std::string detected_path = detect_llama_server_binary();
        if (!detected_path.empty() && std::filesystem::exists(detected_path, ec) && !ec) {
            LOG_INF("Detected llama-server at %s\n", detected_path.c_str());
            default_spawn.command[0] = detected_path;
        } else {
            LOG_INF("Falling back to llama-server resolved via PATH\n");
        }

        return default_spawn;
    }();

    return spawn;
}

const RouterOptions & get_default_router_options() {
    static const RouterOptions opts = {
        /*host      =*/ "127.0.0.1",
        /*port      =*/ 8082,
        /*base_port =*/ 50000,
        /*connection_timeout_s =*/ 5,
        /*read_timeout_s       =*/ 600,
        /*admin_token          =*/ "",
    };

    return opts;
}

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
    out["default_spawn"] = serialize_spawn_config(cfg.default_spawn);
    out["router"] = {{"host", cfg.router.host},
                     {"port", cfg.router.port},
                     {"base_port", cfg.router.base_port},
                     {"connection_timeout_s", cfg.router.connection_timeout_s},
                     {"read_timeout_s", cfg.router.read_timeout_s}};

    if (!cfg.router.admin_token.empty()) {
        out["router"]["admin_token"] = cfg.router.admin_token;
    }

    out["models"] = json::array();
    for (const auto & m : cfg.models) {
        json obj;
        obj["name"]  = m.name;
        obj["path"]  = m.path;
        obj["state"] = m.state.empty() ? "manual" : m.state;
        if (!m.group.empty()) {
            obj["group"] = m.group;
        }
        if (!is_spawn_empty(m.spawn)) {
            obj["spawn"] = serialize_spawn_config(m.spawn);
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
    cfg.default_spawn = get_default_spawn();
    cfg.router        = get_default_router_options();
    cfg.models        = scan_default_models();

    LOG_INF("Discovered %zu default models while generating config\n", cfg.models.size());
    write_config_file(cfg, path);
    LOG_INF("generated default config at %s\n", path.c_str());
    return cfg;
}

RescanResult rescan_auto_models(const RouterConfig & existing) {
    RescanResult result;
    result.config = existing;

    RouterConfig & merged = result.config;

    std::unordered_map<std::string, size_t> existing_paths;
    for (size_t i = 0; i < existing.models.size(); ++i) {
        existing_paths.emplace(expand_user_path(existing.models[i].path), i);
    }

    auto scanned = scan_default_models();
    std::unordered_set<std::string> scanned_paths;
    for (auto & scanned_model : scanned) {
        const auto expanded = expand_user_path(scanned_model.path);
        scanned_paths.insert(expanded);
        auto it = existing_paths.find(expanded);
        if (it != existing_paths.end()) {
            const auto & existing_model = existing.models[it->second];
            if (existing_model.state == "manual") {
                continue;
            }

            continue;
        }

        if (scanned_model.state.empty()) {
            scanned_model.state = "auto";
        }
        merged.models.push_back(std::move(scanned_model));
        existing_paths.emplace(expanded, merged.models.size() - 1);
        ++result.added;
    }

    std::vector<ModelConfig> filtered;
    filtered.reserve(merged.models.size());
    for (const auto & model : merged.models) {
        if (model.state == "manual") {
            filtered.push_back(model);
            continue;
        }

        const auto expanded = expand_user_path(model.path);
        const auto found    = scanned_paths.count(expanded) > 0;
        if (found) {
            filtered.push_back(model);
        } else {
            ++result.removed;
            LOG_INF("Removing auto model (no longer in cache): %s\n", model.name.c_str());
        }
    }
    merged.models = std::move(filtered);

    return result;
}

RouterConfig load_config(const std::string & path) {
    RouterConfig cfg;
    cfg.router        = get_default_router_options();
    cfg.default_spawn = get_default_spawn();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        LOG_WRN("Config file %s missing or inaccessible (ec=%d). Generating default.\n", path.c_str(), ec ? ec.value() : 0);
        return generate_default_config(path);
    }

    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("failed to open config file: " + path);
    }

    json data = json::parse(fin);
    LOG_INF("Loaded config file %s\n", path.c_str());
    if (data.contains("version")) {
        cfg.version = data["version"].get<std::string>();
    }
    if (data.contains("default_spawn")) {
        cfg.default_spawn = parse_spawn_config(data["default_spawn"]);
    }
    if (data.contains("router")) {
        auto r = data["router"];
        if (r.contains("host")) cfg.router.host = r["host"].get<std::string>();
        if (r.contains("port")) cfg.router.port = r["port"].get<int>();
        if (r.contains("base_port")) cfg.router.base_port = r["base_port"].get<int>();
        if (r.contains("connection_timeout_s")) cfg.router.connection_timeout_s = r["connection_timeout_s"].get<int>();
        if (r.contains("read_timeout_s")) cfg.router.read_timeout_s = r["read_timeout_s"].get<int>();
        if (r.contains("admin_token")) cfg.router.admin_token = r["admin_token"].get<std::string>();
    }
    if (data.contains("models")) {
        for (const auto & m : data["models"]) {
            ModelConfig mc;
            mc.name  = m.value("name", "");
            mc.path  = m.value("path", "");
            mc.state = m.value("state", "manual");
            mc.group = m.value("group", "");
            if (m.contains("spawn")) {
                mc.spawn = parse_spawn_config(m["spawn"]);
            }
            cfg.models.push_back(std::move(mc));
        }
    }
    LOG_INF("Config parsed: %zu models, router port %d, base port %d\n", cfg.models.size(), cfg.router.port, cfg.router.base_port);

    const auto rescan_result = rescan_auto_models(cfg);
    cfg                      = rescan_result.config;
    LOG_INF("Rescanned models, found %zu new auto models (removed %zu)\n", rescan_result.added, rescan_result.removed);

    const auto validate_port = [&](int port, const std::string & name) {
        if (port <= 0 || port > 65535) {
            throw std::runtime_error("invalid " + name + " port in config: " + std::to_string(port));
        }
    };

    validate_port(cfg.router.port, "router");
    validate_port(cfg.router.base_port, "base");
    for (const auto & model : cfg.models) {
        if (model.name.empty()) {
            throw std::runtime_error("model entry missing name");
        }

        const std::string path_to_check = expand_user_path(model.path);
        if (!std::filesystem::exists(path_to_check, ec)) {
            throw std::runtime_error("model path does not exist: " + path_to_check);
        }

        const SpawnConfig & spawn = is_spawn_empty(model.spawn) ? cfg.default_spawn : model.spawn;
        if (spawn.command.empty()) {
            throw std::runtime_error("spawn command missing for model: " + model.name);
        }
    }

    if (rescan_result.added > 0 || rescan_result.removed > 0) {
        LOG_INF("Persisting updated configuration after rescan (added %zu, removed %zu)\n", rescan_result.added, rescan_result.removed);
        write_config_file(cfg, path);
    }

    return cfg;
}

std::string get_model_group(const ModelConfig & cfg) {
    return cfg.group.empty() ? cfg.name : cfg.group;
}
