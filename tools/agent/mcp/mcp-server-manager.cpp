#include "mcp-server-manager.h"

#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <regex>

namespace fs = std::filesystem;

mcp_server_manager::~mcp_server_manager() {
    shutdown_all();
}

bool mcp_server_manager::load_config(const std::string & config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        last_error_ = "Cannot open config file: " + config_path;
        return false;
    }

    json config;
    try {
        file >> config;
    } catch (const json::parse_error & e) {
        last_error_ = "JSON parse error: " + std::string(e.what());
        return false;
    }

    if (!config.contains("servers") || !config["servers"].is_object()) {
        last_error_ = "Config must contain 'servers' object";
        return false;
    }

    configs_.clear();

    for (auto & [name, server_json] : config["servers"].items()) {
        mcp_server_config cfg;
        cfg.name = name;

        // Required: command
        if (!server_json.contains("command") || !server_json["command"].is_string()) {
            last_error_ = "Server '" + name + "' missing 'command' string";
            return false;
        }
        cfg.command = expand_env_vars(server_json["command"].get<std::string>());

        // Optional: args (array of strings)
        if (server_json.contains("args") && server_json["args"].is_array()) {
            for (const auto & arg : server_json["args"]) {
                if (arg.is_string()) {
                    cfg.args.push_back(expand_env_vars(arg.get<std::string>()));
                }
            }
        }

        // Optional: env (object of string -> string)
        if (server_json.contains("env") && server_json["env"].is_object()) {
            for (auto & [key, value] : server_json["env"].items()) {
                if (value.is_string()) {
                    cfg.env[key] = expand_env_vars(value.get<std::string>());
                }
            }
        }

        // Optional: enabled (default true)
        cfg.enabled = server_json.value("enabled", true);

        // Optional: timeout (default 60000ms)
        cfg.timeout_ms = server_json.value("timeout", 60000);

        configs_[name] = cfg;
    }

    return true;
}

int mcp_server_manager::start_servers() {
    int started = 0;

    for (auto & [name, cfg] : configs_) {
        if (!cfg.enabled) {
            continue;
        }

        auto client = std::make_unique<mcp_client>();
        if (client->connect(cfg.command, cfg.args, cfg.env, 30000)) {
            clients_[name] = std::move(client);
            started++;
        } else {
            last_error_ = "Failed to start server '" + name + "': " + client->last_error();
        }
    }

    return started;
}

void mcp_server_manager::shutdown_all() {
    for (auto & [name, client] : clients_) {
        if (client) {
            client->shutdown();
        }
    }
    clients_.clear();
}

std::vector<std::pair<std::string, mcp_tool>> mcp_server_manager::list_all_tools() {
    std::vector<std::pair<std::string, mcp_tool>> result;

    for (auto & [server_name, client] : clients_) {
        if (!client || !client->is_connected()) {
            continue;
        }

        auto tools = client->list_tools();
        for (auto & tool : tools) {
            std::string qualified = qualify_name(server_name, tool.name);
            result.emplace_back(qualified, tool);
        }
    }

    return result;
}

mcp_call_result mcp_server_manager::call_tool(const std::string & qualified_name,
                                               const json & arguments) {
    mcp_call_result result;
    result.is_error = true;

    std::string server, tool;
    if (!parse_qualified_name(qualified_name, server, tool)) {
        last_error_ = "Invalid tool name format: " + qualified_name;
        result.content.push_back({{"type", "text"}, {"text", last_error_}});
        return result;
    }

    auto it = clients_.find(server);
    if (it == clients_.end() || !it->second) {
        last_error_ = "Server not found: " + server;
        result.content.push_back({{"type", "text"}, {"text", last_error_}});
        return result;
    }

    if (!it->second->is_connected()) {
        last_error_ = "Server not connected: " + server;
        result.content.push_back({{"type", "text"}, {"text", last_error_}});
        return result;
    }

    // Get timeout from config
    int timeout = 60000;
    auto cfg_it = configs_.find(server);
    if (cfg_it != configs_.end()) {
        timeout = cfg_it->second.timeout_ms;
    }

    return it->second->call_tool(tool, arguments, timeout);
}

std::vector<std::string> mcp_server_manager::server_names() const {
    std::vector<std::string> names;
    for (const auto & [name, _] : configs_) {
        names.push_back(name);
    }
    return names;
}

bool mcp_server_manager::is_server_connected(const std::string & name) const {
    auto it = clients_.find(name);
    return it != clients_.end() && it->second && it->second->is_connected();
}

std::string mcp_server_manager::qualify_name(const std::string & server, const std::string & tool) {
    // Replace any __ in server/tool names with _ to avoid confusion
    std::string safe_server = server;
    std::string safe_tool = tool;

    // Simple replacement (not using regex for performance)
    size_t pos;
    while ((pos = safe_server.find("__")) != std::string::npos) {
        safe_server.replace(pos, 2, "_");
    }
    while ((pos = safe_tool.find("__")) != std::string::npos) {
        safe_tool.replace(pos, 2, "_");
    }

    return "mcp__" + safe_server + "__" + safe_tool;
}

bool mcp_server_manager::parse_qualified_name(const std::string & qualified,
                                               std::string & server,
                                               std::string & tool) {
    // Format: mcp__<server>__<tool>
    const std::string prefix = "mcp__";
    if (qualified.substr(0, prefix.size()) != prefix) {
        return false;
    }

    std::string rest = qualified.substr(prefix.size());
    size_t sep = rest.find("__");
    if (sep == std::string::npos) {
        return false;
    }

    server = rest.substr(0, sep);
    tool = rest.substr(sep + 2);

    return !server.empty() && !tool.empty();
}

std::string mcp_server_manager::expand_env_vars(const std::string & value) {
    std::string result = value;

    // Match ${VAR_NAME} pattern
    std::regex env_regex(R"(\$\{([^}]+)\})");
    std::smatch match;

    while (std::regex_search(result, match, env_regex)) {
        std::string var_name = match[1].str();
        const char * var_value = std::getenv(var_name.c_str());
        std::string replacement = var_value ? var_value : "";

        result = result.substr(0, match.position()) +
                 replacement +
                 result.substr(match.position() + match.length());
    }

    return result;
}

std::string find_mcp_config(const std::string & working_dir) {
    // First check working directory
    fs::path local_config = fs::path(working_dir) / "mcp.json";
    if (fs::exists(local_config)) {
        return local_config.string();
    }

    // Then check user config directory
    const char * home = std::getenv("HOME");
    if (home) {
        fs::path user_config = fs::path(home) / ".config" / "llama-agent" / "mcp.json";
        if (fs::exists(user_config)) {
            return user_config.string();
        }
    }

    return "";
}
