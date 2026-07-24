#include "server-mcp.h"

#include <fstream>

//
// server_mcp_server_config
//

std::vector<server_mcp_server_config> server_mcp_server_config::parse_from_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open MCP config file: " + path);
    }
    json j;
    f >> j;
    return parse_cursor_format(j);
}

std::vector<server_mcp_server_config> server_mcp_server_config::parse_from_json(const std::string & json_str) {
    return parse_cursor_format(json::parse(json_str));
}

std::vector<server_mcp_server_config> server_mcp_server_config::parse_cursor_format(const json & j) {
    std::vector<server_mcp_server_config> result;

    if (!j.contains("mcpServers") || !j.at("mcpServers").is_object()) {
        return result;
    }

    for (const auto & [name, cfg] : j.at("mcpServers").items()) {
        server_mcp_server_config sc;
        sc.name = name;
        sc.command = cfg.value("command", std::string());
        sc.cwd = cfg.value("cwd", std::string());
        sc.timeout_ms = cfg.value("timeout_ms", sc.timeout_ms);

        if (cfg.contains("args") && cfg.at("args").is_array()) {
            for (const auto & a : cfg.at("args")) {
                sc.args.push_back(a.get<std::string>());
            }
        }
        if (cfg.contains("env") && cfg.at("env").is_object()) {
            for (const auto & [k, v] : cfg.at("env").items()) {
                sc.env[k] = v.get<std::string>();
            }
        }

        if (sc.command.empty()) {
            SRV_WRN("MCP server '%s' has no command, skipping\n", name.c_str());
            continue;
        }
        result.push_back(std::move(sc));
    }

    return result;
}

//
// server_mcp
//

static constexpr int MCP_COOLDOWN_SECONDS = 5;

server_mcp::server_mcp(std::vector<server_mcp_server_config> configs)
    : configs(std::move(configs)) {}

server_mcp::~server_mcp() {
    shutdown();

    std::vector<std::shared_ptr<server_mcp_transport>> to_close;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & [name, t] : transports) {
            to_close.push_back(std::move(t));
        }
        transports.clear();
    }
    for (auto & t : to_close) {
        t->close();
    }
}

std::shared_ptr<server_mcp_transport> server_mcp::create_transport(const server_mcp_server_config & cfg) {
    return std::make_shared<server_mcp_stdio>(cfg);
}

void server_mcp::shutdown() {
    stopping.store(true);
}

const server_mcp_server_config * server_mcp::find_config(const std::string & name) const {
    for (const auto & c : configs) {
        if (c.name == name) {
            return &c;
        }
    }
    return nullptr;
}

void server_mcp::start() {
    auto should_stop = [this]() { return stopping.load(); };

    std::vector<server_mcp_tool_def> discovered;
    for (const auto & cfg : configs) {
        auto t = create_transport(cfg);
        if (!t->start()) {
            SRV_WRN("MCP warmup: failed to spawn '%s'\n", cfg.name.c_str());
            continue;
        }
        auto tools = t->list_tools(should_stop);
        SRV_INF("MCP warmup: '%s' discovered %zu tools\n", cfg.name.c_str(), tools.size());
        discovered.insert(discovered.end(), tools.begin(), tools.end());
        t->close();
    }

    std::lock_guard<std::mutex> lock(mutex);
    registry.swap(discovered);
}

std::vector<server_mcp_tool_def> server_mcp::list_tools() const {
    std::lock_guard<std::mutex> lock(mutex);
    return registry;
}

json server_mcp::call_tool(const std::string & server_name,
                           const std::string & tool_name,
                           const json & arguments,
                           const std::function<bool()> & should_stop) {
    auto transport = get_or_create(server_name);
    if (!transport) {
        return {{"error", "MCP server unavailable: " + server_name}};
    }

    auto stop = [this, &should_stop]() {
        return stopping.load() || (should_stop && should_stop());
    };
    return transport->call_tool(tool_name, arguments, stop);
}

std::shared_ptr<server_mcp_transport> server_mcp::get_or_create(const std::string & name) {
    std::vector<std::shared_ptr<server_mcp_transport>> to_close; // closed after unlock
    std::shared_ptr<server_mcp_transport> result;

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping.load()) {
            return nullptr;
        }

        auto now = std::chrono::steady_clock::now();
        auto dead_it = dead_servers.find(name);
        if (dead_it != dead_servers.end()) {
            if (now < dead_it->second) {
                return nullptr;
            }
            dead_servers.erase(dead_it);
        }

        auto it = transports.find(name);
        if (it != transports.end()) {
            if (it->second->is_alive()) {
                return it->second;
            }
            to_close.push_back(std::move(it->second));
            transports.erase(it);
        }

        const server_mcp_server_config * cfg = find_config(name);
        if (cfg) {
            auto fresh = create_transport(*cfg);
            if (fresh->start() && fresh->is_alive()) {
                transports[name] = fresh;
                result = fresh;
            } else {
                to_close.push_back(std::move(fresh));
                dead_servers[name] = now + std::chrono::seconds(MCP_COOLDOWN_SECONDS);
            }
        }
    }

    for (auto & t : to_close) {
        t->close(); // blocking call, no leaks
    }

    return result;
}

