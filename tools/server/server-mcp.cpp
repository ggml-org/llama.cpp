#include "server-mcp.h"

#include <sheredom/subprocess.h>

#include <cstdio>
#include <cstring>
#include <fstream>

#if defined(_WIN32)
#  include <windows.h>
#else
extern char ** environ;
#endif

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
// server_mcp_stdio
//

struct server_mcp_stdio::process_handle {
    subprocess_s sp;
    FILE * in  = nullptr; // child stdin
    FILE * out = nullptr; // child stdout
    FILE * err = nullptr; // child stderr
};

static std::vector<std::string> mcp_parent_env() {
    std::vector<std::string> env;
#if defined(_WIN32)
    LPCH block = GetEnvironmentStringsA();
    if (block) {
        for (LPCH e = block; *e; e += strlen(e) + 1) {
            env.emplace_back(e);
        }
        FreeEnvironmentStringsA(block);
    }
#else
    if (environ) {
        for (char ** e = environ; *e; ++e) {
            env.emplace_back(*e);
        }
    }
#endif
    return env;
}

// parent env with the config overrides applied, in "KEY=VALUE" form
static std::vector<std::string> mcp_build_env(const std::map<std::string, std::string> & overrides) {
    std::vector<std::string> env;
    for (auto & e : mcp_parent_env()) {
        size_t eq = e.find('=');
        std::string key = eq == std::string::npos ? e : e.substr(0, eq);
        if (overrides.find(key) == overrides.end()) {
            env.push_back(e);
        }
    }
    for (auto & [k, v] : overrides) {
        env.push_back(k + "=" + v);
    }
    return env;
}

server_mcp_stdio::server_mcp_stdio(const server_mcp_server_config & config) : config(config) {
    name = config.name;
    timeout_ms = config.timeout_ms;
}

server_mcp_stdio::~server_mcp_stdio() {
    join_pumps();
}

bool server_mcp_stdio::start() {
    std::vector<std::string> argv_s;
    argv_s.push_back(config.command);
    argv_s.insert(argv_s.end(), config.args.begin(), config.args.end());

    int options = subprocess_option_no_window | subprocess_option_search_user_path;
    std::vector<std::string> envp_s;
    if (config.env.empty()) {
        options |= subprocess_option_inherit_environment;
    } else {
        envp_s = mcp_build_env(config.env);
    }

    auto to_ptrs = [](std::vector<std::string> & v) {
        std::vector<const char *> p;
        p.reserve(v.size() + 1);
        for (auto & s : v) {
            p.push_back(s.c_str());
        }
        p.push_back(nullptr);
        return p;
    };
    auto argv = to_ptrs(argv_s);
    auto envp = to_ptrs(envp_s);

    auto handle = std::make_unique<process_handle>();
    int rc = subprocess_create_ex(argv.data(), options,
                                  config.env.empty() ? nullptr : envp.data(),
                                  config.cwd.empty() ? nullptr : config.cwd.c_str(),
                                  &handle->sp);
    if (rc != 0) {
        SRV_WRN("MCP '%s': failed to spawn '%s'\n", config.name.c_str(), config.command.c_str());
        return false;
    }
    handle->in  = subprocess_stdin(&handle->sp);
    handle->out = subprocess_stdout(&handle->sp);
    handle->err = subprocess_stderr(&handle->sp);

    proc = std::move(handle);
    running.store(true);
    reader = std::thread([this] { reader_loop(); });
    writer = std::thread([this] { writer_loop(); });
    errlog = std::thread([this] { errlog_loop(); });
    return true;
}

void server_mcp_stdio::close() {
    join_pumps();
}

bool server_mcp_stdio::is_alive() const {
    return running.load();
}

void server_mcp_stdio::reader_loop() {
    std::string buf;
    char chunk[4096];
    for (;;) {
        size_t n = fread(chunk, 1, sizeof(chunk), proc->out);
        if (n == 0) {
            break; // EOF or error
        }
        buf.append(chunk, n);

        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            json msg;
            try {
                msg = json::parse(line);
            } catch (...) {
                continue; // skip malformed line
            }
            if (!from_server.write(std::move(msg))) {
                return; // consumer gone
            }
        }
    }
    running.store(false);
    to_server.close_write();   // stop the writer
    from_server.close_write(); // EOF to any waiting caller
}

void server_mcp_stdio::writer_loop() {
    auto should_stop = [this] { return !running.load(); };
    json msg;
    while (to_server.read(msg, should_stop)) {
        std::string data = msg.dump();
        data.push_back('\n');
        if (fwrite(data.data(), 1, data.size(), proc->in) != data.size() || fflush(proc->in) != 0) {
            break; // child gone
        }
    }
    running.store(false);
    from_server.close_write();
}

void server_mcp_stdio::errlog_loop() {
    char chunk[4096];
    for (;;) {
        size_t n = fread(chunk, 1, sizeof(chunk), proc->err);
        if (n == 0) {
            break;
        }
        SRV_DBG("MCP '%s' stderr: %.*s", name.c_str(), (int) n, chunk);
    }
}

void server_mcp_stdio::join_pumps() {
    if (!proc) {
        return;
    }
    running.store(false);
    to_server.close_write();   // wake the writer if it waits for a message
    from_server.close_write(); // wake any caller waiting for a reply

    subprocess_terminate(&proc->sp); // child death unblocks the blocked fread/fwrite

    if (writer.joinable()) writer.join();
    if (reader.joinable()) reader.join();
    if (errlog.joinable()) errlog.join();

    subprocess_destroy(&proc->sp); // safe now: no thread touches the FILE* anymore
    proc.reset();
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

