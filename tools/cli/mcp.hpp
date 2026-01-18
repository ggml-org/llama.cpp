#pragma once

#include "../../vendor/sheredom/subprocess.h"
#include "common.h"
#include "log.h"

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

struct mcp_tool {
    std::string name;
    std::string description;
    json        input_schema;
    std::string server_name;
};

class mcp_server {
    std::string                        name;
    std::string                        command;
    std::vector<std::string>           args;
    std::map<std::string, std::string> env;

    struct subprocess_s process = {};
    std::atomic<bool>   running{ false };
    std::thread         read_thread;
    std::thread         err_thread;
    std::atomic<bool>   stop_read{ false };

    std::mutex                        mutex;
    int                               next_id = 1;
    std::map<int, std::promise<json>> pending_requests;

    // Buffer for reading
    std::string read_buffer;

  public:
    mcp_server(const std::string &                        name,
               const std::string &                        cmd,
               const std::vector<std::string> &           args,
               const std::map<std::string, std::string> & env) :
        name(name),
        command(cmd),
        args(args),
        env(env) {}

    mcp_server(const mcp_server &)            = delete;
    mcp_server & operator=(const mcp_server &) = delete;

    mcp_server(mcp_server && other) noexcept :
        name(std::move(other.name)),
        command(std::move(other.command)),
        args(std::move(other.args)),
        env(std::move(other.env)),
        process(other.process),
        running(other.running.load()),
        read_thread(std::move(other.read_thread)),
        err_thread(std::move(other.err_thread)),
        stop_read(other.stop_read.load()),
        next_id(other.next_id),
        pending_requests(std::move(other.pending_requests)),
        read_buffer(std::move(other.read_buffer)) {
        // Zero out the source process to prevent double-free
        other.process = {};
        other.running = false;
    }

    mcp_server & operator=(mcp_server && other) noexcept {
        if (this != &other) {
            stop(); // Clean up current resources

            name             = std::move(other.name);
            command          = std::move(other.command);
            args             = std::move(other.args);
            env              = std::move(other.env);
            process          = other.process;
            running          = other.running.load();
            read_thread      = std::move(other.read_thread);
            err_thread       = std::move(other.err_thread);
            stop_read        = other.stop_read.load();
            next_id          = other.next_id;
            pending_requests = std::move(other.pending_requests);
            read_buffer      = std::move(other.read_buffer);

            // Zero out source
            other.process = {};
            other.running = false;
        }
        return *this;
    }

    ~mcp_server() { stop(); }

    bool start() {
        std::vector<const char *> cmd_args;
        cmd_args.push_back(command.c_str());
        for (const auto & arg : args) {
            cmd_args.push_back(arg.c_str());
        }
        cmd_args.push_back(nullptr);

        std::vector<const char *> env_vars;
        std::vector<std::string>  env_strings;  // keep strings alive
        if (!env.empty()) {
            for (const auto & kv : env) {
                env_strings.push_back(kv.first + "=" + kv.second);
            }
            for (const auto & s : env_strings) {
                env_vars.push_back(s.c_str());
            }
            env_vars.push_back(nullptr);
        }

        // Blocking I/O is simpler with threads
        int options = subprocess_option_search_user_path;
        int result;
        if (env.empty()) {
            options |= subprocess_option_inherit_environment;
            result = subprocess_create(cmd_args.data(), options, &process);
        } else {
            result = subprocess_create_ex(cmd_args.data(), options, env_vars.data(), &process);
        }

        if (result != 0) {
            LOG_ERR("Failed to start MCP server %s: error %d (%s)\n", name.c_str(), errno, strerror(errno));
            return false;
        }

        running     = true;
        read_thread = std::thread(&mcp_server::read_loop, this);
        err_thread  = std::thread(&mcp_server::err_loop, this);

        return true;
    }

    void stop() {
        if (!running) return;
        LOG_INF("Stopping MCP server %s...\n", name.c_str());
        stop_read = true;

        // 1. Close stdin to signal EOF
        if (process.stdin_file) {
            fclose(process.stdin_file);
            process.stdin_file = nullptr;
        }

        // 2. Wait for 10 seconds for normal termination
        bool terminated = false;
        for (int i = 0; i < 100; ++i) { // 100 * 100ms = 10s
             if (subprocess_alive(&process) == 0) {
                 terminated = true;
                 break;
             }
             std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 3. Terminate if still running
        if (!terminated) {
            LOG_WRN("MCP server %s did not exit gracefully, terminating...\n", name.c_str());
            subprocess_terminate(&process);
        }

        // 4. Join threads
        if (read_thread.joinable()) {
            read_thread.join();
        }
        if (err_thread.joinable()) {
            err_thread.join();
        }
        
        // 5. Cleanup
        if (running) {
             subprocess_destroy(&process);
             running = false;
        }
        LOG_INF("MCP server %s stopped.\n", name.c_str());
    }

    json send_request(const std::string & method, const json & params = json::object()) {
        int               id;
        std::future<json> future;
        {
            std::lock_guard<std::mutex> lock(mutex);
            id     = next_id++;
            future = pending_requests[id].get_future();
        }

        json req = {
            { "jsonrpc", "2.0"  },
            { "id",      id     },
            { "method",  method },
            { "params",  params }
        };

        std::string req_str    = req.dump() + "\n";
        FILE *      stdin_file = subprocess_stdin(&process);
        if (stdin_file) {
            fwrite(req_str.c_str(), 1, req_str.size(), stdin_file);
            fflush(stdin_file);
        }

        // Wait for response with timeout
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::timeout) {
            LOG_ERR("Timeout waiting for response from %s (method: %s)\n", name.c_str(), method.c_str());
            std::lock_guard<std::mutex> lock(mutex);
            pending_requests.erase(id);
            return nullptr;
        }

        return future.get();
    }

    void send_notification(const std::string & method, const json & params = json::object()) {
        json req = {
            { "jsonrpc", "2.0"  },
            { "method",  method },
            { "params",  params }
        };

        std::string req_str    = req.dump() + "\n";
        FILE *      stdin_file = subprocess_stdin(&process);
        if (stdin_file) {
            fwrite(req_str.c_str(), 1, req_str.size(), stdin_file);
            fflush(stdin_file);
        }
    }

    // Initialize handshake
    bool initialize() {
        // Send initialize
        json init_params = {
            { "protocolVersion", "2024-11-05" },
            { "capabilities", {
                { "roots", {
                    { "listChanged", false }
                } },
                { "sampling", json::object() }
            } },
            { "clientInfo", {
                { "name", "llama.cpp-cli" },
                { "version", "0.1.0" }  // TODO: use real version
            } }
        };

        json res = send_request("initialize", init_params);
        if (res.is_null() || res.contains("error")) {
            LOG_ERR("Failed to initialize MCP server %s\n", name.c_str());
            return false;
        }

        // Send initialized notification
        send_notification("notifications/initialized");
        return true;
    }

    std::vector<mcp_tool> list_tools() {
        std::vector<mcp_tool> tools;
        json                  res = send_request("tools/list");
        if (res.is_null() || res.contains("error")) {
            LOG_ERR("Failed to list tools from %s\n", name.c_str());
            return tools;
        }

        if (res.contains("result") && res["result"].contains("tools")) {
            for (const auto & t : res["result"]["tools"]) {
                mcp_tool tool;
                tool.name = t["name"].get<std::string>();
                if (t.contains("description")) {
                    tool.description = t["description"].get<std::string>();
                }
                if (t.contains("inputSchema")) {
                    tool.input_schema = t["inputSchema"];
                }
                tool.server_name = name;
                tools.push_back(tool);
            }
        }
        return tools;
    }

    json call_tool(const std::string & tool_name, const json & args) {
        json params = {
            { "name",      tool_name },
            { "arguments", args      }
        };
        json res = send_request("tools/call", params);
        if (res.is_null() || res.contains("error")) {
            return {
                { "error", res.contains("error") ? res["error"] : "Unknown error" }
            };
        }
        if (res.contains("result")) {
            return res["result"];
        }
        return {
            { "error", "No result returned" }
        };
    }

  private:
    void read_loop() {
        char buffer[4096];
        while (!stop_read && running) {
            unsigned bytes_read = subprocess_read_stdout(&process, buffer, sizeof(buffer));

            if (bytes_read == 0) {
                // If blocking read returns 0, it means EOF (process exited or pipe closed).
                // We should NOT call subprocess_alive() here because it calls subprocess_join()
                // which modifies the process struct (closes stdin) and causes race conditions/double-free
                // when stop() is called concurrently.
                // Just break the loop. The process is likely dead or dying.
                if (!stop_read) {
                    LOG_ERR("MCP process died (stdout closed)\n");
                }
                running = false;
                break;
            }

            read_buffer.append(buffer, bytes_read);
            size_t pos;
            while ((pos = read_buffer.find('\n')) != std::string::npos) {
                std::string line = read_buffer.substr(0, pos);
                read_buffer.erase(0, pos + 1);

                if (line.empty()) {
                    continue;
                }

                try {
                    json msg = json::parse(line);
                    if (msg.contains("id")) {
                        // Response
                        int id = msg["id"].get<int>();
                        std::lock_guard<std::mutex> lock(mutex);
                        if (pending_requests.count(id)) {
                            pending_requests[id].set_value(msg);
                            pending_requests.erase(id);
                        } else {
                            // ID not found
                        }
                    } else {
                        // Notification or request from server -> ignore for now or log
                        // MCP servers might send notifications (e.g. logging)
                        LOG_ERR("MCP Notification from %s: %s\n", name.c_str(), line.c_str());
                    }
                } catch (const std::exception & e) {
                    // Not a full JSON yet? Or invalid?
                    // If it was a line, it should be valid JSON-RPC
                    LOG_WRN("Failed to parse JSON from %s: %s\n", name.c_str(), e.what());
                }
            }
        }
    }

    void err_loop() {
        char buffer[1024];
        while (!stop_read && running) {
            unsigned bytes_read = subprocess_read_stderr(&process, buffer, sizeof(buffer));
            if (bytes_read > 0) {
                if (stop_read) break; // Don't log stderr during shutdown
                std::string err_str(buffer, bytes_read);
                // Filter out empty/whitespace-only stderr if desired, or just keep it.
                // User said "extra logging that passes the stderr here is unnecessary" referring to shutdown.
                LOG_WRN("[%s stderr] %s", name.c_str(), err_str.c_str());
            } else {
                // EOF
                break;
            }
        }
    }
};

class mcp_context {
    std::map<std::string, std::shared_ptr<mcp_server>> servers;
    std::vector<mcp_tool>                              tools;
    bool                                               yolo = false;

  public:
    void set_yolo(bool y) { yolo = y; }

    bool load_config(const std::string & config_path, const std::string & enabled_servers_str) {
        std::ifstream f(config_path);
        if (!f) {
            return false;
        }

        json config;
        try {
            f >> config;
        } catch (...) {
            return false;
        }

        std::vector<std::string> enabled_list;
        std::stringstream        ss(enabled_servers_str);
        std::string              item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                enabled_list.push_back(item);
            }
        }

        if (config.contains("mcpServers")) {
            std::string server_list;
            for (auto & [key, val] : config["mcpServers"].items()) {
                if (!server_list.empty()) server_list += ", ";
                server_list += key;
            }
            LOG_INF("MCP configuration found with servers: %s\n", server_list.c_str());

            for (auto & [key, val] : config["mcpServers"].items()) {
                // If enabled_servers_str is empty, enable all? User said "possibility to pick which MCP servers to enable".
                // If the user specifies explicit list, we filter. If not, maybe we shouldn't enable any or enable all?
                // The prompt says "possibility to pick".
                // Let's assume if list provided, use it. If not, enable all? Or none?
                // User provided example implies explicit enabling might be desired.
                // Let's assume if `enabled_servers_str` is not empty, we filter.

                bool enabled = true;
                if (!enabled_list.empty()) {
                    bool found = false;
                    for (const auto & s : enabled_list) {
                        if (s == key) {
                            found = true;
                        }
                    }
                    if (!found) {
                        enabled = false;
                    }
                }

                if (enabled) {
                    std::string                        cmd  = val["command"].get<std::string>();
                    std::vector<std::string>           args = val.value("args", std::vector<std::string>{});
                    std::map<std::string, std::string> env;
                    if (val.contains("env")) {
                        for (auto & [ek, ev] : val["env"].items()) {
                            env[ek] = ev.get<std::string>();
                        }
                    }

                    auto server = std::make_shared<mcp_server>(key, cmd, args, env);
                    LOG_INF("Trying to start MCP server: %s...\n", key.c_str());
                    if (server->start()) {
                        if (server->initialize()) {
                            servers[key] = server;
                            LOG_INF("MCP Server '%s' started and initialized.\n", key.c_str());

                            auto server_tools = server->list_tools();
                            tools.insert(tools.end(), server_tools.begin(), server_tools.end());
                        } else {
                            LOG_ERR("MCP Server '%s' failed to initialize.\n", key.c_str());
                        }
                    }
                }
            }
        }
        return true;
    }

    std::vector<mcp_tool> get_tools() const { return tools; }

    bool get_yolo() const { return yolo; }

    json call_tool(const std::string & tool_name, const json & args) {
        // Find which server has this tool
        std::string server_name;
        for (const auto & t : tools) {
            if (t.name == tool_name) {
                server_name = t.server_name;
                break;
            }
        }

        if (server_name.empty()) {
            return {
                { "error", "Tool not found" }
            };
        }

        if (servers.count(server_name)) {
            return servers[server_name]->call_tool(tool_name, args);
        }
        return {
            { "error", "Server not found" }
        };
    }
};
