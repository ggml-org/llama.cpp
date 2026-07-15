#pragma once

#include "server-common.h"
#include "server-http.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Cursor-compatible MCP server configuration
struct server_mcp_server_config {
    std::string name;                              // key from JSON (e.g. "filesystem")
    std::string command;                           // executable path
    std::vector<std::string> args;                 // arguments
    std::map<std::string, std::string> env;        // environment overrides
    std::string cwd;                               // working directory
    int timeout_ms = 30000;                        // per-tool call timeout

    static std::vector<server_mcp_server_config> parse_from_file(const std::string & path);
    static std::vector<server_mcp_server_config> parse_from_json(const std::string & json_str);
    static std::vector<server_mcp_server_config> parse_cursor_format(const json & j);
};

// Tool definition cached from an MCP server
struct server_mcp_tool_definition {
    std::string name;              // tool name (without server prefix)
    std::string description;
    json input_schema;             // JSON Schema for input parameters
    std::string server_name;       // which MCP server this tool belongs to
};

// Wraps a running MCP server process and manages JSON-RPC over stdio
struct server_mcp_instance {
    std::string server_name;
    std::vector<server_mcp_tool_definition> tools;  // cached from tools/list
    bool initialized = false;
    std::string error;
    uint64_t next_id = 1;
    int timeout_ms = 30000;
    mutable std::recursive_mutex mutex;
    std::string read_buffer; // persistent read buffer for NDJSON framing
    std::atomic<bool> terminating{false};

    // shared with the owning manager; lets an in-flight send_rpc() bail out as soon as
    // shutdown begins. shared_ptr (not a raw pointer) so it stays valid even if the
    // instance outlives the manager via a caller-held shared_ptr.
    std::shared_ptr<std::atomic<bool>> mgr_shutting_down;

    ~server_mcp_instance();

    bool spawn(const server_mcp_server_config & config);
    void terminate();                               // terminate + destroy
    bool is_alive() const;
    std::vector<server_mcp_tool_definition> list_tools();
    json call_tool(const std::string & tool_name, const json & arguments, int timeout_ms);

    // Platform-specific process state; defined in server-mcp.cpp so that platform
    // headers (windows.h) are not pulled into every TU that includes this header
    struct process_handle;
    std::unique_ptr<process_handle> proc;

    json send_rpc(const json & request, int timeout_ms);
    bool read_message(json & out);
    bool write_message(const json & msg);
};

// Owns all configured MCP servers, tracks instances globally (keyed by server name), performs warmup, and handles global cleanup
class server_mcp_manager {
public:
    server_mcp_manager(std::vector<server_mcp_server_config> configs);
    ~server_mcp_manager();

    void warmup();                                  // spawn → list → shutdown each server
    std::vector<server_mcp_tool_definition> get_all_tools() const;

    std::shared_ptr<server_mcp_instance> get_or_create(const std::string & server_name);
    void close_all();

    // Signals shutdown so in-flight send_rpc() calls bail out promptly.
    //
    // ASYNC-SIGNAL-SAFE: performs a single atomic store and takes no locks, because it is
    // called from signal_handler(). It must NOT take the manager mutex (a signal delivered
    // to a thread already inside get_or_create() would self-deadlock) nor the instance mutex
    // (which is exactly what the in-flight send_rpc() holds).
    //
    // Must be called before the HTTP server drains: close_all() runs in clean_up(), which
    // cannot execute until in-flight /tools handlers return, so setting the flags there is
    // too late to unblock them.
    void begin_shutdown();

private:
    std::vector<server_mcp_server_config> configs;
    mutable std::mutex mutex;

    // server_name → global instance for that server
    std::map<std::string, std::shared_ptr<server_mcp_instance>> global_instances;

    // Warmup-cached tool lists (from servers that were successfully warmed up)
    std::vector<server_mcp_tool_definition> warmup_tools;

    // Cooldown map for servers that failed to spawn
    std::map<std::string, std::chrono::steady_clock::time_point> dead_servers;

    // shared with every spawned instance so begin_shutdown() can signal them all
    // with one lock-free store, without iterating global_instances
    std::shared_ptr<std::atomic<bool>> shutting_down = std::make_shared<std::atomic<bool>>(false);

    std::shared_ptr<server_mcp_instance> do_spawn(const std::string & server_name);
};
