#pragma once

// Experimental MCP (Model Context Protocol) client support for llama-server.
// See MCP_SERVER_SPEC.md for the behavioral contract.
//
//   server_mcp           - manager, owns the configured servers and their tools
//   server_mcp_transport - one MCP server session (abstract), JSON-RPC 2.0
//   server_mcp_stdio     - child process speaking NDJSON JSON-RPC over stdio
//
// Blocking I/O lives on the transport's worker threads; callers touch only the in-process queues.
// Shutdown is RAII and blocking (~server_mcp joins everything), so no async-signal-safe path is needed.

#include "server-common.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

//
// Configuration (Cursor-compatible "mcpServers" JSON)
//

struct server_mcp_server_config {
    std::string name; // config key, e.g. "filesystem"
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> env; // merged over the parent env
    std::string cwd;
    int timeout_ms = 30000; // per-tool-call timeout

    // from_file/from_json throw on I/O or parse errors; a missing "mcpServers" yields an empty list, and entries without a "command" are skipped
    static std::vector<server_mcp_server_config> parse_from_file(const std::string & path);
    static std::vector<server_mcp_server_config> parse_from_json(const std::string & json_str);
    static std::vector<server_mcp_server_config> parse_cursor_format(const json & j);
};

// a tool advertised by an MCP server
struct server_mcp_tool_def {
    std::string server_name;
    std::string name; // bare tool name, no "<server>_" prefix
    std::string description;
    json input_schema; // JSON Schema for the arguments, or null
};

//
// server_mcp_transport: one MCP server session over a message conduit.
//
//   caller --send_rpc--> to_server   --[writer thread]--> server stdin
//   caller <--send_rpc-- from_server <--[reader thread]-- server stdout
//
// The base runs the JSON-RPC session; a subclass only pumps I/O into/out of the two queues.
//

struct server_mcp_transport {
    std::string name;
    int timeout_ms = 30000;

    server_pipe<json> to_server;
    server_pipe<json> from_server;

    virtual ~server_mcp_transport() = default;

    virtual bool start() = 0;
    virtual void close() = 0; // blocking and idempotent
    virtual bool is_alive() const = 0; // never blocks behind an in-flight send_rpc()

    bool handshake(const std::function<bool()> & should_stop);

    std::vector<server_mcp_tool_def> list_tools(const std::function<bool()> & should_stop);

    json call_tool(const std::string & tool_name,
                   const json & arguments,
                   const std::function<bool()> & should_stop);

protected:
    // per-transport, not shared: send_rpc() holds it across the wait for a reply, so a shared lock would stall every server behind one slow call
    // all members below are touched only under it
    std::mutex rpc_mutex;
    uint64_t next_id = 1; // reset to 1 per (re)spawn
    bool initialized = false;
    std::string last_error;
    std::vector<server_mcp_tool_def> tools;

    // returns the matching reply, or an {"error": ...} object
    json send_rpc(const json & request, const std::function<bool()> & should_stop);
};

//
// server_mcp_stdio: child process, NDJSON JSON-RPC over stdio (stderr inherited)
//

struct server_mcp_stdio : server_mcp_transport {
    explicit server_mcp_stdio(const server_mcp_server_config & config);
    ~server_mcp_stdio() override;

    bool start() override;
    void close() override;
    bool is_alive() const override;

private:
    server_mcp_server_config config;

    // defined in the .cpp so <windows.h> stays out of this header
    struct process_handle;
    std::unique_ptr<process_handle> proc;

    std::thread reader; // child stdout -> NDJSON framing -> from_server
    std::thread writer; // to_server -> child stdin
    std::thread errlog; // child stderr -> debug log (must be drained or the child blocks)

    // cleared by close() or by the reader on stdout EOF; read without rpc_mutex
    std::atomic<bool> running{false};

    void reader_loop();
    void writer_loop();
    void errlog_loop();
    void join_pumps();
};

//
// server_mcp
// manager lives inside main_server(); declare it before the HTTP context so it outlives every /tools handler.
//

class server_mcp {
public:
    explicit server_mcp(std::vector<server_mcp_server_config> configs);
    ~server_mcp();

    // spawn each server once, list its tools, shut it down. failures are logged, not fatal.
    void start();

    std::vector<server_mcp_tool_def> list_tools() const;

    // lazily (re)spawns the transport. returns the MCP result or an {"error": ...} object. should_stop is OR-ed with the manager's cancel flag.
    json call_tool(const std::string & server_name,
                   const std::string & tool_name,
                   const json & arguments,
                   const std::function<bool()> & should_stop = nullptr);

    // flip the cancel flag so in-flight calls return; blocking teardown is in the destructor. call before the HTTP server drains.
    void shutdown();

private:
    std::vector<server_mcp_server_config> configs;

    mutable std::mutex mutex; // guards transports, dead_servers, registry
    std::map<std::string, std::shared_ptr<server_mcp_transport>> transports;
    std::map<std::string, std::chrono::steady_clock::time_point> dead_servers; // spawn-failure cooldown
    std::vector<server_mcp_tool_def> registry;

    std::atomic<bool> stopping{false};

    const server_mcp_server_config * find_config(const std::string & name) const;

    // the only place that names a concrete transport
    std::shared_ptr<server_mcp_transport> create_transport(const server_mcp_server_config & cfg);

    // nullptr during cooldown or shutdown
    std::shared_ptr<server_mcp_transport> get_or_create(const std::string & name);
};
