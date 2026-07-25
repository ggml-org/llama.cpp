#pragma once

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

    // throw on parse errors; missing "mcpServers" yields an empty list; entries without a "command" are skipped
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
// server_mcp_transport: one MCP server session.
//
//   caller --send_rpc--> to_server   --[writer]--> framing --> server
//   caller <--send_rpc-- from_server <--[reader]-- framing <-- server
//
// each queue item is one complete serialized JSON message.
// subclass owns byte I/O and framing; base owns JSON and the JSON-RPC session (handshake, id correlation).
//

struct server_mcp_transport {
    std::string name;
    int timeout_ms = 30000;

    server_pipe<std::string> to_server;   // serialized messages we send to the server
    server_pipe<std::string> from_server; // serialized messages read from the server

    virtual ~server_mcp_transport() = default;

    virtual bool start() = 0;
    virtual void close() = 0; // blocking and idempotent
    virtual bool is_alive() const = 0; // never blocks behind an in-flight send_rpc()

    // human-readable diagnostics for logging when the transport fails/dies
    // (example: last RPC error, plus any transport-specific detail)
    // may run on a different thread than send_rpc(), so last_error is read under rpc_mutex
    virtual std::string diagnostics() {
        std::lock_guard<std::mutex> lock(rpc_mutex);
        return last_error;
    }

    std::vector<server_mcp_tool_def> list_tools(const std::function<bool()> & should_stop);

    json call_tool(const std::string & tool_name,
                   const json & arguments,
                   const std::function<bool()> & should_stop);

protected:
    // per-transport: send_rpc() holds it across the reply wait, so sharing it would stall every server behind one slow call. guards all members below.
    std::mutex rpc_mutex;
    uint64_t next_id = 1; // reset to 1 per (re)spawn
    bool initialized = false;
    std::string last_error;
    std::vector<server_mcp_tool_def> tools;

    // both assume rpc_mutex is already held by the public caller
    bool ensure_init(const std::function<bool()> & should_stop); // initialize handshake, once
    json send_rpc(const json & request, const std::function<bool()> & should_stop); // returns the reply or an {"error": ...}
};

//
// server_mcp_stdio: child process, NDJSON JSON-RPC over stdio (stderr drained to the debug log)
//

struct server_mcp_stdio : server_mcp_transport {
    explicit server_mcp_stdio(const server_mcp_server_config & config);
    ~server_mcp_stdio() override;

    bool start() override;
    void close() override;
    bool is_alive() const override;
    std::string diagnostics() override;

private:
    server_mcp_server_config config;

    // defined in the .cpp so <windows.h> stays out of this header
    struct process_handle;
    std::unique_ptr<process_handle> proc;

    std::thread reader; // child stdout -> NDJSON de-framing -> from_server
    std::thread writer; // to_server -> NDJSON framing -> child stdin
    std::thread errlog; // child stderr -> debug log (must be drained or the child blocks)

    // cleared by close() or by the reader on stdout EOF; read without rpc_mutex
    std::atomic<bool> running{false};

    // bounded tail of the child's stderr, for diagnostics when it dies
    std::mutex err_mu;
    std::string err_tail;

    void reader_loop();
    void writer_loop();
    void errlog_loop();
    void join_pumps();
};

//
// server_mcp
// declare before the HTTP context so it outlives every /tools handler.
//

class server_mcp {
public:
    server_mcp() = default;
    ~server_mcp();

    // parse the MCP config from params (file and/or inline JSON),
    // then spawn each server once,  list its tools, and shut it down
    // throws on config parse errors; spawn failures are logged.
    void start(const common_params & params);

    // true until start() has parsed at least one server from the config
    bool empty() const { return configs.empty(); }

    std::vector<server_mcp_tool_def> list_tools() const;

    // lazily (re)spawns the transport. returns the MCP result or an {"error": ...}. should_stop is OR-ed with the manager's cancel flag.
    json call_tool(const std::string & server_name,
                   const std::string & tool_name,
                   const json & arguments,
                   const std::function<bool()> & should_stop = nullptr);

    // flip the cancel flag so in-flight calls return; blocking teardown is in the destructor. call before the HTTP server drains.
    // note: multiple calls are idempotent
    void shutdown();

private:
    std::vector<server_mcp_server_config> configs;

    mutable std::mutex mutex; // guards transports, dead_servers, registry

    // shared_ptr: call_tool() hands a transport to the caller and drops the lock for the blocking RPC, so a concurrent evict/respawn must not destroy it mid-call
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
