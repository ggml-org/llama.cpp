#pragma once

#include "mcp-client.h"

#include <string>
#include <vector>
#include <map>
#include <memory>

// Configuration for a single MCP server
struct mcp_server_config {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;
    bool enabled = true;
    int timeout_ms = 60000;  // Tool call timeout
};

// Manages multiple MCP server connections
class mcp_server_manager {
public:
    mcp_server_manager() = default;
    ~mcp_server_manager();

    // Load configuration from JSON file
    // Returns true if config was loaded (false if file not found or invalid)
    bool load_config(const std::string & config_path);

    // Start all enabled servers
    // Returns number of servers successfully started
    int start_servers();

    // Shutdown all servers
    void shutdown_all();

    // Get all tools from all connected servers
    // Returns pairs of (qualified_name, tool)
    std::vector<std::pair<std::string, mcp_tool>> list_all_tools();

    // Call a tool by qualified name (mcp__server__tool)
    mcp_call_result call_tool(const std::string & qualified_name,
                              const json & arguments);

    // Get list of server names
    std::vector<std::string> server_names() const;

    // Check if a server is connected
    bool is_server_connected(const std::string & name) const;

    // Get last error message
    std::string last_error() const { return last_error_; }

private:
    std::map<std::string, mcp_server_config> configs_;
    std::map<std::string, std::unique_ptr<mcp_client>> clients_;
    std::string last_error_;

    // Tool name qualification
    static std::string qualify_name(const std::string & server, const std::string & tool);
    static bool parse_qualified_name(const std::string & qualified,
                                     std::string & server,
                                     std::string & tool);

    // Environment variable substitution in config values
    static std::string expand_env_vars(const std::string & value);
};

// Find MCP config file
// Checks: ./mcp.json, then ~/.config/llama-agent/mcp.json
std::string find_mcp_config(const std::string & working_dir);
