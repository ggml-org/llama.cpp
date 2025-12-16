#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <map>
#include <atomic>

using json = nlohmann::ordered_json;

// MCP tool definition from server
struct mcp_tool {
    std::string name;
    std::string description;
    json input_schema;
};

// Result from calling an MCP tool
struct mcp_call_result {
    bool is_error = false;
    std::vector<json> content;  // Array of content items
};

// MCP client for stdio transport
// Implements JSON-RPC 2.0 over stdin/stdout pipes to MCP server process
class mcp_client {
public:
    mcp_client();
    ~mcp_client();

    // Disable copy
    mcp_client(const mcp_client &) = delete;
    mcp_client & operator=(const mcp_client &) = delete;

    // Launch server process and perform MCP initialize handshake
    // Returns true on success
    bool connect(const std::string & command,
                 const std::vector<std::string> & args,
                 const std::map<std::string, std::string> & env,
                 int timeout_ms = 30000);

    // Check if connected and server is alive
    bool is_connected() const;

    // Get server name (from initialize response)
    std::string server_name() const { return server_name_; }

    // Get list of available tools
    std::vector<mcp_tool> list_tools();

    // Call a tool with given arguments
    // timeout_ms: max time to wait for response (0 = no timeout)
    mcp_call_result call_tool(const std::string & name,
                              const json & arguments,
                              int timeout_ms = 60000);

    // Graceful shutdown
    void shutdown();

    // Get last error message
    std::string last_error() const { return last_error_; }

private:
    // Process management
    pid_t pid_ = -1;
    int stdin_fd_ = -1;   // Write to server
    int stdout_fd_ = -1;  // Read from server

    // Protocol state
    int request_id_ = 0;
    std::string server_name_;
    std::string last_error_;
    bool initialized_ = false;

    // Send JSON-RPC request and wait for response
    json send_request(const std::string & method, const json & params, int timeout_ms);

    // Read a single JSON-RPC message from server
    json read_message(int timeout_ms);

    // Write a JSON-RPC message to server
    bool write_message(const json & msg);

    // Buffer for partial reads
    std::string read_buffer_;
};
