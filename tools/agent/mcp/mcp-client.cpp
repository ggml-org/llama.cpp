#include "mcp-client.h"

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <poll.h>
#include <cstring>
#include <chrono>

mcp_client::mcp_client() = default;

mcp_client::~mcp_client() {
    shutdown();
}

bool mcp_client::connect(const std::string & command,
                         const std::vector<std::string> & args,
                         const std::map<std::string, std::string> & env,
                         int timeout_ms) {
    // Create pipes for stdin/stdout
    int stdin_pipe[2];   // Parent writes to [1], child reads from [0]
    int stdout_pipe[2];  // Child writes to [1], parent reads from [0]

    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0) {
        last_error_ = "Failed to create pipes";
        return false;
    }

    pid_ = fork();
    if (pid_ < 0) {
        last_error_ = "Failed to fork";
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return false;
    }

    if (pid_ == 0) {
        // Child process
        // Redirect stdin
        close(stdin_pipe[1]);
        dup2(stdin_pipe[0], STDIN_FILENO);
        close(stdin_pipe[0]);

        // Redirect stdout
        close(stdout_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdout_pipe[1]);

        // Set environment variables
        for (const auto & [key, value] : env) {
            setenv(key.c_str(), value.c_str(), 1);
        }

        // Build argv
        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(command.c_str()));
        for (const auto & arg : args) {
            argv.push_back(const_cast<char *>(arg.c_str()));
        }
        argv.push_back(nullptr);

        // Execute
        execvp(command.c_str(), argv.data());

        // If we get here, exec failed
        _exit(127);
    }

    // Parent process
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    stdin_fd_ = stdin_pipe[1];
    stdout_fd_ = stdout_pipe[0];

    // Set stdout to non-blocking for timeout handling
    int flags = fcntl(stdout_fd_, F_GETFL, 0);
    fcntl(stdout_fd_, F_SETFL, flags | O_NONBLOCK);

    // Perform MCP initialize handshake
    json init_params = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", json::object()},
        {"clientInfo", {
            {"name", "llama-agent"},
            {"version", "1.0.0"}
        }}
    };

    json response = send_request("initialize", init_params, timeout_ms);
    if (response.is_null()) {
        shutdown();
        return false;
    }

    // Extract server info
    if (response.contains("serverInfo") && response["serverInfo"].contains("name")) {
        server_name_ = response["serverInfo"]["name"].get<std::string>();
    } else {
        server_name_ = "unknown";
    }

    // Send initialized notification
    json notification = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/initialized"}
    };
    write_message(notification);

    initialized_ = true;
    return true;
}

bool mcp_client::is_connected() const {
    if (pid_ <= 0 || !initialized_) {
        return false;
    }

    // Check if process is still running
    int status;
    pid_t result = waitpid(pid_, &status, WNOHANG);
    return result == 0;  // 0 means process still running
}

std::vector<mcp_tool> mcp_client::list_tools() {
    std::vector<mcp_tool> tools;

    if (!is_connected()) {
        last_error_ = "Not connected";
        return tools;
    }

    json response = send_request("tools/list", json::object(), 30000);
    if (response.is_null()) {
        return tools;
    }

    if (!response.contains("tools") || !response["tools"].is_array()) {
        last_error_ = "Invalid tools/list response";
        return tools;
    }

    for (const auto & tool_json : response["tools"]) {
        mcp_tool tool;
        tool.name = tool_json.value("name", "");
        tool.description = tool_json.value("description", "");
        if (tool_json.contains("inputSchema")) {
            tool.input_schema = tool_json["inputSchema"];
        } else {
            // Default empty schema
            tool.input_schema = {{"type", "object"}, {"properties", json::object()}};
        }

        if (!tool.name.empty()) {
            tools.push_back(tool);
        }
    }

    return tools;
}

mcp_call_result mcp_client::call_tool(const std::string & name,
                                       const json & arguments,
                                       int timeout_ms) {
    mcp_call_result result;
    result.is_error = true;

    if (!is_connected()) {
        last_error_ = "Not connected";
        result.content.push_back({{"type", "text"}, {"text", "MCP server not connected"}});
        return result;
    }

    json params = {
        {"name", name},
        {"arguments", arguments}
    };

    json response = send_request("tools/call", params, timeout_ms);
    if (response.is_null()) {
        result.content.push_back({{"type", "text"}, {"text", last_error_}});
        return result;
    }

    result.is_error = response.value("isError", false);

    if (response.contains("content") && response["content"].is_array()) {
        result.content = response["content"].get<std::vector<json>>();
    }

    return result;
}

void mcp_client::shutdown() {
    if (pid_ > 0) {
        // Try graceful shutdown first
        if (stdin_fd_ >= 0) {
            close(stdin_fd_);
            stdin_fd_ = -1;
        }

        // Wait briefly for process to exit
        int status;
        for (int i = 0; i < 10; i++) {
            if (waitpid(pid_, &status, WNOHANG) != 0) {
                break;
            }
            usleep(100000);  // 100ms
        }

        // Force kill if still running
        if (waitpid(pid_, &status, WNOHANG) == 0) {
            kill(pid_, SIGTERM);
            usleep(100000);
            if (waitpid(pid_, &status, WNOHANG) == 0) {
                kill(pid_, SIGKILL);
                waitpid(pid_, &status, 0);
            }
        }

        pid_ = -1;
    }

    if (stdout_fd_ >= 0) {
        close(stdout_fd_);
        stdout_fd_ = -1;
    }

    initialized_ = false;
    read_buffer_.clear();
}

json mcp_client::send_request(const std::string & method, const json & params, int timeout_ms) {
    int id = ++request_id_;

    json request = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method},
        {"params", params}
    };

    if (!write_message(request)) {
        return json();
    }

    // Read response with matching ID
    auto start = std::chrono::steady_clock::now();
    while (true) {
        int remaining = timeout_ms;
        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            remaining = timeout_ms - elapsed;
            if (remaining <= 0) {
                last_error_ = "Request timed out";
                return json();
            }
        }

        json msg = read_message(remaining);
        if (msg.is_null()) {
            return json();
        }

        // Skip notifications (no id)
        if (!msg.contains("id")) {
            continue;
        }

        // Check if this is our response
        if (msg["id"].get<int>() == id) {
            if (msg.contains("error")) {
                last_error_ = msg["error"].value("message", "Unknown error");
                return json();
            }
            if (msg.contains("result")) {
                return msg["result"];
            }
            last_error_ = "Invalid response";
            return json();
        }
    }
}

json mcp_client::read_message(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        // Check for complete message in buffer
        size_t newline_pos = read_buffer_.find('\n');
        if (newline_pos != std::string::npos) {
            std::string line = read_buffer_.substr(0, newline_pos);
            read_buffer_.erase(0, newline_pos + 1);

            // Skip empty lines
            if (line.empty() || (line.size() == 1 && line[0] == '\r')) {
                continue;
            }

            // Remove trailing \r if present
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            try {
                return json::parse(line);
            } catch (const json::parse_error & e) {
                // Not valid JSON, skip
                continue;
            }
        }

        // Calculate remaining timeout
        int remaining = timeout_ms;
        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            remaining = timeout_ms - elapsed;
            if (remaining <= 0) {
                last_error_ = "Read timed out";
                return json();
            }
        }

        // Poll for data
        struct pollfd pfd;
        pfd.fd = stdout_fd_;
        pfd.events = POLLIN;

        int poll_timeout = (timeout_ms > 0) ? remaining : -1;
        int ret = poll(&pfd, 1, poll_timeout);

        if (ret < 0) {
            if (errno == EINTR) continue;
            last_error_ = "Poll error: " + std::string(strerror(errno));
            return json();
        }

        if (ret == 0) {
            last_error_ = "Read timed out";
            return json();
        }

        // Read available data
        char buf[4096];
        ssize_t n = read(stdout_fd_, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            last_error_ = "Read error: " + std::string(strerror(errno));
            return json();
        }

        if (n == 0) {
            last_error_ = "Server disconnected";
            return json();
        }

        read_buffer_.append(buf, n);
    }
}

bool mcp_client::write_message(const json & msg) {
    std::string data = msg.dump() + "\n";

    size_t total = 0;
    while (total < data.size()) {
        ssize_t n = write(stdin_fd_, data.data() + total, data.size() - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            last_error_ = "Write error: " + std::string(strerror(errno));
            return false;
        }
        total += n;
    }

    return true;
}
