#include "server-mcp.h"

#include "server-tools.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>
#endif

using json = nlohmann::ordered_json;

static constexpr const char * MCP_PROTOCOL_VERSION = "2024-11-05";

//
// Config parsing
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
    json j = json::parse(json_str);
    return parse_cursor_format(j);
}

std::vector<server_mcp_server_config> server_mcp_server_config::parse_cursor_format(const json & j) {
    std::vector<server_mcp_server_config> result;
    if (!j.contains("mcpServers") || !j.at("mcpServers").is_object()) {
        return result;
    }
    for (const auto & [name, cfg] : j.at("mcpServers").items()) {
        server_mcp_server_config sc;
        sc.name = name;
        if (cfg.contains("command")) {
            sc.command = cfg.at("command").get<std::string>();
        }
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
        if (cfg.contains("cwd")) {
            sc.cwd = cfg.at("cwd").get<std::string>();
        }
        if (cfg.contains("timeout_ms")) {
            sc.timeout_ms = cfg.at("timeout_ms").get<int>();
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
// Process handle (platform-specific)
//

struct server_mcp_instance::process_handle {
#if defined(_WIN32)
    HANDLE hProcess;
    HANDLE hStdinWrite;
    HANDLE hStdoutRead;
#else
    pid_t pid;
    int stdin_fd;
    int stdout_fd;
#endif
    process_handle();
    ~process_handle();
    void terminate_process();
    bool is_alive() const;
};

server_mcp_instance::process_handle::process_handle() {
#if defined(_WIN32)
    hProcess = NULL;
    hStdinWrite = NULL;
    hStdoutRead = NULL;
#else
    pid = -1;
    stdin_fd = -1;
    stdout_fd = -1;
#endif
}

server_mcp_instance::process_handle::~process_handle() {
    terminate_process();
}

void server_mcp_instance::process_handle::terminate_process() {
#if defined(_WIN32)
    if (hStdinWrite) { CloseHandle(hStdinWrite); hStdinWrite = NULL; }
    if (hStdoutRead) { CloseHandle(hStdoutRead); hStdoutRead = NULL; }
    if (hProcess) { TerminateProcess(hProcess, 1); CloseHandle(hProcess); hProcess = NULL; }
#else
    if (stdin_fd >= 0) { ::close(stdin_fd); stdin_fd = -1; }
    if (stdout_fd >= 0) { ::close(stdout_fd); stdout_fd = -1; }
    if (pid > 0) {
        kill(pid, SIGTERM);
        // wait a bit, then force kill
        for (int i = 0; i < 10; i++) {
            int status = 0;
            pid_t ret = waitpid(pid, &status, WNOHANG);
            if (ret == pid) break;
            if (ret == -1 && errno == ECHILD) break; // already reaped
            if (ret == -1 && errno == EINTR) continue; // retry
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        kill(pid, SIGKILL); // ignore ESRCH
        waitpid(pid, nullptr, WNOHANG); // ignore errors
        pid = -1;
    }
#endif
}

bool server_mcp_instance::process_handle::is_alive() const {
#if defined(_WIN32)
    if (hProcess == NULL) return false;
    DWORD code = 0;
    if (GetExitCodeProcess(hProcess, &code)) {
        return code == STILL_ACTIVE;
    }
    return false;
#else
    if (pid <= 0) return false;
    // Retry kill(pid, 0) on EINTR. If it returns 0, the process exists;
    // additionally reap any zombie via waitpid(WNOHANG) so we can report
    // it as dead and avoid a busy-loop in send_rpc().
    for (;;) {
        int ret = kill(pid, 0);
        if (ret == 0) {
            int status = 0;
            pid_t wp = waitpid(pid, &status, WNOHANG);
            if (wp == pid) {
                // Child is a zombie; treat as dead
                return false;
            }
            if (wp == -1 && errno == ECHILD) {
                // Already reaped by someone else; treat as dead
                return false;
            }
            // Process exists and is not a zombie
            return true;
        }
        if (ret == -1 && errno == EPERM) {
            // Process exists but we have no permission to signal it
            return true;
        }
        if (ret == -1 && errno == EINTR) {
            continue;
        }
        // ESRCH or other error: process does not exist
        return false;
    }
#endif
}

//
// Spawn helper
//

#if defined(_WIN32)
static std::string quote_windows_arg(const std::string & arg) {
    std::string result;
    bool needs_quotes = arg.find(' ') != std::string::npos || arg.find('"') != std::string::npos;
    if (!needs_quotes) {
        return arg;
    }
    result.push_back('"');
    for (char c : arg) {
        if (c == '"') {
            result.push_back('\\');
            result.push_back('"');
        } else if (c == '\\') {
            result.push_back('\\');
            result.push_back('\\');
        } else {
            result.push_back(c);
        }
    }
    result.push_back('"');
    return result;
}
#endif

static bool spawn_process_stdio(
    const std::string & command,
    const std::vector<std::string> & args,
    const std::map<std::string, std::string> & env,
    const std::string & cwd,
    std::unique_ptr<server_mcp_instance::process_handle> & out_proc
) {
    out_proc = std::make_unique<server_mcp_instance::process_handle>();
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
    HANDLE hStdinRead = NULL, hStdinWrite = NULL;
    HANDLE hStdoutRead = NULL, hStdoutWrite = NULL;

    if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0)) return false;
    if (!CreatePipe(&hStdinRead, &hStdinWrite, &sa, 0)) {
        CloseHandle(hStdoutRead);
        CloseHandle(hStdoutWrite);
        return false;
    }
    if (!SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(hStdoutRead);
        CloseHandle(hStdoutWrite);
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        return false;
    }
    if (!SetHandleInformation(hStdinWrite, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(hStdoutRead);
        CloseHandle(hStdoutWrite);
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        return false;
    }

    std::string cmdline = quote_windows_arg(command);
    for (const auto & a : args) {
        cmdline.push_back(' ');
        cmdline += quote_windows_arg(a);
    }

    STARTUPINFOA si = {};
    si.cb = sizeof(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = hStdinRead;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = hStdoutWrite;

    PROCESS_INFORMATION pi = {};

    std::string env_block;
    if (!env.empty()) {
        // merge with parent env
        env_block.reserve(4096);
        // Get parent environment block
        LPCH parent_env = GetEnvironmentStringsA();
        if (parent_env) {
            for (char *p = parent_env; *p; p += strlen(p) + 1) {
                std::string s = p;
                size_t eq = s.find('=');
                if (eq != std::string::npos && eq > 0) {
                    std::string key = s.substr(0, eq);
                    if (env.find(key) == env.end()) {
                        env_block.append(s);
                        env_block.push_back('\0');
                    }
                }
            }
            FreeEnvironmentStringsA(parent_env);
        }
        for (const auto & e : env) {
            env_block.append(e.first);
            env_block.push_back('=');
            env_block.append(e.second);
            env_block.push_back('\0');
        }
        env_block.push_back('\0');
    }

    BOOL ok = CreateProcessA(
        NULL,
        &cmdline[0],
        NULL, NULL, TRUE,
        CREATE_NO_WINDOW,
        env_block.empty() ? NULL : env_block.data(),
        cwd.empty() ? NULL : cwd.c_str(),
        &si, &pi
    );

    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);
    if (!ok) {
        CloseHandle(hStdoutRead);
        CloseHandle(hStdinWrite);
        return false;
    }

    out_proc->hProcess = pi.hProcess;
    out_proc->hStdinWrite = hStdinWrite;
    out_proc->hStdoutRead = hStdoutRead;
    CloseHandle(pi.hThread);
    return true;
#else
    int stdin_pipe[2] = {-1, -1};
    int stdout_pipe[2] = {-1, -1};

    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
        if (stdin_pipe[0] >= 0) { close(stdin_pipe[0]); close(stdin_pipe[1]); }
        if (stdout_pipe[0] >= 0) { close(stdout_pipe[0]); close(stdout_pipe[1]); }
        return false;
    }

    pid_t pid = fork();
    if (pid == 0) {
        // child
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);

        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        if (!cwd.empty() && chdir(cwd.c_str()) != 0) {
            _exit(127);
        }

        if (!env.empty()) {
            // Build merged environment and temporarily replace environ
            std::vector<std::string> env_strings;
            env_strings.reserve(env.size() + 1);
            for (const auto & e : env) {
                env_strings.push_back(e.first + "=" + e.second);
            }
            // Add parent env vars that aren't overridden
            extern char **environ;
            for (char **e = environ; *e; e++) {
                std::string s = *e;
                size_t eq = s.find('=');
                if (eq != std::string::npos) {
                    std::string key = s.substr(0, eq);
                    if (env.find(key) == env.end()) {
                        env_strings.push_back(s);
                    }
                }
            }
            std::vector<char *> env_ptrs;
            env_ptrs.reserve(env_strings.size() + 1);
            for (auto & s : env_strings) {
                env_ptrs.push_back(const_cast<char *>(s.c_str()));
            }
            env_ptrs.push_back(nullptr);

            // Temporarily replace environ (safe in single-threaded child after fork)
            char **old_environ = environ;
            environ = env_ptrs.data();

            std::vector<char *> argv;
            argv.push_back(const_cast<char *>(command.c_str()));
            for (const auto & a : args) {
                argv.push_back(const_cast<char *>(a.c_str()));
            }
            argv.push_back(nullptr);
            execvp(command.c_str(), argv.data());

            // If execvp returns, restore environ and exit
            environ = old_environ;
            _exit(127);
        } else {
            std::vector<char *> argv;
            argv.push_back(const_cast<char *>(command.c_str()));
            for (const auto & a : args) {
                argv.push_back(const_cast<char *>(a.c_str()));
            }
            argv.push_back(nullptr);
            execvp(command.c_str(), argv.data());
            _exit(127);
        }
    }

    if (pid < 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        return false;
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    out_proc->pid = pid;

    // Check if child exited immediately (e.g., command not found)
    {
        int status = 0;
        pid_t ret = waitpid(pid, &status, WNOHANG | WUNTRACED);
        if (ret == pid) {
            // Child exited immediately
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            out_proc->pid = -1;
            return false;
        }
        if (ret == -1 && errno == EINTR) {
            // Child is still running, proceed
        }
    }

    out_proc->stdin_fd = stdin_pipe[1];
    out_proc->stdout_fd = stdout_pipe[0];

    // Set non-blocking so read_message() and write_message() never stall forever
    int flags = fcntl(out_proc->stdin_fd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(out_proc->stdin_fd, F_SETFL, flags | O_NONBLOCK);
    }
    flags = fcntl(out_proc->stdout_fd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(out_proc->stdout_fd, F_SETFL, flags | O_NONBLOCK);
    }

    return true;
#endif
}

//
// server_mcp_instance
//

server_mcp_instance::~server_mcp_instance() = default;

bool server_mcp_instance::spawn(const server_mcp_server_config & config) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (proc) {
        terminate();
    }
    error.clear();
    if (!spawn_process_stdio(config.command, config.args, config.env, config.cwd, proc)) {
        error = "failed to spawn MCP server: " + config.command;
        return false;
    }
    timeout_ms = config.timeout_ms;
    read_buffer.clear();
    initialized = false;
    tools.clear();
    next_id = 1;
    return true;
}

void server_mcp_instance::terminate() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (proc) {
        proc.reset();
    }
    read_buffer.clear();
    initialized = false;
    error.clear();
}

bool server_mcp_instance::is_alive() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!proc) return false;
    return proc->is_alive();
}

static bool try_extract_line(std::string & read_buffer, json & out) {
    size_t pos;
    while ((pos = read_buffer.find('\n')) != std::string::npos) {
        std::string line = read_buffer.substr(0, pos);
        read_buffer.erase(0, pos + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        try {
            out = json::parse(line);
            return true;
        } catch (...) {
            // skip malformed line
        }
    }
    return false;
}

bool server_mcp_instance::read_message(json & out) {
    if (!proc) return false;

    // Drain any complete lines already buffered before touching the fd
    if (try_extract_line(read_buffer, out)) return true;

#if defined(_WIN32)
    char buf[4096];
    DWORD read = 0;
    while (true) {
        DWORD avail = 0;
        if (!PeekNamedPipe(proc->hStdoutRead, NULL, 0, NULL, &avail, NULL) || avail == 0) {
            // No data available right now; try buffered lines one more time
            if (try_extract_line(read_buffer, out)) return true;
            return false;
        }
        BOOL ok = ReadFile(proc->hStdoutRead, buf, sizeof(buf), &read, NULL);
        if (!ok || read == 0) {
            if (GetLastError() == ERROR_BROKEN_PIPE) {
                // Parse any complete lines left in buffer before giving up
                break;
            }
            if (try_extract_line(read_buffer, out)) return true;
            return false;
        }
        read_buffer.append(buf, read);
        if (try_extract_line(read_buffer, out)) return true;
    }
#else
    char buf[4096];
    while (true) {
        ssize_t n = read(proc->stdout_fd, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Try buffered lines before giving up on non-blocking fd
                if (try_extract_line(read_buffer, out)) return true;
                return false;
            }
            return false;
        }
        if (n == 0) {
            // Parse any complete lines left in buffer before giving up
            break;
        }
        read_buffer.append(buf, n);
        if (try_extract_line(read_buffer, out)) return true;
    }
#endif
    // Final attempt to parse any trailing complete line in the buffer
    if (try_extract_line(read_buffer, out)) return true;
    return false;
}

bool server_mcp_instance::write_message(const json & msg) {
    if (!proc) return false;
    std::string data = msg.dump() + "\n";
#if defined(_WIN32)
    DWORD total_written = 0;
    while (total_written < data.size()) {
        DWORD written = 0;
        BOOL ok = WriteFile(proc->hStdinWrite, data.data() + total_written, (DWORD)(data.size() - total_written), &written, NULL);
        if (!ok || written == 0) {
            return false;
        }
        total_written += written;
    }
    return true;
#else
    size_t total = 0;
    while (total < data.size()) {
        ssize_t n = write(proc->stdin_fd, data.data() + total, data.size() - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Retry a few times with small delays instead of hard failure
                for (int retry = 0; retry < 10; ++retry) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    n = write(proc->stdin_fd, data.data() + total, data.size() - total);
                    if (n >= 0) { total += n; break; }
                    if (errno != EAGAIN && errno != EWOULDBLOCK) return false;
                }
                if (n < 0) return false;
                continue;
            }
            return false;
        }
        if (n == 0) return false;
        total += n;
    }
    return true;
#endif
}

json server_mcp_instance::send_rpc(const json & request, int timeout_ms) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!proc) {
        return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
    }

    if (terminating || (mgr_shutting_down && mgr_shutting_down->load())) {
        return {{"error", {{"code", -32603}, {"message", "instance is terminating"}}}};
    }

    if (!write_message(request)) {
        return {{"error", {{"code", -32603}, {"message", "failed to write request"}}}};
    }

    // Track request id for correlation (JSON-RPC 2.0)
    uint64_t request_id = request.value("id", (uint64_t)-1);
    bool request_has_id = request.contains("id");

    // Wait for response with timeout
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    json resp;
    while (true) {
        // Guard against proc being destroyed by signal handler during blocking calls
        if (!proc) {
            return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
        }

        if (terminating || (mgr_shutting_down && mgr_shutting_down->load())) {
            return {{"error", {{"code", -32603}, {"message", "instance is terminating"}}}};
        }

        // Check if process is still alive
        if (!proc->is_alive()) {
            return {{"error", {{"code", -32603}, {"message", "process died"}}}};
        }

        // Check deadline before blocking
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return {{"error", {{"code", -32603}, {"message", "request timed out"}}}};
        }

        // Drain any complete lines already buffered before blocking on select/peek
        if (read_message(resp)) {
            if (!proc) {
                return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
            }
            if (request_has_id && !resp.contains("id")) {
                // Notification or malformed response — keep waiting for actual response
                continue;
            }
            if (request_has_id && resp.contains("id") && resp["id"] != request_id) {
                // Out-of-order response — keep waiting
                continue;
            }
            return resp;
        }

        // Try to read with a small timeout
#if defined(_WIN32)
        // On Windows, check for data availability without waiting for process exit
        DWORD available = 0;
        if (PeekNamedPipe(proc->hStdoutRead, NULL, 0, NULL, &available, NULL) && available > 0) {
            if (read_message(resp)) {
                if (!proc) {
                    return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
                }
                if (request_has_id && !resp.contains("id")) {
                    // Notification or malformed response — keep waiting for actual response
                    continue;
                }
                if (request_has_id && resp.contains("id") && resp["id"] != request_id) {
                    // Out-of-order response — keep waiting
                    continue;
                }
                return resp;
            }
        } else {
            // Avoid busy-looping when the pipe is idle
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Check if process exited
        if (WaitForSingleObject(proc->hProcess, 0) == WAIT_OBJECT_0) {
            // Process exited, try one more read
            if (read_message(resp)) {
                if (!proc) {
                    return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
                }
                if (request_has_id && !resp.contains("id")) {
                    // Notification or malformed response — keep waiting
                    continue;
                }
                if (request_has_id && resp.contains("id") && resp["id"] != request_id) {
                    // Out-of-order response — keep waiting
                    continue;
                }
                return resp;
            }
            return {{"error", {{"code", -32603}, {"message", "process exited"}}}};
        }
#else
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(proc->stdout_fd, &rfds);
        struct timeval tv;
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        if (remaining.count() <= 0) {
            return {{"error", {{"code", -32603}, {"message", "request timed out"}}}};
        }
        long tv_usec = remaining.count() * 1000;
        if (tv_usec > 50 * 1000) tv_usec = 50 * 1000;
        tv.tv_sec = 0;
        tv.tv_usec = tv_usec;
        int sel = select(proc->stdout_fd + 1, &rfds, NULL, NULL, &tv);
        if (sel > 0 && FD_ISSET(proc->stdout_fd, &rfds)) {
            if (read_message(resp)) {
                if (!proc) {
                    return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
                }
                if (request_has_id && !resp.contains("id")) {
                    // Notification or malformed response — keep waiting for actual response
                    continue;
                }
                if (request_has_id && resp.contains("id") && resp["id"] != request_id) {
                    // Out-of-order response — keep waiting
                    continue;
                }
                return resp;
            }
            // Data was available but read_message failed; check if child died
            int status = 0;
            pid_t wp = waitpid(proc->pid, &status, WNOHANG);
            if (wp == proc->pid || (wp == -1 && errno == ECHILD)) {
                return {{"error", {{"code", -32603}, {"message", "process exited"}}}};
            }
        } else if (sel < 0) {
            if (errno == EINTR) {
                // Signal may have destroyed proc; check before retrying
                if (!proc) {
                    return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
                }
                continue;
            }
            return {{"error", {{"code", -32603}, {"message", "select failed"}}}};
        }
#endif
        // Deadline is checked at the top of the next iteration
    }
}

std::vector<server_mcp_tool_definition> server_mcp_instance::list_tools() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!proc) return {};
    if (initialized) return tools;

    error.clear();

    // initialize handshake
    json init_req = {
        {"jsonrpc", "2.0"},
        {"id", next_id++},
        {"method", "initialize"},
        {"params", {
            {"protocolVersion", MCP_PROTOCOL_VERSION},
            {"capabilities", json::object()},
            {"clientInfo", {{"name", "llama.cpp"}, {"version", "1.0"}}}
        }}
    };

    json init_resp = send_rpc(init_req, 30000);
    if (init_resp.contains("error")) {
        error = "initialize failed: " + init_resp["error"].value("message", "unknown error");
        return {};
    }
    if (!init_resp.contains("result")) {
        error = "initialize failed: missing result in response";
        return {};
    }

    // Send initialized notification
    json init_notif = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
    if (!write_message(init_notif)) {
        error = "initialize failed: failed to send initialized notification";
        return {};
    }

    // List tools
    json list_req = {
        {"jsonrpc", "2.0"},
        {"id", next_id++},
        {"method", "tools/list"}
    };

    json list_resp = send_rpc(list_req, 30000);
    if (list_resp.contains("error")) {
        error = "tools/list failed: " + list_resp["error"].value("message", "unknown error");
        return {};
    }
    if (!list_resp.contains("result")) {
        error = "tools/list failed: missing result in response";
        return {};
    }

    tools.clear();
    if (list_resp["result"].contains("tools")) {
        for (const auto & t : list_resp["result"]["tools"]) {
            server_mcp_tool_definition def;
            def.name = t.value("name", "");
            def.description = t.value("description", "");
            if (t.contains("inputSchema")) {
                def.input_schema = t["inputSchema"];
            }
            def.server_name = server_name;
            tools.push_back(def);
        }
    }

    initialized = true;
    return tools;
}

json server_mcp_instance::call_tool(const std::string & tool_name, const json & arguments, int timeout_ms) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!proc) {
        return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
    }

    if (!initialized) {
        list_tools(); // ensure initialized
        if (!initialized) {
            return {{"error", {{"code", -32603}, {"message", "initialization failed: " + error}}}};
        }
    }

    json call_req = {
        {"jsonrpc", "2.0"},
        {"id", next_id++},
        {"method", "tools/call"},
        {"params", {
            {"name", tool_name},
            {"arguments", arguments}
        }}
    };

    json resp = send_rpc(call_req, timeout_ms);
    if (resp.contains("error")) {
        return resp;
    }
    if (resp.contains("result")) {
        return resp["result"];
    }
    return {{"error", {{"code", -32603}, {"message", "invalid response"}}}};
}

//
// server_mcp_manager
//

server_mcp_manager::server_mcp_manager(std::vector<server_mcp_server_config> configs)
    : configs(std::move(configs)) {}

server_mcp_manager::~server_mcp_manager() {
    close_all();
}

void server_mcp_manager::warmup() {
    std::vector<server_mcp_tool_definition> discovered;
    for (const auto & cfg : configs) {
        auto instance = std::make_unique<server_mcp_instance>();
        instance->server_name = cfg.name;
        if (!instance->spawn(cfg)) {
            SRV_WRN("MCP warmup failed for '%s': %s\n", cfg.name.c_str(), instance->error.c_str());
        } else {
            auto tools = instance->list_tools();
            if (!instance->error.empty()) {
                SRV_WRN("MCP warmup had errors for '%s': %s\n", cfg.name.c_str(), instance->error.c_str());
            } else {
                SRV_INF("MCP warmup succeeded for '%s', discovered %zu tools\n", cfg.name.c_str(), tools.size());
            }
        }
        // Always insert discovered tools, even if warmup had errors,
        // so that tools are listed in GET /tools for servers that may
        // succeed on lazy spawn.
        auto & instance_tools = instance->tools;
        discovered.insert(discovered.end(), instance_tools.begin(), instance_tools.end());
        instance->terminate();
    }
    std::lock_guard<std::mutex> lock(mutex);
    warmup_tools.swap(discovered);
}

std::vector<server_mcp_tool_definition> server_mcp_manager::get_all_tools() const {
    std::lock_guard<std::mutex> lock(mutex);
    return warmup_tools;
}

std::shared_ptr<server_mcp_instance> server_mcp_manager::get_or_create(const std::string & server_name) {
    if (shutting_down->load()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex);

    // Prune expired cooldowns
    auto now = std::chrono::steady_clock::now();
    for (auto it = dead_servers.begin(); it != dead_servers.end(); ) {
        if (now >= it->second) {
            it = dead_servers.erase(it);
        } else {
            ++it;
        }
    }

    // Check cooldown first
    auto dead_it = dead_servers.find(server_name);
    if (dead_it != dead_servers.end() && now < dead_it->second) {
        return nullptr;
    }

    // Find existing global instance for this server
    auto it = global_instances.find(server_name);
    if (it != global_instances.end()) {
        auto & inst = it->second;
        if (!inst->is_alive()) {
            // respawn if dead — move the old instance out so its destruction
            // (which may block ~500ms in terminate_process) does not hold the manager lock
            auto old_inst = std::move(it->second);
            global_instances.erase(it);

            auto cfg_it = std::find_if(configs.begin(), configs.end(),
                [&](const server_mcp_server_config & c) { return c.name == server_name; });
            if (cfg_it != configs.end()) {
                if (!old_inst->spawn(*cfg_it)) {
                    // failed to respawn, create a new one
                    dead_servers[server_name] = now + std::chrono::seconds(5);
                    return do_spawn(server_name);
                }
                if (!old_inst->is_alive()) {
                    // Spawned but process died immediately (e.g. bad binary)
                    dead_servers[server_name] = now + std::chrono::seconds(5);
                    return do_spawn(server_name);
                }
                // Do NOT call list_tools() under manager lock; let call_tool() lazy-init
                global_instances[server_name] = std::move(old_inst);
                return global_instances[server_name];
            }
            // config not found, old_inst destroyed outside lock
        } else {
            return inst;
        }
    }

    return do_spawn(server_name);
}

std::shared_ptr<server_mcp_instance> server_mcp_manager::do_spawn(const std::string & server_name) {
    if (shutting_down->load()) {
        return nullptr;
    }

    auto cfg_it = std::find_if(configs.begin(), configs.end(),
        [&](const server_mcp_server_config & c) { return c.name == server_name; });
    if (cfg_it == configs.end()) {
        return nullptr;
    }

    // Prune expired cooldowns
    auto now = std::chrono::steady_clock::now();
    for (auto it = dead_servers.begin(); it != dead_servers.end(); ) {
        if (now >= it->second) {
            it = dead_servers.erase(it);
        } else {
            ++it;
        }
    }

    auto instance = std::make_shared<server_mcp_instance>();
    instance->server_name = server_name;
    instance->mgr_shutting_down = shutting_down;
    if (!instance->spawn(*cfg_it)) {
        SRV_WRN("MCP spawn failed for '%s': %s\n", server_name.c_str(), instance->error.c_str());
        dead_servers[server_name] = now + std::chrono::seconds(5);
        return nullptr;
    }

    if (!instance->is_alive()) {
        SRV_WRN("MCP spawn for '%s': process exited immediately\n", server_name.c_str());
        dead_servers[server_name] = now + std::chrono::seconds(5);
        return nullptr;
    }

    // Do NOT call list_tools() under manager lock; let call_tool() lazy-init

    global_instances[server_name] = instance;
    return instance;
}

void server_mcp_manager::begin_shutdown() {
    // Single atomic store: no locks, safe to call from a signal handler.
    // Every live instance shares this flag, so one store bails them all out.
    shutting_down->store(true);
}

void server_mcp_manager::close_all() {
    shutting_down->store(true);

    std::vector<std::shared_ptr<server_mcp_instance>> to_terminate;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & [server_name, instance] : global_instances) {
            (void)server_name;
            to_terminate.push_back(instance);
        }
        global_instances.clear();
        warmup_tools.clear();
        dead_servers.clear();
    }
    for (auto & inst : to_terminate) {
        inst->terminating = true;
        inst->terminate();
    }
}
