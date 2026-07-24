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
#include <poll.h>
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
    // mutable: is_alive() is const but reaps the child, and a reaped pid must be
    // invalidated immediately so it is never signalled again after the OS recycles it
    mutable pid_t pid;
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
        bool reaped = false;
        kill(pid, SIGTERM);
        // wait a bit, then force kill
        for (int i = 0; i < 10; i++) {
            int status = 0;
            pid_t ret = waitpid(pid, &status, WNOHANG);
            if (ret == pid) { reaped = true; break; }
            if (ret == -1 && errno == ECHILD) { reaped = true; break; } // already reaped
            if (ret == -1 && errno == EINTR) continue; // retry
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        // Only signal again if the child was NOT reaped above: once waitpid() has
        // succeeded the pid is free for reuse, and kill()ing it could hit an
        // unrelated process that the OS has since given the same pid.
        if (!reaped) {
            kill(pid, SIGKILL); // ignore ESRCH
            // Blocking wait: SIGKILL cannot be caught or ignored, so this cannot
            // hang. A WNOHANG here would usually race the kill and leak a zombie.
            while (waitpid(pid, nullptr, 0) == -1 && errno == EINTR) {}
        }
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
                // Child is a zombie; it is now reaped, so invalidate the pid to
                // keep terminate_process() from signalling a recycled pid later
                pid = -1;
                return false;
            }
            if (wp == -1 && errno == ECHILD) {
                // Already reaped by someone else; treat as dead
                pid = -1;
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
// MCP config strings are UTF-8 (they come out of a JSON document). The ANSI
// CreateProcessA/STARTUPINFOA family reinterprets them in the active code page,
// which mangles any non-ASCII command, argument, cwd or env value - so the whole
// spawn path uses the wide variants and converts explicitly.
static std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), NULL, 0);
    if (n <= 0) {
        return std::wstring();
    }
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), &w[0], n);
    return w;
}

static std::wstring quote_windows_arg(const std::wstring & arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos) {
        return arg;
    }

    std::wstring result;
    result.push_back(L'"');

    size_t backslashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }

        if (c == L'"') {
            result.append(backslashes * 2 + 1, L'\\');
        } else {
            result.append(backslashes, L'\\');
        }
        backslashes = 0;
        result.push_back(c);
    }

    result.append(backslashes * 2, L'\\');
    result.push_back(L'"');
    return result;
}

static bool search_path_ext(const std::wstring & name, const wchar_t * ext, std::wstring & out) {
    wchar_t buf[MAX_PATH * 4];
    const DWORD cap = (DWORD) (sizeof(buf) / sizeof(buf[0]));
    DWORD n = SearchPathW(NULL, name.c_str(), ext, cap, buf, NULL);
    if (n > 0 && n < cap) {
        out.assign(buf, n);
        return true;
    }
    return false;
}

// CreateProcess only ever appends ".exe" to an extensionless command and never
// consults PATHEXT, so a "command": "npx" entry (npm ships npx.cmd, never
// npx.exe) fails on Windows while working on POSIX, where execvp() does a full
// PATH search. Resolve the command ourselves so both platforms accept the same
// Cursor-format config.
static std::wstring resolve_windows_command(const std::wstring & command) {
    std::wstring found;

    // exact match first (absolute/relative path, or a command that already has an extension)
    if (search_path_ext(command, NULL, found)) {
        return found;
    }

    std::wstring pathext;
    DWORD need = GetEnvironmentVariableW(L"PATHEXT", NULL, 0);
    if (need > 0) {
        pathext.resize(need);
        DWORD got = GetEnvironmentVariableW(L"PATHEXT", &pathext[0], need);
        pathext.resize(got);
    }
    if (pathext.empty()) {
        pathext = L".COM;.EXE;.BAT;.CMD";
    }

    for (size_t start = 0; start <= pathext.size(); ) {
        size_t sep = pathext.find(L';', start);
        std::wstring ext = pathext.substr(start, sep == std::wstring::npos ? std::wstring::npos : sep - start);
        if (!ext.empty() && search_path_ext(command, ext.c_str(), found)) {
            return found;
        }
        if (sep == std::wstring::npos) {
            break;
        }
        start = sep + 1;
    }

    // give up and let CreateProcessW report the error
    return command;
}
#endif

static bool spawn_process_stdio(
    const std::string & command,
    const std::vector<std::string> & args,
    const std::map<std::string, std::string> & env,
    const std::string & cwd,
    std::unique_ptr<server_mcp_instance::process_handle> & out_proc,
    std::string & out_err
) {
    out_proc = std::make_unique<server_mcp_instance::process_handle>();
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
    HANDLE hStdinRead = NULL, hStdinWrite = NULL;
    HANDLE hStdoutRead = NULL, hStdoutWrite = NULL;

    // 1 MiB rather than the ~4 KiB system default: a large tools/call request
    // should not have to interleave with the child draining its stdin
    const DWORD pipe_buf_size = 1024 * 1024;

    if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, pipe_buf_size)) return false;
    if (!CreatePipe(&hStdinRead, &hStdinWrite, &sa, pipe_buf_size)) {
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

    const std::wstring wcommand = resolve_windows_command(utf8_to_wide(command));
    const std::wstring wcwd     = utf8_to_wide(cwd);

    // A .cmd/.bat target is not a PE image, and CreateProcess only falls back to
    // the command interpreter implicitly when lpApplicationName is NULL - which it
    // is not here - so route batch targets through ComSpec explicitly.
    // NOTE: cmd.exe re-parses this command line with its own rules (the mechanism
    // behind CVE-2024-24576). Arguments come from the operator-controlled MCP
    // config, never from model or client input.
    bool is_batch = false;
    {
        // only look at the final component, so a dot in a directory name
        // (e.g. "C:\tools\node.js\npx") is not mistaken for an extension
        size_t sep = wcommand.find_last_of(L"\\/");
        size_t dot = wcommand.find_last_of(L'.');
        if (dot != std::wstring::npos && (sep == std::wstring::npos || dot > sep)) {
            std::wstring ext = wcommand.substr(dot);
            for (wchar_t & c : ext) {
                if (c >= L'A' && c <= L'Z') c = (wchar_t) (c - L'A' + L'a');
            }
            is_batch = (ext == L".cmd" || ext == L".bat");
        }
    }

    std::wstring app;
    std::wstring cmdline;
    if (is_batch) {
        wchar_t comspec[MAX_PATH];
        DWORD n = GetEnvironmentVariableW(L"ComSpec", comspec, MAX_PATH);
        app = (n > 0 && n < MAX_PATH) ? std::wstring(comspec, n) : std::wstring(L"cmd.exe");
        cmdline  = quote_windows_arg(app);
        cmdline += L" /c ";
        cmdline += quote_windows_arg(wcommand);
    } else {
        app     = wcommand;
        cmdline = quote_windows_arg(wcommand);
    }
    for (const auto & a : args) {
        cmdline.push_back(L' ');
        cmdline += quote_windows_arg(utf8_to_wide(a));
    }

    STARTUPINFOW si = {};
    si.cb = sizeof(STARTUPINFOW);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = hStdinRead;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi = {};

    std::wstring env_block;
    if (!env.empty()) {
        // merge with parent env
        env_block.reserve(4096);
        // Get parent environment block
        LPWCH parent_env = GetEnvironmentStringsW();
        if (parent_env) {
            for (wchar_t *p = parent_env; *p; p += wcslen(p) + 1) {
                std::wstring s = p;
                size_t eq = s.find(L'=');
                if (eq != std::wstring::npos && eq > 0) {
                    std::string key;
                    {
                        std::wstring wkey = s.substr(0, eq);
                        int n = WideCharToMultiByte(CP_UTF8, 0, wkey.data(), (int) wkey.size(), NULL, 0, NULL, NULL);
                        if (n > 0) {
                            key.resize((size_t) n);
                            WideCharToMultiByte(CP_UTF8, 0, wkey.data(), (int) wkey.size(), &key[0], n, NULL, NULL);
                        }
                    }
                    if (env.find(key) == env.end()) {
                        env_block.append(s);
                        env_block.push_back(L'\0');
                    }
                }
            }
            FreeEnvironmentStringsW(parent_env);
        }
        for (const auto & e : env) {
            env_block.append(utf8_to_wide(e.first));
            env_block.push_back(L'=');
            env_block.append(utf8_to_wide(e.second));
            env_block.push_back(L'\0');
        }
        env_block.push_back(L'\0');
    }

    BOOL ok = CreateProcessW(
        app.empty() ? NULL : app.c_str(),
        &cmdline[0],
        NULL, NULL, TRUE,
        CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
        env_block.empty() ? NULL : (LPVOID) env_block.data(),
        wcwd.empty() ? NULL : wcwd.c_str(),
        &si, &pi
    );
    const DWORD spawn_err = ok ? 0 : GetLastError();

    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);
    if (!ok) {
        CloseHandle(hStdoutRead);
        CloseHandle(hStdinWrite);
        // distinguish "not found" from "bad image", "access denied", bad cwd, ...
        out_err = "CreateProcess failed with error " + std::to_string((unsigned long) spawn_err);
        if (spawn_err == ERROR_FILE_NOT_FOUND || spawn_err == ERROR_PATH_NOT_FOUND) {
            out_err += " (command not found in PATH/PATHEXT)";
        }
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
        out_err = std::string("pipe() failed: ") + strerror(errno);
        if (stdin_pipe[0] >= 0) { close(stdin_pipe[0]); close(stdin_pipe[1]); }
        if (stdout_pipe[0] >= 0) { close(stdout_pipe[0]); close(stdout_pipe[1]); }
        return false;
    }

    // Prevent subsequently spawned MCP servers from inheriting these pipes.
    int pipe_fds[] = {stdin_pipe[0], stdin_pipe[1], stdout_pipe[0], stdout_pipe[1]};
    for (int fd : pipe_fds) {
        int flags;
        do {
            flags = fcntl(fd, F_GETFD);
        } while (flags < 0 && errno == EINTR);

        int ret;
        do {
            ret = flags < 0 ? -1 : fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
        } while (ret < 0 && errno == EINTR);

        if (ret < 0) {
            out_err = std::string("fcntl(FD_CLOEXEC) failed: ") + strerror(errno);
            close(stdin_pipe[0]); close(stdin_pipe[1]);
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            return false;
        }
    }

    pid_t pid = fork();
    if (pid == 0) {
        // child
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);

        // If llama-server was started with fd 0 or 1 closed, pipe() can hand back
        // the very fd we are about to dup2() onto. dup2(fd, fd) is a no-op that
        // does NOT clear FD_CLOEXEC, and the close() that follows would then
        // destroy the descriptor we just installed. Install each end explicitly.
        auto install_fd = [](int fd, int target) {
            if (fd != target) {
                dup2(fd, target);
                close(fd);
                return;
            }
            // fd is already the target: keep it, but it must survive execvp()
            int fl = fcntl(fd, F_GETFD);
            if (fl >= 0) {
                fcntl(fd, F_SETFD, fl & ~FD_CLOEXEC);
            }
        };
        install_fd(stdin_pipe[0], STDIN_FILENO);
        install_fd(stdout_pipe[1], STDOUT_FILENO);

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
        out_err = std::string("fork() failed: ") + strerror(errno);
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
        // plain WNOHANG: WUNTRACED would also report a merely *stopped* child,
        // and status is never decoded here, so a stop would be misread as an exit
        pid_t ret = waitpid(pid, &status, WNOHANG);
        if (ret == pid) {
            // Child exited immediately - most often execvp() failing with _exit(127)
            if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
                out_err = "command not found or not executable";
            } else {
                out_err = "process exited immediately";
            }
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
    std::string spawn_err;
    if (!spawn_process_stdio(config.command, config.args, config.env, config.cwd, proc, spawn_err)) {
        error = "failed to spawn MCP server: " + config.command;
        if (!spawn_err.empty()) {
            error += " (" + spawn_err + ")";
        }
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
    size_t total = 0;

    // Bound the write by the same budget the caller gives a request: a child that
    // stops draining its stdin must never park this thread - it holds the instance
    // mutex, so a permanent block here would also wedge terminate() and shutdown.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

#if defined(_WIN32)
    // A blocking WriteFile on a full pipe waits for the child with no way to
    // cancel it, so switch the handle to non-blocking and poll it instead.
    DWORD nowait = PIPE_NOWAIT;
    SetNamedPipeHandleState(proc->hStdinWrite, &nowait, NULL, NULL);

    while (total < data.size()) {
        DWORD written = 0;
        BOOL ok = WriteFile(proc->hStdinWrite, data.data() + total, (DWORD) (data.size() - total), &written, NULL);
        if (ok && written > 0) {
            total += written;
            continue;
        }
        if (!ok) {
            DWORD err = GetLastError();
            // ERROR_NO_DATA is what a PIPE_NOWAIT handle reports for "pipe full"
            if (err != ERROR_NO_DATA && err != ERROR_PIPE_BUSY) {
                break;
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#else
    while (total < data.size()) {
        ssize_t n = write(proc->stdin_fd, data.data() + total, data.size() - total);
        if (n > 0) {
            total += (size_t) n;
            continue;
        }
        if (n == 0) break;
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) break;

        // Pipe is full. Wait for room against the deadline rather than a fixed
        // 10x1ms budget, which is far too tight for a legitimately large frame.
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        int slice_ms = remaining.count() > 50 ? 50 : (int) remaining.count();

        struct pollfd pfd;
        pfd.fd      = proc->stdin_fd;
        pfd.events  = POLLOUT;
        pfd.revents = 0;
        int pr = poll(&pfd, 1, slice_ms);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) break;
    }
#endif

    if (total == data.size()) {
        return true;
    }

    // A partial write leaves a truncated JSON line in the child's stdin: its next
    // readline() would splice the fragment onto the following request, so every
    // later exchange on this instance is desynced. The stream is unrecoverable -
    // drop the process and let the manager respawn it on the next call.
    if (total > 0) {
        SRV_WRN("MCP server '%s': partial write (%zu/%zu bytes), dropping desynced process\n",
                server_name.c_str(), total, data.size());
        terminate();
    }
    return false;
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

        // Evaluated once per iteration, and honoured on every path below: a child
        // that streams an endless run of non-matching messages keeps read_message()
        // returning true, so the "keep waiting" paths must not skip this check.
        auto now = std::chrono::steady_clock::now();
        const bool expired = now >= deadline;

        // Drain buffered/pending lines BEFORE the liveness check. read_message()
        // also pulls whatever is still sitting in the pipe, so a child that answered
        // and then exited still gets its response delivered, instead of having it
        // discarded as "process died".
        if (read_message(resp)) {
            if (!proc) {
                return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
            }
            if (request_has_id && !resp.contains("id")) {
                // Notification or malformed response - keep waiting for actual response
                if (expired) {
                    return {{"error", {{"code", -32603}, {"message", "request timed out"}}}};
                }
                continue;
            }
            if (request_has_id && resp.contains("id") && resp["id"] != request_id) {
                // Out-of-order response - keep waiting
                if (expired) {
                    return {{"error", {{"code", -32603}, {"message", "request timed out"}}}};
                }
                continue;
            }
            return resp;
        }

        if (expired) {
            return {{"error", {{"code", -32603}, {"message", "request timed out"}}}};
        }

        // Check if process is still alive
        if (!proc->is_alive()) {
            return {{"error", {{"code", -32603}, {"message", "process died"}}}};
        }

        // Wait for the child to produce something; the drain at the top of the loop
        // is what actually parses it, so there is no response handling down here.
#if defined(_WIN32)
        DWORD available = 0;
        if (PeekNamedPipe(proc->hStdoutRead, NULL, 0, NULL, &available, NULL) && available > 0) {
            continue;
        }
        // Avoid busy-looping when the pipe is idle
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (WaitForSingleObject(proc->hProcess, 0) == WAIT_OBJECT_0) {
            return {{"error", {{"code", -32603}, {"message", "process exited"}}}};
        }
#else
        // poll instead of select: stdout_fd can exceed FD_SETSIZE on servers with many open descriptors
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        if (remaining.count() <= 0) {
            return {{"error", {{"code", -32603}, {"message", "request timed out"}}}};
        }
        int slice_ms = remaining.count() > 50 ? 50 : (int) remaining.count();
        struct pollfd pfd;
        pfd.fd = proc->stdout_fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int sel = poll(&pfd, 1, slice_ms);
        if (sel > 0) {
            if (pfd.revents & POLLIN) {
                // readable: let the drain at the top of the loop consume it
                continue;
            }
            if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) {
                // EOF or error with nothing left to read. poll() reports these
                // immediately and forever, so looping here would burn a core until
                // the deadline. If the child is gone say so; if it is still running
                // with its stdout closed, no response can ever arrive - fail fast.
                const pid_t child = proc->pid;
                int status = 0;
                pid_t wp = waitpid(child, &status, WNOHANG);
                if (wp == child || (wp == -1 && errno == ECHILD)) {
                    proc->pid = -1; // reaped: never signal this pid again
                    return {{"error", {{"code", -32603}, {"message", "process exited"}}}};
                }
                return {{"error", {{"code", -32603}, {"message", "child closed stdout"}}}};
            }
        } else if (sel < 0) {
            if (errno == EINTR) {
                // Signal may have destroyed proc; check before retrying
                if (!proc) {
                    return {{"error", {{"code", -32603}, {"message", "process not running"}}}};
                }
                continue;
            }
            return {{"error", {{"code", -32603}, {"message", "poll failed"}}}};
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
            // The child is alive but unusable: nothing else tears such an instance
            // down, so without this every later call re-pays the full handshake
            // timeout while holding this instance's mutex. Drop it and let
            // get_or_create() respawn on the next call.
            const std::string init_err = error; // terminate() clears error
            terminate();
            return {{"error", {{"code", -32603}, {"message", "initialization failed: " + init_err}}}};
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

    std::shared_ptr<server_mcp_instance> existing;
    {
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
        if (it == global_instances.end()) {
            return do_spawn(server_name);
        }
        existing = it->second;
    }

    // Probe liveness WITHOUT the manager lock: is_alive() takes the per-instance
    // mutex, which an in-flight send_rpc() holds for up to timeout_ms. Holding the
    // manager lock across that would serialize every MCP server - and every /tools
    // request - behind one slow tool call.
    if (existing->is_alive()) {
        return existing;
    }

    // Dead: re-acquire and re-check before mutating the map, since another thread
    // may have replaced this instance while the lock was released.
    std::lock_guard<std::mutex> lock(mutex);
    auto now = std::chrono::steady_clock::now();

    auto it = global_instances.find(server_name);
    if (it == global_instances.end()) {
        return do_spawn(server_name);
    }
    if (it->second != existing) {
        // someone else already respawned it
        return it->second;
    }

    // respawn - move the old instance out so its destruction
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
