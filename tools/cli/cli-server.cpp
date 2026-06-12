#include "cli-server.h"

#include <sheredom/subprocess.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#if defined(__linux__)
#include <limits.h>
#endif

// stdin/stdout command protocol between the CLI and the spawned server,
// keep in sync with tools/server/server-models.cpp
#define CMD_ROUTER_TO_CHILD_EXIT   "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_READY  "cmd_child_to_router:ready"
#define CMD_CHILD_TO_ROUTER_PREFIX "cmd_child_to_router:"

// address for the spawned server; always loopback, never exposed
#define CHILD_ADDR "127.0.0.1"

static constexpr int    CLI_SERVER_STOP_TIMEOUT_SEC = 10;
static constexpr size_t CLI_SERVER_MAX_LOG_LINES    = 200;

static bool line_starts_with(const std::string & line, const char * prefix) {
    return line.rfind(prefix, 0) == 0;
}

// same logic as get_free_port() in tools/server/server-models.cpp
static int get_free_port() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return -1;
    }
    typedef SOCKET native_socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
    typedef int native_socket_t;
#define INVALID_SOCKET_VAL -1
#define CLOSE_SOCKET(s) close(s)
#endif

    native_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VAL) {
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    int port = -1;

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(0);

#ifdef _WIN32
    int namelen = sizeof(serv_addr);
#else
    socklen_t namelen = sizeof(serv_addr);
#endif
    if (bind(sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) == 0 &&
            getsockname(sock, (struct sockaddr *) &serv_addr, &namelen) == 0) {
        port = ntohs(serv_addr.sin_port);
    }

    CLOSE_SOCKET(sock);
#ifdef _WIN32
    WSACleanup();
#endif

    return port;
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }
    int len = WideCharToMultiByte(CP_UTF8, 0, ws, -1, nullptr, 0, nullptr, nullptr);
    std::string out(len > 0 ? len - 1 : 0, '\0');
    if (len > 1) {
        WideCharToMultiByte(CP_UTF8, 0, ws, -1, out.data(), len, nullptr, nullptr);
    }
    return out;
}
#endif

// same logic as get_environment() in tools/server/server-models.cpp
static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

// same logic as get_server_exec_path() in tools/server/server-models.cpp,
// but resolving the llama-server binary next to the current executable
static std::filesystem::path get_server_bin_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    std::filesystem::path self_path(buf);
    return self_path.parent_path() / "llama-server.exe";
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);
    std::filesystem::path self_path;
    if (_NSGetExecutablePath(small_path, &size) == 0) {
        self_path = std::filesystem::path(small_path);
    } else {
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) != 0) {
            throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
        }
        self_path = std::filesystem::path(buf.data());
    }
    try {
        self_path = std::filesystem::canonical(self_path);
    } catch (...) {
        // ignore, use the raw path
    }
    return self_path.parent_path() / "llama-server";
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    std::filesystem::path self_path(std::string(path, count));
    return self_path.parent_path() / "llama-server";
#endif
}

// helper to convert vector<string> to char **
// pointers are only valid as long as the original vector is valid
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char *>(s.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

cli_server::~cli_server() {
    stop();
}

bool cli_server::start(const std::vector<std::string> & args, bool pass_output_) {
    pass_output = pass_output_;

    std::filesystem::path bin_path;
    try {
        bin_path = get_server_bin_path();
    } catch (const std::exception & e) {
        last_error = e.what();
        return false;
    }

    std::error_code ec;
    if (!std::filesystem::exists(bin_path, ec)) {
        last_error = "llama-server binary not found at " + bin_path.string() +
                     "\nllama-cli requires llama-server to run a local model";
        return false;
    }

    port = get_free_port();
    if (port <= 0) {
        last_error = "failed to get a free port number";
        return false;
    }

    std::vector<std::string> child_args;
    child_args.push_back(bin_path.string());
    child_args.insert(child_args.end(), args.begin(), args.end());
    child_args.push_back("--host");
    child_args.push_back(CHILD_ADDR);
    child_args.push_back("--port");
    child_args.push_back(std::to_string(port));

    std::vector<std::string> child_env = get_environment();
    // make the server run in child mode: it will report readiness on stdout
    // and exit as soon as its stdin reaches EOF (the value is unused)
    child_env.push_back("LLAMA_SERVER_ROUTER_PORT=0");

    std::vector<char *> argv = to_char_ptr_array(child_args);
    std::vector<char *> envp = to_char_ptr_array(child_env);

    subproc = std::make_shared<subprocess_s>();
    int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
    if (subprocess_create_ex(argv.data(), options, envp.data(), subproc.get()) != 0) {
        last_error = "failed to spawn " + bin_path.string();
        subproc.reset();
        return false;
    }

    started     = true;
    child_stdin = subprocess_stdin(subproc.get());

    log_thread = std::thread([this]() {
        FILE * child_stdout = subprocess_stdout(subproc.get());
        std::vector<char> buf(128 * 1024);
        if (child_stdout) {
            while (fgets(buf.data(), buf.size(), child_stdout) != nullptr) {
                std::string line(buf.data());
                if (line_starts_with(line, CMD_CHILD_TO_ROUTER_READY)) {
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        ready.store(true);
                    }
                    cv.notify_all();
                } else if (!line_starts_with(line, CMD_CHILD_TO_ROUTER_PREFIX)) {
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        output_lines.push_back(line);
                        if (output_lines.size() > CLI_SERVER_MAX_LOG_LINES) {
                            output_lines.pop_front();
                        }
                    }
                    if (pass_output) {
                        fputs(line.c_str(), stderr);
                    }
                }
            }
        }
        // EOF means the child exited (or crashed)
        {
            std::lock_guard<std::mutex> lk(mtx);
            exited.store(true);
        }
        cv.notify_all();
    });

    return true;
}

bool cli_server::wait_ready(const std::function<bool()> & is_aborted) {
    std::unique_lock<std::mutex> lk(mtx);
    while (!ready.load() && !exited.load()) {
        if (is_aborted()) {
            return false;
        }
        cv.wait_for(lk, std::chrono::milliseconds(100));
    }
    return ready.load();
}

void cli_server::stop() {
    if (!started) {
        return;
    }

    if (!exited.load() && child_stdin != nullptr) {
        fprintf(child_stdin, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
        fflush(child_stdin);

        // wait for a graceful exit, force-kill after timeout
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait_for(lk, std::chrono::seconds(CLI_SERVER_STOP_TIMEOUT_SEC), [this]() {
            return exited.load();
        });
    }

    if (!exited.load()) {
        subprocess_terminate(subproc.get());
    }

    if (log_thread.joinable()) {
        log_thread.join();
    }

    int exit_code = 0;
    subprocess_join(subproc.get(), &exit_code);
    subprocess_destroy(subproc.get());

    started     = false;
    child_stdin = nullptr;
    subproc.reset();
}

std::string cli_server::address() const {
    return std::string("http://") + CHILD_ADDR + ":" + std::to_string(port);
}

std::string cli_server::recent_output() const {
    std::lock_guard<std::mutex> lk(mtx);
    std::string out;
    for (const auto & line : output_lines) {
        out += line;
    }
    return out;
}
