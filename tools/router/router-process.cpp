#include "router-constants.h"
#include "router-process.h"

#include "log.h"

#include <cpp-httplib/httplib.h>
#include <chrono>
#include <cstdio>
#include <cerrno>
#include <filesystem>
#include <cstring>
#include <exception>
#include <sstream>
#include <thread>

#if !defined(_WIN32)
#    include <csignal>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

bool process_running(const ProcessHandle & handle) {
#if defined(_WIN32)
    if (!handle.valid) {
        return false;
    }
    DWORD status = WaitForSingleObject(handle.proc_info.hProcess, 0);
    return status == WAIT_TIMEOUT;
#else
    if (handle.pid <= 0) {
        return false;
    }

    int  status = 0;
    auto r      = waitpid(handle.pid, &status, WNOHANG);
    if (r == handle.pid) {
        return false;
    }

    return kill(handle.pid, 0) == 0;
#endif
}

bool wait_for_process_exit(const ProcessHandle & handle, int timeout_ms) {
#if defined(_WIN32)
    if (!handle.valid) {
        return true;
    }
    return WaitForSingleObject(handle.proc_info.hProcess, timeout_ms) == WAIT_OBJECT_0;
#else
    if (handle.pid <= 0) {
        return true;
    }

    const auto start = std::chrono::steady_clock::now();
    while (true) {
        int  status = 0;
        auto r      = waitpid(handle.pid, &status, WNOHANG);
        if (r == handle.pid) {
            return true;
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(ROUTER_POLL_INTERVAL_MS));
    }
#endif
}

void close_process(ProcessHandle & handle) {
#if defined(_WIN32)
    if (!handle.valid) {
        return;
    }
    if (handle.stdout_thread.joinable()) {
        handle.stdout_thread.join();
    }
    if (handle.stderr_thread.joinable()) {
        handle.stderr_thread.join();
    }
    if (handle.stdout_read != nullptr) {
        CloseHandle(handle.stdout_read);
        handle.stdout_read = nullptr;
    }
    if (handle.stderr_read != nullptr) {
        CloseHandle(handle.stderr_read);
        handle.stderr_read = nullptr;
    }
    WaitForSingleObject(handle.proc_info.hProcess, 0);
    CloseHandle(handle.proc_info.hThread);
    CloseHandle(handle.proc_info.hProcess);
    handle.valid = false;
#else
    if (handle.pid > 0) {
        if (handle.stdout_thread.joinable()) {
            handle.stdout_thread.join();
        }
        if (handle.stderr_thread.joinable()) {
            handle.stderr_thread.join();
        }
        if (handle.stdout_fd != -1) {
            close(handle.stdout_fd);
            handle.stdout_fd = -1;
        }
        if (handle.stderr_fd != -1) {
            close(handle.stderr_fd);
            handle.stderr_fd = -1;
        }

        LOG_DBG("Closing process pid=%d\n", static_cast<int>(handle.pid));
        int status = 0;
        waitpid(handle.pid, &status, WNOHANG);
        handle.pid = -1;
    }
#endif
}

void terminate_process(ProcessHandle & handle) {
#if defined(_WIN32)
    if (!handle.valid) {
        return;
    }
    TerminateProcess(handle.proc_info.hProcess, 1);
    close_process(handle);
#else
    if (handle.pid <= 0) {
        return;
    }
    LOG_WRN("Sending SIGTERM to pid=%d\n", static_cast<int>(handle.pid));
    kill(handle.pid, SIGTERM);
    if (!wait_for_process_exit(handle, ROUTER_PROCESS_SHUTDOWN_TIMEOUT_MS)) {
        LOG_ERR("Process pid=%d did not terminate, sending SIGKILL\n", static_cast<int>(handle.pid));
        kill(handle.pid, SIGKILL);
        wait_for_process_exit(handle, 1000);
    }
    close_process(handle);
#endif
}

ProcessHandle spawn_process(const std::vector<std::string> & args) {
    ProcessHandle handle;
    if (args.empty()) {
        LOG_ERR("spawn_process called with empty args\n");
        return handle;
    }

    const std::string binary = args[0];
    std::error_code   ec;
    if (!std::filesystem::exists(binary, ec)) {
        LOG_ERR("Binary not found: %s\n", binary.c_str());
        return handle;
    }

#if defined(_WIN32)
    std::ostringstream cmdline;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            cmdline << ' ';
        }
        const std::string & part = args[i];
        if (part.find(' ') != std::string::npos) {
            cmdline << '"' << part << '"';
        } else {
            cmdline << part;
        }
    }

    LOG_INF("Spawn command: %s\n", cmdline.str().c_str());

    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(sa);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE stdout_read = nullptr, stdout_write = nullptr;
    HANDLE stderr_read = nullptr, stderr_write = nullptr;

    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) || !CreatePipe(&stderr_read, &stderr_write, &sa, 0)) {
        LOG_ERR("pipe creation failed while spawning %s\n", args[0].c_str());
        if (stdout_read) CloseHandle(stdout_read);
        if (stdout_write) CloseHandle(stdout_write);
        if (stderr_read) CloseHandle(stderr_read);
        if (stderr_write) CloseHandle(stderr_write);
        return handle;
    }

    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA        si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    si.dwFlags |= STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError  = stderr_write;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

    std::string cmd = cmdline.str();
    if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(stdout_write);
        CloseHandle(stderr_write);

        handle.proc_info   = pi;
        handle.valid       = true;
        handle.stdout_read = stdout_read;
        handle.stderr_read = stderr_read;

        handle.stdout_thread = std::thread([fd = handle.stdout_read]() {
            HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
            char   buf[4096];
            DWORD  read = 0, written = 0;
            while (ReadFile(fd, buf, sizeof(buf), &read, nullptr) && read > 0) {
                WriteFile(hStdout, buf, read, &written, nullptr);
            }
        });

        handle.stderr_thread = std::thread([fd = handle.stderr_read]() {
            HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
            char   buf[4096];
            DWORD  read = 0, written = 0;
            while (ReadFile(fd, buf, sizeof(buf), &read, nullptr) && read > 0) {
                WriteFile(hStderr, buf, read, &written, nullptr);
            }
        });
    } else {
        CloseHandle(stdout_read);
        CloseHandle(stdout_write);
        CloseHandle(stderr_read);
        CloseHandle(stderr_write);
    }

#else
    std::ostringstream cmd_str;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            cmd_str << ' ';
        }
        const std::string & part = args[i];
        if (part.find(' ') != std::string::npos) {
            cmd_str << '"' << part << '"';
        } else {
            cmd_str << part;
        }
    }

    LOG_INF("Spawn command: %s\n", cmd_str.str().c_str());

    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};

    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        LOG_ERR("pipe creation failed while spawning %s\n", args[0].c_str());
        if (stdout_pipe[0] != -1) {
            close(stdout_pipe[0]);
        }
        if (stdout_pipe[1] != -1) {
            close(stdout_pipe[1]);
        }
        if (stderr_pipe[0] != -1) {
            close(stderr_pipe[0]);
        }
        if (stderr_pipe[1] != -1) {
            close(stderr_pipe[1]);
        }
        return handle;
    }

    pid_t pid = fork();
    if (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::vector<char *> cargs;
        cargs.reserve(args.size() + 1);
        for (const auto & arg : args) {
            cargs.push_back(const_cast<char *>(arg.c_str()));
        }
        cargs.push_back(nullptr);
        execvp(cargs[0], cargs.data());
        _exit(1);
    } else if (pid > 0) {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        handle.pid       = pid;
        handle.stdout_fd = stdout_pipe[0];
        handle.stderr_fd = stderr_pipe[0];

        handle.stdout_thread = std::thread([fd = handle.stdout_fd]() {
            char    buf[4096];
            ssize_t n;
            while ((n = read(fd, buf, sizeof(buf))) > 0) {
                write(STDOUT_FILENO, buf, n);
            }
        });

        handle.stderr_thread = std::thread([fd = handle.stderr_fd]() {
            char    buf[4096];
            ssize_t n;
            while ((n = read(fd, buf, sizeof(buf))) > 0) {
                write(STDERR_FILENO, buf, n);
            }
        });

        LOG_INF("Spawned child pid=%d\n", static_cast<int>(pid));
    } else {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        LOG_ERR("fork failed while spawning %s\n", args[0].c_str());
    }
#endif

    return handle;
}

bool wait_for_backend_ready(int port, const std::string & health_endpoint, int timeout_ms, const ProcessHandle * process) {
    httplib::Client client("127.0.0.1:" + std::to_string(port));
    const auto      start = std::chrono::steady_clock::now();
    auto            next_log_ms = 0;

    const std::string endpoint = health_endpoint.empty() ? "/health" : health_endpoint;

    LOG_INF("Waiting up to %d ms for backend readiness on port %d (endpoint %s)\n",
            timeout_ms,
            port,
            endpoint.c_str());

    while (true) {
        try {
            auto res = client.Get(endpoint.c_str());
            if (res && res->status == 200) {
                LOG_INF("Backend on port %d reports ready\n", port);
                return true;
            }
            LOG_DBG("Health check on port %d returned status %d\n", port, res ? res->status : -1);
        } catch (const std::exception & e) {
            LOG_DBG("Health check for port %d failed: %s\n", port, e.what());
        }

        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed_ms >= timeout_ms) {
            break;
        }

        if (process && !process_running(*process)) {
            LOG_ERR("Backend process for port %d exited after %lld ms while waiting for readiness\n",
                    port,
                    static_cast<long long>(elapsed_ms));
            return false;
        }

        if (elapsed_ms >= next_log_ms) {
            LOG_INF("Still waiting for backend on port %d (elapsed %lld ms)\n", port, static_cast<long long>(elapsed_ms));
            next_log_ms += 1000;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(ROUTER_BACKEND_HEALTH_POLL_MS));
    }

    return false;
}
