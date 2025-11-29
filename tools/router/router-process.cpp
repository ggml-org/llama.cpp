#include "router-constants.h"
#include "router-process.h"

#include "log.h"

#include <cpp-httplib/httplib.h>
#include <exception>
#include <chrono>
#include <sstream>
#include <thread>

#if !defined(_WIN32)
#    include <csignal>
#    include <fcntl.h>
#    include <sys/wait.h>
#    include <unistd.h>
#    include <vector>
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
    WaitForSingleObject(handle.proc_info.hProcess, 0);
    CloseHandle(handle.proc_info.hThread);
    CloseHandle(handle.proc_info.hProcess);
    handle.valid = false;
#else
    if (handle.pid > 0) {
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

ProcessHandle spawn_process(const std::vector<std::string> & args, const std::string & log_path) {
    ProcessHandle handle;
    if (args.empty()) {
        LOG_ERR("spawn_process called with empty args\n");
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

    STARTUPINFOA        si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);

    HANDLE              log_handle    = INVALID_HANDLE_VALUE;
    SECURITY_ATTRIBUTES sec_attrs{};
    sec_attrs.nLength              = sizeof(sec_attrs);
    sec_attrs.bInheritHandle       = TRUE;
    sec_attrs.lpSecurityDescriptor = nullptr;

    if (!log_path.empty()) {
        log_handle = CreateFileA(
            log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sec_attrs, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (log_handle != INVALID_HANDLE_VALUE) {
            si.dwFlags |= STARTF_USESTDHANDLES;
            si.hStdOutput = log_handle;
            si.hStdError  = log_handle;
        }
    }

    std::string cmd = cmdline.str();
    if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        handle.proc_info = pi;
        handle.valid     = true;
    }

    if (log_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(log_handle);
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

    pid_t pid = fork();
    if (pid == 0) {
        if (!log_path.empty()) {
            int fd = open(log_path.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
            if (fd >= 0) {
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                close(fd);
            }
        }

        std::vector<char *> cargs;
        cargs.reserve(args.size() + 1);
        for (const auto & arg : args) {
            cargs.push_back(const_cast<char *>(arg.c_str()));
        }
        cargs.push_back(nullptr);
        LOG_INF("Starting child process: %s\n", cargs[0]);
        execvp(cargs[0], cargs.data());
        _exit(1);
    } else if (pid > 0) {
        handle.pid = pid;
        LOG_INF("Spawned child pid=%d\n", static_cast<int>(pid));
    } else {
        LOG_ERR("fork failed while spawning %s\n", args[0].c_str());
    }
#endif

    return handle;
}

bool wait_for_backend_ready(int port, int timeout_ms) {
    httplib::Client client("127.0.0.1:" + std::to_string(port));
    const auto      start = std::chrono::steady_clock::now();

    while (true) {
        try {
            auto res = client.Get("/health");
            if (res && res->status == 200) {
                return true;
            }
        } catch (const std::exception & e) {
            LOG_DBG("Health check for port %d failed: %s\n", port, e.what());
        }

        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed_ms >= timeout_ms) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(ROUTER_BACKEND_HEALTH_POLL_MS));
    }

    return false;
}
