#include "router-process.h"

#include "log.h"

#include <chrono>
#include <sstream>
#include <thread>

#if !defined(_WIN32)
#    include <csignal>
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

#if !defined(_WIN32)
static bool wait_process(pid_t pid, int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        int  status = 0;
        auto r      = waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            return true;
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
#endif

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
    if (!wait_process(handle.pid, 1000)) {
        LOG_ERR("Process pid=%d did not terminate, sending SIGKILL\n", static_cast<int>(handle.pid));
        kill(handle.pid, SIGKILL);
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

    std::string cmd = cmdline.str();
    if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        handle.proc_info = pi;
        handle.valid     = true;
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
