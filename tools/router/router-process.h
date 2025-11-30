#pragma once

#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/types.h>
#endif

struct ProcessHandle {
#if defined(_WIN32)
    PROCESS_INFORMATION proc_info{};
    bool valid = false;
    HANDLE stdout_read = nullptr;
    HANDLE stderr_read = nullptr;
    std::thread stdout_thread;
    std::thread stderr_thread;
#else
    pid_t pid = -1;
    int  stdout_fd = -1;
    int  stderr_fd = -1;
    std::thread stdout_thread;
    std::thread stderr_thread;
#endif

    ProcessHandle() = default;
    ProcessHandle(const ProcessHandle &) = delete;
    ProcessHandle & operator=(const ProcessHandle &) = delete;
    ProcessHandle(ProcessHandle &&) = default;
    ProcessHandle & operator=(ProcessHandle &&) = default;
};

bool         process_running(const ProcessHandle & handle);
void         close_process(ProcessHandle & handle);
void         terminate_process(ProcessHandle & handle);
bool         wait_for_process_exit(const ProcessHandle & handle, int timeout_ms);
ProcessHandle spawn_process(const std::vector<std::string> & args);
bool         wait_for_backend_ready(int port,
                                    const std::string & health_endpoint,
                                    int timeout_ms,
                                    const ProcessHandle * process = nullptr);
