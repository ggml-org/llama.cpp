#pragma once

#include <string>
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
#else
    pid_t pid = -1;
#endif
};

bool         process_running(const ProcessHandle & handle);
void         close_process(ProcessHandle & handle);
void         terminate_process(ProcessHandle & handle);
ProcessHandle spawn_process(const std::vector<std::string> & args);
