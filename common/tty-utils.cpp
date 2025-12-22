#include "tty-utils.h"

#include <unistd.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace common_tty_utils {
    bool is_stdin_a_terminal() {
    #if defined(_WIN32)
        HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
        if (hStdin == INVALID_HANDLE_VALUE) {
            return false;
        }
        DWORD mode;
        return GetConsoleMode(hStdin, &mode);
    #else
        return isatty(STDIN_FILENO);
    #endif
    }

    bool is_stdout_a_terminal() {
    #if defined(_WIN32)
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hStdin == INVALID_HANDLE_VALUE) {
            return false;
        }
        DWORD mode;
        return GetConsoleMode(hStdout, &mode);
    #else
        return isatty(STDOUT_FILENO);
    #endif
    }

    bool is_stderr_a_terminal() {
    #if defined(_WIN32)
        HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
        if (hStdin == INVALID_HANDLE_VALUE) {
            return false;
        }
        DWORD mode;
        return GetConsoleMode(hStderr, &mode);
    #else
        return isatty(STDERR_FILENO);
    #endif
    }
} // namespace common_tty_utils
