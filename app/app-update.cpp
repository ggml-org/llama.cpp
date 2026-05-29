#include "app-update.h"

#include <cstdio>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cstdlib>
#endif

#ifndef LLAMA_APP_ID_BINARY
#define LLAMA_APP_ID_BINARY ""
#endif

#ifndef LLAMA_APP_SCRIPT_URL
#define LLAMA_APP_SCRIPT_URL "https://llama.app"
#endif

int llama_update(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    if (std::string(LLAMA_APP_ID_BINARY).empty()) {
        printf("update: this build has no release channel configured\n");
        return 0;
    }

#if defined(_WIN32)
    // the installer swaps llama.exe, so this process must release its lock first
    // spawn it detached and return at once, exiting frees the binary before the swap
    char cmd[] = "powershell -NoProfile -Command \"irm " LLAMA_APP_SCRIPT_URL "/install.ps1 | iex\"";
    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi)) {
        printf("update: cannot start the installer\n");
        return 1;
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    printf("update: installer running in a new window, llama exits to free the binary\n");
    return 0;
#else
    // posix replaces a running binary fine, the installer runs in place
    return system("curl -fsSL " LLAMA_APP_SCRIPT_URL "/install.sh | sh");
#endif
}
