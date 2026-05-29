#include "app-update.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef LLAMA_APP_ID_BINARY
#define LLAMA_APP_ID_BINARY ""
#endif

#ifndef LLAMA_APP_SCRIPT_URL
#define LLAMA_APP_SCRIPT_URL "https://llama.app"
#endif

// extracts the backend segment from the baked variant id "arch/os/backend/config"
static std::string backend_of(const std::string & id) {
    size_t a = id.find('/');
    if (a == std::string::npos) {
        return std::string();
    }
    size_t b = id.find('/', a + 1);
    if (b == std::string::npos) {
        return std::string();
    }
    size_t c = id.find('/', b + 1);
    if (c == std::string::npos) {
        return id.substr(b + 1);
    }
    return id.substr(b + 1, c - b - 1);
}

int llama_update(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    const std::string id = LLAMA_APP_ID_BINARY;
    if (id.empty()) {
        printf("update: this build has no release channel configured\n");
        return 0;
    }

    // pin the backend, the install script re-detects the hardware config which is stable
    const std::string backend = backend_of(id);
    if (!backend.empty()) {
        std::string force = "FORCE_";
        for (char ch : backend) {
            force += (char) toupper((unsigned char) ch);
        }
#if defined(_WIN32)
        _putenv_s(force.c_str(), "1");
#else
        setenv(force.c_str(), "1", 1);
#endif
    }

    // hand over to the install script, it owns download, decompress and the in place swap
#if defined(_WIN32)
    const char * cmd = "powershell -NoProfile -Command \"irm " LLAMA_APP_SCRIPT_URL "/install.ps1 | iex\"";
#else
    const char * cmd = "curl -fsSL " LLAMA_APP_SCRIPT_URL "/install.sh | sh";
#endif
    return system(cmd);
}
