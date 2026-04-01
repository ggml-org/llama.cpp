#pragma once

#include <cstdlib>
#include <string>

// Get the user config directory for llama-agent
inline std::string get_config_dir() {
#ifdef _WIN32
    const char * appdata = std::getenv("APPDATA");
    if (appdata) {
        return std::string(appdata) + "\\llama-agent";
    }
    return "";
#else
    const char * home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.llama-agent";
    }
    return "";
#endif
}
