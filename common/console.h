// Console functions

#pragma once

#include "common.h"

#include <functional>
#include <string>
#include <vector>

enum display_type {
    DISPLAY_TYPE_RESET = 0,
    DISPLAY_TYPE_INFO,
    DISPLAY_TYPE_PROMPT,
    DISPLAY_TYPE_REASONING,
    DISPLAY_TYPE_USER_INPUT,
    DISPLAY_TYPE_ERROR
};

namespace console {
    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_type display);
    bool readline(std::string & line, bool multiline_input);

    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;
    void set_completion_callback(completion_callback cb);

    namespace spinner {
        void start();
        void stop();
    }

    // Enable/disable quiet mode. When quiet, log() is suppressed but output() still prints.
    void set_quiet(bool quiet);

    // note: the logging API below output directly to stdout
    // it can negatively impact performance if used on inference thread
    // only use in in a dedicated CLI thread
    // for logging in inference thread, use log.h instead

    // Informational/chrome messages (banner, prompts, status).
    // Suppressed when quiet mode is enabled via set_quiet(true).
    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void log(const char * fmt, ...);

    // Model-generated output. Always printed regardless of quiet mode.
    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void output(const char * fmt, ...);

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void error(const char * fmt, ...);

    void flush();
}
