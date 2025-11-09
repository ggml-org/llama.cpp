// Console functions

#pragma once

#include <string>

namespace console {
    enum display_t {
        reset = 0,
        prompt,
        user_input,
        error
    };

    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_t display);
    bool readline(std::string & line, bool multiline_input);
    bool readline_with_timeout(std::string & line, bool multiline_input, int timeout_seconds, bool & timed_out);
}
