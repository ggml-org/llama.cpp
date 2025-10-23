// Console functions

#pragma once

#include <string>
#include <cstdio>

namespace console {
    enum display_t {
        reset = 0,
        prompt,
        user_input,
        error,
        reasoning
    };

    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_t display);
    bool readline(std::string & line, bool multiline_input);

    FILE* get_output_handle();

    template<typename... Args>
    void write(const char* format, Args... args) {
        FILE* out = get_output_handle();
        fprintf(out, format, args...);
        fflush(out);
    }

    inline void write(const char* str) {
        write("%s", str);
    }

    inline void write(const std::string & data) {
        write(data.c_str());
    }
}
