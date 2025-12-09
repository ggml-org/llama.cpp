// Console functions

#pragma once

#include <string>
#include "log.h"

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
    display_t    get_display();
    const char * get_display_color();
    bool readline(std::string & line, bool multiline_input);

    void write_console(const char * format, ...);

    template<typename... Args>
    void write(const char * format, Args... args) {
        if (get_display() == user_input || !common_log_is_active(common_log_main())) {
            write_console(format, args...);

        } else {
            const char * color = get_display_color();
            std::string colored_format = std::string(color) + format + LOG_COL_DEFAULT;
            common_log_add(common_log_main(), GGML_LOG_LEVEL_CONT, colored_format.c_str(), args...);
        }
    }

    inline void write(const char * data) {
        write("%s", data);
    }

    inline void write(const std::string & data) {
        write("%s", data.c_str());
    }
}
