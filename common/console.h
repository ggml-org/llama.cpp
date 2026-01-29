// Console functions

#pragma once

#include "common.h"

#include <memory>
#include <string>

enum display_type {
    DISPLAY_TYPE_RESET = 0,
    DISPLAY_TYPE_INFO,
    DISPLAY_TYPE_PROMPT,
    DISPLAY_TYPE_REASONING,
    DISPLAY_TYPE_USER_INPUT,
    DISPLAY_TYPE_ERROR
};

class console {
private:
    console();

public:
    ~console();

    static void init(bool use_simple_io, bool use_advanced_display);
    static void cleanup();

    static void set_display(display_type display);
    static bool readline(std::string & line, bool multiline_input);
    static void start();
    static void stop();


    // note: the logging API below output directly to stdout
    // it can negatively impact performance if used on inference thread
    // only use in in a dedicated CLI thread
    // for logging in inference thread, use log.h instead

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    static void log(const char * fmt, ...);

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    static void error(const char * fmt, ...);

    static void flush();

private:
    static std::unique_ptr<console> instance;
};
