// view layer for llama-cli (the "view" in MVC)
//
// the view owns all user-facing input/output; it knows nothing about HTTP,
// process management or chat state

#pragma once

#include "console.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct cli_view {
    // returns matches as (replacement line, cursor position)
    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;

    virtual ~cli_view() = default;

    virtual void init(bool simple_io, bool use_color) = 0;
    virtual void cleanup() = 0;

    // read a line from the user; returns true if the input continues on another line
    virtual bool readline(std::string & line, bool multiline_input) = 0;
    virtual void set_completion_callback(completion_callback cb) = 0;

    virtual void print(const std::string & text) = 0;           // assistant / generic output
    virtual void print_reasoning(const std::string & text) = 0; // reasoning stream
    virtual void print_info(const std::string & text) = 0;      // metadata (banner, timings)
    virtual void print_user(const std::string & text) = 0;      // user input marker / echo
    virtual void print_error(const std::string & text) = 0;

    virtual void spinner_start() = 0;
    virtual void spinner_stop() = 0;
    virtual void flush() = 0;
};

// cli_view implementation backed by common/console
struct cli_view_console : cli_view {
    void init(bool simple_io, bool use_color) override;
    void cleanup() override;

    bool readline(std::string & line, bool multiline_input) override;
    void set_completion_callback(completion_callback cb) override;

    void print(const std::string & text) override;
    void print_reasoning(const std::string & text) override;
    void print_info(const std::string & text) override;
    void print_user(const std::string & text) override;
    void print_error(const std::string & text) override;

    void spinner_start() override;
    void spinner_stop() override;
    void flush() override;

private:
    void set_display(display_type display);

    display_type curr_display = DISPLAY_TYPE_RESET;
};
