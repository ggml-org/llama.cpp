// view layer for llama-cli (the "view" in MVC)
//
// the view owns all user-facing input/output; the controller only reports
// semantic events and data objects, and the view decides how to present
// them (styles, markers, spinners, layout); it knows nothing about HTTP,
// process management or chat state

#pragma once

#include "console.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct cli_timings {
    double prompt_per_second    = 0.0;
    double predicted_per_second = 0.0;
};

struct cli_command_info {
    std::string usage;       // e.g. "/read <file>"
    std::string description; // e.g. "add a text file"
};

// properties of the connected server, shown on startup
struct cli_server_info {
    std::string build_info;
    std::string model_name;
    std::string server_base;
    bool is_local_server   = false; // server is spawned and managed by llama-cli
    bool has_system_prompt = false;
    bool has_vision        = false;
    bool has_audio         = false;
    bool has_video         = false;

    std::vector<cli_command_info> commands;
};

struct cli_view {
    // returns matches as (replacement line, cursor position)
    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;

    virtual ~cli_view() = default;

    virtual void init(bool simple_io, bool use_color) = 0;
    virtual void cleanup() = 0;

    // input
    // read a line from the user; returns true if the input continues on another line
    virtual bool readline(std::string & line, bool multiline_input) = 0;
    virtual void set_completion_callback(completion_callback cb) = 0;

    // generic events
    virtual void show_loading(const std::string & message) = 0; // enter a busy state
    virtual void hide_loading() = 0;                            // leave the busy state
    virtual void show_banner(const cli_server_info & info) = 0;
    virtual void show_message(const std::string & text) = 0;    // discrete informational message
    virtual void show_error(const std::string & message) = 0;
    virtual void show_timings(const cli_timings & timings) = 0;

    // user input flow
    virtual void prompt_user() = 0;                       // interactive input starts
    virtual void echo_user(const std::string & text) = 0; // non-interactive input (e.g. from -p)
    virtual void end_user_input() = 0;                    // input finished, output follows

    // assistant response flow
    virtual void begin_response() = 0;                             // waiting for the first token
    virtual void on_reasoning_delta(const std::string & text) = 0; // streamed reasoning fragment
    virtual void on_content_delta(const std::string & text) = 0;   // streamed content fragment
    virtual void end_response() = 0;                               // response finished (or aborted)
};

// cli_view implementation backed by common/console
struct cli_view_console : cli_view {
    void init(bool simple_io, bool use_color) override;
    void cleanup() override;

    bool readline(std::string & line, bool multiline_input) override;
    void set_completion_callback(completion_callback cb) override;

    void show_loading(const std::string & message) override;
    void hide_loading() override;
    void show_banner(const cli_server_info & info) override;
    void show_message(const std::string & text) override;
    void show_error(const std::string & message) override;
    void show_timings(const cli_timings & timings) override;

    void prompt_user() override;
    void echo_user(const std::string & text) override;
    void end_user_input() override;

    void begin_response() override;
    void on_reasoning_delta(const std::string & text) override;
    void on_content_delta(const std::string & text) override;
    void end_response() override;

private:
    void set_display(display_type display);
    void stop_spinner();

    display_type curr_display = DISPLAY_TYPE_RESET;
    bool is_busy     = false; // a spinner is being shown
    bool is_thinking = false; // inside a streamed reasoning block
};
