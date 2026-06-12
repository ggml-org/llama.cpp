#include "cli-view.h"

void cli_view_console::init(bool simple_io, bool use_color) {
    console::init(simple_io, use_color);
    console::set_display(DISPLAY_TYPE_RESET);
    curr_display = DISPLAY_TYPE_RESET;
}

void cli_view_console::cleanup() {
    console::cleanup();
}

bool cli_view_console::readline(std::string & line, bool multiline_input) {
    return console::readline(line, multiline_input);
}

void cli_view_console::set_completion_callback(completion_callback cb) {
    console::set_completion_callback(std::move(cb));
}

void cli_view_console::set_display(display_type display) {
    if (curr_display != display) {
        console::set_display(display);
        curr_display = display;
    }
}

void cli_view_console::print(const std::string & text) {
    set_display(DISPLAY_TYPE_RESET);
    console::log("%s", text.c_str());
}

void cli_view_console::print_reasoning(const std::string & text) {
    set_display(DISPLAY_TYPE_REASONING);
    console::log("%s", text.c_str());
}

void cli_view_console::print_info(const std::string & text) {
    set_display(DISPLAY_TYPE_INFO);
    console::log("%s", text.c_str());
    set_display(DISPLAY_TYPE_RESET);
}

void cli_view_console::print_user(const std::string & text) {
    set_display(DISPLAY_TYPE_USER_INPUT);
    console::log("%s", text.c_str());
}

void cli_view_console::print_error(const std::string & text) {
    console::error("%s", text.c_str()); // restores the current display on its own
}

void cli_view_console::spinner_start() {
    console::spinner::start();
}

void cli_view_console::spinner_stop() {
    console::spinner::stop();
}

void cli_view_console::flush() {
    console::flush();
}
