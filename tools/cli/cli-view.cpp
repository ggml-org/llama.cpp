#include "cli-view.h"

#include <algorithm>

static const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

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

void cli_view_console::stop_spinner() {
    if (is_busy) {
        is_busy = false;
        console::spinner::stop();
    }
}

void cli_view_console::show_loading(const std::string & message) {
    set_display(DISPLAY_TYPE_RESET);
    console::log("%s... ", message.c_str());
    if (!is_busy) {
        is_busy = true;
        console::spinner::start();
    }
}

void cli_view_console::hide_loading() {
    stop_spinner();
    console::log("\n");
}

void cli_view_console::show_banner(const cli_server_info & info) {
    set_display(DISPLAY_TYPE_RESET);
    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    if (!info.build_info.empty()) {
        console::log("build      : %s\n", info.build_info.c_str());
    }
    console::log("model      : %s\n", info.model_name.empty() ? "(unknown)" : info.model_name.c_str());
    console::log("server     : %s%s\n", info.server_base.c_str(), info.is_local_server ? " (managed by llama-cli)" : "");

    std::string modalities = "text";
    if (info.has_vision) {
        modalities += ", vision";
    }
    if (info.has_audio) {
        modalities += ", audio";
    }
    if (info.has_video) {
        modalities += ", video";
    }
    console::log("modalities : %s\n", modalities.c_str());

    if (info.has_system_prompt) {
        console::log("using custom system prompt\n");
    }

    if (!info.commands.empty()) {
        size_t width = 0;
        for (const auto & cmd : info.commands) {
            width = std::max(width, cmd.usage.size());
        }
        console::log("\n");
        console::log("available commands:\n");
        for (const auto & cmd : info.commands) {
            console::log("  %-*s    %s\n", (int) width, cmd.usage.c_str(), cmd.description.c_str());
        }
    }
    console::log("\n");
}

void cli_view_console::show_message(const std::string & text) {
    if (is_busy) {
        // break the pending loading line
        stop_spinner();
        console::log("\n");
    }
    set_display(DISPLAY_TYPE_RESET);
    console::log("%s\n", text.c_str());
}

void cli_view_console::show_error(const std::string & message) {
    if (is_busy) {
        // break the pending loading line
        stop_spinner();
        console::log("\n");
    }
    console::error("%s\n", message.c_str()); // restores the current display on its own
}

void cli_view_console::show_timings(const cli_timings & timings) {
    set_display(DISPLAY_TYPE_INFO);
    console::log("\n[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n",
            timings.prompt_per_second, timings.predicted_per_second);
    set_display(DISPLAY_TYPE_RESET);
}

void cli_view_console::prompt_user() {
    set_display(DISPLAY_TYPE_USER_INPUT);
    console::log("\n> ");
}

void cli_view_console::echo_user(const std::string & text) {
    static constexpr size_t MAX_ECHO_LENGTH = 500;
    set_display(DISPLAY_TYPE_USER_INPUT);
    if (text.size() > MAX_ECHO_LENGTH) {
        console::log("\n> %s ... (truncated)\n", text.substr(0, MAX_ECHO_LENGTH).c_str());
    } else {
        console::log("\n> %s\n", text.c_str());
    }
}

void cli_view_console::end_user_input() {
    set_display(DISPLAY_TYPE_RESET);
    console::log("\n");
}

void cli_view_console::begin_response() {
    if (!is_busy) {
        is_busy = true;
        console::spinner::start();
    }
}

void cli_view_console::on_reasoning_delta(const std::string & text) {
    stop_spinner();
    set_display(DISPLAY_TYPE_REASONING);
    if (!is_thinking) {
        is_thinking = true;
        console::log("[Start thinking]\n");
    }
    console::log("%s", text.c_str());
    console::flush();
}

void cli_view_console::on_content_delta(const std::string & text) {
    stop_spinner();
    if (is_thinking) {
        is_thinking = false;
        set_display(DISPLAY_TYPE_REASONING);
        console::log("\n[End thinking]\n\n");
    }
    set_display(DISPLAY_TYPE_RESET);
    console::log("%s", text.c_str());
    console::flush();
}

void cli_view_console::end_response() {
    stop_spinner();
    is_thinking = false;
    set_display(DISPLAY_TYPE_RESET);
    console::log("\n");
}
