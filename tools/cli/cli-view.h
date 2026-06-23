#pragma once

#include "common.h"
#include "console.h"

// note: make this view implementation generic, so that we can move to TUI in the future if we want to
namespace view {
    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;

    static void set_completion_callback(completion_callback cb) {
        console::set_completion_callback(std::move(cb));
    }

    static void init(const common_params & params) {
        // TODO: avoid using atexit() here by making `console` a singleton
        console::init(params.simple_io, params.use_color);
        atexit([]() { console::cleanup(); });
    }

    struct spinner {
        spinner(const std::string & message) {
            console::log("%s\n", message.c_str());
            console::spinner::start();
        }
        ~spinner() {
            console::spinner::stop();
        }
    };

    struct user_turn {
        user_turn() {
            console::set_display(DISPLAY_TYPE_USER_INPUT);
        }
        ~user_turn() {
            console::set_display(DISPLAY_TYPE_RESET);
        }
        void echo(const std::string & buffer) {
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
        }
        std::string read_input(bool multiline_input) {
            console::log("\n> ");
            std::string buffer;
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, multiline_input);
                buffer += line;
            } while (another_line);
            return buffer;
        }
    };

    enum assistant_display_mode {
        ASSISTANT_DISPLAY_MODE_REASONING,
        ASSISTANT_DISPLAY_MODE_CONTENT,
    };
    struct assistant_turn {
        assistant_display_mode mode = ASSISTANT_DISPLAY_MODE_CONTENT;
        assistant_turn() {
            console::set_display(DISPLAY_TYPE_RESET);
        }
        ~assistant_turn() {
            console::set_display(DISPLAY_TYPE_RESET);
        }
        void push(assistant_display_mode m, const std::string & buffer) {
            if (m != mode) {
                switch (m) {
                    case ASSISTANT_DISPLAY_MODE_CONTENT:
                        console::set_display(DISPLAY_TYPE_RESET);
                        break;
                    case ASSISTANT_DISPLAY_MODE_REASONING:
                        console::set_display(DISPLAY_TYPE_REASONING);
                        break;
                }
            }
            mode = m;
            console::log("%s", buffer.c_str());
            console::flush();
        }
    };

    static void show_error(const std::string & title, const std::string & message = "") {
        console::spinner::stop();
        console::error("Error: %s\n", title.c_str());
        if (!message.empty()) {
            console::log("%s\n", message.c_str());
        }
    }

    static void show_message(const std::string & message) {
        console::log("%s\n", message.c_str());
    }

    static void show_banner(const std::vector<std::string> & lines) {
        for (const auto & line : lines) {
            console::log("%s\n", line.c_str());
        }
    }
};
