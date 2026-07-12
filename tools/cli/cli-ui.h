#pragma once

#include "common.h"
#include "console.h"

#include <array>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string_view>

// TODO?: Make this reusable, enums, docs
static const std::array<std::string_view, 8> cmds = {
    "/audio ",
    "/clear",
    "/exit",
    "/glob ",
    "/image ",
    "/read ",
    "/regen",
    "/video ",
};

static std::vector<std::pair<std::string, size_t>> auto_completion_callback(std::string_view line, size_t cursor_byte_pos) {
    std::vector<std::pair<std::string, size_t>> matches;
    std::string cmd;

    if (line.length() > 1 && line.front() == '/' && !std::any_of(cmds.begin(), cmds.end(), [line](std::string_view prefix) {
        return string_starts_with(line, prefix);
    })) {
        auto it = cmds.begin();

        while ((it = std::find_if(it, cmds.end(), [line](std::string_view cmd_line) {
            return string_starts_with(cmd_line, line);
        })) != cmds.end()) {
            matches.emplace_back(*it, it->length());
            ++it;
        }
    } else {
        auto it = std::find_if(cmds.begin(), cmds.end(), [line](std::string_view prefix) {
            return prefix.back() == ' ' && string_starts_with(line, prefix);
        });

        if (it != cmds.end()) {
            cmd = *it;
        }
    }

    if (!cmd.empty() && cmd != "/glob " && line.length() >= cmd.length() && cursor_byte_pos >= cmd.length()) {
        const std::string path_prefix  = std::string(line.substr(cmd.length(), cursor_byte_pos - cmd.length()));
        const std::string path_postfix = std::string(line.substr(cursor_byte_pos));
        auto cur_dir = std::filesystem::current_path();
        std::string cur_dir_str = cur_dir.string();
        std::string expanded_prefix = path_prefix;

#if !defined(_WIN32)
        if (string_starts_with(path_prefix, '~')) {
            const char * home = std::getenv("HOME");
            if (home && home[0]) {
                expanded_prefix = home + path_prefix.substr(1);
            }
        }
        if (string_starts_with(expanded_prefix, '/')) {
#else
        if (std::isalpha(static_cast<unsigned char>(expanded_prefix[0])) && expanded_prefix.find(':') == 1) {
#endif
            cur_dir = std::filesystem::path(expanded_prefix).parent_path();
            cur_dir_str.clear();
        } else if (!path_prefix.empty()) {
            cur_dir /= std::filesystem::path(path_prefix).parent_path();
        }

        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(cur_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.exists(ec)) {
                ec.clear();
                continue;
            }

            const std::string path_full = entry.path().string();
            std::string path_entry = !cur_dir_str.empty() && string_starts_with(path_full, cur_dir_str) ? path_full.substr(cur_dir_str.length() + 1) : path_full;

            if (entry.is_directory(ec)) {
                path_entry.push_back(std::filesystem::path::preferred_separator);
            }

            if (expanded_prefix.empty() || string_starts_with(path_entry, expanded_prefix)) {
                const std::string updated_line = cmd + path_entry;
                matches.emplace_back(updated_line + path_postfix, updated_line.length());
            }

            if (ec) {
                ec.clear();
            }
        }

        if (matches.empty()) {
            const std::string updated_line = cmd + path_prefix;
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        // Add the longest common prefix
        if (!expanded_prefix.empty() && matches.size() > 1) {
            const std::string_view match0(matches[0].first);
            const std::string_view match1(matches[1].first);
            auto it = std::mismatch(match0.begin(), match0.end(), match1.begin(), match1.end());
            size_t len = it.first - match0.begin();

            for (size_t i = 2; i < matches.size(); ++i) {
                const std::string_view matchi(matches[i].first);
                auto cmp = std::mismatch(match0.begin(), match0.end(), matchi.begin(), matchi.end());
                len = std::min(len, static_cast<size_t>(cmp.first - match0.begin()));
            }

            const std::string updated_line = std::string(match0.substr(0, len));
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a.first.compare(0, a.second, b.first, 0, b.second) < 0;
        });
    }

    return matches;
}

// note: make this view implementation generic, so that we can move to TUI in the future if we want to
namespace ui {
    static void init(const common_params & params) {
        // TODO: avoid using atexit() here by making `console` a singleton
        console::init(params.simple_io, params.use_color);
        atexit([]() { console::cleanup(); });

        console::set_completion_callback(auto_completion_callback);
    }

    struct spinner {
        spinner(const std::string & message) {
            if (!message.empty()) {
                console::log("%s ", message.c_str());
            }
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
        std::string read_input(bool multiline_input, const char * prompt = nullptr) {
            if (prompt) {
                console::log("%s", prompt);
            } else {
                console::log("\n> ");
            }
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
        bool trailing_newline = true;
        bool is_inside_reasoning = false;
        assistant_turn() {
            console::set_display(DISPLAY_TYPE_RESET);
        }
        ~assistant_turn() {
            console::set_display(DISPLAY_TYPE_RESET);
            add_newline_if_needed();
        }
        void push(assistant_display_mode m, const std::string & buffer) {
            if (m != mode) {
                add_newline_if_needed();
                switch (m) {
                    case ASSISTANT_DISPLAY_MODE_CONTENT:
                        {
                            if (is_inside_reasoning) {
                                console::log("[End thinking]\n\n");
                                is_inside_reasoning = false;
                            }
                            console::set_display(DISPLAY_TYPE_RESET);
                        } break;
                    case ASSISTANT_DISPLAY_MODE_REASONING:
                        {
                            console::set_display(DISPLAY_TYPE_REASONING);
                            is_inside_reasoning = true;
                            console::log("\n[Start thinking]\n\n");
                        } break;
                }
            }
            mode = m;
            if (buffer.empty()) {
                return;
            }
            trailing_newline = buffer.back() == '\n';
            console::log("%s", buffer.c_str());
            console::flush();
        }
        void add_newline_if_needed() {
            if (!trailing_newline) {
                console::log("\n");
                console::flush();
            }
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

    static void show_info(const std::string & message) {
        console::set_display(DISPLAY_TYPE_INFO);
        console::log("%s\n", message.c_str());
        console::set_display(DISPLAY_TYPE_RESET);
    }
}
