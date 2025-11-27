// run-chat.cpp - Console/chat mode functionality for llama-run
//
// This file contains the implementation of interactive chat mode and signal handling.

#include "run-chat.h"
#include "server-context.h"
#include "server-common.h"
#include "readline/readline.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <csignal>
#include <iostream>

using json = nlohmann::ordered_json;

#if defined(_WIN32)
#include <windows.h>
#endif

// Static globals for signal handling
static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, force terminating...\n");
        exit(1);
    }

    shutdown_handler(signal);
}

void setup_signal_handlers(std::function<void(int)> handler) {
    shutdown_handler = handler;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

void run_chat_mode(const common_params & params, server_context & ctx_server) {
    // Initialize readline
    readline::Prompt prompt_config;
    prompt_config.prompt = "> ";
    prompt_config.alt_prompt = ". ";
    prompt_config.placeholder = "Send a message";
    readline::Readline rl(prompt_config);
    rl.history_enable();

    // Initialize server routes
    server_routes routes(params, ctx_server);

    // Message history
    json messages = json::array();

    // Flag to check if we should stop (used by should_stop callback)
    std::atomic<bool> stop_requested = false;
    auto should_stop = [&]() { return stop_requested.load(); };

    while (true) {
        // Read user input
        std::string user_input;
        try {
            user_input = rl.readline();
        } catch (const readline::eof_error&) {
            printf("\n");
            break;
        } catch (const readline::interrupt_error&) {
            printf("\nUse Ctrl + d or /bye to exit.\n");
            continue;
        }

        if (user_input.empty()) {
            continue;
        }

        if (user_input == "/bye") {
            break;
        }

        // Add user message to history
        messages.push_back({
            {"role", "user"},
            {"content", user_input}
        });

        // Create request for chat completions endpoint
        server_http_req req{
            {}, {}, "",
            safe_json_to_str(json{
                {"messages", messages},
                {"stream", true}
            }),
            should_stop
        };

        // Reset stop flag
        stop_requested = false;

        // Call the chat completions endpoint
        auto res = routes.post_chat_completions(req);

        std::string curr_text;
        if (res->is_stream()) {
            std::string chunk;
            bool interrupted = false;

            while (res->next(chunk)) {
                // Check for interrupt (Ctrl-C) during streaming
                if (rl.check_interrupt()) {
                    printf("\n");
                    interrupted = true;
                    stop_requested = true;
                    break;
                }

                std::vector<std::string> lines = string_split<std::string>(chunk, '\n');
                for (auto & line : lines) {
                    if (line.empty()) {
                        continue;
                    }
                    if (line == "[DONE]") {
                        break;
                    }
                    std::string & data = line;
                    if (string_starts_with(line, "data: ")) {
                        data = line.substr(6);
                    }
                    try {
                        auto data_json = json::parse(data);
                        if (data_json.contains("choices") && !data_json["choices"].empty() &&
                                data_json["choices"][0].contains("delta") &&
                                data_json["choices"][0]["delta"].contains("content") &&
                                !data_json["choices"][0]["delta"]["content"].is_null()) {
                            std::string new_text = data_json["choices"][0]["delta"]["content"].get<std::string>();
                            curr_text += new_text;
                            std::cout << new_text << std::flush;
                        }
                    } catch (const std::exception & e) {
                        LOG_ERR("%s: error parsing JSON: %s\n", __func__, e.what());
                    }
                }
            }

            if (!interrupted) {
                std::cout << std::endl;
                if (!curr_text.empty()) {
                    messages.push_back({
                        {"role", "assistant"},
                        {"content", curr_text}
                    });
                }
            } else {
                // Remove the user message since generation was interrupted
                messages.erase(messages.end() - 1);
            }
        } else {
            std::cout << res->data << std::endl;
        }
    }

    LOG_INF("%s: exiting chat mode\n", __func__);
}
