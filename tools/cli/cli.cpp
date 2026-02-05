#include "arg.h"
#include "chat.h"
#include "common.h"
#include "console.h"
// #include "log.h"

#include "mcp.hpp"
#include "server-context.h"
#include "server-task.h"

#include <signal.h>

#include <atomic>
#include <fstream>
#include <thread>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

static const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

static std::atomic<bool> g_is_interrupted = false;

static bool should_stop() {
    return g_is_interrupted.load();
}

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
static void signal_handler(int /*unused*/) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

struct cli_context {
    server_context          ctx_server;
    json                    messages = json::array();
    json                    tools    = json::array();
    std::vector<raw_buffer> input_files;
    task_params             defaults;
    std::set<std::string>   approved_requests;

    // thread for showing "loading" animation
    std::atomic<bool> loading_show;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;

        defaults.stream                                 = true;  // make sure we always use streaming mode
        defaults.timings_per_token                      = true;  // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress
    }

    std::string generate_completion(result_timings & out_timings, server_task_result_ptr & out_result) {
        server_response_reader rd = ctx_server.get_response_reader();
        auto chat_params = format_chat();
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;           // copy
            task.cli_prompt = chat_params.prompt; // copy
            task.cli_files  = input_files;        // copy
            task.cli        = true;

            // chat template settings
            task.params.chat_parser_params = common_chat_parser_params(chat_params);
            task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            if (!chat_params.parser.empty()) {
                task.params.chat_parser_params.parser.load(chat_params.parser);
            }

            rd.post_task({ std::move(task) });
        }

        // wait for first result
        console::spinner::start();
        server_task_result_ptr result = rd.next(should_stop);

        console::spinner::stop();
        std::string curr_content;
        bool        is_thinking = false;

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    console::error("Error: %s\n", err_data.dump().c_str());
                }
                out_result = std::move(result);
                return curr_content;
            }
            auto *res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = res_partial->timings;
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (is_thinking) {
                            console::log("\n[End thinking]\n\n");
                            console::set_display(DISPLAY_TYPE_RESET);
                            is_thinking = false;
                        }
                        curr_content += diff.content_delta;
                        console::log("%s", diff.content_delta.c_str());
                        console::flush();
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        console::set_display(DISPLAY_TYPE_REASONING);
                        if (!is_thinking) {
                            console::log("[Start thinking]\n");
                        }
                        is_thinking = true;
                        console::log("%s", diff.reasoning_content_delta.c_str());
                        console::flush();
                    }
                }
            }
            auto *res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = res_final->timings;
                out_result  = std::move(result);
                break;
            }
            result = rd.next(should_stop);
        }
        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return mtmd_default_marker();
        } 
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

    common_chat_params format_chat() const {
        auto meta = ctx_server.get_meta();
        auto & chat_params = meta.chat_params;

        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools                 = common_chat_tools_parse_oaicompat(tools);
        inputs.tool_choice           = tools.empty() ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_AUTO;
        inputs.json_schema           = ""; // TODO
        inputs.grammar               = ""; // TODO
        inputs.use_jinja             = chat_params.use_jinja;
        inputs.parallel_tool_calls   = false;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = chat_params.enable_thinking;

        // Apply chat template to the list of messages
        return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
    }
};

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR;  // by default, less verbose logs

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    // TODO: maybe support it later?
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        console::error("--no-conversation is not supported by llama-cli\n");
        console::error("please use llama-completion instead\n");
    }

    common_init();

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: avoid using atexit() here by making `console` a singleton
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // Initialize MCP
    mcp_context mcp;
    if (!params.mcp_config.empty()) {
        if (mcp.load_config(params.mcp_config, params.mcp_servers)) {
            mcp.set_yolo(params.mcp_yolo);
            auto mcp_tools = mcp.get_tools();
            for (const auto & t : mcp_tools) {
                json tool = {
                    { "type",     "function"                                                                    },
                    { "function",
                     { { "name", t.name }, { "description", t.description }, { "parameters", t.input_schema } } }
                };
                ctx_cli.tools.push_back(tool);
            }
            if (!ctx_cli.tools.empty()) {
                console::log("Enabled %d MCP tools\n", (int) ctx_cli.tools.size());
            }
        } else {
            console::error("Failed to load MCP config: %s\n", params.mcp_config.c_str());
        }
    }

    console::log("\nLoading model... ");  // followed by loading animation
    console::spinner::start();
    if (!ctx_cli.ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    console::spinner::stop();
    console::log("\n");

    std::thread inference_thread([&ctx_cli]() { ctx_cli.ctx_server.start_loop(); });

    auto        inf        = ctx_cli.ctx_server.get_meta();
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    if (!params.system_prompt.empty()) {
        ctx_cli.messages.push_back({
            { "role",    "system"             },
            { "content", params.system_prompt }
        });
    }

    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        console::log("using custom system prompt\n");
    }
    console::log("\n");
    console::log("available commands:\n");
    console::log("  /exit or Ctrl+C     stop or exit\n");
    console::log("  /regen              regenerate the last response\n");
    console::log("  /clear              clear the chat history\n");
    console::log("  /read               add a text file\n");
    if (inf.has_inp_image) {
        console::log("  /image <file>       add an image file\n");
    }
    if (inf.has_inp_audio) {
        console::log("  /audio <file>       add an audio file\n");
    }
    console::log("\n");

    // interactive loop
    std::string cur_msg;
    while (true) {
        std::string buffer;
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        if (params.prompt.empty()) {
            console::log("\n> ");
            std::string line;
            bool        another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                std::string marker = ctx_cli.load_input_file(fname, true);
                if (marker.empty()) {
                    console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                    break;
                }
                console::log("Loaded media from '%s'\n", fname.c_str());
                cur_msg += marker;
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
            params.prompt.clear();  // only use it once
        }
        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() && buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } 
        if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                console::error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            ctx_cli.input_files.clear();
            console::log("Chat history cleared.\n");
            continue;
        } else if ((string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                   (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname  = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_file(fname, true);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded media from '%s'\n", fname.c_str());
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname  = string_strip(buffer.substr(6));
            std::string marker = ctx_cli.load_input_file(fname, false);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded text from '%s'\n", fname.c_str());
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                { "role",    "user"  },
                { "content", cur_msg }
            });
            cur_msg.clear();
        }
        result_timings timings;
        while (true) {
            server_task_result_ptr result;
            std::string            assistant_content = ctx_cli.generate_completion(timings, result);
            auto *                 res_final         = dynamic_cast<server_task_result_cmpl_final *>(result.get());

            if (res_final && !res_final->oaicompat_msg.tool_calls.empty()) {
                ctx_cli.messages.push_back(res_final->oaicompat_msg.to_json_oaicompat());

                for (const auto & tc : res_final->oaicompat_msg.tool_calls) {
                    json args;
                    try {
                        if (tc.arguments.empty()) {
                            args = json::object();
                        } else {
                            args = json::parse(tc.arguments);
                        }
                    } catch (...) {
                        json err_msg = {
                            { "role",         "tool"                    },
                            { "content",      "Error parsing arguments" },
                            { "tool_call_id", tc.id                     },
                            { "name",         tc.name                   }
                        };
                        ctx_cli.messages.push_back(err_msg);
                        continue;
                    }

                    std::string request_id = tc.name + tc.arguments;
                    if (!mcp.get_yolo() &&
                        ctx_cli.approved_requests.find(request_id) == ctx_cli.approved_requests.end()) {
                        // Prompt user
                        fprintf(stdout, "\n\n\033[1;33mTool call: %s\033[0m\n", tc.name.c_str());
                        fprintf(stdout, "Arguments: %s\n", args.dump(2).c_str());
                        fprintf(stdout, "Approve? [y]es, [n]o, [A]lways allow feature: ");
                        fflush(stdout);

                        char        c = ' ';
                        std::string line;

                        console::readline(line, false);
                        if (!line.empty()) {
                            c = line[0];
                        }

                        if (c == 'y' || c == 'Y') {
                            // approved once
                        } else if (c == 'A') {
                            ctx_cli.approved_requests.insert(request_id);
                        } else {
                            json err_msg = {
                                { "role",         "tool"                       },
                                { "content",      "User denied tool execution" },
                                { "tool_call_id", tc.id                        },
                                { "name",         tc.name                      }
                            };
                            ctx_cli.messages.push_back(err_msg);
                            continue;
                        }
                    }

                    json res      = mcp.call_tool(tc.name, args);
                    json tool_msg = {
                        { "role",         "tool"     },
                        { "content",      res.dump() },
                        { "tool_call_id", tc.id      },
                        { "name",         tc.name    }
                    };
                    ctx_cli.messages.push_back(tool_msg);
                }
                // continue loop to generate with tool results
            } else {
                json assistant_msg = {
                    { "role",    "assistant"       },
                    { "content", assistant_content }
                };
                ctx_cli.messages.push_back(assistant_msg);
                break;
            }
        }
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", timings.prompt_per_second,
                         timings.predicted_per_second);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);

    console::log("\nExiting...\n");
    ctx_cli.ctx_server.terminate();
    inference_thread.join();

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    return 0;
}
