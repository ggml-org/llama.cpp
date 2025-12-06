#include "common.h"
#include "arg.h"
#include "console.h"
#include "log.h"

#include "server-context.h"
#include "server-task.h"

#include <atomic>
#include <fstream>
#include <signal.h>

constexpr int POLLING_SECONDS = 1;

const char * LLAMA_ASCII_LOGO = R"(
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

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;
        defaults.stream = true; // make sure we always use streaming mode
    }

    std::string generate_completion() {
        auto queues = ctx_server.get_queues();
        server_response_reader rd(queues, POLLING_SECONDS);
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id        = queues.first.get_new_id();
            task.params    = defaults;    // copy
            task.cli_input = messages;    // copy
            task.cli_files = input_files; // copy
            rd.post_task({std::move(task)});
        }

        server_task_result_ptr result = rd.next(should_stop);
        std::string curr_content;
        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    LOG_ERR("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    LOG_ERR("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                curr_content += res_partial->content;
                LOG("%s", res_partial->content.c_str());
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                break;
            }
            result = rd.next(should_stop);
        }
        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_files(const std::string & fname) {
        input_files.clear();
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        raw_buffer buf;
        buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        input_files.push_back(std::move(buf));
        return mtmd_default_marker();
    }
};

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    common_init();

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    LOG("Loading model...\n");
    if (!ctx_cli.ctx_server.load_model(params)) {
        LOG_ERR("Failed to load the model\n");
        return 1;
    }
    ctx_cli.ctx_server.init();

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    auto inf = ctx_cli.ctx_server.get_info();
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    if (!params.system_prompt.empty()) {
        ctx_cli.messages.push_back({
            {"role",    "system"},
            {"content", params.system_prompt}
        });
    }

    LOG("\n");
    LOG("%s\n", LLAMA_ASCII_LOGO);
    LOG("build      : %s\n", inf.build_info.c_str());
    LOG("model      : %s\n", inf.model_name.c_str());
    LOG("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        LOG("using custom system prompt\n");
    }
    LOG("\n");
    LOG("available commands:\n");
    LOG("  /exit or Ctrl+C   stop or exit\n");
    LOG("  /regen            regenerate the last response\n");
    LOG("  /clear            clear the chat history\n");
    if (inf.has_inp_image) {
        LOG("  /image <file>     add an image file\n");
    }
    if (inf.has_inp_audio) {
        LOG("  /audio <file>     add an audio file\n");
    }
    LOG("\n");

    // interactive loop
    std::string cur_msg;
    while (true) {
        std::string buffer;
        console::set_display(console::user_input);
        if (params.prompt.empty()) {
            LOG("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            buffer = params.prompt;
            LOG("\n> %s\n", buffer.c_str());
            params.prompt.clear(); // only use it once
        }
        console::set_display(console::reset);
        LOG("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        if (buffer.empty()) {
            continue;
        }

        // remove trailing newline
        if (buffer.back() == '\n') {
            buffer.pop_back();
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                LOG_ERR("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            LOG("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_files(fname);
            if (marker.empty()) {
                LOG_ERR("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            LOG("Loaded image from '%s'\n", fname.c_str());
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", cur_msg}
            });
            cur_msg.clear();
        }
        std::string assistant_content = ctx_cli.generate_completion();
        ctx_cli.messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });
        LOG("\n");
    }

    LOG("\nExiting...\n");
    ctx_cli.ctx_server.terminate();
    inference_thread.join();
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    return 0;
}
