#include "common.h"
#include "arg.h"
#include "console.h"

#include "agent-loop.h"
#include "tool-registry.h"
#include "permission.h"

#include <atomic>
#include <fstream>
#include <thread>
#include <signal.h>
#include <filesystem>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

const char * LLAMA_AGENT_LOGO = R"(
  _ _                                                _
 | | | __ _ _ __ ___   __ _        __ _  __ _  ___ _ __ | |_
 | | |/ _` | '_ ` _ \ / _` |      / _` |/ _` |/ _ \ '_ \| __|
 | | | (_| | | | | | | (_| |  _  | (_| | (_| |  __/ | | | |_
 |_|_|\__,_|_| |_| |_|\__,_| (_)  \__,_|\__, |\___|_| |_|\__|
                                        |___/
)";

static std::atomic<bool> g_is_interrupted = false;

static bool should_stop() {
    return g_is_interrupted.load();
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR;

    // Check for custom flags before common_params_parse
    bool yolo_mode = false;
    int max_iterations = 50;  // Default value

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--yolo") {
            yolo_mode = true;
            // Remove from argv
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;  // Re-check this position
        } else if (arg == "--max-iterations" || arg == "-mi") {
            if (i + 1 < argc) {
                try {
                    max_iterations = std::stoi(argv[i + 1]);
                    if (max_iterations < 1) max_iterations = 1;
                    if (max_iterations > 1000) max_iterations = 1000;
                } catch (...) {
                    fprintf(stderr, "Invalid --max-iterations value: %s\n", argv[i + 1]);
                    return 1;
                }
                // Remove both the flag and its value
                for (int j = i; j < argc - 2; j++) {
                    argv[j] = argv[j + 2];
                }
                argc -= 2;
                i--;  // Re-check this position
            } else {
                fprintf(stderr, "--max-iterations requires a value\n");
                return 1;
            }
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        console::error("--no-conversation is not supported by llama-agent\n");
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);

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

    // Create server context
    server_context ctx_server;

    console::log("\nLoading model... ");
    console::spinner::start();
    if (!ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    ctx_server.init();
    console::spinner::stop();
    console::log("\n");

    // Start inference thread
    std::thread inference_thread([&ctx_server]() {
        ctx_server.start_loop();
    });

    auto inf = ctx_server.get_info();

    // Get working directory
    std::string working_dir = fs::current_path().string();

    // Configure agent
    agent_config config;
    config.working_dir = working_dir;
    config.max_iterations = max_iterations;
    config.tool_timeout_ms = 120000;
    config.verbose = (params.verbosity >= LOG_LEVEL_INFO);
    config.yolo_mode = yolo_mode;

    // Create agent loop
    agent_loop agent(ctx_server, params, config, g_is_interrupted);

    // Display startup info
    console::log("\n");
    console::log("%s\n", LLAMA_AGENT_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("working dir: %s\n", working_dir.c_str());
    if (yolo_mode) {
        console::set_display(DISPLAY_TYPE_ERROR);
        console::log("mode       : YOLO (all permissions auto-approved)\n");
        console::set_display(DISPLAY_TYPE_RESET);
    }
    console::log("\n");

    console::log("commands:\n");
    console::log("  /exit       exit the agent\n");
    console::log("  /clear      clear conversation history\n");
    console::log("  /tools      list available tools\n");
    console::log("  ESC/Ctrl+C  abort generation\n");
    console::log("\n");

    // Interactive loop
    while (true) {
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        console::log("\n› ");

        std::string buffer;
        std::string line;
        bool another_line = true;
        do {
            another_line = console::readline(line, params.multiline_input);
            buffer += line;
        } while (another_line);

        console::set_display(DISPLAY_TYPE_RESET);

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // Remove trailing newline
        if (!buffer.empty() && buffer.back() == '\n') {
            buffer.pop_back();
        }

        // Skip empty input
        if (buffer.empty()) {
            continue;
        }

        // Process commands
        if (buffer == "/exit" || buffer == "/quit") {
            break;
        }
        if (buffer == "/clear") {
            agent.clear();
            console::log("Conversation cleared.\n");
            continue;
        }
        if (buffer == "/tools") {
            console::log("\nAvailable tools:\n");
            for (const auto * tool : tool_registry::instance().get_all_tools()) {
                console::log("  %s:\n", tool->name.c_str());
                console::log("    %s\n", tool->description.c_str());
            }
            continue;
        }

        console::log("\n");

        // Run agent loop
        agent_loop_result result = agent.run(buffer);

        console::log("\n");

        // Display result
        switch (result.stop_reason) {
            case agent_stop_reason::COMPLETED:
                console::set_display(DISPLAY_TYPE_INFO);
                console::log("[Completed in %d iteration(s)]\n", result.iterations);
                console::set_display(DISPLAY_TYPE_RESET);
                break;
            case agent_stop_reason::MAX_ITERATIONS:
                console::set_display(DISPLAY_TYPE_ERROR);
                console::log("[Stopped: max iterations reached (%d)]\n", result.iterations);
                console::set_display(DISPLAY_TYPE_RESET);
                break;
            case agent_stop_reason::USER_CANCELLED:
                console::log("[Cancelled by user]\n");
                g_is_interrupted.store(false);
                break;
            case agent_stop_reason::ERROR:
                console::error("[Error occurred]\n");
                break;
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);
    console::log("\nExiting...\n");

    ctx_server.terminate();
    inference_thread.join();

    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_server.get_llama_context());

    return 0;
}
