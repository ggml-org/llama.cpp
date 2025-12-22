#include "common.h"
#include "arg.h"
#include "console.h"

#include "agent-loop.h"
#include "tool-registry.h"
#include "permission.h"
#include "skills/skills-manager.h"

#ifndef _WIN32
#include "mcp/mcp-server-manager.h"
#include "mcp/mcp-tool-wrapper.h"
#endif

#include <atomic>
#include <fstream>
#include <iostream>
#include <thread>
#include <signal.h>
#include <filesystem>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Get the user config directory for llama-agent
static std::string get_config_dir() {
#ifdef _WIN32
    const char * appdata = std::getenv("APPDATA");
    if (appdata) {
        return std::string(appdata) + "\\llama-agent";
    }
    return "";
#else
    const char * xdg_config = std::getenv("XDG_CONFIG_HOME");
    if (xdg_config) {
        return std::string(xdg_config) + "/llama-agent";
    }
    const char * home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.config/llama-agent";
    }
    return "";
#endif
}

const char * LLAMA_AGENT_LOGO = R"(
      _ _                                                  _
     | | | __ _ _ __ ___   __ _      __ _  __ _  ___ _ __ | |_
    | | |/ _` | '_ ` _ \ / _` |___ / _` |/ _` |/ _ \ '_ \| __|
   | | | (_| | | | | | | (_| |___| (_| | (_| |  __/ | | | |_
  |_|_|\__,_|_| |_| |_|\__,_|    \__,_|\__, |\___|_| |_|\__|
                                       |___/
)";

static std::atomic<bool> g_is_interrupted = false;

static bool should_stop() {
    return g_is_interrupted.load();
}

static bool is_stdin_terminal() {
#ifdef _WIN32
    return _isatty(_fileno(stdin));
#else
    return isatty(fileno(stdin));
#endif
}

static std::string read_stdin_prompt() {
    std::string result;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!result.empty()) {
            result += "\n";
        }
        result += line;
    }
    return result;
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
    bool enable_skills = true;
    std::vector<std::string> extra_skills_paths;

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
        } else if (arg == "--no-skills") {
            enable_skills = false;
            // Remove from argv
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;  // Re-check this position
        } else if (arg == "--skills-path") {
            if (i + 1 < argc) {
                extra_skills_paths.push_back(argv[i + 1]);
                // Remove both the flag and its value
                for (int j = i; j < argc - 2; j++) {
                    argv[j] = argv[j + 2];
                }
                argc -= 2;
                i--;  // Re-check this position
            } else {
                fprintf(stderr, "--skills-path requires a value\n");
                return 1;
            }
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

#ifndef _WIN32
    // Load MCP servers (Unix only - requires fork/pipe)
    mcp_server_manager mcp_mgr;
    int mcp_tools_count = 0;
    std::string mcp_config = find_mcp_config(working_dir);
    if (!mcp_config.empty()) {
        if (mcp_mgr.load_config(mcp_config)) {
            int started = mcp_mgr.start_servers();
            if (started > 0) {
                register_mcp_tools(mcp_mgr);
                mcp_tools_count = (int)mcp_mgr.list_all_tools().size();
            }
        }
    }
#else
    int mcp_tools_count = 0;
#endif

    // Discover skills (agentskills.io spec)
    skills_manager skills_mgr;
    int skills_count = 0;
    if (enable_skills) {
        std::vector<std::string> skill_paths;

        // Project-local skills (highest priority)
        skill_paths.push_back(working_dir + "/.llama-agent/skills");

        // User-global skills
        std::string config_dir = get_config_dir();
        if (!config_dir.empty()) {
            skill_paths.push_back(config_dir + "/skills");
        }

        // Extra paths from --skills-path flags
        skill_paths.insert(skill_paths.end(),
            extra_skills_paths.begin(), extra_skills_paths.end());

        skills_count = skills_mgr.discover(skill_paths);
    }

    // Configure agent
    agent_config config;
    config.working_dir = working_dir;
    config.max_iterations = max_iterations;
    config.tool_timeout_ms = 120000;
    config.verbose = (params.verbosity >= LOG_LEVEL_INFO);
    config.yolo_mode = yolo_mode;
    config.enable_skills = enable_skills;
    config.skills_search_paths = extra_skills_paths;
    config.skills_prompt_section = skills_mgr.generate_prompt_section();

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
    if (mcp_tools_count > 0) {
        console::log("mcp tools  : %d\n", mcp_tools_count);
    }
    if (skills_count > 0) {
        console::log("skills     : %d\n", skills_count);
    }
    console::log("\n");

    // Resolve initial prompt from -p/--prompt flag or stdin
    std::string initial_prompt;
    if (!params.prompt.empty()) {
        initial_prompt = params.prompt;
        params.prompt.clear();  // Only use once
    } else if (!is_stdin_terminal()) {
        initial_prompt = read_stdin_prompt();
        // Trim trailing whitespace
        while (!initial_prompt.empty() && (initial_prompt.back() == '\n' || initial_prompt.back() == '\r')) {
            initial_prompt.pop_back();
        }
        // When reading from stdin pipe, always use single-turn mode
        // (stdin is at EOF, so interactive input would spin forever)
        params.single_turn = true;
    }

    // Non-interactive mode: if we have a prompt and single_turn, skip the help text
    if (initial_prompt.empty() || !params.single_turn) {
        console::log("commands:\n");
        console::log("  /exit       exit the agent\n");
        console::log("  /clear      clear conversation history\n");
        console::log("  /tools      list available tools\n");
        console::log("  /skills     list available skills\n");
        console::log("  ESC/Ctrl+C  abort generation\n");
        console::log("\n");
    }

    // Track if we have an initial prompt to process
    bool first_turn = !initial_prompt.empty();

    // Main loop
    while (true) {
        std::string buffer;

        if (first_turn) {
            // Use the initial prompt
            buffer = initial_prompt;
            first_turn = false;
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            console::log("\n› %s\n", buffer.c_str());
            console::set_display(DISPLAY_TYPE_RESET);
        } else {
            // Interactive input
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            console::log("\n› ");

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
            if (buffer == "/skills") {
                const auto & skills = skills_mgr.get_skills();
                if (skills.empty()) {
                    console::log("\nNo skills discovered.\n");
                    console::log("Skills are loaded from:\n");
                    console::log("  ./.llama-agent/skills/  (project-local)\n");
                    console::log("  ~/.config/llama-agent/skills/  (user-global)\n");
                } else {
                    console::log("\nAvailable skills:\n");
                    for (const auto & skill : skills) {
                        console::log("  %s:\n", skill.name.c_str());
                        console::log("    %s\n", skill.description.c_str());
                        console::log("    Path: %s\n", skill.path.c_str());
                    }
                }
                continue;
            }
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
            case agent_stop_reason::AGENT_ERROR:
                console::error("[Error occurred]\n");
                break;
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);
    console::log("\nExiting...\n");

#ifndef _WIN32
    // Shutdown MCP servers
    mcp_mgr.shutdown_all();
#endif

    ctx_server.terminate();
    inference_thread.join();

    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_server.get_llama_context());

    return 0;
}
