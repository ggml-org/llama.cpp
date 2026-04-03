#include "common.h"
#include "arg.h"
#include "console.h"

#include "agent-loop.h"
#include "config-dir.h"
#include "tool-registry.h"
#include "permission.h"
#include "skills/skills-manager.h"
#include "agents-md/agents-md-manager.h"

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


const char * LLAMA_AGENT_LOGO = R"(
    ____                                                   __
   / / /___ _____ ___  ____ _      ____ _____ ____  ____  / /_
  / / / __ `/ __ `__ \/ __ `/_____/ __ `/ __ `/ _ \/ __ \/ __/
 / / / /_/ / / / / / / /_/ /_____/ /_/ / /_/ /  __/ / / / /_
/_/_/\__,_/_/ /_/ /_/\__,_/      \__,_/\__, /\___/_/ /_/\__/
                                      /____/
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
    bool enable_agents_md = true;
    bool enable_compaction = true;
    bool enable_session = true;
    bool resume_session = false;
    std::string session_path;  // explicit path, or auto-generated
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
        } else if (arg == "--no-agents-md") {
            enable_agents_md = false;
            // Remove from argv
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;
        } else if (arg == "--no-compaction") {
            enable_compaction = false;
            // Remove from argv
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;  // Re-check this position
        } else if (arg == "--session") {
            if (i + 1 < argc) {
                session_path = argv[i + 1];
                for (int j = i; j < argc - 2; j++) {
                    argv[j] = argv[j + 2];
                }
                argc -= 2;
                i--;
            } else {
                fprintf(stderr, "--session requires a file path\n");
                return 1;
            }
        } else if (arg == "--resume") {
            resume_session = true;
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;
        } else if (arg == "--no-session") {
            enable_session = false;
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;
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

    // Apply verbosity setting immediately after init to suppress verbose logs
    common_log_set_verbosity_thold(params.verbosity);

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

    console::spinner::stop();
    console::log("\n");

    // Start inference thread
    std::thread inference_thread([&ctx_server]() {
        ctx_server.start_loop();
    });

    auto inf = ctx_server.get_meta();

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
        skill_paths.push_back(working_dir + "/.agents/skills");

        // User-global skills
        std::string config_dir = get_config_dir();
        if (!config_dir.empty()) {
            skill_paths.push_back(config_dir + "/skills");
        }

        // User-global skills (alternative path: ~/.agents/skills)
#ifdef _WIN32
        const char * home_skills = std::getenv("APPDATA");
        if (home_skills) {
            skill_paths.push_back(std::string(home_skills) + "\\agents\\skills");
        }
#else
        const char * home_skills = std::getenv("HOME");
        if (home_skills) {
            skill_paths.push_back(std::string(home_skills) + "/.agents/skills");
        }
#endif

        // Extra paths from --skills-path flags
        skill_paths.insert(skill_paths.end(),
            extra_skills_paths.begin(), extra_skills_paths.end());

        skills_count = skills_mgr.discover(skill_paths);
    }

    // Discover AGENTS.md files (agents.md spec)
    agents_md_manager agents_md_mgr;
    int agents_md_count = 0;
    if (enable_agents_md) {
        // Pass config_dir for global AGENTS.md support (~/.llama-agent/AGENTS.md)
        std::string agents_config_dir = get_config_dir();
        agents_md_count = agents_md_mgr.discover(working_dir, agents_config_dir);

        // Warn if content is very large
        size_t total_size = agents_md_mgr.total_content_size();
        if (total_size > 50 * 1024) {
            console::log("Warning: AGENTS.md content is large (%zu bytes). "
                        "Consider reducing size for better performance.\n", total_size);
        }
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
    config.enable_agents_md = enable_agents_md;
    config.agents_md_prompt_section = agents_md_mgr.generate_prompt_section();
    config.compaction.enabled = enable_compaction;

    // Session persistence
    session_file sf;
    session_file * sf_ptr = nullptr;
    loaded_session loaded;
    const loaded_session * resume_ptr = nullptr;

    if (enable_session && session_path.empty()) {
        // Auto-generate session path based on config dir + working directory
        std::string config_dir = get_config_dir();
        if (!config_dir.empty()) {
            std::string session_dir = session_file::get_session_dir(config_dir, working_dir);
            if (resume_session) {
                session_path = session_file::find_latest_session(session_dir);
                if (session_path.empty()) {
                    console::log("No previous session found, starting new.\n");
                    session_path = session_file::new_session_path(session_dir);
                }
            } else {
                session_path = session_file::new_session_path(session_dir);
            }
        }
    }

    if (!session_path.empty()) {
        if (resume_session || std::filesystem::exists(session_path)) {
            auto maybe = session_file::load(session_path);
            if (maybe) {
                loaded = std::move(*maybe);
                resume_ptr = &loaded;
            }
        }
        if (sf.open(session_path)) {
            sf_ptr = &sf;
            if (resume_ptr) {
                sf.set_message_count(resume_ptr->total_messages_in_file);
            }
        }
    }

    // Create agent loop
    agent_loop agent(ctx_server, params, config, g_is_interrupted, sf_ptr, resume_ptr);

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
    if (agents_md_count > 0) {
        console::log("agents.md  : %d file(s)\n", agents_md_count);
    }
    if (!session_path.empty()) {
        console::log("session    : %s%s\n", session_path.c_str(),
                      resume_ptr ? " (resumed)" : " (new)");
    }
    console::log("\n");

    // Display resumed conversation history
    // Helper: extract text from content that may be a string or array of content blocks.
    auto extract_text = [](const json & msg) -> std::string {
        if (!msg.contains("content")) {
            return "";
        }
        const auto & c = msg["content"];
        if (c.is_string()) {
            return c.get<std::string>();
        }
        if (c.is_array()) {
            std::string text;
            for (const auto & block : c) {
                if (block.contains("type") && block["type"] == "image_url") {
                    text += "[image]\n";
                } else if (block.contains("text") && block["text"].is_string()) {
                    text += block["text"].get<std::string>();
                }
            }
            return text;
        }
        return "";
    };

    if (resume_ptr && !resume_ptr->messages.empty()) {
        for (const auto & m : resume_ptr->messages) {
            std::string role = m.value("role", "");
            if (role == "user") {
                console::set_display(DISPLAY_TYPE_USER_INPUT);
                console::log("› %s\n", extract_text(m).c_str());
                console::set_display(DISPLAY_TYPE_RESET);
            } else if (role == "assistant") {
                std::string content = extract_text(m);
                if (!content.empty()) {
                    console::log("%s\n", content.c_str());
                }
                if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                    for (const auto & tc : m["tool_calls"]) {
                        if (tc.contains("function")) {
                            std::string name = tc["function"].value("name", "");
                            console::log("› %s\n", name.c_str());
                        }
                    }
                }
            } else if (role == "tool") {
                std::string output = extract_text(m);
                if (output.length() > 500) {
                    output = output.substr(0, 500) + "\n... (truncated)";
                }
                console::log("%s\n", output.c_str());
            }
        }
        console::log("--- session resumed ---\n");
    }

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
        console::log("  /stats      show token usage statistics\n");
        console::log("  /tools      list available tools\n");
        console::log("  /skills     list available skills\n");
        console::log("  /agents     list discovered AGENTS.md files\n");
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
            if (buffer == "/stats") {
                const auto & stats = agent.get_stats();
                console::log("\nSession Statistics:\n");
                console::log("  Prompt tokens:  %d\n", stats.total_input);
                console::log("  Output tokens:  %d\n", stats.total_output);
                if (stats.total_cached > 0) {
                    console::log("  Cached tokens:  %d\n", stats.total_cached);
                }
                console::log("  Total tokens:   %d\n", stats.total_input + stats.total_output);

                if (stats.total_prompt_ms > 0) {
                    console::log("  Prompt time:    %.2fs\n", stats.total_prompt_ms / 1000.0);
                }
                if (stats.total_predicted_ms > 0) {
                    console::log("  Gen time:       %.2fs\n", stats.total_predicted_ms / 1000.0);
                    double avg_speed = stats.total_output * 1000.0 / stats.total_predicted_ms;
                    console::log("  Avg speed:      %.1f tok/s\n", avg_speed);
                }
                continue;
            }
            if (buffer == "/skills") {
                const auto & skills = skills_mgr.get_skills();
                if (skills.empty()) {
                    console::log("\nNo skills discovered.\n");
                    console::log("Skills are loaded from:\n");
                    console::log("  ./.llama-agent/skills/  (project-local)\n");
                    console::log("  ~/.llama-agent/skills/  (user-global)\n");
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
            if (buffer == "/agents") {
                const auto & files = agents_md_mgr.get_files();
                if (files.empty()) {
                    console::log("\nNo AGENTS.md files discovered.\n");
                    console::log("AGENTS.md files are searched from:\n");
                    console::log("  ./AGENTS.md to git root  (project-specific)\n");
                    console::log("  ~/.llama-agent/AGENTS.md  (global)\n");
                } else {
                    console::log("\nDiscovered AGENTS.md files (closest first):\n");
                    for (const auto & file : files) {
                        console::log("  %s", file.relative_path.c_str());
                        if (file.depth == 0) {
                            console::log(" (highest precedence)");
                        }
                        console::log("\n    %zu bytes\n", file.content.size());
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
