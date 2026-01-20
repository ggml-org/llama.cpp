#include "agent-loop.h"
#include "console.h"

#include <chrono>
#include <functional>
#include <mutex>
#include <sstream>

// Global mutex to serialize generate_completion calls across all agent instances
// This works around a race condition in the server's cli_input handling
static std::mutex g_completion_mutex;

#if defined(_WIN32)
#include <conio.h>
#else
#include <sys/select.h>
#include <unistd.h>
#include <termios.h>
#endif

// Check for ESC key press without blocking
static bool check_escape_key() {
#if defined(_WIN32)
    if (_kbhit()) {
        int ch = _getch();
        if (ch == 27) { // ESC
            return true;
        }
    }
    return false;
#else
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);

    struct timeval tv = {0, 0}; // Zero timeout = non-blocking

    if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0) {
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1 && ch == 27) { // ESC
            return true;
        }
    }
    return false;
#endif
}

agent_loop::agent_loop(server_context & server_ctx,
                       const common_params & params,
                       const agent_config & config,
                       std::atomic<bool> & is_interrupted)
    : server_ctx_(server_ctx)
    , params_(&params)
    , config_(config)
    , is_interrupted_(is_interrupted)
    , messages_(json::array())
{
    // Initialize task defaults from params
    task_defaults_.sampling    = params.sampling;
    task_defaults_.speculative = params.speculative;
    task_defaults_.n_keep      = params.n_keep;
    task_defaults_.n_predict   = params.n_predict;
    task_defaults_.antiprompt  = params.antiprompt;
    task_defaults_.stream      = true;
    task_defaults_.timings_per_token = true;
    task_defaults_.oaicompat_chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    task_defaults_.oaicompat_chat_syntax.parse_tool_calls = true;

    // Initialize tool context
    tool_ctx_.working_dir = config.working_dir.empty() ? "." : config.working_dir;
    tool_ctx_.is_interrupted = &is_interrupted_;
    tool_ctx_.timeout_ms = config.tool_timeout_ms;

    // Subagent support: store pointers to server context and config
    tool_ctx_.server_ctx_ptr = &server_ctx_;
    tool_ctx_.agent_config_ptr = const_cast<agent_config *>(&config_);
    tool_ctx_.common_params_ptr = const_cast<common_params *>(&params);
    tool_ctx_.session_stats_ptr = &stats_;
    tool_ctx_.subagent_depth = 0;

    // Set up permission manager
    permission_mgr_.set_project_root(tool_ctx_.working_dir);
    permission_mgr_.set_yolo_mode(config.yolo_mode);

    // Base prompt shared with subagents for KV cache prefix sharing
    // Subagent prompts start with this exact text to maximize cache hits
    static const char * BASE_PROMPT_PREFIX = R"(You are llama-agent, a powerful local AI coding assistant running on llama.cpp.

You help users with software engineering tasks by reading files, writing code, running commands, and navigating codebases. You run entirely on the user's machine - no data leaves their system.

# Tools

You have access to the following tools:

- **bash**: Execute shell commands. Use for git, build commands, running tests, etc.
- **read**: Read file contents with line numbers. Always read files before editing them.
- **write**: Create new files or overwrite existing ones.
- **edit**: Make targeted edits using search/replace. The old_string must match exactly. Use replace_all=true to replace all occurrences of a word or phrase.
- **glob**: Find files matching a pattern. Use to explore project structure.

)";

    // Store base prompt for subagents to inherit (enables KV cache prefix sharing)
    tool_ctx_.base_system_prompt = BASE_PROMPT_PREFIX;

    // Add system prompt for tool usage
    std::string system_prompt = R"(You are llama-agent, a powerful local AI coding assistant running on llama.cpp.

You help users with software engineering tasks by reading files, writing code, running commands, and navigating codebases. You run entirely on the user's machine - no data leaves their system.

# Tools

You have access to the following tools:

- **bash**: Execute shell commands. Use for git, build commands, running tests, etc.
- **read**: Read file contents with line numbers. Always read files before editing them.
- **write**: Create new files or overwrite existing ones.
- **edit**: Make targeted edits using search/replace. The old_string must match exactly. Use replace_all=true to replace all occurrences of a word or phrase.
- **glob**: Find files matching a pattern. Use to explore project structure.

## Using the edit tool
The edit tool finds and replaces text in files. Key points:
- **old_string must match exactly** - include correct whitespace and indentation
- **Always read the file first** - so you know the exact text to match
- **Use replace_all=true** when replacing a word or short phrase everywhere in the file
- **Use more context** when there are multiple matches and you only want to change one

# Guidelines

## Be direct and concise
- Give short, clear responses. No filler or excessive explanation.
- Use markdown for code blocks and formatting.
- No emojis unless the user asks for them.

## Think step by step
- Break complex tasks into smaller steps.
- After each tool result, analyze what you learned and decide the next action.
- When stuck, explain your reasoning and ask for clarification.

## Read before you write
- ALWAYS read a file before editing it.
- Understand existing code patterns before making changes.
- Check if similar code exists before creating new files.

## Be careful with destructive operations
- Double-check paths before deleting or overwriting files.
- Prefer targeted edits over full file rewrites.
- Run tests after making changes when possible.

# Tool Usage

## Parallel execution
When multiple operations are independent, execute them together. For example, reading multiple files or running independent commands.

## Search strategy
When looking for code:
1. Use `glob` to find candidate files
2. Use `read` to examine promising files
3. Use `bash` with grep for text search across files

## Code references
When referring to code, use the format `file_path:line_number` so users can navigate easily.

Example: "The bug is in the `processRequest` function in src/server.cpp:142"

# Examples

<example>
User: Fix the typo in README.md
Assistant: Let me read the file first.
[Tool: read README.md]
I see the typo on line 15. Let me fix it.
[Tool: edit README.md - fixes "teh" -> "the"]
Done. Fixed "teh" to "the" on line 15.
</example>

<example>
User: Find where the API routes are defined
Assistant: Let me search for route definitions.
[Tool: glob **/*route*.{js,ts}]
Found src/routes/api.ts and src/routes/auth.ts.
[Tool: read src/routes/api.ts]
The API routes are defined in src/routes/api.ts. The main endpoints are:
- GET /users (line 12)
- POST /users (line 24)
- GET /users/:id (line 36)
</example>

<example>
User: Run the tests and fix any failures
Assistant: Running the test suite.
[Tool: bash npm test]
3 tests failed. Let me examine each failure:
1. test/auth.test.js - "should validate token"
[Tool: read test/auth.test.js]
[Tool: read src/auth.js]
The issue is on src/auth.js:45 - the token expiry check is inverted.
[Tool: edit src/auth.js - fixes the condition]
Let me run the tests again.
[Tool: bash npm test]
All tests passing now.
</example>

When the task is complete, provide a brief summary of what you did.)";

    // Append AGENTS.md section if available (agents.md spec)
    if (!config.agents_md_prompt_section.empty()) {
        system_prompt += R"(

# Project Context

This project has AGENTS.md files with specific guidance for this codebase.
Follow these project-specific instructions, especially for:
- Build and test commands
- Code style preferences
- File organization conventions
- PR and commit guidelines

When project instructions conflict with general guidelines, prefer project-specific guidance.

)";
        system_prompt += config.agents_md_prompt_section;
    }

    // Append skills section if available (agentskills.io spec)
    if (!config.skills_prompt_section.empty()) {
        system_prompt += R"(

# Available Skills

Skills are specialized capabilities you can use for specific tasks.
When a user's request matches a skill description, read the skill file to get detailed instructions.
Use the `read` tool with the skill's location path to load the full instructions.

## Running Skill Scripts

Some skills include executable scripts in their `<scripts>` section. To run a skill script:

1. Use the `bash` tool with the full path: `<skill_dir>/<script>`
2. Example: `python /path/to/skill/scripts/analyze.py --file code.py`
3. Only script output is returned - source code stays out of context

If a skill has `<allowed_tools>`, it declares which tools it needs. This helps you understand the skill's scope.

)";
        system_prompt += config.skills_prompt_section;
    }

    messages_.push_back({
        {"role", "system"},
        {"content", system_prompt}
    });
}

// Constructor for subagents with filtered tools and custom system prompt
agent_loop::agent_loop(server_context & server_ctx,
                       const common_params & params,
                       const agent_config & config,
                       std::atomic<bool> & is_interrupted,
                       const std::set<std::string> & allowed_tools,
                       const std::vector<std::string> & bash_patterns,
                       const std::string & custom_system_prompt,
                       int subagent_depth,
                       tool_call_callback on_tool_call)
    : server_ctx_(server_ctx)
    , params_(&params)
    , config_(config)
    , is_interrupted_(is_interrupted)
    , messages_(json::array())
    , allowed_tools_(allowed_tools)
    , bash_patterns_(bash_patterns)
    , on_tool_call_(on_tool_call)
    , is_subagent_(true)
{
    // Initialize task defaults from params
    task_defaults_.sampling    = params.sampling;
    task_defaults_.speculative = params.speculative;
    task_defaults_.n_keep      = params.n_keep;
    task_defaults_.n_predict   = params.n_predict;
    task_defaults_.antiprompt  = params.antiprompt;
    task_defaults_.stream      = true;
    task_defaults_.timings_per_token = true;
    task_defaults_.oaicompat_chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    task_defaults_.oaicompat_chat_syntax.parse_tool_calls = true;

    // Initialize tool context
    tool_ctx_.working_dir = config.working_dir.empty() ? "." : config.working_dir;
    tool_ctx_.is_interrupted = &is_interrupted_;
    tool_ctx_.timeout_ms = config.tool_timeout_ms;

    // Subagent support: store pointers to server context and config
    tool_ctx_.server_ctx_ptr = &server_ctx_;
    tool_ctx_.agent_config_ptr = const_cast<agent_config *>(&config_);
    tool_ctx_.common_params_ptr = const_cast<common_params *>(&params);
    tool_ctx_.session_stats_ptr = &stats_;
    tool_ctx_.subagent_depth = subagent_depth;

    // Set up permission manager
    permission_mgr_.set_project_root(tool_ctx_.working_dir);
    permission_mgr_.set_yolo_mode(config.yolo_mode);

    // Use custom system prompt provided by caller
    messages_.push_back({
        {"role", "system"},
        {"content", custom_system_prompt}
    });
}

void agent_loop::clear() {
    // Keep system prompt, clear rest
    if (messages_.size() > 1) {
        json system_msg = messages_[0];
        messages_ = json::array();
        messages_.push_back(system_msg);
    }
    permission_mgr_.clear_session();

    // Reset stats when conversation is cleared
    stats_ = session_stats{};
}

common_chat_msg agent_loop::generate_completion(result_timings & out_timings) {
    server_response_reader rd = server_ctx_.get_response_reader();
    {
        // Serialize task posting to work around server race condition with concurrent cli_input tasks
        std::lock_guard<std::mutex> lock(g_completion_mutex);

        server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
        task.id        = rd.get_new_id();
        task.index     = 0;
        task.params    = task_defaults_;

        // Build tools JSON in OAI-compatible format
        // Subagents use filtered tools, main agent uses all tools
        auto chat_tools = allowed_tools_.empty()
            ? tool_registry::instance().to_chat_tools()
            : tool_registry::instance().to_chat_tools_filtered(allowed_tools_);
        json tools_json = common_chat_tools_to_json_oaicompat<json>(chat_tools);

        // Pass both messages and tools as extended cli_input format
        // Use deep copy to avoid any shared state issues across concurrent agents
        task.cli_input = json::object();
        task.cli_input["messages"] = json::parse(messages_.dump());
        task.cli_input["tools"] = json::parse(tools_json.dump());
        task.cli_input["tool_choice"] = "auto";

        rd.post_task(std::move(task));
    }

    auto should_stop = [this]() {
        if (is_interrupted_.load()) {
            return true;
        }
        // Check for ESC key to abort generation
        if (check_escape_key()) {
            is_interrupted_.store(true);
            return true;
        }
        return false;
    };

    // Wait for first result (spinner only for main agent)
    if (!is_subagent_) {
        console::spinner::start();
    }
    server_task_result_ptr result = rd.next(should_stop);

    if (!is_subagent_) {
        console::spinner::stop();
    }
    std::string full_content;
    bool is_thinking = false;
    bool was_aborted = false;

    while (result) {
        if (should_stop()) {
            was_aborted = true;
            break;
        }
        if (result->is_error()) {
            json err_data = result->to_json();
            if (err_data.contains("message") && !is_subagent_) {
                console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
            }
            common_chat_msg empty_msg;
            return empty_msg;
        }

        auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
        if (res_partial) {
            out_timings = std::move(res_partial->timings);
            for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                if (!diff.content_delta.empty()) {
                    if (!is_subagent_) {
                        if (is_thinking) {
                            console::log("\n───\n\n");
                            console::set_display(DISPLAY_TYPE_RESET);
                        }
                        console::log("%s", diff.content_delta.c_str());
                        console::flush();
                    }
                    if (is_thinking) {
                        is_thinking = false;
                    }
                    full_content += diff.content_delta;
                }
                if (!diff.reasoning_content_delta.empty()) {
                    if (!is_subagent_) {
                        console::set_display(DISPLAY_TYPE_REASONING);
                        if (!is_thinking) {
                            console::log("───\n");
                        }
                        console::log("%s", diff.reasoning_content_delta.c_str());
                        console::flush();
                    }
                    is_thinking = true;
                }
            }
        }

        auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        if (res_final) {
            out_timings = std::move(res_final->timings);
            // Use the server-parsed message which handles all chat template formats
            // (Hermes 2 Pro, Qwen3-Coder, Llama 3.x, DeepSeek, etc.)
            if (!res_final->oaicompat_msg.empty()) {
                return res_final->oaicompat_msg;
            }
            // Fallback to raw content if no parsed message
            if (!res_final->content.empty()) {
                full_content = res_final->content;
            }
            break;
        }

        result = rd.next(should_stop);
    }

    // Reset interrupted flag for next interaction
    is_interrupted_.store(false);

    if (was_aborted) {
        if (!is_subagent_) {
            console::log("\n[Generation aborted]\n");
        }
        // Return partial content without tool calls so conversation can continue
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = full_content;
        return msg;
    }

    // Fallback: return content without tool calls
    // (Server should have parsed if parse_tool_calls=true, but handle edge cases)
    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = full_content;
    return msg;
}

tool_result agent_loop::execute_tool_call(const common_chat_tool_call & call) {
    auto & registry = tool_registry::instance();

    // Check if tool exists
    const tool_def * tool = registry.get_tool(call.name);
    if (!tool) {
        return {false, "", "Unknown tool: " + call.name};
    }

    // Parse arguments
    json args;
    try {
        args = json::parse(call.arguments);
    } catch (const json::parse_error & e) {
        return {false, "", std::string("Invalid JSON arguments: ") + e.what()};
    }

    // Determine permission type
    permission_type ptype = permission_type::BASH;
    if (call.name == "read") ptype = permission_type::FILE_READ;
    else if (call.name == "write") ptype = permission_type::FILE_WRITE;
    else if (call.name == "edit") ptype = permission_type::FILE_EDIT;
    else if (call.name == "glob") ptype = permission_type::GLOB;

    // Build permission request
    permission_request req;
    req.type = ptype;
    req.tool_name = call.name;
    req.details = call.arguments;

    // Check for external directory access on file operations
    if (call.name == "read" || call.name == "write" || call.name == "edit") {
        std::string file_path = args.value("file_path", "");
        if (!file_path.empty()) {
            // Make path absolute for comparison
            std::filesystem::path path(file_path);
            if (path.is_relative()) {
                path = std::filesystem::path(tool_ctx_.working_dir) / path;
            }
            if (permission_mgr_.is_external_path(path.string())) {
                permission_request ext_req;
                ext_req.type = permission_type::EXTERNAL_DIR;
                ext_req.tool_name = call.name;
                ext_req.details = "External file: " + path.string();
                ext_req.is_dangerous = true;
                ext_req.description = "Operation outside working directory";

                auto response = permission_mgr_.prompt_user(ext_req);
                if (response == permission_response::DENY_ONCE ||
                    response == permission_response::DENY_ALWAYS) {
                    return {false, "", "Blocked: File is outside working directory"};
                }
            }
        }
    }

    // Check for dangerous commands
    if (call.name == "bash") {
        std::string cmd = args.value("command", "");
        req.details = cmd;
        // Check for dangerous patterns
        for (const auto & pattern : {"rm -rf", "sudo ", "chmod 777"}) {
            if (cmd.find(pattern) != std::string::npos) {
                req.is_dangerous = true;
                break;
            }
        }
    }

    // Check doom loop
    std::hash<std::string> hasher;
    std::string args_hash = std::to_string(hasher(call.arguments));
    if (permission_mgr_.is_doom_loop(call.name, args_hash)) {
        req.description = "Detected repeated identical tool calls (doom loop)";
        auto response = permission_mgr_.prompt_user(req);
        if (response == permission_response::DENY_ONCE ||
            response == permission_response::DENY_ALWAYS) {
            return {false, "", "Blocked: Detected repeated identical tool calls"};
        }
    }

    // Check permission
    permission_state state = permission_mgr_.check_permission(req);
    if (state == permission_state::DENY || state == permission_state::DENY_SESSION) {
        return {false, "", "Permission denied for " + call.name};
    }

    if (state == permission_state::ASK) {
        auto response = permission_mgr_.prompt_user(req);
        if (response == permission_response::DENY_ONCE ||
            response == permission_response::DENY_ALWAYS) {
            return {false, "", "User denied permission for " + call.name};
        }
    }

    // Record this call
    permission_mgr_.record_tool_call(call.name, args_hash);

    // Display tool execution (only for main agent)
    if (!is_subagent_) {
        console::set_display(DISPLAY_TYPE_INFO);
        // Show tool name with relevant details
        if (call.name == "bash") {
            std::string cmd = args.value("command", "");
            // Truncate long commands for display
            if (cmd.length() > 100) {
                cmd = cmd.substr(0, 100) + "...";
            }
            console::log("\n› %s %s", call.name.c_str(), cmd.c_str());
        } else if (call.name == "read" || call.name == "write" || call.name == "edit") {
            std::string path = args.value("path", args.value("file_path", ""));
            console::log("\n› %s %s", call.name.c_str(), path.c_str());
        } else {
            console::log("\n› %s ", call.name.c_str());
        }
        console::spinner::start();
        console::set_display(DISPLAY_TYPE_RESET);
    }

    // Execute the tool with timing
    // Use filtered execution for subagents with bash restrictions (e.g., read-only explore)
    auto start_time = std::chrono::steady_clock::now();
    tool_result result = bash_patterns_.empty()
        ? registry.execute(call.name, args, tool_ctx_)
        : registry.execute_filtered(call.name, args, tool_ctx_, bash_patterns_);
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (!is_subagent_) {
        console::spinner::stop();

        // Display result summary
        if (result.success) {
            // Truncate long output for display
            std::string display_output = result.output;
            if (display_output.length() > 500) {
                display_output = display_output.substr(0, 500) + "\n... (truncated)";
            }
            console::log("%s\n", display_output.c_str());
        } else {
            // Show output if available (e.g., bash stderr), plus error if set
            if (!result.output.empty()) {
                std::string display_output = result.output;
                if (display_output.length() > 500) {
                    display_output = display_output.substr(0, 500) + "\n... (truncated)";
                }
                console::error("%s\n", display_output.c_str());
            }
            if (!result.error.empty()) {
                console::error("Error: %s\n", result.error.c_str());
            }
            if (result.output.empty() && result.error.empty()) {
                console::error("Error: Tool failed with no output\n");
            }
        }

        // Display elapsed time
        console::set_display(DISPLAY_TYPE_INFO);
        if (elapsed_ms < 1000) {
            console::log("└─ %lldms\n", (long long)elapsed_ms);
        } else {
            console::log("└─ %.1fs\n", elapsed_ms / 1000.0);
        }
        console::set_display(DISPLAY_TYPE_RESET);
    } else if (on_tool_call_) {
        // Report to parent via callback for subagents
        std::string args_summary = call.arguments;
        if (args_summary.length() > 60) {
            args_summary = args_summary.substr(0, 60) + "...";
        }
        on_tool_call_(call.name, args_summary, static_cast<int>(elapsed_ms));
    }

    return result;
}

void agent_loop::add_tool_result_message(const std::string & tool_name,
                                          const std::string & call_id,
                                          const tool_result & result) {
    json msg;
    msg["role"] = "tool";
    msg["tool_call_id"] = call_id;
    msg["name"] = tool_name;

    if (result.success) {
        msg["content"] = result.output;
    } else {
        // Include output if available (e.g., bash stderr), plus error message if set
        if (!result.output.empty() && !result.error.empty()) {
            msg["content"] = result.output + "\nError: " + result.error;
        } else if (!result.output.empty()) {
            msg["content"] = result.output;
        } else if (!result.error.empty()) {
            msg["content"] = "Error: " + result.error;
        } else {
            msg["content"] = "Error: Tool failed with no output";
        }
    }

    messages_.push_back(msg);
}

agent_loop_result agent_loop::run(const std::string & user_prompt) {
    agent_loop_result result;
    result.iterations = 0;

    // Add user message
    messages_.push_back({
        {"role", "user"},
        {"content", user_prompt}
    });

    while (result.iterations < config_.max_iterations) {
        if (is_interrupted_.load()) {
            result.stop_reason = agent_stop_reason::USER_CANCELLED;
            return result;
        }

        result.iterations++;

        if (config_.verbose && !is_subagent_) {
            console::log("\n[Iteration %d/%d]\n", result.iterations, config_.max_iterations);
        }

        // Generate completion - returns parsed message with tool calls
        result_timings timings;
        common_chat_msg parsed = generate_completion(timings);

        // Accumulate session statistics
        if (timings.prompt_n > 0) {
            stats_.total_input += timings.prompt_n;
            stats_.total_prompt_ms += timings.prompt_ms;
        }
        if (timings.predicted_n > 0) {
            stats_.total_output += timings.predicted_n;
            stats_.total_predicted_ms += timings.predicted_ms;
        }
        if (timings.cache_n > 0) {
            stats_.total_cached += timings.cache_n;
        }

        if (parsed.content.empty() && parsed.tool_calls.empty() && is_interrupted_.load()) {
            result.stop_reason = agent_stop_reason::USER_CANCELLED;
            return result;
        }

        // Add assistant message to history
        json assistant_msg;
        assistant_msg["role"] = "assistant";
        assistant_msg["content"] = parsed.content;

        if (!parsed.tool_calls.empty()) {
            assistant_msg["tool_calls"] = json::array();
            for (const auto & call : parsed.tool_calls) {
                json tc;
                tc["id"] = call.id.empty() ? ("call_" + std::to_string(result.iterations)) : call.id;
                tc["type"] = "function";
                tc["function"] = {
                    {"name", call.name},
                    {"arguments", call.arguments}
                };
                assistant_msg["tool_calls"].push_back(tc);
            }
        }

        messages_.push_back(assistant_msg);

        // If no tool calls, we're done
        if (parsed.tool_calls.empty()) {
            result.stop_reason = agent_stop_reason::COMPLETED;
            result.final_response = parsed.content;
            return result;
        }

        if (!is_subagent_) {
            console::log("\n");
        }

        // Execute each tool call
        for (const auto & call : parsed.tool_calls) {
            if (is_interrupted_.load()) {
                result.stop_reason = agent_stop_reason::USER_CANCELLED;
                return result;
            }

            tool_result tool_res = execute_tool_call(call);
            std::string call_id = call.id.empty() ? ("call_" + std::to_string(result.iterations)) : call.id;
            add_tool_result_message(call.name, call_id, tool_res);
        }
    }

    result.stop_reason = agent_stop_reason::MAX_ITERATIONS;
    result.final_response = "Reached maximum iterations (" + std::to_string(config_.max_iterations) + ")";
    return result;
}

// Streaming version of run() for API use
agent_loop_result agent_loop::run_streaming(
    const std::string & user_prompt,
    agent_event_callback on_event,
    std::function<bool()> should_stop,
    permission_manager_async * async_perms) {

    agent_loop_result result;
    result.iterations = 0;

    // Default should_stop to check is_interrupted_
    if (!should_stop) {
        should_stop = [this]() { return is_interrupted_.load(); };
    }

    // Add user message
    messages_.push_back({
        {"role", "user"},
        {"content", user_prompt}
    });

    while (result.iterations < config_.max_iterations) {
        if (should_stop()) {
            result.stop_reason = agent_stop_reason::USER_CANCELLED;
            on_event(agent_event::completed(result.stop_reason, stats_));
            return result;
        }

        result.iterations++;

        // Emit iteration start event
        on_event(agent_event::iteration_start(result.iterations, config_.max_iterations));

        // Generate completion with streaming
        result_timings timings;
        common_chat_msg parsed = generate_completion_streaming(timings, on_event, should_stop);

        // Accumulate session statistics
        if (timings.prompt_n > 0) {
            stats_.total_input += timings.prompt_n;
            stats_.total_prompt_ms += timings.prompt_ms;
        }
        if (timings.predicted_n > 0) {
            stats_.total_output += timings.predicted_n;
            stats_.total_predicted_ms += timings.predicted_ms;
        }
        if (timings.cache_n > 0) {
            stats_.total_cached += timings.cache_n;
        }

        if (parsed.content.empty() && parsed.tool_calls.empty() && should_stop()) {
            result.stop_reason = agent_stop_reason::USER_CANCELLED;
            on_event(agent_event::completed(result.stop_reason, stats_));
            return result;
        }

        // Add assistant message to history
        json assistant_msg;
        assistant_msg["role"] = "assistant";
        assistant_msg["content"] = parsed.content;

        if (!parsed.tool_calls.empty()) {
            assistant_msg["tool_calls"] = json::array();
            for (const auto & call : parsed.tool_calls) {
                json tc;
                tc["id"] = call.id.empty() ? ("call_" + std::to_string(result.iterations)) : call.id;
                tc["type"] = "function";
                tc["function"] = {
                    {"name", call.name},
                    {"arguments", call.arguments}
                };
                assistant_msg["tool_calls"].push_back(tc);
            }
        }

        messages_.push_back(assistant_msg);

        // If no tool calls, we're done
        if (parsed.tool_calls.empty()) {
            result.stop_reason = agent_stop_reason::COMPLETED;
            result.final_response = parsed.content;
            on_event(agent_event::completed(result.stop_reason, stats_));
            return result;
        }

        // Execute each tool call
        for (const auto & call : parsed.tool_calls) {
            if (should_stop()) {
                result.stop_reason = agent_stop_reason::USER_CANCELLED;
                on_event(agent_event::completed(result.stop_reason, stats_));
                return result;
            }

            // Emit tool start event
            on_event(agent_event::tool_start(call.name, call.arguments));

            auto start_time = std::chrono::steady_clock::now();

            // Use async permission handling if async_perms is provided
            tool_result tool_res = async_perms
                ? execute_tool_call_async(call, on_event, *async_perms, should_stop)
                : execute_tool_call(call);

            auto end_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            // Emit tool result event
            on_event(agent_event::tool_result(call.name, tool_res.success, tool_res.output, elapsed_ms));

            std::string call_id = call.id.empty() ? ("call_" + std::to_string(result.iterations)) : call.id;
            add_tool_result_message(call.name, call_id, tool_res);
        }
    }

    result.stop_reason = agent_stop_reason::MAX_ITERATIONS;
    result.final_response = "Reached maximum iterations (" + std::to_string(config_.max_iterations) + ")";
    on_event(agent_event::completed(result.stop_reason, stats_));
    return result;
}

// Streaming version of generate_completion
common_chat_msg agent_loop::generate_completion_streaming(
    result_timings & out_timings,
    agent_event_callback on_event,
    std::function<bool()> should_stop) {

    server_response_reader rd = server_ctx_.get_response_reader();
    {
        std::lock_guard<std::mutex> lock(g_completion_mutex);

        server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
        task.id        = rd.get_new_id();
        task.index     = 0;
        task.params    = task_defaults_;

        auto chat_tools = allowed_tools_.empty()
            ? tool_registry::instance().to_chat_tools()
            : tool_registry::instance().to_chat_tools_filtered(allowed_tools_);
        json tools_json = common_chat_tools_to_json_oaicompat<json>(chat_tools);

        task.cli_input = json::object();
        task.cli_input["messages"] = json::parse(messages_.dump());
        task.cli_input["tools"] = json::parse(tools_json.dump());
        task.cli_input["tool_choice"] = "auto";

        rd.post_task(std::move(task));
    }

    server_task_result_ptr result = rd.next(should_stop);

    std::string full_content;
    bool was_aborted = false;

    while (result) {
        if (should_stop()) {
            was_aborted = true;
            break;
        }
        if (result->is_error()) {
            json err_data = result->to_json();
            std::string err_msg = err_data.value("message", "Unknown error");
            on_event(agent_event::error(err_msg));
            common_chat_msg empty_msg;
            return empty_msg;
        }

        auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
        if (res_partial) {
            out_timings = std::move(res_partial->timings);
            for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                if (!diff.content_delta.empty()) {
                    on_event(agent_event::text_delta(diff.content_delta));
                    full_content += diff.content_delta;
                }
                if (!diff.reasoning_content_delta.empty()) {
                    on_event(agent_event::reasoning_delta(diff.reasoning_content_delta));
                }
            }
        }

        auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        if (res_final) {
            out_timings = std::move(res_final->timings);
            if (!res_final->oaicompat_msg.empty()) {
                return res_final->oaicompat_msg;
            }
            if (!res_final->content.empty()) {
                full_content = res_final->content;
            }
            break;
        }

        result = rd.next(should_stop);
    }

    if (was_aborted) {
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = full_content;
        return msg;
    }

    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = full_content;
    return msg;
}

// Async version of execute_tool_call for API use
// Uses async permission manager and emits events instead of blocking on console
tool_result agent_loop::execute_tool_call_async(
    const common_chat_tool_call & call,
    agent_event_callback on_event,
    permission_manager_async & async_perms,
    std::function<bool()> should_stop) {

    auto & registry = tool_registry::instance();

    // Check if tool exists
    const tool_def * tool = registry.get_tool(call.name);
    if (!tool) {
        return {false, "", "Unknown tool: " + call.name};
    }

    // Parse arguments
    json args;
    try {
        args = json::parse(call.arguments);
    } catch (const json::parse_error & e) {
        return {false, "", std::string("Invalid JSON arguments: ") + e.what()};
    }

    // Determine permission type
    permission_type ptype = permission_type::BASH;
    if (call.name == "read") {
        ptype = permission_type::FILE_READ;
    } else if (call.name == "write") {
        ptype = permission_type::FILE_WRITE;
    } else if (call.name == "edit") {
        ptype = permission_type::FILE_EDIT;
    } else if (call.name == "glob") {
        ptype = permission_type::GLOB;
    }

    // Build permission request
    permission_request req;
    req.type = ptype;
    req.tool_name = call.name;
    req.details = call.arguments;

    // Check for external directory access on file operations
    if (call.name == "read" || call.name == "write" || call.name == "edit") {
        std::string file_path = args.value("file_path", "");
        if (!file_path.empty()) {
            std::filesystem::path path(file_path);
            if (path.is_relative()) {
                path = std::filesystem::path(tool_ctx_.working_dir) / path;
            }
            if (async_perms.is_external_path(path.string())) {
                permission_request ext_req;
                ext_req.type = permission_type::EXTERNAL_DIR;
                ext_req.tool_name = call.name;
                ext_req.details = "External file: " + path.string();
                ext_req.is_dangerous = true;
                ext_req.description = "Operation outside working directory";

                // Request permission asynchronously
                std::string req_id = async_perms.request_permission(ext_req);
                on_event(agent_event::permission_required(req_id, call.name, ext_req.details, true));

                // Wait for response
                auto response = async_perms.wait_for_response(req_id, 300000); // 5 min timeout
                if (!response || !response->allowed) {
                    on_event(agent_event::permission_resolved(req_id, false));
                    return {false, "", "Blocked: File is outside working directory"};
                }
                on_event(agent_event::permission_resolved(req_id, true));
            }
        }
    }

    // Check for dangerous commands
    if (call.name == "bash") {
        std::string cmd = args.value("command", "");
        req.details = cmd;
        for (const auto & pattern : {"rm -rf", "sudo ", "chmod 777"}) {
            if (cmd.find(pattern) != std::string::npos) {
                req.is_dangerous = true;
                break;
            }
        }
    }

    // Check doom loop
    std::hash<std::string> hasher;
    std::string args_hash = std::to_string(hasher(call.arguments));
    if (async_perms.is_doom_loop(call.name, args_hash)) {
        req.description = "Detected repeated identical tool calls (doom loop)";

        std::string req_id = async_perms.request_permission(req);
        on_event(agent_event::permission_required(req_id, call.name, req.details, true));

        auto response = async_perms.wait_for_response(req_id, 300000);
        if (!response || !response->allowed) {
            on_event(agent_event::permission_resolved(req_id, false));
            return {false, "", "Blocked: Detected repeated identical tool calls"};
        }
        on_event(agent_event::permission_resolved(req_id, true));
    }

    // Check permission
    permission_state state = async_perms.check_permission(req);
    if (state == permission_state::DENY || state == permission_state::DENY_SESSION) {
        return {false, "", "Permission denied for " + call.name};
    }

    if (state == permission_state::ASK) {
        // Request permission asynchronously
        std::string req_id = async_perms.request_permission(req);
        on_event(agent_event::permission_required(req_id, call.name, req.details, req.is_dangerous));

        // Wait for response (with timeout)
        auto response = async_perms.wait_for_response(req_id, 300000); // 5 min timeout

        if (should_stop()) {
            async_perms.cancel(req_id);
            return {false, "", "Operation cancelled"};
        }

        if (!response) {
            on_event(agent_event::permission_resolved(req_id, false));
            return {false, "", "Permission request timed out"};
        }

        on_event(agent_event::permission_resolved(req_id, response->allowed));

        if (!response->allowed) {
            return {false, "", "User denied permission for " + call.name};
        }
    }

    // Record this call
    async_perms.record_tool_call(call.name, args_hash);

    // Execute the tool
    tool_result result = bash_patterns_.empty()
        ? registry.execute(call.name, args, tool_ctx_)
        : registry.execute_filtered(call.name, args, tool_ctx_, bash_patterns_);

    return result;
}
