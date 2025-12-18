#include "agent-loop.h"
#include "console.h"

#include <chrono>
#include <functional>
#include <sstream>

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

    // Set up permission manager
    permission_mgr_.set_project_root(tool_ctx_.working_dir);
    permission_mgr_.set_yolo_mode(config.yolo_mode);

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

    messages_.push_back({
        {"role", "system"},
        {"content", system_prompt}
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
}

// Parse a single function block: <function=name>...<parameter=key>value</parameter>...</function>
static bool parse_function_block(const std::string & block, common_chat_tool_call & tc) {
    // Parse function name: <function=name>
    size_t func_start = block.find("<function=");
    if (func_start == std::string::npos) return false;

    size_t func_name_start = func_start + 10;
    size_t func_name_end = block.find(">", func_name_start);
    if (func_name_end == std::string::npos) return false;

    tc.name = block.substr(func_name_start, func_name_end - func_name_start);

    // Find function block end
    size_t func_end = block.find("</function>", func_name_end);
    if (func_end == std::string::npos) func_end = block.size();

    std::string func_body = block.substr(func_name_end + 1, func_end - func_name_end - 1);

    // Parse parameters
    json args = json::object();
    size_t param_pos = 0;
    while ((param_pos = func_body.find("<parameter=", param_pos)) != std::string::npos) {
        size_t param_name_start = param_pos + 11;
        size_t param_name_end = func_body.find(">", param_name_start);
        if (param_name_end == std::string::npos) break;

        std::string param_name = func_body.substr(param_name_start, param_name_end - param_name_start);

        // Find parameter value (between > and </parameter> or next <parameter=)
        size_t value_start = param_name_end + 1;
        // Skip leading newline if present
        if (value_start < func_body.size() && func_body[value_start] == '\n') {
            value_start++;
        }

        size_t param_end = func_body.find("</parameter>", value_start);
        size_t next_param = func_body.find("<parameter=", value_start);

        size_t value_end;
        if (param_end != std::string::npos && (next_param == std::string::npos || param_end < next_param)) {
            value_end = param_end;
        } else if (next_param != std::string::npos) {
            value_end = next_param;
        } else {
            value_end = func_body.size();
        }

        std::string param_value = func_body.substr(value_start, value_end - value_start);
        // Trim trailing newline/whitespace
        while (!param_value.empty() && (param_value.back() == '\n' || param_value.back() == '\r')) {
            param_value.pop_back();
        }
        // Trim leading/trailing whitespace for type inference
        std::string trimmed = param_value;
        while (!trimmed.empty() && std::isspace(trimmed.front())) trimmed.erase(0, 1);
        while (!trimmed.empty() && std::isspace(trimmed.back())) trimmed.pop_back();

        // Convert to appropriate JSON type
        std::string lower_trimmed = trimmed;
        for (auto & c : lower_trimmed) c = std::tolower(c);

        if (lower_trimmed == "true") {
            args[param_name] = true;
        } else if (lower_trimmed == "false") {
            args[param_name] = false;
        } else {
            // Try to parse as number
            bool is_number = !trimmed.empty();
            bool has_dot = false;
            for (size_t i = 0; i < trimmed.size(); i++) {
                char c = trimmed[i];
                if (c == '-' && i == 0) continue;
                if (c == '.' && !has_dot) { has_dot = true; continue; }
                if (!std::isdigit(c)) { is_number = false; break; }
            }
            if (is_number && !trimmed.empty() && trimmed != "-" && trimmed != ".") {
                if (has_dot) {
                    args[param_name] = std::stod(trimmed);
                } else {
                    args[param_name] = std::stoll(trimmed);
                }
            } else {
                args[param_name] = param_value;
            }
        }
        param_pos = value_end;
    }

    tc.arguments = args.dump();
    return true;
}

// Parse tool calls from qwen3-coder/nemotron XML format
// Supports both:
//   <tool_call><function=name>...</function></tool_call>
//   <function=name>...</function> (without wrapper)
static common_chat_msg parse_tool_calls_xml(const std::string & content) {
    common_chat_msg msg;
    msg.role = "assistant";

    std::string remaining = content;

    // First, try to find <tool_call> wrapped format
    size_t tool_call_start = remaining.find("<tool_call>");
    // If no <tool_call>, look for bare <function= tags
    size_t func_start = remaining.find("<function=");

    // Determine the earliest tool/function occurrence
    size_t first_tool = std::string::npos;
    bool has_wrapper = false;
    if (tool_call_start != std::string::npos && (func_start == std::string::npos || tool_call_start < func_start)) {
        first_tool = tool_call_start;
        has_wrapper = true;
    } else if (func_start != std::string::npos) {
        first_tool = func_start;
        has_wrapper = false;
    }

    // Extract content before any tool calls
    if (first_tool != std::string::npos) {
        msg.content = remaining.substr(0, first_tool);
        // Trim trailing whitespace from content
        while (!msg.content.empty() && std::isspace(msg.content.back())) {
            msg.content.pop_back();
        }
    } else {
        msg.content = content;
        return msg;  // No tool calls
    }

    // Parse tool calls
    if (has_wrapper) {
        // Parse <tool_call>...<function=...>...</function>...</tool_call> format
        while ((tool_call_start = remaining.find("<tool_call>")) != std::string::npos) {
            size_t tool_call_end = remaining.find("</tool_call>", tool_call_start);
            if (tool_call_end == std::string::npos) break;

            std::string tool_block = remaining.substr(tool_call_start + 11, tool_call_end - tool_call_start - 11);
            remaining = remaining.substr(tool_call_end + 12);

            common_chat_tool_call tc;
            tc.id = "call_" + std::to_string(msg.tool_calls.size());
            if (parse_function_block(tool_block, tc)) {
                msg.tool_calls.push_back(tc);
            }
        }
    } else {
        // Parse bare <function=...>...</function> format
        while ((func_start = remaining.find("<function=")) != std::string::npos) {
            size_t func_end = remaining.find("</function>", func_start);
            if (func_end == std::string::npos) break;

            std::string func_block = remaining.substr(func_start, func_end - func_start + 11);
            remaining = remaining.substr(func_end + 11);

            common_chat_tool_call tc;
            tc.id = "call_" + std::to_string(msg.tool_calls.size());
            if (parse_function_block(func_block, tc)) {
                msg.tool_calls.push_back(tc);
            }
        }
    }

    return msg;
}

common_chat_msg agent_loop::generate_completion(result_timings & out_timings) {
    server_response_reader rd = server_ctx_.get_response_reader();
    {
        server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
        task.id        = rd.get_new_id();
        task.index     = 0;
        task.params    = task_defaults_;

        // Build tools JSON in OAI-compatible format
        auto chat_tools = tool_registry::instance().to_chat_tools();
        json tools_json = common_chat_tools_to_json_oaicompat<json>(chat_tools);

        // Pass both messages and tools as extended cli_input format
        task.cli_input = {
            {"messages", messages_},
            {"tools", tools_json},
            {"tool_choice", "auto"}
        };
        rd.post_task({std::move(task)});
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

    // Wait for first result
    console::spinner::start();
    server_task_result_ptr result = rd.next(should_stop);

    console::spinner::stop();
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
            if (err_data.contains("message")) {
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
                    if (is_thinking) {
                        console::log("\n───\n\n");
                        console::set_display(DISPLAY_TYPE_RESET);
                        is_thinking = false;
                    }
                    full_content += diff.content_delta;
                    console::log("%s", diff.content_delta.c_str());
                    console::flush();
                }
                if (!diff.reasoning_content_delta.empty()) {
                    console::set_display(DISPLAY_TYPE_REASONING);
                    if (!is_thinking) {
                        console::log("───\n");
                    }
                    is_thinking = true;
                    console::log("%s", diff.reasoning_content_delta.c_str());
                    console::flush();
                }
            }
        }

        auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        if (res_final) {
            out_timings = std::move(res_final->timings);
            // Use the raw content for our own parsing
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
        console::log("\n[Generation aborted]\n");
        // Return partial content without tool calls so conversation can continue
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = full_content;
        return msg;
    }

    // Parse tool calls ourselves using the qwen3-coder/nemotron XML format
    return parse_tool_calls_xml(full_content);
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

    // Display tool execution
    console::set_display(DISPLAY_TYPE_INFO);
    console::log("\n› %s\n", call.name.c_str());
    console::set_display(DISPLAY_TYPE_RESET);

    // Execute the tool with timing
    auto start_time = std::chrono::steady_clock::now();
    tool_result result = registry.execute(call.name, args, tool_ctx_);
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Display result summary
    if (result.success) {
        // Truncate long output for display
        std::string display_output = result.output;
        if (display_output.length() > 500) {
            display_output = display_output.substr(0, 500) + "\n... (truncated)";
        }
        console::log("%s\n", display_output.c_str());
    } else {
        console::error("Error: %s\n", result.error.c_str());
    }

    // Display elapsed time
    console::set_display(DISPLAY_TYPE_INFO);
    if (elapsed_ms < 1000) {
        console::log("└─ %lldms\n", elapsed_ms);
    } else {
        console::log("└─ %.1fs\n", elapsed_ms / 1000.0);
    }
    console::set_display(DISPLAY_TYPE_RESET);

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
        msg["content"] = "Error: " + result.error;
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

        if (config_.verbose) {
            console::log("\n[Iteration %d/%d]\n", result.iterations, config_.max_iterations);
        }

        // Generate completion - returns parsed message with tool calls
        result_timings timings;
        common_chat_msg parsed = generate_completion(timings);

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

        console::log("\n");

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
