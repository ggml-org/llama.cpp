#include "agent-loop.h"
#include "console.h"

#include <functional>
#include <sstream>

agent_loop::agent_loop(server_context & server_ctx,
                       const common_params & params,
                       const agent_config & config,
                       std::atomic<bool> & is_interrupted)
    : server_ctx_(server_ctx)
    , params_(params)
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

    // Add system prompt for tool usage
    std::string system_prompt = R"(You are an AI coding assistant with access to tools. Use tools to help the user with their coding tasks.

Available tools:
- bash: Execute shell commands
- read: Read file contents
- write: Create or overwrite files
- edit: Make targeted edits to files using search/replace
- glob: Find files matching a pattern

When using tools, think step by step about what you need to do. After getting tool results, analyze them and continue working toward the user's goal. When the task is complete, provide a summary of what you did.)";

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

        args[param_name] = param_value;
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

    auto should_stop = [this]() { return is_interrupted_.load(); };

    // Wait for first result
    console::spinner::start();
    server_task_result_ptr result = rd.next(should_stop);

    console::spinner::stop();
    std::string full_content;
    bool is_thinking = false;

    while (result) {
        if (should_stop()) {
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
                        console::log("\n[End thinking]\n\n");
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
                        console::log("[Start thinking]\n");
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

    is_interrupted_.store(false);

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
    console::log("\n[Tool: %s]\n", call.name.c_str());
    console::set_display(DISPLAY_TYPE_RESET);

    // Execute the tool
    tool_result result = registry.execute(call.name, args, tool_ctx_);

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
