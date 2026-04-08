#include "agent-loop.h"
#include "agent-loop-internal.h"
#include "console.h"
#include "terminal-image.h"

#include <chrono>
#include <sstream>

// ---------------------------------------------------------------------------
// Constructor / lifecycle
// ---------------------------------------------------------------------------

agent_loop::agent_loop(server_context & server_ctx,
                       const common_params & params,
                       const agent_config & config,
                       std::atomic<bool> & is_interrupted,
                       session_file * sf,
                       const loaded_session * resume)
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

    // Cache vocab pointer — valid for the lifetime of the model (survives sleep/wake).
    auto * lctx = server_ctx.get_llama_context();
    GGML_ASSERT(lctx != nullptr && "llama_context must be available at agent construction");
    vocab_ = llama_model_get_vocab(llama_get_model(lctx));

    // Initialize tool context
    tool_ctx_.working_dir = config.working_dir.empty() ? "." : config.working_dir;
    tool_ctx_.is_interrupted = &is_interrupted_;
    tool_ctx_.timeout_ms = config.tool_timeout_ms;
    tool_ctx_.has_vision = server_ctx_.get_meta().has_inp_image;

    // Set up permission manager
    permission_mgr_.set_project_root(tool_ctx_.working_dir);
    permission_mgr_.set_yolo_mode(config.yolo_mode);

    // Add system prompt for tool usage
    const bool has_vision = tool_ctx_.has_vision;
    std::string system_prompt = R"(You are llama-agent, a powerful local AI coding assistant running on llama.cpp.

You help users with software engineering tasks by reading files, writing code, running commands, and navigating codebases. You run entirely on the user's machine - no data leaves their system.
)";
    if (has_vision) {
        system_prompt += R"(
You have vision capabilities. When you use the `read` tool on an image file, you will see the image and can analyze its visual contents. Use this to identify objects, read text in screenshots, understand diagrams, classify images, and more.

)";
    }
    system_prompt += R"(# Tools

You have access to the following tools:

- **bash**: Execute shell commands. Use for git, build commands, running tests, etc.
)";
    if (has_vision) {
        system_prompt += "- **read**: Read file contents with line numbers. Can also read image files (png, jpg, gif, webp, bmp) for visual analysis. Always read files before editing them.\n";
    } else {
        system_prompt += "- **read**: Read file contents with line numbers. Always read files before editing them.\n";
    }
    system_prompt += R"(
- **write**: Create new files or overwrite existing ones.
- **edit**: Make targeted edits using search/replace. The old_string must match exactly. Use replace_all=true to replace all occurrences of a word or phrase.
- **glob**: Find files matching a pattern (e.g. `*.cpp`, `*.{jpg,png}`). Use to explore project structure.
- **update_plan**: Update and display your task plan. Use for multi-step tasks to show progress while staying in the tool-calling loop.

## Using the edit tool
The edit tool finds and replaces text in files. Key points:
- **old_string must match exactly** - include correct whitespace and indentation
- **Always read the file first** - so you know the exact text to match
- **Prefer small, targeted edits** - make multiple small edits rather than one large replacement. This reduces the chance of old_string mismatches.
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

## Stay on task
- Keep going until the task is completely resolved before yielding back to the user.
- If work remains, always include a tool call. Never emit a bare text "progress report" without one.
- Use `update_plan` to communicate progress — it shows the user your status while keeping you in the loop.
- Only respond with plain text when the task is fully complete or you need clarification that blocks all progress.

# Tool Usage

## Parallel execution
When multiple operations are independent, execute them together. For example, reading multiple files or running independent commands.

## Search strategy
When looking for code:
1. Use `glob` to find candidate files
2. Use `read` to examine promising files
3. Use `bash` with grep for text search across files

## update_plan
For multi-step tasks, use `update_plan` to track progress:
1. Call it at the start to outline steps (all "pending").
2. Set each step to "in_progress" then "completed" as you work.
3. Always follow `update_plan` with the next tool call — never stop after it.

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

    // Inject environment context (working directory + date)
    {
        system_prompt += "\n# Environment\n\n";
        system_prompt += "Current working directory: " + tool_ctx_.working_dir + "\n";

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#if defined(_WIN32)
        localtime_s(&tm_buf, &time);
#else
        localtime_r(&time, &tm_buf);
#endif
        char date_buf[11];
        std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", &tm_buf);
        system_prompt += "Current date: " + std::string(date_buf) + "\n";
    }

    messages_.push_back({
        {"role", "system"},
        {"content", system_prompt}
    });

    // Session persistence
    session_file_ = sf;

    if (resume && !resume->messages.empty()) {
        // Reinsert compaction summary so the LLM has pre-compaction context
        if (!resume->previous_summary.empty()) {
            messages_.push_back({
                {"role", "user"},
                {"content", "<context-summary>\n" + resume->previous_summary + "\n</context-summary>"}
            });
        }
        for (const auto & m : resume->messages) {
            messages_.push_back(m);
        }
        previous_summary_ = resume->previous_summary;
    } else if (session_file_) {
        session_file_->write_header(config_.working_dir);
    }
}

void agent_loop::clear() {
    // Keep system prompt, clear rest
    if (messages_.size() > 1) {
        json system_msg = messages_[0];
        messages_ = json::array();
        messages_.push_back(system_msg);
    }
    permission_mgr_.clear_session();

    // Reset compaction state
    previous_summary_.clear();
    last_prompt_tokens_ = 0;
    last_completion_overflowed_ = false;

    // Reset session file
    if (session_file_) {
        session_file_->reopen();
        session_file_->write_header(config_.working_dir);
    }

    // Reset stats when conversation is cleared
    stats_ = session_stats{};
}

void agent_loop::add_context_message(const std::string & role, const std::string & content) {
    messages_.push_back({{"role", role}, {"content", content}});
    if (session_file_) {
        session_file_->append_message(messages_.back());
    }
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

json agent_loop::build_oai_request_body(const std::vector<common_chat_tool> & chat_tools,
                                        bool has_vision) {
    // Deep copy messages so we don't mutate the canonical messages_.
    json messages = messages_;

    // Strip reasoning_content from all but the last assistant message.
    // Gemma 4 (and similar models) re-summarize prior reasoning on every turn,
    // causing repetitive and ever-growing chain-of-thought.  Google recommends:
    // "Thoughts from previous model turns must not be added before the next
    //  user turn begins."
    for (int i = (int)messages.size() - 2; i >= 0; --i) {
        if (messages[i].value("role", "") == "assistant") {
            messages[i].erase("reasoning_content");
        }
    }

    // Strip image_url blocks when the model lacks vision support to avoid
    // oaicompat_chat_params_parse throwing "image input is not supported".
    if (!has_vision) {
        for (auto & msg : messages) {
            if (!msg.contains("content") || !msg["content"].is_array()) {
                continue;
            }
            json filtered = json::array();
            for (const auto & part : msg["content"]) {
                if (part.value("type", "") != "image_url") {
                    filtered.push_back(part);
                }
            }
            msg["content"] = filtered;
        }
    }

    json body;
    body["messages"]          = std::move(messages);
    body["tools"]             = common_chat_tools_to_json_oaicompat(chat_tools);
    body["stream"]            = true;
    body["timings_per_token"] = true;
    return body;
}

// ---------------------------------------------------------------------------
// Message helpers
// ---------------------------------------------------------------------------

void agent_loop::add_tool_result_message(const std::string & tool_name,
                                          const std::string & call_id,
                                          const tool_result & result) {
    json msg;
    msg["role"] = "tool";
    msg["tool_call_id"] = call_id;
    msg["name"] = tool_name;

    if (result.success) {
        if (!result.content.empty()) {
            msg["content"] = result.content;  // structured content array (e.g. text + image)
        } else {
            msg["content"] = result.output;
        }
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
    if (session_file_) session_file_->append_message(msg);
}

json agent_loop::build_assistant_msg(const common_chat_msg & parsed, int iteration) {
    json msg;
    msg["role"] = "assistant";
    msg["content"] = parsed.content;

    if (!parsed.reasoning_content.empty()) {
        msg["reasoning_content"] = parsed.reasoning_content;
    }

    if (!parsed.tool_calls.empty()) {
        msg["tool_calls"] = json::array();
        for (const auto & call : parsed.tool_calls) {
            json tc;
            tc["id"] = call.id.empty() ? ("call_" + std::to_string(iteration)) : call.id;
            tc["type"] = "function";
            tc["function"] = {
                {"name", call.name},
                {"arguments", call.arguments}
            };
            msg["tool_calls"].push_back(tc);
        }
    }

    return msg;
}

void agent_loop::accumulate_stats(const result_timings & timings) {
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
}

// ---------------------------------------------------------------------------
// Main agent loops
// ---------------------------------------------------------------------------

agent_loop_result agent_loop::run(const json & user_content) {
    agent_loop_result result;
    result.iterations = 0;

    // Add user message (content can be a string or an array of content blocks)
    messages_.push_back({
        {"role", "user"},
        {"content", user_content}
    });
    if (session_file_) session_file_->append_message(messages_.back());

    while (config_.max_iterations <= 0 || result.iterations < config_.max_iterations) {
        if (is_interrupted_.load()) {
            result.stop_reason = agent_stop_reason::USER_CANCELLED;
            return result;
        }

        result.iterations++;

        if (config_.verbose) {
            if (config_.max_iterations > 0) {
                console::log("\n[Iteration %d/%d]\n", result.iterations, config_.max_iterations);
            } else {
                console::log("\n[Iteration %d]\n", result.iterations);
            }
        }

        // Generate completion - returns parsed message with tool calls
        result_timings timings;
        common_chat_msg parsed = generate_completion(timings);

        accumulate_stats(timings);

        // Overflow recovery: compact and retry this iteration
        if (parsed.content.empty() && parsed.tool_calls.empty() && last_completion_overflowed_) {
            last_completion_overflowed_ = false;
            if (config_.compaction.enabled && try_compact()) {
                console::log("[Context compacted, retrying...]\n");
                result.iterations--;
                continue;
            }
        }

        if (parsed.content.empty() && parsed.tool_calls.empty() && is_interrupted_.load()) {
            result.stop_reason = agent_stop_reason::USER_CANCELLED;
            return result;
        }

        // Threshold compaction after successful completion
        if (config_.compaction.enabled) {
            try_compact();
        }

        // Empty response — don't save to history, just end the turn
        if (parsed.content.empty() && parsed.tool_calls.empty()) {
            result.stop_reason = agent_stop_reason::COMPLETED;
            result.final_response = "";
            return result;
        }

        // Add assistant message to history
        json assistant_msg = build_assistant_msg(parsed, result.iterations);
        messages_.push_back(assistant_msg);
        if (session_file_) session_file_->append_message(assistant_msg);

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

// Streaming version of run() for API use
agent_loop_result agent_loop::run_streaming(
    const json & user_content,
    agent_event_callback on_event,
    std::function<bool()> should_stop,
    permission_manager_async * async_perms) {

    agent_loop_result result;
    result.iterations = 0;

    // Default should_stop to check is_interrupted_
    if (!should_stop) {
        should_stop = [this]() { return is_interrupted_.load(); };
    }

    // Add user message (content can be a string or an array of content blocks)
    messages_.push_back({
        {"role", "user"},
        {"content", user_content}
    });
    if (session_file_) session_file_->append_message(messages_.back());

    while (config_.max_iterations <= 0 || result.iterations < config_.max_iterations) {
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

        accumulate_stats(timings);

        // Overflow recovery: compact and retry this iteration
        if (parsed.content.empty() && parsed.tool_calls.empty() && last_completion_overflowed_) {
            last_completion_overflowed_ = false;
            if (config_.compaction.enabled && try_compact()) {
                on_event(agent_event::compaction_completed((int32_t) messages_.size()));
                result.iterations--;
                continue;
            }
            on_event(agent_event::error("Context overflow: compaction could not free enough space"));
            result.stop_reason = agent_stop_reason::AGENT_ERROR;
            on_event(agent_event::completed(result.stop_reason, stats_));
            return result;
        }

        if (parsed.content.empty() && parsed.tool_calls.empty() && should_stop()) {
            result.stop_reason = agent_stop_reason::USER_CANCELLED;
            on_event(agent_event::completed(result.stop_reason, stats_));
            return result;
        }

        // Threshold compaction after successful completion
        if (config_.compaction.enabled) {
            if (try_compact()) {
                on_event(agent_event::compaction_completed((int32_t) messages_.size()));
            }
        }

        // Empty response — don't save to history, just end the turn
        if (parsed.content.empty() && parsed.tool_calls.empty()) {
            result.stop_reason = agent_stop_reason::COMPLETED;
            result.final_response = "";
            on_event(agent_event::completed(result.stop_reason, stats_));
            return result;
        }

        // Add assistant message to history
        json assistant_msg = build_assistant_msg(parsed, result.iterations);
        messages_.push_back(assistant_msg);
        if (session_file_) session_file_->append_message(assistant_msg);

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
