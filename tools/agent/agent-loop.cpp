#include "agent-loop.h"
#include "console.h"
#include "terminal-image.h"

#include <chrono>
#include <functional>
#include <mutex>
#include <sstream>

// Protect the shared prompt-formatting + task-posting path without serializing
// the full generation loop.
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

// Per-tool-call state used during streaming to display args incrementally
struct tool_stream_state {
    std::string accumulated_args;  // full JSON args text accumulated so far
    std::string name;              // tool name (set on first delta)
    bool header_printed = false;   // whether "› tool_name path" was shown
    std::string displayed_path;    // file_path already printed
    size_t displayed_lines = 0;    // number of content lines already printed
    size_t displayed_bytes = 0;    // byte offset in content_buffer past last printed line
    size_t content_scan_pos = 0;   // byte offset in accumulated_args where content value starts (0 = not found)
    std::string content_buffer;    // decoded content accumulated so far (avoids re-decoding from start)
    size_t content_raw_end = 0;    // how many bytes of accumulated_args have been decoded into content_buffer
    bool content_complete = false; // true once the closing quote of the content field was found
};

// Extract a string-valued field from a partial (possibly incomplete) JSON object.
// Returns the decoded string value, or empty string if the key is not yet present.
// `complete` is set to true if the closing quote of the value was found.
static std::string extract_partial_json_string(
    const std::string & json_fragment,
    const std::string & key,
    bool & complete)
{
    complete = false;
    // Look for "key": or "key" :
    std::string needle = "\"" + key + "\"";
    auto kpos = json_fragment.find(needle);
    if (kpos == std::string::npos) {
        return "";
    }
    // Skip past key, then find ':'
    size_t pos = kpos + needle.size();
    while (pos < json_fragment.size() && json_fragment[pos] == ' ') pos++;
    if (pos >= json_fragment.size() || json_fragment[pos] != ':') {
        return "";
    }
    pos++; // skip ':'
    while (pos < json_fragment.size() && json_fragment[pos] == ' ') pos++;
    if (pos >= json_fragment.size() || json_fragment[pos] != '"') {
        return "";  // value not started yet or not a string
    }
    pos++; // skip opening quote

    // Scan value, handling JSON escape sequences
    std::string result;
    result.reserve(json_fragment.size() - pos);
    for (size_t i = pos; i < json_fragment.size(); ) {
        char c = json_fragment[i];
        if (c == '"') {
            complete = true;
            break;
        }
        if (c == '\\' && i + 1 < json_fragment.size()) {
            char esc = json_fragment[i + 1];
            switch (esc) {
                case '"':  result += '"';  i += 2; break;
                case '\\': result += '\\'; i += 2; break;
                case '/':  result += '/';  i += 2; break;
                case 'n':  result += '\n'; i += 2; break;
                case 'r':  result += '\r'; i += 2; break;
                case 't':  result += '\t'; i += 2; break;
                case 'b':  result += '\b'; i += 2; break;
                case 'f':  result += '\f'; i += 2; break;
                case 'u': {
                    if (i + 5 < json_fragment.size()) {
                        char hex[5] = {
                            json_fragment[i+2], json_fragment[i+3],
                            json_fragment[i+4], json_fragment[i+5], 0
                        };
                        unsigned cp = std::strtoul(hex, nullptr, 16);
                        if (cp < 0x80) {
                            result += (char)cp;
                        } else if (cp < 0x800) {
                            result += (char)(0xC0 | (cp >> 6));
                            result += (char)(0x80 | (cp & 0x3F));
                        } else {
                            result += (char)(0xE0 | (cp >> 12));
                            result += (char)(0x80 | ((cp >> 6) & 0x3F));
                            result += (char)(0x80 | (cp & 0x3F));
                        }
                        i += 6;
                    } else {
                        return result; // incomplete \uXXXX, stop here
                    }
                    break;
                }
                default:
                    result += esc;
                    i += 2;
                    break;
            }
        } else {
            result += c;
            i++;
        }
    }
    return result;
}

// Incrementally decode more of a JSON string field into tcs.content_buffer.
// Finds the field start once (caching in content_scan_pos), then decodes only
// new bytes on each call.  Returns true if the field's closing quote was found.
static bool decode_field_incremental(
    tool_stream_state & tcs,
    const std::string & key)
{
    // Find the value start position once
    if (tcs.content_scan_pos == 0) {
        std::string needle = "\"" + key + "\"";
        auto kpos = tcs.accumulated_args.find(needle);
        if (kpos == std::string::npos) return false;
        size_t pos = kpos + needle.size();
        while (pos < tcs.accumulated_args.size() && tcs.accumulated_args[pos] == ' ') pos++;
        if (pos >= tcs.accumulated_args.size() || tcs.accumulated_args[pos] != ':') return false;
        pos++;
        while (pos < tcs.accumulated_args.size() && tcs.accumulated_args[pos] == ' ') pos++;
        if (pos >= tcs.accumulated_args.size() || tcs.accumulated_args[pos] != '"') return false;
        pos++; // skip opening quote
        tcs.content_scan_pos = pos;
        tcs.content_raw_end = pos;
    }

    // Decode only the new bytes since last call
    const auto & s = tcs.accumulated_args;
    for (size_t i = tcs.content_raw_end; i < s.size(); ) {
        char c = s[i];
        if (c == '"') {
            tcs.content_raw_end = i + 1;
            return true; // field complete
        }
        if (c == '\\') {
            if (i + 1 >= s.size()) {
                // Backslash at end of buffer — wait for the escape char
                tcs.content_raw_end = i;
                return false;
            }
            char esc = s[i + 1];
            switch (esc) {
                case '"':  tcs.content_buffer += '"';  i += 2; break;
                case '\\': tcs.content_buffer += '\\'; i += 2; break;
                case '/':  tcs.content_buffer += '/';  i += 2; break;
                case 'n':  tcs.content_buffer += '\n'; i += 2; break;
                case 'r':  tcs.content_buffer += '\r'; i += 2; break;
                case 't':  tcs.content_buffer += '\t'; i += 2; break;
                case 'b':  tcs.content_buffer += '\b'; i += 2; break;
                case 'f':  tcs.content_buffer += '\f'; i += 2; break;
                case 'u': {
                    if (i + 5 < s.size()) {
                        char hex[5] = { s[i+2], s[i+3], s[i+4], s[i+5], 0 };
                        unsigned cp = std::strtoul(hex, nullptr, 16);
                        if (cp < 0x80) {
                            tcs.content_buffer += (char)cp;
                        } else if (cp < 0x800) {
                            tcs.content_buffer += (char)(0xC0 | (cp >> 6));
                            tcs.content_buffer += (char)(0x80 | (cp & 0x3F));
                        } else {
                            tcs.content_buffer += (char)(0xE0 | (cp >> 12));
                            tcs.content_buffer += (char)(0x80 | ((cp >> 6) & 0x3F));
                            tcs.content_buffer += (char)(0x80 | (cp & 0x3F));
                        }
                        i += 6;
                    } else {
                        tcs.content_raw_end = i; // incomplete \uXXXX, wait for more
                        return false;
                    }
                    break;
                }
                default:
                    tcs.content_buffer += esc;
                    i += 2;
                    break;
            }
        } else {
            tcs.content_buffer += c;
            i++;
        }
        tcs.content_raw_end = i;
    }
    return false;
}

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

common_chat_msg agent_loop::generate_completion(result_timings & out_timings) {
    server_response_reader rd = server_ctx_.get_response_reader();
    {
        // Keep formatting + posting atomic.
        std::lock_guard<std::mutex> lock(g_completion_mutex);

        server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
        task.id        = rd.get_new_id();
        task.index     = 0;

        // Route through the same OAI-compat code path as the HTTP server.
        auto meta = server_ctx_.get_meta();
        auto chat_tools = tool_registry::instance().to_chat_tools();
        json body = build_oai_request_body(chat_tools, meta.has_inp_image);
        std::vector<raw_buffer> files;
        json data = oaicompat_chat_params_parse(body, meta.chat_params, files);

        task.params = server_task::params_from_json_cmpl(vocab_, *params_, meta.slot_n_ctx, data);

        task.cli        = true;
        task.cli_prompt = data.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);

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

    // Wait for first result
    console::spinner::start();
    server_task_result_ptr result;
    try {
        result = rd.next(should_stop);
    } catch (const std::exception & e) {
        console::spinner::stop();
        LOG_WRN("Failed to parse model output: %s\n", e.what());
        common_chat_msg msg;
        msg.role = "assistant";
        return msg;
    }
    console::spinner::stop();
    std::string full_content;
    bool is_thinking = false;
    bool was_aborted = false;
    std::vector<tool_stream_state> tc_states;

    while (result) {
        if (should_stop()) {
            was_aborted = true;
            break;
        }
        if (result->is_error()) {
            auto * err = dynamic_cast<server_task_result_error *>(result.get());
            if (err && err->err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
                last_completion_overflowed_ = true;
            }
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
                    }
                    console::log("%s", diff.content_delta.c_str());
                    console::flush();
                    if (is_thinking) {
                        is_thinking = false;
                    }
                    full_content += diff.content_delta;
                }
                if (!diff.reasoning_content_delta.empty()) {
                    console::set_display(DISPLAY_TYPE_REASONING);
                    if (!is_thinking) {
                        console::log("───\n");
                    }
                    console::log("%s", diff.reasoning_content_delta.c_str());
                    console::flush();
                    is_thinking = true;
                }
                // Stream tool call arguments as they are generated
                if (diff.tool_call_index != std::string::npos) {
                    size_t idx = diff.tool_call_index;
                    if (idx >= tc_states.size()) {
                        tc_states.resize(idx + 1);
                    }
                    auto & tcs = tc_states[idx];

                    if (!diff.tool_call_delta.name.empty()) {
                        tcs.name = diff.tool_call_delta.name;
                    }
                    tcs.accumulated_args += diff.tool_call_delta.arguments;

                    // Close reasoning block if still open
                    if (is_thinking) {
                        console::log("\n───\n");
                        console::set_display(DISPLAY_TYPE_RESET);
                        is_thinking = false;
                    }

                    // Print tool header on first appearance
                    if (!tcs.header_printed && !tcs.name.empty()) {
                        console::set_display(DISPLAY_TYPE_INFO);
                        console::log("\n› %s", tcs.name.c_str());
                        console::set_display(DISPLAY_TYPE_RESET);
                        tcs.header_printed = true;
                    }

                    // Extract file_path once complete
                    if (tcs.header_printed && tcs.displayed_path.empty()) {
                        bool path_complete = false;
                        std::string fp = extract_partial_json_string(tcs.accumulated_args, "file_path", path_complete);
                        if (path_complete && !fp.empty()) {
                            console::set_display(DISPLAY_TYPE_INFO);
                            console::log(" %s", fp.c_str());
                            console::set_display(DISPLAY_TYPE_RESET);
                            tcs.displayed_path = fp;
                        }
                    }

                    // For write/edit: stream content lines incrementally
                    const char * content_field = nullptr;
                    if (tcs.name == "write") {
                        content_field = "content";
                    } else if (tcs.name == "edit") {
                        content_field = "new_string";
                    }
                    if (content_field && tcs.header_printed && !tcs.content_complete) {
                        bool field_complete = decode_field_incremental(tcs, content_field);
                        if (field_complete) tcs.content_complete = true;

                        // Print new complete lines starting from where we left off
                        const std::string & buf = tcs.content_buffer;
                        size_t pos = tcs.displayed_bytes;
                        while (pos < buf.size()) {
                            size_t nl = buf.find('\n', pos);
                            if (nl != std::string::npos) {
                                console::set_display(DISPLAY_TYPE_TOOL_STREAM);
                                console::log("\n  %.*s", (int)(nl - pos), buf.c_str() + pos);
                                console::set_display(DISPLAY_TYPE_RESET);
                                tcs.displayed_lines++;
                                pos = nl + 1;
                                tcs.displayed_bytes = pos;
                            } else if (field_complete) {
                                if (pos < buf.size()) {
                                    console::set_display(DISPLAY_TYPE_TOOL_STREAM);
                                    console::log("\n  %s", buf.c_str() + pos);
                                    console::set_display(DISPLAY_TYPE_RESET);
                                    tcs.displayed_lines++;
                                    tcs.displayed_bytes = buf.size();
                                }
                                break;
                            } else {
                                break; // incomplete last line, wait for more
                            }
                        }
                    }
                    console::flush();
                }
            }
        }

        auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        if (res_final) {
            out_timings = std::move(res_final->timings);
            last_prompt_tokens_ = res_final->n_prompt_tokens;

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

        try {
            result = rd.next(should_stop);
        } catch (const std::exception & e) {
            LOG_WRN("Failed to parse model output: %s\n", e.what());
            break;
        }
    }

    // Ensure spinner is stopped before returning (may have been started during tool call arg generation)
    console::spinner::stop();

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
    else if (call.name == "update_plan") ptype = permission_type::GLOB;

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
    if (call.name == "bash") {
        std::string cmd = args.value("command", "");
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

    // Execute the tool with timing
    auto start_time = std::chrono::steady_clock::now();
    tool_result result = registry.execute(call.name, args, tool_ctx_);
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    console::spinner::stop();
    console::log("\n");

    // Display result summary
    if (result.success) {
        // Truncate long output for display
        std::string display_output = result.output;
        if (display_output.length() > 500) {
            display_output = display_output.substr(0, 500) + "\n... (truncated)";
        }
        console::log("%s\n", display_output.c_str());
        if (!result.image_bytes.empty()) {
            render_image_to_terminal(result.image_bytes.data(), result.image_bytes.size(),
                                     result.image_mime);
        }
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

agent_loop_result agent_loop::run(const std::string & user_prompt) {
    agent_loop_result result;
    result.iterations = 0;

    // Add user message
    messages_.push_back({
        {"role", "user"},
        {"content", user_prompt}
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

        // Route through the same OAI-compat code path as the HTTP server.
        auto meta = server_ctx_.get_meta();
        auto chat_tools = tool_registry::instance().to_chat_tools();
        json body = build_oai_request_body(chat_tools, meta.has_inp_image);
        std::vector<raw_buffer> files;
        json data = oaicompat_chat_params_parse(body, meta.chat_params, files);

        task.params = server_task::params_from_json_cmpl(vocab_, *params_, meta.slot_n_ctx, data);

        task.cli        = true;
        task.cli_prompt = data.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);

        rd.post_task(std::move(task));
    }

    server_task_result_ptr result;
    try {
        result = rd.next(should_stop);
    } catch (const std::exception & e) {
        LOG_WRN("Failed to parse model output: %s\n", e.what());
        common_chat_msg msg;
        msg.role = "assistant";
        return msg;
    }

    std::string full_content;
    bool was_aborted = false;

    while (result) {
        if (should_stop()) {
            was_aborted = true;
            break;
        }
        if (result->is_error()) {
            auto * err = dynamic_cast<server_task_result_error *>(result.get());
            if (err && err->err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
                last_completion_overflowed_ = true;
                // Don't emit error event — overflow may be recovered via compaction
                common_chat_msg empty_msg;
                return empty_msg;
            }
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
            last_prompt_tokens_ = res_final->n_prompt_tokens;
            if (!res_final->oaicompat_msg.empty()) {
                return res_final->oaicompat_msg;
            }
            if (!res_final->content.empty()) {
                full_content = res_final->content;
            }
            break;
        }

        try {
            result = rd.next(should_stop);
        } catch (const std::exception & e) {
            LOG_WRN("Failed to parse model output: %s\n", e.what());
            break;
        }
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
    } else if (call.name == "update_plan") {
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

                // Wait for response (cancellable via should_stop)
                auto response = async_perms.wait_for_response_or_stop(req_id, 300000, should_stop);
                if (should_stop()) {
                    on_event(agent_event::permission_resolved(req_id, false));
                    return {false, "", "Operation cancelled"};
                }
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

        auto response = async_perms.wait_for_response_or_stop(req_id, 300000, should_stop);
        if (should_stop()) {
            on_event(agent_event::permission_resolved(req_id, false));
            return {false, "", "Operation cancelled"};
        }
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

        // Wait for response (cancellable via should_stop)
        auto response = async_perms.wait_for_response_or_stop(req_id, 300000, should_stop);

        if (should_stop()) {
            on_event(agent_event::permission_resolved(req_id, false));
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
    return registry.execute(call.name, args, tool_ctx_);
}

std::string agent_loop::generate_summary(const json & messages_to_summarize,
                                          const std::string & previous_summary) {
    std::string conv_text = compaction_serialize_conversation(messages_to_summarize, 0, messages_to_summarize.size());

    // Build the user prompt
    std::string prompt_text = "<conversation>\n" + conv_text + "</conversation>\n\n";
    if (!previous_summary.empty()) {
        prompt_text += "<previous-summary>\n" + previous_summary + "\n</previous-summary>\n\n";
        prompt_text += UPDATE_SUMMARIZATION_PROMPT;
    } else {
        prompt_text += SUMMARIZATION_PROMPT;
    }

    // Build temporary messages for the summarization call
    json summary_messages = json::array();
    summary_messages.push_back({{"role", "system"}, {"content", SUMMARIZATION_SYSTEM_PROMPT}});
    summary_messages.push_back({{"role", "user"}, {"content", prompt_text}});

    // Render through chat template
    auto meta = server_ctx_.get_meta();
    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(summary_messages);
    inputs.tools                 = {};
    inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
    inputs.use_jinja             = meta.chat_params.use_jinja;
    inputs.parallel_tool_calls   = false;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking       = false;
    auto chat_params = common_chat_templates_apply(meta.chat_params.tmpls.get(), inputs);

    // Post summarization task
    server_response_reader rd = server_ctx_.get_response_reader();
    {
        std::lock_guard<std::mutex> lock(g_completion_mutex);

        server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
        task.id     = rd.get_new_id();
        task.index  = 0;
        task.params = task_defaults_;
        int32_t ctx_size = server_ctx_.get_meta().slot_n_ctx;
        int32_t effective_reserve = std::min(config_.compaction.reserve_tokens, ctx_size / 4);
        task.params.n_predict = (int32_t)(0.8f * effective_reserve);
        task.params.chat_parser_params.parse_tool_calls = false;

        task.cli        = true;
        task.cli_prompt = std::move(chat_params.prompt);

        rd.post_task(std::move(task));
    }

    // Collect full response
    std::string summary_text;
    auto should_stop_fn = [this]() { return is_interrupted_.load(); };

    try {
        server_task_result_ptr result = rd.next(should_stop_fn);
        while (result) {
            if (result->is_error()) {
                LOG_WRN("Compaction summary generation failed\n");
                break;
            }

            auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (partial) {
                for (const auto & diff : partial->oaicompat_msg_diffs) {
                    summary_text += diff.content_delta;
                }
            }

            auto * final_result = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (final_result) {
                if (!final_result->oaicompat_msg.content.empty()) {
                    summary_text = final_result->oaicompat_msg.content;
                }
                break;
            }

            result = rd.next(should_stop_fn);
        }
    } catch (const std::exception & e) {
        LOG_WRN("Compaction summary generation error: %s\n", e.what());
    }

    return summary_text;
}

bool agent_loop::try_compact() {
    int32_t ctx_size = server_ctx_.get_meta().slot_n_ctx;
    if (ctx_size <= 0) {
        return false;
    }

    // Clamp settings proportionally to context size for small contexts
    int32_t effective_reserve = std::min(config_.compaction.reserve_tokens, ctx_size / 4);
    int32_t effective_keep    = std::min(config_.compaction.keep_recent_tokens, ctx_size / 3);

    // Check if compaction is needed
    bool threshold_hit = last_prompt_tokens_ > ctx_size - effective_reserve;
    if (!threshold_hit && !last_completion_overflowed_) {
        return false;
    }

    return do_compact(effective_keep);
}

bool agent_loop::compact() {
    int32_t ctx_size = server_ctx_.get_meta().slot_n_ctx;
    int32_t effective_keep = (ctx_size > 0)
        ? std::min(config_.compaction.keep_recent_tokens, ctx_size / 3)
        : config_.compaction.keep_recent_tokens;
    return do_compact(effective_keep);
}

bool agent_loop::do_compact(int32_t effective_keep) {
    size_t cut_idx = compaction_find_cut_point(messages_, effective_keep);
    if (cut_idx <= 1) {
        return false; // nothing to summarize (only system prompt before cut)
    }

    // Extract messages to summarize: [1 .. cut_idx)
    json to_summarize = json::array();
    for (size_t i = 1; i < cut_idx; i++) {
        to_summarize.push_back(messages_[i]);
    }

    if (to_summarize.empty()) {
        return false;
    }

    LOG_INF("Compacting context: %d prompt tokens, summarizing %zu messages, keeping %zu\n",
            last_prompt_tokens_, to_summarize.size(), messages_.size() - cut_idx);

    std::string summary = generate_summary(to_summarize, previous_summary_);
    if (summary.empty()) {
        LOG_WRN("Compaction failed: empty summary\n");
        return false;
    }

    previous_summary_ = summary;

    // Write compaction entry to session file before rebuilding messages
    if (session_file_) {
        size_t kept_count = messages_.size() - cut_idx;
        size_t kept_from = session_file_->message_count() - kept_count;
        session_file_->append_compaction(previous_summary_, kept_from);
    }

    // Rebuild messages: system + summary + recent messages
    json new_messages = json::array();
    new_messages.push_back(messages_[0]); // system prompt

    new_messages.push_back({
        {"role", "user"},
        {"content", "<context-summary>\n" + summary + "\n</context-summary>"}
    });

    for (size_t i = cut_idx; i < messages_.size(); i++) {
        new_messages.push_back(messages_[i]);
    }

    messages_ = std::move(new_messages);

    LOG_INF("Compaction complete: %zu messages in context\n", messages_.size());
    return true;
}
