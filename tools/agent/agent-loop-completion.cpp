#include "agent-loop.h"
#include "agent-loop-internal.h"
#include "console.h"
#include "terminal-image.h"

#include <chrono>
#include <functional>
#include <mutex>

#if defined(_WIN32)
#include <conio.h>
#else
#include <sys/select.h>
#include <unistd.h>
#include <termios.h>
#endif

// Definition of the shared mutex declared in agent-loop-internal.h
std::mutex g_completion_mutex;

// Check for ESC key press without blocking
bool check_escape_key() {
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

// Extract a string-valued field from a partial (possibly incomplete) JSON object.
// Returns the decoded string value, or empty string if the key is not yet present.
// `complete` is set to true if the closing quote of the value was found.
std::string extract_partial_json_string(
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
bool decode_field_incremental(
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

        task.params = server_task::params_from_json_cmpl(vocab_, *params_, meta.slot_n_ctx, meta.logit_bias_eog, data);

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

        task.params = server_task::params_from_json_cmpl(vocab_, *params_, meta.slot_n_ctx, meta.logit_bias_eog, data);

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
