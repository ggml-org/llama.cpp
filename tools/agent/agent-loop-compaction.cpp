#include "agent-loop.h"
#include "agent-loop-internal.h"
#include "console.h"

#include <mutex>

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
