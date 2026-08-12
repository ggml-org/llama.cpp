#include "dataset.h"

#include "common.h"
#include "jsonl.h"

#include "nlohmann/json.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace {

enum token_field : uint8_t {
    TOKEN_FIELD_NONE      = 0,
    TOKEN_FIELD_ASSISTANT = 1 << 0,
    TOKEN_FIELD_REASONING = 1 << 1,
    TOKEN_FIELD_CONTENT   = 1 << 2,
};

struct field_marker {
    std::string begin;
    std::string end;
    uint8_t field;
};

struct dataset_line_state {
    json data;
    std::vector<const json *> message_data;
    std::vector<common_chat_msg> messages;
    llama_prune_dataset_record record;
    std::string error;
    std::mutex error_mutex;
    std::atomic<bool> failed { false };
    bool valid = false;
};

struct message_task {
    size_t line_index;
    size_t first_message;
    size_t message_count;
};

std::string line_error(int64_t line, const std::string & message) {
    return "JSONL line " + std::to_string(line) + ": " + message;
}

void set_error(dataset_line_state & state, const std::string & error) {
    std::lock_guard<std::mutex> lock(state.error_mutex);
    if (state.error.empty()) state.error = error;
    state.failed.store(true);
}

}

llama_prune_ppl_mask llama_prune_ppl_mask_parse(const std::string & value) {
    if (value == "all") return llama_prune_ppl_mask::ALL;
    if (value == "assistant") return llama_prune_ppl_mask::ASSISTANT;
    if (value == "reasoning") return llama_prune_ppl_mask::REASONING;
    if (value == "content") return llama_prune_ppl_mask::CONTENT;
    throw std::runtime_error("invalid perplexity mask: " + value);
}

const char * llama_prune_ppl_mask_name(llama_prune_ppl_mask value) {
    switch (value) {
        case llama_prune_ppl_mask::ALL:       return "all";
        case llama_prune_ppl_mask::ASSISTANT: return "assistant";
        case llama_prune_ppl_mask::REASONING: return "reasoning";
        case llama_prune_ppl_mask::CONTENT:   return "content";
    }
    return "unknown";
}

bool llama_prune_token_is_evaluated(const llama_prune_dataset_record & record, size_t token_index, llama_prune_ppl_mask mask) {
    if (token_index == 0 || token_index >= record.tokens.size()) return false;
    if (mask == llama_prune_ppl_mask::ALL) return true;
    const uint8_t field = record.token_fields[token_index];
    if (mask == llama_prune_ppl_mask::ASSISTANT) return (field & TOKEN_FIELD_ASSISTANT) != 0;
    if (mask == llama_prune_ppl_mask::REASONING) return (field & TOKEN_FIELD_REASONING) != 0;
    return (field & TOKEN_FIELD_CONTENT) != 0;
}

llama_prune_dataset llama_prune_dataset_load(
        const std::string & path,
        const llama_model * model,
        const common_chat_templates * templates,
        int32_t n_threads) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<common_jsonl_line> lines = common_jsonl_read_lines(path, COMMON_JSONL_EMPTY_LINE_KEEP);
    std::vector<std::unique_ptr<dataset_line_state>> states;
    states.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) states.push_back(std::make_unique<dataset_line_state>());

    const size_t worker_limit = common_jsonl_worker_limit(n_threads);
    const size_t n_json_workers = common_jsonl_worker_count(n_threads, lines.size());
    {
        common_jsonl_worker_pool pool(n_json_workers);
        pool.parallel_for(lines.size(), [&](size_t i, size_t) {
            dataset_line_state & state = *states[i];
            if (lines[i].text.empty()) {
                set_error(state, "empty record");
                return;
            }
            try {
                state.data = json::parse(lines[i].text);
                if (!state.data.is_object() || !state.data.contains("messages") || !state.data["messages"].is_array() || state.data["messages"].empty()) {
                    set_error(state, "'messages' must be a non-empty array");
                    return;
                }
                const json & messages = state.data["messages"];
                state.message_data.reserve(messages.size());
                for (const json & message : messages) state.message_data.push_back(&message);
                state.messages.resize(state.message_data.size());
                state.valid = true;
            } catch (const std::exception & e) {
                set_error(state, std::string("invalid JSON: ") + e.what());
            }
        });
    }
    for (common_jsonl_line & line : lines) std::string().swap(line.text);

    const size_t message_parallel_min = 8;
    std::vector<message_task> message_tasks;
    size_t valid_lines = 0;
    for (size_t i = 0; i < states.size(); ++i) {
        const dataset_line_state & state = *states[i];
        if (!state.valid) continue;
        ++valid_lines;
        if (worker_limit > 1 && state.message_data.size() >= message_parallel_min) {
            for (size_t j = 0; j < state.message_data.size(); ++j) message_tasks.push_back({ i, j, 1 });
        } else {
            message_tasks.push_back({ i, 0, state.message_data.size() });
        }
    }

    const size_t n_workers = std::min(worker_limit, std::max(valid_lines, message_tasks.size()));
    std::vector<common_chat_templates_ptr> template_copies;
    std::vector<const common_chat_templates *> worker_templates(n_workers, templates);
    if (templates != nullptr && n_workers > 1) {
        template_copies.reserve(n_workers - 1);
        for (size_t i = 1; i < n_workers; ++i) {
            template_copies.push_back(common_chat_templates_init(model, ""));
            worker_templates[i] = template_copies.back().get();
        }
    }

    {
        common_jsonl_worker_pool pool(n_workers);
        pool.parallel_for(message_tasks.size(), [&](size_t task_index, size_t) {
            const message_task & task = message_tasks[task_index];
            dataset_line_state & state = *states[task.line_index];
            try {
                for (size_t i = task.first_message; i < task.first_message + task.message_count; ++i) {
                    state.messages[i] = common_jsonl_parse_chat_message(*state.message_data[i], COMMON_JSONL_CHAT_PARSE_STRICT_REASONING);
                }
            } catch (const std::exception & e) {
                set_error(state, e.what());
            }
        });

        for (std::unique_ptr<dataset_line_state> & state : states) {
            state->message_data.clear();
            state->data = json();
        }

        pool.parallel_for(states.size(), [&](size_t line_index, size_t worker_index) {
            dataset_line_state & state = *states[line_index];
            if (!state.valid || state.failed.load()) return;
            try {
                std::vector<field_marker> markers;
                size_t marker_id = 0;
                for (common_chat_msg & message : state.messages) {
                    auto mark = [&](std::string & text, uint8_t field) {
                        if (text.empty()) return;
                        const std::string id = std::to_string(marker_id++);
                        const std::string prefix(1, '\x1e');
                        const std::string suffix(1, '\x1f');
                        field_marker marker { prefix + "LLAMA_PRUNE_FIELD_" + id + "_BEGIN" + suffix, prefix + "LLAMA_PRUNE_FIELD_" + id + "_END" + suffix, field };
                        text = marker.begin + text + marker.end;
                        markers.push_back(std::move(marker));
                    };
                    if (message.role == "assistant") {
                        mark(message.reasoning_content, TOKEN_FIELD_ASSISTANT | TOKEN_FIELD_REASONING);
                        mark(message.content, TOKEN_FIELD_ASSISTANT | TOKEN_FIELD_CONTENT);
                        if (!message.reasoning_content.empty()) {
                            message.content = message.reasoning_content + (message.content.empty() ? "" : "\n" + message.content);
                            message.reasoning_content.clear();
                        }
                    }
                }

                common_chat_templates_inputs inputs;
                inputs.messages = std::move(state.messages);
                inputs.add_generation_prompt = false;
                inputs.use_jinja = true;
                inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
                std::string prompt;
                try {
                    prompt = common_chat_templates_apply(worker_templates[worker_index], inputs).prompt;
                } catch (const std::exception & e) {
                    throw std::runtime_error(std::string("chat template failed: ") + e.what());
                }

                struct span { size_t begin; size_t end; uint8_t field; };
                std::vector<span> spans;
                for (const field_marker & marker : markers) {
                    const size_t begin_marker = prompt.find(marker.begin);
                    if (begin_marker == std::string::npos) throw std::runtime_error("chat template did not preserve a message field");
                    prompt.erase(begin_marker, marker.begin.size());
                    const size_t end_marker = prompt.find(marker.end, begin_marker);
                    if (end_marker == std::string::npos) throw std::runtime_error("chat template produced an unterminated field");
                    prompt.erase(end_marker, marker.end.size());
                    spans.push_back({ begin_marker, end_marker, marker.field });
                }

                llama_prune_dataset_record & record = state.record;
                record.line = lines[line_index].number;
                record.tokens = common_tokenize(vocab, prompt, false, true);
                if (record.tokens.size() < 2) throw std::runtime_error("rendered conversation has fewer than two tokens");
                record.token_fields.assign(record.tokens.size(), TOKEN_FIELD_NONE);
                size_t offset = 0;
                for (size_t i = 0; i < record.tokens.size(); ++i) {
                    const std::string piece = common_token_to_piece(vocab, record.tokens[i], true);
                    size_t start = offset;
                    if (!piece.empty() && prompt.compare(offset, piece.size(), piece) != 0) {
                        const size_t found = prompt.find(piece, offset);
                        if (found != std::string::npos) start = found;
                    }
                    for (const span & current : spans) {
                        if (start >= current.begin && start < current.end) {
                            record.token_fields[i] = current.field;
                            break;
                        }
                    }
                    offset = start + piece.size();
                }
            } catch (const std::exception & e) {
                set_error(state, e.what());
            }
        });
    }

    llama_prune_dataset result;
    result.records.reserve(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        dataset_line_state & state = *states[i];
        if (state.failed.load()) throw std::runtime_error(line_error(lines[i].number, state.error));
        result.total_tokens += state.record.tokens.size();
        result.records.push_back(std::move(state.record));
    }
    if (result.records.empty()) throw std::runtime_error("JSONL dataset has no records");
    return result;
}
