#include "dataset.h"

#include "common.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
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

std::string line_error(int64_t line, const std::string & message) {
    return "JSONL line " + std::to_string(line) + ": " + message;
}

std::string required_string(const json & object, const char * key, int64_t line, bool required) {
    auto it = object.find(key);
    if (it == object.end()) {
        if (required) throw std::runtime_error(line_error(line, std::string("missing '") + key + "'"));
        return {};
    }
    if (!it->is_string()) throw std::runtime_error(line_error(line, std::string("'") + key + "' must be a string"));
    return it->get<std::string>();
}

}

aikar_ppl_mask aikar_ppl_mask_parse(const std::string & value) {
    if (value == "all") return aikar_ppl_mask::ALL;
    if (value == "assistant") return aikar_ppl_mask::ASSISTANT;
    if (value == "reasoning") return aikar_ppl_mask::REASONING;
    if (value == "content") return aikar_ppl_mask::CONTENT;
    throw std::runtime_error("invalid perplexity mask: " + value);
}

const char * aikar_ppl_mask_name(aikar_ppl_mask value) {
    switch (value) {
        case aikar_ppl_mask::ALL:       return "all";
        case aikar_ppl_mask::ASSISTANT: return "assistant";
        case aikar_ppl_mask::REASONING: return "reasoning";
        case aikar_ppl_mask::CONTENT:   return "content";
    }
    return "unknown";
}

bool aikar_token_is_evaluated(const aikar_dataset_record & record, size_t token_index, aikar_ppl_mask mask) {
    if (token_index == 0 || token_index >= record.tokens.size()) return false;
    if (mask == aikar_ppl_mask::ALL) return true;
    const uint8_t field = record.token_fields[token_index];
    if (mask == aikar_ppl_mask::ASSISTANT) return (field & TOKEN_FIELD_ASSISTANT) != 0;
    if (mask == aikar_ppl_mask::REASONING) return (field & TOKEN_FIELD_REASONING) != 0;
    return (field & TOKEN_FIELD_CONTENT) != 0;
}

aikar_dataset aikar_dataset_load(
        const std::string & path,
        const llama_model * model,
        const common_chat_templates * templates) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open JSONL dataset: " + path);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    aikar_dataset result;
    std::string line_text;
    int64_t line_number = 0;
    auto last_progress = std::chrono::steady_clock::now();
    while (std::getline(in, line_text)) {
        ++line_number;
        if (line_text.empty()) throw std::runtime_error(line_error(line_number, "empty record"));
        json root;
        try {
            root = json::parse(line_text);
        } catch (const std::exception & e) {
            throw std::runtime_error(line_error(line_number, std::string("invalid JSON: ") + e.what()));
        }
        if (!root.is_object() || !root.contains("messages") || !root["messages"].is_array() || root["messages"].empty()) {
            throw std::runtime_error(line_error(line_number, "'messages' must be a non-empty array"));
        }

        std::vector<common_chat_msg> messages;
        std::vector<field_marker> markers;
        size_t marker_id = 0;
        for (const json & item : root["messages"]) {
            if (!item.is_object()) throw std::runtime_error(line_error(line_number, "each message must be an object"));
            common_chat_msg message;
            message.role = required_string(item, "role", line_number, true);
            if (common_chat_role_from_string(message.role) == COMMON_CHAT_ROLE_UNKNOWN) {
                throw std::runtime_error(line_error(line_number, "unsupported role: " + message.role));
            }
            message.content = required_string(item, "content", line_number, message.role != "assistant");
            message.reasoning_content = required_string(item, "reasoning", line_number, false);
            if (item.contains("reasoning_content")) {
                if (!message.reasoning_content.empty()) throw std::runtime_error(line_error(line_number, "use only one of 'reasoning' and 'reasoning_content'"));
                message.reasoning_content = required_string(item, "reasoning_content", line_number, false);
            }
            if (message.role != "assistant" && !message.reasoning_content.empty()) {
                throw std::runtime_error(line_error(line_number, "reasoning is supported only for assistant messages"));
            }
            if (message.role == "assistant" && message.content.empty() && message.reasoning_content.empty()) {
                throw std::runtime_error(line_error(line_number, "assistant message must contain 'content' or 'reasoning'"));
            }

            auto mark = [&](std::string & text, uint8_t field) {
                if (text.empty()) return;
                const std::string id = std::to_string(marker_id++);
                const std::string prefix(1, '\x1e');
                const std::string suffix(1, '\x1f');
                field_marker marker { prefix + "AIKAR_FIELD_" + id + "_BEGIN" + suffix, prefix + "AIKAR_FIELD_" + id + "_END" + suffix, field };
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
            messages.push_back(std::move(message));
        }

        common_chat_templates_inputs inputs;
        inputs.messages = std::move(messages);
        inputs.add_generation_prompt = false;
        inputs.use_jinja = true;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        std::string marked_prompt;
        try {
            marked_prompt = common_chat_templates_apply(templates, inputs).prompt;
        } catch (const std::exception & e) {
            throw std::runtime_error(line_error(line_number, std::string("chat template failed: ") + e.what()));
        }

        struct span { size_t begin; size_t end; uint8_t field; };
        std::vector<span> spans;
        std::string prompt = marked_prompt;
        for (const field_marker & marker : markers) {
            const size_t begin_marker = prompt.find(marker.begin);
            if (begin_marker == std::string::npos) throw std::runtime_error(line_error(line_number, "chat template did not preserve a message field"));
            prompt.erase(begin_marker, marker.begin.size());
            const size_t end_marker = prompt.find(marker.end, begin_marker);
            if (end_marker == std::string::npos) throw std::runtime_error(line_error(line_number, "chat template produced an unterminated field"));
            prompt.erase(end_marker, marker.end.size());
            spans.push_back({ begin_marker, end_marker, marker.field });
        }

        aikar_dataset_record record;
        record.line = line_number;
        record.tokens = common_tokenize(vocab, prompt, false, true);
        if (record.tokens.size() < 2) throw std::runtime_error(line_error(line_number, "rendered conversation has fewer than two tokens"));
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
        result.total_tokens += record.tokens.size();
        result.records.push_back(std::move(record));
        const auto now = std::chrono::steady_clock::now();
        if (now - last_progress >= std::chrono::seconds(5)) {
            std::cerr << "aikar-prune: tokenized " << result.records.size() << " records, " << result.total_tokens << " tokens\n";
            last_progress = now;
        }
    }
    if (result.records.empty()) throw std::runtime_error("JSONL dataset has no records");
    return result;
}
