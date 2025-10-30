#include "chat-parser.h"
#include "common.h"
#include "log.h"
#include "regex-partial.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::ordered_json;

common_chat_msg_parser::common_chat_msg_parser(const std::string & input, bool is_partial, const common_chat_syntax & syntax)
    : input_(input), is_partial_(is_partial), syntax_(syntax)
{
    result_.role = "assistant";

    while (true) {
        std::string id = std::to_string(std::rand());
        if (input.find(id) == std::string::npos) {
            healing_marker_ = id;
            break;
        }
    }
}

std::string common_chat_msg_parser::str(const common_string_range & rng) const {
    GGML_ASSERT(rng.begin <= rng.end);
    return input_.substr(rng.begin, rng.end - rng.begin);
}

void common_chat_msg_parser::add_content(const std::string &content) {
    result_.content += content;
}

void common_chat_msg_parser::add_reasoning_content(const std::string &reasoning_content) {
    result_.reasoning_content += reasoning_content;
}

bool common_chat_msg_parser::add_tool_call(const std::string & name, const std::string & id, const std::string & arguments) {
    if (name.empty()) {
        return false;
    }

    common_chat_tool_call tool_call;
    tool_call.name = name;
    tool_call.arguments = arguments;
    tool_call.id = id;

    // LOG_DBG("Tool call arguments:\n\traw: %s\n\tresult: %s\n", arguments.c_str(), tool_call.arguments.c_str());
    result_.tool_calls.emplace_back(tool_call);

    return true;
}
bool common_chat_msg_parser::add_tool_call(const json & tool_call) {
    std::string name = tool_call.contains("name") ? tool_call.at("name") : "";
    std::string id = tool_call.contains("id") ? tool_call.at("id") : "";
    std::string arguments = "";
    if (tool_call.contains("arguments")) {
        if (tool_call.at("arguments").is_object()) {
            arguments = tool_call.at("arguments").dump();
        } else {
            arguments = tool_call.at("arguments");
        }
    }

    return add_tool_call(name, id, arguments);
}

bool common_chat_msg_parser::add_tool_calls(const json & arr) {
    for (const auto & item : arr) {
        if (!add_tool_call(item)) {
            return false;
        }
    }
    return true;
}

bool common_chat_msg_parser::add_tool_call_short_form(const json & tool_call) {
    if (!tool_call.is_object() || tool_call.size() != 1) {
        return false;
    }

    // Get the tool name (the single key in the object)
    auto it = tool_call.begin();
    std::string name = it.key();

    if (name.empty()) {
        return false;
    }

    // Get the arguments (the nested object)
    const json & args_json = it.value();
    std::string arguments = "";

    if (args_json.is_object()) {
        arguments = args_json.dump();
    } else if (args_json.is_string()) {
        arguments = args_json;
    } else if (!args_json.is_null()) {
        // For other types, convert to string representation
        arguments = args_json.dump();
    }

    return add_tool_call(name, "", arguments);
}
void common_chat_msg_parser::finish() {
    if (!is_partial_ && pos_ != input_.size()) {
        throw std::runtime_error("Unexpected content at end of input");// + input_.substr(pos_));
    }
}

bool common_chat_msg_parser::consume_spaces() {
    const auto length = input_.size();
    auto consumed = false;
    while (pos_ < length && std::isspace(input_[pos_])) {
        ++pos_;
        consumed = true;
    }
    return consumed;
}

bool common_chat_msg_parser::try_consume_literal(const std::string & literal) {
    auto pos = pos_;
    for (auto i = 0u; i < literal.size(); ++i) {
        if (pos >= input_.size()) {
            return false;
        }
        if (input_[pos] != literal[i]) {
            return false;
        }
        ++pos;
    }
    pos_ = pos;
    return true;
}

std::optional<common_chat_msg_parser::find_regex_result>  common_chat_msg_parser::try_find_literal(const std::string & literal) {
    auto idx = input_.find(literal, pos_);
    if (idx != std::string::npos) {
        find_regex_result res;
        res.prelude = input_.substr(pos_, idx - pos_);
        auto end = idx + literal.size();
        res.groups.emplace_back(common_string_range{idx, end});
        move_to(end);
        return res;
    }
    if (is_partial_) {
        idx = string_find_partial_stop(input_, literal);
        if (idx != std::string::npos && idx >= pos_) {
            find_regex_result res;
            res.prelude = input_.substr(pos_, idx - pos_);
            auto end = input_.size();
            res.groups.emplace_back(common_string_range{idx, end});
            move_to(end);
            return res;
        }
    }
    return std::nullopt;
}

void common_chat_msg_parser::consume_literal(const std::string & literal) {
    if (!try_consume_literal(literal)) {
        throw common_chat_msg_partial_exception(literal);
    }
}

bool common_chat_msg_parser::try_parse_reasoning(const std::string & start_think, const std::string & end_think) {
    std::string pending_reasoning_prefix;

    if (syntax_.reasoning_format == COMMON_REASONING_FORMAT_NONE) {
        return false;
    }

    auto set_reasoning_prefix = [&](size_t prefix_pos) {
        if (!syntax_.thinking_forced_open || syntax_.reasoning_in_content) {
            return;
        }
        if (prefix_pos + start_think.size() > input_.size()) {
            pending_reasoning_prefix.clear();
            return;
        }
        // Capture the exact literal that opened the reasoning section so we can
        // surface it back to callers. This ensures formats that force the
        // reasoning tag open (e.g. DeepSeek R1) retain their original prefix
        // instead of dropping it during parsing.
        pending_reasoning_prefix = input_.substr(prefix_pos, start_think.size());
    };

    auto handle_reasoning = [&](const std::string & reasoning, bool closed) {
        auto stripped_reasoning = string_strip(reasoning);
        if (stripped_reasoning.empty()) {
            return;
        }
        if (syntax_.reasoning_in_content) {
            add_content(syntax_.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK ? "<think>" : start_think);
            add_content(stripped_reasoning);
            if (closed) {
                add_content(syntax_.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK ? "</think>" : end_think);
            }
        } else {
            if (!pending_reasoning_prefix.empty()) {
                add_reasoning_content(pending_reasoning_prefix);
                pending_reasoning_prefix.clear();
            }
            add_reasoning_content(stripped_reasoning);
        }
    };

    const size_t saved_pos = pos_;
    const size_t saved_content_size = result_.content.size();
    const size_t saved_reasoning_size = result_.reasoning_content.size();

    auto restore_state = [&]() {
        move_to(saved_pos);
        result_.content.resize(saved_content_size);
        result_.reasoning_content.resize(saved_reasoning_size);
    };

    // Allow leading whitespace to be preserved as content when reasoning is present at the start
    size_t cursor = pos_;
    size_t whitespace_end = cursor;
    while (whitespace_end < input_.size() && std::isspace(static_cast<unsigned char>(input_[whitespace_end]))) {
        ++whitespace_end;
    }

    if (whitespace_end >= input_.size()) {
        restore_state();
        if (syntax_.thinking_forced_open) {
            auto rest = input_.substr(saved_pos);
            if (!rest.empty()) {
                handle_reasoning(rest, /* closed */ !is_partial());
            }
            move_to(input_.size());
            return true;
        }
        return false;
    }

    cursor = whitespace_end;
    const size_t remaining = input_.size() - cursor;
    const size_t start_prefix = std::min(start_think.size(), remaining);
    const bool has_start_tag = input_.compare(cursor, start_prefix, start_think, 0, start_prefix) == 0;

    if (has_start_tag && start_prefix < start_think.size()) {
        move_to(input_.size());
        return true;
    }

    if (has_start_tag) {
        if (whitespace_end > pos_) {
            add_content(input_.substr(pos_, whitespace_end - pos_));
        }
        set_reasoning_prefix(cursor);
        cursor += start_think.size();
    } else if (syntax_.thinking_forced_open) {
        cursor = whitespace_end;
    } else {
        restore_state();
        return false;
    }
    while (true) {
        if (cursor >= input_.size()) {
            move_to(input_.size());
            return true;
        }

        size_t end_pos = input_.find(end_think, cursor);
        if (end_pos == std::string::npos) {
            std::string_view remaining_view(input_.data() + cursor, input_.size() - cursor);
            size_t partial_off = string_find_partial_stop(remaining_view, end_think);
            size_t reasoning_end = partial_off == std::string::npos ? input_.size() : cursor + partial_off;
            if (reasoning_end > cursor) {
                handle_reasoning(input_.substr(cursor, reasoning_end - cursor), /* closed */ partial_off == std::string::npos && !is_partial());
            }
            move_to(input_.size());
            return true;
        }

        if (end_pos > cursor) {
            handle_reasoning(input_.substr(cursor, end_pos - cursor), /* closed */ true);
        } else {
            handle_reasoning("", /* closed */ true);
        }

        cursor = end_pos + end_think.size();

        while (cursor < input_.size() && std::isspace(static_cast<unsigned char>(input_[cursor]))) {
            ++cursor;
        }

        const size_t next_remaining = input_.size() - cursor;
        if (next_remaining == 0) {
            move_to(cursor);
            return true;
        }

        const size_t next_prefix = std::min(start_think.size(), next_remaining);
        if (input_.compare(cursor, next_prefix, start_think, 0, next_prefix) == 0) {
            if (next_prefix < start_think.size()) {
                move_to(input_.size());
                return true;
            }
            set_reasoning_prefix(cursor);
            cursor += start_think.size();
            continue;
        }

        move_to(cursor);
        return true;
    }
}

std::string common_chat_msg_parser::consume_rest() {
    auto rest = input_.substr(pos_);
    pos_ = input_.size();
    return rest;
}

// Tries to find the regex, consumes it (pos right after it) and gives the prelude (right before it) and the groups to the callback.
std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_find_regex(const common_regex & regex, size_t from, bool add_prelude_to_content) {
    auto m = regex.search(input_, from == std::string::npos ? pos_ : from);
    if (m.type == COMMON_REGEX_MATCH_TYPE_NONE) {
        return std::nullopt;
    }
    auto prelude = input_.substr(pos_, m.groups[0].begin - pos_);
    pos_ = m.groups[0].end;

    if (add_prelude_to_content) {
        add_content(prelude);
    }
    if (m.type == COMMON_REGEX_MATCH_TYPE_PARTIAL) {
        if (is_partial()) {
            throw common_chat_msg_partial_exception(regex.str());
        }
        return std::nullopt;
    }
    return find_regex_result{prelude, m.groups};
}

common_chat_msg_parser::find_regex_result common_chat_msg_parser::consume_regex(const common_regex & regex) {
    if (auto result = try_consume_regex(regex)) {
        return *result;
    }
    throw common_chat_msg_partial_exception(regex.str());
}

std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_consume_regex(const common_regex & regex) {
    auto m = regex.search(input_, pos_);
    if (m.type == COMMON_REGEX_MATCH_TYPE_NONE) {
        return std::nullopt;
    }
    if (m.type == COMMON_REGEX_MATCH_TYPE_PARTIAL) {
        if (is_partial()) {
            throw common_chat_msg_partial_exception(regex.str());
        }
        return std::nullopt;
    }
    if (m.groups[0].begin != pos_) {
        // Didn't match at the current position.
        return std::nullopt;
    }
    pos_ = m.groups[0].end;

    return find_regex_result {
        /* .prelude = */ "",
        m.groups,
    };
}

std::optional<common_json> common_chat_msg_parser::try_consume_json() {
    auto it = input_.cbegin() + pos_;
    const auto end = input_.cend();
    common_json result;
    if (!common_json_parse(it, end, healing_marker_, result)) {
        return std::nullopt;
    }
    pos_ = std::distance(input_.cbegin(), it);
    if (result.healing_marker.marker.empty()) {
        // No healing marker, just return the parsed json
        return result;
    }
    if (!is_partial()) {
        throw common_chat_msg_partial_exception("JSON");
    }
    return result;
}

common_json common_chat_msg_parser::consume_json() {
    if (auto result = try_consume_json()) {
        return *result;
    }
    throw common_chat_msg_partial_exception("JSON");
}

common_chat_msg_parser::consume_json_result common_chat_msg_parser::consume_json_with_dumped_args(
    const std::vector<std::vector<std::string>> & args_paths,
    const std::vector<std::vector<std::string>> & content_paths
) {
    if (auto result = try_consume_json_with_dumped_args(args_paths, content_paths)) {
        return *result;
    }
    throw common_chat_msg_partial_exception("JSON");
}

std::optional<common_chat_msg_parser::consume_json_result> common_chat_msg_parser::try_consume_json_with_dumped_args(
    const std::vector<std::vector<std::string>> & args_paths,
    const std::vector<std::vector<std::string>> & content_paths
) {
    auto partial = try_consume_json();
    if (!partial) {
        return std::nullopt;
    }
    auto is_arguments_path = [&](const std::vector<std::string> & path) {
        return std::find(args_paths.begin(), args_paths.end(), path) != args_paths.end();
    };
    auto is_content_path = [&](const std::vector<std::string> & path) {
        return std::find(content_paths.begin(), content_paths.end(), path) != content_paths.end();
    };

    if (partial->healing_marker.marker.empty()) {
        if (args_paths.empty()) {
            // No arguments to dump, and JSON was parsed fully.
            return consume_json_result {
                partial->json,
                /* .is_partial = */ false,
            };
        }
        if (is_arguments_path({})) {
            // Entire JSON is the arguments and was parsed fully.
            return consume_json_result {
                partial->json.dump(/* indent */ -1, /* indent_char */ ' ', /* ensure_ascii */ true),
                /* .is_partial = */ false,
            };
        }
    }

    LOG_DBG("Parsed partial JSON: %s (json_healing_marker: %s)\n", partial->json.dump().c_str(), partial->healing_marker.json_dump_marker.c_str());

    auto found_healing_marker = false;
    std::vector<std::string> path;
    std::function<json(const json &)> remove_unsupported_healings_and_dump_args = [&](const json & j) -> json {
        if (is_arguments_path(path)) {
            auto arguments = j.dump(/* indent */ -1, /* indent_char */ ' ', /* ensure_ascii */ true);
            if (is_partial() && !partial->healing_marker.marker.empty()) {
                auto idx = arguments.find(partial->healing_marker.json_dump_marker);
                if (idx != std::string::npos) {
                    arguments.resize(idx);
                    found_healing_marker = true;
                }
                if (arguments == "\"") {
                    // This happens because of completing `:"$magic` after `"arguments"`
                    arguments = "";
                }
            }
            return arguments;
        }
        if (is_content_path(path)) {
            if (!j.is_string()) {
                throw std::runtime_error("Content path must be a string");
            }
            std::string str = j;
            auto idx = str.find(partial->healing_marker.marker); // not using json_dump_marker as we're inside a string
            if (idx != std::string::npos) {
                str.resize(idx);
                found_healing_marker = true;
            }
            return str;
        }
        if (j.is_object()) {
            auto obj = json::object();
            for (const auto & p : j.items()) {
                const auto & key = p.key();
                const auto & value = p.value();
                const std::string key_str = key; // NOLINT
                auto idx = key_str.find(healing_marker_);
                if (idx != std::string::npos) {
                    found_healing_marker = true;
                    break;
                }
                path.push_back(key_str);
                if (value.is_string()) {
                    const std::string value_str = value;
                    if (value_str.find(healing_marker_) != std::string::npos) {
                        found_healing_marker = true;
                        if (is_content_path(path)) {
                            if (partial->healing_marker.marker == partial->healing_marker.json_dump_marker) {
                                // The healing occurred inside the string: good. Otherwise we just ditch the entire key/value pair.
                                obj[key] = remove_unsupported_healings_and_dump_args(value);
                            }
                        }
                        break;
                    }
                    obj[key] = value;
                } else {
                    obj[key] = remove_unsupported_healings_and_dump_args(value);
                }
                path.pop_back();
            }
            return obj;
        }
        if (j.is_array()) {
            auto arr = json::array();
            for (const auto & value : j) {
                if (value.is_string()) {
                    std::string str = value;
                    auto idx = str.find(healing_marker_);
                    if (idx != std::string::npos) {
                        // Don't heal array values that aren't in the arguments.
                        found_healing_marker = true;
                        break;
                    }
                }
                arr.push_back(remove_unsupported_healings_and_dump_args(value));
            }
            return arr;
        }
        return j;
    };

    auto cleaned = remove_unsupported_healings_and_dump_args(partial->json);
    LOG_DBG("Cleaned up JSON %s to %s (json_healing_marker : '%s')\n", partial->json.dump().c_str(), cleaned.dump().c_str(), partial->healing_marker.json_dump_marker.c_str());
    return consume_json_result {
        cleaned,
        /* .is_partial = */ found_healing_marker,
    };
}

void common_chat_msg_parser::clear_tools() {
    result_.tool_calls.clear();
}
