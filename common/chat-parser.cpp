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
#include <unordered_set>
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

// Qwen3-Coder XML tool call parser implementation
namespace {
    // Constants for DoS protection
    static constexpr size_t MAX_INPUT_SIZE = 1024 * 1024; // 1MB limit
    static constexpr size_t MAX_PARAMETER_COUNT = 100; // Maximum parameters per function
    static constexpr size_t MAX_TAG_NAME_LENGTH = 256; // Maximum tag name length
    static constexpr size_t MAX_ATTRIBUTE_LENGTH = 1024; // Maximum attribute length

    // Helper function to set error details
    void set_error(common_chat_msg_parser::XmlParseError & error,
                   common_chat_msg_parser::XmlParseErrorType type,
                   size_t position,
                   const std::string & context,
                   const std::string & message) {
        error.type = type;
        error.position = position;
        error.context = context;
        error.message = message;
    }

    // Simple XML tag parser - safer than regex, using string_view for performance
    struct XmlTag {
        std::string name;
        std::string attribute;
        std::string content;
        size_t start_pos = 0;
        size_t end_pos = 0;
    };

    // Find XML tag with optional attribute - ITERATIVE implementation to avoid stack overflow
    std::optional<XmlTag> find_xml_tag(std::string_view text, std::string_view tag_name, size_t start_pos = 0,
                                       common_chat_msg_parser::XmlParseError * error = nullptr) {
        // Input validation for DoS protection
        if (text.size() > MAX_INPUT_SIZE) {
            LOG_DBG("XML input too large: %zu bytes (max: %zu)\n", text.size(), MAX_INPUT_SIZE);
            if (error) {
                set_error(*error, common_chat_msg_parser::XmlParseErrorType::INPUT_TOO_LARGE, 0,
                         std::string(text.substr(0, std::min(text.size(), size_t(100)))),
                         "XML input exceeds maximum size limit of " + std::to_string(MAX_INPUT_SIZE) + " bytes");
            }
            return std::nullopt;
        }

        if (tag_name.size() > MAX_TAG_NAME_LENGTH) {
            LOG_DBG("Tag name too long: %zu chars (max: %zu)\n", tag_name.size(), MAX_TAG_NAME_LENGTH);
            if (error) {
                set_error(*error, common_chat_msg_parser::XmlParseErrorType::TAG_NAME_TOO_LONG, 0,
                         std::string(tag_name),
                         "Tag name exceeds maximum length of " + std::to_string(MAX_TAG_NAME_LENGTH) + " characters");
            }
            return std::nullopt;
        }

        if (start_pos >= text.size()) {
            return std::nullopt;
        }

        // PERFORMANCE OPTIMIZATION: Use string_view to avoid allocations
        // Pre-compute tag patterns
        const std::string open_tag_start = std::string("<") + std::string(tag_name);
        const std::string close_tag = std::string("</") + std::string(tag_name) + ">";

        // ITERATIVE search to avoid recursion and potential stack overflow
        size_t search_pos = start_pos;
        while (search_pos < text.size()) {
            // Look for opening tag
            size_t open_pos = text.find(open_tag_start, search_pos);
            if (open_pos == std::string::npos) {
                return std::nullopt;
            }

            // Validate that this is actually the start of our tag (not a substring)
            // Check that the character after tag name is either '>' or '=' or whitespace
            size_t check_pos = open_pos + open_tag_start.length();
            if (check_pos < text.size()) {
                char next_char = text[check_pos];
                if (next_char != '>' && next_char != '=' && !std::isspace(next_char)) {
                    // This is a false match (e.g., looking for "tool" but found "tool_call")
                    // Continue searching from the next position
                    search_pos = open_pos + 1;
                    continue;
                }
            }

            // Find the end of the opening tag
            size_t open_end = text.find('>', open_pos);
            if (open_end == std::string::npos) {
                return std::nullopt;
            }

            XmlTag tag;
            tag.start_pos = open_pos;

            // Extract attribute if present (for tags like <function=name> or <function = "name">)
            // PERFORMANCE: Use string_view for substring operations
            size_t tag_content_start = open_pos + 1 + tag_name.length();
            if (tag_content_start < open_end) {
                // Look for '=' in the tag content
                size_t eq_pos = text.find('=', tag_content_start);
                if (eq_pos != std::string::npos && eq_pos < open_end) {
                    // Skip whitespace after '='
                    size_t attr_start = eq_pos + 1;
                    while (attr_start < open_end && std::isspace(text[attr_start])) {
                        attr_start++;
                    }

                    if (attr_start < open_end) {
                        size_t attr_end = open_end;

                        // Handle quoted attribute values
                        if (text[attr_start] == '"' || text[attr_start] == '\'') {
                            char quote_char = text[attr_start];
                            attr_start++; // Skip opening quote

                            // Find closing quote
                            size_t quote_end = text.find(quote_char, attr_start);
                            if (quote_end != std::string::npos && quote_end < open_end) {
                                attr_end = quote_end;
                            } else {
                                // No closing quote found, treat as unquoted
                                attr_start--; // Go back to include the quote
                            }
                        } else {
                            // Unquoted attribute - trim trailing whitespace
                            while (attr_end > attr_start && std::isspace(text[attr_end - 1])) {
                                attr_end--;
                            }
                        }

                        if (attr_start < attr_end) {
                            std::string_view attr_view = text.substr(attr_start, attr_end - attr_start);
                            // Validate attribute length
                            if (attr_view.size() <= MAX_ATTRIBUTE_LENGTH) {
                                tag.attribute = std::string(attr_view);
                            } else {
                                LOG_DBG("Attribute too long: %zu chars (max: %zu)\n", attr_view.size(), MAX_ATTRIBUTE_LENGTH);
                                if (error) {
                                    set_error(*error, common_chat_msg_parser::XmlParseErrorType::ATTRIBUTE_TOO_LONG,
                                             open_pos, std::string(attr_view.substr(0, 100)),
                                             "Attribute exceeds maximum length of " + std::to_string(MAX_ATTRIBUTE_LENGTH) + " characters");
                                }
                                return std::nullopt;
                            }
                        }
                    }
                }
            }

            // Look for closing tag - PERFORMANCE: Search from after opening tag
            size_t close_pos = text.find(close_tag, open_end + 1);
            if (close_pos == std::string::npos) {
                return tag;
            }

            tag.end_pos = close_pos + close_tag.length();
            tag.name = std::string(tag_name);

            // PERFORMANCE: Use string_view for content extraction
            size_t content_start = open_end + 1;
            size_t content_length = close_pos - content_start;
            if (content_length > 0) {
                std::string_view content_view = text.substr(content_start, content_length);
                tag.content = std::string(content_view);
            }

            return tag;
        }

        return std::nullopt;
    }

    // Find all XML tags with a specific name and attribute pattern - with limits, using string_view
    std::vector<XmlTag> find_all_xml_tags(std::string_view text, std::string_view tag_name,
                                          common_chat_msg_parser::XmlParseError * error = nullptr) {
        std::vector<XmlTag> tags;
        size_t pos = 0;
        size_t tag_count = 0;

        while (pos < text.length() && tag_count < MAX_PARAMETER_COUNT) {
            auto tag = find_xml_tag(text, tag_name, pos, error);
            if (!tag) {
                break;
            }
            tags.push_back(*tag);
            pos = tag->end_pos;
            ++tag_count;
        }

        if (tag_count >= MAX_PARAMETER_COUNT) {
            LOG_DBG("Too many tags found: %zu (max: %zu)\n", tag_count, MAX_PARAMETER_COUNT);
            if (error) {
                set_error(*error, common_chat_msg_parser::XmlParseErrorType::TOO_MANY_PARAMETERS, pos,
                         std::string(text.substr(pos, std::min(text.size() - pos, size_t(100)))),
                         "Too many " + std::string(tag_name) + " tags found (max: " + std::to_string(MAX_PARAMETER_COUNT) + ")");
            }
        }

        return tags;
    }

    // Trim whitespace from string using string_view for performance
    std::string trim_whitespace(std::string_view str) {
        size_t start = str.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) {
            return "";
        }
        size_t end = str.find_last_not_of(" \t\n\r");
        return std::string(str.substr(start, end - start + 1));
    }

    // Safe integer parsing with overflow protection using string_view
    bool safe_parse_int(std::string_view str, int & result) {
        try {
            // Check for potential overflow by using long long first
            std::string str_copy(str); // stoll requires std::string
            long long temp = std::stoll(str_copy);
            if (temp > std::numeric_limits<int>::max() || temp < std::numeric_limits<int>::min()) {
                return false; // Overflow
            }
            result = static_cast<int>(temp);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    // Safe float parsing with overflow protection using string_view
    bool safe_parse_float(std::string_view str, float & result) {
        try {
            std::string str_copy(str); // stod requires std::string
            double temp = std::stod(str_copy);
            if (temp > std::numeric_limits<float>::max() || temp < std::numeric_limits<float>::lowest()) {
                return false; // Overflow
            }
            result = static_cast<float>(temp);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    // Convert parameter value based on tool schema type - FIXED JSON injection vulnerability, using string_view
    std::string convert_qwen3_param_value(std::string_view param_value,
                                         std::string_view param_name,
                                         const nlohmann::json & param_config,
                                         std::string_view /* func_name */) {
        std::string trimmed_value = trim_whitespace(param_value);

        // Handle null value
        if (trimmed_value == "null") {
            return "null";
        }

        // If we have schema information, use it
        if (param_config.contains(param_name)) {
            const auto & schema = param_config.at(std::string(param_name));
            if (schema.contains("type")) {
                const auto & t = schema.at("type");
                // Handle union types like ["number","null"]
                if (t.is_array()) {
                    std::vector<std::string> types;
                    for (const auto & tv : t) {
                        if (tv.is_string()) {
                            types.push_back((std::string) tv);
                        }
                    }
                    auto list_contains = [&](const char * s) {
                        for (const auto & x : types) {
                            if (x == s) return true;
                        }
                        return false;
                    };
                    auto has = [&](std::string_view ty) {
                        for (const auto & s : types) {
                            if (s == ty) return true;
                        }
                        // Back-compat synonyms
                        if (ty == "string")  return list_contains("str") || list_contains("text");
                        if (ty == "integer") return list_contains("int");
                        if (ty == "number")  return list_contains("float");
                        if (ty == "boolean") return list_contains("bool");
                        return false;
                    };
                    if (has("null") && trimmed_value == "null") {
                        return "null";
                    }
                    if (has("object") || has("array")) {
                        try {
                            auto parsed = json::parse(trimmed_value);
                            return parsed.dump();
                        } catch (...) {
                            return json(trimmed_value).dump();
                        }
                    }
                    if (has("integer")) {
                        int int_val;
                        if (safe_parse_int(trimmed_value, int_val)) {
                            return std::to_string(int_val);
                        }
                        // if integer parse fails, try number or fall through
                    }
                    if (has("number")) {
                        float float_val;
                        if (safe_parse_float(trimmed_value, float_val)) {
                            return std::to_string(float_val);
                        }
                    }
                    if (has("boolean")) {
                        if (trimmed_value == "true" || trimmed_value == "false") {
                            return trimmed_value;
                        }
                        return "false";
                    }
                    if (has("string")) {
                        return json(trimmed_value).dump();
                    }
                    // Unknown union types: fall through to generic inference below
                } else if (t.is_string()) {
                    std::string param_type = t;
                    // Convert based on type
                    if (param_type == "string" || param_type == "str" || param_type == "text") {
                        // SECURITY FIX: Use nlohmann::json for proper escaping instead of manual concatenation
                        return json(trimmed_value).dump();
                    } else if (param_type == "integer" || param_type == "int") {
                        int int_val;
                        if (safe_parse_int(trimmed_value, int_val)) {
                            return std::to_string(int_val);
                        } else {
                            // SECURITY FIX: Use proper JSON escaping for fallback string
                            return json(trimmed_value).dump();
                        }
                    } else if (param_type == "number" || param_type == "float") {
                        float float_val;
                        if (safe_parse_float(trimmed_value, float_val)) {
                            return std::to_string(float_val);
                        } else {
                            // SECURITY FIX: Use proper JSON escaping for fallback string
                            return json(trimmed_value).dump();
                        }
                    } else if (param_type == "boolean" || param_type == "bool") {
                        if (trimmed_value == "true" || trimmed_value == "false") {
                            return trimmed_value;
                        }
                        return "false";
                    } else if (param_type == "object" || param_type == "array") {
                        try {
                            auto parsed = json::parse(trimmed_value);
                            return parsed.dump();
                        } catch (...) {
                            // SECURITY FIX: Use proper JSON escaping for fallback string
                            return json(trimmed_value).dump();
                        }
                    }
                }
                // If schema.type exists but is not string/array, fall through
            }
        }

        // Without schema, try to infer type from value
        // First check if it's valid JSON (object or array)
        try {
            auto parsed_json = json::parse(trimmed_value);
            return parsed_json.dump(); // It's valid JSON, return as-is
        } catch (...) {
            // Not valid JSON, continue with other type checks
        }

        // Check if it's a number
        int int_val;
        if (safe_parse_int(trimmed_value, int_val)) {
            return std::to_string(int_val); // It's an integer
        }

        float float_val;
        if (safe_parse_float(trimmed_value, float_val)) {
            return std::to_string(float_val); // It's a float
        }

        // Check if it's a boolean
        if (trimmed_value == "true" || trimmed_value == "false") {
            return trimmed_value;
        }

        // Default to string - SECURITY FIX: Use proper JSON escaping
        return json(trimmed_value).dump();
    }

    // Get parameter configuration from tools using string_view
    nlohmann::json get_param_config(std::string_view func_name,
                                   const std::vector<common_chat_tool> & tools) {
        for (const auto & tool : tools) {
            if (tool.name == func_name) {
                try {
                    auto params = json::parse(tool.parameters);
                    if (params.contains("properties")) {
                        return params["properties"];
                    }
                    return params;
                } catch (...) {
                    return json::object();
                }
            }
        }
        return json::object();
    }
}

bool common_chat_msg_parser::parse_qwen3_xml_tool_call(const std::string & content,
                                                       const std::vector<common_chat_tool> & tools) {
    XmlParseError error;
    bool result = parse_qwen3_xml_tool_call(content, tools, error);
    last_xml_error_ = error;
    return result;
}

bool common_chat_msg_parser::parse_qwen3_xml_tool_call(const std::string & content,
                                                       const std::vector<common_chat_tool> & tools,
                                                       XmlParseError & error) {
    // Clear any previous error
    error.clear();

    // Input validation for DoS protection
    if (content.size() > MAX_INPUT_SIZE) {
        LOG_DBG("XML content too large: %zu bytes (max: %zu)\n", content.size(), MAX_INPUT_SIZE);
        set_error(error, XmlParseErrorType::INPUT_TOO_LARGE, 0,
                 content.substr(0, std::min(content.size(), size_t(100))),
                 "XML content exceeds maximum size limit of " + std::to_string(MAX_INPUT_SIZE) + " bytes");
        return false;
    }

    // Validate tools vector size
    if (tools.size() > MAX_PARAMETER_COUNT) {
        LOG_DBG("Too many tools provided: %zu (max: %zu)\n", tools.size(), MAX_PARAMETER_COUNT);
        set_error(error, XmlParseErrorType::TOO_MANY_TOOLS, 0, "",
                 "Too many tools provided: " + std::to_string(tools.size()) + " (max: " + std::to_string(MAX_PARAMETER_COUNT) + ")");
        return false;
    }

    // PERFORMANCE OPTIMIZATION: Create hash set for O(1) function lookup
    std::unordered_set<std::string> valid_functions;
    if (!tools.empty()) {
        valid_functions.reserve(tools.size());
        for (const auto & tool : tools) {
            valid_functions.insert(tool.name);
        }
    }

    // PERFORMANCE: Use string_view to avoid unnecessary string copies
    std::string_view content_view(content);

    // Find tool_call tag
    auto tool_call_tag = find_xml_tag(content_view, "tool_call", 0, &error);
    if (!tool_call_tag) {
        if (!error.has_error()) {
            set_error(error, XmlParseErrorType::INVALID_XML_STRUCTURE, 0, content.substr(0, std::min(content.size(), size_t(100))),
                     "No valid <tool_call> tag found in content");
        }
        return false;
    }

    // Extract content before the tool call - with bounds checking
    if (tool_call_tag->start_pos > 0 && tool_call_tag->start_pos <= content.size()) {
        std::string content_before = content.substr(0, tool_call_tag->start_pos);
        // Don't trim whitespace here as it might be significant for the content
        if (!content_before.empty()) {
            add_content(content_before);
        }
    }

    if (!tool_call_tag->end_pos) {
        return true;
    }

    // Find function tag within tool_call - use string_view for performance
    std::string_view tool_call_content_view(tool_call_tag->content);
    auto function_tag = find_xml_tag(tool_call_content_view, "function", 0, &error);
    if (!function_tag || function_tag->attribute.empty()) {
        LOG_DBG("Invalid or missing function tag in tool_call\n");
        if (!error.has_error()) {
            set_error(error, XmlParseErrorType::INVALID_XML_STRUCTURE, tool_call_tag->start_pos,
                     tool_call_tag->content.substr(0, std::min(tool_call_tag->content.size(), size_t(100))),
                     "Invalid or missing <function> tag with attribute in <tool_call>");
        }
        return false;
    }

    std::string function_name = trim_whitespace(function_tag->attribute);

    // Validate function name
    if (function_name.empty() || function_name.size() > MAX_TAG_NAME_LENGTH) {
        LOG_DBG("Invalid function name: '%s' (length: %zu, max: %zu)\n",
                function_name.c_str(), function_name.size(), MAX_TAG_NAME_LENGTH);
        set_error(error, XmlParseErrorType::INVALID_FUNCTION_NAME,
                 tool_call_tag->start_pos + function_tag->start_pos,
                 function_name,
                 "Invalid function name: '" + function_name + "' (length: " + std::to_string(function_name.size()) + ", max: " + std::to_string(MAX_TAG_NAME_LENGTH) + ")");
        return false;
    }

    // PERFORMANCE OPTIMIZATION: Use hash set for O(1) function lookup instead of O(n) loop
    if (!tools.empty() && valid_functions.find(function_name) == valid_functions.end()) {
        LOG_DBG("Function '%s' not found in available tools\n", function_name.c_str());
        set_error(error, XmlParseErrorType::FUNCTION_NOT_FOUND,
                 tool_call_tag->start_pos + function_tag->start_pos,
                 function_name,
                 "Function '" + function_name + "' not found in available tools");
        return false;
    }

    // Get parameter configuration for this function - use string_view
    auto param_config = get_param_config(std::string_view(function_name), tools);

    // Parse parameters within function tag - use string_view for performance
    json arguments = json::object();
    std::string_view function_content_view(function_tag->content);
    auto parameter_tags = find_all_xml_tags(function_content_view, "parameter", &error);

    // Check if error occurred during parameter parsing
    if (error.has_error()) {
        return false;
    }

    // Limit parameter count for DoS protection
    size_t param_count = 0;
    for (const auto & param_tag : parameter_tags) {
        if (param_count >= MAX_PARAMETER_COUNT) {
            LOG_DBG("Too many parameters for function '%s': %zu (max: %zu)\n",
                    function_name.c_str(), param_count, MAX_PARAMETER_COUNT);
            set_error(error, XmlParseErrorType::TOO_MANY_PARAMETERS,
                     tool_call_tag->start_pos + function_tag->start_pos,
                     function_name,
                     "Too many parameters for function '" + function_name + "': " + std::to_string(param_count) + " (max: " + std::to_string(MAX_PARAMETER_COUNT) + ")");
            break;
        }

        if (param_tag.attribute.empty()) {
            LOG_DBG("Skipping parameter with empty attribute\n");
            continue; // Skip malformed parameter tags
        }

        std::string param_name = trim_whitespace(param_tag.attribute);
        std::string param_value = param_tag.content;

        // Validate parameter name
        if (param_name.empty() || param_name.size() > MAX_TAG_NAME_LENGTH) {
            LOG_DBG("Invalid parameter name: '%s' (length: %zu, max: %zu)\n",
                    param_name.c_str(), param_name.size(), MAX_TAG_NAME_LENGTH);
            continue;
        }

        // Convert value based on schema type - use string_view for performance
        try {
            std::string converted_value = convert_qwen3_param_value(
                std::string_view(param_value),
                std::string_view(param_name),
                param_config,
                std::string_view(function_name)
            );
            arguments[param_name] = json::parse(converted_value);
            ++param_count;
        } catch (const std::exception & e) {
            LOG_DBG("Failed to convert parameter '%s': %s, using raw value\n", param_name.c_str(), e.what());
            set_error(error, XmlParseErrorType::PARAMETER_CONVERSION_FAILED,
                     tool_call_tag->start_pos + function_tag->start_pos + param_tag.start_pos,
                     param_name + "=" + param_value,
                     "Failed to convert parameter '" + param_name + "': " + e.what());
            // Fallback to trimmed raw value with proper JSON escaping
            arguments[param_name] = trim_whitespace(param_value);
            ++param_count;
        }
    }

    // Add the tool call with error handling
    try {
        std::string args_json = arguments.dump();
        return add_tool_call(function_name, "", args_json);
    } catch (const std::exception & e) {
        LOG_DBG("Failed to serialize arguments for function '%s': %s\n", function_name.c_str(), e.what());
        set_error(error, XmlParseErrorType::JSON_SERIALIZATION_FAILED,
                 tool_call_tag->start_pos,
                 function_name,
                 "Failed to serialize arguments for function '" + function_name + "': " + e.what());
        return false;
    }
}

