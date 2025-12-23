#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"

#include "log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

using json = nlohmann::ordered_json;

// Helper functions shared between analyzer and generator

bool string_ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ============================================================================
// String Manipulation Helpers
// ============================================================================

void trim_whitespace(std::string & str) {
    if (str.empty()) {
        return;
    }
    size_t first = str.find_first_not_of(" \n\t\r");
    if (first == std::string::npos) {
        str.clear();
        return;
    }
    size_t last = str.find_last_not_of(" \n\t\r");
    str = str.substr(first, (last - first + 1));
}

void trim_trailing_newlines(std::string & str) {
    while (!str.empty() && (str.back() == '\n' || str.back() == '\r')) {
        str.pop_back();
    }
}

void strip_markers(std::string & str, const std::string & prefix, const std::string & suffix) {
    if (str.empty()) {
        return;
    }
    if (str.find(prefix) == 0) {
        str = str.substr(prefix.length());
        LOG_DBG("Stripped prefix '%s' from string\n", prefix.c_str());
    }
    size_t suffix_pos = str.find(suffix);
    if (suffix_pos != std::string::npos) {
        str = str.substr(0, suffix_pos) + str.substr(suffix_pos + suffix.length());
        LOG_DBG("Stripped suffix '%s' from string\n", suffix.c_str());
    }
}

size_t count_non_whitespace(const std::string & str) {
    size_t count = 0;
    for (char c : str) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            count++;
        }
    }
    return count;
}

size_t find_last_of_any(const std::string & str, const std::string & chars, size_t start_pos) {
    size_t last_pos = std::string::npos;
    for (char c : chars) {
        size_t pos = str.rfind(c, start_pos);
        if (pos != std::string::npos && (last_pos == std::string::npos || pos > last_pos)) {
            last_pos = pos;
        }
    }
    return last_pos;
}

// ============================================================================
// Quote and Position Helpers
// ============================================================================

size_t find_with_quote_adjustment(const std::string & str, const std::string & target) {
    size_t pos = str.find("\"" + target + "\"");
    if (pos != std::string::npos) {
        return pos + 1;  // Skip the opening quote
    }
    return str.find(target);
}

// ============================================================================
// Template Application Helpers
// ============================================================================

std::string apply_template(const minja::chat_template & tmpl, const json & message) {
    minja::chat_template_inputs inputs;
    inputs.messages = { message };
    try {
        return tmpl.apply(inputs);
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return "";
    }
}

std::string apply_template(const minja::chat_template & tmpl, const json & messages, const json & tools) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.tools = tools.empty() ? json() : tools;
    try {
        return tmpl.apply(inputs);
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return "";
    }
}

// ============================================================================
// Pattern Matching Helpers
// ============================================================================

bool contains_thinking_pattern(const std::string & str, std::string & found_start, std::string & found_end) {
    std::vector<std::pair<std::string, std::string>> thinking_patterns = {
        { "<|START_THINKING|>", "<|END_THINKING|>" },
        { "<|thinking|>", "</thinking>" },
        { "[THINKING]", "[/THINKING]" },
        { "<thinking>", "</thinking>" }
    };

    for (const auto & [start_tag, end_tag] : thinking_patterns) {
        if (str.find(start_tag) != std::string::npos && str.find(end_tag) != std::string::npos) {
            found_start = start_tag;
            found_end = end_tag;
            return true;
        }
    }
    return false;
}

bool has_closed_reasoning_section(const std::string & str1, const std::string & str2,
                                   std::string & found_start, std::string & found_end) {
    std::vector<std::tuple<std::string, std::string, std::string, std::string>> thinking_patterns = {
        { "<|START_THINKING|>", "THINKING|>", "<|END_THINKING|>", "<|START_THINKING|>" },
        { "<|thinking|>", "thinking>", "</thinking>", "<|thinking>" },
        { "[THINKING]", "THINKING]", "[/THINKING]", "[THINKING]" },
        { "<thinking>", "thinking>", "</thinking>", "<thinking>" }
    };

    for (const auto & [full_start, partial_start, end_tag, actual_start] : thinking_patterns) {
        bool found_in_str1 = (str1.find(full_start) != std::string::npos ||
                              str1.find(partial_start) != std::string::npos) &&
                             str1.find(end_tag) != std::string::npos;
        bool found_in_str2 = (str2.find(full_start) != std::string::npos ||
                              str2.find(partial_start) != std::string::npos) &&
                             str2.find(end_tag) != std::string::npos;

        if (found_in_str1 || found_in_str2) {
            found_start = actual_start;
            found_end = end_tag;
            return true;
        }
    }
    return false;
}

// ============================================================================
// Tag Extraction Helpers
// ============================================================================

std::string extract_tag_name(const std::string & tag) {
    if (tag.empty() || tag[0] != '<') {
        return "";
    }
    std::string tag_name = tag.substr(1);
    size_t end_bracket = tag_name.find_first_of(" >");
    if (end_bracket != std::string::npos) {
        tag_name = tag_name.substr(0, end_bracket);
    }
    return tag_name;
}

std::string create_closing_tag(const std::string & opening_tag) {
    if (opening_tag.empty()) {
        return "";
    }
    if (opening_tag[0] == '<') {
        std::string name = extract_tag_name(opening_tag);
        return "</" + name + ">";
    }
    if (opening_tag.front() == '[' && opening_tag.back() == ']') {
        std::string name = opening_tag.substr(1, opening_tag.length() - 2);
        return "[/" + name + "]";
    }
    return "";
}

std::string extract_bracket_tag_name(const std::string & tag) {
    if (tag.empty() || tag[0] != '[' || tag.back() != ']') {
        return "";
    }
    return tag.substr(1, tag.length() - 2);
}

// ============================================================================
// Common String Helpers
// ============================================================================

std::string find_common_prefix(const std::vector<std::string> & strings) {
    if (strings.empty()) {
        return "";
    }
    if (strings.size() == 1) {
        return strings[0];
    }

    std::string common = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        const std::string & current = strings[i];
        std::string         temp_common;
        for (size_t j = 0; j < common.length() && j < current.length(); ++j) {
            if (common[j] == current[j]) {
                temp_common += common[j];
            } else {
                break;
            }
        }
        common = temp_common;
    }
    return common;
}

std::string find_common_suffix_generic(const std::vector<std::string> & strings) {
    if (strings.empty()) {
        return "";
    }
    if (strings.size() == 1) {
        return strings[0];
    }

    std::string common = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        const std::string & current = strings[i];
        std::string         temp_common;
        size_t              min_len = std::min(common.length(), current.length());
        for (size_t j = 0; j < min_len; ++j) {
            size_t pos_common = common.length() - j - 1;
            size_t pos_current = current.length() - j - 1;
            if (common[pos_common] == current[pos_current]) {
                temp_common = common[pos_common] + temp_common;
            } else {
                break;
            }
        }
        common = temp_common;
    }
    return common;
}

std::string find_common_substring_limited(const std::vector<std::string> & strings,
                                           size_t max_length, const std::string & delimiters) {
    std::string common = find_common_prefix(strings);
    if (common.length() > max_length) {
        size_t pos = find_last_of_any(common, delimiters, common.length() - 1);
        if (pos != std::string::npos && pos > 0) {
            return common.substr(0, pos + 1);
        }
        return common.substr(0, max_length);
    }
    return common;
}

std::string find_first_pattern(const std::string & str, const std::vector<std::string> & candidates,
                                size_t start_pos) {
    for (const auto & pattern : candidates) {
        size_t pos = str.find(pattern, start_pos);
        if (pos != std::string::npos) {
            return pattern;
        }
    }
    return "";
}

// ============================================================================
// Token Collection Helpers
// ============================================================================

void collect_non_empty_tokens(const DiscoveredPattern & discovered, std::vector<std::string> & tokens) {
    if (!discovered.tool_call_opener.empty()) {
        tokens.push_back(discovered.tool_call_opener);
    }
    if (!discovered.tool_call_closer.empty()) {
        tokens.push_back(discovered.tool_call_closer);
    }
    if (!discovered.function_opener.empty()) {
        tokens.push_back(discovered.function_opener);
    }
    if (!discovered.function_closer.empty()) {
        tokens.push_back(discovered.function_closer);
    }
    if (!discovered.parameter_opener.empty()) {
        tokens.push_back(discovered.parameter_opener);
    }
    if (!discovered.parameter_closer.empty()) {
        tokens.push_back(discovered.parameter_closer);
    }
    if (!discovered.tool_call_start_marker.empty()) {
        tokens.push_back(discovered.tool_call_start_marker);
    }
    if (!discovered.tool_call_end_marker.empty()) {
        tokens.push_back(discovered.tool_call_end_marker);
    }
    if (!discovered.reasoning_start_marker.empty()) {
        tokens.push_back(discovered.reasoning_start_marker);
    }
    if (!discovered.reasoning_end_marker.empty()) {
        tokens.push_back(discovered.reasoning_end_marker);
    }
    if (!discovered.content_start_marker.empty()) {
        tokens.push_back(discovered.content_start_marker);
    }
    if (!discovered.content_end_marker.empty()) {
        tokens.push_back(discovered.content_end_marker);
    }
}

std::string apply_template(const minja::chat_template &    tmpl,
                        const struct templates_params & inputs,
                        const std::optional<json> &     messages_override,
                        const std::optional<json> &     tools_override,
                        const std::optional<json> &     additional_context) {
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = messages_override ? *messages_override : inputs.messages;
    if (tools_override) {
        tmpl_inputs.tools = *tools_override;
    } else {
        tmpl_inputs.tools = inputs.tools.empty() ? json() : inputs.tools;
    }
    tmpl_inputs.add_generation_prompt            = inputs.add_generation_prompt;
    tmpl_inputs.extra_context                    = inputs.extra_context;
    tmpl_inputs.extra_context["enable_thinking"] = inputs.enable_thinking;
    if (additional_context) {
        tmpl_inputs.extra_context.merge_patch(*additional_context);
    }

    minja::chat_template_options tmpl_opts;
    try {
        auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
        return result;
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return "";
    }
}
