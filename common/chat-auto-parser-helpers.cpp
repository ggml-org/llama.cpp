#include "chat-auto-parser-helpers.h"

#include "chat-auto-parser.h"
#include "chat.h"
#include "log.h"

#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

bool string_ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

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
    str         = str.substr(first, (last - first + 1));
}

void trim_trailing_newlines(std::string & str) {
    while (!str.empty() && (str.back() == '\n' || str.back() == '\r')) {
        str.pop_back();
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

std::string extract_tag_name(const std::string & tag) {
    if (tag.empty() || tag[0] != '<') {
        return "";
    }
    std::string tag_name    = tag.substr(1);
    size_t      end_bracket = tag_name.find_first_of(" >");
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
            size_t pos_common  = common.length() - j - 1;
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
                                          size_t                           max_length,
                                          const std::string &              delimiters) {
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

std::string apply_template(common_chat_template      &    tmpl,
                           const struct templates_params & inputs,
                           const std::optional<json> &     messages_override,
                           const std::optional<json> &     tools_override,
                           const std::optional<json> &     additional_context) {
    struct templates_params final_inputs(inputs);
    final_inputs.messages = messages_override ? *messages_override : inputs.messages;
    if (tools_override) {
        final_inputs.tools = *tools_override;
    } else {
        final_inputs.tools = inputs.tools.empty() ? json() : inputs.tools;
    }
    final_inputs.add_generation_prompt            = inputs.add_generation_prompt;
    final_inputs.extra_context                    = inputs.extra_context;
    final_inputs.extra_context["enable_thinking"] = inputs.enable_thinking;
    if (additional_context) {
        final_inputs.extra_context.merge_patch(*additional_context);
    }

    try {
        return common_chat_template_direct_apply(tmpl, inputs);
    } catch (const std::exception & e) {
        LOG_ERR("Template application failed: %s\n", e.what());
        return "";
    }
}

std::string adjust_to_token_boundary(const std::string & str) {
    if (str.empty()) {
        return str;
    }

    // Check if the string ends in the middle of a <|...|> token
    // Look for unmatched <| at the end

    // Find the last <| in the string
    size_t last_open = str.rfind("<|");
    if (last_open == std::string::npos) {
        return str;  // No special tokens
    }

    // Find if there's a |> after the last <|
    size_t matching_close = str.find("|>", last_open + 2);
    if (matching_close != std::string::npos) {
        // The token is complete, return as-is
        return str;
    }

    // The string is truncated mid-token
    // Truncate to just before the incomplete token
    std::string result = str.substr(0, last_open);

    // Trim any trailing whitespace
    while (!result.empty() && (result.back() == ' ' || result.back() == '\t' || result.back() == '\n')) {
        result.pop_back();
    }

    return result;
}

// Fullwidth vertical bar: ｜ (U+FF5C) is 3 bytes in UTF-8: 0xEF 0xBD 0x9C
static const std::string FULLWIDTH_PIPE   = "\xef\xbd\x9c";        // ｜
static const std::string TOKEN_OPENER_STD = "<|";
static const std::string TOKEN_OPENER_FW  = "<" + FULLWIDTH_PIPE;  // <｜
static const std::string TOKEN_CLOSER_STD = "|>";
static const std::string TOKEN_CLOSER_FW  = FULLWIDTH_PIPE + ">";  // ｜>

size_t find_token_opener(const std::string & str, size_t start_pos) {
    size_t pos_std = str.find(TOKEN_OPENER_STD, start_pos);
    size_t pos_fw  = str.find(TOKEN_OPENER_FW, start_pos);

    if (pos_std == std::string::npos) {
        return pos_fw;
    }
    if (pos_fw == std::string::npos) {
        return pos_std;
    }
    return std::min(pos_std, pos_fw);
}

size_t find_token_closer(const std::string & str, size_t start_pos) {
    size_t pos_std = str.find(TOKEN_CLOSER_STD, start_pos);
    size_t pos_fw  = str.find(TOKEN_CLOSER_FW, start_pos);

    if (pos_std == std::string::npos) {
        return pos_fw;
    }
    if (pos_fw == std::string::npos) {
        return pos_std;
    }
    return std::min(pos_std, pos_fw);
}

size_t get_token_opener_length(const std::string & str, size_t pos) {
    if (pos >= str.length()) {
        return 0;
    }
    if (str.compare(pos, TOKEN_OPENER_FW.length(), TOKEN_OPENER_FW) == 0) {
        return TOKEN_OPENER_FW.length();  // 4 bytes for <｜
    }
    if (str.compare(pos, TOKEN_OPENER_STD.length(), TOKEN_OPENER_STD) == 0) {
        return TOKEN_OPENER_STD.length();  // 2 bytes for <|
    }
    return 0;
}

size_t get_token_closer_length(const std::string & str, size_t pos) {
    if (pos >= str.length()) {
        return 0;
    }
    if (str.compare(pos, TOKEN_CLOSER_FW.length(), TOKEN_CLOSER_FW) == 0) {
        return TOKEN_CLOSER_FW.length();  // 4 bytes for ｜>
    }
    if (str.compare(pos, TOKEN_CLOSER_STD.length(), TOKEN_CLOSER_STD) == 0) {
        return TOKEN_CLOSER_STD.length();  // 2 bytes for |>
    }
    return 0;
}

std::string strip_eos_token(const std::string & str) {
    if (str.empty()) {
        return str;
    }

    // Find the last token in the string
    // We need to find a token that looks like an EOS marker
    // Common patterns:
    // - <|eot_id|>, <|eos|>, <|end|>, <|endoftext|>
    // - <｜end▁of▁sentence｜> (DeepSeek fullwidth)

    size_t last_closer = std::string::npos;
    size_t search_pos  = str.length();

    // Search backwards for the last token closer
    while (search_pos > 0) {
        // Check for fullwidth closer first (it's longer)
        if (search_pos >= TOKEN_CLOSER_FW.length()) {
            size_t check_pos = search_pos - TOKEN_CLOSER_FW.length();
            if (str.compare(check_pos, TOKEN_CLOSER_FW.length(), TOKEN_CLOSER_FW) == 0) {
                last_closer = check_pos;
                break;
            }
        }
        // Check for standard closer
        if (search_pos >= TOKEN_CLOSER_STD.length()) {
            size_t check_pos = search_pos - TOKEN_CLOSER_STD.length();
            if (str.compare(check_pos, TOKEN_CLOSER_STD.length(), TOKEN_CLOSER_STD) == 0) {
                last_closer = check_pos;
                break;
            }
        }
        search_pos--;
    }

    if (last_closer == std::string::npos) {
        return str;  // No token closer found
    }

    // Find the corresponding opener
    size_t opener_search_start = (last_closer > 100) ? last_closer - 100 : 0;
    size_t last_opener         = std::string::npos;
    size_t opener_len          = 0;

    for (size_t pos = opener_search_start; pos < last_closer; pos++) {
        size_t len = get_token_opener_length(str, pos);
        if (len > 0) {
            last_opener = pos;
            opener_len  = len;
        }
    }

    if (last_opener == std::string::npos) {
        return str;  // No matching opener found
    }

    // Extract the token content to check if it's an EOS marker
    size_t closer_len     = get_token_closer_length(str, last_closer);
    size_t content_start  = last_opener + opener_len;
    size_t content_length = last_closer - content_start;

    if (content_length == 0 || content_length > 50) {
        return str;  // Invalid or too long token content
    }

    std::string token_content = str.substr(content_start, content_length);

    // Convert to lowercase for comparison (ASCII only, sufficient for EOS markers)
    std::string lower_content;
    for (char c : token_content) {
        lower_content += (c >= 'A' && c <= 'Z') ? (c + 32) : c;
    }

    // Check if this looks like an EOS token
    // True EOS tokens:
    //   - <|eos|>, <|eot_id|>, <|end_of_text|>, <|endoftext|>
    //   - <｜end▁of▁sentence｜> (DeepSeek fullwidth)
    // NOT EOS tokens (structural markers):
    //   - <|END_ACTION|>, <|TOOL_CALL_END|>, <|end_thinking|>, etc.

    bool is_eos = false;

    // Check for specific EOS patterns
    if (lower_content == "eos" || lower_content == "eot_id" || lower_content == "eot" ||
        lower_content == "end_of_text" || lower_content == "endoftext") {
        is_eos = true;
    }
    // DeepSeek's end_of_sentence uses fullwidth underscore (▁) which is preserved in lower_content
    // The token content would be "end▁of▁sentence" (with ▁ = U+2581)
    else if (token_content.find("sentence") != std::string::npos ||
             token_content.find("\xe2\x96\x81of\xe2\x96\x81sentence") != std::string::npos) {
        is_eos = true;
    }

    if (!is_eos) {
        return str;  // Not an EOS token
    }

    // Strip the EOS token
    std::string result = str.substr(0, last_opener);

    LOG_DBG("Stripped EOS token '%s' from string\n",
            str.substr(last_opener, last_closer + closer_len - last_opener).c_str());

    return result;
}

std::string find_string_difference(const std::string & base, const std::string & extended) {
    size_t common_prefix = 0;
    while (common_prefix < base.length() && common_prefix < extended.length() &&
           base[common_prefix] == extended[common_prefix]) {
        common_prefix++;
    }
    return extended.substr(common_prefix);
}

std::string extract_json_field_name(const std::string &              opener,
                                    const std::string &              default_name,
                                    const std::vector<std::string> & candidates) {
    for (const auto & candidate : candidates) {
        std::string pattern = "\"" + candidate + "\"";
        if (opener.find(pattern) != std::string::npos) {
            LOG_DBG("Found JSON field name '%s' in opener\n", candidate.c_str());
            return candidate;
        }
    }
    return default_name;
}

std::string find_closing_pattern(const std::string & diff, size_t func_pos) {
    std::vector<std::string> closers = { "</", "}", "]", ">", " " };

    std::string best_pattern;
    size_t      best_pos = std::string::npos;

    for (const auto & pattern : closers) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos != std::string::npos) {
            if (pos < best_pos) {
                if (pattern == "</") {
                    size_t end_pos = diff.find('>', pos);
                    if (end_pos != std::string::npos) {
                        best_pattern = diff.substr(pos, end_pos - pos + 1);
                        best_pos     = pos;
                    }
                } else {
                    best_pattern = pattern;
                    best_pos     = pos;
                }
            }
        }
    }
    return best_pattern;
}

std::string find_tool_call_start(const std::string & diff) {
    std::vector<std::string> start_patterns = { "<", "[", "{", "call", "func", "tool", "TOOL" };
    for (const auto & pattern : start_patterns) {
        size_t pos = diff.find(pattern);
        if (pos < 5) {
            if (pattern == "<") {
                size_t end_pos = diff.find('>', pos);
                if (end_pos != std::string::npos) {
                    return diff.substr(pos, end_pos - pos + 1);
                }
            }
            if (pattern == "[" || pattern == "{") {
                size_t chunk_len = std::min(diff.length() - pos, (size_t) 60);
                return diff.substr(pos, chunk_len);
            }

            size_t end_pos = diff.find_first_of(">]} \n", pos);
            if (end_pos != std::string::npos) {
                if (diff[end_pos] == '>' || diff[end_pos] == ']' || diff[end_pos] == '}') {
                    return diff.substr(pos, end_pos - pos + 1);
                }
                return diff.substr(pos, end_pos - pos);
            }
            return diff.substr(pos, pattern.length());
        }
    }
    return "";
}

std::string find_tool_call_end(const std::string & diff, size_t func_pos) {
    char        opener_char = 0;
    std::string start_tag_name;

    std::string openers         = "[{<";
    size_t      last_opener_pos = std::string::npos;
    for (char c : openers) {
        size_t p = diff.rfind(c, func_pos);
        if (p != std::string::npos) {
            if (last_opener_pos == std::string::npos || p > last_opener_pos) {
                last_opener_pos = p;
                opener_char     = c;
            }
        }
    }

    size_t unclosed_bracket = diff.rfind('[', func_pos);
    if (unclosed_bracket != std::string::npos) {
        size_t closer = diff.find(']', unclosed_bracket);
        if (closer == std::string::npos || closer > func_pos) {
            opener_char = '[';
        }
    }

    if (opener_char == '<') {
        size_t tag_start = diff.find('<', last_opener_pos);
        if (tag_start != std::string::npos) {
            // Include '=' in search to handle <function=name> style tags
            // where the closing tag is </function>, not </function=name>
            size_t tag_end = diff.find_first_of(" >=\n", tag_start);
            if (tag_end != std::string::npos) {
                start_tag_name = diff.substr(tag_start + 1, tag_end - (tag_start + 1));
            }
        }
    }

    if (!start_tag_name.empty()) {
        std::string expected_closer = "</" + start_tag_name + ">";
        size_t      pos             = diff.find(expected_closer, func_pos);
        if (pos != std::string::npos) {
            if (opener_char == '[') {
                size_t bracket_pos = diff.rfind(']', pos);
                if (bracket_pos != std::string::npos && bracket_pos > func_pos) {
                    return diff.substr(bracket_pos, (pos + expected_closer.length()) - bracket_pos);
                }
            }
            return expected_closer;
        }
    }

    std::vector<std::string> end_patterns = { "</", "]", "}", ">", "```", "\n", " " };
    std::string              best_pattern;
    size_t                   best_pos = std::string::npos;

    auto is_structural = [](const std::string & s) {
        if (s.empty()) {
            return false;
        }
        return s[0] == ']' || s[0] == '}' || s[0] == '>' || (s.size() >= 2 && s.substr(0, 2) == "</") ||
               (s.size() >= 3 && s.substr(0, 3) == "```");
    };

    for (const auto & pattern : end_patterns) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos == std::string::npos) {
            continue;
        }

        bool current_is_struct = is_structural(pattern);
        bool best_is_struct    = is_structural(best_pattern);

        bool better = false;
        if (best_pattern.empty()) {
            better = true;
        } else if (pos < best_pos) {
            better = !(best_is_struct && !current_is_struct) &&
                     !(opener_char == '[' && best_pattern[0] == ']' && pattern[0] == '}');
        } else {
            if (!best_is_struct && current_is_struct && pos < best_pos + 400) {
                better = true;
            } else if (best_is_struct && current_is_struct && opener_char == '[' && pattern[0] == ']' &&
                       best_pattern[0] == '}') {
                if (pos < best_pos + 100) {
                    better = true;
                }
            }
        }

        if (better) {
            best_pattern = pattern;
            best_pos     = pos;

            if (current_is_struct && (pattern == "]" || pattern == "}" || pattern == "```")) {
                size_t tag_start = diff.find('<', best_pos + pattern.length());
                if (tag_start != std::string::npos && tag_start < best_pos + pattern.length() + 5) {
                    size_t tag_end = diff.find('>', tag_start);
                    if (tag_end != std::string::npos) {
                        best_pattern = diff.substr(best_pos, tag_end - best_pos + 1);
                    }
                }
            }
        }
    }

    return best_pattern;
}

std::string infer_tool_call_opener(const std::string & diff1, const std::string & diff2, const std::string & diff3) {
    std::vector<std::string> differences = { diff1, diff2, diff3 };
    return find_common_prefix(differences);
}

std::string infer_tool_call_closer(const std::string & diff1, const std::string & diff2, const std::string & diff3) {
    std::vector<std::string> differences = { diff1, diff2, diff3 };
    return find_common_suffix_generic(differences);
}

internal_discovered_pattern extract_patterns_from_differences(const std::string & tool1_diff,
                                                              const std::string & tool2_diff,
                                                              const std::string & tool3_diff,
                                                              const std::string & tool1_full) {
    LOG_DBG("%s\n", __func__);

    internal_discovered_pattern patterns;

    size_t func1_pos = tool1_diff.rfind("test_function_name");
    size_t func2_pos = tool2_diff.rfind("test_function_name");

    if (func1_pos != std::string::npos && func2_pos != std::string::npos) {
        patterns.tool_call_opener = tool1_diff.substr(0, func1_pos);

        if (tool1_full.length() >= tool1_diff.length()) {
            size_t diff_start = tool1_full.length() - tool1_diff.length();

            if (diff_start > 0 && tool1_full[diff_start - 1] == '<' && !patterns.tool_call_opener.empty() &&
                patterns.tool_call_opener[0] != '<') {
                patterns.tool_call_opener = "<" + patterns.tool_call_opener;
            }
        }

        if (func1_pos == 0 && !tool1_full.empty()) {
            size_t func_in_full = tool1_full.rfind("test_function_name");
            if (func_in_full != std::string::npos && func_in_full > 0) {
                // Look backwards from function name to find prefix pattern
                // Find where the prefix ends (skip whitespace immediately before function name)
                size_t prefix_end = func_in_full;
                while (prefix_end > 0 && (tool1_full[prefix_end - 1] == ' ' || tool1_full[prefix_end - 1] == '\t')) {
                    prefix_end--;
                }

                // Find where the prefix starts by looking for newline or alphanumeric boundary
                size_t prefix_start = prefix_end;
                while (prefix_start > 0) {
                    char c = tool1_full[prefix_start - 1];
                    // Stop at newline
                    if (c == '\n' || c == '\r') {
                        break;
                    }
                    // Stop if we hit alphanumeric (probably content, not a prefix delimiter)
                    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                        prefix_start = prefix_end;  // Reset - no valid prefix found
                        break;
                    }
                    prefix_start--;
                }

                // Extract the prefix if we found something meaningful
                if (prefix_start < prefix_end) {
                    std::string prefix      = tool1_full.substr(prefix_start, prefix_end - prefix_start);
                    // Validate: prefix should contain non-whitespace and be reasonable length
                    bool        has_content = false;
                    for (char c : prefix) {
                        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                            has_content = true;
                            break;
                        }
                    }
                    if (has_content && prefix.length() >= 2 && prefix.length() <= 20) {
                        LOG_DBG("Found prefix pattern in full output: '%s'\n", prefix.c_str());
                        patterns.function_opener        = prefix;
                        patterns.tool_call_start_marker = prefix;
                    }
                }
            }
        }

        patterns.tool_name_field = extract_json_field_name(patterns.tool_call_opener, "name",
                                                           { "tool_name", "name", "function_name", "function" });

        patterns.tool_args_field =
            extract_json_field_name(patterns.tool_call_opener + tool1_diff.substr(func1_pos), "arguments",
                                    { "parameters", "arguments", "args", "params", "input" });

        patterns.tool_id_field =
            extract_json_field_name(tool1_diff, "", { "tool_call_id", "tool_id", "id", "call_id" });

        size_t param1_pos       = tool2_diff.find("\"param1\"");
        bool   param_has_quotes = (param1_pos != std::string::npos);
        size_t param2_pos       = tool2_diff.find("\"param2\"");
        size_t value1_pos       = tool2_diff.find("\"value1\"");

        if (param1_pos == std::string::npos) {
            param1_pos = tool2_diff.find("param1");
        }
        if (param_has_quotes && param1_pos != std::string::npos) {
            param1_pos++;
        }
        if (param2_pos == std::string::npos) {
            param2_pos = tool2_diff.find("param2");
        }
        if (param_has_quotes && param2_pos != std::string::npos) {
            param2_pos++;
        }
        if (value1_pos == std::string::npos) {
            value1_pos = tool2_diff.find("value1");
        }
        // Only skip quote if value was actually found quoted
        bool value_has_quotes = (value1_pos != std::string::npos && tool2_diff[value1_pos] == '"');
        if (value_has_quotes) {
            value1_pos++;
        }

        if (param1_pos != std::string::npos && value1_pos != std::string::npos) {
            size_t      search_start = (param1_pos > 20) ? param1_pos - 20 : 0;
            std::string pre_param    = tool2_diff.substr(search_start, param1_pos - search_start);

            size_t delim_pos = pre_param.find_last_of('\n');
            if (delim_pos == std::string::npos) {
                delim_pos = pre_param.find_last_of('>');
            }

            if (delim_pos != std::string::npos) {
                patterns.parameter_key_prefix = pre_param.substr(delim_pos + 1);

                // If prefix is empty after '>', check for GLM-style key-value tags
                // Pattern: <arg_key>param1</arg_key><arg_value>value1</arg_value>
                // In this case, the '>' ends the opening tag, and we should include the whole tag
                if (patterns.parameter_key_prefix.empty() && delim_pos > 0) {
                    // Look for matching '<' before the '>'
                    size_t open_bracket = pre_param.rfind('<', delim_pos);
                    if (open_bracket != std::string::npos) {
                        // Extract the whole tag as the prefix
                        patterns.parameter_key_prefix = pre_param.substr(open_bracket);
                    }
                }
            } else {
                size_t start_marker = pre_param.find_last_of("<{[ \"");
                if (start_marker != std::string::npos) {
                    patterns.parameter_key_prefix = pre_param.substr(start_marker);
                } else {
                    patterns.parameter_key_prefix = pre_param;
                }
            }

            trim_whitespace(patterns.parameter_key_prefix);

            size_t key_end = param1_pos + std::string("param1").length();
            if (value1_pos > key_end) {
                patterns.parameter_key_suffix = tool2_diff.substr(key_end, value1_pos - key_end);
            }

            size_t value1_end = value1_pos + std::string("value1").length();
            if (value1_end < tool2_diff.length()) {
                // Try to find XML-style closing tag like </parameter>
                size_t close_start = tool2_diff.find("</", value1_end);
                if (close_start != std::string::npos) {
                    size_t close_end = tool2_diff.find('>', close_start);
                    if (close_end != std::string::npos) {
                        patterns.parameter_closer = tool2_diff.substr(close_start, close_end - close_start + 1);
                    }
                }
            }
        }

        const std::string & func_context = tool1_diff;
        size_t              open_pos     = func_context.rfind('<', func1_pos);
        if (open_pos != std::string::npos && open_pos < func1_pos) {
            size_t close_pos = func_context.find('>', open_pos);
            if (close_pos != std::string::npos && close_pos < func1_pos) {
                bool is_adjacent = true;
                for (size_t k = close_pos + 1; k < func1_pos; ++k) {
                    char c = func_context[k];
                    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                        is_adjacent = false;
                        break;
                    }
                }
                if (is_adjacent) {
                    patterns.function_opener = func_context.substr(open_pos, close_pos - open_pos + 1);
                }
            } else {
                patterns.function_opener = func_context.substr(open_pos, func1_pos - open_pos);
            }
        }

        if (func1_pos > 0 && patterns.function_opener.empty()) {
            size_t prefix_end = func1_pos;
            // Skip whitespace immediately before function name
            while (prefix_end > 0 && (func_context[prefix_end - 1] == ' ' || func_context[prefix_end - 1] == '\t')) {
                prefix_end--;
            }

            // Find prefix start - look for newline or alphanumeric boundary
            size_t prefix_start = prefix_end;
            while (prefix_start > 0) {
                char c = func_context[prefix_start - 1];
                if (c == '\n' || c == '\r') {
                    break;
                }
                if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                    prefix_start = prefix_end;  // Reset - no valid prefix
                    break;
                }
                prefix_start--;
            }

            if (prefix_start < prefix_end) {
                // ...
            }
        }

        // Fallback: look for standard delimiters
        if (patterns.function_opener.empty()) {
            for (int i = (int) func1_pos - 1; i >= 0; i--) {
                if (func_context[i] == '{' || func_context[i] == '[' || func_context[i] == '(' ||
                    func_context[i] == '<') {
                    patterns.function_opener = func_context.substr(i, func1_pos - i);
                    break;
                }
            }
        }

        size_t func_name_end = func1_pos + std::string("test_function_name").length();
        if (func_name_end < func_context.length()) {
            char next_char = func_context[func_name_end];
            if (next_char == '>' || next_char == ']' || next_char == '}') {
                patterns.function_name_suffix = std::string(1, next_char);
            } else if (next_char == '"') {
                if (func_name_end + 1 < func_context.length() && func_context[func_name_end + 1] == '>') {
                    patterns.function_name_suffix = "\">";
                } else {
                    patterns.function_name_suffix = "\"";
                }
            } else if (next_char == '<') {
                // Check if it's an XML-like tag suffix (e.g. <|tool_call_argument_begin|>)
                // But NOT if it's a closing tag (e.g., </tool_call>) - that should be function_closer
                if (func_name_end + 1 < func_context.length() && func_context[func_name_end + 1] == '/') {
                    // This is a closing tag like </tool_call>, not a suffix
                    // Leave function_name_suffix empty; function_closer will capture this
                } else {
                    size_t tag_close = func_context.find('>', func_name_end);
                    if (tag_close != std::string::npos) {
                        // It seems to be a tag, use it as suffix
                        patterns.function_name_suffix = func_context.substr(func_name_end, tag_close - func_name_end + 1);
                    }
                }
            } else if (next_char == '[') {
                // Bracket-tag format: [CALL_ID]id[ARGS] (Mistral Small 3.2 style)
                // Find where the JSON arguments start (at '{')
                size_t json_start = func_context.find('{', func_name_end);
                if (json_start != std::string::npos) {
                    patterns.function_name_suffix = func_context.substr(func_name_end, json_start - func_name_end);
                    LOG_DBG("Found bracket-tag suffix: '%s'\n", patterns.function_name_suffix.c_str());
                }
            } else if (next_char == ':') {
                // Indexed format: function_name:0<|marker|> or function_name:0{args}
                // Find where the suffix ends - either at a tag marker or at the JSON args start
                size_t suffix_end = func_name_end + 1;
                // Skip the index digits
                while (suffix_end < func_context.length() &&
                       std::isdigit(static_cast<unsigned char>(func_context[suffix_end]))) {
                    suffix_end++;
                }
                if (suffix_end < func_context.length()) {
                    char after_index = func_context[suffix_end];
                    if (after_index == '<') {
                        // There's a marker after the index (e.g., :0<|tool_call_argument_begin|>)
                        size_t tag_close = func_context.find('>', suffix_end);
                        if (tag_close != std::string::npos) {
                            patterns.function_name_suffix =
                                func_context.substr(func_name_end, tag_close - func_name_end + 1);
                        } else {
                            patterns.function_name_suffix =
                                func_context.substr(func_name_end, suffix_end - func_name_end);
                        }
                    } else {
                        // Just the index part (e.g., :0)
                        patterns.function_name_suffix = func_context.substr(func_name_end, suffix_end - func_name_end);
                    }
                }
            } else if (next_char == '\n' || next_char == '\r') {
                // Check for markdown code block pattern (e.g., DeepSeek R1): \n```json\n{...}\n```<end>
                size_t code_block_start = func_context.find("```", func_name_end);
                if (code_block_start != std::string::npos && code_block_start < func_name_end + 10) {
                    // Found code block start after function name
                    // Skip the optional language tag (e.g., "json")
                    size_t newline_after_lang = func_context.find('\n', code_block_start + 3);
                    if (newline_after_lang != std::string::npos) {
                        // function_name_suffix should include everything up to (and including) the newline after language tag
                        patterns.function_name_suffix =
                            func_context.substr(func_name_end, newline_after_lang - func_name_end + 1);
                        LOG_DBG("Found markdown code block suffix: '%s'\n", patterns.function_name_suffix.c_str());
                    }
                }
            }
        }

        // Function closer
        size_t search_start = func_name_end;
        if (!patterns.function_name_suffix.empty()) {
            search_start += patterns.function_name_suffix.length();
        }
        patterns.function_closer = find_closing_pattern(func_context, search_start);

        // Fix for XML-style tag formats where function_closer was detected as "}" (JSON closing)
        // but should be the actual tag closer (e.g., <|tool_call_end|> or <｜tool▁call▁end｜>)
        if (patterns.function_closer == "}" && !patterns.function_opener.empty() &&
            patterns.function_opener[0] == '<') {
            // This is an XML-style tag format, so the closer should be a tag, not just "}"
            // Find the next tag marker after the search position
            size_t next_tag = func_context.find('<', search_start);
            if (next_tag != std::string::npos) {
                // Handle both standard <|...|> and fullwidth <｜...｜> formats
                size_t closer_pos = find_token_closer(func_context, next_tag);
                if (closer_pos != std::string::npos) {
                    size_t closer_len        = get_token_closer_length(func_context, closer_pos);
                    patterns.function_closer = func_context.substr(next_tag, closer_pos - next_tag + closer_len);
                    LOG_DBG("Adjusted function_closer from '}' to tag '%s' for XML-style format\n",
                            patterns.function_closer.c_str());
                }
            }
        }

        if (patterns.function_closer == "}" && !patterns.function_name_suffix.empty() &&
            patterns.function_name_suffix.find("```") != std::string::npos) {
            // function_name_suffix contains a code block opener, look for the closing code block
            size_t code_block_end = func_context.find("```", search_start);
            if (code_block_end != std::string::npos) {
                // Found closing code block, extract everything from ``` to end of tool call
                // The closer should be \n```<per_call_end> (everything from ``` to the end marker)
                size_t after_block = code_block_end + 3;
                // Find the next tag marker (e.g., <|tool_call_end|>)
                size_t next_tag    = func_context.find('<', after_block);
                if (next_tag != std::string::npos) {
                    size_t tag_end = func_context.find('>', next_tag);
                    if (tag_end != std::string::npos) {
                        // Don't include leading newline - the JSON args parser consumes trailing whitespace
                        // So start exactly at the ``` (code_block_end)
                        patterns.function_closer = func_context.substr(code_block_end, tag_end - code_block_end + 1);
                        LOG_DBG("Detected markdown code block args, adjusted function_closer to: '%s'\n",
                                patterns.function_closer.c_str());
                    }
                }
            }
        }

        // Tool call start marker
        if (patterns.function_opener.length() > 0 &&
            patterns.tool_call_opener.length() > patterns.function_opener.length()) {
            size_t opener_start = patterns.tool_call_opener.length() - patterns.function_opener.length();
            if (opener_start > 0) {
                std::string before_func    = patterns.tool_call_opener.substr(0, opener_start);
                size_t      last_bracket   = before_func.find_last_of('[');
                size_t      tool_obj_brace = std::string::npos;
                if (last_bracket != std::string::npos && last_bracket + 1 < before_func.length()) {
                    tool_obj_brace = before_func.find('{', last_bracket + 1);
                }

                if (tool_obj_brace != std::string::npos) {
                    patterns.tool_call_start_marker = before_func.substr(0, tool_obj_brace);
                } else if (last_bracket != std::string::npos) {
                    patterns.tool_call_start_marker = before_func.substr(0, last_bracket + 1);
                } else {
                    patterns.tool_call_start_marker = before_func;
                }
            }
        } else if (patterns.tool_call_start_marker.empty()) {
            // Only search if not already set (e.g., by >>> prefix detection)
            patterns.tool_call_start_marker = find_tool_call_start(tool1_diff);
        }

        if (patterns.tool_call_opener.empty()) {
            patterns.tool_call_opener = infer_tool_call_opener(tool1_diff, tool2_diff, tool3_diff);
            if (func1_pos != std::string::npos && patterns.tool_call_opener.length() > func1_pos) {
                patterns.tool_call_opener = patterns.tool_call_opener.substr(0, func1_pos);
            }
        }
        if (patterns.tool_call_closer.empty()) {
            patterns.tool_call_closer = infer_tool_call_closer(tool1_diff, tool2_diff, tool3_diff);
        }

        patterns.tool_call_end_marker = find_tool_call_end(func_context, func1_pos);

        if (!patterns.tool_call_end_marker.empty() && patterns.tool_call_end_marker.length() > 1) {
            size_t eos_pos = patterns.tool_call_end_marker.find("<|");
            if (eos_pos == 1) {
                // Check if there's a bracket/brace before the token
                char first_char = patterns.tool_call_end_marker[0];
                if (first_char == ']' || first_char == '}') {
                    // Check if this is an actual EOS token (contains "eot_id" or "eos")
                    std::string token_content = patterns.tool_call_end_marker.substr(eos_pos);
                    if (token_content.find("eot_id") != std::string::npos ||
                        token_content.find("eos") != std::string::npos) {
                        // This is an EOS token, strip it
                        patterns.tool_call_end_marker = patterns.tool_call_end_marker.substr(0, 1);
                    }
                }
            }
        }

        // Trim whitespace
        if (!patterns.tool_call_end_marker.empty()) {
            size_t first = patterns.tool_call_end_marker.find_first_not_of(" \n\t");
            size_t last  = patterns.tool_call_end_marker.find_last_not_of(" \n\t");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_end_marker = patterns.tool_call_end_marker.substr(first, (last - first + 1));
            }
        }

        // If tool_call_end_marker matches function_closer, it found the wrong tag.
        // Use tool_call_closer instead which is derived from common suffix of diffs.
        if (!patterns.function_closer.empty() && patterns.tool_call_end_marker == patterns.function_closer) {
            if (!patterns.tool_call_closer.empty()) {
                // Try to extract a proper closing tag from tool_call_closer
                // Use rfind to get the LAST closing tag (e.g.,  not </function>)
                size_t close_start = patterns.tool_call_closer.rfind("</");
                if (close_start != std::string::npos) {
                    size_t close_end = patterns.tool_call_closer.find('>', close_start);
                    if (close_end != std::string::npos) {
                        patterns.tool_call_end_marker =
                            patterns.tool_call_closer.substr(close_start, close_end - close_start + 1);
                    }
                }
            }
        } else if (patterns.tool_call_end_marker == ">" && !patterns.tool_call_closer.empty() &&
                   patterns.tool_call_closer.length() > 3) {
            // If the specific end marker is just ">", but the common suffix (tool_call_closer) is substantial (e.g. <|tool_calls_section_end|>)
            // then prefer the common suffix, as finding ">" might just be hitting the end of the last function call
            if (patterns.tool_call_closer.find(patterns.tool_call_end_marker) != std::string::npos) {
                patterns.tool_call_end_marker = patterns.tool_call_closer;
            }
        }

        if (patterns.tool_call_start_marker.empty()) {
            std::vector<std::string> diffs  = { tool1_diff, tool2_diff, tool3_diff };
            patterns.tool_call_start_marker = find_common_substring_limited(diffs, 20, " \n\t<[{");
        }

        // Truncate if needed, but skip if func_pos is 0 (marker found via full output)
        if (func1_pos != std::string::npos && func1_pos > 0 && patterns.tool_call_start_marker.length() > func1_pos) {
            std::string candidate   = patterns.tool_call_start_marker.substr(0, func1_pos);
            size_t      last_opener = candidate.find_last_of("{[");
            if (last_opener != std::string::npos) {
                patterns.tool_call_start_marker = candidate.substr(0, last_opener);
            } else {
                patterns.tool_call_start_marker = candidate;
            }
        }

        // Ensure we don't truncate in the middle of <|...|> tokens
        patterns.tool_call_start_marker = adjust_to_token_boundary(patterns.tool_call_start_marker);
        patterns.tool_call_end_marker   = adjust_to_token_boundary(patterns.tool_call_end_marker);

        // Final trim
        if (!patterns.tool_call_start_marker.empty()) {
            size_t first = patterns.tool_call_start_marker.find_first_not_of(" \n\t\r");
            size_t last  = patterns.tool_call_start_marker.find_last_not_of(" \n\t\r");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_start_marker = patterns.tool_call_start_marker.substr(first, (last - first + 1));
            }
        }
    }

    return patterns;
}

internal_tool_format determine_format_from_patterns(const internal_discovered_pattern & patterns) {
    LOG_DBG("%s\n", __func__);

    if (patterns.tool_call_opener.empty() && patterns.tool_call_closer.empty() && patterns.function_opener.empty() &&
        patterns.function_closer.empty() && patterns.parameter_opener.empty() && patterns.parameter_closer.empty() &&
        patterns.argument_separator.empty() && patterns.tool_call_start_marker.empty() &&
        patterns.tool_call_end_marker.empty()) {
        LOG_DBG("All patterns are empty - template doesn't support tool calls\n");
        return FORMAT_UNKNOWN;
    }

    // Check for markdown code block format (Cohere Command-R Plus)
    // STRUCTURAL PATTERN: Action:\n```json\n[...]\n```
    // Key indicators:
    // 1. tool_call_start_marker contains "Action:" or similar plain text marker
    // 2. function_name_suffix or tool_call_closer contains "```" (markdown code fence)
    // 3. tool_call_opener starts with "[" indicating JSON array
    bool has_code_fence = false;
    if (!patterns.function_name_suffix.empty() && patterns.function_name_suffix.find("```") != std::string::npos) {
        has_code_fence = true;
    }
    if (!patterns.tool_call_closer.empty() && patterns.tool_call_closer.find("```") != std::string::npos) {
        has_code_fence = true;
    }
    bool has_action_marker = false;
    if (!patterns.tool_call_start_marker.empty()) {
        std::string marker_lower = patterns.tool_call_start_marker;
        std::transform(marker_lower.begin(), marker_lower.end(), marker_lower.begin(), ::tolower);
        if (marker_lower.find("action") != std::string::npos) {
            has_action_marker = true;
        }
    }
    if (has_code_fence && has_action_marker) {
        LOG_DBG("Detected MARKDOWN_CODE_BLOCK format (Action: + ```json code fence)\n");
        return FORMAT_MARKDOWN_CODE_BLOCK;
    }

    // Check for recipient-based routing format (e.g., Functionary v3.2)
    // STRUCTURAL PATTERN: The same marker is used for both content routing and tool routing
    // Key indicators:
    // 1. tool_call_start_marker == function_opener (same marker used for both)
    // 2. No parameter markers (arguments are plain dict/JSON, not wrapped in tags)
    // 3. No XML-style tags (differentiates from FUNC_TAG_WITH_NAME)
    // 4. function_opener doesn't start with structural chars like {, [, < (differentiates from other formats)
    if (!patterns.tool_call_start_marker.empty() && !patterns.function_opener.empty() &&
        patterns.tool_call_start_marker == patterns.function_opener) {
        // Check this isn't an XML-tagged format (opener would start with '<')
        if (patterns.function_opener[0] != '<' && patterns.function_opener[0] != '{' &&
            patterns.function_opener[0] != '[') {
            // Check there are no parameter markers
            if (patterns.parameter_opener.empty() && patterns.parameter_closer.empty()) {
                LOG_DBG("Detected RECIPIENT_BASED format (tool_call_start_marker == function_opener = '%s')\n",
                        patterns.tool_call_start_marker.c_str());
                return FORMAT_RECIPIENT_BASED;
            }
        }
    }

    if (!patterns.tool_call_opener.empty()) {
        if (patterns.tool_call_opener.find("{\"name\":") != std::string::npos ||
            patterns.tool_call_opener.find("{&quot;name&quot;:") != std::string::npos) {
            LOG_DBG("Detected JSON_NATIVE format from tool_call_opener JSON structure\n");
            return FORMAT_JSON_NATIVE;
        }
    }

    if (!patterns.function_opener.empty() && patterns.function_opener.find('<') == 0) {
        bool has_substantial_param_markers = false;
        if (!patterns.parameter_opener.empty()) {
            has_substantial_param_markers = (count_non_whitespace(patterns.parameter_opener) > 1);
        }
        if (!has_substantial_param_markers && !patterns.parameter_closer.empty()) {
            has_substantial_param_markers = (count_non_whitespace(patterns.parameter_closer) > 1);
        }

        if (!has_substantial_param_markers) {
            if ((!patterns.tool_call_opener.empty() && (patterns.tool_call_opener.find('[') != std::string::npos ||
                                                        patterns.tool_call_opener.find('{') != std::string::npos)) ||
                (!patterns.tool_call_start_marker.empty() &&
                 (patterns.tool_call_start_marker.find('[') != std::string::npos ||
                  patterns.tool_call_start_marker.find('{') != std::string::npos))) {
                LOG_DBG("Detected JSON_NATIVE format (XML markers but JSON structure)\n");
                return FORMAT_JSON_NATIVE;
            }
        }

        LOG_DBG("Detected XML_CONSTRUCTED format from function_opener\n");
        return FORMAT_XML_CONSTRUCTED;
    }

    if (!patterns.function_opener.empty() && patterns.function_opener.find('{') == 0) {
        LOG_DBG("Detected JSON_NATIVE format from function_opener\n");
        return FORMAT_JSON_NATIVE;
    }

    // Check for bracket-tag format: [TOOL_CALLS]name[CALL_ID]id[ARGS]{...}
    // Detected when function_name_suffix contains bracket tags like [CALL_ID]...[ARGS]
    if (!patterns.function_name_suffix.empty() && patterns.function_name_suffix.find('[') != std::string::npos &&
        patterns.function_name_suffix.find(']') != std::string::npos) {
        LOG_DBG("Detected BRACKET_TAG format from function_name_suffix containing bracket tags\n");
        return FORMAT_BRACKET_TAG;
    }

    if (!patterns.tool_call_start_marker.empty() &&
        (patterns.tool_call_start_marker.find('<') == 0 || patterns.tool_call_start_marker.find('[') == 0)) {
        bool is_prefix_marker =
            patterns.tool_call_start_marker.find("<|") == 0 || patterns.tool_call_start_marker.find("[|") == 0;
        // Check for bracket-tag format: [TAG] style without | (e.g., [TOOL_CALLS])
        bool is_bracket_tag = patterns.tool_call_start_marker.find('[') == 0 &&
                              patterns.tool_call_start_marker.find("[|") != 0 &&
                              patterns.tool_call_start_marker.find(']') != std::string::npos;
        if (is_bracket_tag) {
            LOG_DBG("Detected BRACKET_TAG format from tool_call_start_marker\n");
            return FORMAT_BRACKET_TAG;
        }
        if (is_prefix_marker) {
            LOG_DBG("Detected JSON_NATIVE format from tool_call_start_marker (instruction-based)\n");
            return FORMAT_JSON_NATIVE;
        }

        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_start_marker\n");
        return FORMAT_XML_CONSTRUCTED;
    }

    if (!patterns.tool_call_start_marker.empty() && patterns.tool_call_start_marker.find('{') == 0) {
        LOG_DBG("Detected JSON_NATIVE format from tool_call_start_marker\n");
        return FORMAT_JSON_NATIVE;
    }

    if (!patterns.tool_call_end_marker.empty() && patterns.tool_call_end_marker.find('>') == 0) {
        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_end_marker\n");
        return FORMAT_XML_CONSTRUCTED;
    }

    if (!patterns.tool_call_end_marker.empty() && patterns.tool_call_end_marker.find('}') == 0) {
        LOG_DBG("Detected JSON_NATIVE format from tool_call_end_marker\n");
        return FORMAT_JSON_NATIVE;
    }

    LOG_DBG("Format could not be determined from patterns\n");
    return FORMAT_UNKNOWN;
}

internal_discovered_pattern analyze_by_differential(const common_chat_template & tmpl) {
    internal_discovered_pattern patterns;

    try {
        LOG_DBG("%s\n", __func__);

        auto caps                      = tmpl.original_caps();
        bool minja_supports_tool_calls = caps.supports_tool_calls;
        if (!minja_supports_tool_calls) {
            LOG_DBG("Template doesn't support standard tool calls (per minja caps detection)\n");
        }

        // Define tools for testing
        json tools = {
            { { "type", "function" },
             { "function",
                { { "name", "test_function_name" },
                  { "description", "A test function" },
                  { "parameters",
                    { { "type", "object" },
                      { "properties",
                        { { "param1", { { "type", "string" }, { "description", "First parameter" } } },
                          { "param2", { { "type", "string" }, { "description", "Second parameter" } } } } },
                      { "required", json::array({ "param1", "param2" }) } } } } } },
            { { "type", "function" },
             { "function",
                { { "name", "another_test_function" },
                  { "description", "Another test function" },
                  { "parameters",
                    { { "type", "object" },
                      { "properties",
                        { { "param1", { { "type", "string" }, { "description", "First parameter" } } } } },
                      { "required", json::array({ "param1" }) } } } } }           }
        };

        // Test payload 1: Tool definitions + user + assistant with content only (no tool calls)
        json user_msg = {
            { "role",    "user"                        },
            { "content", "Please help me with a task." }
        };

        json assistant_content_only = {
            { "role",    "assistant"                                },
            { "content", "I'll help you with that task right away." }
        };

        // Test payload 2: Tool definitions + user + assistant with content + tool calls
        json assistant_content_with_tool = {
            { "role",       "assistant"                                                                              },
            { "content",    "I'll help you with that task right away."                                               },
            { "tool_calls",
             json::array(
                  { { { "id", "call_0001" },
                      { "type", "function" },
                      { "function",
                        { { "name", "test_function_name" },
                          { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
        };

        // Also test with content = null + tool calls (some templates check for this)
        json assistant_null_content_with_tool = {
            { "role",       "assistant"                                                                              },
            { "content",    nullptr                                                                                  },
            { "tool_calls",
             json::array(
                  { { { "id", "call_0001" },
                      { "type", "function" },
                      { "function",
                        { { "name", "test_function_name" },
                          { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
        };

        struct templates_params inputs;
        inputs.tools = tools;
        inputs.add_generation_prompt = false;

        // Helper function to safely render template, handling null content issues
        auto safe_render = [&](const json & messages) -> std::string {
            try {
                // First try with the original messages
                inputs.messages = messages;
                return common_chat_template_direct_apply(tmpl, inputs);
            } catch (const std::exception & e) {
                // If it fails, try replacing null content with empty string
                json fixed_messages = messages;
                for (auto & msg : fixed_messages) {
                    if (msg.contains("content") && msg["content"].is_null()) {
                        msg["content"] = "";
                    }
                }
                inputs.messages = fixed_messages;
                try {
                    return common_chat_template_direct_apply(tmpl, inputs);
                } catch (...) {
                    return "";
                }
            }
        };

        // Render payload 1: content only
        std::string output_content_only = safe_render({ user_msg, assistant_content_only });

        // Render payload 2: content + tool calls
        std::string output_content_with_tool = safe_render({ user_msg, assistant_content_with_tool });

        // Render payload 3: null content + tool calls
        std::string output_null_content_with_tool = safe_render({ user_msg, assistant_null_content_with_tool });

        LOG_DBG("Output 1 (content only): %s\n", output_content_only.c_str());
        LOG_DBG("Output 2 (content + tools): %s\n", output_content_with_tool.c_str());
        LOG_DBG("Output 3 (null + tools): %s\n", output_null_content_with_tool.c_str());

        // Check if the template renders tool calls in any scenario
        // Test 1: content vs content+tool_calls (for templates that render both)
        // Test 2: content vs null+tool_calls (for templates that only render tools when content is null)
        bool renders_tool_calls_with_content    = (output_content_only != output_content_with_tool);
        bool renders_tool_calls_without_content = (output_content_only != output_null_content_with_tool);

        if (!renders_tool_calls_with_content && !renders_tool_calls_without_content) {
            LOG_DBG("Template does NOT render tool calls in any scenario\n");
            // Return empty patterns to indicate no tool support
            return patterns;
        }

        LOG_DBG("Template renders tool calls, proceeding with differential analysis\n");

        // If we get here, the template does support tool calls
        // Use the original differential analysis approach but now we know it's valid
        json base_msg = {
            { "role",    "assistant" },
            { "content", "MARKER"    }
        };

        // Use nullptr for content to trigger tool_calls branch in templates that check "content is none"
        // Include "id" field as some templates (e.g., Mistral Nemo) require it
        json tool_msg1 = {
            { "role",       "assistant"                                                                          },
            { "content",    nullptr                                                                              },
            { "tool_calls",
             json::array(
                  { { { "id", "call_0001" },
                      { "type", "function" },
                      { "function", { { "name", "test_function_name" }, { "arguments", json::object() } } } } }) }
        };

        json tool_msg2 = {
            { "role",       "assistant"                                                                              },
            { "content",    nullptr                                                                                  },
            { "tool_calls",
             json::array(
                  { { { "id", "call_0001" },
                      { "type", "function" },
                      { "function",
                        { { "name", "test_function_name" },
                          { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
        };

        json tool_msg3 = {
            { "role",       "assistant"                                                                             },
            { "content",    nullptr                                                                                 },
            { "tool_calls",
             json::array(
                  { { { "id", "call_0001" },
                      { "type", "function" },
                      { "function", { { "name", "test_function_name" }, { "arguments", json::object() } } } },
                    { { "id", "call_0002" },
                      { "type", "function" },
                      { "function", { { "name", "another_test_function" }, { "arguments", json::object() } } } } }) }
        };

        inputs.messages  = { user_msg, base_msg };
        auto base_output = safe_render({ user_msg, base_msg });

        inputs.messages   = { user_msg, tool_msg1 };
        auto tool1_output = safe_render({ user_msg, tool_msg1 });

        // Detect if template renders null content as "None" (Python/Jinja string representation)
        // This happens when templates concatenate content without null checks, e.g.:
        //   {{ '<|im_start|>' + message.role + '\n' + content }}
        // Check if "None" appears in the tool output where it shouldn't
        if (tool1_output.find("None") != std::string::npos) {
            // Verify this is actually from null content by checking if it goes away with empty string
            json tool_msg1_empty_content       = tool_msg1;
            tool_msg1_empty_content["content"] = "";
            auto tool1_output_empty            = safe_render({ user_msg, tool_msg1_empty_content });
            if (tool1_output_empty.find("None") == std::string::npos) {
                LOG_DBG("Template renders null content as 'None', switching to empty string\n");
                patterns.requires_nonnull_content = true;
                tool1_output                      = tool1_output_empty;

                // Update tool messages to use empty string instead of null
                tool_msg1["content"] = "";
                tool_msg2["content"] = "";
                tool_msg3["content"] = "";
            }
        }

        inputs.messages   = { user_msg, tool_msg2 };
        auto tool2_output = safe_render({ user_msg, tool_msg2 });

        inputs.messages   = { user_msg, tool_msg3 };
        auto tool3_output = safe_render({ user_msg, tool_msg3 });

        std::string tool1_diff = find_string_difference(base_output, tool1_output);
        std::string tool2_diff = find_string_difference(base_output, tool2_output);
        std::string tool3_diff = find_string_difference(base_output, tool3_output);

        LOG_DBG("Tool1 diff length: %zu\n", tool1_diff.length());
        LOG_DBG("Tool2 diff length: %zu\n", tool2_diff.length());
        LOG_DBG("Tool3 diff length: %zu\n", tool3_diff.length());

        if (tool1_diff.empty() && tool2_diff.empty() && tool3_diff.empty()) {
            LOG_DBG("All diffs are empty - trying without add_generation_prompt\n");
            // Try with add_generation_prompt variations
            json alternative_base_msg = {
                { "role",    "assistant" },
                { "content", "MARKER"    }
            };

            templates_params alt_inputs;
            alt_inputs.tools                 = tools;
            alt_inputs.messages              = { user_msg, alternative_base_msg };
            alt_inputs.add_generation_prompt = false;
            auto alt_base                    = common_chat_template_direct_apply(tmpl, alt_inputs);

            alt_inputs.messages = { user_msg, tool_msg1 };
            auto alt_tool1      = common_chat_template_direct_apply(tmpl, alt_inputs);

            tool1_diff = find_string_difference(alt_base, alt_tool1);
            if (!tool1_diff.empty()) {
                // If we found a diff using the alternative approach, we must use the corresponding
                // full output for pattern extraction (otherwise diff indices will be invalid)
                tool1_output = alt_tool1;

                alt_inputs.messages = { user_msg, tool_msg2 };
                tool2_diff          = find_string_difference(alt_base, common_chat_template_direct_apply(tmpl, inputs));
                alt_inputs.messages = { user_msg, tool_msg3 };
                tool3_diff          = find_string_difference(alt_base, common_chat_template_direct_apply(tmpl, inputs));
            }
        }

        patterns = extract_patterns_from_differences(tool1_diff, tool2_diff, tool3_diff, tool1_output);

        LOG_DBG("=== ENDING TEMPLATE DIFFERENTIAL ANALYSIS ===\n");

    } catch (const std::exception & e) {
        LOG_DBG("Template differential analysis failed: %s\n", e.what());
    }

    return patterns;
}
