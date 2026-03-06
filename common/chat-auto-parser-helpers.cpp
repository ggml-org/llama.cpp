#include "chat-auto-parser-helpers.h"

#include "chat-auto-parser.h"
#include "chat.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <cctype>
#include <numeric>

using json = nlohmann::ordered_json;

std::string trim_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    if (start == str.length()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    return str.substr(start, end - start + 1);
}

std::string trim_leading_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    return str.substr(start);
}

std::string trim_trailing_whitespace(const std::string & str) {
    if (str.empty()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    // If first char is also whitespace, return empty string
    if (end == 0 && std::isspace(static_cast<unsigned char>(str[0]))) {
        return "";
    }

    return str.substr(0, end + 1);
}

std::string trim_trailing_newlines(const std::string & str) {
    size_t end = str.length();
    while (end > 0 && str[end - 1] == '\n') {
        end--;
    }

    return str.substr(0, end);
}

static size_t common_prefix_len(const std::string & left, const std::string & right) {
    size_t prefix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (prefix_len < min_len && left[prefix_len] == right[prefix_len]) {
        prefix_len++;
    }
    return prefix_len;
}

static size_t common_suffix_len(const std::string & left, const std::string & right) {
    size_t suffix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (suffix_len < min_len && left[left.length() - 1 - suffix_len] == right[right.length() - 1 - suffix_len]) {
        suffix_len++;
    }
    return suffix_len;
}

diff_split calculate_diff_split(const std::string & left, const std::string & right) {
    diff_split result;

    auto left_seg = segmentize_markers(left);
    auto right_seg = segmentize_markers(right);

    if (left_seg.empty()) {
        result.right = right;
        return result;
    }
    if (right_seg.empty()) {
        result.left = left;
        return result;
    }

    auto left_start = left_seg.begin();
    auto left_end = --left_seg.end();
    auto right_start = right_seg.begin();
    auto right_end = --right_seg.end();

    auto test = [&] () {
        return left_start != left_end && right_start != right_end;
    };

    bool left_fully_consumed = false;
    bool right_fully_consumed = false;

    while (test()) {
        bool advanced = false;
        if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_start++;
            advanced = true;
        }
        if (*left_end == *right_end) {
            result.suffix = left_end->value + result.suffix;
            if (left_start != left_end) {
                left_end--;
            } else {
                left_fully_consumed = true;
            }
            if (right_start != right_end) {
                right_end--;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if (!advanced) {
            break;
        }
    }

    if (left_start == left_end && right_start != right_end) {
        if (*left_start == *right_end) {
            result.suffix = right_end->value + result.suffix;
            right_end--;
            left_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(right_start->value);
            right_start++;
            left_fully_consumed = true;
        }
    } else if (right_start == right_end && left_start != left_end) {
        if (*left_end == *right_start) {
            result.suffix = left_end->value + result.suffix;
            left_end--;
            right_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_fully_consumed = true;
        }
    } else if (left_start == left_end && right_start == right_end && *left_start == *right_start && left_start->type == segment_type::MARKER) {
        result.prefix.append(right_start->value);
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    auto eat_segment = [](std::string & str, segment & seg) -> std::string { return str.append(seg.value); };

    bool can_have_text_suffix = left_end->type == segment_type::TEXT && right_end->type == segment_type::TEXT;
    bool can_have_text_prefix = right_start->type == segment_type::TEXT && left_start->type == segment_type::TEXT;

    std::string remainder_left = std::accumulate(left_start, left_fully_consumed ? left_end : ++left_end, std::string(), eat_segment);
    std::string remainder_right = std::accumulate(right_start, right_fully_consumed ? right_end : ++right_end, std::string(), eat_segment);

    size_t suffix_len = can_have_text_suffix ? common_suffix_len(remainder_left, remainder_right) : 0;
    // avoid overlaps between prefix and suffix
    size_t prefix_len = can_have_text_prefix ? common_prefix_len(remainder_left.substr(0, remainder_left.size() - suffix_len),
        remainder_right.substr(0, remainder_right.size() - suffix_len)) : 0;

    result.prefix.append(remainder_left.substr(0, prefix_len));
    result.suffix = remainder_left.substr(remainder_left.length() - suffix_len, suffix_len) + result.suffix;
    result.left = remainder_left.substr(prefix_len, remainder_left.length() - prefix_len - suffix_len);
    result.right = remainder_right.substr(prefix_len, remainder_right.length() - prefix_len - suffix_len);

    if (result.left == "" && result.right == "") {
        // degenerate case, no diff
        result.prefix = left;
        result.suffix = "";
        // pick prefix = all as representation
    }
    return result;
}

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common prefix of left and right
    size_t common_prefix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_prefix_len < min_len && left[common_prefix_len] == right[common_prefix_len]) {
        common_prefix_len++;
    }

    // If there's no common prefix, return empty string
    if (common_prefix_len == 0) {
        return "";
    }

    // Find the common prefix in the full string
    std::string common_prefix = left.substr(0, common_prefix_len);
    size_t      pos           = full.find(common_prefix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

    // Return everything before the common prefix
    return full.substr(0, pos);
}

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common suffix of left and right (compare from the end)
    size_t common_suffix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_suffix_len < min_len &&
           left[left.length() - 1 - common_suffix_len] == right[right.length() - 1 - common_suffix_len]) {
        common_suffix_len++;
    }

    // If there's no common suffix, return empty string
    if (common_suffix_len == 0) {
        return "";
    }

    // Extract the common suffix
    std::string common_suffix = left.substr(left.length() - common_suffix_len);

    // Find the last occurrence of the common suffix in the full string
    size_t pos = full.rfind(common_suffix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

    // Return everything after the common suffix
    return full.substr(pos + common_suffix_len);
}

// TODO: segmentize will treat a JSON array inside tags as a tag: <calls>[{ "fun": { ... } }]</calls> will be three markers
// not too worried about that because it hasn't turned out as a problem anywhere, but noting here in case it will
// Might have to put some restrictions on tag contents as well (like "no { }")
std::vector<segment> segmentize_markers(const std::string & text) {
    std::vector<segment> retval;
    bool in_marker = false;
    char marker_opener = '\0';

    auto is_marker_opener = [](char c) -> bool { return c == '<' || c == '['; };
    auto is_marker_closer = [](char op, char c) -> bool { return (op == '<' && c == '>') || (op == '[' && c == ']'); };

    size_t last_border = 0;

    for (size_t cur_pos = 0; cur_pos < text.length(); cur_pos++) {
        if (!in_marker && is_marker_opener(text[cur_pos])) {
            if (last_border < cur_pos) {
                retval.push_back(segment(segment_type::TEXT, text.substr(last_border, cur_pos - last_border)));
            }
            last_border = cur_pos;
            in_marker = true;
            marker_opener = text[cur_pos];
        } else if (in_marker && is_marker_closer(marker_opener, text[cur_pos])) {
            // no need to check because last_border will always be smaller
                retval.push_back(segment(segment_type::MARKER, text.substr(last_border, cur_pos - last_border + 1)));
            last_border = cur_pos + 1;
            in_marker = false;
            marker_opener = '\0';
        }
    }
    if (last_border < text.length()) {
            retval.push_back(segment(segment_type::TEXT, text.substr(last_border)));
    }
    return retval;
}

std::vector<segment> prune_whitespace_segments(const std::vector<segment> & segments) {
    std::vector<segment> result;
    for (const auto & seg : segments) {
        if (!trim_whitespace(seg.value).empty()) {
            result.push_back(seg);
        }
    }
    return result;
}

namespace autoparser {

std::string apply_template(const common_chat_template & tmpl, const template_params & params) {
    templates_params tmpl_params;
    tmpl_params.messages              = params.messages;
    tmpl_params.tools                 = params.tools;
    tmpl_params.add_generation_prompt = params.add_generation_prompt;
    tmpl_params.enable_thinking       = params.enable_thinking;

    if (params.extra_context) {
        tmpl_params.extra_context = *params.extra_context;
    }
    tmpl_params.extra_context["enable_thinking"] = params.enable_thinking;

    try {
        return common_chat_template_direct_apply(tmpl, tmpl_params);
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return "";
    }
}

std::optional<compare_variants_result> compare_variants(
    const common_chat_template &                   tmpl,
    const template_params &                        params_A,
    const std::function<void(template_params &)> & params_modifier) {
    // Create variant B by copying A
    template_params params_B = params_A;

    // Apply modifier to create variant B
    if (params_modifier) {
        params_modifier(params_B);
    }

    // Apply template to both variants
    std::string output_A = apply_template(tmpl, params_A);
    std::string output_B = apply_template(tmpl, params_B);

    // Check for template application failures
    if (output_A.empty() || output_B.empty()) {
        return std::nullopt;
    }

    // Calculate diff and return result with both outputs
    compare_variants_result result;
    result.diff     = calculate_diff_split(output_A, output_B);
    result.output_A = output_A;
    result.output_B = output_B;

    return result;
}

}  // namespace autoparser

