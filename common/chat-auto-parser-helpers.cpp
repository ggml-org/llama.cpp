#include "chat-auto-parser-helpers.h"

#include "chat-diff-analyzer.h"
#include "nlohmann/json.hpp"

#include <cctype>

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

// Helper to find unmatched bracket/tag in a string
// Finds an unmatched bracket in a string.
// search_backwards=true:  finds unclosed opening bracket at end (returns bracket position)
// search_backwards=false: finds unopened closing bracket at start (returns position after bracket)
static size_t find_unmatched_bracket(const std::string & str, bool search_backwards) {
    if (str.empty()) {
        return std::string::npos;
    }

    // Compute iteration bounds and bracket types based on direction
    const char * primary_brackets = search_backwards ? "<[" : ">]";

    for (size_t i = 0; i < str.length(); ++i) {
        // Map iteration index to actual position based on direction
        size_t pos = search_backwards ? (str.length() - 1 - i) : i;
        char   c   = str[pos];

        // Check if this is a primary bracket we're looking for
        if (c == primary_brackets[0] || c == primary_brackets[1]) {
            // Get the matching bracket: < matches >, [ matches ], and vice versa
            char match_bracket = (c == '<' || c == '>') ? (c == '<' ? '>' : '<') : (c == '[' ? ']' : '[');

            // Search for matching bracket in the appropriate range
            size_t inner_start = search_backwards ? (pos + 1) : 0;
            size_t inner_end   = search_backwards ? str.length() : pos;
            bool   found_match = false;

            for (size_t j = inner_start; j < inner_end; ++j) {
                if (str[j] == match_bracket) {
                    found_match = true;
                    break;
                }
            }

            if (!found_match) {
                return search_backwards ? pos : (pos + 1);
            }
        }
    }

    return std::string::npos;
}

static size_t find_unclosed_bracket_at_end(const std::string & str) {
    return find_unmatched_bracket(str, true);
}

static size_t find_unopened_bracket_at_start(const std::string & str) {
    return find_unmatched_bracket(str, false);
}

// Returns true if `s` contains an unmatched bracket.
// search_backwards=true:  looks for opening bracket without matching closing after it
// search_backwards=false: looks for closing bracket without matching opening before it
static bool contains_unmatched_bracket(const std::string & s, char opening, char closing, bool search_backwards) {
    if (s.empty()) {
        return false;
    }

    char primary = search_backwards ? opening : closing;

    for (size_t i = 0; i < s.length(); ++i) {
        // Map iteration index to actual position based on direction
        size_t pos = search_backwards ? (s.length() - 1 - i) : i;

        if (s[pos] == primary) {
            // Search for matching bracket in the appropriate range
            size_t inner_start   = search_backwards ? (pos + 1) : 0;
            size_t inner_end     = search_backwards ? s.length() : pos;
            char   match_bracket = search_backwards ? closing : opening;
            bool   found_match   = false;

            for (size_t j = inner_start; j < inner_end; ++j) {
                if (s[j] == match_bracket) {
                    found_match = true;
                    break;
                }
            }

            if (!found_match) {
                return true;
            }
        }
    }
    return false;
}

static bool contains_unopened_closing(const std::string & s, char opening, char closing) {
    return contains_unmatched_bracket(s, opening, closing, false);
}

static bool contains_unclosed_opening(const std::string & s, char opening, char closing) {
    return contains_unmatched_bracket(s, opening, closing, true);
}

// Moves incomplete tags from prefix/suffix into left/right parts
// Only moves tags when we detect the split pattern in BOTH left and right
static diff_split fix_tag_boundaries(diff_split result) {
    // Check if prefix ends with an unclosed bracket/tag
    // No fixed window: search the entire neighboring strings for matching brackets
    size_t unclosed_pos = find_unclosed_bracket_at_end(result.prefix);
    if (unclosed_pos != std::string::npos) {
        char opening_bracket = result.prefix[unclosed_pos];
        char closing_bracket = (opening_bracket == '<') ? '>' : ']';

        // Look for the specific closing bracket that matches our opening bracket
        bool left_has_pattern   = contains_unopened_closing(result.left, opening_bracket, closing_bracket);
        bool right_has_pattern  = contains_unopened_closing(result.right, opening_bracket, closing_bracket);
        bool suffix_has_pattern = contains_unopened_closing(result.suffix, opening_bracket, closing_bracket);

        // Move the tag if both sides satisfy: has pattern OR is empty (and other has pattern)
        // This handles cases like: left="" right="_begin|>..." or left="stuff>" right="stuff>"
        bool left_satisfies  = left_has_pattern || (result.left.empty() && suffix_has_pattern);
        bool right_satisfies = right_has_pattern || (result.right.empty() && suffix_has_pattern);

        if (left_satisfies && right_satisfies) {
            // Move the unclosed tag from prefix to left/right
            std::string tag_part = result.prefix.substr(unclosed_pos);
            result.prefix        = result.prefix.substr(0, unclosed_pos);
            result.left          = tag_part + result.left;
            result.right         = tag_part + result.right;
        }
    }

    // Check if suffix starts with an unopened bracket/tag
    size_t unopened_end = find_unopened_bracket_at_start(result.suffix);
    if (unopened_end != std::string::npos) {
        char closing_bracket =
            result.suffix[unopened_end - 1];  // -1 because unopened_end is position after the bracket
        char opening_bracket = (closing_bracket == '>') ? '<' : '[';

        // Check if BOTH left and right have the pattern of unclosed opening bracket at the end
        bool left_has_pattern   = contains_unclosed_opening(result.left, opening_bracket, closing_bracket);
        bool right_has_pattern  = contains_unclosed_opening(result.right, opening_bracket, closing_bracket);
        bool prefix_has_pattern = contains_unclosed_opening(result.prefix, opening_bracket, closing_bracket);

        // Move the tag if both sides satisfy: has pattern OR is empty (and other has pattern)
        bool left_satisfies  = left_has_pattern || (result.left.empty() && prefix_has_pattern);
        bool right_satisfies = right_has_pattern || (result.right.empty() && prefix_has_pattern);

        if (left_satisfies && right_satisfies) {
            // Move the unopened tag from suffix to left/right
            std::string tag_part = result.suffix.substr(0, unopened_end);
            result.suffix        = result.suffix.substr(unopened_end);
            result.left          = result.left + tag_part;
            result.right         = result.right + tag_part;
        }
    }

    return result;
}

diff_split calculate_diff_split(const std::string & left, const std::string & right) {
    diff_split result;

    // Find longest common prefix
    size_t prefix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (prefix_len < min_len && left[prefix_len] == right[prefix_len]) {
        prefix_len++;
    }
    result.prefix = left.substr(0, prefix_len);

    // Find longest common suffix, ending no later than the end of the longest common prefix
    size_t suffix_len = 0;
    while (suffix_len < min_len - prefix_len) {
        size_t left_pos  = left.length() - 1 - suffix_len;
        size_t right_pos = right.length() - 1 - suffix_len;

        // Ensure we're not going into the prefix region
        if (left_pos < prefix_len || right_pos < prefix_len) {
            break;
        }

        if (left[left_pos] == right[right_pos]) {
            suffix_len++;
        } else {
            break;
        }
    }
    result.suffix = left.substr(left.length() - suffix_len);

    // Extract the remainders (the parts between prefix and suffix)
    result.left  = left.substr(prefix_len, left.length() - prefix_len - suffix_len);
    result.right = right.substr(prefix_len, right.length() - prefix_len - suffix_len);

    // Fix tag boundaries by moving incomplete tags to left/right
    // We iterate because:
    // 1. fix_tag_boundaries may move content from prefix/suffix to left/right
    // 2. After that, we find common suffix in left/right to extract
    // 3. The extracted suffix might contain tag parts that need fixing
    // We apply fix AFTER suffix extraction to ensure incomplete tags aren't left in suffix
    diff_split prev_result;
    do {
        prev_result = result;

        // First, find and extract any common suffix from left/right
        size_t suffix_len = 0;
        size_t min_len    = std::min(result.left.length(), result.right.length());
        while (suffix_len < min_len) {
            size_t left_pos  = result.left.length() - 1 - suffix_len;
            size_t right_pos = result.right.length() - 1 - suffix_len;
            if (result.left[left_pos] == result.right[right_pos]) {
                suffix_len++;
            } else {
                break;
            }
        }

        if (suffix_len > 0) {
            std::string common_suffix = result.left.substr(result.left.length() - suffix_len);
            result.suffix             = common_suffix + result.suffix;
            result.left               = result.left.substr(0, result.left.length() - suffix_len);
            result.right              = result.right.substr(0, result.right.length() - suffix_len);
        }

        // Then apply fix_tag_boundaries to move incomplete tags from prefix/suffix to left/right
        result = fix_tag_boundaries(result);

    } while (!(result == prev_result) && result.left != left && result.right != right);

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

