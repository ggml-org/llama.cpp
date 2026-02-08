#pragma once

#include "chat-diff-analyzer.h"
#include <string>

std::string trim_whitespace(const std::string & str);
std::string trim_leading_whitespace(const std::string & str);
std::string trim_trailing_whitespace(const std::string & str);
std::string trim_trailing_newlines(const std::string & str);

// calculate a diff split (longest common prefix, longest common suffix excluding prefix,
// mismatched part on the left, mismatched part on the right) between two strings
diff_split calculate_diff_split(const std::string & left, const std::string & right);

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right);

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right);

// Segmentize text into markers and non-marker fragments
std::vector<segment> segmentize_markers(const std::string & text);

// Prune whitespace-only segments from a vector of segments
std::vector<segment> prune_whitespace_segments(const std::vector<segment> & segments);
