#pragma once

#include "chat.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Forward declaration of minja::chat_template
namespace minja {
struct chat_template;
}

// Structure to hold discovered patterns through differential analysis
struct DiscoveredPattern;

// ============================================================================
// String Manipulation Helpers
// ============================================================================

void trim_whitespace(std::string & str);
void trim_trailing_newlines(std::string & str);
void strip_markers(std::string & str, const std::string & prefix, const std::string & suffix);
size_t count_non_whitespace(const std::string & str);
size_t find_last_of_any(const std::string & str, const std::string & chars, size_t start_pos);

// ============================================================================
// Quote and Position Helpers
// ============================================================================

size_t find_with_quote_adjustment(const std::string & str, const std::string & target);

// ============================================================================
// Template Application Helpers
// ============================================================================

std::string apply_template(const minja::chat_template & tmpl, const json & message);
std::string apply_template(const minja::chat_template & tmpl, const json & messages, const json & tools);

// ============================================================================
// Pattern Matching Helpers
// ============================================================================

bool contains_thinking_pattern(const std::string & str, std::string & found_start, std::string & found_end);
bool has_closed_reasoning_section(const std::string & str1, const std::string & str2,
                                   std::string & found_start, std::string & found_end);

// ============================================================================
// Tag Extraction Helpers
// ============================================================================

std::string extract_tag_name(const std::string & tag);
std::string create_closing_tag(const std::string & opening_tag);
std::string extract_bracket_tag_name(const std::string & tag);

// ============================================================================
// Common String Helpers
// ============================================================================

std::string find_common_prefix(const std::vector<std::string> & strings);
std::string find_common_suffix_generic(const std::vector<std::string> & strings);
std::string find_common_substring_limited(const std::vector<std::string> & strings,
                                           size_t max_length, const std::string & delimiters);
std::string find_first_pattern(const std::string & str, const std::vector<std::string> & candidates,
                                size_t start_pos = 0);

// ============================================================================
// Token Collection Helpers
// ============================================================================

void collect_non_empty_tokens(const DiscoveredPattern & discovered, std::vector<std::string> & tokens);

// ============================================================================
// Additional Helper Functions
// ============================================================================

bool string_ends_with(const std::string & str, const std::string & suffix);
std::string apply_template(const minja::chat_template &    tmpl,
                           const struct templates_params & inputs,
                           const std::optional<json> &     messages_override  = std::nullopt,
                           const std::optional<json> &     tools_override     = std::nullopt,
                           const std::optional<json> &     additional_context = std::nullopt);
