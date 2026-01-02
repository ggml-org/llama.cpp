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

// ============================================================================
// String Manipulation Helpers
// ============================================================================

void trim_whitespace(std::string & str);
void trim_trailing_newlines(std::string & str);
size_t count_non_whitespace(const std::string & str);
size_t find_last_of_any(const std::string & str, const std::string & chars, size_t start_pos);

// ============================================================================
// Tag Extraction Helpers
// ============================================================================

std::string extract_tag_name(const std::string & tag);
std::string create_closing_tag(const std::string & opening_tag);

// ============================================================================
// Common String Helpers
// ============================================================================

std::string find_common_prefix(const std::vector<std::string> & strings);
std::string find_common_suffix_generic(const std::vector<std::string> & strings);
std::string find_common_substring_limited(const std::vector<std::string> & strings,
                                           size_t max_length, const std::string & delimiters);

// ============================================================================
// Additional Helper Functions
// ============================================================================

bool string_ends_with(const std::string & str, const std::string & suffix);
std::string apply_template(const minja::chat_template &    tmpl,
                           const struct templates_params & inputs,
                           const std::optional<json> &     messages_override  = std::nullopt,
                           const std::optional<json> &     tools_override     = std::nullopt,
                           const std::optional<json> &     additional_context = std::nullopt);

// ============================================================================
// Special Token Boundary Detection (<|...|> tokens)
// ============================================================================

// Adjust a marker string to ensure it ends at a complete <|...|> token boundary
// This prevents truncation mid-token
std::string adjust_to_token_boundary(const std::string & str);

// ============================================================================
// Template Pattern Analysis Helpers
// ============================================================================

// Internal structure for differential analysis (used during pattern extraction)
struct InternalDiscoveredPattern {
    std::string tool_call_opener;
    std::string tool_call_closer;
    std::string function_opener;
    std::string function_closer;
    std::string function_name_suffix;
    std::string parameter_opener;
    std::string parameter_closer;
    std::string argument_separator;
    std::string parameter_key_prefix;
    std::string parameter_key_suffix;
    std::string tool_call_start_marker;
    std::string tool_call_end_marker;
    std::string reasoning_start_marker;
    std::string reasoning_end_marker;
    std::string content_start_marker;
    std::string content_end_marker;
    std::string tool_name_field  = "name";
    std::string tool_args_field  = "arguments";
    std::string tool_id_field;
    // Flag: template renders null content as "None" string, requires empty string instead
    bool requires_nonnull_content = false;
};

// Internal enum for format classification
enum InternalToolFormat {
    FORMAT_JSON_NATIVE,
    FORMAT_XML_CONSTRUCTED,
    FORMAT_BRACKET_TAG,   // [TOOL_CALLS]name[CALL_ID]id[ARGS]{...} (Mistral Small 3.2)
    FORMAT_CONTENT_ONLY,
    FORMAT_UNKNOWN
};

// Find the suffix that differentiates an extended string from a base string
std::string find_string_difference(const std::string & base, const std::string & extended);

// Extract JSON field name from an opener string
std::string extract_json_field_name(const std::string & opener, const std::string & default_name,
                                     const std::vector<std::string> & candidates);

// Find a closing pattern in a string starting from a given position
std::string find_closing_pattern(const std::string & diff, size_t func_pos);

// Find the tool call start marker in a difference string
std::string find_tool_call_start(const std::string & diff);

// Find the tool call end marker in a difference string
std::string find_tool_call_end(const std::string & diff, size_t func_pos);

// Infer the tool call opener from multiple difference strings
std::string infer_tool_call_opener(const std::string & diff1, const std::string & diff2,
                                    const std::string & diff3);

// Infer the tool call closer from multiple difference strings
std::string infer_tool_call_closer(const std::string & diff1, const std::string & diff2,
                                    const std::string & diff3);

// Extract patterns from differences between tool calls
InternalDiscoveredPattern extract_patterns_from_differences(const std::string & tool1_diff,
                                                              const std::string & tool2_diff,
                                                              const std::string & tool3_diff,
                                                              const std::string & tool1_full = "");

// Determine the format classification from discovered patterns
InternalToolFormat determine_format_from_patterns(const InternalDiscoveredPattern & patterns);

// Analyze template using differential analysis (internal use)
InternalDiscoveredPattern analyze_by_differential(const minja::chat_template & tmpl);