#pragma once

#include <optional>
#include <string>
#include <vector>

#include "chat.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

namespace minja {
class chat_template;
}

void   trim_whitespace(std::string & str);
void   trim_trailing_newlines(std::string & str);
size_t count_non_whitespace(const std::string & str);
size_t find_last_of_any(const std::string & str, const std::string & chars, size_t start_pos);

std::string extract_tag_name(const std::string & tag);
std::string create_closing_tag(const std::string & opening_tag);

std::string find_common_prefix(const std::vector<std::string> & strings);
std::string find_common_suffix_generic(const std::vector<std::string> & strings);
std::string find_common_substring_limited(const std::vector<std::string> & strings,
                                          size_t                           max_length,
                                          const std::string &              delimiters);

bool        string_ends_with(const std::string & str, const std::string & suffix);
std::string apply_template(common_chat_template      &    tmpl,
                           const struct templates_params & inputs,
                           const std::optional<json> &     messages_override  = std::nullopt,
                           const std::optional<json> &     tools_override     = std::nullopt,
                           const std::optional<json> &     additional_context = std::nullopt);

// Adjust a marker string to ensure it ends at a complete <|...|> token boundary
// This prevents truncation mid-token
std::string adjust_to_token_boundary(const std::string & str);

// Find the position of a token opener (<| or <｜) in a string
// Returns std::string::npos if not found
size_t find_token_opener(const std::string & str, size_t start_pos = 0);

// Find the position of a token closer (|> or ｜>) in a string
// Returns std::string::npos if not found
size_t find_token_closer(const std::string & str, size_t start_pos = 0);

// Get the length of the token opener at the given position (2 for <| or 4 for <｜)
// Returns 0 if no valid opener at position
size_t get_token_opener_length(const std::string & str, size_t pos);

// Get the length of the token closer at the given position (2 for |> or 4 for ｜>)
// Returns 0 if no valid closer at position
size_t get_token_closer_length(const std::string & str, size_t pos);

// Strip EOS/end-of-sentence tokens from the end of a string
// Handles both standard (<|eos|>, <|eot_id|>) and fullwidth (<｜end▁of▁sentence｜>) formats
std::string strip_eos_token(const std::string & str);

// Internal structure for differential analysis (used during pattern extraction)
struct internal_discovered_pattern {
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
    std::string tool_name_field = "name";
    std::string tool_args_field = "arguments";
    std::string tool_id_field;
    // For markdown code block format (Cohere Command-R Plus)
    std::string code_block_marker;    // e.g., "Action:"
    std::string code_block_language;  // e.g., "json"
    // Flag: template renders null content as "None" string, requires empty string instead
    bool        requires_nonnull_content = false;
};

// Internal enum for format classification
enum internal_tool_format {
    FORMAT_JSON_NATIVE,
    FORMAT_XML_CONSTRUCTED,
    FORMAT_BRACKET_TAG,          // [TOOL_CALLS]name[CALL_ID]id[ARGS]{...} (Mistral Small 3.2)
    FORMAT_RECIPIENT_BASED,      // >>>recipient\n{content} (Functionary v3.2)
    FORMAT_MARKDOWN_CODE_BLOCK,  // Action:\n```json\n[...]\n``` (Cohere Command-R Plus)
    FORMAT_CONTENT_ONLY,
    FORMAT_UNKNOWN
};

// Find the suffix that differentiates an extended string from a base string
std::string find_string_difference(const std::string & base, const std::string & extended);

// Extract JSON field name from an opener string
std::string extract_json_field_name(const std::string &              opener,
                                    const std::string &              default_name,
                                    const std::vector<std::string> & candidates);

// Find a closing pattern in a string starting from a given position
std::string find_closing_pattern(const std::string & diff, size_t func_pos);

// Find the tool call start marker in a difference string
std::string find_tool_call_start(const std::string & diff);

// Find the tool call end marker in a difference string
std::string find_tool_call_end(const std::string & diff, size_t func_pos);

// Infer the tool call opener from multiple difference strings
std::string infer_tool_call_opener(const std::string & diff1, const std::string & diff2, const std::string & diff3);

// Infer the tool call closer from multiple difference strings
std::string infer_tool_call_closer(const std::string & diff1, const std::string & diff2, const std::string & diff3);

// Extract patterns from differences between tool calls
internal_discovered_pattern extract_patterns_from_differences(const std::string & tool1_diff,
                                                            const std::string & tool2_diff,
                                                            const std::string & tool3_diff,
                                                            const std::string & tool1_full = "");

// Determine the format classification from discovered patterns
internal_tool_format determine_format_from_patterns(const internal_discovered_pattern & patterns);

// Analyze template using differential analysis (internal use)
internal_discovered_pattern analyze_by_differential(const common_chat_template & tmpl);
