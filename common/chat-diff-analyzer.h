#pragma once

#include "chat.h"
#include "jinja/caps.h"
#include "peg-parser.h"
#include "nlohmann/json.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

class common_chat_peg_builder;

// ============================================================================
// Parameters for template application
// ============================================================================
struct template_params {
    json                messages;
    json                tools;
    bool                add_generation_prompt = false;
    bool                enable_thinking       = true;
    std::optional<json> extra_context         = std::nullopt;
};

struct diff_split {
    std::string prefix;
    std::string suffix;
    std::string left;
    std::string right;

    bool operator==(struct diff_split & other) const {
        return prefix == other.prefix && suffix == other.suffix && left == other.left && right == other.right;
    }
};

// Result of compare_variants containing diff and original outputs
struct compare_variants_result {
    diff_split  diff;
    std::string output_A;
    std::string output_B;
};

namespace autoparser {

struct templates_params;

// ============================================================================
// Analysis Result Enums
// ============================================================================

// Reasoning handling mode (derived from R1-R3 comparisons)
enum class reasoning_mode {
    NONE,           // No reasoning markers detected
    TAG_BASED,      // Standard tag-based: <think>...</think>
    DELIMITER,      // Delimiter-based: [BEGIN FINAL RESPONSE] (reasoning ends at delimiter)
    FORCED_OPEN,    // Template ends with open reasoning tag (empty start, non-empty end)
    FORCED_CLOSED,  // Template ends with open reasoning tag on enabled thinking but
                    // with both opened and closed tag for disabled thinking
    TOOLS_ONLY      // Only reason on tool calls, not on normal content
};

inline std::ostream & operator<<(std::ostream & os, const reasoning_mode & mode) {
    switch (mode) {
        case reasoning_mode::NONE:
            return os << "NONE";
        case reasoning_mode::TAG_BASED:
            return os << "TAG_BASED";
        case reasoning_mode::DELIMITER:
            return os << "DELIMITER";
        case reasoning_mode::FORCED_OPEN:
            return os << "FORCED_OPEN";
        case reasoning_mode::FORCED_CLOSED:
            return os << "FORCED_CLOSED";
        case reasoning_mode::TOOLS_ONLY:
            return os << "TOOLS_ONLY";
        default:
            return os << "UNKNOWN";
    }
}

// Content wrapping mode (derived from C1 comparison)
enum class content_mode {
    PLAIN,                   // No content markers
    ALWAYS_WRAPPED,          // Content always wrapped with markers
    WRAPPED_WITH_REASONING,  // Content wrapped only when reasoning present
};

inline std::ostream & operator<<(std::ostream & os, const content_mode & mode) {
    switch (mode) {
        case content_mode::PLAIN:
            return os << "PLAIN";
        case content_mode::ALWAYS_WRAPPED:
            return os << "ALWAYS_WRAPPED";
        case content_mode::WRAPPED_WITH_REASONING:
            return os << "WRAPPED_WITH_REASONING";
        default:
            return os << "UNKNOWN";
    }
}

// Call ID position in tool calls (for non-JSON formats)
enum class call_id_position {
    NONE,                   // No call ID support detected
    PRE_FUNC_NAME,          // Call ID before function name: [CALL_ID]id[FUNC]name{args}
    BETWEEN_FUNC_AND_ARGS,  // Call ID between function and args: [FUNC]name[CALL_ID]id{args}
    POST_ARGS,              // Call ID after arguments: [FUNC]name{args}[CALL_ID]id
};

inline std::ostream & operator<<(std::ostream & os, const call_id_position & pos) {
    switch (pos) {
        case call_id_position::NONE:
            return os << "NONE";
        case call_id_position::PRE_FUNC_NAME:
            return os << "PRE_FUNC_NAME";
        case call_id_position::BETWEEN_FUNC_AND_ARGS:
            return os << "BETWEEN_FUNC_AND_ARGS";
        case call_id_position::POST_ARGS:
            return os << "POST_ARGS";
        default:
            return os << "UNKNOWN";
    }
}

// Tool call format classification (derived from T1-T5, A1-A3 comparisons)
enum class tool_format {
    NONE,             // No tool support detected
    JSON_NATIVE,      // Pure JSON: {"name": "X", "arguments": {...}}
    TAG_WITH_JSON,    // Tag-based with JSON args: <function=X>{...}</function>
    TAG_WITH_TAGGED,  // Tag-based with tagged args: <param=key>value</param>
};

inline std::ostream & operator<<(std::ostream & os, const tool_format & format) {
    switch (format) {
        case tool_format::NONE:
            return os << "NONE";
        case tool_format::JSON_NATIVE:
            return os << "JSON_NATIVE";
        case tool_format::TAG_WITH_JSON:
            return os << "TAG_WITH_JSON";
        case tool_format::TAG_WITH_TAGGED:
            return os << "TAG_WITH_TAGGED";
        default:
            return os << "UNKNOWN";
    }
}

// ============================================================================
// Sub-structs for tool analysis
// ============================================================================

struct tool_format_analysis {
    tool_format mode = tool_format::NONE;

    std::string section_start;   // e.g., "<tool_call>", "[TOOL_CALLS]", ""
    std::string section_end;     // e.g., "</tool_call>", ""
    std::string per_call_start;  // e.g., "<|tool_call_begin|>", "" (for multi-call templates)
    std::string per_call_end;    // e.g., "<|tool_call_end|>", ""

    bool fun_name_is_key = false;       // In JSON format function name is JSON key, i.e. { "<funname>": { ... arguments ... } }
    bool tools_array_wrapped = false;   // Tool calls wrapped in JSON array [...]
    bool uses_python_dicts = false;     // Tool call args use Python dict format (single-quoted strings)

    std::string              function_field = "function";
    std::string              name_field     = "name";
    std::string              args_field     = "arguments";
    std::string              id_field;
    std::string              gen_id_field;
    std::vector<std::string> parameter_order;
};

struct tool_function_analysis {
    std::string name_prefix;  // e.g., "<function=", "\"name\": \"", "functions."
    std::string name_suffix;  // e.g., ">", "\"", ":0"
    std::string close;        // e.g., "</function>", "" (for tag-based)
};

struct tool_arguments_analysis {
    std::string start;          // e.g., "<|tool_call_argument_begin|>", "<args>"
    std::string end;            // e.g., "<|tool_call_argument_end|>", "</args>"
    std::string name_prefix;   // e.g., "<param=", "<arg_key>", "\""
    std::string name_suffix;   // e.g., ">", "</arg_key>", "\":"
    std::string value_prefix;  // e.g., "", "<arg_value>", ""
    std::string value_suffix;  // e.g., "</param>", "</arg_value>", ""
    std::string separator;     // e.g., "", "\n", ","
};

struct tool_id_analysis {
    call_id_position pos = call_id_position::NONE;

    std::string prefix;  // e.g., "[CALL_ID]" (marker before call ID value)
    std::string suffix;  // e.g., "" (marker after call ID value, before next section)
};

// ============================================================================
// Parser build context (shared interface for build_parser methods)
// ============================================================================

struct analyze_content;

struct parser_build_context {
    common_chat_peg_builder & p;
    const templates_params &          inputs;
    common_peg_parser                 reasoning_parser;
    bool                              extracting_reasoning = false;
    const analyze_content *           content              = nullptr;

    parser_build_context(common_chat_peg_builder & p, const templates_params & inputs);
};

// ============================================================================
// Base class for analyzers with parser building
// ============================================================================

struct analyze_base {
    virtual ~analyze_base() = default;
    virtual common_peg_parser build_parser(parser_build_context & ctx) const = 0;

  protected:
    const common_chat_template * tmpl = nullptr;

    analyze_base() = default;
    explicit analyze_base(const common_chat_template & tmpl) : tmpl(&tmpl) {}
};

// ============================================================================
// Reasoning analyzer
// ============================================================================

struct analyze_reasoning : analyze_base {
    reasoning_mode mode = reasoning_mode::NONE;

    std::string start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
    std::string end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"

    analyze_reasoning() = default;
    analyze_reasoning(const common_chat_template & tmpl, bool supports_tools);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Look for reasoning markers in rendered content
    void compare_reasoning_presence();

    // Compare generation prompt with enable_thinking=true vs false
    void compare_thinking_enabled();

    // Check if reasoning is always possible or only in tool calls
    void compare_reasoning_scope();
};

// ============================================================================
// Content analyzer
// ============================================================================

struct analyze_content : analyze_base {
    content_mode mode = content_mode::PLAIN;

    std::string start;  // e.g., "<response>", ">>>all\n", ""
    std::string end;    // e.g., "</response>", ""

    bool requires_nonnull_content = false;

    analyze_content() = default;
    analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

    bool is_always_wrapped() const;
    common_peg_parser build_optional_wrapped(parser_build_context & ctx) const;
};

// ============================================================================
// Tool analyzer
// ============================================================================

struct analyze_tools : analyze_base {
    tool_format_analysis    format;
    tool_function_analysis  function;
    tool_arguments_analysis arguments;
    tool_id_analysis        call_id;

    analyze_tools() = default;
    analyze_tools(const common_chat_template & tmpl,
                  const jinja::caps &          caps,
                  const analyze_reasoning &    reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Extract tool calling 'haystack' for further analysis and delegate further analysis based on format
    void analyze_tool_calls(const analyze_reasoning & reasoning);

    // Analyze format based on position of function and argument name in needle
    void analyze_tool_call_format(const std::string &       haystack,
                                  const std::string &       fun_name_needle,
                                  const std::string &       arg_name_needle,
                                  const analyze_reasoning & reasoning);

    // Analyze specifics of JSON native format (entire tool call is a JSON object)
    void analyze_tool_call_format_json_native(const std::string & clean_haystack,
                                              const std::string & fun_name_needle,
                                              const std::string & arg_name_needle);

    // Analyze specifics of non-JSON native format (tags for function name or for function name and arguments)
    void analyze_tool_call_format_non_json(const std::string & clean_haystack,
                                           const std::string & fun_name_needle);

    // Check for and extract specific per-call markers for non-native-JSON templates with parallel call support
    void check_per_call_markers();

    // Extract function name markers
    void extract_function_markers();

    // Delegates to separate functions for: separator analysis, argument name analysis, argument value analysis
    void analyze_arguments();

    // Extract argument name markers
    void extract_argument_name_markers();

    // Extract argument value markers
    void extract_argument_value_markers();

    // Extract argument separator, if specified (eg. <arg=foo>...</arg><sep><arg=bar>...</arg>)
    void extract_argument_separator();

    // Extract argument wrapper markers, if present (eg. '<args><arg=foo>...</arg><arg=bar>...</arg></args>')
    void extract_args_markers();

    // Extract call ID markers, if present
    void extract_call_id_markers();

    // Per-format tool parser builders
    common_peg_parser build_tool_parser_json_native(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_json(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_tagged(parser_build_context & ctx) const;
};

// ============================================================================
// Top-level template analyzer (merges differential_analyzer + diff_analysis_result)
// ============================================================================

struct autoparser {
    jinja::caps        jinja_caps;
    analyze_reasoning  reasoning;
    analyze_content    content;
    analyze_tools      tools;

    // Preserved tokens for tokenizer (union of all non-empty markers)
    std::vector<std::string> preserved_tokens;

    // Constructor: runs full differential analysis on a template
    explicit autoparser(const common_chat_template & tmpl);

    // Build the PEG parser for this template
    common_peg_arena build_parser(const templates_params & inputs) const;

  private:
    // Collect tokens from entire analysis to preserve
    void collect_preserved_tokens();
};

}  // namespace autoparser

enum segment_type { TEXT, MARKER };

inline std::ostream & operator<<(std::ostream & os, const segment_type & type) {
    switch (type) {
        case segment_type::TEXT:
            return os << "TEXT";
        case segment_type::MARKER:
            return os << "MARKER";
        default:
            return os << "UNKNOWN";
    }
}

struct segment {
    segment_type type;
    std::string  value;

    segment(segment_type type, std::string value) : type(type), value(std::move(value)) {}

    bool operator==(const segment & other) const {
        return type == other.type && value == other.value;
    }

    bool operator!=(const segment & other) const {
        return !(*this == other);
    }
};
