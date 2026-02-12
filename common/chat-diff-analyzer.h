#pragma once

#include "chat.h"
#include "jinja/caps.h"
#include "nlohmann/json.hpp"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

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

// ============================================================================
// Marker Registry: All markers extracted via differential analysis
// ============================================================================

// Markers extracted from differential analysis of template outputs
// Each marker is derived from a specific comparison in the analysis matrix
struct marker_registry {
    // === Reasoning markers (from Phase 1: R1-R3) ===
    std::string reasoning_start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
    std::string reasoning_end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"

    // === Content markers (from Phase 2: C1-C2) ===
    std::string content_start;  // e.g., "<response>", ">>>all\n", ""
    std::string content_end;    // e.g., "</response>", ""

    // === Tool section markers (from Phase 3: T1-T2) ===
    std::string tool_section_start;  // e.g., "<tool_call>", "[TOOL_CALLS]", ""
    std::string tool_section_end;    // e.g., "</tool_call>", ""
    std::string per_call_start;      // e.g., "<|tool_call_begin|>", "" (for multi-call templates)
    std::string per_call_end;        // e.g., "<|tool_call_end|>", ""
    std::string call_separator;      // e.g., ",", "\n", "" (between multiple calls)

    // === Function markers (from Phase 3: T3-T5) ===
    std::string func_name_prefix;  // e.g., "<function=", "\"name\": \"", "functions."
    std::string func_name_suffix;  // e.g., ">", "\"", ":0"
    std::string func_close;        // e.g., "</function>", "" (for tag-based)
    std::string args_start;        // e.g., "{", "<|tool_call_argument_begin|>"
    std::string args_end;          // e.g., "}", ""

    // === Argument markers (from Phase 4: A1-A3, for tagged args format) ===
    std::string arg_name_prefix;   // e.g., "<param=", "<arg_key>", "\""
    std::string arg_name_suffix;   // e.g., ">", "</arg_key>", "\":"
    std::string arg_value_prefix;  // e.g., "", "<arg_value>", ""
    std::string arg_value_suffix;  // e.g., "</param>", "</arg_value>", ""
    std::string arg_separator;     // e.g., "", "\n", ","

    // === Call ID markers (for non-JSON formats with tool call IDs) ===
    std::string call_id_prefix;  // e.g., "[CALL_ID]" (marker before call ID value)
    std::string call_id_suffix;  // e.g., "" (marker after call ID value, before next section)
};

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

struct reasoning_analysis {
    reasoning_mode mode = reasoning_mode::NONE;

    std::string start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
    std::string end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"
};

struct content_analysis {
    content_mode mode = content_mode::PLAIN;

    std::string start;  // e.g., "<response>", ">>>all\n", ""
    std::string end;    // e.g., "</response>", ""

    bool requires_nonnull_content = false;
};

struct tool_format_analysis {
    tool_format mode = tool_format::NONE;

    std::string section_start;   // e.g., "<tool_call>", "[TOOL_CALLS]", ""
    std::string section_end;     // e.g., "</tool_call>", ""
    std::string per_call_start;  // e.g., "<|tool_call_begin|>", "" (for multi-call templates)
    std::string per_call_end;    // e.g., "<|tool_call_end|>", ""

    bool fun_name_is_key = false;      // In JSON format function name is JSON key, i.e. { "<funname>": { ... arguments ... } }
    bool tools_array_wrapped = false;  // Tool calls wrapped in JSON array [...]

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

struct tool_analysis {
    tool_format_analysis    format;
    tool_function_analysis  function;
    tool_arguments_analysis arguments;
    tool_id_analysis        call_id;
};

// Complete result of differential analysis
struct diff_analysis_result {
    jinja::caps        jinja_caps;
    reasoning_analysis reasoning;
    content_analysis   content;
    tool_analysis      tools;

    // Preserved tokens for tokenizer (union of all non-empty markers)
    std::vector<std::string> preserved_tokens;
};

// Performs systematic differential analysis on chat templates
// Uses comparison matrix to extract markers without heuristics
class differential_analyzer {
  public:
    // Main entry point: Run full differential analysis on a template
    static diff_analysis_result analyze(const common_chat_template & tmpl);

    // Phase-specific analysis (can be called individually for testing)
    static reasoning_analysis analyze_reasoning(const common_chat_template & tmpl, bool supports_tools);
    static content_analysis   analyze_content(const common_chat_template & tmpl, const reasoning_analysis & reasoning);
    static tool_analysis      analyze_tools(const common_chat_template & tmpl,
                                            const jinja::caps &          caps,
                                            const reasoning_analysis &   reasoning);

    // Factorized differential comparison function (public for testing)
    // Takes base params and a single modifier lambda to create variant B
    // Returns compare_variants_result containing diff and both outputs, or std::nullopt on failure
    static std::optional<compare_variants_result> compare_variants(
        const common_chat_template &                   tmpl,
        const template_params &                        params_A,
        const std::function<void(template_params &)> & params_modifier);

  private:
    // Comparison helpers (implement the comparison matrix from the plan)

    // 1. Reasoning analysis:
    // Look for reasoning markers in rendered content
    static void compare_reasoning_presence(const common_chat_template & tmpl, reasoning_analysis & reasoning);

    // Compare generation prompt with enable_thinking=true vs false
    static void compare_thinking_enabled(const common_chat_template & tmpl, reasoning_analysis & reasoning);

    // Check if reasoning is always possible or only in tool calls
    static void compare_reasoning_scope(const common_chat_template & tmpl, reasoning_analysis & reasoning);

    // 2. Content (fully inside analyze_content mentioned above)

    // 3. Tool calls
    //    a. format
    // Extract tool calling 'haystack' for further analysis and delegate further analysis based on format
    static tool_format_analysis analyze_tool_calls(const common_chat_template & tmpl,
                                                   const reasoning_analysis &   reasoning);

    // Analyze format based on position of function and argument name in needle
    static tool_format_analysis analyze_tool_call_format(const std::string &        haystack,
                                                         const std::string &        fun_name_needle,
                                                         const std::string &        arg_name_needle,
                                                         const reasoning_analysis & reasoning);

    // Analyze specifics of JSON native format (entire tool call is a JSON object)
    static void analyze_tool_call_format_json_native(const std::string &    clean_haystack,
                                                     const std::string &    fun_name_needle,
                                                     const std::string &    arg_name_needle,
                                                     tool_format_analysis & format);

    // Analyze specifics of non-JSON native format (tags for function name or for function name and arguments)
    static void analyze_tool_call_format_non_json(const std::string &    clean_haystack,
                                                  const std::string &    fun_name_needle,
                                                  tool_format_analysis & format);

    // Check for and extract specific per-call markers for non-native-JSON templates with parallel call support
    static void check_per_call_markers(const common_chat_template & tmpl, tool_format_analysis & result);

    // Logic below is only for non-JSON-native tool calling formats
    // 3. b. function name
    // Extract function name markers
    static tool_function_analysis extract_function_markers(const common_chat_template & tmpl,
                                                           const tool_format_analysis & analysis);

    // 4. c. function arguments
    // Delegates to separate functions for: separator analysis, argument name analysis, argument value analysis
    static tool_arguments_analysis analyze_arguments(const common_chat_template & tmpl,
                                                     const tool_analysis &        analysis);

    // Extract argument name markers
    static void extract_argument_name_markers(const common_chat_template & tmpl,
                                              tool_arguments_analysis &    args_analysis);

    // Extract argument value markers
    static void extract_argument_value_markers(const common_chat_template & tmpl,
                                               const tool_analysis &        analysis,
                                               tool_arguments_analysis &    args_analysis);

    // Extract argument separator, if specified (eg. <arg=foo>...</arg><sep><arg=bar>...</arg>)
    static void extract_argument_separator(const common_chat_template & tmpl,
                                           tool_arguments_analysis &    args_analysis);

    // Extract argument wrapper markers, if present (eg. '<args><arg=foo>...</arg><arg=bar>...</arg></args>')
    static void extract_args_markers(const common_chat_template & tmpl,
                                     const tool_analysis &        analysis,
                                     tool_arguments_analysis &    args_analysis);

    // 4. d. function call id
    // Extract call ID markers, if present
    static tool_id_analysis extract_call_id_markers(const common_chat_template & tmpl,
                                                    tool_format_analysis       & analysis);

    // Collect tokens from entire analysis to preserve
    static void collect_preserved_tokens(diff_analysis_result & result);

    static std::string apply_template(const common_chat_template & tmpl, const template_params & params);
};

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
};
