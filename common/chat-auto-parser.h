#pragma once

#include "chat.h"
#include "common.h"
#include "jinja/runtime.h"

#include <chrono>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Phase 1 result: Content and reasoning structure (analyzed without tools)
struct content_structure {
    // Reasoning handling mode
    enum reasoning_mode_type {
        REASONING_NONE,         // No reasoning markers detected
        REASONING_OPTIONAL,     // <think>...</think> may appear before content
        REASONING_FORCED_OPEN,  // Template ends with open reasoning tag (thinking_forced_open)
    };

    reasoning_mode_type reasoning_mode = REASONING_NONE;
    std::string         reasoning_start;  // e.g., "<think>", "<|START_THINKING|>"
    std::string         reasoning_end;    // e.g., "</think>", "<|END_THINKING|>"

    // Content wrapping mode
    enum content_mode_type {
        CONTENT_PLAIN,                   // No content markers
        CONTENT_ALWAYS_WRAPPED,          // <response>...</response> always present
        CONTENT_WRAPPED_WITH_REASONING,  // Content wrapped only when reasoning present
    };

    content_mode_type content_mode = CONTENT_PLAIN;
    std::string       content_start;  // e.g., "<response>", "<|START_RESPONSE|>"
    std::string       content_end;    // e.g., "</response>", "<|END_RESPONSE|>"
};

// Phase 2 result: Tool call structure (layered on Phase 1)
struct tool_call_structure {
    bool supports_tools = false;

    // Container markers (what wraps all tool calls)
    std::string tool_section_start;  // e.g., "<tool_call>", "[TOOL_CALLS]", "<TOOLCALL>", ""
    std::string tool_section_end;    // e.g., "</tool_call>", "]", "</TOOLCALL>", ""

    // Function format (how individual functions are structured)
    enum function_format {
        FUNC_JSON_OBJECT,       // {"name": "X", "arguments": {...}}
        FUNC_TAG_WITH_NAME,     // <function=X>{...}</function>
        FUNC_TAG_NAME_ONLY,     // <X>...</X> where X is function name (rare)
        FUNC_PREFIXED_INDEXED,  // <|tool_call_begin|>functions.X:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
        FUNC_NAME_AS_KEY,       // [{"function_name": {...arguments...}}] (Apertus-style)
        FUNC_BRACKET_TAG,       // [TOOL_CALLS]X[CALL_ID]id[ARGS]{...} (Mistral Small 3.2 style)
        FUNC_RECIPIENT_BASED,   // >>>recipient\n{content} where recipient is "all" (content) or function name (tools)
        FUNC_MARKDOWN_CODE_BLOCK,  // Action:\n```json\n[...]\n``` (Cohere Command-R Plus style)
    };

    function_format function_format = FUNC_JSON_OBJECT;

    // For FUNC_JSON_OBJECT format - field names (may vary between templates)
    std::string name_field = "name";       // Could be "tool_name", "function"
    std::string args_field = "arguments";  // Could be "parameters", "params", "input"
    std::string id_field;                  // Optional: "id", "tool_call_id", ""

    // For FUNC_TAG_WITH_NAME format
    std::string function_prefix;  // e.g., "<function="
    std::string function_suffix;  // e.g., ">"
    std::string function_close;   // e.g., "</function>"

    // For FUNC_PREFIXED_INDEXED format (e.g., Kimi-K2)
    std::string per_call_start;      // e.g., "<|tool_call_begin|>"
    std::string function_namespace;  // e.g., "functions." (prefix before function name)
    std::string args_marker;         // e.g., "<|tool_call_argument_begin|>"
    std::string per_call_end;        // e.g., "<|tool_call_end|>"

    // For FUNC_BRACKET_TAG format (e.g., Mistral Small 3.2)
    std::string id_marker;  // e.g., "[CALL_ID]" - marker before tool call ID

    // For FUNC_MARKDOWN_CODE_BLOCK format (e.g., Cohere Command-R Plus)
    std::string code_block_marker;    // e.g., "Action:" - text marker before code block
    std::string code_block_language;  // e.g., "json" - language identifier in code fence

    // Argument format (how arguments are structured within a function)
    enum argument_format {
        ARGS_JSON,            // Standard JSON object: {"key": "value", ...}
        ARGS_TAGGED,          // XML-style: <param=key>value</param>
        ARGS_KEY_VALUE_TAGS,  // <arg_key>key</arg_key><arg_value>value</arg_value> (GLM-4.6)
    };

    argument_format argument_format = ARGS_JSON;

    // For ARGS_TAGGED format
    std::string arg_prefix;     // e.g., "<param=", "<parameter="
    std::string arg_suffix;     // e.g., ">"
    std::string arg_close;      // e.g., "</param>", "</parameter>"
    std::string arg_separator;  // e.g., "", "\n"

    // Flag: template renders null content as "None" string, requires empty string instead
    bool requires_nonnull_content = false;
};

// Combined result of unified template analysis
struct template_analysis_result {
    content_structure   content;
    tool_call_structure tools;

    // Preserved tokens for tokenizer (union of all markers)
    std::vector<std::string> preserved_tokens;
};

// Template analyzer that uses two-phase differential analysis
class template_analyzer {
  public:
    // Main entry point: Unified two-phase analysis
    static template_analysis_result analyze_template(const common_chat_template & tmpl);

    // Phase 1 - Analyze content and reasoning structure (no tools)
    static content_structure analyze_content_structure(const common_chat_template & tmpl);

    // Phase 2 - Analyze tool call structure (layered on Phase 1)
    static tool_call_structure analyze_tool_structure(const common_chat_template & tmpl,
                                                      const content_structure &    content);

  private:
    // Phase 1 detection helpers
    static void detect_reasoning_markers(const common_chat_template & tmpl, content_structure & cs);
    static void detect_content_markers(const common_chat_template & tmpl, content_structure & cs);
    static content_structure::reasoning_mode_type detect_reasoning_mode(const content_structure & cs,
                                                                        const std::string &       prompt);

    // Phase 2 detection helpers
    static void detect_tool_markers(const common_chat_template & tmpl, tool_call_structure & ts);
    static void detect_function_format(const common_chat_template & tmpl, tool_call_structure & ts);
    static void detect_argument_format(const common_chat_template & tmpl, tool_call_structure & ts);

    // Phase 2 helper methods
    static void analyze_json_format(tool_call_structure & ts, const struct internal_discovered_pattern & discovered);
    static void analyze_xml_format(tool_call_structure & ts, const struct internal_discovered_pattern & discovered);
    static void analyze_bracket_tag_format(tool_call_structure &                      ts,
                                           const struct internal_discovered_pattern & discovered);
    static void analyze_recipient_based_format(tool_call_structure &                      ts,
                                               const struct internal_discovered_pattern & discovered);
    static void analyze_markdown_code_block_format(tool_call_structure &                      ts,
                                                   const struct internal_discovered_pattern & discovered);

    // Helper to collect preserved tokens from analysis result
    static void collect_preserved_tokens(template_analysis_result & result);
};

struct templates_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    json                                  json_schema;
    bool                                  parallel_tool_calls = true;
    common_reasoning_format               reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    bool                                  stream = true;
    std::string                           grammar;
    bool                                  add_generation_prompt = false;
    bool                                  enable_thinking = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    json                                  extra_context;
    bool                                  add_bos = false;
    bool                                  add_eos = false;
    bool                                  is_inference = true;
    bool                                  add_inference = false;
    bool                                  mark_input = true; // whether to mark input strings in the jinja context
};

class universal_peg_generator {
  public:
    // Generate parser from analysis result
    static common_chat_params generate_parser(const template_analysis_result & analysis,
                                              const common_chat_template &     tmpl,
                                              const struct templates_params &  inputs);

  private:
    // Build unified parser (single code path for all formats)
    static common_peg_arena build_parser(const template_analysis_result & analysis,
                                         const common_chat_template &     tmpl,
                                         const struct templates_params &  inputs,
                                         bool                             thinking_forced_open);
};
