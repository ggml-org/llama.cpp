#pragma once

#include "chat.h"

#include <chrono>
#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// ============================================================================
// UNIFIED AUTO-PARSER DATA STRUCTURES
// ============================================================================

// Phase 1 result: Content and reasoning structure (analyzed without tools)
struct ContentStructure {
    // Reasoning handling mode
    enum ReasoningMode {
        REASONING_NONE,         // No reasoning markers detected
        REASONING_OPTIONAL,     // <think>...</think> may appear before content
        REASONING_FORCED_OPEN,  // Template ends with open reasoning tag (thinking_forced_open)
    };

    ReasoningMode reasoning_mode = REASONING_NONE;
    std::string   reasoning_start;  // e.g., "<think>", "<|START_THINKING|>"
    std::string   reasoning_end;    // e.g., "</think>", "<|END_THINKING|>"

    // Content wrapping mode
    enum ContentMode {
        CONTENT_PLAIN,                   // No content markers
        CONTENT_ALWAYS_WRAPPED,          // <response>...</response> always present
        CONTENT_WRAPPED_WITH_REASONING,  // Content wrapped only when reasoning present
    };

    ContentMode content_mode = CONTENT_PLAIN;
    std::string content_start;  // e.g., "<response>", "<|START_RESPONSE|>"
    std::string content_end;    // e.g., "</response>", "<|END_RESPONSE|>"
};

// Phase 2 result: Tool call structure (layered on Phase 1)
struct ToolCallStructure {
    bool supports_tools = false;

    // Container markers (what wraps all tool calls)
    std::string tool_section_start;  // e.g., "<tool_call>", "[TOOL_CALLS]", "<TOOLCALL>", ""
    std::string tool_section_end;    // e.g., "</tool_call>", "]", "</TOOLCALL>", ""

    // Function format (how individual functions are structured)
    enum FunctionFormat {
        FUNC_JSON_OBJECT,       // {"name": "X", "arguments": {...}}
        FUNC_TAG_WITH_NAME,     // <function=X>{...}</function>
        FUNC_TAG_NAME_ONLY,     // <X>...</X> where X is function name (rare)
        FUNC_PREFIXED_INDEXED,  // <|tool_call_begin|>functions.X:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
        FUNC_NAME_AS_KEY,       // [{"function_name": {...arguments...}}] (Apertus-style)
        FUNC_BRACKET_TAG,       // [TOOL_CALLS]X[CALL_ID]id[ARGS]{...} (Mistral Small 3.2 style)
        FUNC_RECIPIENT_BASED,   // >>>recipient\n{content} where recipient is "all" (content) or function name (tools)
        FUNC_MARKDOWN_CODE_BLOCK,  // Action:\n```json\n[...]\n``` (Cohere Command-R Plus style)
    };

    FunctionFormat function_format = FUNC_JSON_OBJECT;

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
    enum ArgumentFormat {
        ARGS_JSON,            // Standard JSON object: {"key": "value", ...}
        ARGS_TAGGED,          // XML-style: <param=key>value</param>
        ARGS_KEY_VALUE_TAGS,  // <arg_key>key</arg_key><arg_value>value</arg_value> (GLM-4.6)
    };

    ArgumentFormat argument_format = ARGS_JSON;

    // For ARGS_TAGGED format
    std::string arg_prefix;     // e.g., "<param=", "<parameter="
    std::string arg_suffix;     // e.g., ">"
    std::string arg_close;      // e.g., "</param>", "</parameter>"
    std::string arg_separator;  // e.g., "", "\n"

    // Flag: template renders null content as "None" string, requires empty string instead
    bool requires_nonnull_content = false;
};

// Combined result of unified template analysis
struct TemplateAnalysisResult {
    ContentStructure  content;
    ToolCallStructure tools;

    // Preserved tokens for tokenizer (union of all markers)
    std::vector<std::string> preserved_tokens;
};

// ============================================================================
// TEMPLATE ANALYZER
// ============================================================================

// Template analyzer that uses two-phase differential analysis
class TemplateAnalyzer {
  public:
    // Main entry point: Unified two-phase analysis
    static TemplateAnalysisResult analyze_template(const minja::chat_template & tmpl);

    // Phase 1 - Analyze content and reasoning structure (no tools)
    static ContentStructure analyze_content_structure(const minja::chat_template & tmpl);

    // Phase 2 - Analyze tool call structure (layered on Phase 1)
    static ToolCallStructure analyze_tool_structure(const minja::chat_template & tmpl,
                                                    const ContentStructure &     content);

  private:
    // Phase 1 detection helpers
    static void detect_reasoning_markers(const minja::chat_template & tmpl, ContentStructure & cs);
    static void detect_content_markers(const minja::chat_template & tmpl, ContentStructure & cs);
    static ContentStructure::ReasoningMode detect_reasoning_mode(const minja::chat_template & tmpl,
                                                                 const ContentStructure &     cs,
                                                                 const std::string &          prompt);

    // Phase 2 detection helpers
    static void detect_tool_markers(const minja::chat_template & tmpl, ToolCallStructure & ts);
    static void detect_function_format(const minja::chat_template & tmpl, ToolCallStructure & ts);
    static void detect_argument_format(const minja::chat_template & tmpl, ToolCallStructure & ts);

    // Phase 2 helper methods
    static void analyze_json_format(ToolCallStructure & ts, const struct InternalDiscoveredPattern & discovered);
    static void analyze_xml_format(ToolCallStructure & ts, const struct InternalDiscoveredPattern & discovered);
    static void analyze_bracket_tag_format(ToolCallStructure & ts, const struct InternalDiscoveredPattern & discovered);
    static void analyze_recipient_based_format(ToolCallStructure &                      ts,
                                               const struct InternalDiscoveredPattern & discovered);
    static void analyze_markdown_code_block_format(ToolCallStructure &                      ts,
                                                   const struct InternalDiscoveredPattern & discovered);

    // Helper to collect preserved tokens from analysis result
    static void collect_preserved_tokens(TemplateAnalysisResult & result);
};

// ============================================================================
// TEMPLATE PARAMETERS
// ============================================================================

struct templates_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice;
    json                                  json_schema;
    bool                                  parallel_tool_calls;
    common_reasoning_format               reasoning_format;
    bool                                  stream;
    std::string                           grammar;
    bool                                  add_generation_prompt;
    bool                                  enable_thinking;
    std::chrono::system_clock::time_point now;
    json                                  extra_context;
    bool                                  add_bos;
    bool                                  add_eos;
    bool                                  is_inference;
    bool                                  add_inference;
};

// ============================================================================
// PEG PARSER GENERATOR
// ============================================================================

class UniversalPEGGenerator {
  public:
    // Generate parser from analysis result
    static common_chat_params generate_parser(const TemplateAnalysisResult &  analysis,
                                              const minja::chat_template &    tmpl,
                                              const struct templates_params & inputs);

  private:
    // Build unified parser (single code path for all formats)
    static common_peg_arena build_parser(const TemplateAnalysisResult &  analysis,
                                         const minja::chat_template &    tmpl,
                                         const struct templates_params & inputs,
                                         bool                            thinking_forced_open);
};
