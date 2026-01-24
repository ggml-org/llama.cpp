#pragma once

#include "chat.h"
#include "peg-parser.h"

#include <map>
#include <optional>

class common_chat_peg_builder : public common_peg_parser_builder {
  public:
    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING       = "reasoning";
    static constexpr const char * CONTENT         = "content";

    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }

    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }

    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }

    common_peg_parser tag_with_safe_content(const std::string &       tag_name,
                                            const std::string &       marker,
                                            const common_peg_parser & p);
};

inline common_peg_arena build_chat_peg_parser(
    const std::function<common_peg_parser(common_chat_peg_builder & builder)> & fn) {
    common_chat_peg_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

class common_chat_peg_mapper {
  public:
    common_chat_msg & result;

    common_chat_peg_mapper(common_chat_msg & msg) : result(msg) {}

    virtual ~common_chat_peg_mapper() = default;

    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
    virtual void map(const common_peg_ast_node & node);
};

struct content_structure;
struct tool_call_structure;

class common_chat_peg_unified_builder : public common_chat_peg_builder {
  public:
    // Tag constants
    static constexpr const char * TOOL           = "tool";
    static constexpr const char * TOOL_OPEN      = "tool-open";
    static constexpr const char * TOOL_CLOSE     = "tool-close";
    static constexpr const char * TOOL_ID        = "tool-id";
    static constexpr const char * TOOL_NAME      = "tool-name";
    static constexpr const char * TOOL_ARGS      = "tool-args";
    static constexpr const char * TOOL_ARG       = "tool-arg";
    static constexpr const char * TOOL_ARG_OPEN  = "tool-arg-open";
    static constexpr const char * TOOL_ARG_CLOSE = "tool-arg-close";
    static constexpr const char * TOOL_ARG_NAME         = "tool-arg-name";
    static constexpr const char * TOOL_ARG_VALUE        = "tool-arg-value";
    static constexpr const char * TOOL_ARG_STRING_VALUE = "tool-arg-string-value";  // For schema-declared string types

    // Low-level tag methods
    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }

    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }

    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }

    common_peg_parser tool_id(const common_peg_parser & p) { return atomic(tag(TOOL_ID, p)); }

    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }

    common_peg_parser tool_args(const common_peg_parser & p) { return tag(TOOL_ARGS, p); }

    common_peg_parser tool_arg(const common_peg_parser & p) { return tag(TOOL_ARG, p); }

    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_OPEN, p)); }

    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_CLOSE, p)); }

    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_NAME, p)); }

    common_peg_parser tool_arg_value(const common_peg_parser & p) { return tag(TOOL_ARG_VALUE, p); }

    // Use for schema-declared string types - won't be treated as potential JSON container
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag(TOOL_ARG_STRING_VALUE, p); }

    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return tag(TOOL_ARG_VALUE, p); }

    // High-level building methods

    // Build reasoning block based on ContentStructure
    common_peg_parser build_reasoning_block(const content_structure & cs,
                                            common_reasoning_format  reasoning_format,
                                            bool                     thinking_forced_open);

    // Build content block based on ContentStructure
    common_peg_parser build_content_block(const content_structure & cs,
                                         common_reasoning_format reasoning_format,
                                         const std::string &    tool_section_start = "");

    // Build complete tool section based on ToolCallStructure
    common_peg_parser build_tool_section(const tool_call_structure & ts,
                                         const nlohmann::json &    tools,
                                         bool                      parallel_tool_calls,
                                         bool                      force_tool_calls);

    // Build single function parser based on ToolCallStructure
    common_peg_parser build_function(const tool_call_structure & ts,
                                     const std::string &       name,
                                     const nlohmann::json &    schema);

    // Build arguments parser based on ToolCallStructure
    common_peg_parser build_arguments(const tool_call_structure & ts, const nlohmann::json & params);

    // Legacy-compatible helper for building standard JSON tool calls
    // Used by tests and manual parsers
    common_peg_parser standard_json_tools(const std::string &    section_start,
                                          const std::string &    section_end,
                                          const nlohmann::json & tools,
                                          bool                   parallel_tool_calls,
                                          bool                   force_tool_calls);

    // Legacy-compatible helper for building XML/tagged style tool calls
    // Used by tests and manual parsers
    common_peg_parser standard_constructed_tools(const std::map<std::string, std::string> & markers,
                                                 const nlohmann::json &                     tools,
                                                 bool                                       parallel_tool_calls,
                                                 bool                                       force_tool_calls);
};

inline common_peg_arena build_chat_peg_unified_parser(
    const std::function<common_peg_parser(common_chat_peg_unified_builder & builder)> & fn) {
    common_chat_peg_unified_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

class common_chat_peg_unified_mapper : public common_chat_peg_mapper {
    std::optional<common_chat_tool_call> pending_tool_call;  // Tool call waiting for name
    common_chat_tool_call *              current_tool        = nullptr;
    int                                  arg_count           = 0;
    bool                                 needs_closing_quote = false;
    std::string                          args_buffer;  // Buffer to delay arguments until tool name is known
    bool                                 buffer_needs_closing_quote = false;  // Track quote state for buffered args

  public:
    common_chat_peg_unified_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}

    void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & parse_result_arg) override;
    void map(const common_peg_ast_node & node) override;
};
