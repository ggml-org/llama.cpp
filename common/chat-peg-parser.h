#pragma once

#include "chat.h"
#include "peg-parser.h"

// ============================================================================
// Tag enum used by both old class-based and new functional mappers
// ============================================================================

// Chat PEG tag enum - all tags used in chat parsing
enum class common_chat_peg_tag : int {
    NONE = 0,
    // Base tags
    REASONING_BLOCK,
    REASONING,
    CONTENT,
    // Native tool call tags
    TOOL,
    TOOL_OPEN,
    TOOL_CLOSE,
    TOOL_ID,
    TOOL_NAME,
    TOOL_ARGS,
    // Constructed tool call tags
    TOOL_ARG,
    TOOL_ARG_OPEN,
    TOOL_ARG_CLOSE,
    TOOL_ARG_NAME,
    TOOL_ARG_STRING_VALUE,
    TOOL_ARG_JSON_VALUE,
};

// Tag to string for debugging/serialization (exhaustive switch)
inline const char * common_chat_peg_tag_to_string(common_chat_peg_tag t) {
    switch (t) {
        case common_chat_peg_tag::NONE:                 return "";
        case common_chat_peg_tag::REASONING_BLOCK:      return "reasoning-block";
        case common_chat_peg_tag::REASONING:            return "reasoning";
        case common_chat_peg_tag::CONTENT:              return "content";
        case common_chat_peg_tag::TOOL:                 return "tool";
        case common_chat_peg_tag::TOOL_OPEN:            return "tool-open";
        case common_chat_peg_tag::TOOL_CLOSE:           return "tool-close";
        case common_chat_peg_tag::TOOL_ID:              return "tool-id";
        case common_chat_peg_tag::TOOL_NAME:            return "tool-name";
        case common_chat_peg_tag::TOOL_ARGS:            return "tool-args";
        case common_chat_peg_tag::TOOL_ARG:             return "tool-arg";
        case common_chat_peg_tag::TOOL_ARG_OPEN:        return "tool-arg-open";
        case common_chat_peg_tag::TOOL_ARG_CLOSE:       return "tool-arg-close";
        case common_chat_peg_tag::TOOL_ARG_NAME:        return "tool-arg-name";
        case common_chat_peg_tag::TOOL_ARG_STRING_VALUE: return "tool-arg-string-value";
        case common_chat_peg_tag::TOOL_ARG_JSON_VALUE:  return "tool-arg-json-value";
    }
    return "unknown";
}

// Alias for the tag enum
using Tag = common_chat_peg_tag;

// ============================================================================
// Original class-based builders/mappers (used by legacy implementations in chat.cpp)
// TODO(ochafik): Remove once --experimental-new-parsers graduates.
// ============================================================================

class common_chat_peg_builder : public common_peg_parser_builder {
  public:
    // Use enum values for compatibility with new tag API
    static constexpr common_chat_peg_tag REASONING_BLOCK = common_chat_peg_tag::REASONING_BLOCK;
    static constexpr common_chat_peg_tag REASONING = common_chat_peg_tag::REASONING;
    static constexpr common_chat_peg_tag CONTENT = common_chat_peg_tag::CONTENT;

    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }
    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }
    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }
};

inline common_peg_arena build_chat_peg_parser(const std::function<common_peg_parser(common_chat_peg_builder & builder)> & fn) {
    common_chat_peg_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

class common_chat_peg_mapper {
  public:
    common_chat_msg & result;

    common_chat_peg_mapper(common_chat_msg & msg) : result(msg) {}

    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
    virtual void map(const common_peg_ast_node & node);
};

class common_chat_peg_native_builder : public common_chat_peg_builder {
  public:
    static constexpr common_chat_peg_tag TOOL = common_chat_peg_tag::TOOL;
    static constexpr common_chat_peg_tag TOOL_OPEN = common_chat_peg_tag::TOOL_OPEN;
    static constexpr common_chat_peg_tag TOOL_CLOSE = common_chat_peg_tag::TOOL_CLOSE;
    static constexpr common_chat_peg_tag TOOL_ID = common_chat_peg_tag::TOOL_ID;
    static constexpr common_chat_peg_tag TOOL_NAME = common_chat_peg_tag::TOOL_NAME;
    static constexpr common_chat_peg_tag TOOL_ARGS = common_chat_peg_tag::TOOL_ARGS;

    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_id(const common_peg_parser & p) { return atomic(tag(TOOL_ID, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_args(const common_peg_parser & p) { return tag(TOOL_ARGS, p); }
};

class common_chat_peg_native_mapper : public common_chat_peg_mapper {
    common_chat_tool_call * current_tool = nullptr;

  public:
    common_chat_peg_native_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}

    void map(const common_peg_ast_node & node) override;
};

inline common_peg_arena build_chat_peg_native_parser(const std::function<common_peg_parser(common_chat_peg_native_builder & builder)> & fn) {
    common_chat_peg_native_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

class common_chat_peg_constructed_builder : public common_chat_peg_builder {
  public:
    static constexpr common_chat_peg_tag TOOL = common_chat_peg_tag::TOOL;
    static constexpr common_chat_peg_tag TOOL_OPEN = common_chat_peg_tag::TOOL_OPEN;
    static constexpr common_chat_peg_tag TOOL_CLOSE = common_chat_peg_tag::TOOL_CLOSE;
    static constexpr common_chat_peg_tag TOOL_NAME = common_chat_peg_tag::TOOL_NAME;
    static constexpr common_chat_peg_tag TOOL_ARG = common_chat_peg_tag::TOOL_ARG;
    static constexpr common_chat_peg_tag TOOL_ARG_OPEN = common_chat_peg_tag::TOOL_ARG_OPEN;
    static constexpr common_chat_peg_tag TOOL_ARG_CLOSE = common_chat_peg_tag::TOOL_ARG_CLOSE;
    static constexpr common_chat_peg_tag TOOL_ARG_NAME = common_chat_peg_tag::TOOL_ARG_NAME;
    static constexpr common_chat_peg_tag TOOL_ARG_STRING_VALUE = common_chat_peg_tag::TOOL_ARG_STRING_VALUE;
    static constexpr common_chat_peg_tag TOOL_ARG_JSON_VALUE = common_chat_peg_tag::TOOL_ARG_JSON_VALUE;

    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_arg(const common_peg_parser & p) { return tag(TOOL_ARG, p); }
    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_OPEN, p)); }
    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_CLOSE, p)); }
    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_NAME, p)); }
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag(TOOL_ARG_STRING_VALUE, p); }
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return tag(TOOL_ARG_JSON_VALUE, p); }
};

class common_chat_peg_constructed_mapper : public common_chat_peg_mapper {
    common_chat_tool_call * current_tool = nullptr;
    int arg_count = 0;
    bool needs_closing_quote = false;

  public:
    common_chat_peg_constructed_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}

    void map(const common_peg_ast_node & node) override;
};

inline common_peg_arena build_chat_peg_constructed_parser(const std::function<common_peg_parser(common_chat_peg_constructed_builder & builder)> & fn) {
    common_chat_peg_constructed_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

// ============================================================================
// Functional mapper infrastructure (used by experimental new PEG parsers in chat-parsers/)
// ============================================================================

// Mapper types: curried functions for AST-to-message conversion
typedef std::function<void(const common_peg_ast_node & node)> common_chat_peg_map_func;
typedef std::function<common_chat_peg_map_func(common_chat_msg & result)> common_chat_peg_mapper_func;

// Helper to apply a mapper to parse results
inline void apply_chat_peg_mapper(
    const common_chat_peg_mapper_func & mapper,
    const common_peg_ast_arena & arena,
    const common_peg_parse_result & parse_result,
    common_chat_msg & msg
) {
    auto map_func = mapper(msg);
    arena.visit(parse_result, map_func);
}

// Alias for the tag enum
using Tag = common_chat_peg_tag;

// Base mapper: handles reasoning and content tags
common_chat_peg_mapper_func common_chat_peg_base_mapper();

// Native mapper: handles tool calls with pre-parsed JSON args
common_chat_peg_mapper_func common_chat_peg_native_mapper_func();

// Constructed mapper: builds JSON args from individual parsed pieces
common_chat_peg_mapper_func common_chat_peg_constructed_mapper_func();

// Short form mapper: handles {"function_name": {...}} format (used by Apertus)
common_chat_peg_mapper_func common_chat_peg_short_form_mapper();

// Generic mapper: handles general purpose parsing
common_chat_peg_mapper_func common_chat_peg_generic_mapper();

// OAI array mapper: handles OpenAI-style tool call arrays
common_chat_peg_mapper_func common_chat_peg_oai_array_mapper();

// Command R7B mapper: handles Command-R7B specific format
common_chat_peg_mapper_func common_chat_peg_command_r7b_mapper();
