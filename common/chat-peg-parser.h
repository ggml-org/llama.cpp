#pragma once

#include "chat.h"
#include "peg-parser.h"

#include <map>
#include <optional>
#include <vector>

class common_chat_peg_mapper {
  public:
    common_chat_msg & result;

    common_chat_peg_mapper(common_chat_msg & msg) : result(msg) {}

    virtual ~common_chat_peg_mapper() = default;

    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
    virtual void map(const common_peg_ast_node & node);
    private:
      // Tool call handling state
      std::optional<common_chat_tool_call> pending_tool_call;  // Tool call waiting for name
      common_chat_tool_call *              current_tool          = nullptr;
      int                                  arg_count             = 0;
      bool                                 closing_quote_pending = false;
      std::string                          args_buffer;  // Buffer to delay arguments until tool name is known

      // Returns a reference to the active argument destination string.
      // Before tool_name is known, writes go to args_buffer; after, to current_tool->arguments.
      std::string & args_target();
};

struct content_structure;
struct tool_call_structure;

class common_chat_peg_builder : public common_peg_parser_builder {
  public:
    // Tag constants (from former common_chat_peg_base_builder)
    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING       = "reasoning";
    static constexpr const char * CONTENT         = "content";

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

    // Low-level tag methods (from former common_chat_peg_base_builder)
    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }

    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }

    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }

    common_peg_parser tag_with_safe_content(const std::string &       tag_name,
                        const std::string &       marker,
                        const common_peg_parser & p);

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
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_VALUE, p)); }

    // Legacy-compatible helper for building standard JSON tool calls
    // Used by tests and manual parsers
    // name_key/args_key: JSON key names for function name and arguments
    //   Empty or "name"/"arguments" will accept both common variations
    //   Supports dot notation for nested objects (e.g., "function.name")
    // array_wrapped: if true, tool calls are wrapped in JSON array [...]
    // function_is_key: if true, function name is the JSON key (e.g., {"func_name": {...}})
    // call_id_key: JSON key for string call ID (e.g., "id")
    // gen_call_id_key: JSON key for generated integer call ID (e.g., "tool_call_id")
    // parameters_order: order in which JSON fields should be parsed
    common_peg_parser standard_json_tools(const std::string &              section_start,
                                          const std::string &              section_end,
                                          const nlohmann::json &           tools,
                                          bool                             parallel_tool_calls,
                                          bool                             force_tool_calls,
                                          const std::string &              name_key = "",
                                          const std::string &              args_key = "",
                                          bool                             array_wrapped = false,
                                          bool                             function_is_key = false,
                                          const std::string &              call_id_key = "",
                                          const std::string &              gen_call_id_key = "",
                                          const std::vector<std::string> & parameters_order = {});

    // Legacy-compatible helper for building XML/tagged style tool calls
    // Used by tests and manual parsers
    common_peg_parser standard_constructed_tools(const std::map<std::string, std::string> & markers,
                                                 const nlohmann::json &                     tools,
                                                 bool                                       parallel_tool_calls,
                                                 bool                                       force_tool_calls);

  private:
    // Implementation helpers for standard_json_tools — one per JSON tool call layout mode
    common_peg_parser build_json_tools_function_is_key(const nlohmann::json & tools,
                                                       const std::string &    args_key,
                                                       const std::string &    effective_args_key,
                                                       const std::string &    call_id_key,
                                                       const std::string &    gen_call_id_key);

    common_peg_parser build_json_tools_nested_keys(const nlohmann::json & tools,
                                                   const std::string &    effective_name_key,
                                                   const std::string &    effective_args_key,
                                                   const std::string &    call_id_key,
                                                   const std::string &    gen_call_id_key);

    common_peg_parser build_json_tools_flat_keys(const nlohmann::json &           tools,
                                                 const std::string &              effective_name_key,
                                                 const std::string &              effective_args_key,
                                                 const std::string &              call_id_key,
                                                 const std::string &              gen_call_id_key,
                                                 const std::vector<std::string> & parameters_order);
};

inline common_peg_arena build_chat_peg_parser(
  const std::function<common_peg_parser(common_chat_peg_builder & builder)> & fn) {
  common_chat_peg_builder builder;
  builder.set_root(fn(builder));
  return builder.build();
}

class tag_based_peg_mapper {
  public:
    std::map<std::string, std::string> tags;

    void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
};

struct tagged_parse_result {
    common_peg_parse_result              result;
    std::map<std::string, std::string> tags;
};

struct tagged_peg_parser {
    common_peg_arena arena;
    bool debug = false;

    tagged_peg_parser & withDebug() {
      debug = true;
      return *this;
    }

    tagged_peg_parser & withoutDebug() {
      debug = false;
      return *this;
    }

    tagged_parse_result parse_and_extract(const std::string & input, bool is_partial = false) const;
    tagged_parse_result parse_anywhere_and_extract(const std::string & input) const;
};

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);

