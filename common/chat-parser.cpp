#include "chat-parser.h"

#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "peg-parser.h"
#include "regex-partial.h"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

/**
 * All common_chat_parse_* moved from chat.cpp to chat-parser.cpp below
 * to reduce incremental compile time for parser changes.
 */
static void common_chat_parse_generic(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    static const std::vector<std::vector<std::string>> content_paths = {
        { "response" },
    };
    static const std::vector<std::vector<std::string>> args_paths = {
        { "tool_call",  "arguments" },
        { "tool_calls", "arguments" },
    };
    auto data = builder.consume_json_with_dumped_args(args_paths, content_paths);
    if (data.value.contains("tool_calls")) {
        if (!builder.add_tool_calls(data.value.at("tool_calls")) || data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool calls");
        }
    } else if (data.value.contains("tool_call")) {
        if (!builder.add_tool_call(data.value.at("tool_call")) || data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call");
        }
    } else if (data.value.contains("response")) {
        const auto & response = data.value.at("response");
        builder.add_content(response.is_string() ? response.template get<std::string>() : response.dump(2));
        if (data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete response");
        }
    } else {
        throw common_chat_msg_partial_exception("Expected 'tool_call', 'tool_calls' or 'response' in JSON");
    }
}

static void common_chat_parse_gpt_oss(common_chat_msg_parser & builder) {
    static const std::string constraint = "(?: (<\\|constrain\\|>)?([a-zA-Z0-9_-]+))";
    static const std::string recipient("(?: to=functions\\.([^<\\s]+))");

    static const common_regex start_regex("<\\|start\\|>assistant");
    static const common_regex analysis_regex("<\\|channel\\|>analysis");
    static const common_regex final_regex("<\\|channel\\|>final" + constraint + "?");
    static const common_regex preamble_regex("<\\|channel\\|>commentary");
    static const common_regex tool_call1_regex(recipient + "<\\|channel\\|>(analysis|commentary)" + constraint + "?");
    static const common_regex tool_call2_regex("<\\|channel\\|>(analysis|commentary)" + recipient + constraint + "?");

    auto consume_end = [&](bool include_end = false) {
        if (auto res = builder.try_find_literal("<|end|>")) {
            return res->prelude + (include_end ? builder.str(res->groups[0]) : "");
        }
        return builder.consume_rest();
    };

    auto handle_tool_call = [&](const std::string & name) {
        if (auto args = builder.try_consume_json_with_dumped_args({ {} })) {
            if (builder.syntax().parse_tool_calls) {
                if (!builder.add_tool_call(name, "", args->value) || args->is_partial) {
                    throw common_chat_msg_partial_exception("incomplete tool call");
                }
            } else if (args->is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
    };

    auto regex_match = [](const common_regex & regex, const std::string & input) -> std::optional<common_regex_match> {
        auto match = regex.search(input, 0, true);
        if (match.type == COMMON_REGEX_MATCH_TYPE_FULL) {
            return match;
        }
        return std::nullopt;
    };

    do {
        auto header_start_pos = builder.pos();
        auto content_start    = builder.try_find_literal("<|message|>");
        if (!content_start) {
            throw common_chat_msg_partial_exception("incomplete header");
        }

        auto header = content_start->prelude;

        if (auto match = regex_match(tool_call1_regex, header)) {
            auto group = match->groups[1];
            auto name  = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        if (auto match = regex_match(tool_call2_regex, header)) {
            auto group = match->groups[2];
            auto name  = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        if (regex_match(analysis_regex, header)) {
            builder.move_to(header_start_pos);
            if (builder.syntax().reasoning_format == COMMON_REASONING_FORMAT_NONE ||
                builder.syntax().reasoning_in_content) {
                builder.add_content(consume_end(true));
            } else {
                builder.try_parse_reasoning("<|channel|>analysis<|message|>", "<|end|>");
            }
            continue;
        }

        if (regex_match(final_regex, header) || regex_match(preamble_regex, header)) {
            builder.add_content(consume_end());
            continue;
        }

        // Possibly a malformed message, attempt to recover by rolling
        // back to pick up the next <|start|>
        LOG_DBG("%s: unknown header from message: %s\n", __func__, header.c_str());
        builder.move_to(header_start_pos);
    } while (builder.try_find_regex(start_regex, std::string::npos, false));

    auto remaining = builder.consume_rest();
    if (!remaining.empty()) {
        LOG_DBG("%s: content after last message: %s\n", __func__, remaining.c_str());
    }
}

static void common_chat_parse_content_only(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    builder.add_content(builder.consume_rest());
}

static void common_chat_parse(common_chat_msg_parser & builder) {
    LOG_DBG("Parsing input with format %s: %s\n", common_chat_format_name(builder.syntax().format),
            builder.input().c_str());

    switch (builder.syntax().format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            common_chat_parse_content_only(builder);
            break;
        case COMMON_CHAT_FORMAT_GENERIC:
            common_chat_parse_generic(builder);
            break;
        case COMMON_CHAT_FORMAT_GPT_OSS:
            common_chat_parse_gpt_oss(builder);
            break;
        default:
            throw std::runtime_error(std::string("Unsupported format: ") +
                                     common_chat_format_name(builder.syntax().format));
    }
    builder.finish();
}

common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax) {
    if (syntax.format == COMMON_CHAT_FORMAT_PEG_SIMPLE || syntax.format == COMMON_CHAT_FORMAT_PEG_NATIVE ||
        syntax.format == COMMON_CHAT_FORMAT_PEG_CONSTRUCTED) {
        return common_chat_peg_parse(syntax.parser, input, is_partial, syntax);
    }
    common_chat_msg_parser builder(input, is_partial, syntax);
    try {
        common_chat_parse(builder);
    } catch (const common_chat_msg_partial_exception & ex) {
        LOG_DBG("Partial parse: %s\n", ex.what());
        if (!is_partial) {
            builder.clear_tools();
            builder.move_to(0);
            common_chat_parse_content_only(builder);
        }
    }
    auto msg = builder.result();
    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat<json>({ msg }).at(0).dump().c_str());
    }
    return msg;
}

common_chat_msg common_chat_peg_parse(const common_peg_arena &   parser,
                                      const std::string &        input,
                                      bool                       is_partial,
                                      const common_chat_syntax & syntax) {
    if (parser.empty()) {
        throw std::runtime_error("Failed to parse due to missing parser definition.");
    }

    LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(syntax.format), input.c_str());

    common_peg_parse_context ctx(input, is_partial);
    ctx.debug   = syntax.debug;
    auto result = parser.parse(ctx);

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial && result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            common_chat_msg msg;
            msg.role = "assistant";
            if (syntax.format == COMMON_CHAT_FORMAT_PEG_NATIVE || syntax.format == COMMON_CHAT_FORMAT_PEG_CONSTRUCTED) {
                auto mapper = common_chat_peg_unified_mapper(msg);
                mapper.from_ast(ctx.ast, result);
            } else {
                auto mapper = common_chat_peg_mapper(msg);
                mapper.from_ast(ctx.ast, result);
            }
            if (ctx.debug) {
                fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", ctx.ast.dump().c_str());
                fflush(stderr);
            }
            return msg;
        }
        throw std::runtime_error(std::string("Failed to parse input at pos ") + std::to_string(result.end) + ": " +
                                 input.substr(result.end));
    }

    common_chat_msg msg;
    msg.role = "assistant";

    if (syntax.format == COMMON_CHAT_FORMAT_PEG_NATIVE || syntax.format == COMMON_CHAT_FORMAT_PEG_CONSTRUCTED) {
        auto mapper = common_chat_peg_unified_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    } else {
        // Generic mapper
        auto mapper = common_chat_peg_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    }
    if (ctx.debug) {
        fprintf(stderr, "\nAST for %s parse:\n%s\n", is_partial ? "partial" : "full", ctx.ast.dump().c_str());
        fflush(stderr);
    }

    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat<json>({ msg }).at(0).dump().c_str());
    }
    return msg;
}
