// Generic tool call format (fallback)
// Format: JSON with tool_call/tool_calls or response field
// Single: {"tool_call": {"name": "func", "arguments": {...}}}
// Multiple: {"tool_calls": [{"name": "func", "arguments": {...}}]}
// Response: {"response": "..."}

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_generic_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto tool_call_schemas = json::array();
    foreach_function(inputs.tools, [&](const auto & function, const auto & name, const auto & parameters, const auto &) {
        auto tool_schema = json {
            {"type", "object"},
            {"properties", {
                {"name", {
                    {"type", "string"},
                    {"const", name},
                }},
                {"arguments", parameters},
            }},
            {"required", json::array({"name", "arguments"})},
        };
        if (function.contains("description")) {
            tool_schema["description"] = function.at("description");
        }
        if (inputs.parallel_tool_calls) {
            tool_schema.at("properties")["id"] = {
                {"type", "string"},
                {"minLength", 4},
            };
            tool_schema.at("required").push_back("id");
        }
        tool_call_schemas.emplace_back(tool_schema);
    });
    const auto tool_call =
        inputs.parallel_tool_calls
            ? json {
                {"type", "object"},
                {"properties", {
                    {"tool_calls", {
                        {"type", "array"},
                        {"items", tool_call_schemas.size() == 1 ? tool_call_schemas[0] : json {
                            {"anyOf", tool_call_schemas},
                        }},
                        {"minItems", 1},
                    }},
                }},
                {"required", json::array({"tool_calls"})},
            }
            : json {
                {"type", "object"},
                {"properties", {
                    {"tool_call", tool_call_schemas.size() == 1 ? tool_call_schemas[0] : json {
                        {"anyOf", tool_call_schemas},
                    }},
                }},
                {"required", json::array({"tool_call"})},
            };
    const auto schema =
        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED
            ? json {
                {"anyOf", json::array({
                    tool_call,
                    {
                        {"type", "object"},
                        {"properties", {
                            {"response", inputs.json_schema.is_null()
                                ? json {{"type", "string"}}
                                : inputs.json_schema
                            },
                        }},
                        {"required", json::array({"response"})},
                    },
                })}
            }
            : tool_call;

    // Build PEG parser for generic JSON format
    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // The generic format uses JSON with specific structure
        // {"tool_call": {...}} or {"tool_calls": [...]} or {"response": "..."}
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Parse as JSON and extract tool calls
            return p.tag(Tag::TOOL_ARGS, p.json());
        }

        // Content only - parse as JSON and extract response
        return p.tag(Tag::CONTENT, p.json());
    });

    auto tweaked_messages = common_chat_template::add_system(
        inputs.messages,
        "Respond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request");

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_GENERIC;
    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
