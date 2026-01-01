// Ministral/Mistral Large 3 tool call format
// Format: [TOOL_CALLS]name[ARGS]{"param": value}
// With optional [THINK]...[/THINK] reasoning blocks

#include "chat-parsers-internal.h"
#include "chat.h"

common_chat_params common_chat_params_init_ministral_3_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Build up messages to follow the format: https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/blob/main/chat_template.jinja
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto role = msg.value("role", "");
        if (role != "system" && role != "assistant") {
            // Only adjust system and assistant messages. Interestingly, the system message may contain thinking.
            adjusted_messages.push_back(msg);
            continue;
        }

        auto content = json::array();

        // If message contains `reasoning_content`, add it as a block of type `thinking`
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            content.push_back({
                {"type", "thinking"},
                {"thinking", msg.at("reasoning_content").get<std::string>()},
            });
        }

        // If message contains `content`, add it as a block of type `text`
        if (msg.contains("content")) {
            if (msg.at("content").is_string()) {
                content.push_back({
                    {"type", "text"},
                    {"text", msg.at("content").get<std::string>()},
                });
            } else if (msg.at("content").is_array()) {
                auto blocks = msg.at("content");
                content.insert(content.end(), blocks.begin(), blocks.end());
            }
        }

        auto adjusted = msg;
        adjusted["content"] = content;
        adjusted.erase("reasoning_content");
        adjusted_messages.push_back(adjusted);
    }

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    data.prompt = apply(tmpl, inputs, /* messages_override = */ adjusted_messages);
    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
        "[TOOL_CALLS]",
        "[ARGS]",
    };

    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        auto reasoning = extract_reasoning ? p.optional("[THINK]" + p.tag(Tag::REASONING, p.until("[/THINK]")) + "[/THINK]") : p.eps();

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            // Ministral wants to emit json surrounded by code fences
            return reasoning << "```json" << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema)) << "```";
        }

        // Tool call parser
        // Format: [TOOL_CALLS]func1[ARGS]{...}[TOOL_CALLS]func2[ARGS]{...}
        // Note: [TOOL_CALLS] prefix appears before EACH tool call
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers = {
                    {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"}
                };
            }

            // Format: [TOOL_CALLS]func1[ARGS]{...}[TOOL_CALLS]func2[ARGS]{...}
            // Note: No separator - each call has its own [TOOL_CALLS] prefix
            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.tag(Tag::TOOL_OPEN, p.literal("[TOOL_CALLS]"))
                    // Wrap name + delimiter in atomic so TOOL_NAME isn't emitted prematurely
                    // when one tool name is a prefix of another (e.g., special_function vs special_function_with_opt).
                    + p.atomic(p.literal_tag(Tag::TOOL_NAME, name) + p.literal("[ARGS]"))
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    + p.tag(Tag::TOOL_CLOSE, p.eps()));
            });

            auto tool_calls = 
                p.space()
                + p.repeat(any_tool_call, 1, inputs.parallel_tool_calls ? -1 : 1);

            if (require_tools) {
                return reasoning << tool_calls;
            }
            // Allow either: content before tool calls, or content only
            auto content_before = p.tag(Tag::CONTENT, p.until("[TOOL_CALLS]"));
            auto with_tools = content_before << tool_calls;
            auto content_only = p.tag(Tag::CONTENT, p.rest());
            return reasoning << p.choice({with_tools, content_only});
        }

        return reasoning << p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
