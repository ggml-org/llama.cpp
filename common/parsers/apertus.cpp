#include "parsers.h"

// Normalize assistant history to content.blocks: the template rejects mixed string and block formats.
static json apertus_1_5_adjust_messages(const json & messages) {
    json adjusted = json::array();

    for (const auto & message : messages) {
        if (message.value("role", "") != "assistant") {
            adjusted.push_back(message);
            continue;
        }

        json blocks = json::array();

        if (message.contains("reasoning_content") && message.at("reasoning_content").is_string()) {
            const auto reasoning = message.at("reasoning_content").get<std::string>();
            if (!reasoning.empty()) {
                blocks.push_back({
                    { "type", "thoughts" },
                    { "text", reasoning },
                });
            }
        }

        if (message.contains("content") && message.at("content").is_string()) {
            const auto content = message.at("content").get<std::string>();
            if (!content.empty()) {
                blocks.push_back({
                    { "type", "response" },
                    { "text", content },
                });
            }
        }

        if (message.contains("tool_calls") && message.at("tool_calls").is_array() &&
            !message.at("tool_calls").empty()) {
            blocks.push_back({
                { "type", "tool_calls" },
                { "calls", message.at("tool_calls") },
            });
        }

        json converted = message;
        converted["content"] = {
            { "blocks", std::move(blocks) },
        };
        converted.erase("reasoning_content");
        converted.erase("tool_calls");
        adjusted.push_back(std::move(converted));
    }

    return adjusted;
}

// Apertus 1.5 format:
// - reasoning: <|inner_prefix|>{reasoning}<|inner_suffix|>
// - tool calls: <|tools_prefix|>[{"function_name": { ... }}]<|tools_suffix|>
common_chat_params common_chat_params_init_apertus_1_5(
        const common_chat_template &          tmpl,
        const autoparser::generation_params & inputs) {
    static const std::string ASSISTANT_START = "<|assistant_start|>";
    static const std::string INNER_START     = "<|inner_prefix|>";
    static const std::string INNER_END       = "<|inner_suffix|>";
    static const std::string TOOLS_START     = "<|tools_prefix|>";
    static const std::string TOOLS_END       = "<|tools_suffix|>";

    common_chat_params data;

    auto adjusted_messages = apertus_1_5_adjust_messages(inputs.messages);
    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs, adjusted_messages);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs, adjusted_messages);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = INNER_START;
    data.thinking_end_tags  = {INNER_END};
    data.preserved_tokens   = {
        INNER_START,
        INNER_END,
        TOOLS_START,
        TOOLS_END,
    };
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, ASSISTANT_START         },
        { COMMON_CHAT_ROLE_TOOL,      "<|tool_output_start|>" },
        { COMMON_CHAT_ROLE_USER,      "<|user_start|>"        },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|system_start|>"      },
    };

    if (inputs.has_continuation()) {
        const auto & message = inputs.continue_msg;

        data.generation_prompt = ASSISTANT_START;
        if (!message.reasoning_content.empty()) {
            data.generation_prompt += INNER_START + message.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += INNER_END;
            }
        }
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += message.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty() &&
                           inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    const bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    const bool extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    const bool include_grammar     = has_response_format || has_tools;

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(ASSISTANT_START);
        auto end               = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning) {
            reasoning = p.optional(
                p.literal(INNER_START) + p.reasoning(p.until(INNER_END)) + p.literal(INNER_END));
        }

        if (has_response_format) {
            return generation_prompt + reasoning +
                   p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) + end;
        }

        if (!has_tools) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](size_t, const json & tool) {
            const auto & function = tool.at("function");
            const auto name = function.at("name").get<std::string>();
            // Do not emit a prefix name before the complete JSON key arrives.
            auto key = p.atomic(p.tool_open(p.literal("{")) + p.space() + p.literal("\"") +
                p.tool_name(p.literal(name)) + p.literal("\"") + p.space() + p.literal(":"));
            tool_choice |= p.rule("tool-" + name, p.tool(key + p.space() +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", common_chat_tool_parameters(function))) +
                p.space() + p.tool_close(p.literal("}"))));
        });
        auto tool_calls = tool_choice;
        if (inputs.parallel_tool_calls) {
            tool_calls += p.zero_or_more(p.space() + p.literal(",") + p.space() + tool_choice);
        }
        // Allow the EOG token once the tool-call JSON is complete.
        auto suffix = p.optional(p.literal(TOOLS_END));
        auto tool_section = p.trigger_rule("tool-call", p.literal(TOOLS_START) + p.space() +
            p.literal("[") + p.space() + tool_calls + p.space() + p.literal("]") + p.space() + suffix);

        auto content_before = p.content(p.until(TOOLS_START));
        auto content_after  = p.content(p.rest());
        auto tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? tool_section : p.optional(tool_section);
        return generation_prompt + reasoning + content_before + tools + content_after + end;
    });

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (has_tools && data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, TOOLS_START },
            };
        }
    }

    return data;
}
