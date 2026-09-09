#include "parsers.h"

common_chat_params common_chat_params_init_gpt_oss(const common_chat_template &    tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    // Copy reasoning to the "thinking" field as expected by the gpt-oss template
    auto adjusted_messages = json::array();
    for (auto msg : inputs.messages) {
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            msg["thinking"] = msg.at("reasoning_content");
            if (msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty()) {
                msg.erase("content");
            }
        }
        adjusted_messages.push_back(msg);
    }

    auto prompt = common_chat_template_direct_apply_impl(tmpl, inputs, /* messages_override= */ adjusted_messages);

    // Check if we need to replace the return token with end token during
    // inference and without generation prompt. For more details see:
    // https://github.com/ggml-org/llama.cpp/issues/15417
    if (inputs.is_inference && !inputs.add_generation_prompt) {
        static constexpr std::string_view return_token = "<|return|>";
        static constexpr std::string_view end_token    = "<|end|>";
        if (size_t pos = prompt.rfind(return_token); pos != std::string::npos) {
            prompt.replace(pos, return_token.length(), end_token);
        }
    }

    data.prompt            = prompt;
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, /* messages_override= */ adjusted_messages);
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|start|>assistant" },
        { COMMON_CHAT_ROLE_USER,      "<|start|>user"      },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|start|>developer" },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|start|>system"    },
        { COMMON_CHAT_ROLE_TOOL,      "<|start|>functions" },
    };

    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    data.thinking_start_tag = "<|channel|>analysis<|message|>";
    data.thinking_end_tags  = {"<|end|>"};

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = {
        "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>",
    };

    // Adjust prompt for continuation
    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = "<|start|>assistant<|channel|>analysis<|message|>" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "<|end|><|start|>assistant<|channel|>final<|message|>" + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start           = p.rule("start", p.literal("<|start|>assistant"));
        auto end             = p.rule("end", p.literal("<|end|>"));
        auto content         = p.rule("message-content", p.until("<|end|>"));
        auto channel         = p.literal("<|channel|>") + (p.literal("commentary") | p.literal("analysis"));
        auto constrain_type  = p.chars("[A-Za-z0-9_-]", 1, -1);

        // Occasionally, gpt-oss-20b will prefix channels with this commentary
        auto stray_commentary = p.optional(p.literal("<|channel|>commentary") + p.optional(p.literal(" to=assistant")));
        auto start_analysis = stray_commentary + p.literal("<|channel|>analysis<|message|>");

        if (extract_reasoning) {
            p.rule("analysis", start_analysis + p.reasoning(content) + end);
        } else {
            p.rule("analysis", p.content(start_analysis + content + end));
        }

        auto analysis = p.ref("analysis");
        auto preamble = p.rule("preamble", p.literal("<|channel|>commentary<|message|>") + p.content(content) + end);
        auto final_msg = p.rule("final", stray_commentary + p.literal("<|channel|>final<|message|>") + p.content(content));

        // Consume any unsolicited tool calls, e.g. builtin functions
        auto unsolicited = p.rule("unsolicited", p.atomic(p.optional(channel) + p.literal(" to=") + content + end));

        auto any = p.rule("any", preamble | analysis);

        if (has_response_format) {
            auto constraint = p.optional(p.space() + p.optional(p.literal("<|constrain|>")) + constrain_type);
            auto response_format = p.rule("response-format",
                p.literal("<|channel|>final") + constraint + p.literal("<|message|>") +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));

            return p.zero_or_more(start + analysis) + start + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto & params   = function.at("parameters");

                auto func_name  = p.literal(" to=functions.") + p.tool_name(p.literal(name));
                auto constraint = p.optional(p.space() + p.optional(p.literal("<|constrain|>")) + constrain_type);
                auto args       = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", params));

                // recipient in role header
                //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
                auto tool_in_role = p.tool(p.tool_open(func_name + channel + constraint + p.literal("<|message|>")) + args);

                // recipient in channel header
                //   <|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
                auto tool_in_channel = p.tool(p.tool_open(channel + func_name + constraint + p.literal("<|message|>")) + args);

                tool_choice |= p.rule("tool-" + name, tool_in_role | tool_in_channel);
            });

            auto tool_call  = p.trigger_rule("tool-call", tool_choice);

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return p.zero_or_more(start + any) + start + tool_call;
            }

            return p.zero_or_more(start + any) + start + (tool_call | final_msg);
        }

        return p.zero_or_more(start + any) + start + (final_msg | unsolicited);
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "^\\s+to$" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "^<\\|channel\\|>(?:commentary|analysis)\\s+to=functions$" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>assistant(\\s+to)" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>assistant(<\\|channel\\|>(?:commentary|analysis)\\s+to)" }
        };
    }

    return data;
}
