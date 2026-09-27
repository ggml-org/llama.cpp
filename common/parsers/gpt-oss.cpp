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

    auto parser = build_chat_peg_parser(inputs.specials, [&](common_chat_peg_builder & p) {
        auto start_assistant = p.token("<|start|>") + p.literal("assistant");
        auto start           = p.rule("start", start_assistant);
        auto end             = p.rule("end", p.token("<|end|>"));
        auto content         = p.rule("message-content", p.until(p.token("<|end|>")));
        auto channel         = p.token("<|channel|>") + (p.literal("commentary") | p.literal("analysis"));
        auto constrain_type  = p.chars("[A-Za-z0-9_-]", 1, -1);

        // Occasionally, gpt-oss-20b will prefix channels with this commentary
        auto stray_commentary = p.optional(p.token("<|channel|>") + p.literal("commentary") + p.optional(p.literal(" to=assistant")));
        auto start_analysis = stray_commentary + p.token("<|channel|>") + p.literal("analysis") + p.token("<|message|>");

        if (extract_reasoning) {
            p.rule("analysis", start_analysis + p.reasoning(content) + end);
        } else {
            p.rule("analysis", p.content(start_analysis + content + end));
        }

        auto analysis = p.ref("analysis");
        auto preamble = p.rule("preamble", p.token("<|channel|>") + p.literal("commentary") + p.token("<|message|>") + p.content(content) + end);
        auto final_msg = p.rule("final", stray_commentary + p.token("<|channel|>") + p.literal("final") + p.token("<|message|>") + p.content(content));

        // Consume any unsolicited tool calls, e.g. builtin functions
        auto unsolicited = p.rule("unsolicited", p.atomic(p.optional(channel) + p.literal(" to=") + content + end));

        auto any = p.rule("any", preamble | analysis);

        if (has_response_format) {
            auto constraint = p.optional(p.space() + p.optional(p.token("<|constrain|>")) + constrain_type);
            auto response_format = p.rule("response-format",
                p.token("<|channel|>") + p.literal("final") + constraint + p.token("<|message|>") +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));

            return p.zero_or_more(start + analysis) + start + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto in_role    = p.choice();
            auto in_channel = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto   params   = common_chat_tool_parameters(function);

                auto func_name  = p.tool_name(p.literal(name));
                auto constraint = p.optional(p.space() + p.optional(p.token("<|constrain|>")) + constrain_type);
                auto args       = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", params));

                // recipient in role header
                //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
                in_role |= p.rule("tool-" + name + "-in-role", func_name + channel + constraint + p.token("<|message|>") + args);

                // recipient in channel header
                //   <|start|>assistant<|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
                in_channel |= p.rule("tool-" + name + "-in-channel", func_name + constraint + p.token("<|message|>") + args);
            });

            // The recipient prefix is the trigger, so the grammar constrains the name that follows it
            auto to_functions = p.tool_open(p.literal(" to=functions."));
            auto tool_call    = p.trigger_rule("tool-call", {
                { start_assistant + to_functions,                                                    in_role    },
                { start_assistant + p.token("<|channel|>") + p.literal("commentary") + to_functions, in_channel },
                { start_assistant + p.token("<|channel|>") + p.literal("analysis") + to_functions,   in_channel },
            });

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return p.zero_or_more(start + any) + tool_call;
            }

            return p.zero_or_more(start + any) + (tool_call | start + final_msg);
        }

        return p.zero_or_more(start + any) + start + (final_msg | unsolicited);
    });

    data.parser = parser.save();

    if (include_grammar) {
        // The grammar scans for the triggers itself, so the sampler runs it from the first token
        data.grammar_lazy = false;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO);
        });
    }

    return data;
}
