#include "parsers.h"

// LLM-jp-4.1: the GPT-OSS (Harmony) format with two differences
//  - the tokenizer emits a space after every special token: "<|channel|> analysis<|message|> ..."
//  - parallel tool calls are consecutive assistant messages, all but the last closed by <|end|>
common_chat_params common_chat_params_init_llm_jp_harmony(const common_chat_template &          tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    // Copy reasoning to the "thinking" field as expected by the template
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
        // tokenizer space after special tokens; not p.space() since GBNF `space` allows one space only
        auto sp          = p.chars("[ ]", 0, -1);
        auto channel_tag = p.literal("<|channel|>") + sp;
        // one space only: keep an intentional leading space in the body
        auto message     = p.literal("<|message|>") + p.optional(p.literal(" "));

        auto start          = p.rule("start", p.literal("<|start|>") + sp + p.literal("assistant"));
        auto end            = p.rule("end", p.literal("<|end|>"));
        auto content        = p.rule("message-content", p.until("<|end|>"));
        auto channel        = channel_tag + (p.literal("commentary") | p.literal("analysis"));
        auto constrain_type = p.chars("[A-Za-z0-9_-]", 1, -1);
        auto constraint     = p.optional(p.space() + p.optional(p.literal("<|constrain|>") + sp) + constrain_type);

        auto start_analysis = channel_tag + p.literal("analysis") + message;
        if (extract_reasoning) {
            p.rule("analysis", start_analysis + p.reasoning(content) + end);
        } else {
            p.rule("analysis", p.content(start_analysis + content + end));
        }

        auto analysis  = p.ref("analysis");
        auto preamble  = p.rule("preamble", channel_tag + p.literal("commentary") + message + p.content(content) + end);
        auto final_msg = p.rule("final", channel_tag + p.literal("final") + message + p.content(content));

        auto any = p.rule("any", preamble | analysis);

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                channel_tag + p.literal("final") + constraint + message +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));

            return p.zero_or_more(start + analysis) + start + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto   params   = common_chat_tool_parameters(function);

                auto func_name = p.literal(" to=functions.") + p.tool_name(p.literal(name));
                auto args      = p.tool_args(p.schema(p.json(), "tool-" + std::to_string(tool_index) + "-schema", params));

                // recipient in role header
                //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
                auto tool_in_role = p.tool(p.tool_open(func_name + channel + constraint + message) + args);

                // recipient in channel header
                //   <|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
                auto tool_in_channel = p.tool(p.tool_open(channel + func_name + constraint + message) + args);

                tool_choice |= p.rule("tool-" + std::to_string(tool_index), tool_in_role | tool_in_channel);
            });

            // parallel calls are separated by <|end|>; inside the trigger rule so the lazy grammar covers all of them
            auto tool_calls = inputs.parallel_tool_calls
                ? tool_choice + p.zero_or_more(end + start + tool_choice)
                : tool_choice;
            auto tool_call  = p.trigger_rule("tool-call", tool_calls);

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return p.zero_or_more(start + any) + start + tool_call;
            }

            return p.zero_or_more(start + any) + start + (tool_call | final_msg);
        }

        return p.zero_or_more(start + any) + start + final_msg;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "^\\s+to$" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "^<\\|channel\\|>\\s*(?:commentary|analysis)\\s+to=functions$" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>\\s*assistant(\\s+to)" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>\\s*assistant(<\\|channel\\|>\\s*(?:commentary|analysis)\\s+to)" }
        };
    }

    return data;
}
