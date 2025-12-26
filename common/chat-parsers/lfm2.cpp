// LFM2 tool call format
// Format: <|tool_call_start|>[{"name": "...", "arguments": {...}}]<|tool_call_end|>

#include "chat-parsers-internal.h"

// Helper to find case-insensitive substring (same as in chat.cpp)
static size_t ifind_string(const std::string & str, const std::string & pattern) {
    auto it = std::search(
        str.begin(), str.end(),
        pattern.begin(), pattern.end(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); }
    );
    return it == str.end() ? std::string::npos : std::distance(str.begin(), it);
}

common_chat_params common_chat_params_init_lfm2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    const auto is_json_schema_provided = !inputs.json_schema.is_null();
    const auto is_grammar_provided = !inputs.grammar.empty();
    const auto are_tools_provided = inputs.tools.is_array() && !inputs.tools.empty();

    // the logic requires potentially modifying the messages
    auto tweaked_messages = inputs.messages;

    auto replace_json_schema_marker = [](json & messages) -> bool {
        static std::string marker1 = "force json schema.\n";
        static std::string marker2 = "force json schema.";

        if (messages.empty() || messages.at(0).at("role") != "system") {
            return false;
        }

        std::string content = messages.at(0).at("content");

        for (const auto & marker : {marker1, marker2}) {
            const auto pos = ifind_string(content, marker);
            if (pos != std::string::npos) {
                content.replace(pos, marker.length(), "");
                // inject modified content back into the messages
                messages.at(0).at("content") = content;
                return true;
            }
        }

        return false;
    };

    // Lfm2 model does not natively work with json, but can generally understand the tools structure
    // For the llama server compatibility with json tools semantic,
    // the client can add "Follow json schema." line into the system message prompt to force the json output.
    // if (are_tools_provided && (is_json_schema_provided || is_grammar_provided)) {
    //     // server/utils.hpp prohibits that branch for the custom grammar anyways
    //     throw std::runtime_error("Tools call must not use \"json_schema\" or \"grammar\", use non-tool invocation if you want to use custom grammar");
    // } else if (are_tools_provided && replace_json_schema_marker(tweaked_messages)) {
    
    data.format = COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS;
    data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};

    // Build PEG parser
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Tool call: <|tool_call_start|> + JSON array + <|tool_call_end|>
        auto tool_call = p.tag(Tag::TOOL,
            p.atomic_tag(Tag::TOOL_OPEN, p.literal("<|tool_call_start|>"))
            + p.tag(Tag::TOOL_ARGS, p.json())
            + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("<|tool_call_end|>"))
        );

        auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
        auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
        auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, min_calls, max_calls));

        bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        if (require_tools) {
            return tool_calls;
        }
        return p.tag(Tag::CONTENT, p.until("<|tool_call_start|>")) << tool_calls;
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    data.grammar_triggers = {{COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "\\s*<\\|tool_call_start\\|>\\s*\\["}};
// } else if (are_tools_provided && (!is_json_schema_provided && !is_grammar_provided)) {
    data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};
    // } else if (is_json_schema_provided) {
    //     data.grammar = json_schema_to_grammar(inputs.json_schema);
    // } else if (is_grammar_provided) {
    //     data.grammar = inputs.grammar;
    // }
    
    replace_json_schema_marker(tweaked_messages);
    data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);

    return data;
}
