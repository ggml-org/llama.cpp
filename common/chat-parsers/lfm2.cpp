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

    // The logic requires potentially modifying the messages
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
                // Inject modified content back into the messages
                messages.at(0).at("content") = content;
                return true;
            }
        }

        return false;
    };

    // LFM2 model does not natively work with JSON, but can generally understand the tools structure
    //
    // Example of the pytorch dialog structure:
    //     <|startoftext|><|im_start|>system
    //     List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
    //     <|im_start|>user
    //     What is the current status of candidate ID 12345?<|im_end|>
    //     <|im_start|>assistant
    //     <|tool_call_start|>[{"name": "get_candidate_status", "arguments": {"candidate_id": "12345"}}]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
    //     <|im_start|>tool
    //     <|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}<|tool_response_end|><|im_end|>
    //     <|im_start|>assistant
    //     The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
    //
    // For the llama server compatibility with JSON tools semantic,
    // the client can add "force json schema." line into the system message prompt to force the JSON output.
    //
    // When the marker is present, we build a custom schema with full validation for:
    // - Tool name (exact match via const)
    // - Parameter types (full schema validation)
    // - Required id field
    // - maxItems constraint when parallel_tool_calls=false
    //
    // When the marker is absent, we don't build a grammar (the model generates unconstrained).

    // Branch 1: Error - tools + custom grammar not allowed (server prohibits this combination)
    if (are_tools_provided && (is_json_schema_provided || is_grammar_provided)) {
        throw std::runtime_error("Tools call must not use \"json_schema\" or \"grammar\", use non-tool invocation if you want to use custom grammar");
    }

    // Branch 2: Tools + "force json schema" marker → Full schema validation
    bool force_json_schema = are_tools_provided && replace_json_schema_marker(tweaked_messages);

    if (force_json_schema) {
        data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};

        // Build PEG parser with full schema validation
        auto parser = build_chat_peg_parser([&](auto & p) {
            using Tag = common_chat_peg_tag;

            static const json id_schema {
                {"type", "string"},
            };
            // Tool call: <|tool_call_start|>[{"name": "...", "arguments": {...}, "id": "..."}]<|tool_call_end|>
            // LFM2 format with ID at end: {"name": "...", "arguments": {...}, "id": "..."}
            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.literal_tag(Tag::TOOL_OPEN, "{")
                    << "\"name\"" << ":" << ("\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"") << ","
                    << "\"arguments\"" << ":" << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters)) << ","
                    << "\"id\"" << ":" << p.tag(Tag::TOOL_ID, p.schema(p.json(), "tool-id", id_schema))
                    << p.literal_tag(Tag::TOOL_CLOSE, "}"));
            });

            auto tool_calls_parser =
                p.space()
                + p.literal("<|tool_call_start|>[")
                + any_tool_call + p.repeat(p.literal(",") << any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
                + p.literal("]<|tool_call_end|>");

            auto tool_calls = p.trigger_rule("tool-call-root", tool_calls_parser);

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return tool_calls;
            }
            return p.tag(Tag::CONTENT, p.until("<|tool_call_start|>")) << tool_calls;
        });

        common_chat_build_peg_grammar(inputs, parser, data);
        data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

        // Trigger lazy grammar activation on <|tool_call_start|>[ pattern
        data.grammar_triggers = {{COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "\\s*<\\|tool_call_start\\|>\\s*\\["}};
    } else if (are_tools_provided) {
        // Branch 3: Tools without marker - no grammar, just preserved_tokens
        // The model can generate unconstrained tool calls (validated at runtime)
        // LOG_INF("%s: Using tools without json schema or grammar\n", __func__);
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
        data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};
    } else if (is_json_schema_provided) {
        // Branch 4: json_schema passthrough
        // LOG_INF("%s: Using provided json schema to build a grammar\n", __func__);
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
        data.grammar = json_schema_to_grammar(inputs.json_schema);
    } else if (is_grammar_provided) {
        // Branch 5: grammar passthrough
        // LOG_INF("%s: Using provided grammar\n", __func__);
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
        data.grammar = inputs.grammar;
    } else {
        // Branch 6: Plain content (no tools, no schema, no grammar)
        // LOG_INF("%s: Using content relying on the template\n", __func__);
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }

    data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);
    // LOG_DBG("%s: Prompt: %s\n", __func__, data.prompt.c_str());

    return data;
}
