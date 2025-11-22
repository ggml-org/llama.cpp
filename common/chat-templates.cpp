#include "chat-templates.h"
#include "chat-parser.h"
#include "chat-parser-xml-toolcall.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "regex-partial.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cctype>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

typedef minja::chat_template common_chat_template;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template; // Model had builtin template or template overridde was specified.
    std::unique_ptr<common_chat_template> template_default; // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

struct templates_params {
    json messages;
    json tools;
    common_chat_tool_choice tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    json extra_context;
    bool add_bos;
    bool add_eos;
    bool is_inference = true;
};


static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto time = std::chrono::system_clock::to_time_t(now);
    auto local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static std::string wrap_code_as_arguments(common_chat_msg_parser & builder, const std::string & code) {
    std::string arguments;
    if (builder.is_partial()) {
        arguments = (json {{"code", code + builder.healing_marker()}}).dump();
        auto idx = arguments.find(builder.healing_marker());
        if (idx != std::string::npos) {
            arguments.resize(idx);
        }
    } else {
        arguments = (json {{"code", code}}).dump();
    }
    return arguments;
}

/**
 * Takes a prefix regex that must have 1 group to capture the function name, a closing suffix, and expects json parameters in between.
 * Aggregates the prefix, suffix and in-between text into the content.
 */
static void parse_json_tool_calls(
    common_chat_msg_parser & builder,
    const std::optional<common_regex> & block_open,
    const std::optional<common_regex> & function_regex_start_only,
    const std::optional<common_regex> & function_regex,
    const common_regex & close_regex,
    const std::optional<common_regex> & block_close,
    bool allow_raw_python = false,
    const std::function<std::string(const common_chat_msg_parser::find_regex_result & fres)> & get_function_name = nullptr) {

    auto parse_tool_calls = [&]() {
        size_t from = std::string::npos;
        auto first = true;
        while (true) {
            auto start_pos = builder.pos();
            auto res = function_regex_start_only && first
                ? builder.try_consume_regex(*function_regex_start_only)
                : function_regex
                    ? builder.try_find_regex(*function_regex, from)
                    : std::nullopt;

            if (res) {
                std::string name;
                if (get_function_name) {
                    name = get_function_name(*res);
                } else {
                    GGML_ASSERT(res->groups.size() == 2);
                    name = builder.str(res->groups[1]);
                }
                first = false;
                if (name.empty()) {
                    // get_function_name signalled us that we should skip this match and treat it as content.
                    from = res->groups[0].begin + 1;
                    continue;
                }
                from = std::string::npos;

                auto maybe_raw_python = name == "python" && allow_raw_python;
                if (builder.input()[builder.pos()] == '{' || !maybe_raw_python) {
                    if (auto arguments = builder.try_consume_json_with_dumped_args({{}})) {
                        if (!builder.add_tool_call(name, "", arguments->value) || arguments->is_partial) {
                            throw common_chat_msg_partial_exception("incomplete tool call");
                        }
                        builder.consume_regex(close_regex);
                    }
                    continue;
                }
                if (maybe_raw_python) {
                    auto arguments = wrap_code_as_arguments(builder, builder.consume_rest());
                    if (!builder.add_tool_call(name, "", arguments)) {
                        throw common_chat_msg_partial_exception("incomplete tool call");
                    }
                    return;
                }
                throw common_chat_msg_partial_exception("incomplete tool call");
            } else {
                builder.move_to(start_pos);
            }
            break;
        }
        if (block_close) {
            builder.consume_regex(*block_close);
        }
        builder.consume_spaces();
        builder.add_content(builder.consume_rest());
    };
    if (block_open) {
        if (auto res = builder.try_find_regex(*block_open)) {
            parse_tool_calls();
        } else {
            builder.add_content(builder.consume_rest());
        }
    } else {
        parse_tool_calls();
    }
}

static void parse_prefixed_json_tool_call_array(common_chat_msg_parser & builder, const common_regex & prefix, size_t rstrip_prefix = 0) {
    static const std::vector<std::vector<std::string>> args_paths = {{"arguments"}};
    if (auto res = builder.try_find_regex(prefix)) {
        builder.move_back(rstrip_prefix);
        auto tool_calls = builder.consume_json_with_dumped_args(args_paths);
        if (!builder.add_tool_calls(tool_calls.value) || tool_calls.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call array");
        }
    } else {
        builder.add_content(builder.consume_rest());
    }
}

static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

static std::string apply(
    const common_chat_template & tmpl,
    const struct templates_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt)
{
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = messages_override ? *messages_override : inputs.messages;
    if (tools_override) {
        tmpl_inputs.tools = *tools_override;
    } else {
        tmpl_inputs.tools = inputs.tools.empty() ? json() : inputs.tools;
    }
    tmpl_inputs.add_generation_prompt = inputs.add_generation_prompt;
    tmpl_inputs.extra_context = inputs.extra_context;
    tmpl_inputs.extra_context["enable_thinking"] = inputs.enable_thinking;
    if (additional_context) {
        tmpl_inputs.extra_context.merge_patch(*additional_context);
    }
    // TODO: add flag to control date/time, if only for testing purposes.
    // tmpl_inputs.now = std::chrono::system_clock::now();

    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

static common_chat_params common_chat_params_init_generic(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto tool_call_schemas = json::array();
    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & function = tool.at("function");
        auto tool_schema = json {
            {"type", "object"},
            {"properties", {
                {"name", {
                    {"type", "string"},
                    {"const", function.at("name")},
                }},
                {"arguments", function.at("parameters")},
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

    data.grammar_lazy = false;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        builder.add_schema("root", schema);
    });

    auto tweaked_messages = common_chat_template::add_system(
        inputs.messages,
        "Respond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request");

    data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);
    data.format = COMMON_CHAT_FORMAT_GENERIC;
    return data;
}
static void common_chat_parse_generic(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    static const std::vector<std::vector<std::string>> content_paths = {
        {"response"},
    };
    static const std::vector<std::vector<std::string>> args_paths = {
        {"tool_call", "arguments"},
        {"tool_calls", "arguments"},
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

static common_chat_params common_chat_params_init_mistral_nemo(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        auto schemas = json::array();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            schemas.push_back({
                {"type", "object"},
                {"properties", {
                    // Important note: the model is probably trained to take a JSON stringified arguments value.
                    // It's hard to constrain that for now (while reusing the JSON schema conversion), so we're just expecting a plain object.
                    {"name", {
                        {"type", "string"},
                        {"const", function.at("name")},
                    }},
                    {"arguments", function.at("parameters")},
                    {"id", {
                        {"type", "string"},
                        // Nemo's template expects a 9-character alphanumeric ID.
                        {"pattern", "^[a-zA-Z0-9]{9}$"},
                    }},
                }},
                {"required", json::array({"name", "arguments", "id"})},
            });
        });
        auto schema = json {
            {"type", "array"},
            {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
            {"minItems", 1},
        };
        if (!inputs.parallel_tool_calls) {
            schema["maxItems"] = 1;
        }
        builder.add_rule("root", "\"[TOOL_CALLS]\" " + builder.add_schema("tool_calls", schema));
    });
    data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
    data.preserved_tokens = {
        "[TOOL_CALLS]",
    };
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MISTRAL_NEMO;
    return data;
}


// Case-insensitive find
static size_t ifind_string(const std::string & haystack, const std::string & needle, size_t pos = 0) {
    auto it = std::search(
        haystack.begin() + pos, haystack.end(),
        needle.begin(), needle.end(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); }
    );
    return (it == haystack.end()) ? std::string::npos : std::distance(haystack.begin(), it);
}

static common_chat_params common_chat_params_init_lfm2(const common_chat_template & tmpl, const struct templates_params & inputs) {
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
    //
    // Example of the pytorch dialog structure:
    //     <|startoftext|><|im_start|>system
    //     List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
    //     <|im_start|>user
    //     What is the current status of candidate ID 12345?<|im_end|>
    //     <|im_start|>assistant
    //     <|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
    //     <|im_start|>tool
    //     <|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}<|tool_response_end|><|im_end|>
    //     <|im_start|>assistant
    //     The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
    //
    // For the llama server compatibility with json tools semantic,
    // the client can add "Follow json schema." line into the system message prompt to force the json output.
    //
    if (are_tools_provided && (is_json_schema_provided || is_grammar_provided)) {
        // server/utils.hpp prohibits that branch for the custom grammar anyways
        throw std::runtime_error("Tools call must not use \"json_schema\" or \"grammar\", use non-tool invocation if you want to use custom grammar");
    } else if (are_tools_provided && replace_json_schema_marker(tweaked_messages)) {
        LOG_INF("%s: Using tools to build a grammar\n", __func__);

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", function.at("name")},
                        }},
                        {"arguments", function.at("parameters")},
                    }},
                    {"required", json::array({"name", "arguments", "id"})},
                });
            });
            auto schema = json {
                {"type", "array"},
                {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                {"minItems", 1},
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }

            builder.add_rule("root", "\"<|tool_call_start|>\"" + builder.add_schema("tool_calls", schema) + "\"<|tool_call_end|>\"");
        });
        // model has no concept of tool selection mode choice,
        // if the system prompt rendered correctly it will produce a tool call
        // the grammar goes inside the tool call body
        data.grammar_lazy = true;
        data.grammar_triggers = {{COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "\\s*<\\|tool_call_start\\|>\\s*\\["}};
        data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};
        data.format = COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS;
    } else if (are_tools_provided && (!is_json_schema_provided && !is_grammar_provided)) {
        LOG_INF("%s: Using tools without json schema or grammar\n", __func__);
        // output those tokens
        data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};
    } else if (is_json_schema_provided) {
        LOG_INF("%s: Using provided json schema to build a grammar\n", __func__);
        data.grammar = json_schema_to_grammar(inputs.json_schema);
    } else if (is_grammar_provided) {
        LOG_INF("%s: Using provided grammar\n", __func__);
        data.grammar = inputs.grammar;
    } else {
        LOG_INF("%s: Using content relying on the template\n", __func__);
    }

    data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);
    LOG_DBG("%s: Prompt: %s\n", __func__, data.prompt.c_str());

    return data;
}

static common_chat_params common_chat_params_init_magistral(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MAGISTRAL;
    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
    };

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", function.at("name")},
                        }},
                        {"arguments", function.at("parameters")},
                        {"id", {
                            {"type", "string"},
                            {"pattern", "^[a-zA-Z0-9]{9}$"},
                        }},
                    }},
                    {"required", json::array({"name", "arguments", "id"})},
                });
            });
            auto schema = json {
                {"type", "array"},
                {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                {"minItems", 1},
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root", "\"[TOOL_CALLS]\" " + builder.add_schema("tool_calls", schema));
        });
        data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
        data.preserved_tokens.push_back("[TOOL_CALLS]");
    } else {
        data.grammar_lazy = false;
        if (!inputs.json_schema.is_null()) {
            if (!inputs.grammar.empty()) {
                throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
            }
            data.grammar = json_schema_to_grammar(inputs.json_schema);
        } else {
            data.grammar = inputs.grammar;
        }
    }

    return data;
}

static void common_chat_parse_mistral_nemo(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex prefix(regex_escape("[TOOL_CALLS]"));
    parse_prefixed_json_tool_call_array(builder, prefix);
}

static void common_chat_parse_magistral(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("[THINK]", "[/THINK]");

    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex prefix(regex_escape("[TOOL_CALLS]"));
    parse_prefixed_json_tool_call_array(builder, prefix);
}

static common_chat_params common_chat_params_init_command_r7b(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto has_reasoning_content = msg.contains("reasoning_content") && msg.at("reasoning_content").is_string();
        auto has_tool_calls = msg.contains("tool_calls") && msg.at("tool_calls").is_array();
        if (has_reasoning_content && has_tool_calls) {
            auto adjusted_message = msg;
            adjusted_message["tool_plan"] = msg.at("reasoning_content");
            adjusted_message.erase("reasoning_content");
            adjusted_messages.push_back(adjusted_message);
        } else {
            adjusted_messages.push_back(msg);
        }
    }
    data.prompt = apply(tmpl, inputs, /* messages_override= */ adjusted_messages);
    data.format = COMMON_CHAT_FORMAT_COMMAND_R7B;
    if (string_ends_with(data.prompt, "<|START_THINKING|>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "<|END_THINKING|>";
        } else {
            data.thinking_forced_open = true;
        }
    } else if (!inputs.enable_thinking && string_ends_with(data.prompt, "<|CHATBOT_TOKEN|>")) {
        data.prompt += "<|START_THINKING|><|END_THINKING|>";
    }

    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        auto schemas = json::array();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            schemas.push_back({
                {"type", "object"},
                {"properties", {
                    {"tool_call_id", {
                        {"type", "string"},
                        // Command-R's template expects an integer string.
                        {"pattern", "^[0-9]{1,10}$"},
                    }},
                    {"tool_name", {
                        {"type", "string"},
                        {"const", function.at("name")},
                    }},
                    {"parameters", function.at("parameters")},
                }},
                {"required", json::array({"tool_call_id", "tool_name", "parameters"})},
            });
        });
        auto schema = json {
            {"type", "array"},
            {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
            {"minItems", 1},
        };
        if (!inputs.parallel_tool_calls) {
            schema["maxItems"] = 1;
        }
        builder.add_rule("root",
            std::string(data.thinking_forced_open ? "( \"<|END_THINKING|>\" space )? " : "") +
            "\"<|START_ACTION|>\" " + builder.add_schema("tool_calls", schema) + " \"<|END_ACTION|>\"");
    });
    data.grammar_triggers.push_back({
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
        // If thinking_forced_open, then we capture the </think> tag in the grammar,
        // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
        std::string(data.thinking_forced_open ? "[\\s\\S]*?(<\\|END_THINKING\\|>\\s*)" : "(?:<\\|START_THINKING\\|>[\\s\\S]*?<\\|END_THINKING\\|>\\s*)?") +
            "(<\\|START_ACTION\\|>)[\\s\\S]*"
    });
    data.preserved_tokens = {
        "<|START_ACTION|>",
        "<|END_ACTION|>",
        "<|START_RESPONSE|>",
        "<|END_RESPONSE|>",
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    };
    return data;
}

static void common_chat_parse_command_r7b(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<|START_THINKING|>", "<|END_THINKING|>");

    static const common_regex start_action_regex("<\\|START_ACTION\\|>");
    static const common_regex end_action_regex("<\\|END_ACTION\\|>");
    static const common_regex start_response_regex("<\\|START_RESPONSE\\|>");
    static const common_regex end_response_regex("<\\|END_RESPONSE\\|>");

    if (auto res = builder.try_find_regex(start_action_regex)) {
        // If we didn't extract thoughts, prelude includes them.
        auto tool_calls = builder.consume_json_with_dumped_args({{"parameters"}});
        for (const auto & tool_call : tool_calls.value) {
            std::string name = tool_call.contains("tool_name") ? tool_call.at("tool_name") : "";
            std::string id = tool_call.contains("tool_call_id") ? tool_call.at("tool_call_id") : "";
            std::string arguments = tool_call.contains("parameters") ? tool_call.at("parameters") : "";
            if (!builder.add_tool_call(name, id, arguments) || tool_calls.is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
        if (tool_calls.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call");
        }
        builder.consume_regex(end_action_regex);
    } else if (auto res = builder.try_find_regex(start_response_regex)) {
        if (!builder.try_find_regex(end_response_regex)) {
            builder.add_content(builder.consume_rest());
            throw common_chat_msg_partial_exception(end_response_regex.str());
        }
    } else {
        builder.add_content(builder.consume_rest());
    }
}

static void expect_tool_parameters(const std::string & name, const json & parameters, const std::vector<std::string> & expected_properties) {
    if (!parameters.is_object() || !parameters.contains("type") || parameters.at("type") != "object" || !parameters.contains("properties") || !parameters.contains("required")) {
        throw std::runtime_error("Parameters of tool " + name + " must be an object w/ required properties");
    }
    const auto & parameters_properties = parameters.at("properties");
    const auto & parameters_required = parameters.at("required");
    for (const auto & prop : expected_properties) {
        if (!parameters_properties.contains(prop)) {
            throw std::runtime_error("Parameters of tool " + name + " is missing property: " + prop); // NOLINT
        }
        if (std::find(parameters_required.begin(), parameters_required.end(), json(prop)) == parameters_required.end()) {
            throw std::runtime_error("Parameters of tool " + name + " must have property marked as required: " + prop); // NOLINT
        }
    }
    if (parameters_properties.size() != expected_properties.size()) {
        throw std::runtime_error("Parameters of tool " + name + " must only have these properties:" + string_join(expected_properties, ", "));
    }
}

static common_chat_params common_chat_params_init_llama_3_x(const common_chat_template & tmpl, const struct templates_params & inputs, bool allow_python_tag_builtin_tools) {
    auto builtin_tools = json::array();
    common_chat_params data;
    if (!inputs.tools.is_null()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;

            auto handle_builtin_tool = [&](const std::string & name, const json & parameters) {
                if (name == "wolfram_alpha" || name == "web_search" || name == "brave_search") {
                    // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/tool_runtime/wolfram_alpha/wolfram_alpha.py
                    // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/tool_runtime/brave_search/brave_search.py
                    expect_tool_parameters(name, parameters, {"query"});
                } else if (name == "python" || name == "code_interpreter") {
                    // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/inline/tool_runtime/code_interpreter/code_interpreter.py
                    expect_tool_parameters(name, parameters, {"code"});
                } else {
                    return false;
                }

                std::vector<std::string> kvs;
                for (const auto & [key, value] : parameters.at("properties").items()) {
                    kvs.push_back("\"" + key + "=\" " + builder.add_schema(name + "-args-" + key, value)); // NOLINT
                }

                tool_rules.push_back(
                    builder.add_rule(
                        name + "-call",
                        "\"<|python_tag|>" + name + ".call(\" " + string_join(kvs, " \", \" ") + " \")\""));
                builtin_tools.push_back(name);

                return true;
            };

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                // https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/remote/tool_runtime
                if (allow_python_tag_builtin_tools) {
                    handle_builtin_tool(name, parameters);
                }
                tool_rules.push_back(
                    builder.add_rule(
                        name + "-call",
                        "\"{\" space "
                        "( \"\\\"type\\\"\"       space \":\" space \"\\\"function\\\"\"     space \",\" space )? "
                        "  \"\\\"name\\\"\"       space \":\" space \"\\\"" + name + "\\\"\" space \",\" space "
                        "  \"\\\"parameters\\\"\" space \":\" space " + builder.add_schema(name + "-args", parameters) + " "
                        "\"}\" space"));
            });
            // Small models may hallucinate function names so we match anything (*at the start*) that looks like the JSON of a function call, regardless of the name.
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                "(\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\")[\\s\\S]*", // + name + "\"[\\s\\S]*",
            });
            if (!builtin_tools.empty()) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|python_tag|>"});
                data.preserved_tokens.push_back("<|python_tag|>");
            }
            // Allow a few empty lines on top of the usual constrained json schema space rule.
            builder.add_rule("root", string_join(tool_rules, " | "));
            data.additional_stops.push_back("<|eom_id|>");
        });
        data.format = allow_python_tag_builtin_tools && !builtin_tools.empty()
            ? COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS
            : COMMON_CHAT_FORMAT_LLAMA_3_X;
    } else {
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }
    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, /* tools_override= */ std::nullopt, json {
        {"date_string", format_time(inputs.now, "%d %b %Y")},
        {"tools_in_user_message", false},
        {"builtin_tools", builtin_tools.empty() ? json() : builtin_tools},
    });
    return data;
}

static common_chat_params common_chat_params_init_nemotron_v2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Generate the prompt using the apply() function with the template
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_NEMOTRON_V2;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    // When tools are present, build grammar for the <TOOLCALL> format, similar to CommandR, but without tool call ID
    if (!inputs.tools.is_null() && inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = true;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    { "type",       "object"                                                   },
                    { "properties",
                        {
                            { "name",
                            {
                                { "type", "string" },
                                { "const", function.at("name") },
                            } },
                            { "arguments", function.at("parameters") },
                        }                                                                        },
                    { "required",   json::array({ "name", "arguments" }) },
                });
            });
            auto schema = json{
                        { "type",     "array"                                                         },
                        { "items",    schemas.size() == 1 ? schemas[0] : json{ { "anyOf", schemas } } },
                        { "minItems", 1                                                               },
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root",
                                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                                    "\"<TOOLCALL>\" " + builder.add_schema("tool_calls", schema) +
                                    " \"</TOOLCALL>\"");
        });
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
            // If thinking_forced_open, then we capture the </think> tag in the grammar,
            // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
            std::string(data.thinking_forced_open ?
                            "[\\s\\S]*?(</think>\\s*)" :
                            "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                "(<TOOLCALL>)[\\s\\S]*" });
    }
    return data;
}

static common_chat_params common_chat_params_init_apertus(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Generate the prompt using the apply() function with the template
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_APERTUS;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<|inner_prefix|>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "<|inner_suffix|>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    // When tools are present, build grammar for the <|tools_prefix|> format
    if (!inputs.tools.is_null() && inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = true;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    { "type",       "object"                                                   },
                    { "properties",
                        {
                            { function.at("name"), function.at("parameters") }
                        }                                                                        },
                    { "required",   json::array({ function.at("name") }) },
                });
            });
            auto schema = json{
                        { "type",     "array"                                                         },
                        { "items",    schemas.size() == 1 ? schemas[0] : json{ { "anyOf", schemas } } },
                        { "minItems", 1                                                               },
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root",
                                std::string(data.thinking_forced_open ? "( \"<|inner_suffix|>\" space )? " : "") +
                                    "\"<|tools_prefix|>\"" + builder.add_schema("tool_calls", schema) + "\"<|tools_suffix|>\"");
                            });
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
            // If thinking_forced_open, then we capture the <|inner_suffix|> tag in the grammar,
            // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
            std::string(data.thinking_forced_open ?
                            "[\\s\\S]*?(<\\|inner_suffix\\|>\\s*)" :
                            "(?:<\\|inner_prefix\\|>[\\s\\S]*?<\\|inner_suffix\\|>\\s*)?") +
                "(<\\|tools_prefix\\|>)[\\s\\S]*" });
        data.preserved_tokens = {
            "<|system_start|>",
            "<|system_end|>",
            "<|developer_start|>",
            "<|developer_end|>",
            "<|user_start|>",
            "<|user_end|>",
            "<|assistant_start|>",
            "<|assistant_end|>",
            "<|inner_prefix|>",
            "<|inner_suffix|>",
            "<|tools_prefix|>",
            "<|tools_suffix|>",
        };
    }
    return data;
}
static void common_chat_parse_llama_3_1(common_chat_msg_parser & builder, bool with_builtin_tools = false) {
    builder.try_parse_reasoning("<think>", "</think>");

    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex function_regex(
        "\\s*\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"parameters\"\\s*: ");
    static const common_regex close_regex("\\}\\s*");

    static const common_regex function_name_regex("\\s*(\\w+)\\s*\\.\\s*call\\(");
    static const common_regex arg_name_regex("\\s*(\\w+)\\s*=\\s*");

    if (with_builtin_tools) {
        static const common_regex builtin_call_regex("<\\|python_tag\\|>");
        if (auto res = builder.try_find_regex(builtin_call_regex)) {
            auto fun_res = builder.consume_regex(function_name_regex);
            auto function_name = builder.str(fun_res.groups[1]);

            common_healing_marker healing_marker;
            json args = json::object();
            while (true) {
                if (auto arg_res = builder.try_consume_regex(arg_name_regex)) {
                    auto arg_name = builder.str(arg_res->groups[1]);
                    auto partial = builder.consume_json();
                    args[arg_name] = partial.json;
                    healing_marker.marker = partial.healing_marker.marker;
                    healing_marker.json_dump_marker = partial.healing_marker.json_dump_marker;
                    builder.consume_spaces();
                    if (!builder.try_consume_literal(",")) {
                        break;
                    }
                } else {
                    break;
                }
            }
            builder.consume_literal(")");
            builder.consume_spaces();

            auto arguments = args.dump();
            if (!builder.add_tool_call(function_name, "", arguments)) {
                throw common_chat_msg_partial_exception("Incomplete tool call");
            }
            return;
        }
    }
    parse_json_tool_calls(
        builder,
        /* block_open= */ std::nullopt,
        /* function_regex_start_only= */ function_regex,
        /* function_regex= */ std::nullopt,
        close_regex,
        std::nullopt);

}

static common_chat_params common_chat_params_init_deepseek_r1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    auto prompt = apply(tmpl, inputs);

    // Hacks to fix the official (broken) prompt.
    // It is advisable to use --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja instead,
    // until the official template is fixed.
    if (tmpl.source().find("{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}") != std::string::npos) {
        // Don't leave the chat dangling after tool results
        if (string_ends_with(prompt, "<｜tool▁outputs▁end｜>")) {
            prompt += "<｜end▁of▁sentence｜>";
            if (inputs.add_generation_prompt) {
                prompt += "<｜Assistant｜>";
            }
        }
        // Fix up tool call delta example added by Minja
        prompt = std::regex_replace(
            prompt,
            std::regex("(<｜tool▁call▁end｜>)[\\s\\r\\n]*(<｜tool▁outputs▁begin｜>|<｜User｜>)"),
            "$1<｜tool▁calls▁end｜><｜end▁of▁sentence｜>$2");
    }
    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_DEEPSEEK_R1;
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED && inputs.json_schema.is_null();
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_rule(name + "-call",
                    "( \"<｜tool▁call▁begin｜>\" )? \"function<｜tool▁sep｜>" + name + "\\n"
                    "```json\\n\" " + builder.add_schema(name + "-args", parameters) + " "
                    "\"```<｜tool▁call▁end｜>\""));
            });
            // Distill Qwen 7B & 32B models seem confused re/ syntax of their tool call opening tag,
            // so we accept common variants (then it's all constrained)
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                "( \"<｜tool▁calls▁begin｜>\" | \"<｜tool_calls_begin｜>\" | \"<｜tool calls begin｜>\" | \"<｜tool\\\\_calls\\\\_begin｜>\" | \"<｜tool▁calls｜>\" ) "
                "(" + string_join(tool_rules, " | ") + ")" + (inputs.parallel_tool_calls ? "*" : "") + " "
                "\"<｜tool▁calls▁end｜>\""
                " space");
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                // If thinking_forced_open, then we capture the </think> tag in the grammar,
                // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                    "(<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)[\\s\\S]*"
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁call▁begin｜>",
                "<｜tool▁sep｜>",
                "<｜tool▁call▁end｜>",
                "<｜tool▁calls▁end｜",
            };
        });
    }
    return data;
}

static common_chat_params common_chat_params_init_deepseek_v3_1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Pass thinking context for DeepSeek V3.1 template
    json additional_context = {
        {"thinking", inputs.enable_thinking},
    };

    auto prompt = apply(tmpl, inputs,
                       /* messages_override= */ inputs.messages,
                       /* tools_override= */ std::nullopt,
                       additional_context);
    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_DEEPSEEK_V3_1;
    if (string_ends_with(data.prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED && inputs.json_schema.is_null();
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_rule(name + "-call",
                    "( \"<｜tool▁call▁begin｜>\" )? \"" + name + "<｜tool▁sep｜>"
                    "\" " + builder.add_schema(name + "-args", parameters) + " "
                    "\"<｜tool▁call▁end｜>\""));
            });
            // Distill Qwen 7B & 32B models seem confused re/ syntax of their tool call opening tag,
            // so we accept common variants (then it's all constrained)
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                "( \"<｜tool▁calls▁begin｜>\" | \"<｜tool_calls_begin｜>\" | \"<｜tool calls begin｜>\" | \"<｜tool\\\\_calls\\\\_begin｜>\" | \"<｜tool▁calls｜>\" ) "
                "(" + string_join(tool_rules, " | ") + ")" + (inputs.parallel_tool_calls ? "*" : "") + " "
                "\"<｜tool▁calls▁end｜>\""
                " space");
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                // If thinking_forced_open, then we capture the </think> tag in the grammar,
                // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                    "(<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)[\\s\\S]*"
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁call▁begin｜>",
                "<｜tool▁sep｜>",
                "<｜tool▁call▁end｜>",
                "<｜tool▁calls▁end｜>",
            };
        });
    }
    return data;
}

static void common_chat_parse_deepseek_r1(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex tool_calls_begin("(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)");
    static const common_regex tool_calls_end("<｜tool▁calls▁end｜>");
    static const common_regex function_regex("(?:<｜tool▁call▁begin｜>)?function<｜tool▁sep｜>([^\n]+)\n```json\n");
    static const common_regex close_regex("```[\\s\\r\\n]*<｜tool▁call▁end｜>");

    parse_json_tool_calls(
        builder,
        /* block_open= */ tool_calls_begin,
        /* function_regex_start_only= */ std::nullopt,
        function_regex,
        close_regex,
        tool_calls_end);
}

static void common_chat_parse_deepseek_v3_1_content(common_chat_msg_parser & builder) {
    static const common_regex function_regex("(?:<｜tool▁call▁begin｜>)?([^\\n<]+)(?:<｜tool▁sep｜>)");

    static const common_regex close_regex("(?:[\\s]*)?<｜tool▁call▁end｜>");
    static const common_regex tool_calls_begin("(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)");
    static const common_regex tool_calls_end("<｜tool▁calls▁end｜>");

    if (!builder.syntax().parse_tool_calls) {
        LOG_DBG("%s: not parse_tool_calls\n", __func__);
        builder.add_content(builder.consume_rest());
        return;
    }

    LOG_DBG("%s: parse_tool_calls\n", __func__);

    parse_json_tool_calls(
        builder,
        /* block_open= */ tool_calls_begin,
        /* function_regex_start_only= */ std::nullopt,
        function_regex,
        close_regex,
        tool_calls_end);
}

static void common_chat_parse_deepseek_v3_1(common_chat_msg_parser & builder) {
    // DeepSeek V3.1 outputs reasoning content between "<think>" and "</think>" tags, followed by regular content
    // First try to parse using the standard reasoning parsing method
    LOG_DBG("%s: thinking_forced_open: %s\n", __func__, std::to_string(builder.syntax().thinking_forced_open).c_str());

    auto start_pos = builder.pos();
    auto found_end_think = builder.try_find_literal("</think>");
    builder.move_to(start_pos);

    if (builder.syntax().thinking_forced_open && !builder.is_partial() && !found_end_think) {
        LOG_DBG("%s: no end_think, not partial, adding content\n", __func__);
        common_chat_parse_deepseek_v3_1_content(builder);
    } else if (builder.try_parse_reasoning("<think>", "</think>")) {
        // If reasoning was parsed successfully, the remaining content is regular content
        LOG_DBG("%s: parsed reasoning, adding content\n", __func__);
        // </think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\nJSON\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
        common_chat_parse_deepseek_v3_1_content(builder);
    } else {
        if (builder.syntax().reasoning_format == COMMON_REASONING_FORMAT_NONE) {
          LOG_DBG("%s: reasoning_format none, adding content\n", __func__);
          common_chat_parse_deepseek_v3_1_content(builder);
          return;
        }
        // If no reasoning tags found, check if we should treat everything as reasoning
        if (builder.syntax().thinking_forced_open) {
            // If thinking is forced open but no tags found, treat everything as reasoning
            LOG_DBG("%s: thinking_forced_open, adding reasoning content\n", __func__);
            builder.add_reasoning_content(builder.consume_rest());
        } else {
            LOG_DBG("%s: no thinking_forced_open, adding content\n", __func__);
            // <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>JSON<｜tool▁call▁end｜>
            common_chat_parse_deepseek_v3_1_content(builder);
        }
    }
}


static common_chat_params common_chat_params_init_minimax_m2(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_MINIMAX_M2;

    // Handle thinking tags based on prompt ending
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!params.enable_thinking) {
            // Close the thinking tag immediately if thinking is disabled
            data.prompt += "</think>\n\n";
        } else {
            // Mark thinking as forced open (template started with <think>)
            data.thinking_forced_open = true;
        }
    }

    // Preserve MiniMax-M2 special tokens
    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<minimax:tool_call>",
        "</minimax:tool_call>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<minimax:tool_call>\n",
        /* form.tool_start  = */ "<invoke name=\"",
        /* form.tool_sep    = */ "\">\n",
        /* form.key_start   = */ "<parameter name=\"",
        /* form.key_val_sep = */ "\">",
        /* form.val_end     = */ "</parameter>\n",
        /* form.tool_end    = */ "</invoke>\n",
        /* form.scope_end   = */ "</minimax:tool_call>",
    };
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static void common_chat_parse_minimax_m2(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<minimax:tool_call>",
        /* form.tool_start  = */ "<invoke name=\"",
        /* form.tool_sep    = */ "\">",
        /* form.key_start   = */ "<parameter name=\"",
        /* form.key_val_sep = */ "\">",
        /* form.val_end     = */ "</parameter>",
        /* form.tool_end    = */ "</invoke>",
        /* form.scope_end   = */ "</minimax:tool_call>",
    };
    builder.consume_reasoning_with_xml_tool_calls(form, "<think>", "</think>");
}

static common_chat_params common_chat_params_init_qwen3_coder_xml(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_QWEN3_CODER_XML;

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<tool_call>\n",
        /* form.tool_start  = */ "<function=",
        /* form.tool_sep    = */ ">\n",
        /* form.key_start   = */ "<parameter=",
        /* form.key_val_sep = */ ">\n",
        /* form.val_end     = */ "\n</parameter>\n",
        /* form.tool_end    = */ "</function>\n",
        /* form.scope_end   = */ "</tool_call>",
    };
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static void common_chat_parse_qwen3_coder_xml(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<tool_call>";
        form.tool_start  = "<function=";
        form.tool_sep    = ">";
        form.key_start   = "<parameter=";
        form.key_val_sep = ">";
        form.val_end     = "</parameter>";
        form.tool_end    = "</function>";
        form.scope_end   = "</tool_call>";
        form.trim_raw_argval = true;
        return form;
    })();
    builder.consume_reasoning_with_xml_tool_calls(form);
}

static common_chat_params common_chat_params_init_kimi_k2(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_KIMI_K2;

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
        "<|im_end|>",
        "<|im_system|>",
        "<|im_middle|>",
    };

    data.additional_stops.insert(data.additional_stops.end(), {
        "<|im_end|>",
        "<|im_middle|>"
    });
    // build grammar for tool call
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<|tool_calls_section_begin|>";
        form.tool_start  = "<|tool_call_begin|>";
        form.tool_sep    = "<|tool_call_argument_begin|>{";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}<|tool_call_end|>";
        form.scope_end   = "<|tool_calls_section_end|>";
        form.raw_argval  = false;
        form.last_val_end = "";
        return form;
    })();
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static void common_chat_parse_kimi_k2(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<|tool_calls_section_begin|>";
        form.tool_start  = "<|tool_call_begin|>";
        form.tool_sep    = "<|tool_call_argument_begin|>{";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}<|tool_call_end|>";
        form.scope_end   = "<|tool_calls_section_end|>";
        form.raw_argval  = false;
        form.last_val_end = "";
        return form;
    })();
    builder.consume_reasoning_with_xml_tool_calls(form, "<think>", "</think>");
}

static common_chat_params common_chat_params_init_apriel_1_5(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_APRIEL_1_5;

    data.preserved_tokens = {
        "<thinking>",
        "</thinking>",
        "<tool_calls>",
        "</tool_calls>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<tool_calls>[";
        form.tool_start  = "{\"name\": \"";
        form.tool_sep    = "\", \"arguments\": {";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}, ";
        form.scope_end   = "]</tool_calls>";
        form.raw_argval  = false;
        form.last_val_end = "";
        form.last_tool_end = "}";
        return form;
    })();
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static void common_chat_parse_apriel_1_5(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<tool_calls>[";
        form.tool_start  = "{\"name\": \"";
        form.tool_sep    = "\", \"arguments\": {";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}, ";
        form.scope_end   = "]</tool_calls>";
        form.raw_argval  = false;
        form.last_val_end = "";
        form.last_tool_end = "}";
        return form;
    })();
    builder.consume_reasoning_with_xml_tool_calls(form, "<thinking>", "</thinking>");
}

static common_chat_params common_chat_params_init_xiaomi_mimo(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_XIAOMI_MIMO;

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "\n";
        form.tool_start  = "<tool_call>\n{\"name\": \"";
        form.tool_sep    = "\", \"arguments\": {";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}\n</tool_call>";
        form.scope_end   = "";
        form.raw_argval  = false;
        form.last_val_end = "";
        return form;
    })();
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static void common_chat_parse_xiaomi_mimo(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "";
        form.tool_start  = "<tool_call>\n{\"name\": \"";
        form.tool_sep    = "\", \"arguments\": {";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}\n</tool_call>";
        form.scope_end   = "";
        form.raw_argval  = false;
        form.last_val_end = "";
        return form;
    })();
    builder.consume_reasoning_with_xml_tool_calls(form);
}

static common_chat_params common_chat_params_init_gpt_oss(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Copy reasoning to the "thinking" field as expected by the gpt-oss template
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto has_reasoning_content = msg.contains("reasoning_content") && msg.at("reasoning_content").is_string();
        auto has_tool_calls = msg.contains("tool_calls") && msg.at("tool_calls").is_array();

        if (has_reasoning_content && has_tool_calls) {
            auto adjusted_message = msg;
            adjusted_message["thinking"] = msg.at("reasoning_content");
            adjusted_messages.push_back(adjusted_message);
        } else {
            adjusted_messages.push_back(msg);
        }
    }

    auto prompt = apply(tmpl, inputs, /* messages_override= */ adjusted_messages);

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

    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_GPT_OSS;

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = {
        "<|channel|>",
        "<|constrain|>",
        "<|message|>",
        "<|start|>",
        "<|end|>",
    };

    if (!inputs.json_schema.is_null()) {
        data.grammar_lazy = false;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schema = inputs.json_schema;
            builder.resolve_refs(schema);

            auto not_end = builder.add_rule("not-end",
                "[^<] | \"<\" [^|] | \"<|\" [^e] | \"<|e\" [^n] | \"<|en\" [^d] | \"<|end\" [^|] | \"<|end|\" [^>]");
            auto analysis = builder.add_rule("analysis",
                "\"<|channel|>analysis<|message|>\" ( " + not_end + " )* \"<|end|>\"");
            auto constraint = builder.add_rule("constraint", "\"<|constrain|>\"? [a-zA-Z0-9_-]+");
            auto final = builder.add_rule("final",
                "\"<|channel|>final\" ( \" \" " + constraint + " )? \"<|message|>\" " +
                builder.add_schema("response", schema)
            );

            builder.add_rule("root", "( " + analysis + " \"<|start|>assistant\" )? " + final);
        });
    }

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            // tool calls can appear in commentary or analysis channels
            auto channel = builder.add_rule("channel", "\"<|channel|>\" ( \"commentary\" | \"analysis\" )");

            std::vector<std::string> tool_rules_recipient_in_role;
            std::vector<std::string> tool_rules_recipient_in_channel;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                tool_rules_recipient_in_role.push_back(
                    builder.add_rule(name + "-call",
                        "\"" + name + "\"" + channel + " \" <|constrain|>json\"? \"<|message|>\" " +
                        builder.add_schema(name + "-args", parameters)
                    )
                );

                tool_rules_recipient_in_channel.push_back(
                    builder.add_rule(name + "-call",
                        "\"" + name + "\"" + " \" <|constrain|>json\"? \"<|message|>\" " +
                        builder.add_schema(name + "-args", parameters)
                    )
                );
            });

            auto recipient_in_channel = builder.add_rule("recipient_in_channel",
                channel + " \" to=functions.\" ( " +
                string_join(tool_rules_recipient_in_channel, " | ") + " )"
            );

            if (data.grammar_lazy) {
                auto recipient_in_role = builder.add_rule("recipient_in_role",
                    "\"<|start|>assistant\"? \" to=functions.\" ( " +
                    string_join(tool_rules_recipient_in_role, " | ") + " )"
                );

                builder.add_rule("root", recipient_in_role + " | " + recipient_in_channel);
            } else {
                auto not_end = builder.add_rule("not-end",
                    "[^<] | \"<\" [^|] | \"<|\" [^e] | \"<|e\" [^n] | \"<|en\" [^d] | \"<|end\" [^|] | \"<|end|\" [^>]");
                auto analysis = builder.add_rule("analysis",
                    "\"<|channel|>analysis<|message|>\" ( " + not_end + " )* \"<|end|>\"");
                auto commentary = builder.add_rule("commentary",
                    "\"<|channel|>commentary<|message|>\" ( " + not_end + " )* \"<|end|>\"");

                auto recipient_in_role = builder.add_rule("recipient_in_role",
                    "\" to=functions.\" ( " + string_join(tool_rules_recipient_in_role, " | ") + " )"
                );

                builder.add_rule("root",
                    "( " + analysis + " \"<|start|>assistant\" )? " +
                    "( " + commentary + " \"<|start|>assistant\" )? " +
                    "( " + recipient_in_role + " | " + recipient_in_channel + " )"
                );
            }

            // Trigger on tool calls that appear in the commentary channel
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                "<\\|channel\\|>(commentary|analysis) to"
            });

            // Trigger tool calls that appear in the role section, either at the
            // start or in the middle.
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                "^ to"
            });

            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                "<\\|start\\|>assistant to"
            });
        });
    }

    return data;
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
        if (auto args = builder.try_consume_json_with_dumped_args({{}})) {
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
        auto content_start = builder.try_find_literal("<|message|>");
        if (!content_start) {
            throw common_chat_msg_partial_exception("incomplete header");
        }

        auto header = content_start->prelude;

        if (auto match = regex_match(tool_call1_regex, header)) {
            auto group = match->groups[1];
            auto name = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        if (auto match = regex_match(tool_call2_regex, header)) {
            auto group = match->groups[2];
            auto name = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        if (regex_match(analysis_regex, header)) {
            builder.move_to(header_start_pos);
            if (builder.syntax().reasoning_format == COMMON_REASONING_FORMAT_NONE || builder.syntax().reasoning_in_content) {
                builder.add_content(consume_end(true));
            } else {
                builder.try_parse_reasoning("<|channel|>analysis<|message|>", "<|end|>");
            }
            continue;
        }

        if(regex_match(final_regex, header) || regex_match(preamble_regex, header)) {
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

static common_chat_params common_chat_params_init_glm_4_5(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.grammar_lazy = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    std::string prompt = apply(tmpl, inputs);

    // match the existing trimming behavior
    if (inputs.add_bos && string_starts_with(prompt, tmpl.bos_token())) {
        prompt.erase(0, tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(prompt, tmpl.eos_token())) {
        prompt.erase(prompt.size() - tmpl.eos_token().size());
    }
    if (string_ends_with(prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    // add GLM preserved tokens
    data.preserved_tokens = {
        "<|endoftext|>",
        "[MASK]",
        "[gMASK]",
        "[sMASK]",
        "<sop>",
        "<eop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
        "<|begin_of_audio|>",
        "<|end_of_audio|>",
        "<|begin_of_transcription|>",
        "<|end_of_transcription|>",
        "<|code_prefix|>",
        "<|code_middle|>",
        "<|code_suffix|>",
        "/nothink",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>"
    };

    // extra GLM 4.5 stop word
    data.additional_stops.insert(data.additional_stops.end(), {
        "<|user|>",
        "<|observation|>"
    });

    // build grammar for tool call
    static const xml_tool_call_format form {
        /* form.scope_start = */ "",
        /* form.tool_start  = */ "\n<tool_call>",
        /* form.tool_sep    = */ "\n",
        /* form.key_start   = */ "<arg_key>",
        /* form.key_val_sep = */ "</arg_key>\n<arg_value>",
        /* form.val_end     = */ "</arg_value>\n",
        /* form.tool_end    = */ "</tool_call>\n",
        /* form.scope_end   = */ "",
    };
    build_grammar_xml_tool_call(data, inputs.tools, form);

    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_GLM_4_5;
    return data;
}

static void common_chat_parse_glm_4_5(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form {
        /* form.scope_start  = */ "",
        /* form.tool_start   = */ "<tool_call>",
        /* form.tool_sep     = */ "",
        /* form.key_start    = */ "<arg_key>",
        /* form.key_val_sep  = */ "</arg_key>",
        /* form.val_end      = */ "</arg_value>",
        /* form.tool_end     = */ "</tool_call>",
        /* form.scope_end    = */ "",
        /* form.key_val_sep2 = */ "<arg_value>",
    };
    builder.consume_reasoning_with_xml_tool_calls(form, "<think>", "</think>");
}

static common_chat_params common_chat_params_init_firefunction_v2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    LOG_DBG("%s\n", __func__);
    common_chat_params data;
    const std::optional<json> tools_override = json();
    const std::optional<json> additional_context = json {
        {"datetime", format_time(inputs.now, "%b %d %Y %H:%M:%S GMT")},
        {"functions", json(inputs.tools.empty() ? "" : inputs.tools.dump(2))},
    };
    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, tools_override, additional_context);
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", function.at("name")},
                        }},
                        {"arguments", function.at("parameters")},
                    }},
                    {"required", json::array({"name", "arguments", "id"})},
                });
            });
            auto schema = json {
                {"type", "array"},
                {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                {"minItems", 1},
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root", "\" functools\"? " + builder.add_schema("tool_calls", schema));
        });
        data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, " functools["});
        data.preserved_tokens = {
            " functools[",
        };
        data.format = COMMON_CHAT_FORMAT_FIREFUNCTION_V2;
    } else {
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }
    return data;
}
static void common_chat_parse_firefunction_v2(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    static const common_regex prefix(regex_escape(" functools["));
    parse_prefixed_json_tool_call_array(builder, prefix, /* rstrip_prefix= */ 1);
}

static common_chat_params common_chat_params_init_functionary_v3_2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
    // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
    // If the function is python, we also allow raw python code (if the line after `python\n` doesn't start w/ opening `{`), which the model seems to prefer for multiline code.
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2;
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> first_tool_rules;
            std::vector<std::string> subsequent_tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                std::string args_pattern = "[\\s\\S]*";
                auto args_rule = builder.add_schema(name + "-args", parameters);
                if (name == "python") {
                    args_rule = builder.add_rule(name + "-maybe-raw-args", args_rule + " | [^{] .*");
                } else {
                    args_pattern = "\\{" + args_pattern;
                }
                auto call_rule = builder.add_rule(name + "-call", "\"" + name + "\\n\" " + args_rule);
                first_tool_rules.push_back(call_rule);
                if (inputs.parallel_tool_calls) {
                    subsequent_tool_rules.push_back(builder.add_rule(name + "-call2", "\">>>\" " + call_rule));
                }
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    "((?:[\\s\\S]+?>>>)?" + regex_escape(name) + "\n)" + args_pattern,
                });
            });
            data.preserved_tokens = {
                "<|end_header_id|>",
            };
            auto first_rule = first_tool_rules.empty() ? "" : builder.add_rule("first_tool_call", string_join(first_tool_rules, " | ")) + " space";
            if (inputs.parallel_tool_calls) {
                auto subsequent_rule = builder.add_rule("subsequent_tool_call", string_join(subsequent_tool_rules, " | ")) + " space";
                builder.add_rule("root", first_rule + " (" + subsequent_rule + ")*");
            } else {
                builder.add_rule("root", first_rule);
            }

        });
    }
    return data;
}
static void common_chat_parse_functionary_v3_2(common_chat_msg_parser & builder) {
    static const common_regex function_regex_start_only(R"((\w+\n\{|python\n|all\n))");
    static const common_regex function_regex(R"(>>>(\w+\n\{|python\n|all\n))");
    static const common_regex close_regex(R"(\s*)");

    parse_json_tool_calls(
        builder,
        std::nullopt,
        function_regex_start_only,
        function_regex,
        close_regex,
        std::nullopt,
        /* allow_raw_python= */ true,
        /* get_function_name= */ [&](const auto & res) -> std::string {
            auto at_start = res.groups[0].begin == 0;
            auto name = builder.str(res.groups[1]);
            if (!name.empty() && name.back() == '{') {
                // Unconsume the opening brace '{' to ensure the JSON parsing goes well.
                builder.move_back(1);
            }
            auto idx = name.find_last_not_of("\n{");
            name = name.substr(0, idx + 1);
            if (at_start && name == "all") {
                return "";
            }
            return name;
        });
}

static common_chat_params common_chat_params_init_functionary_v3_1_llama_3_1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
    common_chat_params data;

    if (!inputs.tools.is_null()) {
        std::string python_code_argument_name;
        auto has_raw_python = false;

        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                const auto & parameters = function.at("parameters");
                std::string name = function.at("name");
                if (name == "python" || name == "ipython") {
                    if (!parameters.contains("type")) {
                        throw std::runtime_error("Missing type in python tool");
                    }
                    has_raw_python = true;
                    const auto & type = parameters.at("type");
                    if (type == "object") {
                        auto properties = parameters.at("properties");
                        for (auto it = properties.begin(); it != properties.end(); ++it) {
                            if (it.value().at("type") == "string") {
                                if (!python_code_argument_name.empty()) {
                                    throw std::runtime_error("Multiple string arguments found in python tool");
                                }
                                python_code_argument_name = it.key();
                            }
                        }
                        if (python_code_argument_name.empty()) {
                            throw std::runtime_error("No string argument found in python tool");
                        }
                    } else if (type != "string") {
                        throw std::runtime_error("Invalid type in python tool: " + type.dump());
                    }
                }
                tool_rules.push_back(builder.add_rule(name + "-call", "\"<function=" + name + ">\" " + builder.add_schema(name + "-args", parameters) + " \"</function>\" space"));
            });
            if (has_raw_python) {
                tool_rules.push_back(builder.add_rule("python-call", "\"<|python_tag|>\" .*"));
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|python_tag|>"});
                data.preserved_tokens.push_back("<|python_tag|>");
            }
            auto tool_call = builder.add_rule("tool_call", string_join(tool_rules, " | ")) + " space";
            builder.add_rule("root", inputs.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<function="});
        });
        data.format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1;
    } else {
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }

    data.prompt = apply(tmpl, inputs);
    // TODO: if (has_raw_python)
    return data;
}
static void common_chat_parse_functionary_v3_1_llama_3_1(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    // This version of Functionary still supports the llama 3.1 tool call format for the python tool.
    static const common_regex python_tag_regex(regex_escape("<|python_tag|>"));

    static const common_regex function_regex(R"(<function=(\w+)>)");
    static const common_regex close_regex(R"(</function>)");

    parse_json_tool_calls(
        builder,
        /* block_open= */ std::nullopt,
        /* function_regex_start_only= */ std::nullopt,
        function_regex,
        close_regex,
        std::nullopt);

    if (auto res = builder.try_find_regex(python_tag_regex)) {
        auto arguments = wrap_code_as_arguments(builder, builder.consume_rest());
        builder.add_tool_call("python", "", arguments);
        return;
    }
}

static common_chat_params common_chat_params_init_hermes_2_pro(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    json extra_context = json {
        {"enable_thinking", inputs.enable_thinking},
    };
    extra_context.update(inputs.extra_context);

    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, /* tools_override= */ std::nullopt, extra_context);
    data.format = COMMON_CHAT_FORMAT_HERMES_2_PRO;
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!extra_context["enable_thinking"]) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (!inputs.tools.is_null()) {
        // (content)?(<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>)*
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            std::vector<std::string> tool_call_alts;
            std::vector<std::string> escaped_names;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_schema(name + "-call", {
                    {"type", "object"},
                    {"properties", json {
                        {"name", json {{"const", name}}},
                        {"arguments", parameters},
                    }},
                    {"required", json::array({"name", "arguments"})},
                }));
                tool_call_alts.push_back(builder.add_rule(
                    name + "-function-tag",
                    "\"<function\" ( \"=" + name + "\" | \" name=\\\"" + name + "\\\"\" ) \">\" space " +
                    builder.add_schema(name + "-args", parameters) + " "
                    "\"</function>\" space"));

                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                    "<function=" + name + ">",
                });
                auto escaped_name = regex_escape(name);
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                    "<function\\s+name\\s*=\\s*\"" + escaped_name + "\"",
                });
                escaped_names.push_back(escaped_name);
            });
            auto any_tool_call = builder.add_rule("any_tool_call", "( " + string_join(tool_rules, " | ") + " ) space");
            std::vector<std::string> alt_tags {
                any_tool_call,
                "\"<tool_call>\" space "     + any_tool_call + " \"</tool_call>\"",
                // The rest is just to accommodate common "good bad" outputs.
                "\"<function_call>\" space " + any_tool_call + " \"</function_call>\"",
                "\"<response>\"  space "     + any_tool_call + " \"</response>\"",
                "\"<tools>\"     space "     + any_tool_call + " \"</tools>\"",
                "\"<json>\"      space "     + any_tool_call + " \"</json>\"",
                "\"<xml>\"      space "     + any_tool_call + " \"</xml>\"",
                "\"<JSON>\"      space "     + any_tool_call + " \"</JSON>\"",
            };
            auto wrappable_tool_call = builder.add_rule("wrappable_tool_call", "( " + string_join(alt_tags, " | ") + " ) space");
            tool_call_alts.push_back(wrappable_tool_call);
            tool_call_alts.push_back(
                "( \"```\\n\" | \"```json\\n\" | \"```xml\\n\" ) space " + wrappable_tool_call + " space \"```\" space ");
            auto tool_call = builder.add_rule("tool_call", string_join(tool_call_alts, " | "));
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                (inputs.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call));
            // Trigger on some common known "good bad" outputs (only from the start and with a json that's about a specific argument name to avoid false positives)
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                // If thinking_forced_open, then we capture the </think> tag in the grammar,
                // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") + (
                    "\\s*("
                    "(?:<tool_call>"
                    "|<function"
                    "|(?:```(?:json|xml)?\n\\s*)?(?:<function_call>|<tools>|<xml><json>|<response>)?"
                    "\\s*\\{\\s*\"name\"\\s*:\\s*\"(?:" + string_join(escaped_names, "|") + ")\""
                    ")"
                    ")[\\s\\S]*"
                ),
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<function",
                "<tools>",
                "</tools>",
                "<response>",
                "</response>",
                "<function_call>",
                "</function_call>",
                "<json>",
                "</json>",
                "<JSON>",
                "</JSON>",
                "```",
                "```json",
                "```xml",
            };
        });
    }

    return data;
}
static void common_chat_parse_hermes_2_pro(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex open_regex(
        "(?:"
            "(```(?:xml|json)?\\n\\s*)?" // match 1 (block_start)
            "("                          // match 2 (open_tag)
                "<tool_call>"
                "|<function_call>"
                "|<tool>"
                "|<tools>"
                "|<response>"
                "|<json>"
                "|<xml>"
                "|<JSON>"
            ")?"
            "(\\s*\\{\\s*\"name\")" // match 3 (named tool call)
        ")"
        "|<function=([^>]+)>"            // match 4 (function name)
        "|<function name=\"([^\"]+)\">"  // match 5 (function name again)
    );

    while (auto res = builder.try_find_regex(open_regex)) {
        const auto & block_start = res->groups[1];
        std::string block_end = block_start.empty() ? "" : "```";

        const auto & open_tag = res->groups[2];
        std::string close_tag;

        if (!res->groups[3].empty()) {
            builder.move_to(res->groups[3].begin);
            close_tag = open_tag.empty() ? "" : "</" + builder.str(open_tag).substr(1);

            if (auto tool_call = builder.try_consume_json_with_dumped_args({{"arguments"}})) {
                if (!builder.add_tool_call(tool_call->value) || tool_call->is_partial) {
                    throw common_chat_msg_partial_exception("incomplete tool call");
                }
                builder.consume_spaces();
                builder.consume_literal(close_tag);
                builder.consume_spaces();
                if (!block_end.empty()) {
                    builder.consume_literal(block_end);
                    builder.consume_spaces();
                }
            } else {
                throw common_chat_msg_partial_exception("failed to parse tool call");
            }
        } else {
            auto function_name = builder.str(res->groups[4]);
            if (function_name.empty()) {
                function_name = builder.str(res->groups[5]);
            }
            GGML_ASSERT(!function_name.empty());

            close_tag = "</function>";

            if (auto arguments = builder.try_consume_json_with_dumped_args({{}})) {
                if (!builder.add_tool_call(function_name, "", arguments->value) || arguments->is_partial) {
                    throw common_chat_msg_partial_exception("incomplete tool call");
                }
                builder.consume_spaces();
                builder.consume_literal(close_tag);
                builder.consume_spaces();
                if (!block_end.empty()) {
                    builder.consume_literal(block_end);
                    builder.consume_spaces();
                }
            }
        }
    }

    builder.add_content(builder.consume_rest());
}

static common_chat_params common_chat_params_init_granite(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Pass thinking context for Granite template
    json additional_context = {
        {"thinking", inputs.enable_thinking},
    };

    data.prompt = apply(tmpl, inputs, /* messages_override= */ std::nullopt, /* tools_override= */ std::nullopt, additional_context);
    data.format = COMMON_CHAT_FORMAT_GRANITE;

    if (string_ends_with(data.prompt, "<think>\n") || string_ends_with(data.prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (!inputs.tools.is_null()) {
        // Granite uses <|tool_call|> followed by JSON list
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_rule(name + "-call", builder.add_schema(name +
"-args", {
                    {"type", "object"},
                    {"properties", {
                        {"name", {{"const", name}}},
                        {"arguments", parameters},
                    }},
                    {"required", json::array({"name", "arguments"})},
                })));
            });

            auto tool_call = builder.add_rule("tool_call", string_join(tool_rules, " | "));
            auto tool_list = builder.add_rule("tool_list", "\"[\" space " + tool_call + " (\",\" space " + tool_call + ")* space \"]\"");

            if (data.thinking_forced_open) {
                builder.add_rule("root", "\"</think>\" space \"<response>\" space [^<]* \"</response>\" space \"<|tool_call|>\" space " + tool_list);
            } else {
                builder.add_rule("root", "\"<|tool_call|>\" space " + tool_list);
            }

            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                "<|tool_call|>"
            });

            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<response>",
                "</response>",
                "<|tool_call|>",
            };
        });
    } else {
        // Handle thinking tags for non-tool responses
        if (data.thinking_forced_open && inputs.enable_thinking) {
            data.grammar_lazy = false;
            data.grammar = build_grammar([&](const common_grammar_builder & builder) {
                builder.add_rule("root", "\"</think>\" space \"<response>\" space .* \"</response>\" space");
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<response>",
                "</response>",
            };
        }
    }

    return data;
}

static void common_chat_parse_granite(common_chat_msg_parser & builder) {
    // Parse thinking tags
    static const common_regex start_think_regex(regex_escape("<think>"));
    static const common_regex end_think_regex(regex_escape("</think>"));
    // Granite models output partial tokens such as "<" and "<think".
    // By leveraging try_consume_regex()/try_find_regex() throwing
    // common_chat_msg_partial_exception for these partial tokens,
    // processing is interrupted and the tokens are not passed to add_content().
    if (auto res = builder.try_consume_regex(start_think_regex)) {
        // Restore position for try_parse_reasoning()
        builder.move_to(res->groups[0].begin);
        builder.try_find_regex(end_think_regex, std::string::npos, false);
        // Restore position for try_parse_reasoning()
        builder.move_to(res->groups[0].begin);
    }
    builder.try_parse_reasoning("<think>", "</think>");

    // Parse response tags
    static const common_regex start_response_regex(regex_escape("<response>"));
    static const common_regex end_response_regex(regex_escape("</response>"));
    // Granite models output partial tokens such as "<" and "<response".
    // Same hack as reasoning parsing.
    if (builder.try_consume_regex(start_response_regex)) {
        builder.try_find_regex(end_response_regex);
    }

    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // Look for tool calls
    static const common_regex tool_call_regex(regex_escape("<|tool_call|>"));
    if (auto res = builder.try_find_regex(tool_call_regex)) {
        builder.move_to(res->groups[0].end);

        // Expect JSON array of tool calls
        if (auto tool_call = builder.try_consume_json_with_dumped_args({{{"arguments"}}})) {
            if (!builder.add_tool_calls(tool_call->value) || tool_call->is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
    } else {
        builder.add_content(builder.consume_rest());
    }
}

static void common_chat_parse_nemotron_v2(common_chat_msg_parser & builder) {
    // Parse thinking tags
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // Look for tool calls
    static const common_regex tool_call_regex(regex_escape("<TOOLCALL>"));
    if (auto res = builder.try_find_regex(tool_call_regex)) {
        builder.move_to(res->groups[0].end);

        // Expect JSON array of tool calls
        auto tool_calls_data = builder.consume_json();
        if (tool_calls_data.json.is_array()) {
            if (!builder.try_consume_literal("</TOOLCALL>")) {
                throw common_chat_msg_partial_exception("Incomplete tool call");
            }
            builder.add_tool_calls(tool_calls_data.json);
        } else {
            throw common_chat_msg_partial_exception("Incomplete tool call");
        }
    }
    builder.add_content(builder.consume_rest());
}

static void common_chat_parse_apertus(common_chat_msg_parser & builder) {
    // Parse thinking tags
    builder.try_parse_reasoning("<|inner_prefix|>", "<|inner_suffix|>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // Look for tool calls
    static const common_regex tool_call_regex(regex_escape("<|tools_prefix|>"));
    if (auto res = builder.try_find_regex(tool_call_regex)) {
        builder.move_to(res->groups[0].end);

        auto tool_calls_data = builder.consume_json();
        if (tool_calls_data.json.is_array()) {
            builder.consume_spaces();
            if (!builder.try_consume_literal("<|tools_suffix|>")) {
                throw common_chat_msg_partial_exception("Incomplete tool call");
            }
            for (const auto & value : tool_calls_data.json) {
                if (value.is_object()) {
                    builder.add_tool_call_short_form(value);
                }
            }
        } else {
            throw common_chat_msg_partial_exception("Incomplete tool call");
        }
    }
    builder.add_content(builder.consume_rest());
}


static void common_chat_parse_lfm2(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // LFM2 format: <|tool_call_start|>[{"name": "get_current_time", "arguments": {"location": "Paris"}}]<|tool_call_end|>
    static const common_regex tool_call_start_regex(regex_escape("<|tool_call_start|>"));
    static const common_regex tool_call_end_regex(regex_escape("<|tool_call_end|>"));

    // Loop through all tool calls
    while (auto res = builder.try_find_regex(tool_call_start_regex, std::string::npos, /* add_prelude_to_content= */ true)) {
        builder.move_to(res->groups[0].end);

        // Parse JSON array format: [{"name": "...", "arguments": {...}}]
        auto tool_calls_data = builder.consume_json();

        // Consume end marker
        builder.consume_spaces();
        if (!builder.try_consume_regex(tool_call_end_regex)) {
            throw common_chat_msg_partial_exception("Expected <|tool_call_end|>");
        }

        // Process each tool call in the array
        if (tool_calls_data.json.is_array()) {
            for (const auto & tool_call : tool_calls_data.json) {
                if (!tool_call.is_object()) {
                    throw common_chat_msg_partial_exception("Tool call must be an object");
                }

                if (!tool_call.contains("name")) {
                    throw common_chat_msg_partial_exception("Tool call missing 'name' field");
                }

                std::string function_name = tool_call.at("name");
                std::string arguments = "{}";

                if (tool_call.contains("arguments")) {
                    if (tool_call.at("arguments").is_object()) {
                        arguments = tool_call.at("arguments").dump();
                    } else if (tool_call.at("arguments").is_string()) {
                        arguments = tool_call.at("arguments");
                    }
                }

                if (!builder.add_tool_call(function_name, "", arguments)) {
                    throw common_chat_msg_partial_exception("Incomplete tool call");
                }
            }
        } else {
            throw common_chat_msg_partial_exception("Expected JSON array for tool calls");
        }

        // Consume any trailing whitespace after this tool call
        builder.consume_spaces();
    }

    // Consume any remaining content after all tool calls
    auto remaining = builder.consume_rest();
    if (!string_strip(remaining).empty()) {
        builder.add_content(remaining);
    }
}

static void common_chat_parse_seed_oss(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<seed:tool_call>",
        /* form.tool_start  = */ "<function=",
        /* form.tool_sep    = */ ">",
        /* form.key_start   = */ "<parameter=",
        /* form.key_val_sep = */ ">",
        /* form.val_end     = */ "</parameter>",
        /* form.tool_end    = */ "</function>",
        /* form.scope_end   = */ "</seed:tool_call>",
    };
    builder.consume_reasoning_with_xml_tool_calls(form, "<seed:think>", "</seed:think>");
}

static common_chat_params common_chat_params_init_without_tools(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    data.grammar_lazy = false;
    if (!inputs.json_schema.is_null()) {
        if (!inputs.grammar.empty()) {
            throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
        }
        data.grammar = json_schema_to_grammar(inputs.json_schema);
    } else {
        data.grammar = inputs.grammar;
    }
    return data;
}

static common_chat_params common_chat_params_init_seed_oss(
    const common_chat_template         & tmpl,
    templates_params                   & params,
    const common_chat_templates_inputs & inputs)
{
    common_chat_params data;
    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_SEED_OSS;
    if (string_ends_with(data.prompt, "<seed:think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</seed:think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (params.tools.is_array() && !params.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(params.tools, [&](const json & tool) {
                const auto & function   = tool.at("function");
                std::string  name       = function.at("name");
                auto         parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                // Create rule for Seed-OSS function call format
                std::string param_rules;
                if (parameters.contains("properties")) {
                    for (const auto & [key, value] : parameters.at("properties").items()) {
                        param_rules += "\"<parameter=" + key + ">\"" + builder.add_schema(name + "-arg-" + key, value) +
                                       "\"</parameter>\"";
                    }
                }

                tool_rules.push_back(builder.add_rule(name + "-call",
                                                      "\"<seed:tool_call>\" space \"<function=" + name + ">\" space " +
                                                          param_rules +
                                                          " \"</function>\" space \"</seed:tool_call>\""));
            });

            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<seed:tool_call>" });

            data.preserved_tokens = {
                "<seed:think>", "</seed:think>", "<seed:tool_call>", "</seed:tool_call>",
                "<function=",   "</function>",   "<parameter=",      "</parameter>",
            };

            builder.add_rule("root", string_join(tool_rules, " | "));
        });
    }
    return data;
}


common_chat_params common_chat_templates_apply_jinja(
    const struct common_chat_templates        * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    templates_params params;
    params.tools = common_chat_tools_to_json_oaicompat<json>(inputs.tools);
    const auto & tmpl = params.tools.is_array() && tmpls->template_tool_use
        ? *tmpls->template_tool_use
        : *tmpls->template_default;
    const auto & src = tmpl.source();
    const auto & caps = tmpl.original_caps();
    params.messages = common_chat_msgs_to_json_oaicompat<json>(inputs.messages, /* concat_text= */ !tmpl.original_caps().requires_typed_content);
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.tool_choice = inputs.tool_choice;
    params.enable_thinking = inputs.enable_thinking;
    params.grammar = inputs.grammar;
    params.now = inputs.now;
    params.add_bos = tmpls->add_bos;
    params.add_eos = tmpls->add_eos;

    params.extra_context = json::object();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    if (inputs.parallel_tool_calls && !tmpl.original_caps().supports_parallel_tool_calls) {
        LOG_DBG("Disabling parallel_tool_calls because the template does not support it\n");
        params.parallel_tool_calls = false;
    } else {
        params.parallel_tool_calls = inputs.parallel_tool_calls;
    }

    if (params.tools.is_array()) {
        if (params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !params.grammar.empty()) {
            throw std::runtime_error("Cannot specify grammar with tools");
        }
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN("Template supports tool calls but does not natively describe tools. The fallback behaviour used may produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    // DeepSeek V3.1: detect based on specific patterns in the template
    if (src.find("message['prefix'] is defined and message['prefix'] and thinking") != std::string::npos &&
        params.json_schema.is_null()) {
        return common_chat_params_init_deepseek_v3_1(tmpl, params);
    }

    // DeepSeek R1: use handler in all cases except json schema (thinking / tools).
    if (src.find("<｜tool▁calls▁begin｜>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_deepseek_r1(tmpl, params);
    }

    // Command R7B: : use handler in all cases except json schema (thinking / tools).
    if (src.find("<|END_THINKING|><|START_ACTION|>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_command_r7b(tmpl, params);
    }

    // Granite (IBM) - detects thinking / tools support
    if (src.find("elif thinking") != std::string::npos && src.find("<|tool_call|>") != std::string::npos) {
        return common_chat_params_init_granite(tmpl, params);
    }

    // GLM 4.5: detect by <arg_key> and <arg_value> tags (check before Hermes since both use <tool_call>)
    if (src.find("[gMASK]<sop>") != std::string::npos &&
        src.find("<arg_key>") != std::string::npos &&
        src.find("<arg_value>") != std::string::npos &&
        params.json_schema.is_null()) {
        return common_chat_params_init_glm_4_5(tmpl, params);
    }

    // Qwen3-Coder XML format detection (must come before Hermes 2 Pro)
    // Detect via explicit XML markers unique to Qwen3-Coder to avoid false positives in other templates.
    // Require presence of <tool_call>, <function=...>, and <parameter=...> blocks.
    if (src.find("<tool_call>") != std::string::npos &&
        src.find("<function>") != std::string::npos &&
        src.find("<function=") != std::string::npos &&
        src.find("<parameters>") != std::string::npos &&
        src.find("<parameter=") != std::string::npos) {
        return common_chat_params_init_qwen3_coder_xml(tmpl, params);
    }

    // Xiaomi MiMo format detection (must come before Hermes 2 Pro)
    if (src.find("<tools>") != std::string::npos &&
        src.find("# Tools") != std::string::npos &&
        src.find("</tools>") != std::string::npos &&
        src.find("<tool_calls>") != std::string::npos &&
        src.find("</tool_calls>") != std::string::npos &&
        src.find("<tool_response>") != std::string::npos) {
        return common_chat_params_init_xiaomi_mimo(tmpl, params);
    }

    // Hermes 2/3 Pro, Qwen 2.5 Instruct (w/ tools)
    if (src.find("<tool_call>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_hermes_2_pro(tmpl, params);
    }

    // GPT-OSS
    if (src.find("<|channel|>") != std::string::npos) {
        return common_chat_params_init_gpt_oss(tmpl, params);
    }

    // Seed-OSS
    if (src.find("<seed:think>") != std::string::npos) {
        return common_chat_params_init_seed_oss(tmpl, params, inputs);
    }

    // Nemotron v2
    if (src.find("<SPECIAL_10>") != std::string::npos) {
        return common_chat_params_init_nemotron_v2(tmpl, params);
    }

    // Apertus format detection
    if (src.find("<|system_start|>") != std::string::npos && src.find("<|tools_prefix|>") != std::string::npos) {
        return common_chat_params_init_apertus(tmpl, params);
    }

    // LFM2 (w/ tools)
    if (src.find("List of tools: <|tool_list_start|>[") != std::string::npos &&
        src.find("]<|tool_list_end|>") != std::string::npos) {
        return common_chat_params_init_lfm2(tmpl, params);
    }

    // MiniMax-M2 format detection
    if (src.find("]~!b[") != std::string::npos && src.find("]~b]") != std::string::npos) {
        return common_chat_params_init_minimax_m2(tmpl, params);
    }

    // Kimi K2 format detection
    if (src.find("<|im_system|>tool_declare<|im_middle|>") != std::string::npos &&
        src.find("<|tool_calls_section_begin|>") != std::string::npos &&
        src.find("## Return of") != std::string::npos) {
        return common_chat_params_init_kimi_k2(tmpl, params);
    }

    // Apriel 1.5 format detection
    if (src.find("<thinking>") != std::string::npos &&
        src.find("</thinking>") != std::string::npos &&
        src.find("<available_tools>") != std::string::npos &&
        src.find("<|assistant|>") != std::string::npos &&
        src.find("<|tool_result|>") != std::string::npos &&
        src.find("<tool_calls>[") != std::string::npos &&
        src.find("]</tool_calls>") != std::string::npos) {
        return common_chat_params_init_apriel_1_5(tmpl, params);
    }

    // Use generic handler when mixing tools + JSON schema.
    // TODO: support that mix in handlers below.
    if ((params.tools.is_array() && params.json_schema.is_object())) {
        return common_chat_params_init_generic(tmpl, params);
    }

    // Functionary prepends "all\n" to plain content outputs, so we use its handler in all cases.
    if (src.find(">>>all") != std::string::npos) {
        return common_chat_params_init_functionary_v3_2(tmpl, params);
    }

    // Firefunction v2 requires datetime and functions in the context even w/o tools, so we also use its handler in all cases.
    if (src.find(" functools[") != std::string::npos) {
        return common_chat_params_init_firefunction_v2(tmpl, params);
    }

    // Functionary v3.1 (w/ tools)
    if (src.find("<|start_header_id|>") != std::string::npos
        && src.find("<function=") != std::string::npos) {
        return common_chat_params_init_functionary_v3_1_llama_3_1(tmpl, params);
    }

    // Llama 3.1, 3.2, 3.3 (also requires date_string so using it even w/o tools)
    if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        auto allow_python_tag_builtin_tools = src.find("<|python_tag|>") != std::string::npos;
        return common_chat_params_init_llama_3_x(tmpl, params, allow_python_tag_builtin_tools);
    }

    if (src.find("[THINK]") != std::string::npos && src.find("[/THINK]") != std::string::npos) {
        return common_chat_params_init_magistral(tmpl, params);
    }

    // Plain handler (no tools)
    if (params.tools.is_null() || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
        return common_chat_params_init_without_tools(tmpl, params);
    }

    // Mistral Nemo (w/ tools)
    if (src.find("[TOOL_CALLS]") != std::string::npos) {
        return common_chat_params_init_mistral_nemo(tmpl, params);
    }

    // Generic fallback
    return common_chat_params_init_generic(tmpl, params);
}

void common_chat_parse_content_only(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    builder.add_content(builder.consume_rest());
}

void common_chat_parse(common_chat_msg_parser & builder) {
    LOG_DBG("Parsing input with format %s: %s\n", common_chat_format_name(builder.syntax().format), builder.input().c_str());

    switch (builder.syntax().format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            common_chat_parse_content_only(builder);
            break;
        case COMMON_CHAT_FORMAT_GENERIC:
            common_chat_parse_generic(builder);
            break;
        case COMMON_CHAT_FORMAT_MISTRAL_NEMO:
            common_chat_parse_mistral_nemo(builder);
            break;
        case COMMON_CHAT_FORMAT_MAGISTRAL:
            common_chat_parse_magistral(builder);
            break;
        case COMMON_CHAT_FORMAT_LLAMA_3_X:
            common_chat_parse_llama_3_1(builder);
            break;
        case COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS:
            common_chat_parse_llama_3_1(builder, /* with_builtin_tools= */ true);
            break;
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1:
            common_chat_parse_deepseek_r1(builder);
            break;
        case COMMON_CHAT_FORMAT_DEEPSEEK_V3_1:
            common_chat_parse_deepseek_v3_1(builder);
            break;
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2:
            common_chat_parse_functionary_v3_2(builder);
            break;
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1:
            common_chat_parse_functionary_v3_1_llama_3_1(builder);
            break;
        case COMMON_CHAT_FORMAT_HERMES_2_PRO:
            common_chat_parse_hermes_2_pro(builder);
            break;
        case COMMON_CHAT_FORMAT_FIREFUNCTION_V2:
            common_chat_parse_firefunction_v2(builder);
            break;
        case COMMON_CHAT_FORMAT_COMMAND_R7B:
            common_chat_parse_command_r7b(builder);
            break;
        case COMMON_CHAT_FORMAT_GRANITE:
            common_chat_parse_granite(builder);
            break;
        case COMMON_CHAT_FORMAT_GPT_OSS:
            common_chat_parse_gpt_oss(builder);
            break;
        case COMMON_CHAT_FORMAT_SEED_OSS:
            common_chat_parse_seed_oss(builder);
            break;
        case COMMON_CHAT_FORMAT_NEMOTRON_V2:
            common_chat_parse_nemotron_v2(builder);
            break;
        case COMMON_CHAT_FORMAT_APERTUS:
            common_chat_parse_apertus(builder);
            break;
        case COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS:
            common_chat_parse_lfm2(builder);
            break;
        case COMMON_CHAT_FORMAT_MINIMAX_M2:
            common_chat_parse_minimax_m2(builder);
            break;
        case COMMON_CHAT_FORMAT_GLM_4_5:
            common_chat_parse_glm_4_5(builder);
            break;
        case COMMON_CHAT_FORMAT_KIMI_K2:
            common_chat_parse_kimi_k2(builder);
            break;
        case COMMON_CHAT_FORMAT_QWEN3_CODER_XML:
            common_chat_parse_qwen3_coder_xml(builder);
            break;
        case COMMON_CHAT_FORMAT_APRIEL_1_5:
            common_chat_parse_apriel_1_5(builder);
            break;
        case COMMON_CHAT_FORMAT_XIAOMI_MIMO:
            common_chat_parse_xiaomi_mimo(builder);
            break;
        default:
            throw std::runtime_error(std::string("Unsupported format: ") + common_chat_format_name(builder.syntax().format));
    }
    builder.finish();
}
