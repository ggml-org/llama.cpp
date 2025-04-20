#include "chat.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "minja/chat-template.hpp"
#include "minja/minja.hpp"

#include <optional>

typedef minja::chat_template common_chat_template;

struct common_chat_templates {
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
    bool extract_reasoning     = true;
};

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice) {
    if (tool_choice == "auto") {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    if (tool_choice == "none") {
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    if (tool_choice == "required") {
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    }
    throw std::runtime_error("Invalid tool_choice: " + tool_choice);
}

template <>
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {

        if (!messages.is_array()) {
            throw std::runtime_error("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::runtime_error("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::runtime_error("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::runtime_error("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text") {
                            throw std::runtime_error("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::runtime_error("Invalid 'content' type: expected string or array, got " + content.dump() + " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::runtime_error("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::runtime_error("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::runtime_error("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::runtime_error("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name = fc.at("name");
                    tc.arguments = fc.at("arguments");
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::runtime_error("Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & https://github.com/ggml-org/llama.cpp/issues/12279)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id");
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()) + "; messages = " + messages.dump(2));
    }

    return msgs;
}

template <>
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    json messages = json::array();
    for (const auto & msg : msgs) {
        if (!msg.content.empty() && !msg.content_parts.empty()) {
            throw std::runtime_error("Cannot specify both content and content_parts");
        }
        json jmsg {
            {"role", msg.role},
        };
        if (!msg.content.empty()) {
            jmsg["content"] = msg.content;
        } else if (!msg.content_parts.empty()) {
            if (concat_typed_text) {
                std::string text;
                for (const auto & part : msg.content_parts) {
                    if (part.type != "text") {
                        LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                        continue;
                    }
                    if (!text.empty()) {
                        text += '\n';
                    }
                    text += part.text;
                }
                jmsg["content"] = text;
            } else {
                auto & parts = jmsg["content"] = json::array();
                for (const auto & part : msg.content_parts) {
                    parts.push_back({
                        {"type", part.type},
                        {"text", part.text},
                    });
                }
            }
        } else {
            jmsg["content"] = json(); // null
        }
        if (!msg.reasoning_content.empty()) {
            jmsg["reasoning_content"] = msg.reasoning_content;
        }
        if (!msg.tool_name.empty()) {
            jmsg["name"] = msg.tool_name;
        }
        if (!msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        if (!msg.tool_calls.empty()) {
            auto & tool_calls = jmsg["tool_calls"] = json::array();
            for (const auto & tool_call : msg.tool_calls) {
                json tc {
                    {"type", "function"},
                    {"function", {
                        {"name", tool_call.name},
                        {"arguments", tool_call.arguments},
                    }},
                };
                if (!tool_call.id.empty()) {
                    tc["id"] = tool_call.id;
                }
                tool_calls.push_back(tc);
            }
        }
        messages.push_back(jmsg);
    }
    return messages;
}

template <>
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const std::string & messages) {
    return common_chat_msgs_parse_oaicompat(json::parse(messages));
}

template <>
std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::runtime_error("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::runtime_error("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::runtime_error("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::runtime_error("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.at("description"),
                    /* .parameters = */ function.at("parameters").dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

template <>
std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const std::string & tools) {
    return common_chat_tools_parse_oaicompat(json::parse(tools));
}

template <>
json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            {"type", "function"},
            {"function", {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", json::parse(tool.parameters)},
            }},
        });
    }
    return result;
}

bool common_chat_verify_template(const std::string & tmpl, bool use_jinja) {
    if (use_jinja) {
        try {
            common_chat_msg msg;
            msg.role = "user";
            msg.content = "test";

            auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);

            common_chat_templates_inputs inputs;
            inputs.messages = {msg};

            common_chat_templates_apply(tmpls.get(), inputs);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to apply template: %s\n", __func__, e.what());
            return false;
        }
    }
    llama_chat_message chat[] = {{"user", "test"}};
    const int res = llama_chat_apply_template(tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string common_chat_format_single(
        const struct common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & past_msg,
        const common_chat_msg & new_msg,
        bool add_ass,
        bool use_jinja) {

    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates * tmpls, bool use_jinja) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    auto add_simple_msg = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system",    "You are a helpful assistant");
    add_simple_msg("user",      "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user",      "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

#define CHATML_TEMPLATE_SRC \
    "{%- for message in messages -%}\n" \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n" \
    "{%- if add_generation_prompt -%}\n" \
    "  {{- '<|im_start|>assistant\n' -}}\n" \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

const char * common_chat_templates_source(const struct common_chat_templates * tmpls, const char * variant) {
    if (variant != nullptr) {
        if (strcmp(variant, "tool_use") == 0) {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source().c_str();
            }
            return nullptr;
        } else {
            LOG_DBG("%s: unknown template variant: %s\n", __func__, variant);
        }
    }
    return tmpls->template_default->source().c_str();
}

common_chat_templates_ptr common_chat_templates_init(
    const struct llama_model * model,
    const std::string & chat_template_override,
    const std::string & bos_token_override,
    const std::string & eos_token_override)
{
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }
    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    if (model) {
        const auto * vocab = llama_model_get_vocab(model);
        const auto get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos
                    || template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN("common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't work as intended.\n", name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    try {
        tmpls->template_default = std::make_unique<minja::chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to parse chat template (defaulting to chatml): %s \n", __func__, e.what());
        tmpls->template_default = std::make_unique<minja::chat_template>(CHATML_TEMPLATE_SRC, token_bos, token_eos);
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<minja::chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

std::string common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY: return "Content-only";
        case COMMON_CHAT_FORMAT_GENERIC: return "Generic";
        case COMMON_CHAT_FORMAT_MISTRAL_NEMO: return "Mistral Nemo";
        case COMMON_CHAT_FORMAT_LLAMA_3_X: return "Llama 3.x";
        case COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS: return "Llama 3.x with builtin tools";
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1: return "DeepSeek R1";
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING: return "DeepSeek R1 (extract reasoning)";
        case COMMON_CHAT_FORMAT_FIREFUNCTION_V2: return "FireFunction v2";
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2: return "Functionary v3.2";
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1: return "Functionary v3.1 Llama 3.1";
        case COMMON_CHAT_FORMAT_HERMES_2_PRO: return "Hermes 2 Pro";
        case COMMON_CHAT_FORMAT_HERMES_2_PRO_EXTRACT_REASONING: return "Hermes 2 Pro (extract reasoning)";
        case COMMON_CHAT_FORMAT_COMMAND_R7B: return "Command R7B";
        case COMMON_CHAT_FORMAT_COMMAND_R7B_EXTRACT_REASONING: return "Command R7B (extract reasoning)";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

static bool parse_json(std::string::const_iterator & it, const std::string::const_iterator & end, json & out) {
    // // https://json.nlohmann.me/features/parsing/sax_interface/
    struct json_error_locator : public nlohmann::json_sax<json> {
        std::size_t position;
        bool found_error;

        json_error_locator() : position(0), found_error(false) {}

        bool parse_error(std::size_t position, const std::string &, const json::exception &) override { // NOLINT
            this->position = position - 1;
            this->found_error = true;
            return false;
        }
        bool null() override { return true; } // NOLINT
        bool boolean(bool) override { return true; } // NOLINT
        bool number_integer(number_integer_t) override { return true; } // NOLINT
        bool number_unsigned(number_unsigned_t) override { return true; } // NOLINT
        bool number_float(number_float_t, const string_t &) override { return true; } // NOLINT
        bool string(string_t &) override { return true; } // NOLINT
        bool binary(binary_t &) override { return true; } // NOLINT
        bool start_object(std::size_t) override { return true; } // NOLINT
        bool key(string_t &) override { return true; } // NOLINT
        bool end_object() override { return true; }
        bool start_array(std::size_t) override { return true; } // NOLINT
        bool end_array() override { return true; }
    };
    json_error_locator err_loc;
    json::sax_parse(it, end, &err_loc);

    std::string::const_iterator temptative_end;
    if (err_loc.found_error) {
        temptative_end = it + err_loc.position;
    } else {
        temptative_end = end;
    }
    std::string json_sub {it, temptative_end};
    try {
        out = json::parse(json_sub);
        it = temptative_end;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

static bool parse_literal(std::string::const_iterator & it, const std::string::const_iterator & end, const std::string & expected) {
    auto expected_it = expected.begin();
    auto tmp_it = it;
    while (tmp_it != end && expected_it != expected.end() && *tmp_it == *expected_it) {
        ++tmp_it;
        ++expected_it;
    }
    if (expected_it == expected.end()) {
        it = tmp_it;
        return true;
    }
    return false;
}

static std::optional<std::smatch> parse_pattern(std::string::const_iterator & it, const std::string::const_iterator & end, const std::regex & expected) {
    std::smatch match;
    if (std::regex_match(it, end, match, expected)) {
        it = match.suffix().first;
        return match;
    }
    return std::nullopt;
}

static void consume_spaces(std::string::const_iterator & it, const std::string::const_iterator & end) {
    while (it != end && std::isspace(*it)) {
        ++it;
    }
}

/**
 * Takes a prefix regex that must have 1 group to capture the function name, a closing suffix, and expects json parameters in between.
 * Aggregates the prefix, suffix and in-between text into the content.
 */
static common_chat_msg parse_json_tool_calls(
    const std::string& input,
    const std::optional<std::regex> & trigger_opt,
    const std::regex & function_regex,
    const std::regex & close_regex,
    bool allow_raw_python = false) {
    std::smatch match;

    common_chat_msg result;
    result.role = "assistant";


    auto end = input.end();
    auto it = input.begin();

    if (trigger_opt) {
        if (!std::regex_search(it, end, match, *trigger_opt)) {
            result.content = input;
            return result;
        }
        result.content = match.prefix().str();
        it = match.suffix().first;
    }

    while (it != end) {
        std::sregex_iterator rend;
        std::sregex_iterator rit(it, end, function_regex);
        if (rit == rend) {
            result.content += std::string(it, end);
            break;
        }
        auto name = rit->str(1);
        result.content += std::string(it, rit->prefix().second);
        it = rit->suffix().first;

        json arguments;
        if (parse_json(it, end, arguments)) {
            if (!std::regex_search(it, end, match, close_regex)) {
                throw std::runtime_error("Malformed input, missing closing pattern: " + input);
            }
            it = match.suffix().first;
            result.tool_calls.push_back({name, arguments.is_string() ? arguments.get<std::string>() : arguments.dump(), /* id= */ ""});
        } else {
            if (allow_raw_python && name == "python") {
                result.tool_calls.push_back({name, json({{"code", std::string(it, end)}}).dump(), /* id= */ ""});
                break;
            }
            throw std::runtime_error("Failed to parse json tool call arguments: " + input);
        }
    }

    if (!result.tool_calls.empty()) {
        if (!string_strip(result.content).empty()) {
            LOG_WRN("Content found with tool calls: %s\n", result.content.c_str());
        }
        result.content = "";
    }
    return result;
}

static common_chat_tool_call process_tool_call(const json & tool_call) {
    const auto & arguments = tool_call.at("arguments");
    return {
        /* .name = */ tool_call.at("name"),
        /* .arguments = */ arguments.is_string() ? arguments.get<std::string>() : arguments.dump(),
        /* .id = */ tool_call.contains("id") ? tool_call.at("id") : "",
    };
}
static common_chat_msg parse_prefixed_json_tool_call_array(const std::string& input, const std::string & prefix, size_t rstrip_prefix = 0) {
    auto content_end = input.find(prefix);
    size_t tc_start = std::string::npos;

    common_chat_msg result;
    result.role = "assistant";
    if (content_end == std::string::npos) {
        result.content = input;
    } else {
        tc_start = content_end + prefix.size() - rstrip_prefix;
        result.content = input.substr(0, content_end);
        auto tool_calls = json::parse(input.substr(tc_start));
        for (const auto & tool_call : tool_calls) {
            result.tool_calls.emplace_back(process_tool_call(tool_call));
        }
    }
    return result;
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
    const nlohmann::ordered_json & messages,
    const nlohmann::ordered_json & tools,
    bool add_generation_prompt,
    const nlohmann::ordered_json & extra_context = nlohmann::ordered_json())
{
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = messages;
    tmpl_inputs.tools = tools;
    tmpl_inputs.add_generation_prompt = add_generation_prompt;
    tmpl_inputs.extra_context = extra_context;
    // TODO: add flag to control date/time, if only for testing purposes.
    // tmpl_inputs.now = std::chrono::system_clock::now();

    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
    if (string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (string_ends_with(result, tmpl.eos_token())) {
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

    data.prompt = apply(tmpl, tweaked_messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
    data.format = COMMON_CHAT_FORMAT_GENERIC;
    return data;
}
static common_chat_msg common_chat_parse_generic(const std::string & input) {
    json data = json::parse(input);
    common_chat_msg result;
    result.role = "assistant";
    if (data.contains("tool_calls")) {
        for (const auto & tool_call : data.at("tool_calls")) {
            result.tool_calls.push_back({
                tool_call.at("name"),
                tool_call.at("arguments").dump(),
                tool_call.contains("id") ? tool_call.at("id") : "",
            });
        }
    } else if (data.contains("tool_call")) {
        result.tool_calls.push_back({
            data.at("tool_call").at("name"),
            data.at("tool_call").at("arguments").dump(),
            /* id= */ "",
        });
    } else if (data.contains("response")) {
        const auto & response = data.at("response");
        result.content = response.is_string() ? response.get<std::string>() : response.dump(2);
    }
    return result;
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
    data.prompt = apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
    data.format = COMMON_CHAT_FORMAT_MISTRAL_NEMO;
    return data;
}
static common_chat_msg common_chat_parse_mistral_nemo(const std::string & input) {
    return parse_prefixed_json_tool_call_array(input, "[TOOL_CALLS]");
}

static common_chat_params common_chat_params_init_command_r7b(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
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
        builder.add_rule("root", "\"<|START_ACTION|>\" " + builder.add_schema("tool_calls", schema) + " \"<|END_ACTION|>\"");
    });
    data.grammar_triggers.push_back({
        COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
        "<|START_ACTION|>",
    });
    data.preserved_tokens = {
        "<|START_ACTION|>",
        "<|END_ACTION|>",
        "<|START_RESPONSE|>",
        "<|END_RESPONSE|>",
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    };
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
    data.prompt = apply(tmpl, adjusted_messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt, {});
    data.format = inputs.extract_reasoning ? COMMON_CHAT_FORMAT_COMMAND_R7B_EXTRACT_REASONING : COMMON_CHAT_FORMAT_COMMAND_R7B;
    return data;
}
static common_chat_msg common_chat_parse_command_r7b(const std::string & input, bool extract_reasoning) {
    static const std::regex thought_regex("(<\\|START_THINKING\\|>([\\s\\S]*?)<\\|END_THINKING\\|>)([\\s\\S]*)");
    static const std::regex action_regex("<\\|START_ACTION\\|>([\\s\\S]*?)<\\|END_ACTION\\|>");
    static const std::regex response_regex("(?:<\\|START_RESPONSE\\|>)?([\\s\\S]*?)<\\|END_RESPONSE\\|>");

    std::smatch match;

    common_chat_msg result;
    result.role = "assistant";

    std::string rest = input;

    if (std::regex_match(rest, match, thought_regex)) {
        if (extract_reasoning) {
            result.reasoning_content = match[2].str();
        } else if (!match[2].str().empty()) {
            // Let the unparsed thinking tags through in content only if their insides aren't empty.
            result.content = match[1].str();
        }
        rest = match[3].str();
    }
    if (std::regex_match(rest, match, action_regex)) {
        auto actions_str = match[1].str();
        auto actions = json::parse(actions_str);
        for (const auto & action : actions) {
            result.tool_calls.push_back({
                /* .name = */      action.at("tool_name"),
                /* .arguments = */ action.at("parameters").dump(),
                /* .id = */        action.at("tool_call_id"),
            });
        }
    } else if (std::regex_match(rest, match, response_regex)) {
        auto response = match[1].str();
        result.content += response;
    } else {
        result.content += rest;
    }
    return result;
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

static common_chat_params common_chat_params_init_llama_3_1_tool_calls(const common_chat_template & tmpl, const struct templates_params & inputs, bool allow_python_tag_builtin_tools) {
    auto builtin_tools = json::array();
    common_chat_params data;
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
            COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START,
            "\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\"", // + name + "\"[\\s\\S]*",
        });
        if (!builtin_tools.empty()) {
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|python_tag|>"});
            data.preserved_tokens.push_back("<|python_tag|>");
        }
        // Allow a few empty lines on top of the usual constrained json schema space rule.
        builder.add_rule("root", string_join(tool_rules, " | "));
    });
    data.additional_stops.push_back("<|eom_id|>");
    data.prompt = apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt, {
        {"tools_in_user_message", false},
        {"builtin_tools", builtin_tools.empty() ? json() : builtin_tools},
    });
    data.format = allow_python_tag_builtin_tools && !builtin_tools.empty()
        ? COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS
        : COMMON_CHAT_FORMAT_LLAMA_3_X;
    return data;
}
static common_chat_msg common_chat_parse_llama_3_1(const std::string & input, bool with_builtin_tools = false) {
    // TODO: tighten & simplify the parser, don't accept leading text context.
    static const std::regex function_regex(
        "\\s*\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"parameters\"\\s*: ");
    static const std::regex close_regex("\\}\\s*");
    static const std::regex builtin_call_regex("<\\|python_tag\\|>\\s*([^.(]+)\\s*\\.\\s*call\\s*\\(\\s*([\\w]+)\\s*=\\s*([\\s\\S]*?)\\)");

    if (with_builtin_tools) {
        std::smatch match;
        if (std::regex_match(input, match, builtin_call_regex)) {
            try {
                auto name = match[1].str();
                auto arg_name = match[2].str();
                auto arg_value_str = match[3].str();
                auto arg_value = json::parse(arg_value_str);

                common_chat_msg msg;
                msg.role = "assistant";
                msg.tool_calls.push_back({
                    /* .name = */ name,
                    /* .arguments = */ (json {
                        {arg_name, arg_value},
                    }).dump(),
                    /* .id = */ "",
                });
                return msg;
            } catch (const std::exception & e) {
                LOG_WRN("Failed to parse builtin tool call arguments (%s): %s", e.what(), input.c_str());
            }
        }
    }
    return parse_json_tool_calls(input, std::nullopt, function_regex, close_regex);
}

static common_chat_params common_chat_params_init_deepseek_r1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
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
                    "\"<｜tool▁call▁begin｜>function<｜tool▁sep｜>" + name + "\\n"
                    "```json\\n\" " + builder.add_schema(name + "-args", parameters) + " "
                    "\"```<｜tool▁call▁end｜>\""));
            });
            // Distill Qwen 7B & 32B models seem confused re/ syntax of their tool call opening tag,
            // so we accept common variants (then it's all constrained)
            builder.add_rule("root",
                "( \"<｜tool▁calls▁begin｜>\" | \"<｜tool_calls_begin｜>\" | \"<｜tool calls begin｜>\" | \"<｜tool\\\\_calls\\\\_begin｜>\" ) "
                "(" + string_join(tool_rules, " | ") + ")" + (inputs.parallel_tool_calls ? "*" : "") + " "
                "\"<｜tool▁calls▁end｜>\""
                " space");
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<｜tool▁calls▁begin｜>"});
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<｜tool_calls_begin｜>"});
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<｜tool calls begin｜>"});
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<｜tool\\_calls\\_begin｜>"});
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
    auto prompt = apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);

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
    data.format = inputs.extract_reasoning ? COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING : COMMON_CHAT_FORMAT_DEEPSEEK_R1;
    return data;
}
static common_chat_msg handle_think_tag_prelude(const std::string & input, bool extract_reasoning, const std::function<common_chat_msg(const std::string &)> & rest_parser) {
    std::smatch match;
    static const std::regex reasoning_content_regex("((?:<think>)?([\\s\\S\\r\\n]*?)</think>)?([\\s\\S\\r\\n]*)");
    if (std::regex_match(input, match, reasoning_content_regex)) {
        auto rest = match[3].str();
        auto msg = rest_parser(rest);
        auto reasoning_content = string_strip(match[2].str());
        if (extract_reasoning) {
            msg.reasoning_content = reasoning_content;
        } else if (!reasoning_content.empty()) {
            std::ostringstream content;
            content << "<think>" << reasoning_content << "</think>" << msg.content;
            msg.content = content.str();
        }
        return msg;
    }
    return rest_parser(input);
}
static common_chat_msg common_chat_parse_deepseek_r1(const std::string & input, bool extract_reasoning) {
    return handle_think_tag_prelude(input, extract_reasoning, [](const std::string & input) {
        static const std::regex function_regex("<｜tool▁call▁begin｜>function<｜tool▁sep｜>([^\n]+)\n```json\n");
        static const std::regex close_regex("```[\\s\\r\\n]*<｜tool▁call▁end｜>");
        static const std::regex tool_calls_regex("[\\s\\r\\n]*(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>)([\\s\\S\\r\\n]*?)<｜tool▁calls▁end｜>");

        common_chat_msg msg;
        msg.role = "assistant";
        std::smatch match;
        if (std::regex_search(input, match, tool_calls_regex)) {
            auto tool_calls = match[1].str();
            auto msg2 = parse_json_tool_calls(tool_calls, std::nullopt, function_regex, close_regex);
            msg.tool_calls = std::move(msg2.tool_calls);
        } else {
            msg.content = input;
        }
        return msg;
    });
}

static common_chat_params common_chat_params_init_firefunction_v2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    LOG_DBG("%s\n", __func__);
    common_chat_params data;
    data.prompt = apply(tmpl, inputs.messages, /* tools= */ nullptr, inputs.add_generation_prompt, {
        {"datetime", "Jan 29 2025 13:00:00 GMT"},
        {"functions", json(inputs.tools.empty() ? "" : inputs.tools.dump(2))},
    });
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
static common_chat_msg common_chat_parse_firefunction_v2(const std::string & input) {
    return parse_prefixed_json_tool_call_array(input, " functools[", /* rstrip_prefix= */ 1);
}

static common_chat_params common_chat_params_init_functionary_v3_2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
    // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
    common_chat_params data;
    data.prompt = apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
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
                auto args_rule = builder.add_schema(name + "-args", parameters);
                first_tool_rules.push_back(builder.add_rule(name + "-call", "( \"assistant<|end_header_id|>\\n\" )? \"" + name + "\\n\" " + args_rule));
                subsequent_tool_rules.push_back(builder.add_rule(name + "-call2", "\">>>" + name + "\\n\" " + args_rule));
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START,
                    regex_escape(name + "\n"),
                });
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START,
                    regex_escape("assistant<|end_header_id|>\n" + name + "\n"),
                });
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                    regex_escape(">>>" + name + "\n"),
                });
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                    ">>>assistant<|end_header_id|>\n" + name,
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

static common_chat_msg common_chat_parse_functionary_v3_2(const std::string & input) {
    static const std::regex function_regex(R"((?:>>>)?(?:assistant<|end_header_id|>\n)?(\w+)\n)");
    static const std::regex close_regex(R"($|(?=>>>))");

    std::string content;
    auto it = input.begin();
    const auto end = input.end();

    if (parse_literal(it, end, "all\n")) {
        std::smatch match;
        if (std::regex_search(it, end, match, function_regex)) {
            auto fun_it = match.prefix().second;
            content = std::string(it, fun_it);
            it = fun_it;
        } else {
            common_chat_msg res;
            res.role = "assistant";
            res.content = std::string(it, end);
            return res;
        }
    }
    // TODO: tighten & simplify.
    try {
        auto res = parse_json_tool_calls(std::string(it, end), std::nullopt, function_regex, close_regex, /* allow_raw_python= */ true);
        res.content = content + res.content;
        return res;
    } catch (const std::exception & e) {
        LOG_ERR("Failed to parse functionary v3.2 input: %s\n", e.what());
        common_chat_msg res;
        res.role = "assistant";
        res.content = input;
        return res;
    }
}

static common_chat_params common_chat_params_init_functionary_v3_1_llama_3_1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
    common_chat_params data;
    json tools = inputs.tools.is_null() ? inputs.tools : json::array();
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

    data.prompt = apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
    // TODO: if (has_raw_python)
    data.format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1;
    return data;
}
static common_chat_msg common_chat_parse_functionary_v3_1_llama_3_1(const std::string & input) {
    // This version of Functionary still supports the llama 3.1 tool call format for the python tool.
    static const std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
    std::smatch match;
    if (std::regex_search(input, match, python_tag_regex)) {
        auto code = match[1].str();
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = match.prefix().str();
        msg.tool_calls.push_back({
            /* .name = */ "python",
            /* .arguments = */ (json {{"code", code}}).dump(),
            /* .id = */ "",
        });
        return msg;
    }
    static const std::regex function_regex(R"(<function=(\w+)>)");
    static const std::regex close_regex(R"(</function>)");
    // TODO: tighten & simplify.
    return parse_json_tool_calls(input, std::nullopt, function_regex, close_regex);
}

static common_chat_params common_chat_params_init_hermes_2_pro(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    // (content)?(<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>)*
    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;
        std::vector<std::string> tool_call_alts;
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
        builder.add_rule("root", inputs.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
        data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"});
        data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<function"});
        // Trigger on some common known "good bad" outputs (only from the start and with a json that's about a specific argument name to avoid false positives)
        data.grammar_triggers.push_back({
            COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START,
            "(?:```(?:json|xml)?\n\\s*)?(?:<function_call>|<tools>|<xml><json>|<response>)?\\s*\\{\\s*\"", //name\"\\s*:\\s*\"" + escaped_name + "\"",
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

    data.prompt = apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
    data.format = inputs.extract_reasoning ? COMMON_CHAT_FORMAT_HERMES_2_PRO_EXTRACT_REASONING : COMMON_CHAT_FORMAT_HERMES_2_PRO;
    return data;
}
static common_chat_msg common_chat_parse_hermes_2_pro(const std::string& input, bool extract_reasoning) {
    return handle_think_tag_prelude(input, extract_reasoning, [](const std::string & input) {
        static const std::regex open_regex(
            "(?:"
            "(```(?:xml|json)?\\n\\s*)?"         // match 1 (block_start)
            "(<tool_call>"                   // match 2 (open_tag)
            "|<function_call>"
            "|<tool>"
            "|<tools>"
            "|<response>"
            "|<json>"
            "|<xml>"
            "|<JSON>"
            ")?"
            "(\\s*\\{\\s*\"name\"\\s*:[\\s\\S]*)"    // match 3 (named tool call + rest)
            ")"
            "|"
            "(?:<function=([^>]+)>"            // match 4 (function name)
            "|<function name=\"([^\"]+)\">)" // match 5 (function name again)
            "([\\s\\S]*)"                   // match 6 (function arguments + rest)})"
        );

        try {
            common_chat_msg msg;
            msg.role = "assistant";

            std::string::const_iterator it = input.begin();
            const std::string::const_iterator end = input.end();
            std::smatch match;

            while (it != end) {
                if (std::regex_search(it, end, match, open_regex)) {
                    // Add content before the match
                    msg.content += std::string(it, match[0].first);

                    auto block_start = match[1].str();
                    std::string block_end = block_start.empty() ? "" : "```";

                    auto open_tag = match[2].str();
                    std::string close_tag;

                    if (match[3].matched) {
                        close_tag = open_tag.empty() ? "" : "</" + open_tag.substr(1);
                        auto json_it = match[3].first;
                        json tool_call;
                        if (parse_json(json_it, end, tool_call) && tool_call.contains("name") && tool_call.contains("arguments")) {

                            msg.tool_calls.emplace_back(process_tool_call(tool_call));
                            it = json_it;  // Move iterator past parsed JSON

                            // Handle close tags
                            consume_spaces(it, end);
                            if (!close_tag.empty() && !parse_literal(it, end, close_tag)) {
                                throw std::runtime_error("Failed to parse closing tag");
                            }
                            consume_spaces(it, end);
                            if (!block_end.empty() && !parse_literal(it, end, block_end)) {
                                throw std::runtime_error("Failed to parse block end");
                            }
                            consume_spaces(it, end);
                        } else {
                            // Not a valid tool call, treat as content
                            msg.content += std::string(match[0].first, match[0].second);
                            it = match[0].second;
                        }
                    } else {
                        auto function_name = match[4].str();
                        if (function_name.empty()) {
                            function_name = match[5].str();
                        }
                        GGML_ASSERT(!function_name.empty());

                        close_tag = "</function>";
                        // Start parsing from after the opening tags
                        auto json_it = match[6].first;
                        json arguments;
                        if (parse_json(json_it, end, arguments)) {
                            msg.tool_calls.emplace_back(process_tool_call({
                                {"name", function_name},
                                {"arguments", arguments},
                            }));
                            it = json_it;  // Move iterator past parsed JSON

                            // Handle close tags
                            consume_spaces(it, end);
                            if (!close_tag.empty() && !parse_literal(it, end, close_tag)) {
                                throw std::runtime_error("Failed to parse closing tag");
                            }
                            consume_spaces(it, end);
                            if (!block_end.empty() && !parse_literal(it, end, block_end)) {
                                throw std::runtime_error("Failed to parse block end");
                            }
                            consume_spaces(it, end);
                        } else {
                            // Not a valid tool call, treat as content
                            msg.content += std::string(match[0].first, match[0].second);
                            it = match[0].second;
                        }
                    }
                } else {
                    // Add remaining content
                    msg.content += std::string(it, end);
                    break;
                }
            }
            return msg;
        } catch (const std::exception & e) {
            LOG_ERR("Failed to parse hermes 2 pro input: %s\n", e.what());
            common_chat_msg msg;
            msg.role = "assistant";
            msg.content = input;
            return msg;
        }
    });
}

static common_chat_params common_chat_params_init_without_tools(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt = apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
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

static common_chat_params common_chat_templates_apply_jinja(
    const struct common_chat_templates * tmpls,
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
    params.extract_reasoning = inputs.extract_reasoning;
    params.tool_choice = inputs.tool_choice;
    params.grammar = inputs.grammar;
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

    // DeepSeek R1: use handler in all cases except json schema (thinking / tools).
    if (src.find("<｜tool▁calls▁begin｜>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_deepseek_r1(tmpl, params);
    }

    // Command R7B: : use handler in all cases except json schema (thinking / tools).
    if (src.find("<|END_THINKING|><|START_ACTION|>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_command_r7b(tmpl, params);
    }

    // Hermes 2/3 Pro, Qwen 2.5 Instruct (w/ tools)
    if (src.find("<tool_call>") != std::string::npos && params.json_schema.is_null() && params.tools.is_array() && params.json_schema.is_null()) {
        return common_chat_params_init_hermes_2_pro(tmpl, params);
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

    // Plain handler (no tools)
    if (params.tools.is_null() || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
        return common_chat_params_init_without_tools(tmpl, params);
    }

    // Functionary v3.1 (w/ tools)
    if (src.find("<|start_header_id|>") != std::string::npos
        && src.find("<function=") != std::string::npos) {
        return common_chat_params_init_functionary_v3_1_llama_3_1(tmpl, params);
    }

    // Llama 3.1, 3.2, 3.3 (w/ tools)
    if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        auto allow_python_tag_builtin_tools = src.find("<|python_tag|>") != std::string::npos;
        return common_chat_params_init_llama_3_1_tool_calls(tmpl, params, allow_python_tag_builtin_tools);
    }

    // Mistral Nemo (w/ tools)
    if (src.find("[TOOL_CALLS]") != std::string::npos) {
        return common_chat_params_init_mistral_nemo(tmpl, params);
    }

    // Generic fallback
    return common_chat_params_init_generic(tmpl, params);
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    int alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string> contents;
    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({msg.role.c_str(), content.c_str()});
        alloc_size += (msg.role.size() + content.size()) * 1.25;
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(), buf.size());
    }

    common_chat_params params;
    params.prompt = std::string(buf.data(), res);
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

common_chat_params common_chat_templates_apply(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja
        ? common_chat_templates_apply_jinja(tmpls, inputs)
        : common_chat_templates_apply_legacy(tmpls, inputs);
}

static common_chat_msg common_chat_parse_content_only(const std::string & input) {
    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = input;
    return msg;
}

common_chat_msg common_chat_parse(const std::string & input, common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            return common_chat_parse_content_only(input);
        case COMMON_CHAT_FORMAT_GENERIC:
            return common_chat_parse_generic(input);
        case COMMON_CHAT_FORMAT_MISTRAL_NEMO:
            return common_chat_parse_mistral_nemo(input);
        case COMMON_CHAT_FORMAT_LLAMA_3_X:
            return common_chat_parse_llama_3_1(input);
        case COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS:
            return common_chat_parse_llama_3_1(input, /* with_builtin_tools= */ true);
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1:
            return common_chat_parse_deepseek_r1(input, /* extract_reasoning= */ false);
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING:
            return common_chat_parse_deepseek_r1(input, /* extract_reasoning= */ true);
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2:
            return common_chat_parse_functionary_v3_2(input);
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1:
            return common_chat_parse_functionary_v3_1_llama_3_1(input);
        case COMMON_CHAT_FORMAT_HERMES_2_PRO:
            return common_chat_parse_hermes_2_pro(input, /* extract_reasoning= */ false);
        case COMMON_CHAT_FORMAT_HERMES_2_PRO_EXTRACT_REASONING:
            return common_chat_parse_hermes_2_pro(input, /* extract_reasoning= */ true);
        case COMMON_CHAT_FORMAT_FIREFUNCTION_V2:
            return common_chat_parse_firefunction_v2(input);
        case COMMON_CHAT_FORMAT_COMMAND_R7B:
            return common_chat_parse_command_r7b(input, /* extract_reasoning= */ false);
        case COMMON_CHAT_FORMAT_COMMAND_R7B_EXTRACT_REASONING:
            return common_chat_parse_command_r7b(input, /* extract_reasoning= */ true);
        default:
            throw std::runtime_error("Unsupported format: " + common_chat_format_name(format));
    }
}
