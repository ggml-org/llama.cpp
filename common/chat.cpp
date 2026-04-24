#include "chat.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "log.h"

#include "jinja/value.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "peg-parser.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto               time       = std::chrono::system_clock::to_time_t(now);
    auto               local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static json safe_args_parse(const std::string & to_parse) {
    std::string stripped = to_parse;
    if (to_parse.at(0) == '"' && to_parse.at(to_parse.length() - 1) == '"') {
        stripped = to_parse.substr(1, to_parse.length() - 1);
    }
    try {
        return json::parse(stripped);
    } catch (json::exception & e) {
        return stripped;
    }
}

static std::string string_diff(const std::string & last, const std::string & current) {
    if (last.empty()) {
        return current;
    }
    if (!string_starts_with(current, last)) {
        if (string_starts_with(last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    return current.substr(last.size());
}

static bool has_content_or_tool_calls(const common_chat_msg & msg) {
    return !msg.content.empty() || !msg.tool_calls.empty();
}

json common_chat_msg::to_json_oaicompat(bool concat_typed_text) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    json jmsg {
        {"role", role},
    };
    if (!content.empty()) {
        jmsg["content"] = content;
    } else if (!content_parts.empty()) {
        if (concat_typed_text) {
            std::string text;
            bool last_was_media_marker = false;
            // join parts with newline, do not add newline before or after media markers
            for (const auto & part : content_parts) {
                bool add_new_line = true;
                if (part.type == "text") {
                    add_new_line = !last_was_media_marker && !text.empty();
                    last_was_media_marker = false;
                } else if (part.type == "media_marker") {
                    add_new_line = false;
                    last_was_media_marker = true;
                } else {
                    LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                    continue;
                }

                if (add_new_line) {
                    text += '\n';
                }

                text += part.text;
            }
            jmsg["content"] = text;
        } else {
            auto & parts = jmsg["content"] = json::array();
            for (const auto & part : content_parts) {
                parts.push_back({
                    {"type", part.type},
                    {"text", part.text},
                });
            }
        }
    } else {
        jmsg["content"] = "";
    }
    if (!reasoning_content.empty()) {
        jmsg["reasoning_content"] = reasoning_content;
    }
    if (!tool_name.empty()) {
        jmsg["name"] = tool_name;
    }
    if (!tool_call_id.empty()) {
        jmsg["tool_call_id"] = tool_call_id;
    }
    if (!tool_calls.empty()) {
        jmsg["tool_calls"] = json::array();
        auto & jtool_calls = jmsg["tool_calls"];
        for (const auto & tool_call : tool_calls) {
            json tc {
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", json(tool_call.arguments)},
                }},
            };
            if (!tool_call.id.empty()) {
                tc["id"] = tool_call.id;
            }
            // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
            // We only generate a random id for the ones that don't generate one by themselves
            // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
            // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            jtool_calls.push_back(tc);
        }
    }

    return jmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & msg_prv,
                                                                      const common_chat_msg & msg_new) {
    std::vector<common_chat_msg_diff> diffs;
    if (msg_new.tool_calls.size() > msg_prv.tool_calls.size()) {
        diffs.reserve(msg_new.tool_calls.size() - msg_prv.tool_calls.size() + 3);
    } else {
        diffs.reserve(3);
    }

    // TODO: these can become expensive for long messages - how to optimize?
    if (msg_prv.reasoning_content != msg_new.reasoning_content) {
        auto & diff                  = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(msg_prv.reasoning_content, msg_new.reasoning_content);
    }
    if (msg_prv.content != msg_new.content) {
        auto & diff        = diffs.emplace_back();
        diff.content_delta = string_diff(msg_prv.content, msg_new.content);
    }

    if (msg_new.tool_calls.size() < msg_prv.tool_calls.size()) {
        std::string err = "Invalid diff: now finding less tool calls!\n";
        err += "  Previous (" + std::to_string(msg_prv.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_prv.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current (" + std::to_string(msg_new.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_new.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current msg text content:\n" + msg_new.content + "\n";
        throw std::runtime_error(err);
    }

    if (!msg_prv.tool_calls.empty()) {
        const auto   idx  = msg_prv.tool_calls.size() - 1;
        const auto & pref = msg_prv.tool_calls[idx];
        const auto & newf = msg_new.tool_calls[idx];
        // Allow tool name to change during incremental parsing:
        // - empty -> non-empty (initial discovery)
        // - prefix -> longer string (name grows as more input is parsed)
        if (pref.name != newf.name && !pref.name.empty() && !newf.name.empty()) {
            // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
            bool is_prefix = (newf.name.rfind(pref.name, 0) == 0);
            if (!is_prefix) {
                LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", pref.name.c_str(), newf.name.c_str());
                throw std::runtime_error("Invalid diff: tool call mismatch!");
            }
        }
        const auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id || pref.name != newf.name) {
            auto & diff          = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id || pref.name != newf.name) {
                diff.tool_call_delta.id   = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = msg_prv.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        auto & diff          = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
    }

    return diffs;
}

using chat_template_caps = jinja::caps;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template;  // Model had builtin template or template overridden was specified.
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
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
    throw std::invalid_argument("Invalid tool_choice: " + tool_choice);
}

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates) {
    common_chat_templates_inputs inputs;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "test";
    inputs.messages = { msg };
    inputs.enable_thinking = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    auto params = common_chat_templates_apply(chat_templates, inputs);
    return params.supports_thinking;
}

std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {
        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content    = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text" && type != "media_marker") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " +
                                                content.dump() +
                                                " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name           = fc.at("name");
                    const auto & args = fc.at("arguments");
                    if (args.is_string()) {
                        tc.arguments = args;
                    } else {
                        tc.arguments = args.dump();
                    }
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument(
                    "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & "
                    "https://github.com/ggml-org/llama.cpp/issues/12279)");
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
        // @ngxson : disable otherwise it's bloating the API response
        // printf("%s\n", std::string("; messages = ") + messages.dump(2));
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }

    return msgs;
}

static json render_message_to_json(const std::vector<common_chat_msg> & msgs, const jinja::caps & c) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\n", __func__);
    }

    bool only_string_accepted =  c.supports_string_content && !c.supports_typed_content;
    bool only_typed_accepted  = !c.supports_string_content &&  c.supports_typed_content;

    json messages = json::array();
    for (const auto & msg : msgs) {
        if (only_string_accepted) {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ true);
            messages.push_back(jmsg);
        } else if (only_typed_accepted) {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ false);
            if (jmsg.at("content").is_string()) {
                jmsg["content"] = json::array({
                    json{
                        {"type", "text"},
                        {"text", jmsg.at("content").get<std::string>()},
                    }
                });
            }
            messages.push_back(jmsg);
        } else {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ false);
            messages.push_back(jmsg);
        }
    }
    return messages;
}

// DEPRECATED: only used in tests
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    jinja::caps c;
    c.supports_string_content = true;
    c.supports_typed_content = !concat_typed_text;
    return render_message_to_json(msgs, c);
}

json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function", {
                { "name", tool.name },
                { "description", tool.description },
                { "parameters", json::parse(tool.parameters) },
            }},
        });
    }
    return result;
}

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.value("description", ""),
                    /* .parameters = */ function.value("parameters", json::object()).dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

bool common_chat_verify_template(const std::string & tmpl, bool use_jinja) {
    if (use_jinja) {
        try {
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = "test";

            auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);

            common_chat_templates_inputs inputs;
            inputs.messages = { msg };

            common_chat_templates_apply(tmpls.get(), inputs);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to apply template: %s\n", __func__, e.what());
            return false;
        }
    }
    llama_chat_message chat[] = {
        { "user", "test" }
    };
    const int res = llama_chat_apply_template(tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos   = tmpls->add_bos;
    inputs.add_eos   = tmpls->add_eos;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages              = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg                 = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg             = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja            = use_jinja;
    inputs.add_bos              = tmpls->add_bos;
    inputs.add_eos              = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg         = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system", "You are a helpful assistant");
    add_simple_msg("user", "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user", "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

#define CHATML_TEMPLATE_SRC                                                               \
    "{%- for message in messages -%}\n"                                                   \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n"                                                                    \
    "{%- if add_generation_prompt -%}\n"                                                  \
    "  {{- '<|im_start|>assistant\n' -}}\n"                                               \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

// LFM2 format detection: template uses <|tool_list_start|>[...]<|tool_list_end|> around the tool list
// and <|tool_call_start|>[...]<|tool_call_end|> around each tool call
static bool is_lfm2_template(const std::string & src) {
    return src.find("<|tool_list_start|>") != std::string::npos &&
           src.find("<|tool_list_end|>")   != std::string::npos;
}

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates) {
    common_chat_prompt_preset asr_preset;
    asr_preset.system = "";
    asr_preset.user   = "Transcribe audio to text";

    if (chat_templates && chat_templates->template_default && is_lfm2_template(chat_templates->template_default->source())) {
        asr_preset.system = "Perform ASR.";
        asr_preset.user   = "";
    }

    return asr_preset;
}

std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant) {
    if (!variant.empty()) {
        if (variant == "tool_use") {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source();
            }
            return "";
        }
        LOG_DBG("%s: unknown template variant: %s\n", __func__, variant.c_str());
    }
    return tmpls->template_default->source();
}

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override,
                                                     const std::string &        eos_token_override) {
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src  = str;
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

    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (default_template_src.find("<|channel|>") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("in message.content or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
                           "\"<|channel|>final<|message|>\" in message.content %}",
                           "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (default_template_src.find("[TOOL_CALLS]") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("if (message['content'] is none or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if (message['content'] is none or message['content'] == '' or "
                           "message['content']|length == 0) and (message['tool_calls'] is not defined or "
                           "message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
                           "{%- if false %}");
    }

    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool        add_bos   = false;
    bool        add_eos   = false;
    if (model) {
        const auto * vocab     = llama_model_get_vocab(model);
        const auto   get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos ||
                    template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN(
                        "common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't "
                          "work as intended.\n",
                        name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
        add_bos   = llama_vocab_get_add_bos(vocab);
        add_eos   = llama_vocab_get_add_eos(vocab);
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos               = add_bos;
    tmpls->add_eos               = add_eos;
    try {
        tmpls->template_default = std::make_unique<common_chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        LOG_ERR("%s: failed to initialize chat template\n", __func__);
        LOG_ERR("%s: please consider disabling jinja via --no-jinja, or using another chat template\n", __func__);
        throw e;
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<common_chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

const char * common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            return "Content-only";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE:
            return "peg-simple";
        case COMMON_CHAT_FORMAT_PEG_NATIVE:
            return "peg-native";
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return "peg-gemma4";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:
            return "none";
        case COMMON_REASONING_FORMAT_AUTO:
            return "auto";
        case COMMON_REASONING_FORMAT_DEEPSEEK:
            return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY:
            return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

common_reasoning_format common_reasoning_format_from_name(const std::string & format) {
    if (format == "none") {
        return COMMON_REASONING_FORMAT_NONE;
    }
    if (format == "auto") {
        return COMMON_REASONING_FORMAT_AUTO;
    }
    if (format == "deepseek") {
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    if (format == "deepseek-legacy") {
        return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    }
    throw std::runtime_error("Unknown reasoning format: " + format);
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

static void foreach_parameter(const json &                                                         function,
                              const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto &          props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
}

static std::string common_chat_template_direct_apply_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt) {
    jinja::context ctx(tmpl.source());

    nlohmann::ordered_json inp = nlohmann::ordered_json{
        {"messages", messages_override.has_value() ? *messages_override : inputs.messages},
        {"bos_token", tmpl.bos_token()},
        {"eos_token", tmpl.eos_token()},
        {"enable_thinking", inputs.enable_thinking},
    };
    if (tools_override.has_value() || !inputs.tools.empty()) {
        inp["tools"] = tools_override.has_value() ? *tools_override : inputs.tools;
    }
    if (inputs.extra_context.is_object()) {
        // TODO: do we need to merge, or replacing is fine?
        for (const auto & [k, v] : inputs.extra_context.items()) {
            inp[k] = v;
        }
    }
    if (additional_context.has_value()) {
        // TODO: merge properly instead of overwriting (matching old behavior)
        for (const auto & [k, v] : additional_context->items()) {
            inp[k] = v;
        }
    }
    if (inputs.add_generation_prompt) {
        inp["add_generation_prompt"] = true;
    }

    jinja::global_from_json(ctx, inp, inputs.mark_input);

    // render
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    auto parts = jinja::runtime::gather_string_parts(results);

    std::string result = parts->as_string().str();

    // TODO: improve this later
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

std::string common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

static common_chat_params common_chat_params_init_ministral_3(const common_chat_template &    tmpl,
                                                              const autoparser::generation_params & inputs) {
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
                { "type",     "thinking"                                     },
                { "thinking", msg.at("reasoning_content").get<std::string>() },
            });
        }

        // If message contains `content`, add it as a block of type `text`
        if (msg.contains("content")) {
            if (msg.at("content").is_string()) {
                content.push_back({
                    { "type", "text"                               },
                    { "text", msg.at("content").get<std::string>() },
                });
            } else if (msg.at("content").is_array()) {
                auto blocks = msg.at("content");
                content.insert(content.end(), blocks.begin(), blocks.end());
            }
        }

        auto adjusted       = msg;
        adjusted["content"] = content;
        adjusted.erase("reasoning_content");
        adjusted_messages.push_back(adjusted);
    }

    auto has_tools            = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format  = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning    = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar      = true;

    data.supports_thinking  = true;
    data.thinking_start_tag = "[THINK]";
    data.thinking_end_tag   = "[/THINK]";
    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, /* messages_override = */ adjusted_messages);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = {
        "[THINK]",
        "[/THINK]",
        "[TOOL_CALLS]",
        "[ARGS]",
    };

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, "[THINK]");
        auto reasoning =
            extract_reasoning ? p.optional("[THINK]" + p.reasoning(p.until("[/THINK]")) + "[/THINK]") : p.eps();

        // Response format parser
        if (has_response_format) {
            // Ministral wants to emit json surrounded by code fences
            return generation_prompt + (reasoning << "```json" << p.content(p.schema(p.json(), "response-format", inputs.json_schema)) << "```");
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto & schema   = function.at("parameters");

                tool_choice |=
                    p.rule("tool-" + name, p.tool_open(p.tool_name(p.literal(name)) + "[ARGS]") +
                                               p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)));
            });

            auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat("[TOOL_CALLS]" + tool_choice, min_calls, max_calls));

            return generation_prompt + (reasoning << p.content(p.until("[TOOL_CALLS]")) << tool_calls);
        }

        // Content only parser
        include_grammar = false;
        return generation_prompt + (reasoning << p.content(p.rest()));
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
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
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]" }
        };
    }

    return data;
}

static common_chat_params common_chat_params_init_gpt_oss(const common_chat_template &    tmpl,
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
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = {
        "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>",
    };

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

static common_chat_params common_chat_params_init_gemma4(const common_chat_template &    tmpl,
                                                         const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);

    if (inputs.add_generation_prompt && string_ends_with(data.prompt, "<turn|>\n")) {
        // This may happen if the model generates content + tool_call, the
        // template does not add the model's next turn and confuses the model
        // from emitting its proper reasoning token sequence.
        data.prompt += "<|turn>model\n";
    }

    data.format            = COMMON_CHAT_FORMAT_PEG_GEMMA4;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<|channel>thought";
    data.thinking_end_tag   = "<channel|>";

    data.preserved_tokens = {
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|turn>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.prefix(inputs.generation_prompt, "<|channel>"));

        if (extract_reasoning) {
            p.rule("thought", p.literal("<|channel>thought") + p.space() + p.reasoning(p.until("<channel|>")) + p.literal("<channel|>"));
        } else {
            p.rule("thought", p.content(p.literal("<|channel>thought") + p.space() + p.until("<channel|>") + p.literal("<channel|>")));
        }

        auto consume_empty_channels = p.gbnf(p.zero_or_more(p.literal("<|channel>") + p.negate(p.literal("thought"))), "");
        auto thought = (p.peek(p.literal("<|channel>")) + consume_empty_channels + p.ref("thought")) | p.negate(p.literal("<|channel>"));

        if (has_response_format) {
            auto response_format = p.literal("```json") <<
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) <<
                p.literal("```");
            return start + p.optional(thought) + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Gemma4 tool calling syntax
            // Rules should match traversal logic in gemma4_to_json()
            p.rule("gemma4-string-content", p.until("<|\"|>"));
            p.rule("gemma4-string", p.literal("<|\"|>") + p.ref("gemma4-string-content") + p.literal("<|\"|>"));
            p.rule("gemma4-bool", p.json_bool());
            p.rule("gemma4-null", p.json_null());
            p.rule("gemma4-number", p.json_number());
            p.rule("gemma4-dict-key", p.rule("gemma4-dict-key-name", p.chars("[^:}]", 1, -1)) + p.literal(":"));
            p.rule("gemma4-dict-kv", p.ref("gemma4-dict-key") + p.space() + p.ref("gemma4-value"));
            p.rule("gemma4-dict", [&]() {
                auto ws = p.space();
                auto member = p.ref("gemma4-dict-kv");
                auto members = p.sequence({member, p.zero_or_more(p.sequence({p.literal(","), ws, member}))});
                return p.sequence({
                    p.literal("{"), ws,
                    p.choice({p.literal("}"), p.sequence({members, ws, p.literal("}")})})
                });
            });
            p.rule("gemma4-array", [&]() {
                auto ws = p.space();
                auto value = p.ref("gemma4-value");
                auto elements = p.sequence({value, p.zero_or_more(p.sequence({p.literal(","), ws, value}))});
                return p.sequence({
                    p.literal("["), ws,
                    p.choice({p.literal("]"), p.sequence({elements, ws, p.literal("]")})})
                });
            });
            p.rule("gemma4-value", [&]() {
                return p.choice({
                    p.ref("gemma4-string"), p.ref("gemma4-dict"), p.ref("gemma4-array"),
                    p.ref("gemma4-number"), p.ref("gemma4-bool"), p.ref("gemma4-null")
                });
            });

            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                // TODO @aldehir : need to extend json-schema-to-grammar to produce more than JSON rules
                // const auto & params   = function.at("parameters");

                tool_choice |= p.rule("tool-" + name, p.tool(p.sequence({
                    p.tool_open(p.tool_name(p.literal(name)) + p.peek(p.literal("{"))),
                    p.tool_args(p.ref("gemma4-dict")),
                })));
            });

            auto tool_call = p.trigger_rule("tool-call", p.repeat(
                "<|tool_call>call:" + tool_choice + "<tool_call|>",
                /* min = */ inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0,
                /* max = */ inputs.parallel_tool_calls ? -1 : 1
            ));

            auto scan_to_toolcall = p.rule("scan-to-toolcall", p.until("<|tool_call>"));
            auto content = p.rule("content", p.content(p.until_one_of({"<|channel>", "<channel|>", "<|tool_call>"})));
            auto message = p.rule("message", thought + content);
            return start + p.zero_or_more(message) + scan_to_toolcall + tool_call;
        }

        // Gemma 4 may emit an extra <|channel>thought\n<channel|> at the end of the content. It may
        // also emit a single trailing <channel|> token. Consume all complete reasoning blocks and
        // then stop at the first unmatched <channel|> token.
        auto content = p.rule("content", p.content(p.until_one_of({"<|channel>", "<channel|>"})));
        auto message = p.rule("message", thought + content);
        return start + p.one_or_more(message);
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
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call>" },
        };
    }

    return data;
}

// Functionary v3.2 - uses recipient-based format: >>>recipient\n{content}
static common_chat_params common_chat_params_init_functionary_v3_2(const common_chat_template &    tmpl,
                                                                   const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt           = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format           = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = {
        ">>>all",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Functionary v3.2 format:
        // - Normal content: >>>all\n{content}
        // - Tool calls: >>>function_name\n{json_args}
        // Generation prompt ends with ">>>" so model outputs recipient immediately

        // Build content parser for >>>all\n{content}
        // When tools are present, content stops before the next ">>>" (tool call)
        // When no tools, content goes until end
        auto content_until_tool = p.literal("all\n") + p.content(p.until(">>>"));
        auto content_until_end  = p.literal("all\n") + p.content(p.rest());
        auto generation_prompt  = p.literal(inputs.generation_prompt);

        // If no tools or tool_choice is NONE, just parse content
        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            // When no tools, just match the prefix and capture everything after
            return generation_prompt + content_until_end + p.end();
        }

        // Build tool call parsers for each available function
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            // Tool format: >>>function_name\n{json_args}
            auto tool_parser = p.tool(
                p.tool_open(p.tool_name(p.literal(name)) + p.literal("\n")) +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema))
            );

            tool_choice |= p.rule("tool-" + name, tool_parser);
        });

        auto content_only = content_until_end;
        auto tools_only = p.trigger_rule("tools", p.one_or_more(tool_choice));
        auto content_and_tools = content_until_tool + tools_only;

        auto ret = p.eps();
        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            if (inputs.parallel_tool_calls) {
                ret = p.choice({ content_and_tools, tools_only }) + p.end();
            } else {
                ret = p.choice({ content_until_tool + tool_choice, tools_only }) + p.end();
            }
        } else if (inputs.parallel_tool_calls) {
            ret = p.choice({ content_and_tools, content_only, tools_only }) + p.end();
        } else {
            auto content_and_tool = content_until_tool + tool_choice;
            ret = p.choice({ content_and_tool, content_only, tool_choice }) + p.end();
        }
        return generation_prompt + ret;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Grammar trigger for when the model starts outputting a tool call
        // (after the initial ">>>" in the generation prompt but recipient other than "all")
        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, ">>>(?!all)" }
        };
    }

    return data;
}

// Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
// The ID contains both the function name and an incrementing counter
static common_chat_params common_chat_params_init_kimi_k2(const common_chat_template &    tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.preserved_tokens  = {
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    const std::string SECTION_BEGIN = "<|tool_calls_section_begin|>";
    const std::string SECTION_END   = "<|tool_calls_section_end|>";
    const std::string CALL_BEGIN    = "<|tool_call_begin|>";
    const std::string ARGS_BEGIN    = "<|tool_call_argument_begin|>";
    const std::string CALL_END      = "<|tool_call_end|>";

    const std::string THINK_START = "<think>";
    const std::string THINK_END   = "</think>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tag   = THINK_END;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Kimi K2 Thinking format:
        // - Reasoning: <think>{reasoning}</think>
        // - Content: text after reasoning
        // - Tool calls section:
        //   <|tool_calls_section_begin|>
        //   <|tool_call_begin|>functions.<name>:<index><|tool_call_argument_begin|>{json_args}<|tool_call_end|>
        //   ...
        //   <|tool_calls_section_end|>
        // The ID format is: functions.<function_name>:<counter> where counter is 0, 1, 2, ...

        // Tool call markers
        auto end = p.end();

        // Note: this model is CRAZY. It can diverge from its supposed tool calling pattern in so many ways it's not funny.
        // For example, it can call tools at the end of reasoning without closing reasoning...
        auto reasoning = extract_reasoning ? p.optional(THINK_START + p.reasoning(
            p.until_one_of({ THINK_END, "<|tool_calls_section_begin|>", "<|tool_call_begin|>" })) +
            p.optional(p.literal(THINK_END))) : p.eps();
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);


        // Content only parser (no tools)
        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        // Build tool call parsers for each available function
        // The ID format is: functions.<name>:<index>
        // We need to match: functions.<name>:<digits>
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            // Match: functions.<name>:<digits>
            // Capture the full call id (functions.<name>:<digits>) using tool_id tag
            auto tool_id = p.tool_id(p.literal("functions.") + p.tool_name(p.literal(name)) + p.literal(":") + p.chars("[0-9]", 1, -1));
            auto tool_parser = p.tool(
                p.tool_open(tool_id + p.literal(ARGS_BEGIN)) +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)) +
                p.tool_close(p.optional((p.literal(CALL_END))))
            );

            tool_choice |= p.rule("tool-" + name, tool_parser);
        });

        // Tool calls section: <|tool_calls_section_begin|> tool_calls <|tool_calls_section_end|>
        auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
        auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
        // Use trigger_rule so grammar generator knows where to start generating rules
        auto tool_calls = p.rule("tool-calls",
            p.optional(p.literal(SECTION_BEGIN)) +
            p.trigger_rule("tool-call", p.repeat(CALL_BEGIN + tool_choice, min_calls, max_calls) +
                p.optional(p.literal(SECTION_END)))
        );

        auto content_before_tools = p.content(p.until_one_of({ SECTION_BEGIN, CALL_BEGIN }));

        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call_begin|>" }
        };
    }

    return data;
}

// LFM2 format: uses <|tool_list_start|>[...]<|tool_list_end|> in system prompt
// and <|tool_call_start|>[name(arg="val")]<|tool_call_end|> for tool calls.
// - Reasoning: <think>{reasoning}</think> (optional)
// - Content: text before a tool call (optional)
// - Tool calls: Python-style, e.g. [function_name(arg1="value1", arg2="value2")]
//   Tool calls can appear multiple times (parallel tool calls supported)
static common_chat_params common_chat_params_init_lfm2(const common_chat_template &    tmpl,
                                                       const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<|tool_list_start|>",
        "<|tool_list_end|>",
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    const std::string TOOL_CALL_START = "<|tool_call_start|>";
    const std::string TOOL_CALL_END   = "<|tool_call_end|>";
    const std::string THINK_START     = "<think>";
    const std::string THINK_END       = "</think>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tag   = THINK_END;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }
        auto tool_calls = p.rule("tool-calls",
            p.trigger_rule("tool-call",
                p.literal(TOOL_CALL_START) +
                p.python_style_tool_calls(inputs.tools, inputs.parallel_tool_calls) +
                p.literal(TOOL_CALL_END)
            )
        );

        auto content = p.content(p.until(TOOL_CALL_START));

        return generation_prompt + reasoning + content + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, TOOL_CALL_START }
        };
    }
    return data;
}

// LFM2.5 format: uses plain "List of tools: [...]" in system prompt, no wrapper tokens.
// Tool calls are bare [name(arg="val")], though model may optionally emit <|tool_call_start|>.
// - Reasoning: <think>{reasoning}</think> (optional)
// - Content: text before a tool call (optional)
// - Tool calls: Python-style, e.g. [function_name(arg1="value1", arg2="value2")]
//   Tool calls can appear multiple times (parallel tool calls supported)
static common_chat_params common_chat_params_init_lfm2_5(const common_chat_template &    tmpl,
                                                         const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    const std::string THINK_START     = "<think>";
    const std::string THINK_END       = "</think>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tag   = THINK_END;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_calls = p.rule("tool-calls",
            p.trigger_rule("tool-call",
                p.python_style_tool_calls(inputs.tools, inputs.parallel_tool_calls)
            )
        );

        auto content = p.content(p.until_one_of({"<|tool_call_start|>", "["}));
        auto maybe_start = p.optional(p.literal("<|tool_call_start|>"));
        return generation_prompt + reasoning + content + maybe_start + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });
        foreach_function(inputs.tools, [&](const json & tool) {
            const std::string name = tool.at("function").at("name");
            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[" + name + "(" });
        });
    }

    return data;
}

static common_chat_params common_chat_params_init_gigachat_v3(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = false;
    data.preserved_tokens  = {
        "<|message_sep|>\n\n",
        "<|role_sep|>\n",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    const auto *tool_call_start_prefix = "<|message_sep|>\n\nfunction call<|role_sep|>\n";

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto ret = p.eps();
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Build a choice of all available tools
            auto tool_choice = p.choice();
            for (const auto & tool : inputs.tools) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                const auto & schema = function.at("parameters");

                auto tool_name = p.json_member("name", "\"" + p.tool_name(p.literal(name)) + "\"");
                auto tool_args = p.json_member("arguments", p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)));

                auto tool_open = p.tool_open(p.literal("{") << tool_name);

                tool_choice |= p.rule("tool-" + name, tool_open << "," << tool_args << "}");
            }

            // Define the tool call structure
            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = 1; // parallel toolcalls are not supported
            auto tool_call = p.rule("tool-call", p.literal(tool_call_start_prefix) + tool_choice);
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, /* min = */ min_calls, /* max = */ max_calls));

            ret = p.content(p.until("<|message_sep|>\n\n")) << tool_calls;
        } else {
            // Content only parser
            include_grammar = false;
            ret = p.content(p.rest());
        }

        return p.literal(inputs.generation_prompt) + ret;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto schema = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, tool_call_start_prefix}
        };
    }
    return data;
}

static common_chat_params common_chat_params_init_deepseek_v3_2(const common_chat_template &    tmpl,
                                                                 const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tag   = "</think>";
    data.preserved_tokens  = {
        "｜DSML｜",
        "<think>",
        "</think>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    const std::string DSML         = "｜DSML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string FC_START     = "<" + DSML + "function_calls>";
    const std::string FC_END       = "</" + DSML + "function_calls>";
    const std::string INVOKE_START = "<" + DSML + "invoke";
    const std::string INVOKE_END   = "</" + DSML + "invoke>";
    const std::string PARAM_START  = "<" + DSML + "parameter";
    const std::string PARAM_END    = "</" + DSML + "parameter>";

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        } else if (extract_reasoning) {
            // Thinking disabled but reasoning extraction requested: the generation prompt
            // contains an empty <think></think> pair that must still be consumed.
            reasoning = p.optional(p.literal(THINK_START) + p.until(THINK_END) + p.literal(THINK_END));
        }

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                p.literal("```json") + p.space() +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.space() + p.literal("```"));
            return generation_prompt + reasoning + response_format + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            auto params   = function.contains("parameters") ? function.at("parameters") : json::object();
            const auto & props    = params.contains("properties") ? params.at("properties") : json::object();

            std::set<std::string> required;
            if (params.contains("required")) {
                params.at("required").get_to(required);
            }

            auto schema_info = common_schema_info();
            schema_info.resolve_refs(params);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            for (const auto & [param_name, param_schema] : props.items()) {
                bool is_required = required.find(param_name) != required.end();
                bool is_string   = schema_info.resolves_to_string(param_schema);

                auto arg = p.tool_arg(
                    p.tool_arg_open(
                        p.literal(PARAM_START + " name=\"") +
                        p.tool_arg_name(p.literal(param_name)) +
                        p.literal("\" string=\"" + std::string(is_string ? "true" : "false") + "\">")) +
                    (is_string
                         ? p.tool_arg_string_value(p.until(PARAM_END))
                         : p.tool_arg_json_value(p.schema(p.json(),
                                                          "tool-" + name + "-arg-" + param_name + "-schema",
                                                          param_schema, false))) +
                    p.tool_arg_close(p.literal(PARAM_END)));

                auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
                if (is_required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            }

            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < required_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + required_parsers[i];
            }

            if (!optional_parsers.empty()) {
                common_peg_parser any_opt = p.choice();
                for (const auto & opt : optional_parsers) {
                    any_opt |= opt;
                }
                args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
            }

            common_peg_parser invoke_body = args_seq;
            auto func_parser = p.tool(
                p.tool_open(p.literal(INVOKE_START + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                invoke_body + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        auto content_before_tools = p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
        };
    }

    return data;
}

namespace workaround {

static void map_developer_role_to_system(json & messages) {
    for (auto & message : messages) {
        if (message.contains("role")) {
            if (message["role"] == "developer") {
                message["role"] = "system";
            }
        }
    }
}


// if first message is system and template does not support it, merge it with next message
static void system_message_not_supported(json & messages) {
    if (!messages.empty() && messages.front().at("role") == "system") {
        if (messages.size() > 1) {
            LOG_DBG("Merging system prompt into next message\n");
            auto & first_msg = messages.front();
            auto & second_msg = messages[1];
            second_msg["content"] = first_msg.at("content").get<std::string>()
                + "\n" + second_msg.at("content").get<std::string>();
            messages.erase(messages.begin());
        } else {
            LOG_WRN("Removing system prompt due to template not supporting system role\n");
            messages.erase(messages.begin());
        }
    }
}

static void requires_non_null_content(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls") && !message.contains("content")) {
            message["content"] = "";
        }
    }
}

// Gemma4 uses a custom tool_responses field instead of role:tool messages.
//
// This will transform a sequence of messages:
//   assistant(tool_call+) -> tool+ -> assistant(content)
//
// Into a single assistant message containing a tool_responses field:
//   assistant(content + tool_call + tool_responses)
//
// This is necessary for the Gemma4 chat template to properly format the prompt.
// See https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
struct gemma4_model_turn_builder {
    json & messages;
    size_t pos;
    json tool_calls = json::array();
    json tool_responses = json::array();
    json content;
    json reasoning_content;

    gemma4_model_turn_builder(json & msgs, size_t pos) : messages(msgs), pos(pos) {}

    void collect() {
        // Collect the first assistant message
        auto & msg = messages[pos];
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            // According to the prompt formatting guide, we need to preserve reasoning_content
            // between function calls. The current chat templates do not support this, but we will do it anyway.
            reasoning_content = msg.at("reasoning_content");
        }
        for (auto & tc : msg.at("tool_calls")) {
            tool_calls.push_back(tc);
        }
        pos++;

        // Collect tool call results
        while (pos < messages.size() && messages[pos].value("role", "") == "tool") {
            collect_result(messages[pos]);
            pos++;
        }

        // Check if the next assistant message is the final message
        if (pos < messages.size() && messages[pos].value("role", "") == "assistant") {
            auto & next = messages[pos];
            if (!has_tool_calls(next) && has_content(next)) {
                content = next.at("content");
                pos++;
            }
        }
    }

    void collect_result(const json & curr) {
        json response;
        if (curr.contains("content")) {
            const auto & content = curr.at("content");
            if (content.is_string()) {
                // Try to parse the content as JSON; fall back to raw string
                try {
                    response = json::parse(content.get<std::string>());
                } catch (...) {
                    response = content;
                }
            } else {
                response = content;
            }
        }

        std::string name;

        // Match name with corresponding tool call
        size_t idx = tool_responses.size();
        if (idx < tool_calls.size()) {
            auto & tc = tool_calls[idx];
            if (tc.contains("function")) {
                name = tc.at("function").value("name", "");
            }
        }

        // Fallback to the tool call id
        if (name.empty()) {
            name = curr.value("tool_call_id", "");
        }

        tool_responses.push_back({{"name", name}, {"response", response}});
    }

    json build() {
        collect();

        json msg = {
            {"role", "assistant"},
            {"tool_calls", tool_calls},
        };
        if (!tool_responses.empty()) {
            msg["tool_responses"] = tool_responses;
        }
        if (!content.is_null()) {
            msg["content"] = content;
        }
        if (!reasoning_content.is_null()) {
            msg["reasoning_content"] = reasoning_content;
        }
        return msg;
    }

    static bool has_content(const json & msg) {
        if (!msg.contains("content") || msg.at("content").is_null()) {
            return false;
        }
        const auto & content = msg.at("content");
        if (content.is_string() && !content.get<std::string>().empty()) {
            return true;
        }
        if (content.is_array() && !content.empty()) {
            return true;
        }
        return false;
    }

    static bool has_tool_calls(const json & msg) {
        return msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty();
    }
};

static void convert_tool_responses_gemma4(json & messages) {
    json result = json::array();
    size_t i = 0;

    while (i < messages.size()) {
        auto & msg = messages[i];

        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") ||
            !msg.at("tool_calls").is_array() || msg.at("tool_calls").empty()) {
            result.push_back(msg);
            i++;
            continue;
        }

        gemma4_model_turn_builder builder(messages, i);
        result.push_back(builder.build());
        i = builder.pos;
    }

    messages = result;
}

static void func_args_not_string(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls")) {
            for (auto & tool_call : message["tool_calls"]) {
                if (tool_call.contains("function") && tool_call["function"].contains("arguments")) {
                    auto & args = tool_call["function"]["arguments"];
                    if (args.is_string()) {
                        try {
                            args = json::parse(args.get<std::string>());
                        } catch (const std::exception & e) {
                            throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(e.what()));
                        }
                    }
                }
            }
        }
    }
}

}

static json common_chat_extra_context() {
    json ctx = json::object();
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::string datetime_str = format_time(now, "%b %d %Y");
    std::string date_str = format_time(now, "%d %b %Y");
    ctx["datetime"] = datetime_str;
    ctx["date_string"] = date_str;
    return ctx;
}

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const std::string &                   src,
        autoparser::generation_params & params) {
    // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
    // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
    if (src.find("[SYSTEM_PROMPT]") != std::string::npos && src.find("[TOOL_CALLS]") != std::string::npos &&
        src.find("[ARGS]") != std::string::npos && src.find("[CALL_ID]") == std::string::npos) {
        LOG_DBG("Using specialized template: Ministral/Magistral Large 3\n");
        return common_chat_params_init_ministral_3(tmpl, params);
    }

    // GPT-OSS - has unique channel-based structure that needs dedicated handler
    if (src.find("<|channel|>") != std::string::npos) {
        LOG_DBG("Using specialized template: GPT-OSS\n");
        return common_chat_params_init_gpt_oss(tmpl, params);
    }

    // Functionary v3.2 - uses recipient-based format with >>>recipient\n{content}
    // Detection: template has ">>>all" for content and ">>>" prefix for tool calls
    if (src.find(">>>all") != std::string::npos && src.find(">>>${recipient}") != std::string::npos) {
        LOG_DBG("Using specialized template: Functionary v3.2\n");
        return common_chat_params_init_functionary_v3_2(tmpl, params);
    }

    // Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
    // Detection: template has "<|tool_calls_section_begin|>" and "functions." prefix in tool call IDs
    if (src.find("<|tool_calls_section_begin|>") != std::string::npos &&
        src.find("<|tool_call_begin|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Kimi K2 Thinking\n");
        return common_chat_params_init_kimi_k2(tmpl, params);
    }

    if (is_lfm2_template(src)) {
        LOG_DBG("Using specialized template: LFM2\n");
        return common_chat_params_init_lfm2(tmpl, params);
    }

    // LFM2.5 format detection: template uses plain "List of tools: [...]" with no special tokens
    if (src.find("List of tools: [") != std::string::npos &&
        src.find("<|tool_list_start|>") == std::string::npos) {
        LOG_DBG("Using specialized template: LFM2.5\n");
        return common_chat_params_init_lfm2_5(tmpl, params);
    }

    // GigaChatV3 format detection
    if (src.find("<|role_sep|>") != std::string::npos &&
        src.find("<|message_sep|>") != std::string::npos &&
        src.find("<|function_call|>") == std::string::npos) {
        LOG_DBG("Using specialized template: GigaChatV3\n");
        return common_chat_params_init_gigachat_v3(tmpl, params);
    }

    // DeepSeek V3.2 format detection: template defines dsml_token and uses it for tool calls.
    // The template source contains the token as a variable assignment, not as a literal in markup.
    if (src.find("dsml_token") != std::string::npos &&
        src.find("function_calls") != std::string::npos &&
        src.find("DSML") != std::string::npos) {
        LOG_DBG("Using specialized template: DeepSeek V3.2\n");
        return common_chat_params_init_deepseek_v3_2(tmpl, params);
    }

    // Gemma4 format detection
    if (src.find("'<|tool_call>call:'") != std::string::npos) {
        if (src.find("{#- OpenAI Chat Completions:") == std::string::npos) {
            // apply workarounds if using the older gemma4 templates
            LOG_WRN("%s: detected an outdated gemma4 chat template, applying compatibility workarounds. "
                    "Consider updating to the official template.\n", __func__);
            workaround::convert_tool_responses_gemma4(params.messages);
        }
        return common_chat_params_init_gemma4(tmpl, params);
    }

    return std::nullopt;
}

static common_chat_params common_chat_templates_apply_jinja(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    autoparser::generation_params params;
    params.tools = common_chat_tools_to_json_oaicompat(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    const auto & src        = tmpl.source();
    const auto & caps       = tmpl.original_caps();
    params.messages         = render_message_to_json(inputs.messages, tmpl.original_caps());
    params.tool_choice      = inputs.tool_choice;
    params.reasoning_format = inputs.reasoning_format;
    params.enable_thinking  = inputs.enable_thinking;
    params.grammar          = inputs.grammar;
    params.now              = inputs.now;
    params.add_bos          = tmpls->add_bos;
    params.add_eos          = tmpls->add_eos;

    if (src.find("<|channel|>") == std::string::npos) {
        // map developer to system for all models except for GPT-OSS
        workaround::map_developer_role_to_system(params.messages);
    }

    if (!tmpl.original_caps().supports_system_role) {
        workaround::system_message_not_supported(params.messages);
    }

    if (tmpl.original_caps().supports_tool_calls) {
        // some templates will require the content field in tool call messages
        // to still be non-null, this puts an empty string everywhere where the
        // content field is null
        workaround::requires_non_null_content(params.messages);
    }

    if (tmpl.original_caps().supports_object_arguments) {
        workaround::func_args_not_string(params.messages);
    }

    params.add_generation_prompt = false;
    std::string no_gen_prompt    = common_chat_template_direct_apply_impl(tmpl, params);
    params.add_generation_prompt = true;
    std::string gen_prompt       = common_chat_template_direct_apply_impl(tmpl, params);
    auto        diff             = calculate_diff_split(no_gen_prompt, gen_prompt);
    params.generation_prompt     = diff.right + diff.suffix;

    params.add_generation_prompt = inputs.add_generation_prompt;

    params.extra_context = common_chat_extra_context();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    params.parallel_tool_calls = inputs.parallel_tool_calls;

    if (params.tools.is_array()) {
        if (params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !params.grammar.empty()) {
            throw std::runtime_error("Cannot specify grammar with tools");
        }
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN(
                "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    if (inputs.force_pure_content) {
        LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
        // Create the result structure
        common_chat_params data;
        auto params_copy               = params;
        params_copy.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
        data.prompt                    = common_chat_template_direct_apply_impl(tmpl, params_copy);
        data.format                    = COMMON_CHAT_FORMAT_PEG_NATIVE;
        data.generation_prompt         = params.generation_prompt;
        auto parser                    = build_chat_peg_parser([&params](common_chat_peg_builder &p) {
            return p.prefix(params.generation_prompt) << p.content(p.rest());
        });
        data.parser                    = parser.save();
        return data;
    }

    if (auto result = common_chat_try_specialized_template(tmpl, src, params)) {
        result->generation_prompt = params.generation_prompt;
        return *result;
    }

    try {
        LOG_DBG("%s: using differential autoparser\n", __func__);
        struct autoparser::autoparser autoparser;
        autoparser.analyze_template(tmpl);
        auto auto_params = autoparser::peg_generator::generate_parser(tmpl, params, autoparser);
        auto_params.supports_thinking = autoparser.reasoning.mode != autoparser::reasoning_mode::NONE;
        if (auto_params.supports_thinking) {
            auto_params.thinking_start_tag = autoparser.reasoning.start;
            auto_params.thinking_end_tag   = autoparser.reasoning.end;
        }
        auto_params.generation_prompt = params.generation_prompt;
        common_peg_arena arena;
        arena.load(auto_params.parser);
        LOG_DBG("%s: generated parser:\n%s\n\nparser generation prompt: %s\n", __func__, arena.dump(arena.root()).c_str(), auto_params.generation_prompt.c_str());
        return auto_params;
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("Unable to generate parser for this template. Automatic parser generation failed: ") + e.what());
    }
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(const struct common_chat_templates *        tmpls,
                                                             const struct common_chat_templates_inputs & inputs) {
    size_t                          alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text" && part.type != "media_marker") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
                ;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg     = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({ msg.role.c_str(), content.c_str() });
        size_t msg_size = msg.role.size() + content.size();
        alloc_size += msg_size + (msg_size / 4);  // == msg_size * 1.25 but avoiding float ops
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t      res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt,
                                                 buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported, try using --jinja");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(),
                                        buf.size());
    }

    // for safety, we check the result again
    if (res < 0 || (size_t) res > buf.size()) {
        throw std::runtime_error("failed to apply chat template, try using --jinja");
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

common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                               const struct common_chat_templates_inputs & inputs) {
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja ? common_chat_templates_apply_jinja(tmpls, inputs) :
                              common_chat_templates_apply_legacy(tmpls, inputs);
}

common_chat_msg common_chat_parse(const std::string &               input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    return common_chat_peg_parse(params.parser, input, is_partial, params);
}

common_chat_msg common_chat_peg_parse(const common_peg_arena &          src_parser,
                                      const std::string &               input,
                                      bool                              is_partial,
                                      const common_chat_parser_params & params) {
    const common_peg_arena & parser = src_parser.empty() ?
        build_chat_peg_parser([](common_chat_peg_builder & p) { return p.content(p.rest()) + p.end(); }) :
        src_parser;

    if (src_parser.empty()) {
        LOG_DBG("No parser definition detected, assuming pure content parser.");
    }

    const std::string effective_input = params.generation_prompt.empty()
        ? input
        : params.generation_prompt + input;

    //LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(params.format), effective_input.c_str());

    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
    if (params.debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }

    common_peg_parse_context ctx(effective_input, flags);
    auto result = parser.parse(ctx);

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial && result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            common_chat_msg msg;
            msg.role = "assistant";
            std::unique_ptr<common_chat_peg_mapper> mapper;
            if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
                mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
            } else {
                mapper = std::make_unique<common_chat_peg_mapper>(msg);
            }
            mapper->from_ast(ctx.ast, result);

            if (ctx.is_debug()) {
                fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", ctx.ast.dump().c_str());
                fflush(stderr);
            }
            return msg;
        }
        throw std::runtime_error(std::string("Failed to parse input at pos ") + std::to_string(result.end) + ": " +
                                 effective_input.substr(result.end));
    }

    common_chat_msg msg;
    msg.role = "assistant";

    std::unique_ptr<common_chat_peg_mapper> mapper;
    if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
        mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
    } else {
        mapper = std::make_unique<common_chat_peg_mapper>(msg);
    }
    mapper->from_ast(ctx.ast, result);

    if (ctx.is_debug()) {
        fprintf(stderr, "\nAST for %s parse:\n%s\n", is_partial ? "partial" : "full", ctx.ast.dump().c_str());
        fflush(stderr);
    }

    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({ msg }).at(0).dump().c_str());
    }
    return msg;
}

std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates) {
    GGML_ASSERT(chat_templates != nullptr);
    GGML_ASSERT(chat_templates->template_default != nullptr);
    return chat_templates->template_default->caps.to_map();
}
