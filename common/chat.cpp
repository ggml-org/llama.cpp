#include "chat.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"

// #include <minja/chat-template.hpp>
// #include <minja/minja.hpp>

#include "jinja/parser.h"
#include "jinja/value.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
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

template <> json common_chat_msg::to_json_oaicompat() const {
    json message{
        { "role", "assistant" },
    };
    if (!reasoning_content.empty()) {
        message["reasoning_content"] = reasoning_content;
    }
    if (content.empty() && !tool_calls.empty()) {
        message["content"] = json();
    } else {
        message["content"] = content;
    }
    if (!tool_calls.empty()) {
        auto arr = json::array();
        for (const auto & tc : tool_calls) {
            arr.push_back({
                { "type",     "function" },
                { "function",
                 {
                      { "name", tc.name },
                      { "arguments", tc.arguments },
                  }                      },
                { "id",       tc.id      },
                // // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
                // // We only generate a random id for the ones that don't generate one by themselves
                // // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
                // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            });
        }
        message["tool_calls"] = arr;
    }
    return message;
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

struct common_chat_template {
    jinja::program prog;
    std::string bos_tok;
    std::string eos_tok;
    std::string src;
    chat_template_caps caps;

    common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token) {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(src);
        this->prog = jinja::parse_from_tokens(lexer_res);

        this->src = lexer_res.source;
        this->bos_tok = bos_token;
        this->eos_tok = eos_token;

        this->caps = jinja::caps_get(prog);
        // LOG_INF("%s: caps:\n%s\n", __func__, this->caps.to_string().c_str());
    }

    const std::string & source() const { return src; }
    const std::string & bos_token() const { return bos_tok; }
    const std::string & eos_token() const { return eos_tok; }

    // TODO: this is ugly, refactor it somehow
    json add_system(const json & messages, const std::string & system_prompt) const {
        GGML_ASSERT(messages.is_array());
        auto msgs_copy = messages;
        if (!caps.supports_system_role) {
            if (msgs_copy.empty()) {
                msgs_copy.insert(msgs_copy.begin(), json{
                    {"role", "user"},
                    {"content", system_prompt}
                });
            } else {
                auto & first_msg = msgs_copy[0];
                if (!first_msg.contains("content")) {
                    first_msg["content"] = "";
                }
                first_msg["content"] = system_prompt + "\n\n"
                    + first_msg["content"].get<std::string>();
            }
        } else {
            if (msgs_copy.empty() || msgs_copy[0].at("role") != "system") {
                msgs_copy.insert(msgs_copy.begin(), json{
                    {"role", "system"},
                    {"content", system_prompt}
                });
            } else if (msgs_copy[0].at("role") == "system") {
                msgs_copy[0]["content"] = system_prompt;
            }
        }
        return msgs_copy;
    }

    chat_template_caps original_caps() const {
        return caps;
    }

};

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template;  // Model had builtin template or template overridde was specified.
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;

    // Cache for template analysis results (per template source)
    mutable std::mutex                                      analysis_cache_mutex;
    mutable std::map<std::string, template_analysis_result> analysis_cache;

    // Cache for generated parser results (per template + params combination)
    mutable std::mutex                                parser_cache_mutex;
    mutable std::map<std::string, common_chat_params> parser_cache;
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
    common_chat_templates_inputs dummy_inputs;
    common_chat_msg              msg;
    msg.role                          = "user";
    msg.content                       = "test";
    dummy_inputs.messages             = { msg };
    dummy_inputs.enable_thinking      = false;
    const auto rendered_no_thinking   = common_chat_templates_apply(chat_templates, dummy_inputs);
    dummy_inputs.enable_thinking      = true;
    const auto rendered_with_thinking = common_chat_templates_apply(chat_templates, dummy_inputs);
    return rendered_no_thinking.prompt != rendered_with_thinking.prompt;
}

template <> std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
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
                        if (type != "text") {
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
                    tc.name      = fc.at("name");
                    tc.arguments = fc.at("arguments");
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

template <> json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    json messages = json::array();
    for (const auto & msg : msgs) {
        if (!msg.content.empty() && !msg.content_parts.empty()) {
            throw std::runtime_error("Cannot specify both content and content_parts");
        }
        json jmsg{
            { "role", msg.role },
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
                        { "type", part.type },
                        { "text", part.text },
                    });
                }
            }
        } else if (msg.tool_calls.empty()) {
            jmsg["content"] = "";
        } else {
            // Per OpenAI spec, content should be null (not empty string) when there are tool calls.
            jmsg["content"] = nullptr;
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
                json tc{
                    { "type",     "function" },
                    { "function",
                     {
                          { "name", tool_call.name },
                          { "arguments", tool_call.arguments },
                      }                      },
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

template <> std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const std::string & messages) {
    return common_chat_msgs_parse_oaicompat(json::parse(messages));
}

template <> std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
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

template <> std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const std::string & tools) {
    return common_chat_tools_parse_oaicompat(json::parse(tools));
}

template <> json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function",
             {
                  { "name", tool.name },
                  { "description", tool.description },
                  { "parameters", json::parse(tool.parameters) },
              }                      },
        });
    }
    return result;
}

template <> json common_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        json function = json::object();
        if (!diff.tool_call_delta.name.empty()) {
            function["name"] = diff.tool_call_delta.name;
        }
        function["arguments"] = diff.tool_call_delta.arguments;
        tool_call["function"] = function;
        delta["tool_calls"]   = json::array({ tool_call });
    }
    return delta;
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

const char * common_chat_templates_source(const struct common_chat_templates * tmpls, const char * variant) {
    if (variant != nullptr) {
        if (strcmp(variant, "tool_use") == 0) {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source().c_str();
            }
            return nullptr;
        }
        LOG_DBG("%s: unknown template variant: %s\n", __func__, variant);
    }
    return tmpls->template_default->source().c_str();
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
        case COMMON_CHAT_FORMAT_GENERIC:
            return "Generic";
        case COMMON_CHAT_FORMAT_MISTRAL_NEMO:
            return "Mistral Nemo";
        case COMMON_CHAT_FORMAT_GPT_OSS:
            return "GPT-OSS";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE:
            return "peg-simple";
        case COMMON_CHAT_FORMAT_PEG_NATIVE:
            return "peg-native";
        case COMMON_CHAT_FORMAT_PEG_CONSTRUCTED:
            return "peg-constructed";
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

static std::string apply(
    const common_chat_template & tmpl,
    const struct templates_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt)
{
    jinja::context ctx(tmpl.source());

    nlohmann::ordered_json inp = nlohmann::ordered_json{
        {"messages", messages_override.has_value() ? *messages_override : inputs.messages},
        {"tools", tools_override.has_value() ? *tools_override : inputs.tools},
        {"bos_token", tmpl.bos_token()},
        {"eos_token", tmpl.eos_token()},
    };
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
    if (inp["tools"].is_null()) {
        inp["tools"] = json::array();
    }

    jinja::global_from_json(ctx, inp, inputs.mark_input);

    // render
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    auto parts = runtime.gather_string_parts(results);

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

static common_chat_params common_chat_params_init_generic(const common_chat_template &    tmpl,
                                                          const struct templates_params & inputs) {
    common_chat_params data;

    auto tool_call_schemas = json::array();
    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & function    = tool.at("function");
        auto         tool_schema = json{
                    { "type",       "object"                             },
                    { "properties",
                     {
                  { "name",
                            {
                        { "type", "string" },
                        { "const", function.at("name") },
                    } },
                  { "arguments", function.at("parameters") },
              }                                                          },
                    { "required",   json::array({ "name", "arguments" }) },
        };
        if (function.contains("description")) {
            tool_schema["description"] = function.at("description");
        }
        if (inputs.parallel_tool_calls) {
            tool_schema.at("properties")["id"] = {
                { "type",      "string" },
                { "minLength", 4        },
            };
            tool_schema.at("required").push_back("id");
        }
        tool_call_schemas.emplace_back(tool_schema);
    });
    const auto tool_call =
        inputs.parallel_tool_calls ?
            json{
                { "type",       "object"                      },
                { "properties",
                 {
                      { "tool_calls",
                        {
                            { "type", "array" },
                            { "items", tool_call_schemas.size() == 1 ? tool_call_schemas[0] :
                                                                       json{
                                                                           { "anyOf", tool_call_schemas },
                                                                       } },
                            { "minItems", 1 },
                        } },
                  }                                           },
                { "required",   json::array({ "tool_calls" }) },
    } :
            json{
                { "type", "object" },
                { "properties",
                  {
                      { "tool_call", tool_call_schemas.size() == 1 ? tool_call_schemas[0] :
                                                                     json{
                                                                         { "anyOf", tool_call_schemas },
                                                                     } },
                  } },
                { "required", json::array({ "tool_call" }) },
            };
    const auto schema =
        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED ?
            json{
                { "anyOf", json::array({
                               tool_call,
                               {
                                   { "type", "object" },
                                   { "properties",
                                     {
                                         { "response", inputs.json_schema.is_null() ? json{ { "type", "string" } } :
                                                                                      inputs.json_schema },
                                     } },
                                   { "required", json::array({ "response" }) },
                               },
                           }) }
    } :
            tool_call;

    data.grammar_lazy = false;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) { builder.add_schema("root", schema); });

    auto tweaked_messages = tmpl.add_system(
        inputs.messages,
        "Respond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request");

    // ensure all messages has "content" field
    for (auto & message : tweaked_messages) {
        if (!message.contains("content") || message["content"].is_null()) {
            message["content"] = "";
        }
    }

    data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);
    data.format = COMMON_CHAT_FORMAT_GENERIC;
    return data;
}

static common_chat_params common_chat_params_init_ministral_3(const common_chat_template &    tmpl,
                                                              const struct templates_params & inputs) {
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

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = true;

    data.prompt           = apply(tmpl, inputs, /* messages_override = */ adjusted_messages);
    data.format           = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
        "[TOOL_CALLS]",
        "[ARGS]",
    };

    auto parser = build_chat_peg_unified_parser([&](common_chat_peg_unified_builder & p) {
        auto reasoning =
            extract_reasoning ? p.optional("[THINK]" + p.reasoning(p.until("[/THINK]")) + "[/THINK]") : p.eps();

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            // Ministral wants to emit json surrounded by code fences
            return reasoning << "```json" << p.content(p.schema(p.json(), "response-format", inputs.json_schema))
                             << "```";
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

            return reasoning << p.content(p.until("[TOOL_CALLS]")) << tool_calls;
        }

        // Content only parser
        include_grammar = false;
        return reasoning << p.content(p.rest());
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
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]" }
        };
    }

    return data;
}

static common_chat_params common_chat_params_init_gpt_oss(const common_chat_template &    tmpl,
                                                          const struct templates_params & inputs) {
    common_chat_params data;

    // Copy reasoning to the "thinking" field as expected by the gpt-oss template
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto has_reasoning_content = msg.contains("reasoning_content") && msg.at("reasoning_content").is_string();
        auto has_tool_calls        = msg.contains("tool_calls") && msg.at("tool_calls").is_array();

        if (has_reasoning_content && has_tool_calls) {
            auto adjusted_message        = msg;
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
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = {
        "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = true;

    // Build PEG parser for GPT-OSS format
    // Structure: segments separated by <|start|>assistant
    // Each segment: [recipient_in_role]?<|channel|>(analysis|commentary|final)[recipient_in_channel]?[constraint]?<|message|>content[<|end|>]?
    auto parser = build_chat_peg_unified_parser([&](common_chat_peg_unified_builder & p) {
        // Helper: content until <|end|> or <|start|>
        auto segment_content = p.until_one_of({ "<|end|>", "<|start|>" });

        // Analysis channel (reasoning)
        auto analysis_header = "<|channel|>analysis<|message|>";
        auto end_segment = p.optional(p.literal("<|end|>") + p.space() + p.optional(p.literal("<|start|>assistant")));
        auto analysis_segment = p.literal(analysis_header) + p.reasoning(segment_content) + end_segment;

        // Final/commentary channel (content) - without tool recipient
        auto content_header = p.choice({ p.literal("<|channel|>final"), p.literal("<|channel|>commentary") });
        auto content_segment =
            content_header + p.until("<|message|>") + "<|message|>" + p.content(segment_content) + end_segment;

        // JSON schema response format
        if (!inputs.json_schema.is_null()) {
            auto final_header = p.literal("<|channel|>final");
            auto constraint   = p.optional(p.literal(" <|constrain|>") + p.until("<|message|>"));
            return p.optional(analysis_segment) + final_header + constraint + "<|message|>" +
                   p.content(p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto & params   = function.at("parameters");

                // Tool call can appear as:
                // 1. In role header: " to=functions.NAME<|channel|>..."
                // 2. In channel: "<|channel|>(analysis|commentary) to=functions.NAME..."
                auto func_name_in_role    = p.literal(" to=functions.") + p.tool_name(p.literal(name));
                auto func_name_in_channel = p.literal(" to=functions.") + p.tool_name(p.literal(name));

                // Channel header after recipient (in role)
                auto channel_after_recipient =
                    p.literal("<|channel|>") + p.choice({ p.literal("analysis"), p.literal("commentary") });

                // Optional constraint - use until to handle partial matching during streaming
                // (constraint can be " <|constrain|>json" or " <|constrain|>identifier" etc.)
                auto constraint = p.until("<|message|>");

                // Tool arguments
                auto args = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", params));

                // Pattern 1: recipient in role header
                // " to=functions.NAME<|channel|>(analysis|commentary)[constraint]<|message|>ARGS"
                auto tool_in_role = p.tool(p.tool_open(func_name_in_role + channel_after_recipient + constraint) +
                                           "<|message|>" + args);

                // Pattern 2: recipient in channel header
                // "<|channel|>(analysis|commentary) to=functions.NAME[constraint]<|message|>ARGS"
                auto channel_with_recipient =
                    p.literal("<|channel|>") + p.choice({ p.literal("analysis"), p.literal("commentary") });
                auto tool_in_channel = p.tool(p.tool_open(channel_with_recipient + func_name_in_channel + constraint) +
                                              "<|message|>" + args);

                tool_choice |= p.rule("tool-" + name, tool_in_role | tool_in_channel);
            });

            // Build tool calls section
            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;

            // Optional role start before tool call
            auto role_start = p.optional(p.literal("<|start|>assistant"));

            // Tool call with optional preceding content
            auto tool_call = p.trigger_rule(
                "tool-call",
                role_start + p.repeat(tool_choice + p.optional(p.literal("<|start|>assistant")), min_calls, max_calls));

            // Full parser: optional reasoning, optional content, optional tool calls
            if (extract_reasoning) {
                return p.optional(analysis_segment) + p.optional(content_segment) + tool_call;
            }
            return p.content(p.until_one_of({ " to=functions.", "<|channel|>" })) + tool_call;
        }

        // Content only parser
        include_grammar = false;
        if (extract_reasoning) {
            return p.optional(analysis_segment) + content_segment;
        }
        // reasoning_format=NONE:
        // - Final/commentary channels: strip headers (use content_segment)
        // - Analysis channel: keep raw content (fall back to rest)
        auto simple_content_header  = p.choice({ p.literal("<|channel|>final"), p.literal("<|channel|>commentary") });
        auto simple_content_segment = simple_content_header + p.until("<|message|>") + "<|message|>" +
                                      p.content(p.until_one_of({ "<|end|>", "<|start|>" }));
        return simple_content_segment | p.content(p.rest());
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
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Trigger on tool calls that appear in the commentary/analysis channel
        data.grammar_triggers.push_back(
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|channel\\|>(commentary|analysis) to" });

        // Trigger tool calls that appear in the role section
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "^ to" });
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>assistant to" });
    }

    return data;
}

// various workarounds for known issues with certain templates or model behaviors
// TODO @ngxson : improve this (how?)
namespace workaround {

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

static common_chat_params common_chat_params_init_granite(const common_chat_template &    tmpl,
                                                          const struct templates_params & inputs) {
    common_chat_params data;

    // Pass thinking context for Granite template
    json additional_context = {
        { "thinking", inputs.enable_thinking },
    };

    data.prompt = apply(tmpl, inputs, /* messages_override= */ std::nullopt, /* tools_override= */ std::nullopt,
                        additional_context);
    data.format = COMMON_CHAT_FORMAT_GRANITE;

    if (string_ends_with(data.prompt, "<think>\n") || string_ends_with(data.prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
        // reasoning_format=NONE:
        // - Final/commentary channels: strip headers (use content_segment)
        // - Analysis channel: keep raw content (fall back to rest)
        auto simple_content_header  = p.choice({ p.literal("<|channel|>final"), p.literal("<|channel|>commentary") });
        auto simple_content_segment = simple_content_header + p.until("<|message|>") + "<|message|>" +
                                      p.content(p.until_one_of({ "<|end|>", "<|start|>" }));
        return simple_content_segment | p.content(p.rest());
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
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Trigger on tool calls that appear in the commentary/analysis channel
        data.grammar_triggers.push_back(
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|channel\\|>(commentary|analysis) to" });

        // Trigger tool calls that appear in the role section
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "^ to" });
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>assistant to" });
    }

    return data;
}

static common_chat_params common_chat_params_init_without_tools(const common_chat_template &    tmpl,
                                                                const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt       = apply(tmpl, inputs);
    data.format       = COMMON_CHAT_FORMAT_CONTENT_ONLY;
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

static common_chat_params common_chat_params_init_seed_oss(const common_chat_template &         tmpl,
                                                           templates_params &                   params,
                                                           const common_chat_templates_inputs & inputs) {
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

                tool_rules.push_back(
                    builder.add_rule(name + "-call", "\"<seed:tool_call>\" space \"<function=" + name + ">\" space " +
                                                              param_rules + " \"</function>\" space \"</seed:tool_call>\""));
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

// various workarounds for known issues with certain templates or model behaviors
// TODO @ngxson : improve this (how?)
namespace workaround {

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

static void move_tool_calls_to_content(json & messages, int indent_spaces = 2) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls")) {
            auto tool_calls_new = json{
                {"tool_calls", message.at("tool_calls")}
            };
            message.erase("tool_calls");
            auto content = message.at("content");
            std::string content_new = content.is_null() ? "" : content.get<std::string>();
            message["content"] = content_new + tool_calls_new.dump(indent_spaces, ' ', false, json::error_handler_t::replace);
        }
    }
}

// TODO @ngxson : we may remove support for generic schema in the future
static void use_generic_schema(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls") && message.at("tool_calls").is_array()) {
            auto & tool_calls = message.at("tool_calls");
            for (auto & tool_call : tool_calls) {
                if (tool_call.contains("type") && tool_call.at("type") == "function" &&
                    tool_call.contains("function") && tool_call.at("function").is_object()) {
                    // Copy values before erasing to avoid use-after-free
                    json name_value;
                    json arguments_value;
                    json id_value;
                    const auto & function = tool_call.at("function");
                    if (function.contains("name")) {
                        name_value = function.at("name");
                    }
                    if (function.contains("arguments")) {
                        arguments_value = function.at("arguments");
                    }
                    if (tool_call.contains("id")) {
                        id_value = tool_call.at("id");
                    }
                    // Now safely erase and assign in the correct order
                    tool_call.erase("type");
                    tool_call.erase("function");
                    tool_call.erase("id");
                    // Reassign in desired order: name, arguments, id
                    if (!name_value.is_null()) {
                        tool_call["name"] = name_value;
                    }
                    if (!arguments_value.is_null()) {
                        tool_call["arguments"] = arguments_value;
                    }
                    if (!id_value.is_null()) {
                        tool_call["id"] = id_value;
                    }
                }
            }
        }
    }
}

} // namespace workaround

static common_chat_params common_chat_templates_apply_jinja(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    templates_params params;
    params.tools = common_chat_tools_to_json_oaicompat<json>(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    const auto & src  = tmpl.source();
    const auto & caps = tmpl.original_caps();
    params.messages   = common_chat_msgs_to_json_oaicompat<json>(
        inputs.messages, /* concat_typed_text= */ !tmpl.original_caps().requires_typed_content);
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.tool_choice = inputs.tool_choice;
    params.reasoning_format = inputs.reasoning_format;
    params.enable_thinking = inputs.enable_thinking;
    params.grammar = inputs.grammar;
    params.now = inputs.now;
    params.add_bos = tmpls->add_bos;
    params.add_eos = tmpls->add_eos;

    if (!tmpl.original_caps().supports_system_role) {
        workaround::system_message_not_supported(params.messages);
    }

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
            LOG_WRN(
                "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
    // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
    if (src.find("[SYSTEM_PROMPT]") != std::string::npos && src.find("[TOOL_CALLS]") != std::string::npos &&
        src.find("[ARGS]") != std::string::npos && src.find("[CALL_ID]") == std::string::npos) {
        return common_chat_params_init_ministral_3(tmpl, params);
    }

    // GPT-OSS - has unique channel-based structure that needs dedicated handler
    if (src.find("<|channel|>") != std::string::npos) {
        return common_chat_params_init_gpt_oss(tmpl, params);
    }

    // Unified two-phase template analysis with caching
    try {
        // Generate cache key (excludes messages and now)
        const std::string cache_key = make_autoparser_cache_key(src, params);

        // Check analysis cache first (needed for both cached and uncached parser paths)
        template_analysis_result analysis;
        {
            std::lock_guard<std::mutex> lock(tmpls->analysis_cache_mutex);
            auto                        it = tmpls->analysis_cache.find(src);
            if (it != tmpls->analysis_cache.end()) {
                LOG_DBG("Using cached template analysis\n");
                analysis = it->second;
            } else {
                LOG_DBG("Analyzing template (analysis cache miss)\n");
                analysis                   = template_analyzer::analyze_template(tmpl);
                tmpls->analysis_cache[src] = analysis;
            }
        }
        fprintf(stderr, "DEBUG ANALYSIS: function_format=%d, per_call_start='%s', id_marker='%s'\n",
                static_cast<int>(analysis.tools.function_format), analysis.tools.per_call_start.c_str(),
                analysis.tools.id_marker.c_str());

        // Check parser cache
        {
            std::lock_guard<std::mutex> lock(tmpls->parser_cache_mutex);
            auto                        it = tmpls->parser_cache.find(cache_key);
            if (it != tmpls->parser_cache.end()) {
                LOG_DBG("Using cached autoparser result\n");
                // Clone the cached result and update the prompt with current messages
                common_chat_params cached_result = it->second;
                cached_result.prompt             = apply_template(tmpl, params, std::nullopt);

                // Apply the same prompt modifications that generate_parser applies
                // (e.g., appending reasoning end marker when thinking is disabled)
                if (analysis.content.reasoning_mode == content_structure::REASONING_FORCED_OPEN &&
                    !params.enable_thinking) {
                    cached_result.prompt += analysis.content.reasoning_end + "\n";
                }

                return cached_result;
            }
        }

        LOG_DBG("Using unified template analysis (parser cache miss)\n");

        auto auto_params = universal_peg_generator::generate_parser(analysis, tmpl, params);

        // Only use the auto-generated parser if it provides more than basic content-only handling.
        // A PEG parser with thinking support or a non-empty parser (PEG grammar)
        // provides specialized handling that the generic handler doesn't.
        if (auto_params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY || auto_params.thinking_forced_open ||
            !auto_params.parser.empty()) {
            // Cache the result (store a copy without the prompt since that varies per messages)
            {
                std::lock_guard<std::mutex> lock(tmpls->parser_cache_mutex);
                tmpls->parser_cache[cache_key] = auto_params;
            }
            return auto_params;
        }
    } catch (const std::exception & e) {
        LOG_WRN("Automatic parser generation failed: %s - using generic fallback\n", e.what());
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
    if (src.find("<thinking>") != std::string::npos && src.find("</thinking>") != std::string::npos &&
        src.find("<available_tools>") != std::string::npos && src.find("<|assistant|>") != std::string::npos &&
        src.find("<|tool_result|>") != std::string::npos && src.find("<tool_calls>[") != std::string::npos &&
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
    if (src.find("<|start_header_id|>") != std::string::npos && src.find("<function=") != std::string::npos) {
        return common_chat_params_init_functionary_v3_1_llama_3_1(tmpl, params);
    }

    if (src.find("[THINK]") != std::string::npos && src.find("[/THINK]") != std::string::npos) {
        return common_chat_params_init_magistral(tmpl, params);
    }

    // Solar Open
    if (src.find("<|tool_response:begin|>") != std::string::npos &&
        src.find("<|tool_response:name|>") != std::string::npos &&
        src.find("<|tool_response:result|>") != std::string::npos) {
        return common_chat_params_init_solar_open(tmpl, params);
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

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(const struct common_chat_templates *        tmpls,
                                                             const struct common_chat_templates_inputs & inputs) {
    size_t                          alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text") {
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
