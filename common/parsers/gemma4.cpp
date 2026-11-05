#include "parsers.h"

#include "log.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>

namespace {

using gemma4 = common_chat_peg_gemma4_mapper;

// Parsers for a tool's arguments in the Gemma 4 call syntax, shaped by the parameter schema
// rather than converted from it. Only the shape is enforced: object keys are accepted in
// alphabetical order, since the template renders arguments through dictsort, optional keys are
// skipped in place instead of being moved to the end, and nested objects and arrays are shaped
// the same way. Every other value is accepted as written, so the schema guides the model without
// forcing values it cannot see. The grammar is derived from these parsers like any other.
class gemma4_schema_parser {
    common_chat_peg_builder & p_;

    std::map<const common_chat_schema *, std::string> refs_;

  public:
    explicit gemma4_schema_parser(common_chat_peg_builder & p) : p_(p) {}

    common_peg_parser visit(const common_chat_schema & node, const std::string & name) {
        switch (node.kind()) {
            case common_chat_schema::KIND_REF:    return visit_ref(static_cast<const common_chat_schema_ref &>(node), name);
            case common_chat_schema::KIND_ANY_OF: return visit_any_of(static_cast<const common_chat_schema_any_of &>(node), name);
            case common_chat_schema::KIND_ALL_OF: return visit_all_of(static_cast<const common_chat_schema_all_of &>(node), name);
            case common_chat_schema::KIND_ARRAY:  return visit_array(static_cast<const common_chat_schema_array &>(node), name);
            case common_chat_schema::KIND_TUPLE:  return visit_tuple(static_cast<const common_chat_schema_tuple &>(node), name);
            case common_chat_schema::KIND_OBJECT: return visit_object(static_cast<const common_chat_schema_object &>(node), name);
            default:                              return p_.ref("gemma4-value");
        }
    }

  private:
    // Whether a schema contributes a shape of its own, as opposed to accepting any value.
    static bool shaped(const common_chat_schema & node) {
        switch (node.kind()) {
            case common_chat_schema::KIND_REF:
                return static_cast<const common_chat_schema_ref &>(node).target != nullptr;
            case common_chat_schema::KIND_ANY_OF:
                for (const auto & child : static_cast<const common_chat_schema_any_of &>(node).children) {
                    if (shaped(*child)) {
                        return true;
                    }
                }
                return false;
            case common_chat_schema::KIND_ALL_OF: {
                const auto & children = static_cast<const common_chat_schema_all_of &>(node).children;
                return children.size() == 1 && shaped(*children[0]);
            }
            case common_chat_schema::KIND_ARRAY:
            case common_chat_schema::KIND_TUPLE:
            case common_chat_schema::KIND_OBJECT:
                return true;
            default:
                return false;
        }
    }

    // The template writes keys bare. Models keep that for identifiers and quote anything else,
    // with single or double quotes or the string delimiter, so those are accepted as well.
    common_peg_parser key(const std::string & key) {
        auto name = [&]() { return p_.tag(gemma4::DICT_KEY_NAME, p_.literal(key)); };

        bool identifier = !key.empty() && !std::isdigit(static_cast<unsigned char>(key[0]));
        for (char c : key) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                identifier = false;
            }
        }
        if (identifier) {
            return p_.tag(gemma4::DICT_KEY, name() + p_.literal(":"));
        }

        auto alts = p_.choice();
        if (!key.empty() && key.find_first_of(":}") == std::string::npos && std::string("'\"<").find(key[0]) == std::string::npos) {
            alts |= name();
        }
        alts |= p_.literal("<|\"|>") + name() + p_.literal("<|\"|>");
        if (key.find('\'') == std::string::npos) {
            alts |= p_.literal("'") + name() + p_.literal("'");
        }
        if (key.find('"') == std::string::npos) {
            alts |= p_.literal("\"") + name() + p_.literal("\"");
        }
        return p_.tag(gemma4::DICT_KEY, alts + p_.literal(":"));
    }

    common_peg_parser visit_ref(const common_chat_schema_ref & ref, const std::string & name) {
        if (!ref.target) {
            return p_.ref("gemma4-value");
        }
        auto it = refs_.find(ref.target);
        if (it != refs_.end()) {
            return p_.ref(it->second);
        }
        refs_[ref.target] = gbnf_format_rule_name(name);
        return p_.rule(name, [&]() { return visit(*ref.target, name + "-value"); });
    }

    // Shaped alternatives are tried first, then anything else collapses into one generic value.
    common_peg_parser visit_any_of(const common_chat_schema_any_of & node, const std::string & name) {
        if (!shaped(node)) {
            return p_.ref("gemma4-value");
        }
        if (node.children.size() == 1) {
            return visit(*node.children[0], name);
        }
        auto alts  = p_.choice();
        bool loose = false;
        for (size_t i = 0; i < node.children.size(); i++) {
            if (shaped(*node.children[i])) {
                alts |= visit(*node.children[i], name + "-" + std::to_string(i));
            } else {
                loose = true;
            }
        }
        if (loose) {
            alts |= p_.ref("gemma4-value");
        }
        return p_.rule(name, alts);
    }

    common_peg_parser visit_all_of(const common_chat_schema_all_of & node, const std::string & name) {
        if (node.children.size() == 1) {
            return visit(*node.children[0], name);
        }
        return p_.ref("gemma4-value");
    }

    common_peg_parser visit_array(const common_chat_schema_array & node, const std::string & name) {
        auto ws       = p_.space();
        auto item     = node.items && shaped(*node.items) ? visit(*node.items, name + "-item") : p_.ref("gemma4-value");
        auto elements = item + p_.zero_or_more(p_.literal(",") + ws + item);
        return p_.rule(name, p_.tag(gemma4::ARRAY,
            p_.literal("[") + ws + (p_.literal("]") | (elements + ws + p_.literal("]")))));
    }

    common_peg_parser visit_tuple(const common_chat_schema_tuple & node, const std::string & name) {
        auto ws    = p_.space();
        auto items = p_.sequence();
        for (size_t i = 0; i < node.items.size(); i++) {
            if (i > 0) {
                items = items + p_.literal(",") + ws;
            }
            items = items + visit(*node.items[i], name + "-" + std::to_string(i));
        }
        return p_.rule(name, p_.tag(gemma4::ARRAY, p_.literal("[") + ws + items + ws + p_.literal("]")));
    }

    common_peg_parser visit_object(const common_chat_schema_object & node, const std::string & name) {
        auto ws = p_.space();

        std::vector<const common_chat_schema_property *> props;
        for (const auto & prop : node.properties) {
            props.push_back(&prop);
        }
        // jinja's dictsort compares keys case-insensitively
        auto lower = [](const std::string & s) {
            std::string out = s;
            for (auto & c : out) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            return out;
        };
        std::stable_sort(props.begin(), props.end(), [&](const auto * a, const auto * b) {
            return lower(a->name) < lower(b->name);
        });

        if (props.empty()) {
            if (node.additional_properties) {
                return p_.ref("gemma4-dict");
            }
            return p_.rule(name, p_.tag(gemma4::DICT, p_.literal("{") + ws + p_.literal("}")));
        }

        // Members are chained from the back. rest(i) accepts members i.. after a member was
        // already written, so each one carries its comma, and first(i) accepts members i..
        // when nothing was written yet. Optional members are skipped in place.
        std::optional<common_peg_parser> rest;
        std::optional<common_peg_parser> first;
        if (node.additional_properties) {
            auto kv = p_.ref("gemma4-dict-kv");
            rest    = p_.zero_or_more(p_.literal(",") + ws + kv);
            first   = p_.optional(kv + *rest);
        }

        auto then_rest = [&](const common_peg_parser & parser) { return rest ? parser + *rest : parser; };

        for (size_t i = props.size(); i-- > 0;) {
            const auto & prop = *props[i];
            auto kv = p_.tag(gemma4::DICT_KV, key(prop.name) + ws + visit(*prop.schema, name + "-" + std::to_string(i)));

            common_peg_parser rest_i  = then_rest(p_.literal(",") + ws + kv);
            common_peg_parser first_i = then_rest(kv);
            if (!prop.required) {
                rest_i  = then_rest(p_.optional(p_.literal(",") + ws + kv));
                first_i = first ? (first_i | *first) : p_.optional(first_i);
            }

            if (i > 0) {
                rest = p_.rule(name + "-rest-" + std::to_string(i), rest_i);
            }
            first = first_i;
        }

        return p_.rule(name, p_.tag(gemma4::DICT, p_.literal("{") + ws + *first + ws + p_.literal("}")));
    }
};

}  // namespace

namespace workaround {

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

void convert_tool_responses_gemma4(json & messages) {
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

}

common_chat_params common_chat_params_init_gemma4(const common_chat_template &    tmpl,
                                                         const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);

    if (inputs.add_generation_prompt && string_ends_with(data.prompt.str(), "<turn|>\n")) {
        // This may happen if the model generates content + tool_call, the
        // template does not add the model's next turn and confuses the model
        // from emitting its proper reasoning token sequence.
        data.generation_prompt = "<|turn>model\n";
        data.prompt.append(data.generation_prompt);
    }

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_USER,      "<|turn>user"  },
        { COMMON_CHAT_ROLE_ASSISTANT, "<|turn>model" },
    };

    data.format            = COMMON_CHAT_FORMAT_PEG_GEMMA4;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<|channel>thought";
    data.thinking_end_tags  = {"<channel|>"};

    data.preserved_tokens = {
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|turn>",
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = string_ends_with(data.prompt.str(), "<turn|>\n") ? "<|turn>model\n" : "";
        data.generation_prompt += "<|channel>thought\n" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "<channel|>" + msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.optional(p.literal("<|turn>model\n")));

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
            // The tags are what gemma4_to_json() in the mapper reads, the rules may wrap them freely.
            p.rule("gemma4-string", p.tag(gemma4::STRING,
                p.literal("<|\"|>") + p.tag(gemma4::STRING_CONTENT, p.until("<|\"|>")) + p.literal("<|\"|>")));
            p.rule("gemma4-bool", p.tag(gemma4::BOOL, p.json_bool()));
            p.rule("gemma4-null", p.tag(gemma4::NULL_, p.json_null()));
            p.rule("gemma4-number", p.tag(gemma4::NUMBER, p.json_number()));
            // Keys are bare in the template, but models quote keys that are not identifiers
            // with single or double quotes or the string delimiter.
            p.rule("gemma4-dict-key", p.tag(gemma4::DICT_KEY, p.choice({
                p.literal("<|\"|>") + p.tag(gemma4::DICT_KEY_NAME, p.until("<|\"|>")) + p.literal("<|\"|>"),
                p.literal("\"") + p.tag(gemma4::DICT_KEY_NAME, p.chars("[^\"]", 1, -1)) + p.literal("\""),
                p.literal("'") + p.tag(gemma4::DICT_KEY_NAME, p.chars("[^']", 1, -1)) + p.literal("'"),
                p.tag(gemma4::DICT_KEY_NAME, p.chars("[^:}]", 1, -1)),
            }) + p.literal(":")));
            p.rule("gemma4-dict-kv", p.tag(gemma4::DICT_KV, p.ref("gemma4-dict-key") + p.space() + p.ref("gemma4-value")));
            p.rule("gemma4-dict", [&]() {
                auto ws = p.space();
                auto member = p.ref("gemma4-dict-kv");
                auto members = p.sequence({member, p.zero_or_more(p.sequence({p.literal(","), ws, member}))});
                return p.tag(gemma4::DICT, p.sequence({
                    p.literal("{"), ws,
                    p.choice({p.literal("}"), p.sequence({members, ws, p.literal("}")})})
                }));
            });
            p.rule("gemma4-array", [&]() {
                auto ws = p.space();
                auto value = p.ref("gemma4-value");
                auto elements = p.sequence({value, p.zero_or_more(p.sequence({p.literal(","), ws, value}))});
                return p.tag(gemma4::ARRAY, p.sequence({
                    p.literal("["), ws,
                    p.choice({p.literal("]"), p.sequence({elements, ws, p.literal("]")})})
                }));
            });
            p.rule("gemma4-value", [&]() {
                return p.choice({
                    p.ref("gemma4-string"), p.ref("gemma4-dict"), p.ref("gemma4-array"),
                    p.ref("gemma4-number"), p.ref("gemma4-bool"), p.ref("gemma4-null")
                });
            });

            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");

                // Arguments follow the shape of the parameter schema when it is an object,
                // see gemma4_schema_parser, and are a generic dict otherwise.
                auto args = p.ref("gemma4-dict");
                try {
                    auto doc = common_chat_schema_from_json(common_chat_tool_parameters(function));
                    if (dynamic_cast<const common_chat_schema_object *>(doc.root.get())) {
                        args = gemma4_schema_parser(p).visit(*doc.root, "tool-" + std::to_string(tool_index) + "-args");
                    }
                } catch (const std::exception & e) {
                    LOG_WRN("%s: tool %s uses the generic argument parser: %s\n", __func__, name.c_str(), e.what());
                }

                tool_choice |= p.rule("tool-" + std::to_string(tool_index), p.tool(p.sequence({
                    p.tool_open(p.tool_name(p.literal(name)) + p.peek(p.literal("{"))),
                    p.tool_args(args),
                })));
            });

            auto tool_call = p.trigger_rule("tool-call", p.repeat(
                "<|tool_call>call:" + tool_choice + "<tool_call|>",
                /* min = */ inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0,
                /* max = */ inputs.parallel_tool_calls ? -1 : 1
            ));

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return start + thought + tool_call;
            }

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

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call>" },
        };
    }

    return data;
}
