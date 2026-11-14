#include "parsers.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <stdexcept>

namespace {

const std::vector<std::string> markers = {
    "<think>", "</think>", "<tool_call>", "</tool_call>",
    "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
    "<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>",
    "<|user|>", "<|assistant|>", "<|observation|>",
};
const std::vector<std::string> stops = {"<|end_of_text|>", "<|user|>", "<|observation|>"};

std::string join(const std::vector<std::string> & parts, const std::string & separator) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) {
            out += separator;
        }
        out += parts[i];
    }
    return out;
}

std::string regex_escape(char c) {
    const std::string special = R"(\.^$|?*+()[]{}-)";
    return (special.find(c) == std::string::npos ? "" : "\\") + std::string(1, c);
}

// The trie accepts prefixes and mismatches, but never a complete protocol marker.
struct marker_trie {
    bool terminal = false;
    std::map<char, marker_trie> children;

    std::string pattern() const {
        std::vector<std::string> alternatives = {""};
        std::string excluded = "<";
        for (const auto & entry : children) {
            excluded += regex_escape(entry.first);
            if (!entry.second.terminal) {
                alternatives.push_back(regex_escape(entry.first) + entry.second.pattern());
            }
        }
        alternatives.push_back("[^" + excluded + "][^<]*");
        return "(?:" + join(alternatives, "|") + ")";
    }
};

const std::string & no_marker_pattern() {
    static const std::string pattern = [] {
        marker_trie trie;
        for (const auto & marker : markers) {
            auto * node = &trie;
            for (size_t i = 1; i < marker.size(); ++i) {
                node = &node->children[marker[i]];
            }
            node->terminal = true;
        }
        return "[^<]*(?:<" + trie.pattern() + ")*";
    }();
    return pattern;
}

std::string lark_regex(const std::string & pattern) {
    std::string out;
    for (char c : pattern) {
        if (c == '/') {
            out += '\\';
        }
        out += c;
    }
    return out;
}

json strings_without_markers(const json & schema, int depth = 0) {
    if (depth > COMMON_RECURSIVE_LIMIT) {
        throw std::invalid_argument("Flint tool schema exceeds nesting limit");
    }
    if (schema.is_array()) {
        auto out = json::array();
        for (const auto & item : schema) {
            out.push_back(strings_without_markers(item, depth + 1));
        }
        return out;
    }
    if (!schema.is_object()) {
        return schema;
    }
    auto out = json::object();
    for (const auto & item : schema.items()) {
        const auto & key = item.key();
        out[key] = key == "enum" || key == "const" || key == "default" || key == "examples"
                       ? item.value() : strings_without_markers(item.value(), depth + 1);
    }
    auto types = schema.value("type", json());
    bool has_string = types == "string";
    if (types.is_array()) {
        for (const auto & type : types) {
            has_string = has_string || type == "string";
        }
    }
    if (has_string && !schema.contains("pattern") && !schema.contains("enum") && !schema.contains("const")) {
        out["pattern"] = "^" + no_marker_pattern() + "$";
    }
    return out;
}

bool plain_string(const json & schema) {
    return schema.is_object() && schema.value("type", json()) == "string";
}

json parameters(const json & function) {
    auto schema = function.value("parameters", json::object());
    if (schema.is_null() || schema == true) {
        return json::object();
    }
    if (!schema.is_object()) {
        throw std::invalid_argument("Flint tool parameters must be an object schema");
    }
    return schema;
}

std::map<std::string, std::string> native_aliases(const json & tools) {
    std::set<std::string> offered;
    for (const auto & tool : tools) {
        offered.insert(tool.at("function").at("name").get<std::string>());
    }
    std::map<std::string, std::string> aliases;
    const std::map<std::string, std::vector<std::string>> targets = {
        {"read", {"view", "read_file"}}, {"write", {"create", "write_to_file", "write_file"}},
    };
    for (const auto & entry : targets) {
        if (offered.count(entry.first)) {
            continue;
        }
        for (const auto & target : entry.second) {
            if (offered.count(target)) {
                aliases[entry.first] = target;
                break;
            }
        }
    }
    return aliases;
}

std::set<std::string> shell_ids(const std::string & text, const std::string & prefix, const std::string & suffix) {
    std::set<std::string> ids;
    size_t pos = 0;
    while ((pos = text.find(prefix, pos)) != std::string::npos) {
        auto begin = pos + prefix.size();
        auto end = begin;
        while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) {
            ++end;
        }
        if (end > begin && text.compare(end, suffix.size(), suffix) == 0) {
            ids.insert(text.substr(begin, end - begin));
        }
        pos = end;
    }
    return ids;
}

bool running_shells(const json & messages) {
    std::set<std::string> running;
    for (const auto & message : messages) {
        auto calls = message.value("tool_calls", json::array());
        for (const auto & call : calls) {
            const auto & fn = call.at("function");
            if (fn.at("name") != "bash") {
                continue;
            }
            auto args = fn.at("arguments");
            if (args.is_string()) {
                args = json::parse(args.get<std::string>());
            }
            // Match the proxy: an async bash call keeps session tools available for the rest of the history.
            if (args.is_object() && args.value("mode", json()) == "async") {
                return true;
            }
        }
        if (message.at("role") == "tool") {
            auto content = message.value("content", "");
            auto started = shell_ids(content, "<command with shellId: ", " is still running");
            running.insert(started.begin(), started.end());
            for (const auto & id : shell_ids(content, "<shellId: ", " completed")) {
                running.erase(id);
            }
        }
    }
    return !running.empty();
}

json without_shell_session_tools(const json & tools, const json & messages) {
    if (running_shells(messages)) {
        return tools;
    }
    auto filtered = json::array();
    const std::set<std::string> session_tools = {"read_bash", "write_bash", "stop_bash", "list_bash"};
    for (const auto & tool : tools) {
        if (!session_tools.count(tool.at("function").at("name").get<std::string>())) {
            filtered.push_back(tool);
        }
    }
    return filtered;
}

// Port of Albert Zhang's Flint Lark builder in miles PRs #1172 and #1233.
struct flint_lark_builder {
    const llama_vocab * vocab;
    std::vector<std::string> rules;

    std::string token(const std::string & marker) const {
        if (!vocab) {
            return json(marker).dump();
        }
        auto tokens = common_tokenize(vocab, marker, false, true);
        if (tokens.size() != 1 || common_token_to_piece(vocab, tokens[0], true) != marker) {
            throw std::invalid_argument("Flint vocabulary lacks protocol token " + marker);
        }
        return "<[" + std::to_string(tokens[0]) + "]>";
    }

    std::string literal(const std::string & value) const {
        std::vector<std::string> parts;
        size_t pos = 0;
        while (pos < value.size()) {
            size_t next = value.size();
            std::string found;
            for (const auto & marker : markers) {
                auto at = value.find(marker, pos);
                if (at < next) {
                    next = at;
                    found = marker;
                }
            }
            if (next > pos) {
                parts.push_back(json(value.substr(pos, next - pos)).dump());
            }
            if (!found.empty()) {
                parts.push_back(token(found));
            }
            pos = next + found.size();
        }
        return parts.empty() ? "\"\"" : join(parts, " ");
    }

    std::string rule(const std::string & body, const std::string & prefix = "r") {
        auto name = prefix + std::to_string(rules.size() + 1);
        rules.push_back(name + ": " + body);
        return name;
    }

    std::string text(int64_t low = 0, int64_t high = -1) {
        if (low < 0 || high < -1 || (high >= 0 && high < low)) {
            throw std::invalid_argument("Invalid Flint string length bounds");
        }
        if (high == 0) {
            return "\"\"";
        }
        auto length = "(.|\\n){" + std::to_string(std::max<int64_t>(low, 1)) + "," +
                      (high < 0 ? "" : std::to_string(high)) + "}";
        auto name = rule("/" + length + "/ & /" + lark_regex(no_marker_pattern()) + "/", "L");
        return low == 0 ? name + "?" : name;
    }

    std::string raw_string(const json & schema) {
        if (schema.contains("const")) {
            return schema.at("const").dump();
        }
        if (schema.contains("enum")) {
            std::vector<std::string> values;
            for (const auto & value : schema.at("enum")) {
                values.push_back(value.dump());
            }
            return "(" + join(values, " | ") + ")";
        }
        if (schema.contains("pattern")) {
            auto pattern = schema.at("pattern").get<std::string>();
            if (!pattern.empty() && pattern.front() == '^') {
                pattern.erase(0, 1);
            }
            if (!pattern.empty() && pattern.back() == '$') {
                pattern.pop_back();
            }
            return rule("/" + lark_regex(pattern) + "/", "L");
        }
        return text(schema.value("minLength", int64_t(0)), schema.value("maxLength", int64_t(-1)));
    }

    std::string json_value(json schema, const json & root) {
        if (schema.is_object()) {
            for (const auto & key : {"$defs", "definitions"}) {
                if (root.contains(key) && !schema.contains(key)) {
                    schema[key] = root.at(key);
                }
            }
            schema = strings_without_markers(schema);
        } else {
            schema = schema == true ? json::object() : json{{"not", json::object()}};
        }
        schema["x-guidance"] = json{{"whitespace_flexible", false}, {"item_separator", ", "},
                                    {"key_separator", ": "}, {"lenient", true}};
        return rule("%json " + schema.dump(), "j");
    }

    std::string arguments(const json & schema) {
        auto props = schema.value("properties", json::object());
        if (props.is_null()) {
            props = json::object();
        }
        std::set<std::string> required;
        for (const auto & key : schema.value("required", json::array())) {
            required.insert(key.get<std::string>());
        }
        std::vector<std::string> pairs;
        for (const auto & prop : props.items()) {
            auto value = plain_string(prop.value()) ? raw_string(prop.value()) : json_value(prop.value(), schema);
            auto pair = token("<arg_key>") + " " + json(prop.key()).dump() + " " + token("</arg_key>") + " " +
                        token("<arg_value>") + " " + value + " " + token("</arg_value>");
            pairs.push_back("(" + pair + ")" + (required.count(prop.key()) ? "" : "?"));
        }
        if (props.empty() && schema.value("additionalProperties", json(true)) != false) {
            auto key = rule("/[^<]+/", "L");
            auto value = text();
            pairs.push_back("(" + token("<arg_key>") + " " + key + " " + token("</arg_key>") + " " +
                            token("<arg_value>") + " " + value + " " + token("</arg_value>") + ")*");
        }
        return pairs.empty() ? "\"\"" : join(pairs, " ");
    }

    std::string grammar(const json & tools, bool required, bool parallel, const std::string & generation_prompt) {
        for (const auto & marker : markers) {
            token(marker);
        }
        auto reasoning_text = text();
        auto reasoning = "(" + token("<think>") + " " + reasoning_text + " " + token("</think>") + ")?";
        auto content = text();
        std::vector<std::string> calls;
        for (const auto & tool : tools) {
            const auto & fn = tool.at("function");
            auto args = arguments(parameters(fn));
            calls.push_back("(" + token("<tool_call>") + " " + fn.at("name").dump() + " (" + args + ") " +
                            token("</tool_call>") + ")");
        }
        std::string body = literal(generation_prompt) + " " + reasoning + " " + content;
        if (!calls.empty()) {
            auto call = rule(join(calls, " | "));
            body += " (" + call + (parallel ? "+" : "") + ")" + (required ? "" : "?");
        }
        std::vector<std::string> endings;
        for (const auto & stop : stops) {
            endings.push_back(token(stop));
        }
        return "%llguidance {}\nstart: " + body + " (" + join(endings, " | ") + ")\n" + join(rules, "\n");
    }
};

} // namespace

bool is_flint_template(const std::string & src) {
    return src.find("<|end_of_text|>") != std::string::npos &&
           src.find("<|assistant|>") != std::string::npos &&
           src.find("<|observation|>") != std::string::npos &&
           src.find("<arg_key>") != std::string::npos &&
           src.find("<arg_value>") != std::string::npos;
}

common_chat_params common_chat_params_init_flint(const common_chat_template & tmpl,
                                                const autoparser::generation_params & inputs) {
    if (inputs.has_continuation()) {
        throw std::invalid_argument("Flint does not support assistant prefill with its tool grammar");
    }
    auto tools = inputs.tools.is_array() ? inputs.tools : json::array();
    auto noshell = inputs.extra_context.value("flint_noshell", json(false));
    if (!noshell.is_boolean()) {
        throw std::invalid_argument("flint_noshell must be a boolean");
    }
    if (noshell.get<bool>()) {
        tools = without_shell_session_tools(tools, inputs.messages);
    }
    common_chat_params data;
    data.tool_aliases = native_aliases(tools);
    bool named = !inputs.tool_choice_name.empty();
    auto offered = tools;
    tools = json::array();
    for (const auto & tool : offered) {
        if (!named || tool.at("function").at("name").get<std::string>() == inputs.tool_choice_name) {
            tools.push_back(tool);
        }
    }
    if (named && tools.empty()) {
        throw std::invalid_argument("Named tool_choice is not an offered tool");
    }
    if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
        tools = json::array();
    }
    bool required = named || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    if (required && tools.empty()) {
        throw std::invalid_argument("Required tool_choice needs at least one tool");
    }
    auto parser_tools = tools;
    for (const auto & alias : data.tool_aliases) {
        for (const auto & tool : tools) {
            if (tool.at("function").at("name").get<std::string>() == alias.second) {
                auto native = tool;
                native["function"]["name"] = alias.first;
                parser_tools.push_back(native);
                break;
            }
        }
    }
    data.prompt = common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt,
        inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE ? json::array() : offered);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tags = {"</think>"};
    data.preserved_tokens = markers;
    data.additional_stops = stops;
    data.message_delimiters = {{COMMON_CHAT_ROLE_ASSISTANT, "<|assistant|>"},
                               {COMMON_CHAT_ROLE_USER, "<|user|>"}, {COMMON_CHAT_ROLE_TOOL, "<|observation|>"}};
    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto thought_text = p.until("</think>");
        auto thought = inputs.reasoning_format == COMMON_REASONING_FORMAT_NONE
            ? p.content("<think>" + thought_text + "</think>")
            : "<think>" + p.reasoning(thought_text) + "</think>";
        auto calls = p.choice();
        for (const auto & tool : parser_tools) {
            const auto & fn = tool.at("function");
            auto props = parameters(fn).value("properties", json::object());
            auto pairs = p.choice();
            for (const auto & prop : props.items()) {
                auto value = p.until("</arg_value>");
                auto tagged_value = plain_string(prop.value()) ? p.tool_arg_string_value(value) : p.tool_arg_json_value(value);
                pairs |= p.tool_arg(p.tool_arg_open(p.literal("<arg_key>")) + p.tool_arg_name(p.literal(prop.key())) +
                    "</arg_key><arg_value>" + tagged_value + p.tool_arg_close(p.literal("</arg_value>")));
            }
            if (props.empty()) {
                pairs |= p.tool_arg(p.tool_arg_open(p.literal("<arg_key>")) + p.tool_arg_name(p.until("</arg_key>")) +
                    "</arg_key><arg_value>" + p.tool_arg_string_value(p.until("</arg_value>")) +
                    p.tool_arg_close(p.literal("</arg_value>")));
            }
            calls |= p.tool(p.tool_open(p.literal("<tool_call>") + p.tool_name(p.literal(fn.at("name").get<std::string>())) +
                            p.peek(p.choice({p.literal("<arg_key>"), p.literal("</tool_call>")}))) +
                            p.tool_args(p.zero_or_more(pairs)) + p.tool_close(p.literal("</tool_call>")));
        }
        auto body = p.literal(data.generation_prompt) + p.optional(thought) + p.content(p.until_one_of({"<tool_call>", stops[0], stops[1], stops[2]}));
        if (!parser_tools.empty()) {
            body = body + p.zero_or_more(calls);
        }
        return body + p.optional(p.choice({p.literal(stops[0]), p.literal(stops[1]), p.literal(stops[2])})) + p.end();
    });
    if (!inputs.grammar.empty()) {
        data.grammar = inputs.grammar;
    } else if (!inputs.json_schema.is_null()) {
        throw std::invalid_argument("Flint response_format is not supported with its native tool grammar");
    } else {
#ifdef LLAMA_USE_LLGUIDANCE
        flint_lark_builder builder{inputs.vocab, {}};
        data.grammar = builder.grammar(parser_tools, required, inputs.parallel_tool_calls && !named, data.generation_prompt);
#else
        throw std::invalid_argument("Flint native tool grammar requires a build with LLAMA_LLGUIDANCE=ON");
#endif
    }
    return data;
}
