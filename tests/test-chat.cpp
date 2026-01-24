//  Tests chat handling, including grammar generation and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include "chat.h"

#include "log.h"

#include "../src/unicode.h"
#include "../src/llama-grammar.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <functional>
#include <string>

using json = nlohmann::ordered_json;

static std::ostream & operator<<(std::ostream & os, const common_chat_msg_diff & diff) {
    os << "{ content_delta: " << diff.content_delta << "; ";
    os << "reasoning_content_delta: " << diff.reasoning_content_delta << "; ";
    if (diff.tool_call_index != std::string::npos) {
        os << "tool_call_index: " << diff.tool_call_index << "; ";
        os << "tool_call_delta.name: " << diff.tool_call_delta.name << "; ";
        os << "tool_call_delta.id: " << diff.tool_call_delta.id << "; ";
        os << "tool_call_delta.arguments: " << diff.tool_call_delta.arguments << "; ";
    }
    os << "}";
    return os;
}
// operator<< for vector<common_chat_msg_diff>:
static std::ostream & operator<<(std::ostream & os, const std::vector<common_chat_msg_diff> & diffs) {
    os << "[\n";
    for (const auto & diff : diffs) {
        os << "  " << diff << ",\n";
    }
    os << "]";
    return os;
}
static std::ostream & operator<<(std::ostream & os, const common_chat_msg & msg) {
    os << "{ role: " << msg.role << "; ";
    os << "content: " << msg.content << "; ";
    os << "content_parts: [\n";
    for (const auto & part : msg.content_parts) {
        os << "  { type: " << part.type << "; text: " << part.text << " },\n";
    }
    os << "]; ";
    os << "reasoning_content: " << msg.reasoning_content << "; ";
    os << "tool_calls: [\n";
    for (const auto & tool_call : msg.tool_calls) {
        os << "  { name: " << tool_call.name << "; arguments: " << tool_call.arguments << "; id: " << tool_call.id << " },\n";
    }
    os << "]";
    os << "}";
    return os;
}

template <class T> static bool equals(const T & expected, const T & actual) {
    return expected == actual;
}

static common_chat_msg normalize(const common_chat_msg & msg) {
    common_chat_msg normalized = msg;
    for (auto & tool_call : normalized.tool_calls) {
        try {
            tool_call.arguments = json::parse(tool_call.arguments).dump();
        } catch (const std::exception &) {
            // Do nothing
        }
    }
    return normalized;
}


template <>
bool equals(const common_chat_msg & expected, const common_chat_msg & actual) {
    return normalize(expected) == normalize(actual);
}

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (!equals(expected, actual)) {
        std::cerr << "Expected:```\n" << expected << "\n```" << std::endl;
        std::cerr << "Actual:```\n" << actual << "\n```" << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

static std::string read_file(const std::string & path) {
    std::cerr << "# Reading: " << path << '\n' << std::flush;
    std::ifstream fs(path, std::ios_base::binary);
    if (!fs.is_open()) {
        fs = std::ifstream("../" + path, std::ios_base::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
    }
    fs.seekg(0, std::ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    std::string out;
    out.resize(static_cast<size_t>(size));
    fs.read(out.data(), static_cast<std::streamsize>(size));
    return out;
}

static common_chat_templates_ptr read_templates(const std::string & path) {
    return common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, read_file(path)));
}

static std::unique_ptr<llama_grammar> build_grammar(const std::string & grammar_str) {
    return std::unique_ptr<llama_grammar>(
        llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0));
}

// TODO: extract to common helper (copied from test-grammar-integration.cpp)
static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto cpts = unicode_cpts_from_utf8(input);

    auto & stacks_cur = llama_grammar_get_stacks(grammar);

    for (const auto & cpt : cpts) {
        llama_grammar_accept(grammar, cpt);

        if (stacks_cur.empty()) {
            // no stacks means that the grammar failed to match at this point
            return false;
        }
    }

    if (std::any_of(stacks_cur.begin(), stacks_cur.end(), [](const auto & stack) { return stack.empty(); })) {
        // An empty stack means that the grammar has been completed
        return true;
    }

    return false;
}

static std::string renormalize_json(const std::string & json_str) {
    try {
        auto json_obj = json::parse(json_str);
        return json_obj.dump();
    } catch (const std::exception & e) {
        std::cerr << "Failed to parse JSON: " << e.what() << '\n';
        return json_str;
    }
}
static void assert_msg_equals(const common_chat_msg & expected, const common_chat_msg & actual, bool ignore_whitespace_differences = false) {
    assert_equals(expected.role, actual.role);
    if (ignore_whitespace_differences) {
        assert_equals(string_strip(expected.content), string_strip(actual.content));
    } else {
        assert_equals(expected.content, actual.content);
    }
    assert_equals(expected.content_parts.size(), actual.content_parts.size());
    for (size_t i = 0; i < expected.content_parts.size(); i++) {
        const auto & expected_part = expected.content_parts[i];
        const auto & actual_part   = actual.content_parts[i];
        assert_equals(expected_part.type, actual_part.type);
        if (ignore_whitespace_differences) {
            assert_equals(string_strip(expected_part.text), string_strip(actual_part.text));
        } else {
            assert_equals(expected_part.text, actual_part.text);
        }
    }
    if (ignore_whitespace_differences) {
        assert_equals(string_strip(expected.reasoning_content), string_strip(actual.reasoning_content));
    } else {
        assert_equals(expected.reasoning_content, actual.reasoning_content);
    }
    assert_equals(expected.tool_calls.size(), actual.tool_calls.size());
    for (size_t i = 0; i < expected.tool_calls.size(); i++) {
        const auto & expected_tool_call = expected.tool_calls[i];
        const auto & actual_tool_call   = actual.tool_calls[i];
        assert_equals(expected_tool_call.name, actual_tool_call.name);
        assert_equals(renormalize_json(expected_tool_call.arguments), renormalize_json(actual_tool_call.arguments));
        assert_equals(expected_tool_call.id, actual_tool_call.id);
    }
}

common_chat_tool special_function_tool {
    /* .name = */ "special_function",
    /* .description = */ "I'm special",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "arg1": {
                "type": "integer",
                "description": "The arg."
            }
        },
        "required": ["arg1"]
    })",
};
common_chat_tool special_function_tool_with_optional_param {
    /* .name = */ "special_function_with_opt",
    /* .description = */ "I'm special but have optional stuff",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "arg1": {
                "type": "integer",
                "description": "The arg."
            },
            "arg2": {
                "type": "integer",
                "description": "The optional arg."
            }
        },
        "required": ["arg1"]
    })",
};
common_chat_tool python_tool {
    /* .name = */ "python",
    /* .description = */ "an ipython interpreter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute."
            }
        },
        "required": ["code"]
    })",
};
common_chat_tool code_interpreter_tool {
    /* .name = */ "code_interpreter",
    /* .description = */ "an ipython interpreter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute."
            }
        },
        "required": ["code"]
    })",
};
std::vector<common_chat_tool> tools           { special_function_tool, special_function_tool_with_optional_param, python_tool };
std::vector<common_chat_tool> llama_3_1_tools { special_function_tool, code_interpreter_tool };

struct delta_data {
    std::string        delta;
    common_chat_params params;
};

static common_chat_msg simple_assist_msg(const std::string & content, const std::string & reasoning_content = "", const std::string & tool_name = "", const std::string & arguments = "", const std::string & id = "") {
    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = content;
    msg.reasoning_content = reasoning_content;
    if (!tool_name.empty()) {
        msg.tool_calls.push_back({ tool_name, arguments, id });
    }
    return msg;
}

static delta_data init_delta(const struct common_chat_templates * tmpls, const std::vector<std::string> & end_tokens,
                             const common_chat_msg & user_message,
                             const common_chat_msg & delta_message,
                             const std::vector<common_chat_tool> & tools,
                             const common_chat_tool_choice & tool_choice) {
    common_chat_templates_inputs inputs;
    inputs.parallel_tool_calls = true;
    inputs.messages.push_back(user_message);
    inputs.tools       = tools;
    inputs.tool_choice = tool_choice;
    auto params_prefix = common_chat_templates_apply(tmpls, inputs);

    inputs.messages.push_back(delta_message);
    inputs.add_generation_prompt = false;
    auto params_full             = common_chat_templates_apply(tmpls, inputs);

    std::string prefix = params_prefix.prompt;
    std::string full   = params_full.prompt;

    if (full == prefix) {
        throw std::runtime_error("Full message is the same as the prefix");
    }

    size_t common_prefix_length = 0;
    for (size_t i = 0; i < prefix.size() && i < full.size(); ++i) {
        if (prefix[i] != full[i]) {
            break;
        }
        if (prefix[i] == '<') {
            // DeepSeek R1's template (as of 20250209) adds a trailing <think> if add_generation_prompt,
            // but it removes thinking tags for past messages.
            // The prefix and full strings diverge at <think> vs. <｜tool▁calls▁begin｜>, we avoid consuming the leading <.
            continue;
        }
        common_prefix_length = i + 1;
    }
    auto delta = full.substr(common_prefix_length);

    // Strip end tokens
    for (const auto & end_token : end_tokens) {
        // rfind to find the last occurrence
        auto pos = delta.rfind(end_token);
        if (pos != std::string::npos) {
            delta = delta.substr(0, pos);
            break;
        }
    }
    return { delta, params_full };
}

/*
  Applies the template to 1 user message w/ add_generation_prompt=true, then w/ the test message w/ add_generation_prompt=false,
  gets the diff, removes any end tokens and parses the result w/ the grammar, checking that
  the parsed message is the same as the test_message
*/
static void test_templates(const struct common_chat_templates * tmpls, const std::vector<std::string> & end_tokens,
                          const common_chat_msg & test_message,
                          const std::vector<common_chat_tool> & tools = {},
                          const std::string & expected_delta = "",
                          bool expect_grammar_triggered = true,
                          bool test_grammar_if_triggered = true,
                          common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE,
                          bool ignore_whitespace_differences = false
                        ) {
    common_chat_msg user_message;
    user_message.role = "user";
    user_message.content = "Hello, world!";

    for (const auto & tool_choice : std::vector<common_chat_tool_choice> {COMMON_CHAT_TOOL_CHOICE_AUTO, COMMON_CHAT_TOOL_CHOICE_REQUIRED}) {
        auto data = init_delta(tmpls, end_tokens, user_message, test_message, tools, tool_choice);
        if (!expected_delta.empty()) {
            if (ignore_whitespace_differences) {
                assert_equals(string_strip(expected_delta), string_strip(data.delta));
            } else {
                assert_equals(expected_delta, data.delta);
            }
        }

        if (expect_grammar_triggered) {
            // TODO @ngxson : refactor common_chat_parse to avoid passing format/reasoning_format every time
            common_chat_parser_params params;
            params.format = data.params.format;
            params.reasoning_format = reasoning_format;
            const auto msg = common_chat_parse(data.delta, /* is_partial= */ false, params);
            assert_msg_equals(test_message, msg, ignore_whitespace_differences);
        }

        if (!test_message.tool_calls.empty()) {
            GGML_ASSERT(!data.params.grammar.empty());
        }
        if (!data.params.grammar.empty()) {
            auto grammar = build_grammar(data.params.grammar);
            if (!grammar) {
                throw std::runtime_error("Failed to build grammar");
            }
            auto earliest_trigger_pos = std::string::npos;
            auto constrained = data.delta;
            for (const auto & trigger : data.params.grammar_triggers) {
                size_t pos = std::string::npos;
                std::smatch match;
                switch (trigger.type) {
                    case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    {
                        const auto & word = trigger.value;
                        pos = constrained.find(word);
                        break;
                    }
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    {
                        const auto & pattern = trigger.value;
                        if (std::regex_search(constrained, match, std::regex(pattern))) {
                            pos = match.position(1);
                        }
                        break;
                    }
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                    {
                        const auto & pattern = trigger.value;
                        if (std::regex_match(constrained, match, std::regex(pattern))) {
                            auto mpos = std::string::npos;
                            for (size_t i = 1; i < match.size(); ++i) {
                                if (match[i].length() > 0) {
                                    mpos = match.position(i);
                                    break;
                                }
                            }
                            if (mpos == std::string::npos) {
                                mpos = match.position(0);
                            }
                            pos = mpos;
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("Unknown trigger type");
                }
                if (pos == std::string::npos) {
                    continue;
                }
                if (earliest_trigger_pos == std::string::npos || pos < earliest_trigger_pos) {
                    earliest_trigger_pos = pos;
                }
            }
            auto grammar_triggered = false;
            if (earliest_trigger_pos != std::string::npos) {
                constrained = constrained.substr(earliest_trigger_pos);
                grammar_triggered = true;
            }
            if (data.params.grammar_lazy) {
                assert_equals(expect_grammar_triggered, grammar_triggered);
            }

            if (grammar_triggered && test_grammar_if_triggered && !match_string(constrained, grammar.get())) {
                throw std::runtime_error("Failed to match delta against grammar:\n\n" + data.delta +
                    "\n\nConstrained: " + constrained +
                    "\n\nGrammar: " + data.params.grammar);
            }
        }
    }
}

/**
 * Test if streaming=true is consistant with streaming=false for given partial parser
 * Also test if there is any problem with partial message
 */
template <typename T>
static void test_parser_with_streaming(const common_chat_msg & expected, const std::string & raw_message, T parse_msg) {
    constexpr auto utf8_truncate_safe_len = [](const std::string_view s) -> size_t {
        auto len = s.size();
        if (len == 0) return 0;
        auto i = len;
        for (size_t back = 0; back < 4 && i > 0; ++back) {
            --i;
            unsigned char c = s[i];
            if ((c & 0x80) == 0) {
                return len;
            } else if ((c & 0xC0) == 0xC0) {
                size_t expected_len = 0;
                if ((c & 0xE0) == 0xC0) expected_len = 2;
                else if ((c & 0xF0) == 0xE0) expected_len = 3;
                else if ((c & 0xF8) == 0xF0) expected_len = 4;
                else return i;
                if (len - i >= expected_len) {
                    return len;
                } else {
                    return i;
                }
            }
        }
        return len - std::min(len, size_t(3));
    };
    constexpr auto utf8_truncate_safe_view = [utf8_truncate_safe_len](const std::string_view s) {
        return s.substr(0, utf8_truncate_safe_len(s));
    };

    auto merged = simple_assist_msg("");
    auto last_msg = parse_msg("");
    for (size_t i = 1; i <= raw_message.size(); ++i) {
        auto curr_msg = parse_msg(std::string(utf8_truncate_safe_view(std::string_view(raw_message).substr(0, i))));
        if (curr_msg == simple_assist_msg("")) continue;
        LOG_INF("Streaming msg: %s\n", common_chat_msgs_to_json_oaicompat({curr_msg}).dump().c_str());
        for (auto diff: common_chat_msg_diff::compute_diffs(last_msg, curr_msg)) {
            LOG_INF("Streaming diff: %s\n", common_chat_msg_diff_to_json_oaicompat(diff).dump().c_str());
            if (!diff.reasoning_content_delta.empty()) {
                merged.reasoning_content += diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                merged.content += diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                if (!diff.tool_call_delta.name.empty()) {
                    merged.tool_calls.push_back({diff.tool_call_delta.name, "", ""});
                }
                if (!diff.tool_call_delta.arguments.empty()) {
                    GGML_ASSERT(!merged.tool_calls.empty());
                    merged.tool_calls.back().arguments += diff.tool_call_delta.arguments;
                }
            }
            LOG_INF("Streaming merged: %s\n", common_chat_msgs_to_json_oaicompat({merged}).dump().c_str());
        }
        assert_msg_equals(curr_msg, merged, true);
        last_msg = curr_msg;
    }
    assert_msg_equals(expected, parse_msg(raw_message), true);
    assert_msg_equals(expected, merged, true);
}

const common_chat_msg message_user {
    "user",
    "Hey there!",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

const common_chat_msg message_user_parts {
    "user",
    /* .content = */ "",
    /* .content_parts = */ {
        { "text", "Hey" },
        { "text", "there" },
    },
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

const common_chat_msg message_assist                              = simple_assist_msg("Hello, world!\nWhat's up?");
const common_chat_msg message_assist_empty                        = simple_assist_msg("");
const common_chat_msg message_assist_thoughts_unparsed_deepseek   = simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_unparsed_md         = simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```");
const common_chat_msg message_assist_thoughts_unparsed_md_partial = simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}");

const common_chat_msg message_assist_thoughts_unparsed_r7b       = simple_assist_msg("<|START_THINKING|>I'm\nthinking<|END_THINKING|>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_unparsed_magistral = simple_assist_msg("[THINK]raisonnement[/THINK]Réponse");
const common_chat_msg message_assist_thoughts                    = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking");
const common_chat_msg message_assist_thoughts_unopened_unparsed  = simple_assist_msg("I'm\nthinking</think>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_no_content         = simple_assist_msg("", "I'm\nthinking");
const common_chat_msg message_assist_call                        = simple_assist_msg("", "", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_noopt                  = simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_withopt                = simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}");
const common_chat_msg message_assist_call_content                = simple_assist_msg("Hello, world!\nWhat's up?", "", "special_function", "{\"arg1\":1}");
const common_chat_msg message_assist_call_empty_args             = simple_assist_msg("", "", "special_function");
const common_chat_msg message_assist_call_cutoff_args            = simple_assist_msg("", "", "special_function", "{\"arg");
const common_chat_msg message_assist_call_thoughts               = simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\":1}");
const common_chat_msg message_assist_call_thoughts_unparsed      = simple_assist_msg("<think>I'm\nthinking</think>\n\n", "", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_thoughts_content       = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_id                     = simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "123456789");
const common_chat_msg message_assist_call_idx                    = simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "0");
const common_chat_msg message_assist_thoughts_call_idx           = simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}", /* id = */ "0");
const common_chat_msg message_assist_call_python                 = simple_assist_msg("", "", "python", "{\"code\":\"print('hey')\"}");
const common_chat_msg message_assist_call_python_lines           = simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')\"}");
const common_chat_msg message_assist_call_python_lines_unclosed  = simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')");
const common_chat_msg message_assist_call_code_interpreter       = simple_assist_msg("", "", "code_interpreter", "{\"code\":\"print('hey')\"}");

// Use for PEG parser implementations
struct peg_test_case {
    common_chat_templates_inputs params;
    std::string input;
    common_chat_msg expect;
};

struct make_peg_parser {
    common_chat_params params_;
    common_peg_arena arena_;

    make_peg_parser(common_chat_templates * tmpls, const common_chat_templates_inputs & inputs) {
        params_ = common_chat_templates_apply(tmpls, inputs);
        arena_.load(params_.parser);
    }

    common_chat_msg parse(const std::string & msg, bool is_partial) {
        common_chat_parser_params parser_params;
        parser_params.format = params_.format;
        return common_chat_peg_parse(arena_, msg, is_partial, parser_params);
    }
};

static void test_peg_parser(common_chat_templates * tmpls, const std::function<void(peg_test_case &)> & init) {
    peg_test_case tc;
    init(tc);
    if (tc.params.messages.empty()) {
        tc.params.messages = {message_user};
    }
    if (tc.expect.role.empty()) {
        tc.expect.role = "assistant";
    }

    auto parser = make_peg_parser(tmpls, tc.params);

    common_chat_msg msg_accum;
    common_chat_msg msg_prev;
    msg_accum.role = msg_prev.role = "assistant";

    for (size_t i = 1; i <= tc.input.size(); ++i) {
        auto is_partial = i < tc.input.size();
        common_chat_msg msg_current = parser.parse(tc.input.substr(0, i), is_partial);

        for (const auto & diff : common_chat_msg_diff::compute_diffs(msg_prev, msg_current)) {
            if (!diff.reasoning_content_delta.empty()) {
                msg_accum.reasoning_content += diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                msg_accum.content += diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                if (!diff.tool_call_delta.name.empty()) {
                    msg_accum.tool_calls.push_back({diff.tool_call_delta.name, "", ""});
                }
                if (!diff.tool_call_delta.arguments.empty()) {
                    msg_accum.tool_calls.back().arguments += diff.tool_call_delta.arguments;
                }
            }
        }
        assert_msg_equals(msg_current, msg_accum, true);
        msg_prev = msg_current;
    }

    assert_msg_equals(tc.expect, parser.parse(tc.input, false), true);
    assert_msg_equals(tc.expect, msg_accum, true);
}

static void test_msgs_oaicompat_json_conversion() {
    printf("[%s]\n", __func__);
    std::vector<common_chat_msg> msgs{
        message_user,
        message_user_parts,
        message_assist_call,
        message_assist_call_thoughts,
        message_assist_call_thoughts_unparsed,
        message_assist_call_thoughts_content,
        message_assist_call_id,
        message_assist_call_idx,
        message_assist_call_python,
        message_assist_call_code_interpreter,
    };
    for (const auto & msg : msgs) {
        auto oai_json = common_chat_msgs_to_json_oaicompat({msg});
        auto msgs2 = common_chat_msgs_parse_oaicompat(oai_json);
        assert_equals((size_t) 1, msgs2.size());
        auto msg2 = msgs2[0];
        assert_msg_equals(msg, msg2);
    }
    assert_equals(
        std::string(
            "[\n"
            "  {\n"
            "    \"role\": \"user\",\n"
            "    \"content\": [\n"
            "      {\n"
            "        \"type\": \"text\",\n"
            "        \"text\": \"Hey\"\n"
            "      },\n"
            "      {\n"
            "        \"type\": \"text\",\n"
            "        \"text\": \"there\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "]"
        ),
        common_chat_msgs_to_json_oaicompat({message_user_parts}).dump(2));

    assert_equals(
        std::string(
            "[\n"
            "  {\n"
            "    \"role\": \"assistant\",\n"
            "    \"content\": \"\",\n"
            "    \"tool_calls\": [\n"
            "      {\n"
            "        \"type\": \"function\",\n"
            "        \"function\": {\n"
            "          \"name\": \"python\",\n"
            "          \"arguments\": \"{\\\"code\\\":\\\"print('hey')\\\"}\"\n"
            "        }\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "]"
        ),
        common_chat_msgs_to_json_oaicompat({message_assist_call_python}).dump(2));

    auto res = common_chat_msgs_parse_oaicompat(json::parse("[{\"role\": \"assistant\", \"tool_calls\": []}]"));
    assert_equals<size_t>(1, res.size());
    assert_equals<std::string>(res[0].role, "assistant");
    assert_equals(true, res[0].content.empty());
    assert_equals(true, res[0].tool_calls.empty());

    try {
        common_chat_msgs_parse_oaicompat(json::parse("[{\"role\": \"assistant\"}]"));
        throw std::runtime_error("Expected exception");
    } catch (const std::exception & e) {
        if (std::string(e.what()).find("'content'") == std::string::npos) {
            throw std::runtime_error("Expected exception about missing 'content'");
        }
    }
}

static void test_tools_oaicompat_json_conversion() {
    printf("[%s]\n", __func__);
    std::vector<common_chat_tool> tools{
        special_function_tool,
        python_tool,
        code_interpreter_tool,
    };

    for (const auto & tool : tools) {
        auto oai_json = common_chat_tools_to_json_oaicompat({tool});
        auto tools2 = common_chat_tools_parse_oaicompat(oai_json);
        assert_equals((size_t) 1, tools2.size());
        auto tool2 = tools2[0];
        assert_equals(tool.name, tool2.name);
        assert_equals(tool.description, tool2.description);
        assert_equals(json::parse(tool.parameters).dump(2), json::parse(tool2.parameters).dump(2));
    }

    assert_equals(
        std::string(
            "[\n"
            "  {\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "      \"name\": \"special_function\",\n"
            "      \"description\": \"I'm special\",\n"
            "      \"parameters\": {\n"
            "        \"type\": \"object\",\n"
            "        \"properties\": {\n"
            "          \"arg1\": {\n"
            "            \"type\": \"integer\",\n"
            "            \"description\": \"The arg.\"\n"
            "          }\n"
            "        },\n"
            "        \"required\": [\n"
            "          \"arg1\"\n"
            "        ]\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "]"
        ),
        common_chat_tools_to_json_oaicompat({special_function_tool}).dump(2));

    {
        auto tools_no_params = common_chat_tools_parse_oaicompat(json::parse(
            R"([{"type": "function", "function": {"name": "test_func", "description": "A test"}}])"));
        assert_equals((size_t) 1, tools_no_params.size());
        assert_equals(std::string("test_func"), tools_no_params[0].name);
        assert_equals(std::string("A test"), tools_no_params[0].description);
        assert_equals(std::string("{}"), tools_no_params[0].parameters);
    }
    {
        auto tools_no_desc = common_chat_tools_parse_oaicompat(json::parse(
            R"([{"type": "function", "function": {"name": "test_func", "parameters": {"type": "object"}}}])"));
        assert_equals((size_t) 1, tools_no_desc.size());
        assert_equals(std::string("test_func"), tools_no_desc[0].name);
        assert_equals(std::string(""), tools_no_desc[0].description);
    }
    {
        auto tools_minimal = common_chat_tools_parse_oaicompat(json::parse(
            R"([{"type": "function", "function": {"name": "test_func"}}])"));
        assert_equals((size_t) 1, tools_minimal.size());
        assert_equals(std::string("test_func"), tools_minimal[0].name);
        assert_equals(std::string(""), tools_minimal[0].description);
        assert_equals(std::string("{}"), tools_minimal[0].parameters);
    }
}

// for compat; ref: https://github.com/ggml-org/llama.cpp/pull/18961
struct test_parser_params {
    common_chat_format       format                = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format  reasoning_format      = COMMON_REASONING_FORMAT_NONE;
    bool                     reasoning_in_content  = false;
    bool                     thinking_forced_open  = false;
    bool                     parse_tool_calls      = true;
};

static common_chat_msg test_chat_parse(const std::string & input, bool is_partial, const test_parser_params & syntax) {
    common_chat_parser_params params;
    params.format               = syntax.format;
    params.reasoning_format     = syntax.reasoning_format;
    params.reasoning_in_content = syntax.reasoning_in_content;
    params.thinking_forced_open = syntax.thinking_forced_open;
    params.parse_tool_calls     = syntax.parse_tool_calls;
    return common_chat_parse(input, is_partial, params);
}

static void test_template_output_parsers() {
    printf("[%s]\n", __func__);

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    common_chat_templates_inputs inputs_tools_builtin;
    inputs_tools_builtin.messages           = {message_user};
    inputs_tools_builtin.tools              = {python_tool};

    {
        // Not supported yet
        auto tmpls = read_templates("models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja");
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GENERIC, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    }
    {
        auto tmpls = read_templates("models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja");
        std::vector<std::string>   end_tokens{ "<|END_OF_TURN_TOKEN|>" };

        for (const auto & inputs : { inputs_no_tools, inputs_tools }) {
            auto params = common_chat_templates_apply(tmpls.get(), inputs);
            assert_equals(COMMON_CHAT_FORMAT_COMMAND_R7B, params.format);
            assert_equals(false, params.thinking_forced_open);
        }

        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_COMMAND_R7B}));
        assert_msg_equals(message_assist,
            test_chat_parse(
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_COMMAND_R7B}));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            test_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ true,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_r7b,
            test_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_COMMAND_R7B}));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_call_idx,
            test_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_ACTION|>[\n"
                "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
                "]<|END_ACTION|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_no_content,
            test_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_ACTION|>[\n"
                "    {\"tool_call_id\": \"0\", \"tool_name\": \"special",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        test_templates(tmpls.get(), end_tokens, message_assist_call_idx, tools,
                      "<|START_THINKING|><|END_THINKING|>"
                      "<|START_ACTION|>[\n"
                      "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
                      "]<|END_ACTION|>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      COMMON_REASONING_FORMAT_DEEPSEEK);
        test_templates(tmpls.get(), end_tokens, message_assist, tools,
                      "<|START_RESPONSE|>Hello, world!\n"
                      "What's up?<|END_RESPONSE|>",
                      /* expect_grammar_triggered= */ false);
    }
    // TODO @ngxson : generic tool calls is too costly to maintain, consider removing it in the future
    {
        auto tmpls = read_templates("models/templates/google-gemma-2-2b-it.jinja");
        std::vector<std::string>   end_tokens{ "<end_of_turn>" };

        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GENERIC, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GENERIC,
                      common_chat_templates_apply(
                          read_templates("models/templates/microsoft-Phi-3.5-mini-instruct.jinja").get(),
                          inputs_tools)
                          .format);

        // Generic tool calls doesn't generate / parse content-only messages symmetrically.

        assert_equals(
            simple_assist_msg("{ \"tool_call\" : { \"name\" : \"t"),
            test_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"t",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GENERIC,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                    /* .parse_tool_calls = */ false,
                }));
        assert_equals(
            message_assist_empty,
            test_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"t",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));

        assert_equals(
            simple_assist_msg("", "", "puppeteer_screenshot", "{\"name\":\"servethehome_homepage\","),
            test_chat_parse(
                R"({"tool_call": {"name": "puppeteer_screenshot", "arguments": {"name": "servethehome_homepage",)",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));

        assert_equals(
            message_assist_call_empty_args,
            test_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"special_function\"",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));
        assert_equals(
            message_assist_call_cutoff_args,
            test_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"special_function\", \"arguments\" : { \"arg",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));

        assert_msg_equals(message_assist,
            test_chat_parse(
                "{\n"
                "  \"response\": \"Hello, world!\\nWhat's up?\"\n"
                "}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GENERIC}));
#if 0
        test_templates(tmpls.get(), end_tokens, message_assist_call_id, tools,
                      "{\n"
                      "  \"tool_calls\": [\n"
                      "    {\n"
                      "      \"name\": \"special_function\",\n"
                      "      \"arguments\": {\n"
                      "        \"arg1\": 1\n"
                      "      },\n"
                      "      \"id\": \"123456789\"\n"
                      "    }\n"
                      "  ],\n"
                      "  \"content\": \"\"\n"
                      "}");
#endif
    }
    {
        auto tmpls = read_templates("models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja");
        std::vector<std::string>   end_tokens{ "</s>" };

        assert_equals(COMMON_CHAT_FORMAT_MISTRAL_NEMO, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(
            tmpls.get(), end_tokens, message_assist_call_id, tools,
            "[TOOL_CALLS][{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, \"id\": \"123456789\"}]");
    }
    {
        assert_msg_equals(
            simple_assist_msg("Réponse", "raisonnement"),
            test_chat_parse(
                message_assist_thoughts_unparsed_magistral.content,
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_MAGISTRAL,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
    }
    {
        auto tmpls = read_templates("models/templates/Qwen-QwQ-32B.jinja");
        std::vector<std::string> end_tokens{ "<|im_end|>" };

        assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    }
    {
        auto tmpls = read_templates("models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja");
        std::vector<std::string> end_tokens{ "<|im_end|>" };

        assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(
            COMMON_CHAT_FORMAT_HERMES_2_PRO,
            common_chat_templates_apply(
                read_templates("models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja").get(),
                inputs_tools)
                .format);
        assert_equals(
            COMMON_CHAT_FORMAT_HERMES_2_PRO,
            common_chat_templates_apply(
                read_templates("models/templates/Qwen-Qwen2.5-7B-Instruct.jinja").get(),
                inputs_tools)
                .format);

        // Test parsing
        assert_msg_equals(
            simple_assist_msg("", "", "python", ""),
            test_chat_parse(
                "```json\n"
                "<function_call> { \"name\" : \"python\"",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            simple_assist_msg("Let's call something\n"),
            test_chat_parse(
                "Let's call something\n"
                "<tool_call>{\"name\"",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(
            simple_assist_msg("Let's call something\n"),
            test_chat_parse(
                "Let's call something\n"
                "<tool_call>{\"name",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_call_thoughts,
            test_chat_parse(
                // QwQ-32B's template adds a trailing <think> if add_generation_prompt
                "I'm\nthinking</think>\n"
                "<tool_call>{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}</tool_call>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<tool_call>\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(message_assist_call_content,
            test_chat_parse(
                "Hello, world!\nWhat's up?<tool_call>\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<function=special_function>{\"arg1\": 1}</function>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<function name=\"special_function\">\n"
                "{\"arg1\": 1}\n"
                "</function>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<tool>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tool>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<tools>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tools>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<response>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</response>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "```xml\n"
                "<response>\n"
                "    {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</response>\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "```xml\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "```\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "```\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "```json\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "```json\n"
                "\n"
                "                    <function_call> {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}} \n"
                "                    </function_call> \n"
                "``` ",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<json>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</json>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<xml>\n"
                "  {\n"
                "    \"name\": \"special_function\", \"arguments\": {\"arg1\": 1}\n"
                "  }\n"
                "</xml>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<JSON>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</JSON>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "{\n  \"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));

        // Test multiple tool calls
        common_chat_msg message_assist_multiple_calls;
        message_assist_multiple_calls.role = "assistant";
        message_assist_multiple_calls.content = "";
        message_assist_multiple_calls.tool_calls.push_back({"special_function", "{\"arg1\": 1}", ""});
        message_assist_multiple_calls.tool_calls.push_back({"python", "{\"code\":\"print('hello')\"}", ""});

        assert_msg_equals(
            message_assist_multiple_calls,
            test_chat_parse(
                "<tool_call>\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tool_call>\n"
                "<tool_call>\n"
                "{\"name\": \"python\", \"arguments\": {\"code\":\"print('hello')\"}}\n"
                "</tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));

        assert_msg_equals(
            message_assist_multiple_calls,
            test_chat_parse(
                "<function=special_function>{\"arg1\": 1}</function>\n"
                "<function=python>{\"code\":\"print('hello')\"}</function>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));

        assert_msg_equals(
            simple_assist_msg(
                "This is not a tool call:",
                "",
                "special_function",
                "{\"arg1\": 1}"),
            test_chat_parse(
                "This is not a tool call:\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        // assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
        //     test_chat_parse(
        //         "I'm\nthinking</think>Hello, world!\nWhat's up?",
        //         COMMON_CHAT_FORMAT_HERMES_2_PRO));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_md,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ true,
                    /* .thinking_forced_open = */ false,
                    /* .parse_tool_calls = */ false,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_md_partial,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ true,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts_unopened_unparsed,
            test_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<tool_call>\n"
                      "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                      "</tool_call>");

        // Test multiple tool calls with template
        common_chat_msg message_assist_multiple_calls_template;
        message_assist_multiple_calls_template.role = "assistant";
        message_assist_multiple_calls_template.content = "";
        message_assist_multiple_calls_template.tool_calls.push_back({"special_function", "{\"arg1\": 1}", ""});
        message_assist_multiple_calls_template.tool_calls.push_back({"python", "{\"code\":\"print('test')\"}", ""});

        test_templates(tmpls.get(), end_tokens, message_assist_multiple_calls_template, tools,
                      "<tool_call>\n"
                      "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                      "</tool_call>\n"
                      "<tool_call>\n"
                      "{\"name\": \"python\", \"arguments\": {\"code\":\"print('test')\"}}\n"
                      "</tool_call>");

        test_templates(tmpls.get(), end_tokens, message_assist_call_python_lines, tools,
                      "<tool_call>\n"
                      "{\"name\": \"python\", \"arguments\": {\"code\":\"# This is a program:\\nprint('hey')\"}}\n"
                      "</tool_call>");
        assert_msg_equals(
            simple_assist_msg("", /* reasoning_content= */ "<tool_call>nah uhg</tool_call>"),
            test_chat_parse(
                "<think><tool_call>nah uhg</tool_call>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
    }
    {
        auto tmpls = read_templates("models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                      common_chat_templates_apply(tmpls.get(), inputs_tools_builtin).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                      common_chat_templates_apply(
                          read_templates("models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja").get(),
                          inputs_tools_builtin)
                          .format);

        assert_equals(
            message_assist_call,
            test_chat_parse(
                "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LLAMA_3_X}));

        // test_templates(tmpls.get(), end_tokens, message_assist, tools, R"(?)", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call_code_interpreter, llama_3_1_tools,
                      "<|python_tag|>code_interpreter.call(code=\"print('hey')\")");
        test_templates(tmpls.get(), end_tokens, message_assist_call_python, tools,
                      "<|python_tag|>python.call(code=\"print('hey')\")");
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }
    {
        auto tmpls = read_templates("models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }
    {
        auto tmpls = read_templates("models/templates/meetkai-functionary-medium-v3.1.jinja");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY,
                      common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
            common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY,
                        common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

        for (auto is_partial : { false, true }) {
            assert_equals(
                message_assist_call,
                test_chat_parse(
                    "<function=special_function>{\"arg1\": 1}</function>",
                    is_partial,
                    {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1}));
        }

        assert_equals(
            message_assist_call,
            test_chat_parse(
                "<function=special_function>{\"arg1\": 1}<",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1}));

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<function=special_function>{\"arg1\": 1}</function>");
    }
    {
        auto tmpls = read_templates("models/templates/meetkai-functionary-medium-v3.2.jinja");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        assert_msg_equals(
            simple_assist_msg(
                "Hello, world!\nnono\nWhat's up?",
                "",
                "special_function",
                "{\"arg1\": 1}"),
            test_chat_parse(
                "all\n"
                "Hello, world!\n"
                "nono\n"
                "What's up?>>>special_function\n"
                "{\"arg1\": 1}\n",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist_call_python_lines,
            test_chat_parse(
                "python\n"
                "# This is a program:\n"
                "print('hey')",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist_call_python_lines_unclosed,
            test_chat_parse(
                "python\n"
                "# This is a program:\n"
                "print('hey')",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "special_function\n"
                "{\"arg1\": 1} \n                    ",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist,
            test_chat_parse(
                "all\n"
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));

        test_templates(tmpls.get(), end_tokens, message_assist, {},
                      "all\n"
                      "Hello, world!\n"
                      "What's up?",
                      /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "special_function\n"
                      "{\"arg1\": 1}");
    }
    {
        auto tmpls = read_templates("models/templates/fireworks-ai-llama-3-firefunction-v2.jinja");
        std::vector<std::string>   end_tokens{ "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_FIREFUNCTION_V2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      " functools[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]");
    }
    {
        // Original DeepSeek R1 template. Leaves <｜tool▁calls▁begin｜> and others unclosed. Our logic fixes the prompt.
        auto tmpls = read_templates("models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja");
        std::vector<std::string>   end_tokens{ "<｜end▁of▁sentence｜>" };

        for (const auto & inputs : { inputs_no_tools, inputs_tools }) {
            auto params = common_chat_templates_apply(tmpls.get(), inputs);
            assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1, params.format);
            assert_equals(true, params.thinking_forced_open);
        }

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(
            simple_assist_msg("Hello, world!\nWhat's up?", "<think>I'm\nthinking"),
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));
        assert_msg_equals(
            simple_assist_msg("", "I need to remember the correct syntax. It starts with <｜tool▁calls▁begin｜> and ends with"),
            test_chat_parse(
                "I need to remember the correct syntax. It starts with <｜tool▁calls▁begin｜> and ends with",
                /* is_partial= */ true,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_unopened_unparsed,
            test_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));
        assert_msg_equals(message_assist_thoughts,
            // Latest template update (ast of 20250209) adds a trailing <think>\n if add_generation_prompt is true.
            test_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));
        // test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
        //               "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
        //               "```json\n"
        //               "{\"arg1\": 1}\n"
        //               // Look what's not here: <｜tool▁calls▁end｜> (also missing the <｜end▁of▁sentence｜>, but that is removed lazily by the test's delta logic)
        //               "```<｜tool▁call▁end｜>",
        //               /* expect_grammar_triggered= */ true,
        //               /* test_grammar_if_triggered= */ false);
    }
    {
        // Replacement DeepSeek R1 template. Makes the Distill Qwen 7B/32B models happy to call tools and all.
        auto tmpls = read_templates("models/templates/llama-cpp-deepseek-r1.jinja");
        std::vector<std::string>   end_tokens{ "<｜end▁of▁sentence｜>" };

        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));

        assert_msg_equals(message_assist_call_thoughts_unparsed,
            test_chat_parse(
                "<think>I'm\nthinking</think>\n\n"
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "<｜tool▁calls｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));

        assert_msg_equals(message_assist_call_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>\n\n"
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>");
    }
    {
        auto tmpls = read_templates("models/templates/ibm-granite-granite-3.3-2B-Instruct.jinja");
        std::vector<std::string> end_tokens{ "<|end_of_text|>" };

        assert_equals(COMMON_CHAT_FORMAT_GRANITE, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

        assert_equals(COMMON_CHAT_FORMAT_GRANITE, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(simple_assist_msg("<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>"),
            test_chat_parse(
                "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(message_assist_empty,
            test_chat_parse(
                "<think",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_empty,
            test_chat_parse(
                "<think",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(message_assist_thoughts_no_content,
            test_chat_parse(
                "<think>I'm\nthinking",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(
            message_assist_empty,
            test_chat_parse(
                "<think>I'm\nthinking</think><response",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist_call_empty_args,
            test_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\"",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist_call_cutoff_args,
            test_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist_call_cutoff_args,
            test_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls with thinking
        assert_msg_equals(
            message_assist_call_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think><|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, {",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test template generation for regular content
        test_templates(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);
    // TODO @ngxson : generic tool call should be removed in the future
#if 0
        // Test template generation for tool calls
        test_templates(tmpls.get(), end_tokens, message_assist_call_id, tools,
                      "{\n"
                      "  \"tool_calls\": [\n"
                      "    {\n"
                      "      \"name\": \"special_function\",\n"
                      "      \"arguments\": {\n"
                      "        \"arg1\": 1\n"
                      "      },\n"
                      "      \"id\": \"123456789\"\n"
                      "    }\n"
                      "  ],\n"
                      "  \"content\": \"\"\n"
                      "}",
                      /* expect_grammar_triggered= */ false
        );
#endif
    }
    {
        auto tmpls = read_templates("models/templates/openai-gpt-oss-120b.jinja");
        std::vector<std::string> end_tokens{ "<|return|>", "<|call|>" };

        assert_equals(COMMON_CHAT_FORMAT_GPT_OSS, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GPT_OSS, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        assert_msg_equals(simple_assist_msg("", "I'm\nthink"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthink",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function<|message|>{\"arg1",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>analysis to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary<|message|>Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary<|message|>Hello, world!\nWhat's up?<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));

        // Test parse_tool_calls == false
        assert_msg_equals(
            simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                    /* .parse_tool_calls = */ false,
                }));
        assert_msg_equals(
            simple_assist_msg("", "I'm\nthinking"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function<|message|>{\"arg1",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                    /* .parse_tool_calls = */ false,
                }));
        assert_msg_equals(
            simple_assist_msg("", "I'm\nthinking"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                    /* .parse_tool_calls = */ false,
                }));

        // Test reasoning formats
        assert_msg_equals(
            simple_assist_msg(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>Hello, world!\nWhat's up?"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_NONE,
                }));

        assert_msg_equals(
            simple_assist_msg(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>Hello, world!\nWhat's up?"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                    /* .reasoning_in_content = */ true,
                }));

        // Test tool calling in role header
        assert_msg_equals(simple_assist_msg("", "", "special_function", "{\"arg1\": 1}"),
            test_chat_parse(
                " to=functions.special_function<|channel|>commentary <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "", "special_function", "{\"arg1\": 1}"),
            test_chat_parse(
                " to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            test_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
    }
    {
        // Seed-OSS format tests
        auto tmpls = read_templates("models/templates/ByteDance-Seed-OSS.jinja");
        std::vector<std::string> end_tokens{ "<seed:eos>" };

        assert_equals(COMMON_CHAT_FORMAT_SEED_OSS, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_SEED_OSS, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);

        // Test simple reasoning content
        assert_msg_equals(
            simple_assist_msg("Hello, world!", "I'm thinking about the answer"),
            test_chat_parse(
                "<seed:think>I'm thinking about the answer</seed:think>Hello, world!",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_SEED_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test budget reflection tags
        common_chat_msg msg_budget_reflect;
        msg_budget_reflect.role = "assistant";
        msg_budget_reflect.content = "<seed:cot_budget_reflect>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:cot_budget_reflect>I need to calculate this step by step.";
        msg_budget_reflect.reasoning_content = "Token usage: 45/1000\nI should continue thinking to find the best solution.";
        assert_msg_equals(
            msg_budget_reflect,
            test_chat_parse(
                "<seed:think>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:think>"
                "<seed:cot_budget_reflect>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:cot_budget_reflect>"
                "I need to calculate this step by step.",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_SEED_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test tool calls with Seed-OSS format
        common_chat_msg msg_tool_call;
        msg_tool_call.role = "assistant";
        msg_tool_call.tool_calls.push_back({"calculate_sum", "{\"numbers\": [1, 2, 3]}", ""});
        assert_msg_equals(
            msg_tool_call,
            test_chat_parse(
                "<seed:tool_call>\n"
                "<function=calculate_sum>\n"
                "<parameter=numbers>[1, 2, 3]</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_SEED_OSS}));

        // Test reasoning + tool call combination
        common_chat_msg msg_reasoning_tool;
        msg_reasoning_tool.role = "assistant";
        msg_reasoning_tool.content = "";
        msg_reasoning_tool.reasoning_content = "I need to calculate the sum of these numbers";
        msg_reasoning_tool.tool_calls.push_back({"calculate_sum", "{\"numbers\": [1, 2, 3]}", ""});
        assert_msg_equals(
            msg_reasoning_tool,
            test_chat_parse(
                "<seed:think>I need to calculate the sum of these numbers</seed:think>"
                "<seed:tool_call>\n"
                "<function=calculate_sum>\n"
                "<parameter=numbers>[1, 2, 3]</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_SEED_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test deltas: the number of tool calls in partial parses should never decrease
        std::string tool_msg = "<seed:tool_call>\n"
            "<function=fun>\n"
            "<parameter=smth>[1, 2, 3]</parameter>\n"
            "</function>";
        std::size_t previousToolCalls = 0;
        for (std::size_t i = std::string("<seed:tool_call>").length(); i < tool_msg.length() - 1; i++) {
            auto partial = tool_msg.substr(0, i);
            auto partial_res = test_chat_parse(partial, true, { COMMON_CHAT_FORMAT_SEED_OSS, COMMON_REASONING_FORMAT_DEEPSEEK });
            if (partial_res.tool_calls.size() < previousToolCalls) {
                throw std::runtime_error("Tool call size decreased on partial: " + partial + " from " + std::to_string(previousToolCalls) + " to " + std::to_string(partial_res.tool_calls.size()));
            }
            previousToolCalls = partial_res.tool_calls.size();
        }

        // Test multiple parameters in tool call
        common_chat_msg msg_multi_param;
        msg_multi_param.role = "assistant";
        msg_multi_param.tool_calls.push_back({"process_data", "{\"input\": \"test\", \"format\": \"json\"}", ""});
        assert_msg_equals(
            msg_multi_param,
            test_chat_parse(
                "<seed:tool_call>\n"
                "<function=process_data>\n"
                "<parameter=input>test</parameter>\n"
                "<parameter=format>json</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_SEED_OSS}));

        // Test partial parsing for incomplete tool call - don't actually add the call until parsing parameters is done
        assert_msg_equals(
            simple_assist_msg("", "", "calculate_sum", "{\"numbers\":"),
            test_chat_parse(
                "<seed:tool_call>\n"
                "<function=calculate_sum>\n"
                "<parameter=numbers>[1,\n",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_SEED_OSS}));

        // Test incomplete reasoning tag
        assert_msg_equals(
            simple_assist_msg("", "I was thinking"),
            test_chat_parse(
                "<seed:think>I was thinking",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_SEED_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test content without reasoning
        assert_msg_equals(
            simple_assist_msg("This is a simple response without reasoning."),
            test_chat_parse(
                "This is a simple response without reasoning.",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_SEED_OSS}));
    }
    {
        auto tmpls = read_templates("models/templates/NVIDIA-Nemotron-Nano-v2.jinja");
        std::vector<std::string> end_tokens{ "<SPECIAL_12>" };

        assert_equals(COMMON_CHAT_FORMAT_NEMOTRON_V2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_NEMOTRON_V2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_NEMOTRON_V2}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_NEMOTRON_V2}));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think><TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            test_chat_parse(
                "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_NEMOTRON_V2}
            ));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            test_chat_parse(
                "<think>I'm\nthinking</think><TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test template generation for regular content
        test_templates(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?\n",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
                      /* expect_grammar_triggered= */ true
        );
    }
    {
        auto tmpls = read_templates("models/templates/deepseek-ai-DeepSeek-V3.1.jinja");
        std::vector<std::string>   end_tokens{ "<｜end▁of▁sentence｜>" };

        for (const auto & inputs : { inputs_no_tools, inputs_tools }) {
            auto params = common_chat_templates_apply(tmpls.get(), inputs);
            assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_V3_1, params.format);
            assert_equals(true, params.thinking_forced_open);
        }

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "</think>Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_thoughts, tools, "</think>Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(
            simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
            test_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));
        // variant: thinking forced open, reasoning_format none
        assert_msg_equals(
            simple_assist_msg("REASONING</think>ok", ""),
            test_chat_parse(
                "REASONING</think>ok",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_NONE,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                    /* .parse_tool_calls = */ true,
                }));
        // variant: happy path for when it works as the model card says it should
        assert_msg_equals(
            simple_assist_msg("", "", "get_time", "{\"city\":\"Tokyo\"}"),
            test_chat_parse(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                    /* .parse_tool_calls = */ true,
                }));
        // variant: simple + thinking open
        assert_msg_equals(
            simple_assist_msg("", "REASONING", "get_time", "{\"city\":\"Tokyo\"}"),
            test_chat_parse(
                "REASONING</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                    /* .parse_tool_calls = */ true,
                }));
        // variant: simple + multiple tool calls
        common_chat_msg message_assist_multiple_calls;
        message_assist_multiple_calls.role = "assistant";
        message_assist_multiple_calls.content = "CONTENT";
        message_assist_multiple_calls.tool_calls.push_back({"get_time", "{\"city\":\"Paris\"}", ""});
        message_assist_multiple_calls.tool_calls.push_back({"get_weather", "{\"city\":\"Paris\"}", ""});
        assert_msg_equals(
            message_assist_multiple_calls,
            test_chat_parse(
                "CONTENT<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                    /* .parse_tool_calls = */ true,
                }));
        // variant: thinking forced open + tool call in reasoning content
        assert_msg_equals(
            simple_assist_msg("", "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time2<｜tool▁sep｜>{\"city\": \"Tokyo2\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>REASONING", "get_time", "{\"city\":\"Tokyo\"}"),
            test_chat_parse(
                "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time2<｜tool▁sep｜>{\"city\": \"Tokyo2\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>REASONING</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                    /* .parse_tool_calls = */ true,
                }));
        // variant: thinking forced open + tool call in reasoning content + no closing think + not partial
        //          This is a bit of a fine tuning issue on the model's part IMO. It really should not be attempting
        //          to make tool calls in reasoning content according to the model card, but it does sometimes, so
        //          add the reasoning content as regular content and parse the tool calls.
        assert_msg_equals(
            simple_assist_msg("REASONING", "", "get_time", "{\"city\":\"Tokyo\"}"),
            test_chat_parse(
                "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                    /* .parse_tool_calls = */ true,
                }));
        // variant: thinking forced open + tool call in reasoning content + no closing think + partial
        assert_msg_equals(
            simple_assist_msg("", "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>", "", ""),
            test_chat_parse(
                "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ true,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                    /* .parse_tool_calls = */ true,
                }));
        // variant: thinking not forced open + missing reasoning + no tool calls
        assert_msg_equals(
            simple_assist_msg("CONTENT", ""),
            test_chat_parse(
                "CONTENT",
                /* is_partial= */ false,
                {
                    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                    /* .parse_tool_calls = */ true,
                }));
    }
    {
        auto tmpls = read_templates("models/templates/Apertus-8B-Instruct.jinja");
        std::vector<std::string> end_tokens{ "<|assistant_end|>" };

        assert_equals(COMMON_CHAT_FORMAT_APERTUS, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_APERTUS, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            test_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|><|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            test_chat_parse(
                "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}
            ));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            test_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|><|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test template generation for regular content
        test_templates(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                      /* expect_grammar_triggered= */ true
        );

        // TODO @ngxson : not sure why this fails, but not very important for now
        // assert_equals(true, common_chat_templates_support_enable_thinking(tmpls.get()));
    }
    {
        // LFM2 format tests
        auto tmpls = read_templates("models/templates/llama-cpp-lfm2.jinja");
        std::vector<std::string> end_tokens{ "<|im_end|>" };

        auto inputs_tools_forced_json_schema = std::invoke([&]() -> common_chat_templates_inputs {
            common_chat_templates_inputs inputs;
            inputs.messages = {
                std::invoke([&]() -> common_chat_msg {
                    common_chat_msg msg;
                    msg.role = "system";
                    msg.content = "force json schema.\n";
                    return msg;
                }),
                message_user,
            };
            inputs.tools = {special_function_tool};
            return inputs;
        });

        {
            auto params = common_chat_templates_apply(tmpls.get(), inputs_no_tools);
            assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, params.format);
            assert_equals(false, params.grammar_lazy);
            assert_equals(std::string(R"(<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
)"), params.prompt);
        }

        {
            auto params = common_chat_templates_apply(tmpls.get(), inputs_tools);
            assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, params.format);
            assert_equals(false, params.grammar_lazy);
            assert_equals(std::string(R"(<|im_start|>system
List of tools: <|tool_list_start|>[{"type": "function", "function": {"name": "special_function", "description": "I'm special", "parameters": {"type": "object", "properties": {"arg1": {"type": "integer", "description": "The arg."}}, "required": ["arg1"]}}}]<|tool_list_end|><|im_end|>
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
)"), params.prompt);
            assert_equals(true, params.grammar.empty());
        }

        {
            auto params = common_chat_templates_apply(tmpls.get(), inputs_tools_forced_json_schema);
            assert_equals(COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS, params.format);
            assert_equals(true, params.grammar_lazy);
            assert_equals(std::string(R"(<|im_start|>system
List of tools: <|tool_list_start|>[{"type": "function", "function": {"name": "special_function", "description": "I'm special", "parameters": {"type": "object", "properties": {"arg1": {"type": "integer", "description": "The arg."}}, "required": ["arg1"]}}}]<|tool_list_end|><|im_end|>
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
)"), params.prompt);
            assert_equals(false, params.grammar.empty());
        }

        // Test parsing regular content
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test single tool call with JSON format
        common_chat_msg msg_single_tool_call;
        msg_single_tool_call.role = "assistant";
        msg_single_tool_call.tool_calls.push_back({"special_function", "{\"arg1\":1}", ""});
        assert_msg_equals(
            msg_single_tool_call,
            test_chat_parse(
                "<|tool_call_start|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with string argument
        common_chat_msg msg_tool_call_string;
        msg_tool_call_string.role = "assistant";
        msg_tool_call_string.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
        assert_msg_equals(
            msg_tool_call_string,
            test_chat_parse(
                "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with multiple arguments
        common_chat_msg msg_multi_args;
        msg_multi_args.role = "assistant";
        msg_multi_args.tool_calls.push_back({"calculate", "{\"x\":10,\"y\":20,\"operation\":\"add\"}", ""});
        assert_msg_equals(
            msg_multi_args,
            test_chat_parse(
                "<|tool_call_start|>[{\"name\": \"calculate\", \"arguments\": {\"x\": 10, \"y\": 20, \"operation\": \"add\"}}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test multiple tool calls in single array
        common_chat_msg msg_multiple_tools;
        msg_multiple_tools.role = "assistant";
        msg_multiple_tools.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
        msg_multiple_tools.tool_calls.push_back({"get_time", "{\"timezone\":\"UTC\"}", ""});
        assert_msg_equals(
            msg_multiple_tools,
            test_chat_parse(
                "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}, {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with content before
        common_chat_msg msg_content_before_tool;
        msg_content_before_tool.role = "assistant";
        msg_content_before_tool.content = "Let me check the weather for you.";
        msg_content_before_tool.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
        assert_msg_equals(
            msg_content_before_tool,
            test_chat_parse(
                "Let me check the weather for you.<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with content after
        common_chat_msg msg_content_after_tool;
        msg_content_after_tool.role = "assistant";
        msg_content_after_tool.content = "Here's the result.";
        msg_content_after_tool.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
        assert_msg_equals(
            msg_content_after_tool,
            test_chat_parse(
                "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>Here's the result.",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with newlines (common in LLM output)
        common_chat_msg msg_tool_call_newlines;
        msg_tool_call_newlines.role = "assistant";
        msg_tool_call_newlines.tool_calls.push_back({"get_current_time", "{\"location\":\"Paris\"}", ""});
        assert_msg_equals(
            msg_tool_call_newlines,
            test_chat_parse(
                "<|tool_call_start|>[{\n    \"name\": \"get_current_time\",\n    \"arguments\": {\n        \"location\": \"Paris\"\n    }\n}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Note: LFM2 uses JSON format for tool calls: [{"name": "...", "arguments": {...}}]
        // Unlike other formats, LFM2 template does not render tool calls in conversation history,
        // so we don't use test_templates() for tool call generation. Instead, the parsing tests
        // above verify edge cases and format variations for the tool call output format.
    }

    {
        auto tmpls = read_templates("models/templates/MiniMax-M2.jinja");
        std::vector<std::string> end_tokens{ "[e~[" };

        assert_equals(COMMON_CHAT_FORMAT_MINIMAX_M2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_MINIMAX_M2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_MINIMAX_M2}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_MINIMAX_M2,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "<minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_MINIMAX_M2}));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think><minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_MINIMAX_M2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            test_chat_parse(
                "<minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_MINIMAX_M2}
            ));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            test_chat_parse(
                "<think>I'm\nthinking</think><minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_MINIMAX_M2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test streaming
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "<think>I'm\nthinking\n</think>Hello, world!\nWhat's up?\n<minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_MINIMAX_M2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(message_assist_call_thoughts_unparsed,
            "<think>I'm\nthinking</think>\n\n<minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_MINIMAX_M2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_NONE
            }); });
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "<think>I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?\n\n<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>\n",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_MINIMAX_M2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(message_assist_call_withopt,
            "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n<parameter name=\"arg2\">2</parameter>\n</invoke>\n</minimax:tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_MINIMAX_M2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_NONE
            }); });

        // Test template generation for regular content
        test_templates(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                      /* ignore_whitespace_differences= */ true
        );

        // Test template generation for tools with optional parameters
        test_templates(tmpls.get(), end_tokens, message_assist_call_noopt, tools,
                      "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                      /* ignore_whitespace_differences= */ true
        );
        test_templates(tmpls.get(), end_tokens, message_assist_call_withopt, tools,
                      "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n<parameter name=\"arg2\">2</parameter>\n</invoke>\n</minimax:tool_call>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                      /* ignore_whitespace_differences= */ true
        );
    }

    {
        auto tmpls = read_templates("models/templates/GLM-4.6.jinja");
        std::vector<std::string>   end_tokens{ "<|assistant|>", "<|observation|>" };

        assert_equals(COMMON_CHAT_FORMAT_GLM_4_5, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GLM_4_5, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GLM_4_5}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "\n<think>I'm\nthinking</think>\nHello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }), true);

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GLM_4_5}), true);

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            test_chat_parse(
                "\n<think>I'm\nthinking</think>\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }), true);

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            test_chat_parse(
                "\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GLM_4_5}
            ), true);

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            test_chat_parse(
                "\n<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }), true);

        // Test streaming
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "\n<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(message_assist_call_thoughts_unparsed,
            "\n<think>I'm\nthinking</think>\n\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_NONE
            }); });
        test_parser_with_streaming(message_assist_call_withopt,
            "\n<think></think>\n<tool_call>special_function_with_opt\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n<arg_key>arg2</arg_key>\n<arg_value>2</arg_value>\n</tool_call>\n",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
            test_parser_with_streaming(
                simple_assist_msg("", "", "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
                "<tool_call>complex_function\n"
                "<arg_key>name</arg_key>\n"
                "<arg_value>John Doe</arg_value>\n"
                "<arg_key>age</arg_key>\n"
                "<arg_value>30</arg_value>\n"
                "<arg_key>active</arg_key>\n"
                "<arg_value>true</arg_value>\n"
                "<arg_key>score</arg_key>\n"
                "<arg_value>95.5</arg_value>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_GLM_4_5}); });
        test_parser_with_streaming(
                simple_assist_msg("", "", "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"),
                "<tool_call>web_search\n"
                "<arg_key>query</arg_key>\n"
                "<arg_value>\"From Zero\" Linkin Park album tracklist complete songs</arg_value>\n"
                "<arg_key>limit</arg_key>\n"
                "<arg_value>3</arg_value>\n"
                "<arg_key>type</arg_key>\n"
                "<arg_value>text</arg_value>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_GLM_4_5}); });

        // Test interleaved thinking
        test_parser_with_streaming(simple_assist_msg("Hello, world!\n\nWhat's up?", "I'm\nthinkingThinking2", "special_function", "{\"arg1\": 1}"),
            "\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(simple_assist_msg("\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?", "", "special_function", "{\"arg1\": 1}"),
            "\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_GLM_4_5,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_NONE
            }); });

        // Test template generation for regular content
        test_templates(tmpls.get(), end_tokens, message_assist, tools,
                      "\n<think></think>\nHello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "\n<think></think>\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>\n",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ false,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                      /* ignore_whitespace_differences= */ true
        );

        // Test template generation for tools with optional parameters
        test_templates(tmpls.get(), end_tokens, message_assist_call_noopt, tools,
                      "\n<think></think>\n<tool_call>special_function_with_opt\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>\n",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ false,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                      /* ignore_whitespace_differences= */ true
        );
        test_templates(tmpls.get(), end_tokens, message_assist_call_withopt, tools,
                      "\n<think></think>\n<tool_call>special_function_with_opt\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n<arg_key>arg2</arg_key>\n<arg_value>2</arg_value>\n</tool_call>\n",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ false,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                      /* ignore_whitespace_differences= */ true
        );
    }

    {
        auto tmpls = read_templates("models/templates/Kimi-K2-Thinking.jinja");
        std::vector<std::string> end_tokens{ "<|im_end|>" };

        assert_equals(COMMON_CHAT_FORMAT_KIMI_K2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_KIMI_K2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            test_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_KIMI_K2}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            test_chat_parse(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_KIMI_K2}));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            test_chat_parse(
                "<think>I'm\nthinking</think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            test_chat_parse(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_KIMI_K2}
            ));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            test_chat_parse(
                "<think>I'm\nthinking</think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test streaming
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "<think>I'm\nthinking\n</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(message_assist_call_thoughts_unparsed,
            "<think>I'm\nthinking</think>\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_NONE
            }); });
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "<think>I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>\n",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(message_assist_call_withopt,
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function_with_opt:0<|tool_call_argument_begin|>{\"arg1\": 1, \"arg2\": 2}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_NONE
            }); });
        test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": \"123456\"}"),
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": \"123456\"}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": [1, 2, \"345\", 6]}"),
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": [1, 2, \"345\", 6]}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": {\"12\": 34, \"5\": [67, 8], \"9\": \"10\"}}"),
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": {\"12\": 34, \"5\": [67, 8], \"9\": \"10\"}}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    /*  .format = */ COMMON_CHAT_FORMAT_KIMI_K2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(
                simple_assist_msg("", "", "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function:0<|tool_call_argument_begin|>"
                "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_KIMI_K2}); });
        test_parser_with_streaming(
                simple_assist_msg("", "", "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"),
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0<|tool_call_argument_begin|>"
                "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_KIMI_K2}); });
        test_parser_with_streaming(
                simple_assist_msg("", "", "read_file", "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}"),
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
                "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_KIMI_K2}); });
        test_parser_with_streaming(
                simple_assist_msg(
                        "Let me start by examining the relevant files to understand the current implementation.", "",
                        "read_file",
                        "{\"files\": [{\"path\": \"src/app/Partners.tsx\", \"line_ranges\": [\"1-100\"]}]}"),
                "Let me start by examining the relevant files to understand the current implementation."
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
                "{\"files\":[{\"path\":\"src/app/Partners.tsx\",\"line_ranges\":[\"1-100\"]}]}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_KIMI_K2}); });
        auto multi_tool_msg = simple_assist_msg("Let me call multiple tools.", "I'm thinking.");
        multi_tool_msg.tool_calls.push_back({ "read_file", "{\"files\": [{\"path\": \"src/app/Partners.tsx\", \"line_ranges\": [\"1-100\"]}]}", "" });
        multi_tool_msg.tool_calls.push_back({ "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}", "" });
        multi_tool_msg.tool_calls.push_back({ "complex_function", "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}", "" });
        multi_tool_msg.tool_calls.push_back({ "emoji_function", "{\"message\":\"Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\"}", "" });
        test_parser_with_streaming(multi_tool_msg,
                "<think>I'm thinking.</think>Let me call multiple tools."
                "<|tool_calls_section_begin|>"
                "<|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
                "{\"files\":[{\"path\":\"src/app/Partners.tsx\",\"line_ranges\":[\"1-100\"]}]}"
                "<|tool_call_end|>"
                "<|tool_call_begin|>functions.web_search:1<|tool_call_argument_begin|>"
                "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"
                "<|tool_call_end|>"
                "<|tool_call_begin|>functions.complex_function:2<|tool_call_argument_begin|>"
                "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
                "<|tool_call_end|>"
                "<|tool_call_begin|>functions.emoji_function:3<|tool_call_argument_begin|>"
                "{\"message\":\"Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\"}"
                "<|tool_call_end|>"
                "<|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    COMMON_CHAT_FORMAT_KIMI_K2,
                    COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(
                simple_assist_msg("", "I'm thinking", "complex_function_in_think", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
                "<think>I'm thinking<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function_in_think:0<|tool_call_argument_begin|>"
                "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    COMMON_CHAT_FORMAT_KIMI_K2,
                    COMMON_REASONING_FORMAT_DEEPSEEK
            }); });
        test_parser_with_streaming(
                simple_assist_msg("Hello", "I'm thinkingI'm still thinking", "complex_function_in_think", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
                "<think>I'm thinking<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function_in_think:0<|tool_call_argument_begin|>"
                "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
                "<|tool_call_end|><|tool_calls_section_end|>I'm still thinking</think>Hello",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {
                    COMMON_CHAT_FORMAT_KIMI_K2,
                    COMMON_REASONING_FORMAT_DEEPSEEK
            }); });

        // Test template rendering
        common_chat_templates_inputs conversation_with_tools = inputs_tools;
        conversation_with_tools.messages.push_back(simple_assist_msg("Let's do it", "Think first", "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"));
        conversation_with_tools.messages.push_back({
            "tool",
            "Tool response 1",
            /* .content_parts = */ {},
            /* .tool_calls = */ {},
            /* .reasoning_content = */ "",
            /* .tool_name = */ "complex_function",
            /* .tool_call_id = */ "",
        });
        conversation_with_tools.messages.push_back(simple_assist_msg("Continue", "Think next", "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"));
        conversation_with_tools.messages.push_back({
            "tool",
            "Tool response 2",
            /* .content_parts = */ {},
            /* .tool_calls = */ {},
            /* .reasoning_content = */ "",
            /* .tool_name = */ "web_search",
            /* .tool_call_id = */ "",
        });
        conversation_with_tools.messages.push_back(simple_assist_msg("CC", "Think last", "read_file", "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}"));
        conversation_with_tools.messages.push_back({
            "tool",
            "Tool response 3",
            /* .content_parts = */ {},
            /* .tool_calls = */ {},
            /* .reasoning_content = */ "",
            /* .tool_name = */ "read_file",
            /* .tool_call_id = */ "",
        });
        assert_equals(common_chat_templates_apply(tmpls.get(), conversation_with_tools).prompt, std::string("<|im_system|>tool_declare<|im_middle|>[{\"type\": \"function\", \"function\": {\"name\": \"special_function\", \"description\": \"I'm special\", \"parameters\": {\"type\": \"object\", \"properties\": {\"arg1\": {\"type\": \"integer\", \"description\": \"The arg.\"}}, \"required\": [\"arg1\"]}}}]<|im_end|><|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>Hey there!<|im_end|><|im_assistant|>assistant<|im_middle|><think>Think first</think>Let's do it<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function:0<|tool_call_argument_begin|>{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>complex_function<|im_middle|>## Return of functions.complex_function:0\nTool response 1<|im_end|><|im_assistant|>assistant<|im_middle|><think>Think next</think>Continue<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:1<|tool_call_argument_begin|>{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>web_search<|im_middle|>## Return of functions.web_search:1\nTool response 2<|im_end|><|im_assistant|>assistant<|im_middle|><think>Think last</think>CC<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:2<|tool_call_argument_begin|>{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>read_file<|im_middle|>## Return of functions.read_file:2\nTool response 3<|im_end|><|im_assistant|>assistant<|im_middle|>"));

        // Test template generation for regular content
        test_templates(tmpls.get(), end_tokens, message_assist, tools,
                      "<think></think>Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<think></think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                      /* ignore_whitespace_differences= */ true
        );

        // Test template generation for tools with optional parameters
        test_templates(tmpls.get(), end_tokens, message_assist_call_noopt, tools,
                      "<think></think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function_with_opt:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                      /* ignore_whitespace_differences= */ true
        );
        test_templates(tmpls.get(), end_tokens, message_assist_call_withopt, tools,
                      "<think></think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function_with_opt:0<|tool_call_argument_begin|>{\"arg1\": 1, \"arg2\": 2}<|tool_call_end|><|tool_calls_section_end|>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                      /* ignore_whitespace_differences= */ true
        );
    }

    // Test Qwen3-Coder XML format
    {
        // Basic XML tool call parsing
        assert_msg_equals(
            message_assist_call,
            test_chat_parse(
                "<tool_call>\n"
                "  <function=special_function>\n"
                "    <parameter=arg1>\n"
                "      1\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}));

        // Multiple parameters with different types
        common_chat_msg expected_multi_param;
        expected_multi_param.role = "assistant";
        expected_multi_param.tool_calls = {
            { "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}", "" }
        };

        test_parser_with_streaming(expected_multi_param,
                "<tool_call>\n"
                "  <function=complex_function>\n"
                "    <parameter=name>\n"
                "      John Doe\n"
                "    </parameter>\n"
                "    <parameter=age>\n"
                "      30\n"
                "    </parameter>\n"
                "    <parameter=active>\n"
                "      true\n"
                "    </parameter>\n"
                "    <parameter=score>\n"
                "      95.5\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Special characters and Unicode
        common_chat_msg expected_special_chars;
        expected_special_chars.role = "assistant";
        expected_special_chars.tool_calls = {
            { "unicode_function", "{\"message\":\"Hello 世界! 🌍 Special chars: @#$%^&*()\"}", "" }
        };

        test_parser_with_streaming(expected_special_chars,
                "<tool_call>\n"
                "  <function=unicode_function>\n"
                "    <parameter=message>\n"
                "      Hello 世界! 🌍 Special chars: @#$%^&*()\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Multiline content with newlines and indentation
        common_chat_msg expected_multiline;
        expected_multiline.role = "assistant";
        expected_multiline.tool_calls = {
            { "code_function", "{\"code\":\"def hello():\\n    print(\\\"Hello, World!\\\")\\n    return True\"}", "" }
        };

        test_parser_with_streaming(expected_multiline,
                "<tool_call>\n"
                "  <function=code_function>\n"
                "    <parameter=code>\n"
                "def hello():\n"
                "    print(\"Hello, World!\")\n"
                "    return True\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // JSON object as parameter value
        common_chat_msg expected_json_param;
        expected_json_param.role = "assistant";
        expected_json_param.tool_calls = {
            { "json_function", "{\"config\":{\"host\":\"localhost\",\"port\":8080,\"ssl\":false}}", "" }
        };

        test_parser_with_streaming(
            expected_json_param,
                "<tool_call>\n"
                "  <function=json_function>\n"
                "    <parameter=config>\n"
                "      {\"host\": \"localhost\", \"port\": 8080, \"ssl\": false}\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Array as parameter value
        common_chat_msg expected_array_param;
        expected_array_param.role = "assistant";
        expected_array_param.tool_calls = {
            { "array_function", "{\"items\":[\"apple\",\"banana\",\"cherry\"]}", "" }
        };

        test_parser_with_streaming(
            expected_array_param,
                "<tool_call>\n"
                "  <function=array_function>\n"
                "    <parameter=items>\n"
                "      [\"apple\", \"banana\", \"cherry\"]\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Empty parameter
        common_chat_msg expected_empty_param;
        expected_empty_param.role = "assistant";
        expected_empty_param.tool_calls = {
            { "empty_function", "{\"empty_param\":\"\"}", "" }
        };

        test_parser_with_streaming(
            expected_empty_param,
                "<tool_call>\n"
                "  <function=empty_function>\n"
                "    <parameter=empty_param>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Boolean values (true/false)
        common_chat_msg expected_boolean;
        expected_boolean.role = "assistant";
        expected_boolean.tool_calls = {
            { "boolean_function", "{\"enabled\":true,\"debug\":false}", "" }
        };

        test_parser_with_streaming(
            expected_boolean,
                "<tool_call>\n"
                "  <function=boolean_function>\n"
                "    <parameter=enabled>\n"
                "      true\n"
                "    </parameter>\n"
                "    <parameter=debug>\n"
                "      false\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Null value
        common_chat_msg expected_null;
        expected_null.role = "assistant";
        expected_null.tool_calls = {
            { "null_function", "{\"optional_param\":null}", "" }
        };

        test_parser_with_streaming(
            expected_null,
                "<tool_call>\n"
                "  <function=null_function>\n"
                "    <parameter=optional_param>\n"
                "      null\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Negative numbers and scientific notation
        common_chat_msg expected_numbers;
        expected_numbers.role = "assistant";
        expected_numbers.tool_calls = {
            { "math_function", "{\"negative\":-42,\"decimal\":-3.14,\"scientific\":1.23e-4}", "" }
        };

        test_parser_with_streaming(
            expected_numbers,
                "<tool_call>\n"
                "  <function=math_function>\n"
                "    <parameter=negative>\n"
                "      -42\n"
                "    </parameter>\n"
                "    <parameter=decimal>\n"
                "      -3.14\n"
                "    </parameter>\n"
                "    <parameter=scientific>\n"
                "      1.23e-4\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // XML-like content in parameters (should be escaped)
        common_chat_msg expected_xml_content;
        expected_xml_content.role = "assistant";
        expected_xml_content.tool_calls = {
            { "xml_function", "{\"xml_content\":\"<root><item>value</item></root>\"}", "" }
        };

        test_parser_with_streaming(
            expected_xml_content,
                "<tool_call>\n"
                "  <function=xml_function>\n"
                "    <parameter=xml_content>\n"
                "      <root><item>value</item></root>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Quotes and escape characters
        common_chat_msg expected_quotes;
        expected_quotes.role = "assistant";
        expected_quotes.tool_calls = {
            { "quote_function", "{\"message\":\"She said \\\"Hello!\\\" and left.\"}", "" }
        };

        test_parser_with_streaming(
            expected_quotes,
                "<tool_call>\n"
                "  <function=quote_function>\n"
                "    <parameter=message>\n"
                "      She said \"Hello!\" and left.\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Long parameter value (simplified)
        std::string long_text = "This is a long text parameter that should test the parser's ability to handle larger amounts of text data.";

        common_chat_msg expected_long_text;
        expected_long_text.role = "assistant";
        expected_long_text.tool_calls = {
            { "long_function", "{\"long_text\":\"" + long_text + "\"}", "" }
        };

        test_parser_with_streaming(
            expected_long_text,
                "<tool_call>\n"
                "  <function=long_function>\n"
                "    <parameter=long_text>\n"
                "      " + long_text + "\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Mixed content with text before and after tool call
        common_chat_msg expected_mixed_content;
        expected_mixed_content.role = "assistant";
        expected_mixed_content.content = "I'll help you search for products. ";
        expected_mixed_content.tool_calls = {
            { "search_function", "{\"query\":\"laptops\"}", "" }
        };

        test_parser_with_streaming(
            expected_mixed_content,
                "I'll help you search for products. <tool_call>\n"
                "  <function=search_function>\n"
                "    <parameter=query>\n"
                "      laptops\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Compact format (no extra whitespace)
        common_chat_msg expected_compact;
        expected_compact.role = "assistant";
        expected_compact.tool_calls = {
            { "compact_function", "{\"param\":\"value\"}", "" }
        };

        test_parser_with_streaming(
            expected_compact,
                "<tool_call><function=compact_function><parameter=param>value</parameter></function></tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Function name with underscores and numbers
        common_chat_msg expected_complex_name;
        expected_complex_name.role = "assistant";
        expected_complex_name.tool_calls = {
            { "get_user_data_v2", "{\"user_id\":12345}", "" }
        };

        test_parser_with_streaming(
            expected_complex_name,
                "<tool_call>\n"
                "  <function=get_user_data_v2>\n"
                "    <parameter=user_id>\n"
                "      12345\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Parameter names with underscores and numbers
        common_chat_msg expected_complex_params;
        expected_complex_params.role = "assistant";
        expected_complex_params.tool_calls = {
            { "test_function", "{\"param_1\":\"value1\",\"param_2_name\":\"value2\",\"param3\":123}", "" }
        };

        test_parser_with_streaming(
            expected_complex_params,
                "<tool_call>\n"
                "  <function=test_function>\n"
                "    <parameter=param_1>\n"
                "      value1\n"
                "    </parameter>\n"
                "    <parameter=param_2_name>\n"
                "      value2\n"
                "    </parameter>\n"
                "    <parameter=param3>\n"
                "      123\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Very deeply nested XML content in parameter
        common_chat_msg expected_deep_xml;
        expected_deep_xml.role = "assistant";
        expected_deep_xml.tool_calls = {
            { "xml_parser", "{\"xml\":\"<root><level1><level2><level3>deep content</level3></level2></level1></root>\"}", "" }
        };

        test_parser_with_streaming(
            expected_deep_xml,
                "<tool_call>\n"
                "  <function=xml_parser>\n"
                "    <parameter=xml>\n"
                "      <root><level1><level2><level3>deep content</level3></level2></level1></root>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Parameter with only whitespace
        common_chat_msg expected_whitespace_param;
        expected_whitespace_param.role = "assistant";
        expected_whitespace_param.tool_calls = {
            { "whitespace_function", "{\"spaces\":\"\"}", "" }
        };

        test_parser_with_streaming(
            expected_whitespace_param,
                "<tool_call>\n"
                "  <function=whitespace_function>\n"
                "    <parameter=spaces>\n"
                "      \n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Parameter with tabs and mixed whitespace
        common_chat_msg expected_mixed_whitespace;
        expected_mixed_whitespace.role = "assistant";
        expected_mixed_whitespace.tool_calls = {
            { "tab_function", "{\"content\":\"line1\\n\\tindented line\\n    spaces\"}", "" }
        };

        test_parser_with_streaming(
            expected_mixed_whitespace,
                "<tool_call>\n"
                "  <function=tab_function>\n"
                "    <parameter=content>\n"
                "line1\n"
                "\tindented line\n"
                "    spaces\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Control characters and special Unicode
        common_chat_msg expected_control_chars;
        expected_control_chars.role = "assistant";
        expected_control_chars.tool_calls = {
            { "control_function", "{\"text\":\"Line1\\nLine2\\tTabbed\\rCarriage return\"}", "" }
        };

        test_parser_with_streaming(
            expected_control_chars,
                "<tool_call>\n"
                "  <function=control_function>\n"
                "    <parameter=text>\n"
                "Line1\nLine2\tTabbed\rCarriage return\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Emoji and extended Unicode characters
        common_chat_msg expected_emoji;
        expected_emoji.role = "assistant";
        expected_emoji.tool_calls = {
            { "emoji_function", "{\"message\":\"Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\"}", "" }
        };

        test_parser_with_streaming(
            expected_emoji,
                "<tool_call>\n"
                "  <function=emoji_function>\n"
                "    <parameter=message>\n"
                "      Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Mathematical expressions and formulas
        common_chat_msg expected_math;
        expected_math.role = "assistant";
        expected_math.tool_calls = {
            { "math_function", "{\"formula\":\"E = mc² and ∫f(x)dx = F(x) + C\"}", "" }
        };

        test_parser_with_streaming(
            expected_math,
                "<tool_call>\n"
                "  <function=math_function>\n"
                "    <parameter=formula>\n"
                "      E = mc² and ∫f(x)dx = F(x) + C\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // SQL injection-like content (should be safely escaped)
        common_chat_msg expected_sql;
        expected_sql.role = "assistant";
        expected_sql.tool_calls = {
            { "sql_function", "{\"query\":\"SELECT * FROM users WHERE id = 1; DROP TABLE users; --\"}", "" }
        };

        test_parser_with_streaming(
            expected_sql,
                "<tool_call>\n"
                "  <function=sql_function>\n"
                "    <parameter=query>\n"
                "      SELECT * FROM users WHERE id = 1; DROP TABLE users; --\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // HTML/XML injection content
        common_chat_msg expected_html;
        expected_html.role = "assistant";
        expected_html.tool_calls = {
            { "html_function", "{\"content\":\"<script>alert('xss')</script><img src=x onerror=alert(1)>\"}", "" }
        };

        test_parser_with_streaming(
            expected_html,
                "<tool_call>\n"
                "  <function=html_function>\n"
                "    <parameter=content>\n"
                "      <script>alert('xss')</script><img src=x onerror=alert(1)>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Binary-like content (base64)
        common_chat_msg expected_binary;
        expected_binary.role = "assistant";
        expected_binary.tool_calls = {
            { "binary_function", "{\"data\":\"SGVsbG8gV29ybGQhIFRoaXMgaXMgYmFzZTY0IGVuY29kZWQgdGV4dC4=\"}", "" }
        };

        test_parser_with_streaming(
            expected_binary,
                "<tool_call>\n"
                "  <function=binary_function>\n"
                "    <parameter=data>\n"
                "      SGVsbG8gV29ybGQhIFRoaXMgaXMgYmFzZTY0IGVuY29kZWQgdGV4dC4=\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });

        // Very large numbers (should be parsed as scientific notation)
        common_chat_msg expected_large_numbers;
        expected_large_numbers.role = "assistant";
        expected_large_numbers.tool_calls = {
            { "number_function", "{\"big_int\":1e+60}", "" }  // Large number becomes scientific notation
        };

        test_parser_with_streaming(
            expected_large_numbers,
                "<tool_call>\n"
                "  <function=number_function>\n"
                "    <parameter=big_int>\n"
                "      999999999999999999999999999999999999999999999999999999999999\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return test_chat_parse(msg, /* is_partial= */ true, {COMMON_CHAT_FORMAT_QWEN3_CODER_XML}); });
    }

    {
        // Qwen3-Coder template
        auto tmpls = read_templates("models/templates/Qwen3-Coder.jinja");
        common_chat_templates_inputs inputs;
        inputs.messages = { message_user };

        common_chat_tool qwen_union_tool {
            /* .name = */ "qwen_union",
            /* .description = */ "Test tool for union/anyOf handling",
            /* .parameters = */ R"({
                "type": "object",
                "properties": {
                    "priority": { "type": ["number", "null"] },
                    "maybe_text": { "anyOf": [ { "type": "string" } ] },
                    "config": { "anyOf": [ { "type": "object" }, { "type": "null" } ] }
                },
                "required": []
            })",
        };
        inputs.tools = { qwen_union_tool };

        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        assert_equals(COMMON_CHAT_FORMAT_QWEN3_CODER_XML, params.format);
        assert_equals(false, params.grammar.empty());

        // Grammar should compile successfully
        auto grammar = build_grammar(params.grammar);
        GGML_ASSERT(grammar && "Failed to build Qwen3-Coder grammar with union types");
    }
}

static void test_template_output_peg_parsers() {
    printf("[%s]\n", __func__);

    // JSON schemas
    const char * invoice_schema = R"({
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "date": {"type": "string"}
        }
    })";

    {
        // Ministral-3-14B-Reasoning-2512
        auto tmpls = read_templates("models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja");

        // Test basic message
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "Hello, world!\nWhat's up?";
            t.expect = message_assist;
        });

        // Test basic message and reasoning with reasoning_format = none
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?";
            t.expect.content = "[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?";
        });

        // Test basic message and reasoning with reasoning_format = auto
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

            t.expect = message_assist_thoughts;
        });

        // Test tool call
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = R"([TOOL_CALLS]special_function[ARGS]{"arg1":1})";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {special_function_tool};

            t.expect = message_assist_call;
        });

        // Test tool call with reasoning
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "[THINK]I'm\nthinking[/THINK]"
                      R"([TOOL_CALLS]special_function[ARGS]{"arg1":1})";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {special_function_tool};

            t.expect = message_assist_call_thoughts;
        });

        // Test parallel tool calls
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = R"([TOOL_CALLS]special_function[ARGS]{"arg1": 1})"
                      R"([TOOL_CALLS]special_function_with_opt[ARGS]{"arg1": 1, "arg2": 2})";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.parallel_tool_calls = true;
            t.params.tools = {special_function_tool, special_function_tool_with_optional_param};

            t.expect.tool_calls = {{
                /* .name = */      "special_function",
                /* .arguments = */ R"({"arg1": 1})",
                /* .id = */        {},
            }, {
                /* .name = */      "special_function_with_opt",
                /* .arguments = */ R"({"arg1": 1, "arg2": 2})",
                /* .id = */        {},
            }};
        });

        // Test response format
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "[THINK]I need to output the invoice details in JSON[/THINK]"
                      "```json\n"
                      R"({"amount": 123.45, "date": "2025-12-03"})"
                      "\n```";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.json_schema = invoice_schema;

            t.expect.reasoning_content = "I need to output the invoice details in JSON";
            t.expect.content =R"({"amount": 123.45, "date": "2025-12-03"})";
        });
    }

    {
        // NVIDIA Nemotron-3 Nano
        auto tmpls = read_templates("models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja");

        // Test basic message
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "Hello, world!\nWhat's up?";
            t.expect = message_assist;
        });

        // Test basic message and reasoning with reasoning_format = none
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "I'm\nthinking\n</think>\nHello, world!\nWhat's up?";
            t.expect.content = "I'm\nthinking\n</think>\nHello, world!\nWhat's up?";
        });

        // Test basic message and reasoning with reasoning_format = auto
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "I'm\nthinking\n</think>\nHello, world!\nWhat's up?";
            t.params.enable_thinking = true;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

            t.expect = message_assist_thoughts;
        });

        // Test tool call
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {special_function_tool};

            t.expect = message_assist_call;
        });

        // Test tool call with reasoning
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input =
                "I'm\nthinking\n</think>\n"
                "<tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {special_function_tool};

            t.expect = message_assist_call_thoughts;
        });

        // Test parallel tool calls
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n"
                "<tool_call>\n"
                "<function=special_function_with_opt>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "<parameter=arg2>\n"
                "2\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.parallel_tool_calls = true;
            t.params.tools = {special_function_tool, special_function_tool_with_optional_param};

            t.expect.tool_calls = {{
                /* .name = */      "special_function",
                /* .arguments = */ R"({"arg1": 1})",
                /* .id = */        {},
            }, {
                /* .name = */      "special_function_with_opt",
                /* .arguments = */ R"({"arg1": 1, "arg2": 2})",
                /* .id = */        {},
            }};
        });

        // Test tool call with string parameter
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=python>\n"
                "<parameter=code>\n"
                "def hello():\n"
                "    print(\"Hello, world!\")\n"
                "\n"
                "hello()\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {python_tool};

            t.expect.tool_calls = {{
                /* .name = */      "python",
                /* .arguments = */ "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}",
                /* .id = */        {},
            }};
        });

        // Test tool call with string parameter and no closing </parameter> tag
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=python>\n"
                "<parameter=code>\n"
                "def hello():\n"
                "    print(\"Hello, world!\")\n"
                "\n"
                "hello()\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {python_tool};

            t.expect.tool_calls = {{
                /* .name = */      "python",
                /* .arguments = */ "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}",
                /* .id = */        {},
            }};
        });

        // Test response format
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input =
              "I need to output the invoice details in JSON\n"
              "</think>\n"
              R"({"amount": 123.45, "date": "2025-12-03"})";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.json_schema = invoice_schema;

            t.expect.reasoning_content = "I need to output the invoice details in JSON";
            t.expect.content = R"({"amount": 123.45, "date": "2025-12-03"})";
        });
    }

}

static void test_msg_diffs_compute() {
    printf("[%s]\n", __func__);
    {
        common_chat_msg msg1;

        common_chat_msg msg2;
        msg2.content = "Hello, world!";

        common_chat_msg_diff diff;
        diff.content_delta = "Hello, world!";

        assert_equals(
            {diff},
            common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg1;
        msg1.content = "Hello,";

        common_chat_msg msg2;
        msg2.content = "Hello, world!";

        common_chat_msg_diff diff;
        diff.content_delta = " world!";

        assert_equals(
            {diff},
            common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg0;

        common_chat_msg msg1;
        msg1.tool_calls = { { "special_function", "{\"ar", /* .id = */ "123" } };

        common_chat_msg msg2;
        msg2.tool_calls = { { "special_function", "{\"arg1\": 1}", /* .id = */ "123" } };

        common_chat_msg_diff diff01;
        diff01.tool_call_index = 0;
        diff01.tool_call_delta.name = "special_function";
        diff01.tool_call_delta.id = "123";
        diff01.tool_call_delta.arguments = "{\"ar";

        assert_equals(
            {diff01},
            common_chat_msg_diff::compute_diffs(msg0, msg1));

        common_chat_msg_diff diff12;
        diff12.tool_call_index = 0;
        // Note: neither id nor name change here.
        diff12.tool_call_delta.arguments = "g1\": 1}";

        assert_equals(
            {diff12},
            common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg0;

        common_chat_msg msg2;
        msg2.tool_calls = {
            { "f1", "{\"arg1\": 1}", /* .id = */ "123" },
            { "f2", "{\"arg2\": 2}", /* .id = */ "222" },
        };

        common_chat_msg_diff diff1;
        diff1.tool_call_index = 0;
        diff1.tool_call_delta.name = "f1";
        diff1.tool_call_delta.id = "123";
        diff1.tool_call_delta.arguments = "{\"arg1\": 1}";

        common_chat_msg_diff diff2;
        diff2.tool_call_index = 1;
        diff2.tool_call_delta.name = "f2";
        diff2.tool_call_delta.id = "222";
        diff2.tool_call_delta.arguments = "{\"arg2\": 2}";

        assert_equals(
            {diff1, diff2},
            common_chat_msg_diff::compute_diffs(msg0, msg2));
    }
}

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(999);

    // try {
#ifndef _WIN32
        if (argc > 1) {
            common_chat_templates_inputs inputs;
            common_chat_msg msg;
            msg.role = "user";
            msg.content = "Hey";
            inputs.messages = {msg};
            inputs.tools = { special_function_tool };

            std::cout << "| Template | Format |\n";
            std::cout << "|----------|--------|\n";

            for (int i = 1; i < argc; i++) {
                try {
                    std::string path = argv[i];
                    if (path.rfind(".jinja") != path.size() - 6) {
                        std::cerr << "Skipping non-jinja file: " << path << '\n';
                        continue;
                    }
                    auto tmpls = read_templates(path);
                    auto parts  = string_split(path, "/");
                    auto name   = parts[parts.size() - 1];
                    auto format = common_chat_format_name(common_chat_templates_apply(tmpls.get(), inputs).format);
                    std::cout << "| " << name << " | " << format << " |\n";
                } catch (const std::exception & e) {
                    std::cerr << "Failed to process " << argv[i] << ": " << e.what() << '\n';
                }
            }
        } else
#endif
        {
            test_msg_diffs_compute();
            test_msgs_oaicompat_json_conversion();
            test_tools_oaicompat_json_conversion();
            test_template_output_parsers();
            test_template_output_peg_parsers();
            std::cout << "\n[chat] All tests passed!" << '\n';
        }
        return 0;
    // } catch (const std::exception & e) {
    //     std::cerr << "Error: " << e.what() << '\n';
    //     return 1;
    // }
}
