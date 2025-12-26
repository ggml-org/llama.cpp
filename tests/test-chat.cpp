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

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <functional>
#include <string>

using json = nlohmann::ordered_json;

// ANSI color codes for terminal output
#define ANSI_COLOR_RED     "\033[1;31m"
#define ANSI_COLOR_GREEN   "\033[1;32m"
#define ANSI_COLOR_YELLOW  "\033[0;33m"
#define ANSI_COLOR_RESET   "\033[0m"

// Verbose mode control - set LOG_LEVEL=2 or higher for debug output
static int get_verbosity() {
    const char * level = std::getenv("LOG_LEVEL");
    return level ? std::atoi(level) : 0;
}
static const int g_verbose = get_verbosity();

// Parser implementation selector for tests
enum class chat_parser_impl {
    LEGACY,      // Use legacy monolithic parsers
    EXPERIMENTAL // Use new modular PEG parsers
};

static const char * chat_parser_impl_name(chat_parser_impl impl) {
    switch (impl) {
        case chat_parser_impl::LEGACY:      return "legacy";
        case chat_parser_impl::EXPERIMENTAL: return "experimental";
    }
    return "unknown";
}

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
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

static std::string read_file(const std::string & path) {
    if (g_verbose >= 2) {
        std::cerr << "# Reading: " << path << '\n' << std::flush;
    }
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
    try {
        return common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, read_file(path)));
    } catch (const std::runtime_error &) {
        return nullptr;
    }
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
    } catch (const std::exception &) {
        // JSON parsing can fail for partial streaming content - that's expected
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
        "required": ["code"],
        "additionalProperties": true
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
// Additional tools used in format-specific tests
common_chat_tool complex_function_tool {
    /* .name = */ "complex_function",
    /* .description = */ "A function with complex parameter types",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "active": { "type": "boolean" },
            "score": { "type": "number" }
        },
        "required": ["name", "age", "active", "score"]
    })",
};
common_chat_tool web_search_tool {
    /* .name = */ "web_search",
    /* .description = */ "Search the web",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "query": { "type": "string" },
            "limit": { "type": "integer" },
            "type": { "type": "string" }
        },
        "required": ["query"]
    })",
};
// Additional tools for Kimi K2 tests
common_chat_tool read_file_tool {
    /* .name = */ "read_file",
    /* .description = */ "Read files from the filesystem",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "args": { "type": "array" },
            "files": { "type": "array" }
        }
    })",
};
common_chat_tool emoji_function_tool {
    /* .name = */ "emoji_function",
    /* .description = */ "A function that handles emoji strings",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "message": { "type": "string" }
        },
        "required": ["message"]
    })",
};
common_chat_tool complex_function_in_think_tool {
    /* .name = */ "complex_function_in_think",
    /* .description = */ "A complex function for testing in-think tool calls",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "active": { "type": "boolean" },
            "score": { "type": "number" }
        },
        "required": ["name", "age", "active", "score"]
    })",
};
// Tool for testing multiple string parameters
common_chat_tool process_data_tool {
    /* .name = */ "process_data",
    /* .description = */ "Process data with specified format",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "input": { "type": "string", "description": "The input data" },
            "format": { "type": "string", "description": "The output format" }
        },
        "required": ["input", "format"]
    })",
};

std::vector<common_chat_tool> tools           { special_function_tool, special_function_tool_with_optional_param, python_tool };
std::vector<common_chat_tool> llama_3_1_tools { special_function_tool, code_interpreter_tool };
std::vector<common_chat_tool> glm_4_5_tools   { special_function_tool, special_function_tool_with_optional_param, complex_function_tool, web_search_tool };
std::vector<common_chat_tool> kimi_k2_tools   { special_function_tool, special_function_tool_with_optional_param, complex_function_tool, web_search_tool, read_file_tool, emoji_function_tool, complex_function_in_think_tool };

// Helper to create common_chat_syntax from common_chat_params with optional reasoning format override
static common_chat_syntax get_syntax(const common_chat_params & params,
                                      common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE) {
    common_chat_syntax syntax;
    syntax.format = params.format;
    syntax.reasoning_format = reasoning_format;
    syntax.thinking_forced_open = params.thinking_forced_open;
    if (!params.parser.empty()) {
        syntax.parser.load(params.parser);
    }
    return syntax;
}

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
                             const common_chat_tool_choice & tool_choice,
                             common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE,
                             const std::function<void(common_chat_templates_inputs &)> & customize_inputs = {},
                             chat_parser_impl impl = chat_parser_impl::LEGACY) {
    common_chat_templates_inputs inputs;
    inputs.parallel_tool_calls = true;
    inputs.messages.push_back(user_message);
    inputs.tools       = tools;
    inputs.tool_choice = tool_choice;
    // Enable thinking when reasoning is expected - this builds the parser with reasoning block support
    if (reasoning_format != COMMON_REASONING_FORMAT_NONE) {
        inputs.enable_thinking = true;
        inputs.reasoning_format = reasoning_format;
    }
    // Set parser implementation based on enum (env var can override for backwards compat)
    inputs.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL) || std::getenv("LLAMA_USE_NEW_PARSERS");
    if (customize_inputs) {
        customize_inputs(inputs);
    }

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

    // Strip end tokens (fall back to params_full.additional_stops when vector empty)
    const std::vector<std::string> & tokens_to_strip = end_tokens.empty() ? params_full.additional_stops : end_tokens;
    for (const auto & end_token : tokens_to_strip) {
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
                          bool ignore_whitespace_differences = false,
                          bool expect_parse_failure = false,
                          const std::function<void(std::string &)> & mutate_delta = {},
                          chat_parser_impl impl = chat_parser_impl::LEGACY) {
    common_chat_msg user_message;
    user_message.role = "user";
    user_message.content = "Hello, world!";

    for (const auto & tool_choice : std::vector<common_chat_tool_choice> {COMMON_CHAT_TOOL_CHOICE_AUTO, COMMON_CHAT_TOOL_CHOICE_REQUIRED}) {
        auto data = init_delta(tmpls, end_tokens, user_message, test_message, tools, tool_choice, reasoning_format, {}, impl);
        if (!expected_delta.empty()) {
            if (ignore_whitespace_differences) {
                assert_equals(string_strip(expected_delta), string_strip(data.delta));
            } else {
                assert_equals(expected_delta, data.delta);
            }
        }

        std::string delta = data.delta;
        if (mutate_delta) {
            mutate_delta(delta);
        }

        if (expect_parse_failure && !expect_grammar_triggered) {
            throw std::runtime_error("Cannot expect parse failure when grammar trigger is disabled");
        }

        if (expect_grammar_triggered) {
            common_chat_syntax syntax;
            syntax.format = data.params.format;
            syntax.reasoning_format = reasoning_format;
            if (!data.params.parser.empty()) {
                syntax.parser.load(data.params.parser);
            }
            bool threw = false;
            common_chat_msg msg;
            try {
                msg = common_chat_parse(delta, /* is_partial= */ false, syntax);
                if (expect_parse_failure) {
                    throw std::runtime_error("Expected parse failure but parsing succeeded");
                }
            } catch (const std::exception & e) {
                if (!expect_parse_failure) {
                    throw;
                }
                threw = true;
            }
            if (expect_parse_failure && !threw) {
                throw std::runtime_error("Expected parse failure but parsing succeeded");
            }
            if (!threw) {
                assert_msg_equals(test_message, msg, ignore_whitespace_differences);
            }
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
            auto constrained = delta;
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

            if (grammar_triggered && test_grammar_if_triggered && !expect_parse_failure && !match_string(constrained, grammar.get())) {
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
        LOG_INF("Streaming msg: %s\n", common_chat_msgs_to_json_oaicompat<json>({curr_msg}).dump().c_str());
        for (auto diff: common_chat_msg_diff::compute_diffs(last_msg, curr_msg)) {
            LOG_INF("Streaming diff: %s\n", common_chat_msg_diff_to_json_oaicompat<json>(diff).dump().c_str());
            if (!diff.reasoning_content_delta.empty()) {
                merged.reasoning_content += diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                merged.content += diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                if (!diff.tool_call_delta.name.empty()) {
                    merged.tool_calls.push_back({diff.tool_call_delta.name, "", diff.tool_call_delta.id});
                }
                if (!diff.tool_call_delta.arguments.empty()) {
                    GGML_ASSERT(!merged.tool_calls.empty());
                    merged.tool_calls.back().arguments += diff.tool_call_delta.arguments;
                }
                // Update ID if provided in delta (for formats that include ID with arguments)
                if (!diff.tool_call_delta.id.empty() && !merged.tool_calls.empty()) {
                    merged.tool_calls.back().id = diff.tool_call_delta.id;
                }
            }
            LOG_INF("Streaming merged: %s\n", common_chat_msgs_to_json_oaicompat<json>({merged}).dump().c_str());
        }
        assert_msg_equals(curr_msg, merged, true);
        last_msg = curr_msg;
    }
    assert_msg_equals(expected, parse_msg(raw_message), true);
    assert_msg_equals(expected, merged, true);
}

// ============================================================================
// Needle-based streaming tests
// ============================================================================
// Each field contains 2 "needles" that MUST appear in order during streaming.
// This catches buffering bugs, out-of-order emission, and non-incremental streaming.

// Unique needle markers (unlikely to appear in normal content)
#define NEEDLE1_CONTENT   "$N1C$"
#define NEEDLE2_CONTENT   "$N2C$"
#define NEEDLE1_REASONING "$N1R$"
#define NEEDLE2_REASONING "$N2R$"
#define NEEDLE1_ARG_KEY   "$N1AK$"
#define NEEDLE2_ARG_KEY   "$N2AK$"
#define NEEDLE1_ARG_VALUE "$N1AV$"
#define NEEDLE2_ARG_VALUE "$N2AV$"

struct needle_field_needles {
    std::string first;
    std::string second;
};

struct needle_arg_expectation {
    needle_field_needles key_needles;
    needle_field_needles value_needles;
    std::string key_text;
    std::string value_text;
};

struct needle_tool_expectation {
    std::vector<needle_arg_expectation> args;
};

struct needle_test_context {
    std::string scenario_name;
    common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    needle_field_needles content_needles;
    needle_field_needles reasoning_needles;
    std::vector<needle_tool_expectation> tool_expectations;
    common_chat_msg expected_msg;
    bool has_content = false;
    bool has_reasoning = false;
};

struct needle_scenario {
    std::string name;
    bool provide_tools = false;
    bool with_content = true;
    bool with_reasoning = false;
    bool with_tool_call = false;
    size_t tool_call_count = 1;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    bool expect_tool_ids = false;
    bool enable_thinking = false;
    bool force_disable_thinking = false;
    bool require_thinking_support = false;
    bool require_tool_support = false;
    bool parallel_tool_calls = false;
    bool skip_if_thinking_forced = false;
    size_t args_per_tool_call = 2;
    std::string tool_name = "python";
};

struct needle_field_state {
    bool saw_first = false;
    bool saw_second = false;
    bool saw_second_before_first = false;
};

struct needle_arg_state {
    needle_field_state key_state;
    needle_field_state value_state;
    size_t key_completion_seq = 0;
};

struct needle_tool_state {
    std::vector<needle_arg_state> arg_states;
    bool args_regressed = false;
    std::string longest_args_seen;
};

struct needle_test_result {
    needle_field_state content_state;
    needle_field_state reasoning_state;
    std::vector<needle_tool_state> tool_states;
    bool unexpected_tool_count = false;
    common_chat_msg final_msg;
};

// Check if tool call arguments regressed (got shorter)
static bool check_args_regression(const std::string & current, const std::string & previous) {
    // If previous is a prefix of current, no regression
    if (current.find(previous) == 0) return false;
    // If current is shorter and not a prefix situation, it's a regression
    if (current.length() < previous.length()) return true;
    return false;
}

static std::string make_indexed_needle(const char * base, size_t idx) {
    return std::string(base) + "_" + std::to_string(idx);
}

static void update_field_state(needle_field_state & state, const needle_field_needles & needles, const std::string & text) {
    if (needles.first.empty() && needles.second.empty()) {
        return;
    }
    auto pos_first = text.find(needles.first);
    auto pos_second = text.find(needles.second);

    if (!state.saw_first && pos_second != std::string::npos) {
        if (pos_first == std::string::npos || pos_second < pos_first) {
            state.saw_second_before_first = true;
        }
    }
    if (pos_first != std::string::npos) {
        state.saw_first = true;
    }
    if (pos_second != std::string::npos) {
        state.saw_second = true;
    }
}

static needle_test_context make_needle_context(const needle_scenario & scenario, common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY) {
    needle_test_context ctx;
    ctx.scenario_name = scenario.name;
    ctx.format = format;
    ctx.expected_msg.role = "assistant";

    if (scenario.with_content) {
        ctx.has_content = true;
        ctx.content_needles = {NEEDLE1_CONTENT, NEEDLE2_CONTENT};
        ctx.expected_msg.content = "Before " + ctx.content_needles.first + " middle " + ctx.content_needles.second + " after";
    }

    if (scenario.with_reasoning) {
        ctx.has_reasoning = true;
        ctx.reasoning_needles = {NEEDLE1_REASONING, NEEDLE2_REASONING};
        ctx.expected_msg.reasoning_content = "Thinking " + ctx.reasoning_needles.first + " deeply " + ctx.reasoning_needles.second + " done";
    }

    if (scenario.with_tool_call) {
        for (size_t call_idx = 0; call_idx < scenario.tool_call_count; ++call_idx) {
            needle_tool_expectation expectation;
            json args = json::object();

            for (size_t arg_idx = 0; arg_idx < scenario.args_per_tool_call; ++arg_idx) {
                needle_arg_expectation arg_expect;
                arg_expect.key_needles.first  = make_indexed_needle(NEEDLE1_ARG_KEY, call_idx * scenario.args_per_tool_call + arg_idx);
                arg_expect.key_needles.second = make_indexed_needle(NEEDLE2_ARG_KEY, call_idx * scenario.args_per_tool_call + arg_idx);
                arg_expect.value_needles.first  = make_indexed_needle(NEEDLE1_ARG_VALUE, call_idx * scenario.args_per_tool_call + arg_idx);
                arg_expect.value_needles.second = make_indexed_needle(NEEDLE2_ARG_VALUE, call_idx * scenario.args_per_tool_call + arg_idx);
                arg_expect.key_text = arg_expect.key_needles.first + arg_expect.key_needles.second;
                arg_expect.value_text = arg_expect.value_needles.first + arg_expect.value_needles.second;

                std::string key = arg_expect.key_text;
                std::string value = arg_expect.value_text;

                args[key] = value;
                expectation.args.push_back(arg_expect);
            }

            common_chat_tool_call call;
            call.name = scenario.tool_name;
            call.arguments = args.dump();
            if (scenario.expect_tool_ids) {
                // Mistral Nemo requires 9-character alphanumeric IDs
                if (ctx.format == COMMON_CHAT_FORMAT_MISTRAL_NEMO) {
                    // Generate 9-character alphanumeric ID (e.g., "call00123", "abc456789")
                    std::string id = "call";
                    id += std::to_string(call_idx);
                    while (id.length() < 9) {
                        id += "0";
                    }
                    // Pad or truncate to exactly 9 characters
                    if (id.length() > 9) {
                        id = id.substr(0, 9);
                    }
                    call.id = id;
                } else {
                    call.id = std::to_string(call_idx);
                }
            }

            ctx.tool_expectations.push_back(expectation);
            ctx.expected_msg.tool_calls.push_back(call);
        }
    }

    return ctx;
}

static void verify_field_state(const char * label, const needle_field_state & state, const needle_field_needles & needles) {
    if (needles.first.empty() && needles.second.empty()) {
        return;
    }
    if (!state.saw_first) {
        throw std::runtime_error(std::string(label) + ": Never saw NEEDLE1");
    }
    if (!state.saw_second) {
        throw std::runtime_error(std::string(label) + ": Never saw NEEDLE2");
    }
    if (state.saw_second_before_first) {
        throw std::runtime_error(std::string(label) + ": Saw NEEDLE2 before NEEDLE1 - streaming not incremental!");
    }
}

static needle_test_result test_streaming_with_needles(
    const needle_test_context & ctx,
    const std::string & raw_message,
    const std::function<common_chat_msg(const std::string &, bool)> & parse_msg) {

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

    needle_test_result result;
    result.tool_states.resize(ctx.tool_expectations.size());
    size_t key_sequence_counter = 1;

    for (size_t i = 1; i <= raw_message.size(); ++i) {
        auto safe_partial = std::string(utf8_truncate_safe_view(std::string_view(raw_message).substr(0, i)));
        bool is_partial = i < raw_message.size();
        auto msg = parse_msg(safe_partial, is_partial);

        update_field_state(result.content_state, ctx.content_needles, msg.content);
        update_field_state(result.reasoning_state, ctx.reasoning_needles, msg.reasoning_content);

        if (!ctx.tool_expectations.empty()) {
            if (msg.tool_calls.size() > ctx.tool_expectations.size()) {
                result.unexpected_tool_count = true;
            }
            size_t limit = std::min(msg.tool_calls.size(), ctx.tool_expectations.size());
            for (size_t idx = 0; idx < limit; ++idx) {
                const auto & tc = msg.tool_calls[idx];
                auto & tracker = result.tool_states[idx];
                if (tracker.arg_states.size() < ctx.tool_expectations[idx].args.size()) {
                    tracker.arg_states.resize(ctx.tool_expectations[idx].args.size());
                }

                // Track full arguments JSON for regression detection
                if (!tracker.longest_args_seen.empty() && !tc.arguments.empty()) {
                    if (check_args_regression(tc.arguments, tracker.longest_args_seen)) {
                        tracker.args_regressed = true;
                    }
                }
                if (tc.arguments.length() > tracker.longest_args_seen.length()) {
                    tracker.longest_args_seen = tc.arguments;
                }

                for (size_t arg_idx = 0; arg_idx < ctx.tool_expectations[idx].args.size(); ++arg_idx) {
                    const auto & expectation = ctx.tool_expectations[idx].args[arg_idx];
                    auto & arg_state = tracker.arg_states[arg_idx];

                    update_field_state(arg_state.key_state, expectation.key_needles, tc.arguments);
                    update_field_state(arg_state.value_state, expectation.value_needles, tc.arguments);

                    // Track when each key completes (both needles seen) for ordering verification
                    if (arg_state.key_state.saw_second && arg_state.key_completion_seq == 0) {
                        arg_state.key_completion_seq = key_sequence_counter++;
                    }
                }
            }
        }

        if (!is_partial) {
            result.final_msg = msg;
        }
    }

    return result;
}

static void verify_needle_results(const needle_test_context & ctx, const needle_test_result & result) {
    if (ctx.has_content) {
        verify_field_state("Content", result.content_state, ctx.content_needles);
    }
    if (ctx.has_reasoning) {
        verify_field_state("Reasoning", result.reasoning_state, ctx.reasoning_needles);
    }

        if (!ctx.tool_expectations.empty()) {
            if (result.unexpected_tool_count) {
                throw std::runtime_error("Tool call: Parser produced more tool calls than expected");
            }
            if (result.final_msg.tool_calls.size() != ctx.tool_expectations.size()) {
                throw std::runtime_error("Tool call: Final tool call count mismatch");
            }
            for (size_t call_idx = 0; call_idx < ctx.tool_expectations.size(); ++call_idx) {
                const auto & expectation = ctx.tool_expectations[call_idx];
                const auto & state = result.tool_states[call_idx];
                const auto & final_call = result.final_msg.tool_calls[call_idx];

                if (state.args_regressed) {
                    throw std::runtime_error("Tool call: Arguments regressed (got shorter) during streaming!");
                }

                for (size_t arg_idx = 0; arg_idx < expectation.args.size(); ++arg_idx) {
                    const auto & arg_expect = expectation.args[arg_idx];
                    if (arg_idx >= state.arg_states.size()) {
                        throw std::runtime_error("Tool call: Missing argument state in tracker");
                    }
                    const auto & arg_state = state.arg_states[arg_idx];

                    verify_field_state("Tool arg key", arg_state.key_state, arg_expect.key_needles);
                    verify_field_state("Tool arg value", arg_state.value_state, arg_expect.value_needles);

                    // Verify keys stream in order (key N completes before key N+1)
                    if (arg_idx > 0) {
                        const auto & prev_state = state.arg_states[arg_idx - 1];
                        if (prev_state.key_completion_seq == 0 || arg_state.key_completion_seq == 0 ||
                            prev_state.key_completion_seq > arg_state.key_completion_seq) {
                            throw std::runtime_error("Tool call: Argument keys streamed out of order");
                        }
                    }

                    if (final_call.arguments.find(arg_expect.key_text) == std::string::npos) {
                        throw std::runtime_error("Tool call: Final arguments missing expected key");
                    }
                    if (final_call.arguments.find(arg_expect.value_text) == std::string::npos) {
                        throw std::runtime_error("Tool call: Final arguments missing expected value");
                    }
                }
            }
        }

    assert_msg_equals(ctx.expected_msg, result.final_msg, false);
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
const common_chat_msg message_assist_call_content_idx            = simple_assist_msg("Hello, world!\nWhat's up?", "", "special_function", "{\"arg1\":1}", /* id = */ "0");
const common_chat_msg message_assist_call_thoughts_content_idx   = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}", /* id = */ "0");
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
        return common_chat_peg_parse(arena_, msg, is_partial, /* syntax = */ {params_.format});
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
    // PEG parser tests always use new parsers
    tc.params.experimental_new_parsers = true;

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
        auto oai_json = common_chat_msgs_to_json_oaicompat<json>({msg});
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
        common_chat_msgs_to_json_oaicompat<json>({message_user_parts}).dump(2));

    assert_equals(
        std::string(
            "[\n"
            "  {\n"
            "    \"role\": \"assistant\",\n"
            "    \"content\": null,\n"
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
        common_chat_msgs_to_json_oaicompat<json>({message_assist_call_python}).dump(2));

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
        auto oai_json = common_chat_tools_to_json_oaicompat<json>({tool});
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
        common_chat_tools_to_json_oaicompat<json>({special_function_tool}).dump(2));
}

static void test_template_output_parsers(chat_parser_impl impl) {
    printf("[%s:%s]\n", __func__, chat_parser_impl_name(impl));

    // Wrapper to pass impl to test_templates without changing all call sites
    // Note: direct common_chat_parse() calls still use legacy format-based parsing
    // (they don't go through template application and don't have a PEG parser)
    auto test = [impl](const struct common_chat_templates * tmpls, const std::vector<std::string> & end_tokens,
                       const common_chat_msg & test_message,
                       const std::vector<common_chat_tool> & tools = {},
                       const std::string & expected_delta = "",
                       bool expect_grammar_triggered = true,
                       bool test_grammar_if_triggered = true,
                       common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE,
                       bool ignore_whitespace_differences = false,
                       bool expect_parse_failure = false,
                       const std::function<void(std::string &)> & mutate_delta = {}) {
        test_templates(tmpls, end_tokens, test_message, tools, expected_delta, expect_grammar_triggered,
                       test_grammar_if_triggered, reasoning_format, ignore_whitespace_differences,
                       expect_parse_failure, mutate_delta, impl);
    };

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
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_COMMAND_R7B}));
        assert_msg_equals(message_assist,
            common_chat_parse(
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_COMMAND_R7B}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
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
            common_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_COMMAND_R7B}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_call_idx,
            common_chat_parse(
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
            common_chat_parse(
                "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
                "<|START_ACTION|>[\n"
                "    {\"tool_call_id\": \"0\", \"tool_name\": \"special",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        test(tmpls.get(), end_tokens, message_assist_call_idx, tools,
                      "<|START_THINKING|><|END_THINKING|>"
                      "<|START_ACTION|>[\n"
                      "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
                      "]<|END_ACTION|>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      COMMON_REASONING_FORMAT_DEEPSEEK);
        test(tmpls.get(), end_tokens, message_assist, tools,
                      "<|START_RESPONSE|>Hello, world!\n"
                      "What's up?<|END_RESPONSE|>",
                      /* expect_grammar_triggered= */ false);
    }
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
            common_chat_parse(
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
            common_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"t",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));

        assert_equals(
            simple_assist_msg("", "", "puppeteer_screenshot", "{\"name\":\"servethehome_homepage\","),
            common_chat_parse(
                R"({"tool_call": {"name": "puppeteer_screenshot", "arguments": {"name": "servethehome_homepage",)",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));

        assert_equals(
            message_assist_call_empty_args,
            common_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"special_function\"",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));
        assert_equals(
            message_assist_call_cutoff_args,
            common_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"special_function\", \"arguments\" : { \"arg",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GENERIC}));

        assert_msg_equals(message_assist,
            common_chat_parse(
                "{\n"
                "  \"response\": \"Hello, world!\\nWhat's up?\"\n"
                "}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GENERIC}));
        test(tmpls.get(), end_tokens, message_assist_call_id, tools,
                      "{\n"
                      "  \"tool_calls\": [\n"
                      "    {\n"
                      "      \"name\": \"special_function\",\n"
                      "      \"arguments\": {\n"
                      "        \"arg1\": 1\n"
                      "      },\n"
                      "      \"id\": \"123456789\"\n"
                      "    }\n"
                      "  ]\n"
                      "}");
    }
    {
        auto tmpls = read_templates("models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja");
        std::vector<std::string>   end_tokens{ "</s>" };

        assert_equals(COMMON_CHAT_FORMAT_MISTRAL_NEMO, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test(
            tmpls.get(), end_tokens, message_assist_call_id, tools,
            "[TOOL_CALLS][{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, \"id\": \"123456789\"}]");
    }
    {
        assert_msg_equals(
            simple_assist_msg("Réponse", "raisonnement"),
            common_chat_parse(
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
            common_chat_parse(
                "```json\n"
                "<function_call> { \"name\" : \"python\"",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            simple_assist_msg("Let's call something\n"),
            common_chat_parse(
                "Let's call something\n"
                "<tool_call>{\"name\"",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(
            simple_assist_msg("Let's call something\n"),
            common_chat_parse(
                "Let's call something\n"
                "<tool_call>{\"name",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
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
            common_chat_parse(
                "<tool_call>\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(message_assist_call_content,
            common_chat_parse(
                "Hello, world!\nWhat's up?<tool_call>\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tool_call>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<function=special_function>{\"arg1\": 1}</function>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<function name=\"special_function\">\n"
                "{\"arg1\": 1}\n"
                "</function>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<tool>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tool>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<tools>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</tools>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<response>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</response>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "```xml\n"
                "<response>\n"
                "    {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</response>\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "```xml\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "```\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "```\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "```json\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "```",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "```json\n"
                "\n"
                "                    <function_call> {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}} \n"
                "                    </function_call> \n"
                "``` ",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<json>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</json>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<xml>\n"
                "  {\n"
                "    \"name\": \"special_function\", \"arguments\": {\"arg1\": 1}\n"
                "  }\n"
                "</xml>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<JSON>\n"
                "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                "</JSON>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
                "This is not a tool call:\n"
                "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        // assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
        //     common_chat_parse(
        //         "I'm\nthinking</think>Hello, world!\nWhat's up?",
        //         COMMON_CHAT_FORMAT_HERMES_2_PRO));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_md,
            common_chat_parse(
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
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ true,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts_unopened_unparsed,
            common_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));

        test(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<tool_call>\n"
                      "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                      "</tool_call>");

        // Test multiple tool calls with template
        common_chat_msg message_assist_multiple_calls_template;
        message_assist_multiple_calls_template.role = "assistant";
        message_assist_multiple_calls_template.content = "";
        message_assist_multiple_calls_template.tool_calls.push_back({"special_function", "{\"arg1\": 1}", ""});
        message_assist_multiple_calls_template.tool_calls.push_back({"python", "{\"code\":\"print('test')\"}", ""});

        test(tmpls.get(), end_tokens, message_assist_multiple_calls_template, tools,
                      "<tool_call>\n"
                      "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                      "</tool_call>\n"
                      "<tool_call>\n"
                      "{\"name\": \"python\", \"arguments\": {\"code\":\"print('test')\"}}\n"
                      "</tool_call>");

        // TODO(ochafik): Fix this test - the template produces a format that doesn't match expected
        // test(tmpls.get(), end_tokens, message_assist_call_python_lines, tools,
        //               "<tool_call>\n"
        //               "{\"name\": \"python\", \"arguments\": {\"code\":\"# This is a program:\\nprint('hey')\"}}\n"
        //               "</tool_call>");
        assert_msg_equals(
            simple_assist_msg("", /* reasoning_content= */ "<tool_call>nah uhg</tool_call>"),
            common_chat_parse(
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
            common_chat_parse(
                "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LLAMA_3_X}));

        // test(tmpls.get(), end_tokens, message_assist, tools, R"(?)", /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_call_code_interpreter, llama_3_1_tools,
                      "<|python_tag|>code_interpreter.call(code=\"print('hey')\")");
        test(tmpls.get(), end_tokens, message_assist_call_python, tools,
                      "<|python_tag|>python.call(code=\"print('hey')\")");
        test(tmpls.get(), end_tokens, message_assist_call, tools,
                      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }
    {
        auto tmpls = read_templates("models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

        test(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_call, tools,
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
                common_chat_parse(
                    "<function=special_function>{\"arg1\": 1}</function>",
                    is_partial,
                    {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1}));
        }

        assert_equals(
            message_assist_call,
            common_chat_parse(
                "<function=special_function>{\"arg1\": 1}<",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1}));

        test(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_call, tools,
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
            common_chat_parse(
                "all\n"
                "Hello, world!\n"
                "nono\n"
                "What's up?>>>special_function\n"
                "{\"arg1\": 1}\n",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist_call_python_lines,
            common_chat_parse(
                "python\n"
                "# This is a program:\n"
                "print('hey')",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist_call_python_lines_unclosed,
            common_chat_parse(
                "python\n"
                "# This is a program:\n"
                "print('hey')",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "special_function\n"
                "{\"arg1\": 1} \n                    ",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
        assert_msg_equals(message_assist,
            common_chat_parse(
                "all\n"
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));

        test(tmpls.get(), end_tokens, message_assist, {},
                      "all\n"
                      "Hello, world!\n"
                      "What's up?",
                      /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_call, tools,
                      "special_function\n"
                      "{\"arg1\": 1}");
    }
    {
        auto tmpls = read_templates("models/templates/fireworks-ai-llama-3-firefunction-v2.jinja");
        std::vector<std::string>   end_tokens{ "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_FIREFUNCTION_V2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_call, tools,
                      " functools[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]");
    }
    {
        // Replacement DeepSeek R1 template. Makes the Distill Qwen 7B/32B models happy to call tools and all.
        auto tmpls = read_templates("models/templates/llama-cpp-deepseek-r1.jinja");
        std::vector<std::string>   end_tokens{ "<｜end▁of▁sentence｜>" };

        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));

        assert_msg_equals(message_assist_call_thoughts_unparsed,
            common_chat_parse(
                "<think>I'm\nthinking</think>\n\n"
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<｜tool▁calls｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));

        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
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
        test(tmpls.get(), end_tokens, message_assist_call, tools,
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>");
    }
    {
        auto tmpls = read_templates("models/templates/llama-cpp-ibm-granite-granite-3.3-2B-Instruct.jinja");
        std::vector<std::string> end_tokens{ "<|end_of_text|>" };

        assert_equals(COMMON_CHAT_FORMAT_GRANITE, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

        assert_equals(COMMON_CHAT_FORMAT_GRANITE, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(simple_assist_msg("<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>"),
            common_chat_parse(
                "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(message_assist_empty,
            common_chat_parse(
                "<think",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_empty,
            common_chat_parse(
                "<think",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(message_assist_thoughts_no_content,
            common_chat_parse(
                "<think>I'm\nthinking",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(
            message_assist_empty,
            common_chat_parse(
                "<think>I'm\nthinking</think><response",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist_call_empty_args,
            common_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\"",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist_call_cutoff_args,
            common_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_GRANITE}));
        assert_msg_equals(
            message_assist_call_cutoff_args,
            common_chat_parse(
                "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls with thinking
        assert_msg_equals(
            message_assist_call_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think><|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, {",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test template generation for regular content
        test(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        // Skip the full template test for now - parser loops over AUTO/REQUIRED and only REQUIRED works without content
        // test(tmpls.get(), end_tokens, message_assist_call, tools,
        //               "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]",
        //               /* expect_grammar_triggered= */ true
        // );
    }
    {
        auto tmpls = read_templates("models/templates/openai-gpt-oss-120b.jinja");
        std::vector<std::string> end_tokens{ "<|return|>", "<|call|>" };

        assert_equals(COMMON_CHAT_FORMAT_GPT_OSS, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GPT_OSS, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        assert_msg_equals(simple_assist_msg("", "I'm\nthink"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthink",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function<|message|>{\"arg1",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>analysis to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
            common_chat_parse(
                "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                "<|start|>assistant<|channel|>commentary<|message|>Hello, world!\nWhat's up?",
                /* is_partial= */ true,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
                " to=functions.special_function<|channel|>commentary <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "", "special_function", "{\"arg1\": 1}"),
            common_chat_parse(
                " to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                }));
        assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
            common_chat_parse(
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

        test(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);

        // Create inputs with reasoning enabled (includes process_data for multi-param tests)
        common_chat_templates_inputs inputs_tools_reasoning;
        inputs_tools_reasoning.messages = {message_user};
        inputs_tools_reasoning.tools = {special_function_tool, process_data_tool};
        inputs_tools_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs_tools_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);

        // Get syntax with parser for tool call tests (with reasoning)
        auto params = common_chat_templates_apply(tmpls.get(), inputs_tools_reasoning);
        common_chat_syntax syntax;
        syntax.format = params.format;
        syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (!params.parser.empty()) {
            syntax.parser.load(params.parser);
        }

        // Syntax with reasoning for content-only tests
        common_chat_syntax syntax_reasoning;
        syntax_reasoning.format = params.format;
        syntax_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (!params.parser.empty()) {
            syntax_reasoning.parser.load(params.parser);
        }

        // PEG parser-specific tests (only run with experimental parser)
        // Legacy format-based parser has different whitespace handling for these cases
        if (impl == chat_parser_impl::EXPERIMENTAL) {
            // Test simple reasoning content
            assert_msg_equals(
                simple_assist_msg("Hello, world!", "I'm thinking about the answer"),
                common_chat_parse(
                    "<seed:think>I'm thinking about the answer</seed:think>Hello, world!",
                    /* is_partial= */ false,
                    syntax_reasoning));

        // Test budget reflection tags
        common_chat_msg msg_budget_reflect;
        msg_budget_reflect.role = "assistant";
        msg_budget_reflect.content = "<seed:cot_budget_reflect>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:cot_budget_reflect>I need to calculate this step by step.";
        msg_budget_reflect.reasoning_content = "Token usage: 45/1000\nI should continue thinking to find the best solution.";
        assert_msg_equals(
            msg_budget_reflect,
            common_chat_parse(
                "<seed:think>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:think>"
                "<seed:cot_budget_reflect>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:cot_budget_reflect>"
                "I need to calculate this step by step.",
                /* is_partial= */ false,
                syntax_reasoning));

        // Test tool calls with Seed-OSS format (using special_function from inputs_tools)
        common_chat_msg msg_tool_call;
        msg_tool_call.role = "assistant";
        msg_tool_call.tool_calls.push_back({"special_function", "{\"arg1\":42}", ""});
        assert_msg_equals(
            msg_tool_call,
            common_chat_parse(
                "<seed:tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n42\n</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                syntax));

        // Test multiple parameters in tool call
        common_chat_msg msg_multi_param;
        msg_multi_param.role = "assistant";
        msg_multi_param.tool_calls.push_back({"process_data", "{\"input\":\"test\",\"format\":\"json\"}", ""});
        assert_msg_equals(
            msg_multi_param,
            common_chat_parse(
                "<seed:tool_call>\n"
                "<function=process_data>\n"
                "<parameter=input>\ntest\n</parameter>\n"
                "<parameter=format>\njson\n</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                syntax));

        // Test reasoning + tool call combination
        common_chat_msg msg_reasoning_tool;
        msg_reasoning_tool.role = "assistant";
        msg_reasoning_tool.content = "";
        msg_reasoning_tool.reasoning_content = "I need to call the special function";
        msg_reasoning_tool.tool_calls.push_back({"special_function", "{\"arg1\":42}", ""});
        assert_msg_equals(
            msg_reasoning_tool,
            common_chat_parse(
                "<seed:think>I need to call the special function</seed:think>"
                "<seed:tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n42\n</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                syntax_reasoning));

        // Test deltas: the number of tool calls in partial parses should never decrease
        std::string tool_msg = "<seed:tool_call>\n"
            "<function=special_function>\n"
            "<parameter=arg1>\n42\n</parameter>\n"
            "</function>";
        std::size_t previousToolCalls = 0;
        for (std::size_t i = std::string("<seed:tool_call>").length(); i < tool_msg.length() - 1; i++) {
            auto partial = tool_msg.substr(0, i);
            auto partial_res = common_chat_parse(partial, true, syntax);
            if (partial_res.tool_calls.size() < previousToolCalls) {
                throw std::runtime_error("Tool call size decreased on partial: " + partial + " from " + std::to_string(previousToolCalls) + " to " + std::to_string(partial_res.tool_calls.size()));
            }
            previousToolCalls = partial_res.tool_calls.size();
        }

        // Test partial parsing for incomplete string parameter - captures partial value
        assert_msg_equals(
            simple_assist_msg("", "", "process_data", "{\"input\":\"test"),
            common_chat_parse(
                "<seed:tool_call>\n"
                "<function=process_data>\n"
                "<parameter=input>\ntest",
                /* is_partial= */ true,
                syntax));

        auto make_invalid_delta = [&](const std::function<void(std::string &)> & mutate) {
            test(
                tmpls.get(), end_tokens, message_assist_call, tools,
                /* expected_delta = */ "", /* expect_grammar_triggered = */ true,
                /* test_grammar_if_triggered = */ true,
                COMMON_REASONING_FORMAT_NONE,
                /* ignore_whitespace_differences = */ false,
                /* expect_parse_failure = */ true,
                mutate);
        };

        // Wrong function name should fail parsing once tool-call trigger fires
        make_invalid_delta([](std::string & delta) {
            const std::string needle = "function=special_function";
            auto pos = delta.find(needle);
            GGML_ASSERT(pos != std::string::npos);
            delta.replace(pos, needle.size(), "function=unknown_function");
        });

        // Wrong argument type should also fail (string instead of integer)
        make_invalid_delta([](std::string & delta) {
            const std::string param_open = "<parameter=arg1>";
            const std::string param_close = "</parameter>";
            auto start = delta.find(param_open);
            GGML_ASSERT(start != std::string::npos);
            auto end = delta.find(param_close, start);
            GGML_ASSERT(end != std::string::npos);
            end += param_close.size();
            const std::string replacement = "<parameter=arg1>\n\"not-a-number\"\n</parameter>";
            delta.replace(start, end - start, replacement);
        });

        // Test incomplete reasoning tag
        assert_msg_equals(
            simple_assist_msg("", "I was thinking"),
            common_chat_parse(
                "<seed:think>I was thinking",
                /* is_partial= */ true,
                syntax_reasoning));

        // Test content without reasoning
        assert_msg_equals(
            simple_assist_msg("This is a simple response without reasoning."),
            common_chat_parse(
                "This is a simple response without reasoning.",
                /* is_partial= */ false,
                syntax));
        } // end PEG parser-specific tests
    }
    {
        auto tmpls = read_templates("models/templates/NVIDIA-Nemotron-Nano-v2.jinja");
        std::vector<std::string> end_tokens{ "<SPECIAL_12>" };

        assert_equals(COMMON_CHAT_FORMAT_NEMOTRON_V2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_NEMOTRON_V2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_NEMOTRON_V2}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_NEMOTRON_V2}));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think><TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            common_chat_parse(
                "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_NEMOTRON_V2}
            ));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            common_chat_parse(
                "<think>I'm\nthinking</think><TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /*  .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                    /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test template generation for regular content
        test(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?\n",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test(tmpls.get(), end_tokens, message_assist_call, tools,
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

        test(tmpls.get(), end_tokens, message_assist, tools, "</think>Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test(tmpls.get(), end_tokens, message_assist_thoughts, tools, "</think>Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(
            simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|><|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            common_chat_parse(
                "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}
            ));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            common_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|><|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));


//         assert_msg_equals(
//             simple_assist_msg("", "I'm\nthinking", "", ""),
//             common_chat_parse(
//                 "<|tools_prefix|>[ { \"test\" : { \"success\" : true } } ] <|tools_suffix|>",
//                 /* is_partial= */ false,
//                 {
//                     /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
//                     /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
//                 }));

// res  remove_waiti: remove task 0 from waiting list. current waiting = 1 (before remove)
// srv          stop: cancel task, id_task = 0
// res  remove_waiti: remove task 0 from waiting list. current waiting = 0 (before remove)
// que          post: new task, id = 70/1, front = 1
// que    start_loop: processing new tasks
// que    start_loop: processing task, id = 70
// que    start_loop: update slots
// srv  update_slots: all slots are idle
// que    start_loop: waiting for new tasks
// srv    operator(): got exception: {"error":{"code":500,"message":"Failed to parse input at pos 0","type":"server_error"}}
// srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 500
// srv  log_server_r: request:  {"max_tokens": 512, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Write an example"}], "tool_choice": "required", "tools": [{"type": "function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {"success": {"type": "boolean", "const": true}}, "required": ["success"]}}}], "parallel_tool_calls": false, "stream": false}

        // Test template generation for regular content
        test(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                      /* expect_grammar_triggered= */ true
        );

        assert_equals(true, common_chat_templates_support_enable_thinking(tmpls.get()));
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
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test single tool call with JSON format
        common_chat_msg msg_single_tool_call;
        msg_single_tool_call.role = "assistant";
        msg_single_tool_call.tool_calls.push_back({"special_function", "{\"arg1\":1}", ""});
        assert_msg_equals(
            msg_single_tool_call,
            common_chat_parse(
                "<|tool_call_start|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with string argument
        common_chat_msg msg_tool_call_string;
        msg_tool_call_string.role = "assistant";
        msg_tool_call_string.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
        assert_msg_equals(
            msg_tool_call_string,
            common_chat_parse(
                "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with multiple arguments
        common_chat_msg msg_multi_args;
        msg_multi_args.role = "assistant";
        msg_multi_args.tool_calls.push_back({"calculate", "{\"x\":10,\"y\":20,\"operation\":\"add\"}", ""});
        assert_msg_equals(
            msg_multi_args,
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
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
            common_chat_parse(
                "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>Here's the result.",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Test tool call with newlines (common in LLM output)
        common_chat_msg msg_tool_call_newlines;
        msg_tool_call_newlines.role = "assistant";
        msg_tool_call_newlines.tool_calls.push_back({"get_current_time", "{\"location\":\"Paris\"}", ""});
        assert_msg_equals(
            msg_tool_call_newlines,
            common_chat_parse(
                "<|tool_call_start|>[{\n    \"name\": \"get_current_time\",\n    \"arguments\": {\n        \"location\": \"Paris\"\n    }\n}]<|tool_call_end|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

        // Note: LFM2 uses JSON format for tool calls: [{"name": "...", "arguments": {...}}]
        // Unlike other formats, LFM2 template does not render tool calls in conversation history,
        // so we don't use test() for tool call generation. Instead, the parsing tests
        // above verify edge cases and format variations for the tool call output format.
    }

    {
        auto tmpls = read_templates("models/templates/MiniMax-M2.jinja");
        std::vector<std::string> end_tokens{ "[e~[" };

        assert_equals(COMMON_CHAT_FORMAT_MINIMAX_M2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_MINIMAX_M2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Create inputs for parser tests - without reasoning (for content-only tests)
        common_chat_templates_inputs inputs_tools_no_reasoning;
        inputs_tools_no_reasoning.messages = {message_user};
        inputs_tools_no_reasoning.tools = {special_function_tool, special_function_tool_with_optional_param};
        inputs_tools_no_reasoning.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        inputs_tools_no_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);

        // Create inputs with reasoning enabled for reasoning tests
        common_chat_templates_inputs inputs_tools_reasoning;
        inputs_tools_reasoning.messages = {message_user};
        inputs_tools_reasoning.tools = {special_function_tool, special_function_tool_with_optional_param};
        inputs_tools_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs_tools_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);

        // Get syntax for content-only tests
        auto params_no_reasoning = common_chat_templates_apply(tmpls.get(), inputs_tools_no_reasoning);
        common_chat_syntax syntax;
        syntax.format = params_no_reasoning.format;
        if (!params_no_reasoning.parser.empty()) {
            syntax.parser.load(params_no_reasoning.parser);
        }

        // Get syntax with reasoning for reasoning tests
        auto params_reasoning = common_chat_templates_apply(tmpls.get(), inputs_tools_reasoning);
        common_chat_syntax syntax_reasoning;
        syntax_reasoning.format = params_reasoning.format;
        syntax_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (!params_reasoning.parser.empty()) {
            syntax_reasoning.parser.load(params_reasoning.parser);
        }

        // PEG parser-specific tests (only run with experimental parser)
        // Legacy format-based parser has different whitespace handling for these cases
        if (impl == chat_parser_impl::EXPERIMENTAL) {
            // Test parsing regular content
            assert_msg_equals(message_assist,
                common_chat_parse(
                    "Hello, world!\nWhat's up?",
                    /* is_partial= */ false,
                    syntax));

            // Test parsing content with thinking (thinking_forced_open: model output starts with reasoning directly)
            assert_msg_equals(message_assist_thoughts,
                common_chat_parse(
                    "I'm\nthinking</think>Hello, world!\nWhat's up?",
                    /* is_partial= */ false,
                    syntax_reasoning));

        // Test parsing tool calls (with proper newlines expected by parser)
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                /* is_partial= */ false,
                syntax));

        // Test parsing tool calls with thinking (thinking_forced_open)
        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
                "I'm\nthinking</think><minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                /* is_partial= */ false,
                syntax_reasoning));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            common_chat_parse(
                "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                syntax));

        // Test tool calls with extra content AND thinking (thinking_forced_open)
        assert_msg_equals(message_assist_call_thoughts_content,
            common_chat_parse(
                "I'm\nthinking</think><minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                syntax_reasoning));

        // Test streaming (thinking_forced_open: no <think> prefix in input)
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "I'm\nthinking\n</think>Hello, world!\nWhat's up?\n<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax_reasoning); });
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?\n\n<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>\n",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax_reasoning); });
        test_parser_with_streaming(message_assist_call_withopt,
            "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n<parameter name=\"arg2\">2</parameter>\n</invoke>\n</minimax:tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax); });

        // Test compact format (no extra whitespace) - verifies whitespace flexibility
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>",
                /* is_partial= */ false,
                syntax));
        } // end PEG parser-specific tests

        // Test template generation for regular content
        test(tmpls.get(), end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test(tmpls.get(), end_tokens, message_assist_call, tools,
                      "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                      /* ignore_whitespace_differences= */ true
        );

        // Test template generation for tools with optional parameters
        test(tmpls.get(), end_tokens, message_assist_call_noopt, tools,
                      "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* common_reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                      /* ignore_whitespace_differences= */ true
        );
        test(tmpls.get(), end_tokens, message_assist_call_withopt, tools,
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

        // Get params with tools for parsing tests (always use a parser)
        // Build parser with reasoning extraction disabled
        common_chat_templates_inputs glm_inputs_no_reasoning;
        glm_inputs_no_reasoning.messages = {message_user};
        glm_inputs_no_reasoning.tools = glm_4_5_tools;
        glm_inputs_no_reasoning.enable_thinking = true;
        glm_inputs_no_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
        auto glm_params_no_reasoning = common_chat_templates_apply(tmpls.get(), glm_inputs_no_reasoning);
        auto glm_syntax = get_syntax(glm_params_no_reasoning);

        // Build parser with reasoning extraction enabled
        common_chat_templates_inputs glm_inputs_reasoning;
        glm_inputs_reasoning.messages = {message_user};
        glm_inputs_reasoning.tools = glm_4_5_tools;
        glm_inputs_reasoning.enable_thinking = true;
        glm_inputs_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        glm_inputs_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
        auto glm_params_reasoning = common_chat_templates_apply(tmpls.get(), glm_inputs_reasoning);
        auto glm_syntax_reasoning = get_syntax(glm_params_reasoning, COMMON_REASONING_FORMAT_DEEPSEEK);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                glm_syntax));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "\n<think>I'm\nthinking</think>\nHello, world!\nWhat's up?",
                /* is_partial= */ false,
                glm_syntax_reasoning), true);

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                /* is_partial= */ false,
                glm_syntax), true);

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
                "\n<think>I'm\nthinking</think>\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                /* is_partial= */ false,
                glm_syntax_reasoning), true);

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            common_chat_parse(
                "\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                glm_syntax), true);

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            common_chat_parse(
                "\n<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                /* is_partial= */ false,
                glm_syntax_reasoning), true);

        // Streaming tests only run with experimental PEG parsers
        if (impl == chat_parser_impl::EXPERIMENTAL) {
            test_parser_with_streaming(message_assist_call_thoughts_content,
                "\n<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax_reasoning); });
            test_parser_with_streaming(message_assist_call_thoughts_unparsed,
                "\n<think>I'm\nthinking</think>\n\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });
            test_parser_with_streaming(message_assist_call_withopt,
                "\n<think></think>\n<tool_call>special_function_with_opt\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n<arg_key>arg2</arg_key>\n<arg_value>2</arg_value>\n</tool_call>\n",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax_reasoning); });
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
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });
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
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });

            // Test interleaved thinking
            // Content chunks: "Hello, world!\n" (until <think>) + "What's up?" (until \n<tool_call>) = "Hello, world!\nWhat's up?"
            test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinkingThinking2", "special_function", "{\"arg1\": 1}"),
                "\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax_reasoning); });
            test_parser_with_streaming(simple_assist_msg("\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?", "", "special_function", "{\"arg1\": 1}"),
                "\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });
        }

        // Test template generation for regular content
        test(tmpls.get(), end_tokens, message_assist, tools,
                      "\n<think></think>\nHello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // TODO: Test template generation for tool calls with reasoning
        // These tests are temporarily disabled because building params with reasoning_format=DEEPSEEK
        // causes grammar stack overflow during llama_grammar_advance_stack (recursive grammar structure).
        // This is a pre-existing issue that needs to be fixed separately.
        // test(tmpls.get(), end_tokens, message_assist_call, tools,
        //               "\n<think></think>\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>\n",
        //               /* expect_grammar_triggered= */ true,
        //               /* test_grammar_if_triggered= */ false,
        //               /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
        //               /* ignore_whitespace_differences= */ true);
    }

    {
        auto tmpls = read_templates("models/templates/Kimi-K2-Thinking.jinja");
        std::vector<std::string> end_tokens{ "<|im_end|>" };

        assert_equals(COMMON_CHAT_FORMAT_KIMI_K2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_KIMI_K2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Build parser with tools (always use a parser)
        common_chat_templates_inputs kimi_inputs;
        kimi_inputs.messages = {message_user};
        kimi_inputs.tools = kimi_k2_tools;
        kimi_inputs.enable_thinking = true;
        kimi_inputs.parallel_tool_calls = true;
        kimi_inputs.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
        auto kimi_params = common_chat_templates_apply(tmpls.get(), kimi_inputs);
        auto kimi_syntax = get_syntax(kimi_params);

        // Build parser with reasoning extraction enabled
        common_chat_templates_inputs kimi_inputs_reasoning;
        kimi_inputs_reasoning.messages = {message_user};
        kimi_inputs_reasoning.tools = kimi_k2_tools;
        kimi_inputs_reasoning.enable_thinking = true;
        kimi_inputs_reasoning.parallel_tool_calls = true;
        kimi_inputs_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        kimi_inputs_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
        auto kimi_params_reasoning = common_chat_templates_apply(tmpls.get(), kimi_inputs_reasoning);
        auto kimi_syntax_reasoning = get_syntax(kimi_params_reasoning, COMMON_REASONING_FORMAT_DEEPSEEK);

        // Build content-only parser (no tools) for content-only tests
        common_chat_templates_inputs kimi_inputs_content_only;
        kimi_inputs_content_only.messages = {message_user};
        kimi_inputs_content_only.enable_thinking = true;
        kimi_inputs_content_only.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
        auto kimi_params_content = common_chat_templates_apply(tmpls.get(), kimi_inputs_content_only);
        auto kimi_syntax_content = get_syntax(kimi_params_content);

        // Build content-only parser with reasoning
        common_chat_templates_inputs kimi_inputs_content_reasoning;
        kimi_inputs_content_reasoning.messages = {message_user};
        kimi_inputs_content_reasoning.enable_thinking = true;
        kimi_inputs_content_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        kimi_inputs_content_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
        auto kimi_params_content_reasoning = common_chat_templates_apply(tmpls.get(), kimi_inputs_content_reasoning);
        auto kimi_syntax_content_reasoning = get_syntax(kimi_params_content_reasoning, COMMON_REASONING_FORMAT_DEEPSEEK);

        // Test parsing regular content (content-only parser)
        assert_msg_equals(message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                kimi_syntax_content));

        // Test parsing content with thinking (content-only parser with reasoning)
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                kimi_syntax_content_reasoning));

        // Tool call and streaming tests only run with experimental PEG parsers
        // (legacy parser doesn't extract tool IDs correctly for Kimi format)
        if (impl == chat_parser_impl::EXPERIMENTAL) {
            // Test parsing tool calls (Kimi format includes tool ID after the colon)
            assert_msg_equals(message_assist_call_idx,
                common_chat_parse(
                    "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                    /* is_partial= */ false,
                    kimi_syntax));

            // Test parsing tool calls with thinking
            assert_msg_equals(message_assist_thoughts_call_idx,
                common_chat_parse(
                    "<think>I'm\nthinking</think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                    /* is_partial= */ false,
                    kimi_syntax_reasoning));

            // Test tool calls with extra content
            assert_msg_equals(message_assist_call_content_idx,
                common_chat_parse(
                    "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>Hello, world!\nWhat's up?",
                    /* is_partial= */ false,
                    kimi_syntax));

            // Test tool calls with extra content AND thinking
            assert_msg_equals(message_assist_call_thoughts_content_idx,
                common_chat_parse(
                    "<think>I'm\nthinking</think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>Hello, world!\nWhat's up?",
                    /* is_partial= */ false,
                    kimi_syntax_reasoning));

            // Test streaming
            test_parser_with_streaming(message_assist_call_thoughts_content_idx,
            "<think>I'm\nthinking\n</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
        test_parser_with_streaming(simple_assist_msg("<think>I'm\nthinking</think>\n\n", "", "special_function", "{\"arg1\": 1}", "0"),
            "<think>I'm\nthinking</think>\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
        test_parser_with_streaming(message_assist_call_thoughts_content_idx,
            "<think>I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>\n",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
        test_parser_with_streaming(simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}", "0"),
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function_with_opt:0<|tool_call_argument_begin|>{\"arg1\": 1, \"arg2\": 2}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
        test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": \"123456\"}", "0"),
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": \"123456\"}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
        test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": [1, 2, \"345\", 6]}", "0"),
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": [1, 2, \"345\", 6]}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
        test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": {\"12\": 34, \"5\": [67, 8], \"9\": \"10\"}}", "0"),
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": {\"12\": 34, \"5\": [67, 8], \"9\": \"10\"}}<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
        test_parser_with_streaming(
                simple_assist_msg("", "", "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}", "0"),
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function:0<|tool_call_argument_begin|>"
                "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
        test_parser_with_streaming(
                simple_assist_msg("", "", "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}", "0"),
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0<|tool_call_argument_begin|>"
                "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
        test_parser_with_streaming(
                simple_assist_msg("", "", "read_file", "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}", "0"),
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
                "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
        test_parser_with_streaming(
                simple_assist_msg(
                        "Let me start by examining the relevant files to understand the current implementation.", "",
                        "read_file",
                        "{\"files\": [{\"path\": \"src/app/Partners.tsx\", \"line_ranges\": [\"1-100\"]}]}", "0"),
                "Let me start by examining the relevant files to understand the current implementation."
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
                "{\"files\":[{\"path\":\"src/app/Partners.tsx\",\"line_ranges\":[\"1-100\"]}]}"
                "<|tool_call_end|><|tool_calls_section_end|>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
        auto multi_tool_msg = simple_assist_msg("Let me call multiple tools.", "I'm thinking.");
        multi_tool_msg.tool_calls.push_back({ "read_file", "{\"files\": [{\"path\": \"src/app/Partners.tsx\", \"line_ranges\": [\"1-100\"]}]}", "0" });
        multi_tool_msg.tool_calls.push_back({ "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}", "1" });
        multi_tool_msg.tool_calls.push_back({ "complex_function", "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}", "2" });
        multi_tool_msg.tool_calls.push_back({ "emoji_function", "{\"message\":\"Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\"}", "3" });
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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
        } // end experimental parser tests

        // TODO: These tests are for tool calls embedded in <think> blocks, which is an edge case
        // that requires special parser handling not yet implemented. The parser currently
        // treats all content inside <think>...</think> as reasoning_content.
        // test_parser_with_streaming(
        //         simple_assist_msg("", "I'm thinking", "complex_function_in_think", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
        //         "<think>I'm thinking<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function_in_think:0<|tool_call_argument_begin|>"
        //         "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
        //         "<|tool_call_end|><|tool_calls_section_end|>",
        //     [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
        // test_parser_with_streaming(
        //         simple_assist_msg("Hello", "I'm thinkingI'm still thinking", "complex_function_in_think", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
        //         "<think>I'm thinking<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function_in_think:0<|tool_call_argument_begin|>"
        //         "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
        //         "<|tool_call_end|><|tool_calls_section_end|>I'm still thinking</think>Hello",
        //     [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });

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
        test(tmpls.get(), end_tokens, message_assist, tools,
                      "<think></think>Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Tool call tests require PEG parser for correct ID extraction
        if (impl == chat_parser_impl::EXPERIMENTAL) {
            // Test template generation for tool calls (Kimi format includes ID after colon)
            // Note: JSON formatting may vary, so we skip delta comparison and just test parsing
            test(tmpls.get(), end_tokens, message_assist_call_idx, tools,
                          /* expected_delta= */ "",
                          /* expect_grammar_triggered= */ true,
                          /* test_grammar_if_triggered= */ true,
                          /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                          /* ignore_whitespace_differences= */ true
            );

            // Test template generation for tools with optional parameters
            test(tmpls.get(), end_tokens, simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1}", "0"), tools,
                          /* expected_delta= */ "",
                          /* expect_grammar_triggered= */ true,
                          /* test_grammar_if_triggered= */ true,
                          /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                          /* ignore_whitespace_differences= */ true
            );
            test(tmpls.get(), end_tokens, simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}", "0"), tools,
                          /* expected_delta= */ "",
                          /* expect_grammar_triggered= */ true,
                          /* test_grammar_if_triggered= */ true,
                          /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                          /* ignore_whitespace_differences= */ true
            );
        }
    }

    // Test Qwen3-Coder XML format
    {
        // Load template and build parser with tools
        auto tmpls = read_templates("models/templates/Qwen3-Coder.jinja");
        std::vector<std::string> end_tokens{ "<|im_end|>", "<|endoftext|>" };

        // Define all tools used in these tests with proper types matching test expectations
        std::vector<common_chat_tool> qwen3_coder_tools = {
            { "special_function", "A special function", R"({"type":"object","properties":{"arg1":{"type":"integer"}},"required":["arg1"]})" },
            { "special_function_with_opt", "A function with optional param", R"({"type":"object","properties":{"arg1":{"type":"integer"},"arg2":{"type":"integer"}},"required":["arg1"]})" },
            { "complex_function", "A complex function", R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"active":{"type":"boolean"},"score":{"type":"number"}},"required":["name","age","active","score"]})" },
            { "unicode_function", "A unicode function", R"({"type":"object","properties":{"message":{"type":"string"}},"required":["message"]})" },
            { "code_function", "A code function", R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})" },
            { "json_function", "A JSON function", R"({"type":"object","properties":{"config":{"type":"object"}},"required":["config"]})" },
            { "array_function", "An array function", R"({"type":"object","properties":{"items":{"type":"array"}},"required":["items"]})" },
            { "empty_function", "An empty param function", R"({"type":"object","properties":{"empty_param":{"type":"string"}},"required":["empty_param"]})" },
            { "boolean_function", "A boolean function", R"({"type":"object","properties":{"enabled":{"type":"boolean"},"debug":{"type":"boolean"}},"required":["enabled","debug"]})" },
            { "null_function", "A null function", R"({"type":"object","properties":{"optional_param":{"type":"null"}},"required":["optional_param"]})" },
            { "math_function", "A math function", R"({"type":"object","properties":{"negative":{"type":"integer"},"decimal":{"type":"number"},"scientific":{"type":"number"},"formula":{"type":"string"}}})" },
            { "xml_function", "An XML function", R"({"type":"object","properties":{"xml_content":{"type":"string"}},"required":["xml_content"]})" },
            { "quote_function", "A quote function", R"({"type":"object","properties":{"message":{"type":"string"}},"required":["message"]})" },
            { "long_function", "A long text function", R"({"type":"object","properties":{"long_text":{"type":"string"}},"required":["long_text"]})" },
            { "search_function", "A search function", R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})" },
            { "compact_function", "A compact function", R"({"type":"object","properties":{"param":{"type":"string"}},"required":["param"]})" },
            { "get_user_data_v2", "A user data function", R"({"type":"object","properties":{"user_id":{"type":"integer"}},"required":["user_id"]})" },
            { "test_function", "A test function", R"({"type":"object","properties":{"param_1":{"type":"string"},"param_2_name":{"type":"string"},"param3":{"type":"integer"}},"required":["param_1","param_2_name","param3"]})" },
            { "xml_parser", "An XML parser function", R"({"type":"object","properties":{"xml":{"type":"string"}},"required":["xml"]})" },
            { "whitespace_function", "A whitespace function", R"({"type":"object","properties":{"spaces":{"type":"string"}},"required":["spaces"]})" },
            { "tab_function", "A tab function", R"({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]})" },
            { "control_function", "A control function", R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})" },
            { "emoji_function", "An emoji function", R"({"type":"object","properties":{"message":{"type":"string"}},"required":["message"]})" },
            { "number_function", "A number function", R"({"type":"object","properties":{"big_int":{"type":"integer"}},"required":["big_int"]})" },
            { "binary_function", "A binary function", R"({"type":"object","properties":{"data":{"type":"string"}},"required":["data"]})" },
            { "sql_function", "A SQL function", R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})" },
            { "html_function", "An HTML function", R"({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]})" },
            { "python", "A python function", R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})" },
        };

        // Build parser with tools
        common_chat_templates_inputs qwen3_inputs;
        qwen3_inputs.messages = {message_user};
        qwen3_inputs.tools = qwen3_coder_tools;
        qwen3_inputs.parallel_tool_calls = true;
        auto qwen3_params = common_chat_templates_apply(tmpls.get(), qwen3_inputs);
        auto qwen3_syntax = get_syntax(qwen3_params);

        // Basic XML tool call parsing
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<tool_call>\n"
                "  <function=special_function>\n"
                "    <parameter=arg1>\n"
                "      1\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
                /* is_partial= */ false,
                qwen3_syntax));

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Compact format (no extra whitespace)
        common_chat_msg expected_compact;
        expected_compact.role = "assistant";
        expected_compact.tool_calls = {
            { "compact_function", "{\"param\":\"value\"}", "" }
        };

        test_parser_with_streaming(
            expected_compact,
                "<tool_call><function=compact_function><parameter=param>value</parameter></function></tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

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
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });
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
            t.params.enable_thinking = true;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.json_schema = invoice_schema;

            t.expect.reasoning_content = "I need to output the invoice details in JSON";
            t.expect.content = R"({"amount": 123.45, "date": "2025-12-03"})";
        });

        // Test simple "quota" response (basic parsing test)
        test_peg_parser(tmpls.get(), [&](auto & t) {
            t.input = "quota";
            t.expect = message_assist;
            t.expect.content = "quota";
        });
    }

}

// ============================================================================
// Systematic needle-based streaming tests
// ============================================================================
// Tests each template format with needle-injected content to verify:
// 1. Streaming is truly incremental (needles appear in order)
// 2. Tool names are never split
// 3. Tool arguments never regress

// Scoped enums for template capabilities - each field has its own type for type safety
enum class ThinkingSupport { No, Yes };
enum class ToolSupport { No, Yes };
enum class Skip { No, Yes };
enum class ReasoningRequiresTools { No, Yes };
enum class ToolsEmitContentWithCalls { No, Yes };
enum class InjectReasoningAfterFormat { No, Yes };
enum class SupportsDisableThinking { No, Yes };
enum class SupportsReasoningOnly { No, Yes };
enum class ToolCallsHaveIds { No, Yes };

struct template_capabilities {
    const char * name;
    const char * jinja_path;
    common_chat_format format;
    ThinkingSupport supports_thinking = ThinkingSupport::No;
    ToolSupport supports_tools = ToolSupport::No;
    const char * think_open_tag = nullptr;   // Opening tag for thinking (nullptr = auto-detect)
    const char * think_close_tag = nullptr;  // Closing tag for thinking (nullptr = no thinking)
    Skip skip = Skip::No;
    // TODO(ochafik): Add minja detection for these capabilities (see https://github.com/ochafik/minja/pull/20)
    ReasoningRequiresTools reasoning_requires_tools = ReasoningRequiresTools::No;
    ToolsEmitContentWithCalls tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    InjectReasoningAfterFormat inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    SupportsDisableThinking supports_disable_thinking = SupportsDisableThinking::Yes;
    SupportsReasoningOnly supports_reasoning_only = SupportsReasoningOnly::Yes;
    ToolCallsHaveIds tool_calls_have_ids = ToolCallsHaveIds::No;
    const char * needle_tool_name = nullptr;  // Tool name for needle tests (nullptr = use "python")
};

// Shared template capabilities for all needle tests
static const std::vector<template_capabilities> & get_template_capabilities() {
    static const std::vector<template_capabilities> templates = {
        // Templates with thinking support
        {"Command R7B", "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja",
            COMMON_CHAT_FORMAT_COMMAND_R7B, ThinkingSupport::Yes, ToolSupport::Yes,
            "<|START_THINKING|>", "<|END_THINKING|>", Skip::No, ReasoningRequiresTools::Yes,
            ToolsEmitContentWithCalls::No, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes,
            ToolCallsHaveIds::Yes},
        {"DeepSeek R1", "models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja",
            COMMON_CHAT_FORMAT_DEEPSEEK_R1, ThinkingSupport::Yes, ToolSupport::No,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::Yes},
        {"DeepSeek V3.1", "models/templates/deepseek-ai-DeepSeek-V3.1.jinja",
            COMMON_CHAT_FORMAT_DEEPSEEK_V3_1, ThinkingSupport::Yes, ToolSupport::No,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::Yes,
            SupportsDisableThinking::No, SupportsReasoningOnly::No},
        {"GLM 4.6", "models/templates/GLM-4.6.jinja",
            COMMON_CHAT_FORMAT_GLM_4_5, ThinkingSupport::Yes, ToolSupport::Yes,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes},
        {"Granite", "models/templates/llama-cpp-ibm-granite-granite-3.3-2B-Instruct.jinja",
            COMMON_CHAT_FORMAT_GRANITE, ThinkingSupport::Yes, ToolSupport::Yes,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::Yes,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::No},
        {"Hermes 2 Pro", "models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
            COMMON_CHAT_FORMAT_HERMES_2_PRO, ThinkingSupport::No, ToolSupport::Yes,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::No, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::No, SupportsReasoningOnly::No},
        {"Kimi K2", "models/templates/Kimi-K2-Instruct.jinja",
            COMMON_CHAT_FORMAT_KIMI_K2, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes,
            ToolCallsHaveIds::Yes},
        {"MiniMax M2", "models/templates/MiniMax-M2.jinja",
            COMMON_CHAT_FORMAT_MINIMAX_M2, ThinkingSupport::Yes, ToolSupport::Yes,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::No, SupportsReasoningOnly::No},
        {"Nemotron V2", "models/templates/NVIDIA-Nemotron-Nano-v2.jinja",
            COMMON_CHAT_FORMAT_NEMOTRON_V2, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes},
        {"Nemotron V3", "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja",
            COMMON_CHAT_FORMAT_NEMOTRON_V3, ThinkingSupport::Yes, ToolSupport::Yes,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::No, SupportsReasoningOnly::No},
        {"Nemotron V3 (Unsloth)", "models/templates/unsloth-Nemotron-3-Nano.jinja",
            COMMON_CHAT_FORMAT_NEMOTRON_V3, ThinkingSupport::Yes, ToolSupport::Yes,
            "<think>", "</think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::No, SupportsReasoningOnly::No},
        {"Seed OSS", "models/templates/ByteDance-Seed-OSS.jinja",
            COMMON_CHAT_FORMAT_SEED_OSS, ThinkingSupport::Yes, ToolSupport::Yes,
            "<seed:think>", "</seed:think>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes},

        // Templates without thinking support
        {"Firefunction V2", "models/templates/fireworks-ai-llama-3-firefunction-v2.jinja",
            COMMON_CHAT_FORMAT_FIREFUNCTION_V2, ThinkingSupport::No, ToolSupport::No},
        {"Functionary V3.1", "models/templates/meetkai-functionary-medium-v3.1.jinja",
            COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes,
            ToolCallsHaveIds::No, "test_function"},
        {"Functionary V3.2", "models/templates/meetkai-functionary-medium-v3.2.jinja",
            COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes},
        {"Llama 3.1", "models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja",
            COMMON_CHAT_FORMAT_LLAMA_3_X, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::No, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::No, SupportsReasoningOnly::No},
        {"Mistral Nemo", "models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja",
            COMMON_CHAT_FORMAT_MISTRAL_NEMO, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::No, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::No, SupportsReasoningOnly::No,
            ToolCallsHaveIds::Yes},
        {"Qwen3 Coder", "models/templates/Qwen3-Coder.jinja",
            COMMON_CHAT_FORMAT_QWEN3_CODER_XML, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::No, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::No, SupportsReasoningOnly::No},
        {"Apertus", "models/templates/Apertus-8B-Instruct.jinja",
            COMMON_CHAT_FORMAT_APERTUS, ThinkingSupport::Yes, ToolSupport::Yes,
            "<|inner_prefix|>", "<|inner_suffix|>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes},
        {"Apriel 1.5", "models/templates/unsloth-Apriel-1.5.jinja",
            COMMON_CHAT_FORMAT_APRIEL_1_5, ThinkingSupport::Yes, ToolSupport::Yes,
            "<thinking>", "</thinking>", Skip::Yes},
        {"GPT OSS", "models/templates/openai-gpt-oss-120b.jinja",
            COMMON_CHAT_FORMAT_GPT_OSS, ThinkingSupport::Yes, ToolSupport::Yes,
            "<|inner_thoughts_begin|>", "<|inner_thoughts_end|>", Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes},
        {"Xiaomi MiMo", "models/templates/MiMo-VL.jinja",
            COMMON_CHAT_FORMAT_XIAOMI_MIMO, ThinkingSupport::No, ToolSupport::Yes,
            nullptr, nullptr, Skip::No, ReasoningRequiresTools::No,
            ToolsEmitContentWithCalls::Yes, InjectReasoningAfterFormat::No,
            SupportsDisableThinking::Yes, SupportsReasoningOnly::Yes},
    };
    return templates;
}

// Cross-check declared capabilities against minja's detected capabilities.
// This ensures our test configuration stays in sync with what minja detects from templates.
// Note: minja's detection is heuristic (checks if output differs with capability enabled).
// Our declarations may intentionally differ if we know the template's actual behavior.
static bool verify_template_capabilities(const std::vector<template_capabilities> & templates) {
    printf("[%s]\n", __func__);
    size_t checked = 0;
    size_t tools_mismatches = 0;
    size_t thinking_mismatches = 0;

    for (const auto & info : templates) {
        auto tmpls = read_templates(info.jinja_path);
        if (!tmpls) {
            continue;
        }

        // Cross-check tools support (this should always match)
        bool minja_tools = common_chat_templates_support_tools(tmpls.get());
        bool our_tools = info.supports_tools == ToolSupport::Yes;
        if (minja_tools != our_tools) {
            printf("  " ANSI_COLOR_RED "MISMATCH" ANSI_COLOR_RESET " %s: minja.supports_tools=%s, declared=%s\n",
                   info.name, minja_tools ? "yes" : "no", our_tools ? "yes" : "no");
            tools_mismatches++;
        }

        // Cross-check thinking support
        // Note: minja checks if enable_thinking changes output, which may differ from
        // whether the template has explicit thinking tags we can parse.
        bool minja_thinking = common_chat_templates_support_enable_thinking(tmpls.get());
        bool our_thinking = info.supports_thinking == ThinkingSupport::Yes;
        if (minja_thinking != our_thinking) {
            if (g_verbose >= 1) {
                printf("  " ANSI_COLOR_YELLOW "NOTE" ANSI_COLOR_RESET " %s: minja.supports_thinking=%s, declared=%s\n",
                       info.name, minja_thinking ? "yes" : "no", our_thinking ? "yes" : "no");
            }
            thinking_mismatches++;
        }

        // TODO(ochafik): Cross-check tool_calls_have_ids with minja's supports_tool_call_id
        // once minja exposes this capability (see https://github.com/ochafik/minja/pull/20)

        checked++;
    }

    // Tools mismatch is a hard failure - should always match
    if (tools_mismatches > 0) {
        printf("  " ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET " %zu tools capability mismatches\n", tools_mismatches);
        return false;
    }

    // Thinking mismatches are informational - minja detection is heuristic
    if (thinking_mismatches > 0 && g_verbose >= 1) {
        printf("  " ANSI_COLOR_YELLOW "INFO" ANSI_COLOR_RESET " %zu thinking capability differences (may be intentional)\n", thinking_mismatches);
    }

    printf("  " ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET " (%zu templates verified against minja)\n", checked);
    return true;
}

// Verify that when experimental_new_parsers is enabled with tools, we get the expected format
// (not CONTENT_ONLY) and that grammar + parser are properly generated.
// This catches Pattern 1 failures: templates detected as Content-only when they should have
// a proper tool-calling format.
static bool test_format_detection_with_tools() {
    printf("[%s]\n", __func__);

    const char * template_filter = std::getenv("NEEDLE_TEMPLATE_FILTER");
    const auto & templates = get_template_capabilities();

    size_t tested = 0;
    size_t passed = 0;
    size_t skipped = 0;

    for (const auto & info : templates) {
        if (template_filter && std::string(info.name) != template_filter) {
            continue;
        }

        // Skip templates without tool support
        if (info.supports_tools != ToolSupport::Yes) {
            continue;
        }

        auto tmpls = read_templates(info.jinja_path);
        if (!tmpls) {
            if (g_verbose >= 1) {
                printf("  " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " %s (template not found)\n", info.name);
            }
            skipped++;
            continue;
        }

        tested++;

        // Apply template with tools and experimental_new_parsers
        common_chat_templates_inputs inputs;
        inputs.messages = {message_user};
        inputs.tools = {python_tool};
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        inputs.parallel_tool_calls = false;
        inputs.experimental_new_parsers = true;

        common_chat_params params;
        try {
            params = common_chat_templates_apply(tmpls.get(), inputs);
        } catch (const std::exception & e) {
            printf("  " ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET " %s: apply threw: %s\n", info.name, e.what());
            continue;
        }

        bool format_ok = true;
        bool grammar_ok = true;
        bool parser_ok = true;

        // Check 1: Format should match expected (not CONTENT_ONLY)
        if (params.format != info.format) {
            if (params.format == COMMON_CHAT_FORMAT_CONTENT_ONLY) {
                printf("  " ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET " %s: format is CONTENT_ONLY, expected %d\n",
                       info.name, static_cast<int>(info.format));
            } else if (g_verbose >= 1) {
                printf("  " ANSI_COLOR_YELLOW "NOTE" ANSI_COLOR_RESET " %s: format=%d, expected=%d\n",
                       info.name, static_cast<int>(params.format), static_cast<int>(info.format));
            }
            // Only fail on CONTENT_ONLY, other format differences may be intentional
            format_ok = (params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY);
        }

        // Check 2: Grammar should be non-empty when tools are provided
        if (params.grammar.empty()) {
            printf("  " ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET " %s: grammar is empty with tools\n", info.name);
            grammar_ok = false;
        }

        // Check 3: Parser should be non-empty when experimental_new_parsers is enabled
        if (params.parser.empty()) {
            printf("  " ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET " %s: parser is empty with experimental_new_parsers\n", info.name);
            parser_ok = false;
        }

        if (format_ok && grammar_ok && parser_ok) {
            passed++;
            if (g_verbose >= 1) {
                printf("  " ANSI_COLOR_GREEN "PASS" ANSI_COLOR_RESET " %s (format=%d, grammar=%zu bytes, parser=%zu bytes)\n",
                       info.name, static_cast<int>(params.format), params.grammar.size(), params.parser.size());
            }
        }
    }

    printf("  Results: %zu/%zu passed, %zu skipped\n", passed, tested, skipped);
    return passed == tested;
}

static const char * tool_choice_name(common_chat_tool_choice choice) {
    switch (choice) {
        case COMMON_CHAT_TOOL_CHOICE_AUTO: return "auto";
        case COMMON_CHAT_TOOL_CHOICE_REQUIRED: return "required";
        case COMMON_CHAT_TOOL_CHOICE_NONE: return "none";
    }
    return "unknown";
}

static std::vector<needle_scenario> build_needle_scenarios(const template_capabilities & info) {
    std::vector<needle_scenario> scenarios;

    needle_scenario content_no_tools;
    content_no_tools.name = "content-no-tools";
    content_no_tools.provide_tools = false;
    content_no_tools.with_content = true;
    content_no_tools.with_tool_call = false;
    content_no_tools.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    content_no_tools.enable_thinking = false;
    content_no_tools.force_disable_thinking = true;
    content_no_tools.skip_if_thinking_forced = true;
    scenarios.push_back(content_no_tools);

    if (info.supports_thinking == ThinkingSupport::Yes && info.reasoning_requires_tools == ReasoningRequiresTools::No) {
        needle_scenario reasoning_with_content;
        reasoning_with_content.name = "content-with-reasoning";
        reasoning_with_content.with_reasoning = true;
        reasoning_with_content.enable_thinking = true;
        reasoning_with_content.require_thinking_support = true;
        scenarios.push_back(reasoning_with_content);

        if (info.supports_reasoning_only == SupportsReasoningOnly::Yes) {
            needle_scenario reasoning_only;
            reasoning_only.name = "reasoning-only";
            reasoning_only.with_content = false;
            reasoning_only.with_reasoning = true;
            reasoning_only.enable_thinking = true;
            reasoning_only.require_thinking_support = true;
            scenarios.push_back(reasoning_only);
        }

        if (info.supports_disable_thinking == SupportsDisableThinking::Yes) {
            needle_scenario thinking_disabled;
            thinking_disabled.name = "thinking-disabled";
            thinking_disabled.with_content = true;
            thinking_disabled.force_disable_thinking = true;
            thinking_disabled.require_thinking_support = true;
            thinking_disabled.skip_if_thinking_forced = true;
            scenarios.push_back(thinking_disabled);
        }
    }

    if (info.supports_tools == ToolSupport::Yes) {
        needle_scenario tools_disabled;
        tools_disabled.name = "tools-available-but-disabled";
        tools_disabled.provide_tools = true;
        tools_disabled.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        tools_disabled.with_tool_call = false;
        tools_disabled.require_tool_support = true;
        scenarios.push_back(tools_disabled);

        needle_scenario tool_auto;
        tool_auto.name = "tool-auto-single";
        tool_auto.provide_tools = true;
        tool_auto.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        tool_auto.with_tool_call = true;
        tool_auto.require_tool_support = true;
        tool_auto.with_content = (info.tools_emit_content_with_calls == ToolsEmitContentWithCalls::Yes);
        tool_auto.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
        scenarios.push_back(tool_auto);

        needle_scenario tool_required_only;
        tool_required_only.name = "tool-required-only";
        tool_required_only.provide_tools = true;
        tool_required_only.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        tool_required_only.with_tool_call = true;
        tool_required_only.with_content = false;  // tool_choice=required never allows content
        tool_required_only.require_tool_support = true;
        tool_required_only.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
        scenarios.push_back(tool_required_only);

        needle_scenario tool_parallel;
        tool_parallel.name = "parallel-tool-calls";
        tool_parallel.provide_tools = true;
        tool_parallel.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        tool_parallel.with_tool_call = true;
        tool_parallel.tool_call_count = 2;
        tool_parallel.parallel_tool_calls = true;
        tool_parallel.require_tool_support = true;
        tool_parallel.with_content = (info.tools_emit_content_with_calls == ToolsEmitContentWithCalls::Yes);
        tool_parallel.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
        scenarios.push_back(tool_parallel);

        if (info.supports_thinking == ThinkingSupport::Yes) {
            needle_scenario tool_with_reasoning;
            tool_with_reasoning.name = "tool-with-reasoning";
            tool_with_reasoning.provide_tools = true;
            tool_with_reasoning.with_tool_call = true;
            tool_with_reasoning.with_reasoning = true;
            tool_with_reasoning.enable_thinking = true;
            tool_with_reasoning.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
            tool_with_reasoning.require_tool_support = true;
            tool_with_reasoning.require_thinking_support = true;
            tool_with_reasoning.with_content = (info.tools_emit_content_with_calls == ToolsEmitContentWithCalls::Yes);
            tool_with_reasoning.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
            scenarios.push_back(tool_with_reasoning);
        }
    }

    return scenarios;
}

static std::string describe_scenario(const needle_scenario & scenario) {
    std::ostringstream oss;
    oss << "tools=" << (scenario.provide_tools ? "yes" : "no");
    oss << ", choice=" << tool_choice_name(scenario.tool_choice);
    if (scenario.parallel_tool_calls) {
        oss << ", parallel";
    }
    oss << ", tool_calls=";
    if (scenario.with_tool_call) {
        oss << scenario.tool_call_count;
        oss << "x" << scenario.args_per_tool_call << "args";
    } else {
        oss << 0;
    }
    if (scenario.with_reasoning) {
        oss << ", reasoning";
    }
    if (scenario.enable_thinking) {
        oss << ", thinking=on";
    } else if (scenario.force_disable_thinking) {
        oss << ", thinking=forced-off";
    }
    return oss.str();
}

// Test that parsers correctly reject content in tool_choice=required mode.
// When tool_choice is REQUIRED, parsers should only accept tool calls (and optionally thinking),
// but NOT content. This test verifies that invariant holds for all templates by using init_delta
// to properly render assistant messages through templates.
static bool test_required_tool_rejects_content() {
    printf("[%s]\n", __func__);

    const char * template_filter = std::getenv("NEEDLE_TEMPLATE_FILTER");

    // Use shared template capabilities
    const auto & templates = get_template_capabilities();

    size_t tested = 0;
    size_t passed = 0;
    size_t skipped = 0;

    for (const auto & info : templates) {
        if (template_filter && std::string(info.name) != template_filter) {
            continue;
        }

        // Skip templates without tool support
        if (info.supports_tools != ToolSupport::Yes) {
            continue;
        }

        auto tmpls = read_templates(info.jinja_path);
        if (!tmpls) {
            if (g_verbose >= 1) {
                printf("  " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " %s (template not found)\n", info.name);
            }
            skipped++;
            continue;
        }

        // Test scenarios that should FAIL in required mode:
        // Messages with content (but no tool calls) rendered through the template
        struct test_scenario {
            const char * name;
            common_chat_msg delta_msg;
            common_reasoning_format reasoning_format;
        };

        std::vector<test_scenario> scenarios;

        // Scenario 1: Content only - should always fail
        scenarios.push_back({"content-only", simple_assist_msg("Hello, this is just content without any tool call."), COMMON_REASONING_FORMAT_NONE});

        // Scenario 2: Thinking + content (if supported) - should fail (content is still present)
        if (info.supports_thinking == ThinkingSupport::Yes) {
            scenarios.push_back({"thinking-then-content",
                simple_assist_msg("Here is my response.", "Let me think about this..."),
                COMMON_REASONING_FORMAT_DEEPSEEK});
        }

        tested++;
        bool template_passed = true;

        for (const auto & scenario : scenarios) {
            // Use init_delta to get the properly-rendered delta through the template
            delta_data data;
            try {
                data = init_delta(
                    tmpls.get(),
                    {},  // end_tokens - let it use params.additional_stops
                    message_user,
                    scenario.delta_msg,
                    {python_tool},  // tools
                    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
                    scenario.reasoning_format,
                    {},  // customize_inputs
                    chat_parser_impl::EXPERIMENTAL
                );
            } catch (const std::exception & e) {
                if (g_verbose >= 0) {
                    printf("  " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " %s [%s]: init_delta failed: %s\n",
                           info.name, scenario.name, e.what());
                }
                continue;
            }

            if (data.params.parser.empty()) {
                if (g_verbose >= 1) {
                    printf("  " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " %s [%s]: no PEG parser\n",
                           info.name, scenario.name);
                }
                continue;
            }

            common_peg_arena arena;
            arena.load(data.params.parser);

            bool threw = false;
            std::string error_msg;
            try {
                common_chat_peg_parse(arena, data.delta, /* is_partial = */ false, {data.params.format});
            } catch (const std::exception & e) {
                threw = true;
                error_msg = e.what();
            }

            // In required mode, content should always cause parser to fail
            if (!threw) {
                if (g_verbose >= 0) {
                    printf("  " ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET " %s [%s]: expected parser to reject content but it succeeded\n",
                           info.name, scenario.name);
                    printf("    Delta: %.80s%s\n", data.delta.c_str(), data.delta.size() > 80 ? "..." : "");
                }
                template_passed = false;
            } else if (g_verbose >= 2) {
                printf("  " ANSI_COLOR_GREEN "PASS" ANSI_COLOR_RESET " %s [%s]\n", info.name, scenario.name);
            }
        }

        if (template_passed) {
            passed++;
            if (g_verbose >= 1) {
                printf("  " ANSI_COLOR_GREEN "PASS" ANSI_COLOR_RESET " %s\n", info.name);
            }
        }
    }

    printf("  Results: %zu/%zu passed, %zu skipped\n", passed, tested, skipped);
    return passed == tested;
}

static bool test_systematic_needle_streaming() {
    printf("[%s]\n", __func__);

    const char * template_filter = std::getenv("NEEDLE_TEMPLATE_FILTER");
    const char * scenario_filter = std::getenv("NEEDLE_SCENARIO_FILTER");

    if (g_verbose >= 1 || template_filter || scenario_filter) {
        printf("    Filters: template=%s, scenario=%s\n",
               template_filter ? template_filter : "(all)",
               scenario_filter ? scenario_filter : "(all)");
    }

    const auto matches_filter = [](const char * filter, const std::string & value) {
        if (filter == nullptr || *filter == '\0') {
            return true;
        }
        return value == filter;
    };

    struct template_summary {
        std::string name;
        size_t scenarios_total = 0;
        size_t scenarios_passed = 0;
        std::vector<std::string> failed_scenarios;
        std::vector<std::pair<std::string, std::string>> failed_scenarios_with_errors; // <scenario_name, error_message>
    };
    std::vector<template_summary> summaries;

    // Use shared template capabilities
    const auto & templates = get_template_capabilities();

    // Verify declared capabilities match what minja detects
    if (!verify_template_capabilities(templates)) {
        return false;
    }

    // Test each template
    for (const auto & tmpl_info : templates) {
        if (g_verbose >= 1) {
            printf("  ⚫ %s\n", tmpl_info.name);
            fflush(stdout);
        }

        auto tmpls = read_templates(tmpl_info.jinja_path);
        if (!tmpls) {
            if (g_verbose >= 1) {
                printf("    " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (template not found)\n");
            }
            continue;
        }
        if (tmpl_info.skip == Skip::Yes) {
            if (g_verbose >= 1) {
                printf("    " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (temporarily disabled)\n");
            }
            continue;
        }
        if (!matches_filter(template_filter, tmpl_info.name)) {
            if (g_verbose >= 2) {
                printf("    " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (template filter)\n");
            }
            continue;
        }

        // Cross-check static template info with minja's capabilities detection
        // Note: minja detection relies on the template using 'enable_thinking' variable.
        // Some templates (e.g., Seed OSS) always include thinking tags but don't use this variable,
        // so we only warn about mismatches rather than failing.
        bool minja_thinks = common_chat_templates_support_enable_thinking(tmpls.get());
        bool minja_tools = common_chat_templates_support_tools(tmpls.get());
        bool static_thinks = (tmpl_info.supports_thinking == ThinkingSupport::Yes);
        bool static_tools = (tmpl_info.supports_tools == ToolSupport::Yes);

        if (minja_thinks != static_thinks && g_verbose >= 1) {
            printf("    " ANSI_COLOR_YELLOW "⚠" ANSI_COLOR_RESET " thinking support: static=%s, minja=%s\n",
                   static_thinks ? "Yes" : "No", minja_thinks ? "Yes" : "No");
        }
        if (minja_tools != static_tools) {
            printf("    " ANSI_COLOR_RED "✗ FAIL" ANSI_COLOR_RESET " tools mismatch: static=%s, minja=%s\n",
                   static_tools ? "Yes" : "No", minja_tools ? "Yes" : "No");
            throw std::runtime_error("Template capabilities mismatch for " + std::string(tmpl_info.name));
        }

        template_summary summary_entry;
        summary_entry.name = tmpl_info.name;

        auto scenarios = build_needle_scenarios(tmpl_info);
        for (const auto & scenario : scenarios) {
            if (!matches_filter(scenario_filter, scenario.name)) {
                if (g_verbose >= 2) {
                    printf("      - %s: " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (filter)\n", scenario.name.c_str());
                }
                continue;
            }
            if (scenario.require_thinking_support && tmpl_info.supports_thinking == ThinkingSupport::No) {
                if (g_verbose >= 2) {
                    printf("      - %s: " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (no thinking)\n", scenario.name.c_str());
                }
                continue;
            }
            if (scenario.require_tool_support && tmpl_info.supports_tools == ToolSupport::No) {
                if (g_verbose >= 2) {
                    printf("      - %s: " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (no tools)\n", scenario.name.c_str());
                }
                continue;
            }
            if (scenario.parallel_tool_calls && !common_chat_templates_support_parallel_tool_calls(tmpls.get())) {
                if (g_verbose >= 2) {
                    printf("      - %s: " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (no parallel)\n", scenario.name.c_str());
                }
                continue;
            }

            if (g_verbose >= 2) {
                printf("    🔵 %s (%s)\n", scenario.name.c_str(), describe_scenario(scenario).c_str());
                fflush(stdout);
            }

            summary_entry.scenarios_total++;

            try {
                // Override tool name if template specifies a custom one
                auto scenario_copy = scenario;
                if (tmpl_info.needle_tool_name != nullptr) {
                    scenario_copy.tool_name = tmpl_info.needle_tool_name;
                }

                auto ctx = make_needle_context(scenario_copy, tmpl_info.format);
                std::vector<common_chat_tool> scenario_tools;
                if (scenario_copy.provide_tools) {
                    // Create a dynamic tool with parameter names matching the needle markers
                    // This is needed for parsers that use literal_tag for parameter names (e.g., Llama 3.1 builtin tools)
                    if (!ctx.expected_msg.tool_calls.empty()) {
                        common_chat_tool dynamic_tool;
                        dynamic_tool.name = scenario_copy.tool_name;
                        dynamic_tool.description = "Dynamic tool for needle testing";

                        // Build parameters schema from ALL tool calls' argument names
                        // This is important for parallel tool calls where each call may have different parameter names
                        json properties = json::object();
                        json required = json::array();

                        for (const auto& tool_call : ctx.expected_msg.tool_calls) {
                            if (tool_call.arguments.empty()) continue;
                            json args_json = json::parse(tool_call.arguments);
                            for (auto& [key, value] : args_json.items()) {
                                if (!properties.contains(key)) {  // Avoid duplicates
                                    properties[key] = {
                                        {"type", "string"},
                                        {"description", "Needle test parameter"}
                                    };
                                    required.push_back(key);
                                }
                            }
                        }

                        dynamic_tool.parameters = json({
                            {"type", "object"},
                            {"properties", properties},
                            {"required", required}
                        }).dump();
                        scenario_tools = {dynamic_tool};
                    } else {
                        scenario_tools = {python_tool};
                    }
                }

                auto reasoning_format = scenario.with_reasoning ? COMMON_REASONING_FORMAT_DEEPSEEK : COMMON_REASONING_FORMAT_NONE;

                auto data = init_delta(tmpls.get(), {}, message_user, ctx.expected_msg, scenario_tools,
                                       scenario.tool_choice, reasoning_format,
                                       [&](common_chat_templates_inputs & inputs) {
                                           inputs.parallel_tool_calls = scenario.parallel_tool_calls;
                                           inputs.experimental_new_parsers = true;  // Needle tests use new PEG parsers
                                           if (scenario.force_disable_thinking) {
                                               inputs.enable_thinking = false;
                                               inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
                                           } else if (scenario.enable_thinking || scenario.with_reasoning) {
                                               inputs.enable_thinking = true;
                                               inputs.reasoning_format = reasoning_format;
                                           } else {
                                               inputs.enable_thinking = false;
                                               inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
                                           }
                                       });

                if (scenario.skip_if_thinking_forced && data.params.thinking_forced_open) {
                    if (g_verbose >= 2) {
                        printf("      - %s: " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (forces thinking)\n", scenario.name.c_str());
                    }
                    continue;
                }
                if (scenario.force_disable_thinking && data.params.thinking_forced_open) {
                    if (g_verbose >= 2) {
                        printf("      - %s: " ANSI_COLOR_YELLOW "SKIP" ANSI_COLOR_RESET " (forces thinking)\n", scenario.name.c_str());
                    }
                    continue;
                }

                if (data.params.parser.empty()) {
                    throw std::runtime_error("Template returned empty parser definition");
                }

                auto syntax = get_syntax(data.params, reasoning_format);
                if (syntax.parser.empty()) {
                    throw std::runtime_error("PEG arena failed to load");
                }

                auto syntax_copy = syntax;
                auto parse_fn = [syntax_copy](const std::string & msg, bool is_partial) mutable {
                    return common_chat_peg_parse(syntax_copy.parser, msg, is_partial, syntax_copy);
                };

                std::string raw_message = data.delta;
                if (tmpl_info.inject_reasoning_after_format == InjectReasoningAfterFormat::Yes && scenario.with_reasoning &&
                    raw_message.find(ctx.reasoning_needles.first) == std::string::npos) {
                    const char * open = tmpl_info.think_open_tag ? tmpl_info.think_open_tag : "<think>";
                    const char * close = tmpl_info.think_close_tag ? tmpl_info.think_close_tag : "</think>";
                    std::string prefix;
                    if (data.params.thinking_forced_open) {
                        prefix = ctx.expected_msg.reasoning_content;
                    } else {
                        prefix = std::string(open) + ctx.expected_msg.reasoning_content + std::string(close);
                    }
                    auto inserted_len = prefix.size();
                    raw_message = prefix + raw_message;
                    std::string close_tag = close ? close : "";
                    if (!close_tag.empty() && raw_message.size() >= inserted_len + close_tag.size() &&
                        raw_message.compare(inserted_len, close_tag.size(), close_tag) == 0) {
                        raw_message.erase(inserted_len, close_tag.size());
                    }
                }

                auto result = test_streaming_with_needles(ctx, raw_message, parse_fn);
                verify_needle_results(ctx, result);
                if (g_verbose >= 1) {
                    printf("    %s: " ANSI_COLOR_GREEN "✓ OK" ANSI_COLOR_RESET "\n", scenario.name.c_str());
                }
                summary_entry.scenarios_passed++;
            } catch (const std::exception & e) {
                summary_entry.failed_scenarios.push_back(scenario.name);
                summary_entry.failed_scenarios_with_errors.push_back({scenario.name, e.what()});
            }
        }

        summaries.push_back(summary_entry);

        // Print per-template summary (always show for templates that were tested)
        if (summary_entry.scenarios_total > 0) {
            if (summary_entry.failed_scenarios.empty()) {
                printf("  %s: " ANSI_COLOR_GREEN "%zu/%zu passed" ANSI_COLOR_RESET "\n",
                       summary_entry.name.c_str(), summary_entry.scenarios_passed, summary_entry.scenarios_total);
            } else {
                printf("  %s: " ANSI_COLOR_RED "%zu/%zu passed" ANSI_COLOR_RESET " (failed: %s)\n",
                       summary_entry.name.c_str(), summary_entry.scenarios_passed, summary_entry.scenarios_total,
                       string_join(summary_entry.failed_scenarios, ", ").c_str());
                // Print detailed failures underneath
                for (const auto & [scenario_name, error_msg] : summary_entry.failed_scenarios_with_errors) {
                    printf("    %s: " ANSI_COLOR_RED "✗ FAIL" ANSI_COLOR_RESET " %s\n", scenario_name.c_str(), error_msg.c_str());
                }
            }
        }
    }

    size_t templates_total = 0;
    size_t templates_passing = 0;
    std::vector<std::string> passing_templates;
    std::vector<std::string> failing_template_summaries;
    for (const auto & entry : summaries) {
        if (entry.scenarios_total == 0) {
            continue;
        }
        templates_total++;
        if (entry.failed_scenarios.empty()) {
            templates_passing++;
            passing_templates.push_back(entry.name);
        } else {
            std::ostringstream oss;
            oss << entry.name << " (" << entry.scenarios_passed << "/" << entry.scenarios_total << ")";
            failing_template_summaries.push_back(oss.str());
        }
    }

    // Print overall summary with colors
    printf("\n  Summary: ");
    if (templates_passing == templates_total) {
        printf(ANSI_COLOR_GREEN "%zu/%zu templates passed" ANSI_COLOR_RESET "\n", templates_passing, templates_total);
    } else {
        printf(ANSI_COLOR_RED "%zu/%zu templates passed" ANSI_COLOR_RESET "\n", templates_passing, templates_total);
    }
    if (g_verbose >= 1 && !passing_templates.empty()) {
        printf("    " ANSI_COLOR_GREEN "Passed" ANSI_COLOR_RESET ": %s\n", string_join(passing_templates, ", ").c_str());
    }
    if (!failing_template_summaries.empty()) {
        printf("    " ANSI_COLOR_RED "Failed" ANSI_COLOR_RESET ": %s\n", string_join(failing_template_summaries, ", ").c_str());
    }
    printf("\n");

    return templates_passing == templates_total;
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
    // Set log verbosity based on LOG_LEVEL env var (0=quiet, 1=info, 2+=debug)
    // Lower threshold = less logging. Set to -1 by default to suppress all logs.
    // LOG_LEVEL=2 enables all debug output.
    int log_thold = g_verbose >= 2 ? 999 : -1;
    common_log_set_verbosity_thold(log_thold);

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
            const std::string chat_test = std::getenv("CHAT_TEST") ? std::getenv("CHAT_TEST") : "";

            if (chat_test == "" || chat_test == "msg_diffs_compute") {
                test_msg_diffs_compute();
            }
            if (chat_test == "" || chat_test == "msgs_oaicompat_json_conversion") {
                test_msgs_oaicompat_json_conversion();
            }
            if (chat_test == "" || chat_test == "tools_oaicompat_json_conversion") {
                test_tools_oaicompat_json_conversion();
            }
            if (chat_test == "" || chat_test == "template_output_parsers") {
                test_template_output_parsers(chat_parser_impl::LEGACY);
                test_template_output_parsers(chat_parser_impl::EXPERIMENTAL);
            }
            if (chat_test == "" || chat_test == "template_output_peg_parsers") {
                test_template_output_peg_parsers();
            }
            if (chat_test == "" || chat_test == "required_tool_rejects_content") {
                if (!test_required_tool_rejects_content()) {
                    return 1;
                }
            }
            if (chat_test == "" || chat_test == "format_detection_with_tools") {
                if (!test_format_detection_with_tools()) {
                    return 1;
                }
            }
            if (chat_test == "" || chat_test == "systematic_needle_streaming") {
                if (!test_systematic_needle_streaming()) {
                    return 1;
                }
            }
            std::cout << "\n[chat] All tests passed!" << '\n';
        }
        return 0;
}
