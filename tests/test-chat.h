//  Tests chat handling, including grammar generation and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include "chat.h"

#include "common.h"
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
#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

// Parser implementation selector for tests
enum class chat_parser_impl {
    LEGACY,      // Use legacy monolithic parsers
    EXPERIMENTAL // Use new modular PEG parsers
};

const char * chat_parser_impl_name(chat_parser_impl impl);

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
    std::string name;
    std::string jinja_path;
    common_chat_format legacy_format;
    common_chat_format experimental_format;
    ThinkingSupport supports_thinking = ThinkingSupport::No;
    const char * think_open_tag = nullptr;   // Opening tag for thinking (nullptr = auto-detect)
    const char * think_close_tag = nullptr;  // Closing tag for thinking (nullptr = no thinking)
    // TODO(ochafik): Add minja detection for these capabilities (see https://github.com/ochafik/minja/pull/20)
    ReasoningRequiresTools reasoning_requires_tools = ReasoningRequiresTools::No;
    ToolsEmitContentWithCalls tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    InjectReasoningAfterFormat inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    SupportsDisableThinking supports_disable_thinking = SupportsDisableThinking::Yes;
    SupportsReasoningOnly supports_reasoning_only = SupportsReasoningOnly::Yes;
    ToolCallsHaveIds tool_calls_have_ids = ToolCallsHaveIds::No;
    std::vector<std::string> end_tokens;
};

inline std::ostream & operator<<(std::ostream & os, const common_chat_msg_diff & diff) {
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
inline std::ostream & operator<<(std::ostream & os, const std::vector<common_chat_msg_diff> & diffs) {
    os << "[\n";
    for (const auto & diff : diffs) {
        os << "  " << diff << ",\n";
    }
    os << "]";
    return os;
}
inline std::ostream & operator<<(std::ostream & os, const common_chat_msg & msg) {
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

template <class T> inline bool equals(const T & expected, const T & actual) {
    return expected == actual;
}

inline common_chat_msg normalize(const common_chat_msg & msg) {
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
inline bool equals(const common_chat_msg & expected, const common_chat_msg & actual) {
    return normalize(expected) == normalize(actual);
}

template <class T> inline void assert_equals(const T & expected, const T & actual, const std::string & desc = "") {
    if (!equals(expected, actual)) {
        std::ostringstream ss;
        ss << "Expected: " << expected << std::endl;
        ss << "Actual: " << actual << std::endl;
        ss << std::flush;
        throw std::runtime_error("Test failed" + (desc.empty() ? "" : " (" + desc + ")") + ":\n" + ss.str());
    }
}

inline void assert_throws(const std::function<void()> & fn, const std::string & desc = "") {
    try {
        fn();
        throw std::runtime_error("Failed to throw" + (desc.empty() ? "" : " (" + desc + ")"));
    } catch (const std::runtime_error &) {
        // Do nothing
    }
}

common_chat_templates_ptr read_templates(const std::string & path);

// TODO: extract to common helper (copied from test-grammar-integration.cpp)
inline bool match_string(const std::string & input, llama_grammar * grammar) {
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

void assert_msg_equals(const common_chat_msg & expected, const common_chat_msg & actual, bool ignore_whitespace_differences = false);

static common_chat_tool special_function_tool {
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
static common_chat_tool special_function_tool_with_optional_param {
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
static common_chat_tool python_tool {
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
static common_chat_tool code_interpreter_tool {
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
static common_chat_tool complex_function_tool {
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
static common_chat_tool web_search_tool {
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
static common_chat_tool read_file_tool {
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
static common_chat_tool emoji_function_tool {
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
static common_chat_tool complex_function_in_think_tool {
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
static common_chat_tool process_data_tool {
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

// TODO: inline in each chat-parser test file
static std::vector<common_chat_tool> tools           { special_function_tool, special_function_tool_with_optional_param, python_tool };
static std::vector<common_chat_tool> llama_3_1_tools { special_function_tool, code_interpreter_tool };
static std::vector<common_chat_tool> glm_4_5_tools   { special_function_tool, special_function_tool_with_optional_param, complex_function_tool, web_search_tool };
static std::vector<common_chat_tool> kimi_k2_tools   { special_function_tool, special_function_tool_with_optional_param, complex_function_tool, web_search_tool, read_file_tool, emoji_function_tool, complex_function_in_think_tool };

/*
  Applies the template to 1 user message w/ add_generation_prompt=true, then w/ the test message w/ add_generation_prompt=false,
  gets the diff, removes any end tokens and parses the result w/ the grammar, checking that
  the parsed message is the same as the test_message
*/
void test_templates(chat_parser_impl impl, const struct common_chat_templates * tmpls, const std::vector<std::string> & end_tokens,
                          const common_chat_msg & test_message,
                          const std::vector<common_chat_tool> & tools = {},
                          const std::string & expected_delta = "",
                          bool expect_grammar_triggered = true,
                          bool test_grammar_if_triggered = true,
                          common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE,
                          bool ignore_whitespace_differences = false,
                          bool expect_parse_failure = false,
                          const std::function<void(std::string &)> & mutate_delta = {});

static const common_chat_msg message_user {
    "user",
    "Hey there!",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

static const common_chat_msg message_user_parts {
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

inline common_chat_msg simple_assist_msg(const std::string & content, const std::string & reasoning_content = "", const std::string & tool_name = "", const std::string & arguments = "", const std::string & id = "") {
    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = content;
    msg.reasoning_content = reasoning_content;
    if (!tool_name.empty()) {
        msg.tool_calls.push_back({ tool_name, arguments, id });
    }
    return msg;
}

std::unique_ptr<llama_grammar> build_grammar(const std::string & grammar_str);

common_chat_syntax get_syntax(const common_chat_params & params,
                              common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE);


// Use for PEG parser implementations
struct peg_test_case {
    common_chat_templates_inputs params;
    std::string input;
    common_chat_msg expect;
};

void test_peg_parser(chat_parser_impl impl, common_chat_templates * tmpls, const std::function<void(peg_test_case &)> & init);

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
        // LOG_INF("Streaming msg: %s\n", common_chat_msgs_to_json_oaicompat<json>({curr_msg}).dump().c_str());
        for (auto diff: common_chat_msg_diff::compute_diffs(last_msg, curr_msg)) {
            // LOG_INF("Streaming diff: %s\n", common_chat_msg_diff_to_json_oaicompat<json>(diff).dump().c_str());
            if (!diff.reasoning_content_delta.empty()) {
                merged.reasoning_content += diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                merged.content += diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                // Check if this is a new tool call or an update to an existing one
                bool is_new_tool_call = diff.tool_call_index >= merged.tool_calls.size();
                if (is_new_tool_call && !diff.tool_call_delta.name.empty()) {
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
            LOG_DBG("Streaming merged: %s\n", common_chat_msgs_to_json_oaicompat<json>({merged}).dump().c_str());
        }
        assert_msg_equals(curr_msg, merged, true);
        last_msg = curr_msg;
    }
    assert_msg_equals(expected, parse_msg(raw_message), true);
    assert_msg_equals(expected, merged, true);
}

static const common_chat_msg message_assist                              = simple_assist_msg("Hello, world!\nWhat's up?");
static const common_chat_msg message_assist_empty                        = simple_assist_msg("");
static const common_chat_msg message_assist_thoughts_unparsed_deepseek   = simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?");
static const common_chat_msg message_assist_thoughts_unparsed_md         = simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```");
static const common_chat_msg message_assist_thoughts_unparsed_md_partial = simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}");

static const common_chat_msg message_assist_thoughts_unparsed_r7b       = simple_assist_msg("<|START_THINKING|>I'm\nthinking<|END_THINKING|>Hello, world!\nWhat's up?");
static const common_chat_msg message_assist_thoughts_unparsed_magistral = simple_assist_msg("[THINK]raisonnement[/THINK]Réponse");
static const common_chat_msg message_assist_thoughts                    = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking");
static const common_chat_msg message_assist_thoughts_unopened_unparsed  = simple_assist_msg("I'm\nthinking</think>Hello, world!\nWhat's up?");
static const common_chat_msg message_assist_thoughts_no_content         = simple_assist_msg("", "I'm\nthinking");
static const common_chat_msg message_assist_call                        = simple_assist_msg("", "", "special_function", "{\"arg1\": 1}");
static const common_chat_msg message_assist_call_noopt                  = simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1}");
static const common_chat_msg message_assist_call_withopt                = simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}");
static const common_chat_msg message_assist_call_content                = simple_assist_msg("Hello, world!\nWhat's up?", "", "special_function", "{\"arg1\":1}");
static const common_chat_msg message_assist_call_empty_args             = simple_assist_msg("", "", "special_function");
static const common_chat_msg message_assist_call_cutoff_args            = simple_assist_msg("", "", "special_function", "{\"arg");
static const common_chat_msg message_assist_call_thoughts               = simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\":1}");
static const common_chat_msg message_assist_call_thoughts_unparsed      = simple_assist_msg("<think>I'm\nthinking</think>\n\n", "", "special_function", "{\"arg1\": 1}");
static const common_chat_msg message_assist_call_thoughts_content       = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}");
static const common_chat_msg message_assist_call_id                     = simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "123456789");
static const common_chat_msg message_assist_call_idx                    = simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "0");
static const common_chat_msg message_assist_thoughts_call_idx           = simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}", /* id = */ "0");
static const common_chat_msg message_assist_call_content_idx            = simple_assist_msg("Hello, world!\nWhat's up?", "", "special_function", "{\"arg1\":1}", /* id = */ "0");
static const common_chat_msg message_assist_call_thoughts_content_idx   = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}", /* id = */ "0");
static const common_chat_msg message_assist_call_python                 = simple_assist_msg("", "", "python", "{\"code\":\"print('hey')\"}");
static const common_chat_msg message_assist_call_python_lines           = simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')\"}");
static const common_chat_msg message_assist_call_python_lines_unclosed  = simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')");
static const common_chat_msg message_assist_call_code_interpreter       = simple_assist_msg("", "", "code_interpreter", "{\"code\":\"print('hey')\"}");

void run_template_test_suite(chat_parser_impl impl, const template_capabilities & template_caps, const common_chat_templates_ptr & tmpls);

void test_apertus_parser(chat_parser_impl impl);
void test_apriel_1_5_parser(chat_parser_impl impl);
void test_command_r7b_parser(chat_parser_impl impl);
void test_deepseek_r1_parser(chat_parser_impl impl);
void test_deepseek_v3_1_parser(chat_parser_impl impl);
void test_firefunction_v2_parser(chat_parser_impl impl);
void test_functionary_v3_1_llama_3_1_parser(chat_parser_impl impl);
void test_functionary_v3_2_parser(chat_parser_impl impl);
void test_generic_parser(chat_parser_impl impl);
void test_glm_4_5_parser(chat_parser_impl impl);
void test_gpt_oss_parser(chat_parser_impl impl);
void test_granite_parser(chat_parser_impl impl);
void test_hermes_2_pro_parser(chat_parser_impl impl);
void test_kimi_k2_parser(chat_parser_impl impl);
void test_lfm2_parser(chat_parser_impl impl);
void test_llama_3_x_parser(chat_parser_impl impl);
void test_magistral_parser(chat_parser_impl impl);
void test_minimax_m2_parser(chat_parser_impl impl);
void test_ministral_3_parser(chat_parser_impl impl);
void test_mistral_nemo_parser(chat_parser_impl impl);
void test_nemotron_v2_parser(chat_parser_impl impl);
void test_nemotron_v3_parser(chat_parser_impl impl);
void test_qwen3_coder_xml_parser(chat_parser_impl impl);
void test_seed_oss_parser(chat_parser_impl impl);
void test_xiaomi_mimo_parser(chat_parser_impl impl);
