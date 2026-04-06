//  Tests chat handling, including grammar generation and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include "../src/llama-grammar.h"
#include "../src/unicode.h"
#include "chat-auto-parser.h"
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
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
        os << "  { name: " << tool_call.name << "; arguments: " << tool_call.arguments << "; id: " << tool_call.id
           << " },\n";
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
        }
    }
    return normalized;
}

template <> bool equals(const common_chat_msg & expected, const common_chat_msg & actual) {
    return normalize(expected) == normalize(actual);
}

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (!equals(expected, actual)) {
        std::ostringstream oss_expected;
        oss_expected << expected;
        std::ostringstream oss_actual;
        oss_actual << actual;
        LOG_ERR("Expected: %s\n", oss_expected.str().c_str());
        LOG_ERR("Actual: %s\n", oss_actual.str().c_str());
        common_log_flush(common_log_main());
        throw std::runtime_error("Test failed");
    }
}

static std::string read_file(const std::string & path) {
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

// Helper to format a code point as a readable string
static std::string format_codepoint(uint32_t cp) {
    if (cp >= 32 && cp < 127) {
        return std::string("'") + static_cast<char>(cp) + "'";
    } else if (cp == '\n') {
        return "'\\n'";
    } else if (cp == '\r') {
        return "'\\r'";
    } else if (cp == '\t') {
        return "'\\t'";
    } else {
        return "U+" + std::to_string(cp);
    }
}

// Helper to format expected element from grammar stack
static std::string format_expected_element(const llama_grammar_rules & /* rules*/, const llama_grammar_element * elem) {
    if (!elem) {
        return "<end>";
    }

    switch (elem->type) {
        case LLAMA_GRETYPE_END:
            return "<end of rule>";
        case LLAMA_GRETYPE_ALT:
            return "<alternative>";
        case LLAMA_GRETYPE_RULE_REF:
            {
                // Find rule name - just show rule ID for now
                return "<rule-" + std::to_string(elem->value) + ">";
            }
        case LLAMA_GRETYPE_CHAR:
            {
                std::string                   result;
                const llama_grammar_element * pos   = elem;
                bool                          first = true;

                do {
                    if (!first) {
                        result += " | ";
                    }
                    first = false;

                    if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                        // Range like [a-z]
                        result += "[" + format_codepoint(pos->value) + "-" + format_codepoint(pos[1].value) + "]";
                        pos += 2;
                    } else {
                        result += format_codepoint(pos->value);
                        pos += 1;
                    }
                } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

                return result;
            }
        case LLAMA_GRETYPE_CHAR_NOT:
            {
                std::string                   result = "[^";
                const llama_grammar_element * pos    = elem;
                bool                          first  = true;

                do {
                    if (!first) {
                        result += " ";
                    }
                    first = false;

                    if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                        result += format_codepoint(pos->value) + "-" + format_codepoint(pos[1].value);
                        pos += 2;
                    } else {
                        result += format_codepoint(pos->value);
                        pos += 1;
                    }
                } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

                return result + "]";
            }
        case LLAMA_GRETYPE_CHAR_ANY:
            return "<any char>";
        case LLAMA_GRETYPE_TOKEN:
            return "<token-" + std::to_string(elem->value) + ">";
        case LLAMA_GRETYPE_TOKEN_NOT:
            return "<not-token-" + std::to_string(elem->value) + ">";
        default:
            return "<unknown>";
    }
}

// Get description of what the grammar expects at current position
static std::string get_expected_description(const llama_grammar_rules & rules, const llama_grammar_stacks & stacks) {
    if (stacks.empty()) {
        return "<no valid continuations>";
    }

    std::string           result;
    std::set<std::string> seen;

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            if (seen.insert("<end>").second) {
                if (!result.empty()) {
                    result += " OR ";
                }
                result += "<end>";
            }
            continue;
        }

        const llama_grammar_element * elem = stack.back();
        std::string                   desc = format_expected_element(rules, elem);
        if (seen.insert(desc).second) {
            if (!result.empty()) {
                result += " OR ";
            }
            result += desc;
        }
    }

    return result;
}

// Result of a detailed grammar match attempt
struct grammar_match_result {
    bool        success            = false;  // Did the string fully match the grammar?
    size_t      matched_bytes      = 0;      // Bytes successfully matched before failure
    size_t      matched_codepoints = 0;      // Codepoints successfully matched before failure
    size_t      total_bytes        = 0;      // Total bytes in input
    size_t      total_codepoints   = 0;      // Total codepoints in input
    std::string matched_prefix;              // The portion that was successfully matched
    std::string failing_char;                // The character that caused failure (if any)
    std::string expected_description;        // What the grammar expected at failure point
    bool        incomplete = false;          // True if matched all input but grammar expects more
};

// Detailed version of match_string that returns failure information
static grammar_match_result match_string_detailed(const std::string & input, llama_grammar * grammar) {
    grammar_match_result result;
    result.total_bytes = input.size();

    const auto cpts         = unicode_cpts_from_utf8(input);
    result.total_codepoints = cpts.size();

    auto &       stacks_cur = llama_grammar_get_stacks(grammar);
    const auto & rules      = llama_grammar_get_rules(grammar);

    size_t byte_pos = 0;

    for (size_t i = 0; i < cpts.size(); i++) {
        const auto & cpt = cpts[i];

        // Get expected before accepting (for error reporting)
        std::string expected_before = get_expected_description(rules, stacks_cur);

        llama_grammar_accept(grammar, cpt);

        // Calculate byte position for this codepoint
        size_t cpt_bytes = 0;
        if (cpt < 0x80) {
            cpt_bytes = 1;
        } else if (cpt < 0x800) {
            cpt_bytes = 2;
        } else if (cpt < 0x10000) {
            cpt_bytes = 3;
        } else {
            cpt_bytes = 4;
        }

        if (stacks_cur.empty()) {
            // Grammar failed to match at this point
            result.matched_bytes        = byte_pos;
            result.matched_codepoints   = i;
            result.matched_prefix       = input.substr(0, byte_pos);
            result.failing_char         = format_codepoint(cpt);
            result.expected_description = expected_before;
            result.incomplete           = false;
            return result;
        }

        byte_pos += cpt_bytes;
    }

    // All input matched - check if grammar is complete
    result.matched_bytes      = input.size();
    result.matched_codepoints = cpts.size();
    result.matched_prefix     = input;

    if (std::any_of(stacks_cur.begin(), stacks_cur.end(), [](const auto & stack) { return stack.empty(); })) {
        // An empty stack means that the grammar has been completed
        result.success    = true;
        result.incomplete = false;
    } else {
        // Grammar expects more input
        result.success              = false;
        result.incomplete           = true;
        result.expected_description = get_expected_description(rules, stacks_cur);
    }

    return result;
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
        return "";  // ignore parial JSON contents for comparison purposes
    }
}

static void assert_msg_equals(const common_chat_msg & expected,
                              const common_chat_msg & actual,
                              bool                    ignore_whitespace_differences = false) {
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

static common_chat_tool special_function_tool{
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
static common_chat_tool special_function_tool_with_optional_param{
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

static common_chat_tool empty_args_tool{
    /* .name = */ "empty_args",
    /* .description = */ "A tool that takes no arguments",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {}
    })",
};

static common_chat_tool empty_args_tool_no_properties{
    /* .name = */ "empty_args_no_props",
    /* .description = */ "A tool that takes no arguments and has no properties",
    /* .parameters = */ R"({
        "type": "object"
    })",
};

static common_chat_tool python_tool{
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

static common_chat_tool html_tool{
    /* .name = */ "html",
    /* .description = */ "an html validator",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "markup": {
                "type": "string",
                "description": "HTML markup to validate."
            }
        },
        "required": ["markup"]
    })",
};

static common_chat_tool get_time_tool{
    /* .name = */ "get_time",
    /* .description = */ "Get the current time in a city",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    })",
};

static common_chat_tool get_weather_tool{
    /* .name = */ "get_weather",
    /* .description = */ "Get the current weather in a city",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    })",
};

static common_chat_tool todo_list{
    /* .name = */ "todo_list",
    /* .description = */ "Create or update the todo list",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of TODO list items"
            }
        },
        "required": ["todos"]
    })",
};

static common_chat_tool edit_tool{
    /* .name = */ "edit",
    /* .description = */ "Edit file",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Path of file to edit"
            },
            "oldString": {
                "type": "string",
                "description": "String to replace"
            },
            "newString": {
                "type": "string",
                "description": "New (replacement) value"
            }
        },
        "required": ["filename", "oldString", "newString"]
    })",
};

static common_chat_tool magic_tool{
    /* .name = */ "magic",
    /* .description = */ "Magic tool that takes a hash",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "ref": {
                "type": "string"
            }
        },
        "required": ["name", "ref"]
    })",
};

static common_chat_tool magic_int_tool{
    /* .name = */ "magic_int",
    /* .description = */ "Magic tool that takes a hash",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "ref": {
                "type": "integer"
            },
            "name": {
                "type": "string"
            }
        },
        "required": ["ref"]
    })",
};

static common_chat_tool amount_tool{
    /* .name = */ "amount",
    /* .description = */ "Amount converter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "orig": {
                "type": "number"
            }
        },
        "required": ["orig"]
    })",
};

static common_chat_tool toggle_tool{
    /* .name = */ "toggle",
    /* .description = */ "Toggle a feature",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "Whether to enable the feature"
            }
        },
        "required": ["enabled"]
    })",
};

static common_chat_tool nullable_tool{
    /* .name = */ "set_nullable",
    /* .description = */ "Set a nullable value",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "value": {
                "type": "null",
                "description": "A null value"
            }
        },
        "required": ["value"]
    })",
};

static common_chat_tool config_tool{
    /* .name = */ "set_config",
    /* .description = */ "Set configuration",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "description": "Configuration dict"
            }
        },
        "required": ["config"]
    })",
};

static common_chat_tool imaginary_number_tool{
    /* .name = */ "imaginary_number",
    /* .description = */ "Imaginary number converter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "number": {
                "type": "object",
                "properties": {
                    "real": {
                        "type": "number"
                    },
                    "imaginary": {
                        "type": "number"
                    }
                },
                "required": ["real", "imaginary"]
            }
        },
        "required": ["number"]
    })",
};

static common_chat_tool nullable_string_tool{
    /* .name = */ "set_nullable_str",
    /* .description = */ "Set a nullable string value",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "null"],
                "description": "A nullable string"
            }
        },
        "required": ["name"]
    })",
};

static common_chat_tool nullable_string_null_first_tool{
    /* .name = */ "set_nullable_str_nf",
    /* .description = */ "Set a nullable string value with null first in type array",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": {
                "type": ["null", "string"],
                "description": "A nullable string with null first"
            }
        },
        "required": ["name"]
    })",
};

static common_chat_tool nullable_int_tool{
    /* .name = */ "set_nullable_int",
    /* .description = */ "Set a nullable integer value",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "count": {
                "type": ["integer", "null"],
                "description": "A nullable integer"
            }
        },
        "required": ["count"]
    })",
};

static common_chat_tool enum_no_type_tool{
    /* .name = */ "set_unit",
    /* .description = */ "Set a temperature unit",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "unit": {
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["unit"]
    })",
};

static common_chat_tool string_param_tool{
    /* .name = */ "string_param",
    /* .description = */ "Tool with string parameter for testing",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "A text parameter"
            }
        },
        "required": []
    })",
};

static common_chat_tool quoted_unquoted_tool{
    /* .name = */ "quoted_unquoted",
    /* .description = */ "Tool with two string parameters, one for quoted string, one for unquoted",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "quoted": {
                "type": "string",
                "description": "Quoted value"
            },
            "unquoted": {
                "type": "string",
                "description": "Unquoted value"
            }
        },
        "required": ["quoted", "unquoted"]
    })",
};


static common_chat_tool tool_2req_4opt{
    /* .name = */ "tool_2req_4opt",
    /* .description = */ "Tool with 2 required and 4 optional params",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "req1": { "type": "string", "description": "Required string" },
            "req2": { "type": "integer", "description": "Required int" },
            "opt1": { "type": "string", "description": "Optional string 1" },
            "opt2": { "type": "integer", "description": "Optional int 1" },
            "opt3": { "type": "string", "description": "Optional string 2" },
            "opt4": { "type": "integer", "description": "Optional int 2" }
        },
        "required": ["req1", "req2"]
    })",
};

static common_chat_tool tool_2req_5opt{
    /* .name = */ "tool_2req_5opt",
    /* .description = */ "Tool with 2 required and 5 optional params",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "req1": { "type": "string", "description": "Required string" },
            "req2": { "type": "integer", "description": "Required int" },
            "opt1": { "type": "string", "description": "Optional string 1" },
            "opt2": { "type": "integer", "description": "Optional int 1" },
            "opt3": { "type": "string", "description": "Optional string 2" },
            "opt4": { "type": "integer", "description": "Optional int 2" },
            "opt5": { "type": "string", "description": "Optional string 3" }
        },
        "required": ["req1", "req2"]
    })",
};

static std::vector<common_chat_tool> tools{ special_function_tool, special_function_tool_with_optional_param,
                                            python_tool, html_tool, todo_list };

const common_chat_msg message_user{
    "user",
    "Hey there!",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

const common_chat_msg message_user_parts{
    "user",
    /* .content = */ "",
    /* .content_parts = */
    {
     { "text", "Hey" },
     { "text", "there" },
     },
    /* .tool_calls = */
    {                 },
    /* .reasoning_content = */
    "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

static common_chat_msg simple_assist_msg(const std::string & content,
                                         const std::string & reasoning_content = "",
                                         const std::string & tool_name         = "",
                                         const std::string & arguments         = "",
                                         const std::string & id                = "") {
    common_chat_msg msg;
    msg.role              = "assistant";
    msg.content           = content;
    msg.reasoning_content = reasoning_content;
    if (!tool_name.empty() || !id.empty()) {
        msg.tool_calls.push_back({ tool_name, arguments, id });
    }
    return msg;
}

static common_chat_msg message_with_tool_calls(const std::string & tool_name, const std::string & arguments) {
    return simple_assist_msg("", "", tool_name, arguments);
}

static common_chat_msg message_with_tool_calls_and_reasoning(const std::string & tool_name,
                                                             const std::string & arguments,
                                                             const std::string & reasoning) {
    return simple_assist_msg("", reasoning, tool_name, arguments);
}

static common_chat_msg message_with_reasoning_content_and_multiple_tool_calls(
    const std::string &                                      reasoning,
    const std::string &                                      content,
    const std::vector<std::pair<std::string, std::string>> & tool_calls) {
    common_chat_msg msg;
    msg.role              = "assistant";
    msg.content           = content;
    msg.reasoning_content = reasoning;
    for (const auto & [name, args] : tool_calls) {
        msg.tool_calls.push_back({ name, args, "" });
    }
    return msg;
}

static common_chat_msg message_with_content_and_tool_call(const std::string & content,
                                                          const std::string & tool_name,
                                                          const std::string & arguments) {
    return simple_assist_msg(content, "", tool_name, arguments);
}

static common_chat_msg message_with_reasoning_and_tool_call(const std::string & reasoning,
                                                            const std::string & tool_name,
                                                            const std::string & arguments) {
    return simple_assist_msg("", reasoning, tool_name, arguments);
}

const common_chat_msg message_assist       = simple_assist_msg("Hello, world!\nWhat's up?");
const common_chat_msg message_assist_empty = simple_assist_msg("");
const common_chat_msg message_assist_thoughts_unparsed_deepseek =
    simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_unparsed_md =
    simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```");
const common_chat_msg message_assist_thoughts_unparsed_md_partial =
    simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}");

const common_chat_msg message_assist_thoughts_unparsed_r7b =
    simple_assist_msg("<|START_THINKING|>I'm\nthinking<|END_THINKING|>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_unparsed_magistral =
    simple_assist_msg("[THINK]raisonnement[/THINK]Réponse");
const common_chat_msg message_assist_thoughts = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking");
const common_chat_msg message_assist_thoughts_unopened_unparsed =
    simple_assist_msg("I'm\nthinking</think>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_no_content = simple_assist_msg("", "I'm\nthinking");
const common_chat_msg message_assist_call = simple_assist_msg("", "", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_noopt =
    simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_withopt =
    simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}");
const common_chat_msg message_assist_call_content =
    simple_assist_msg("Hello, world!\nWhat's up?", "", "special_function", "{\"arg1\":1}");
const common_chat_msg message_assist_call_empty_args  = simple_assist_msg("", "", "special_function");
const common_chat_msg message_assist_call_cutoff_args = simple_assist_msg("", "", "special_function", "{\"arg");
const common_chat_msg message_assist_call_thoughts =
    simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\":1}");
const common_chat_msg message_assist_call_thoughts_unparsed =
    simple_assist_msg("<think>I'm\nthinking</think>\n\n", "", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_thoughts_content =
    simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_id =
    simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "123456789");
const common_chat_msg message_assist_call_idx =
    simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "0");
const common_chat_msg message_assist_thoughts_call_idx =
    simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}", /* id = */ "0");
const common_chat_msg message_assist_thoughts_partial_call =
    simple_assist_msg("", "I'm\nthinking", "special_function", "", /* id = */ "0");
const common_chat_msg message_assist_call_python = simple_assist_msg("", "", "python", "{\"code\":\"print('hey')\"}");
const common_chat_msg message_assist_call_python_lines =
    simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')\"}");
const common_chat_msg message_assist_call_python_lines_unclosed =
    simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')");
const common_chat_msg message_assist_json_content =
    simple_assist_msg("{\n  \"response\": \"Hello, world!\\nWhat's up?\"\n}");

// Use for PEG parser implementations
struct peg_test_case {
    common_chat_templates_inputs params;
    std::string                  input;
    common_chat_msg              expect;
    bool                         is_partial            = false;
    bool                         expect_reconstruction = false;
};

struct make_peg_parser {
    common_chat_params params_;
    common_peg_arena   arena_;
    bool               detailed_debug_;

    make_peg_parser(common_chat_templates *              tmpls,
                    const common_chat_templates_inputs & inputs,
                    bool                                 detailed_debug = false) {
        detailed_debug_ = detailed_debug;
        params_         = common_chat_templates_apply(tmpls, inputs);
        arena_.load(params_.parser);
    }

    common_chat_msg parse(const std::string & msg, bool is_partial) const {
        common_chat_parser_params parser_params(params_);
        parser_params.debug = detailed_debug_;
        return common_chat_peg_parse(arena_, msg, is_partial, parser_params);
    }
};

// Global template filter for --template flag
static std::string g_template_filter;

// When true, run reconstruction test on every non-partial test and report results
static bool g_force_reconstruction_test = false;

static void test_peg_parser(common_chat_templates *                      tmpls,
                            const std::function<void(peg_test_case &)> & init,
                            bool                                         detailed_debug) {
    // UTF-8-safe truncation helper (same as in test_parser_with_streaming)
    constexpr auto utf8_truncate_safe_len = [](const std::string_view s) -> size_t {
        auto len = s.size();
        if (len == 0) {
            return 0;
        }
        auto i = len;
        for (size_t back = 0; back < 4 && i > 0; ++back) {
            --i;
            unsigned char c = s[i];
            if ((c & 0x80) == 0) {
                return len;
            }
            if ((c & 0xC0) == 0xC0) {
                size_t expected_len = 0;
                if ((c & 0xE0) == 0xC0) {
                    expected_len = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    expected_len = 3;
                } else if ((c & 0xF8) == 0xF0) {
                    expected_len = 4;
                } else {
                    return i;
                }
                if (len - i >= expected_len) {
                    return len;
                }
                return i;
            }
        }
        return len - std::min(len, size_t(3));
    };

    peg_test_case tc;
    init(tc);
    if (tc.params.messages.empty()) {
        tc.params.messages = { message_user };
    }
    if (tc.expect.role.empty()) {
        tc.expect.role = "assistant";
    }

    auto parser = make_peg_parser(tmpls, tc.params, detailed_debug);
    if (detailed_debug) {
        LOG_DBG("Using parser: \n%s\n", parser.arena_.dump(parser.arena_.root()).c_str());
    }

    common_chat_msg msg_accum;
    common_chat_msg msg_prev;
    msg_accum.role = msg_prev.role = "assistant";

    for (size_t i = 1; i <= tc.input.size(); ++i) {
        auto            is_partial  = i < tc.input.size() || tc.is_partial;
        // Use UTF-8 safe truncation to avoid corrupting multi-byte characters
        size_t          safe_len    = utf8_truncate_safe_len(std::string_view(tc.input).substr(0, i));
        std::string     prefix      = tc.input.substr(0, safe_len);
        common_chat_msg msg_current = parser.parse(prefix, is_partial);

        for (const auto & diff : common_chat_msg_diff::compute_diffs(msg_prev, msg_current)) {
            if (!diff.reasoning_content_delta.empty()) {
                msg_accum.reasoning_content += diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                msg_accum.content += diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                // During partial parsing, a new tool call may appear with empty name initially
                // The name gets filled in as more input is parsed
                while (msg_accum.tool_calls.size() <= diff.tool_call_index) {
                    msg_accum.tool_calls.push_back({ "", "", "" });
                }
                // Always update name and id from diff (may change during incremental parsing), but only if the delta
                // actually contains them
                if (!diff.tool_call_delta.name.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].name = diff.tool_call_delta.name;
                }
                if (!diff.tool_call_delta.id.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].id = diff.tool_call_delta.id;
                }
                if (!diff.tool_call_delta.arguments.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].arguments += diff.tool_call_delta.arguments;
                }
            }
        }
        try {
            assert_msg_equals(msg_current, msg_accum, true);
        } catch (std::exception & e) {
            throw std::runtime_error((std::string("Error comparing accumulated message to current: ") + e.what()).c_str());
        }

        msg_prev = msg_current;
    }

    if (!tc.is_partial) {
        assert_msg_equals(tc.expect, parser.parse(tc.input, false), true);
    }
    assert_msg_equals(tc.expect, msg_accum, true);

    // Test grammar if present in params
    if (!parser.params_.grammar.empty()) {
        auto grammar = build_grammar(parser.params_.grammar);
        if (!grammar) {
            throw std::runtime_error("Failed to build grammar: " + parser.params_.grammar);
        }

        // In production, grammar triggers match against the full generated text
        // including the generation prompt. All positions are in full_input coordinates.
        const auto & gen_prompt = parser.params_.generation_prompt;
        std::string full_input = gen_prompt + tc.input;

        // Determine whether the reasoning-budget sampler path applies: tool-call grammar
        // with all WORD triggers and thinking tags present. In production, the reasoning
        // budget sampler inhibits grammar application while inside thinking blocks —
        // triggers inside <think>...</think> are suppressed.
        bool use_reasoning_budget_path = false;
        if (parser.params_.grammar_lazy && !parser.params_.thinking_end_tag.empty()) {
            use_reasoning_budget_path = true;
            for (const auto & trigger : parser.params_.grammar_triggers) {
                if (trigger.type != COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    use_reasoning_budget_path = false;
                    break;
                }
            }
        }

        // Find the earliest trigger position to determine the constrained portion
        auto earliest_trigger_pos = std::string::npos;

        if (use_reasoning_budget_path) {
            // Reasoning-budget path: simulate thinking-aware trigger detection.
            // Walk through full_input tracking thinking state; only match triggers
            // when outside thinking blocks.
            const auto & think_start = parser.params_.thinking_start_tag;
            const auto & think_end   = parser.params_.thinking_end_tag;

            bool in_thinking = false;
            for (size_t i = 0; i < full_input.size(); ++i) {
                if (!in_thinking && !think_start.empty()
                        && full_input.compare(i, think_start.size(), think_start) == 0) {
                    in_thinking = true;
                    i += think_start.size() - 1;
                    continue;
                }
                if (in_thinking && full_input.compare(i, think_end.size(), think_end) == 0) {
                    in_thinking = false;
                    i += think_end.size() - 1;
                    continue;
                }
                if (in_thinking) {
                    continue;
                }
                // Outside thinking — check if any trigger word starts here
                for (const auto & trigger : parser.params_.grammar_triggers) {
                    if (full_input.compare(i, trigger.value.size(), trigger.value) == 0) {
                        if (earliest_trigger_pos == std::string::npos || i < earliest_trigger_pos) {
                            earliest_trigger_pos = i;
                        }
                    }
                }
                if (earliest_trigger_pos != std::string::npos) {
                    break;  // found the earliest
                }
            }

            // If the reasoning-budget path found no trigger outside thinking but the test
            // expects tool calls, this template nests tool calls inside thinking
            // blocks (e.g. Kimi). Fall back to the legacy path for this case.
            if (earliest_trigger_pos == std::string::npos && !tc.expect.tool_calls.empty()) {
                use_reasoning_budget_path = false;
            }
        }

        if (!use_reasoning_budget_path) {
            // Legacy path: find triggers without thinking-awareness
            for (const auto & trigger : parser.params_.grammar_triggers) {
                size_t      pos = std::string::npos;
                std::smatch match;
                switch (trigger.type) {
                    case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                        {
                            const auto & word = trigger.value;
                            pos               = full_input.find(word);
                            break;
                        }
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                        {
                            const auto & compiled = std::regex(trigger.value);
                            if (std::regex_search(full_input, match, compiled)) {
                                pos = match.position(compiled.mark_count());
                            }
                            break;
                        }
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                        {
                            // In production, PATTERN_FULL triggers are checked against
                            // the text generated so far, growing token by token. Simulate
                            // by trying every prefix of full_input.
                            const auto & compiled = std::regex(trigger.value);
                            for (size_t end = gen_prompt.size(); end <= full_input.size(); ++end) {
                                std::string prefix = full_input.substr(0, end);
                                if (std::regex_match(prefix, match, compiled)) {
                                    pos = std::string::npos;
                                    for (size_t gi = 1; gi < match.size(); ++gi) {
                                        if (match[gi].length() > 0) {
                                            pos = match.position(gi);
                                            break;
                                        }
                                    }
                                    if (pos == std::string::npos) {
                                        pos = match.position(0);
                                    }
                                    break;
                                }
                            }
                            break;
                        }
                    default:
                        throw std::runtime_error("Unknown trigger type");
                }
                if (pos != std::string::npos) {
                    if (earliest_trigger_pos == std::string::npos || pos < earliest_trigger_pos) {
                        earliest_trigger_pos = pos;
                    }
                }
            }
        }

        // If the test expects tool calls and the grammar is lazy, the trigger must fire.
        // Otherwise the grammar would never activate in production and tool calls wouldn't
        // be constrained. A silent skip here would hide broken triggers.
        if (parser.params_.grammar_lazy && !tc.expect.tool_calls.empty() && !tc.is_partial
                && earliest_trigger_pos == std::string::npos) {
            std::string trigger_desc;
            for (const auto & trigger : parser.params_.grammar_triggers) {
                trigger_desc += "\n  [type=" + std::to_string(trigger.type) + "] " + trigger.value;
            }
            throw std::runtime_error(
                "Grammar trigger did not fire, but test expects tool calls (lazy grammar).\n"
                ">>> Input: " + full_input + "\n"
                ">>> Triggers (" + std::to_string(parser.params_.grammar_triggers.size()) + "):" + trigger_desc);
        }

        // Determine the constrained portion of input to test against grammar.
        // If the trigger position falls inside the generation prompt, the grammar
        // sampler was already active before model output began — constrain from the
        // start of the model output (i.e. tc.input).
        std::string constrained = full_input;
        bool grammar_triggered = false;
        if (earliest_trigger_pos != std::string::npos) {
            auto constrain_from = std::max(earliest_trigger_pos, gen_prompt.size());
            constrained = full_input.substr(constrain_from);
            grammar_triggered = true;
        } else if (!parser.params_.grammar_lazy) {
            // For non-lazy grammars, the entire input should match
            grammar_triggered = true;
        }

        // Test the constrained portion against the grammar
        if (grammar_triggered && !tc.is_partial) {
            auto result = match_string_detailed(constrained, grammar.get());
            if (!result.success) {
                std::string error_msg;
                if (result.incomplete) {
                    error_msg =
                        "Grammar matched all input but expects more:\n\n"
                        ">>> Input: " + tc.input +
                        "\n\n>>> Constrained: " + constrained +
                        "\n\n>>> Matched prefix (" + std::to_string(result.matched_bytes) + " bytes, " +
                        std::to_string(result.matched_codepoints) + " codepoints): " +
                        (result.matched_prefix.size() > 100 ? result.matched_prefix.substr(0, 100) + "..." : result.matched_prefix) +
                        "\n\n>>> Expected next: " + result.expected_description +
                        "\n\n>>> Grammar: " + parser.params_.grammar;
                } else {
                    error_msg =
                        "Grammar match failed:\n\n"
                        ">>> Input: " + tc.input +
                        "\n\n>>> Constrained: " + constrained +
                        "\n\n>>> Matched prefix (" + std::to_string(result.matched_bytes) + " bytes, " +
                        std::to_string(result.matched_codepoints) + " codepoints): " +
                        (result.matched_prefix.size() > 100 ? result.matched_prefix.substr(0, 100) + "..." : result.matched_prefix) +
                        "\n\n>>> Failing character: " + result.failing_char +
                        "\n\n>>> Expected: " + result.expected_description +
                        "\n\n>>> Grammar: " + parser.params_.grammar;
                }
                throw std::runtime_error(error_msg);
            }
        }
    }

    // Reconstruction test: verify that appending the parsed message to the original
    // messages and re-rendering the template (without generation prompt) reproduces
    // the original prompt + input exactly, or as a proper prefix (the template may
    // append end-of-turn tokens after the assistant message).
    if ((tc.expect_reconstruction || g_force_reconstruction_test) && !tc.is_partial) {
        // Start from tc.expect but copy tool call arguments from the actual parser
        // output, which preserves original JSON formatting (e.g. {"arg1":1} vs {"arg1": 1}).
        auto reconstruction_msg = tc.expect;
        auto parsed_msg         = parser.parse(tc.input, false);
        for (size_t i = 0; i < reconstruction_msg.tool_calls.size() && i < parsed_msg.tool_calls.size(); i++) {
            reconstruction_msg.tool_calls[i].arguments = parsed_msg.tool_calls[i].arguments;
        }
        common_chat_templates_inputs reconstruction_inputs = tc.params;
        reconstruction_inputs.messages.push_back(reconstruction_msg);
        reconstruction_inputs.add_generation_prompt = false;

        auto reconstruction_params = common_chat_templates_apply(tmpls, reconstruction_inputs);
        std::string expected_text  = parser.params_.prompt + tc.input;
        bool match = reconstruction_params.prompt == expected_text ||
            (reconstruction_params.prompt.size() > expected_text.size() &&
             reconstruction_params.prompt.compare(0, expected_text.size(), expected_text) == 0);
        if (!match && g_force_reconstruction_test && !tc.expect_reconstruction) {
            // In forced mode, report mismatch but don't fail
            // Find the first difference position
            size_t diff_pos = 0;
            size_t min_len  = std::min(expected_text.size(), reconstruction_params.prompt.size());
            while (diff_pos < min_len && expected_text[diff_pos] == reconstruction_params.prompt[diff_pos]) {
                diff_pos++;
            }
            size_t ctx_start = diff_pos > 60 ? diff_pos - 60 : 0;
            size_t ctx_end_e = std::min(expected_text.size(), diff_pos + 40);
            size_t ctx_end_r = std::min(reconstruction_params.prompt.size(), diff_pos + 40);
            LOG_ERR("\x1b[31m[RECONSTRUCTION FAIL]\x1b[0m "
                    "first diff at byte %zu (expected len=%zu, reconstructed len=%zu)\n"
                    "  expected:      ...%s...\n"
                    "  reconstructed: ...%s...\n",
                    diff_pos, expected_text.size(), reconstruction_params.prompt.size(),
                    expected_text.substr(ctx_start, ctx_end_e - ctx_start).c_str(),
                    reconstruction_params.prompt.substr(ctx_start, ctx_end_r - ctx_start).c_str());
        } else if (!match) {
            std::string error_msg =
                "Reconstruction mismatch:\n\n"
                ">>> Expected (prompt + input):\n" + expected_text +
                "\n\n>>> Reconstructed:\n" + reconstruction_params.prompt;
            throw std::runtime_error(error_msg);
        } else if (g_force_reconstruction_test) {
            LOG_INF("\x1b[32m[RECONSTRUCTION OK]\x1b[0m\n");
        }
    }
}

// Fluent builder for PEG parser tests
class peg_test_builder;

class peg_tester {
    common_chat_templates_ptr tmpls_;
    std::string               template_path_;
    bool                      detailed_debug_;
    friend class peg_test_builder;

  public:
    explicit peg_tester(const std::string & template_path, const bool detailed_debug = false) :
        tmpls_(read_templates(template_path)),
        template_path_(template_path),
        detailed_debug_(detailed_debug) {}

    const std::string & template_path() const { return template_path_; }

    peg_test_builder test(const std::string & input);
};

class peg_test_builder {
    peg_tester &  tester_;
    peg_test_case tc_;

  public:
    peg_test_builder(peg_tester & tester, const std::string & input) : tester_(tester) { tc_.input = input; }

    // Parameter setters
    peg_test_builder & reasoning_format(common_reasoning_format fmt) {
        tc_.params.reasoning_format = fmt;
        return *this;
    }

    peg_test_builder & tools(std::vector<common_chat_tool> tools) {
        tc_.params.tools = std::move(tools);
        return *this;
    }

    peg_test_builder & enable_thinking(bool val) {
        tc_.params.enable_thinking = val;
        return *this;
    }

    peg_test_builder & parallel_tool_calls(bool val) {
        tc_.params.parallel_tool_calls = val;
        return *this;
    }

    peg_test_builder & json_schema(const std::string & schema) {
        tc_.params.json_schema = schema;
        return *this;
    }

    peg_test_builder & is_partial(bool val) {
        tc_.is_partial = val;
        return *this;
    }

    peg_test_builder & expect_reconstruction(bool val = true) {
        tc_.expect_reconstruction = val;
        return *this;
    }

    // Expect setters
    peg_test_builder & expect(const common_chat_msg & msg) {
        tc_.expect = msg;
        return *this;
    }

    peg_test_builder & expect_content(const std::string & content) {
        tc_.expect.content = content;
        return *this;
    }

    peg_test_builder & expect_reasoning(const std::string & reasoning) {
        tc_.expect.reasoning_content = reasoning;
        return *this;
    }

    peg_test_builder & expect_tool_calls(std::vector<common_chat_tool_call> calls) {
        tc_.expect.tool_calls = std::move(calls);
        return *this;
    }

    // Execute the test
    void run() {
        // Check template filter
        if (!g_template_filter.empty()) {
            // Case-insensitive substring match
            std::string template_path_lower = tester_.template_path();
            std::string filter_lower        = g_template_filter;
            std::transform(template_path_lower.begin(), template_path_lower.end(), template_path_lower.begin(),
                           ::tolower);
            std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
            if (template_path_lower.find(filter_lower) == std::string::npos) {
                // Skip this test
                return;
            }
        }
        LOG_INF("\n\x1b[38;5;126m[%s]\x1b[0m\n%s\n\n", tester_.template_path().c_str(), tc_.input.c_str());
        test_peg_parser(tester_.tmpls_.get(), [this](peg_test_case & t) { t = tc_; }, tester_.detailed_debug_);
    }
};

peg_test_builder peg_tester::test(const std::string & input) {
    return peg_test_builder(*this, input);
}

static void test_msgs_oaicompat_json_conversion() {
    LOG_DBG("%s\n", __func__);
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
    };
    for (const auto & msg : msgs) {
        auto oai_json = common_chat_msgs_to_json_oaicompat({ msg });
        auto msgs2    = common_chat_msgs_parse_oaicompat(oai_json);
        assert_equals((size_t) 1, msgs2.size());
        const auto & msg2 = msgs2[0];
        assert_msg_equals(msg, msg2);
    }
    assert_equals(std::string("[\n"
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
                              "]"),
                  common_chat_msgs_to_json_oaicompat({ message_user_parts }).dump(2));

    // Note: content is "" instead of null due to workaround for templates that render null as "None"
    assert_equals(std::string("[\n"
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
                              "]"),
                  common_chat_msgs_to_json_oaicompat({ message_assist_call_python }).dump(2));

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
    LOG_DBG("%s\n", __func__);
    std::vector<common_chat_tool> tools{
        special_function_tool,
        python_tool,
    };

    for (const auto & tool : tools) {
        auto oai_json = common_chat_tools_to_json_oaicompat({ tool });
        auto tools2   = common_chat_tools_parse_oaicompat(oai_json);
        assert_equals((size_t) 1, tools2.size());
        auto tool2 = tools2[0];
        assert_equals(tool.name, tool2.name);
        assert_equals(tool.description, tool2.description);
        assert_equals(json::parse(tool.parameters).dump(2), json::parse(tool2.parameters).dump(2));
    }

    assert_equals(std::string("[\n"
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
                              "]"),
                  common_chat_tools_to_json_oaicompat({ special_function_tool }).dump(2));
}

static void test_template_output_peg_parsers(bool detailed_debug) {
    LOG_DBG("%s\n", __func__);

    // JSON schemas
    const char * invoice_schema = R"({
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "date": {"type": "string"}
        }
    })";

    {
        // Qwen3.5 (basically same as Nemotron, but keeping separate tests just in case)
        auto tst = peg_tester("models/templates/Qwen3.5-4B.jinja", detailed_debug);

        tst.test("I'm\nthinking</think>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .expect(message_assist_thoughts)
            .run();

                tst.test("I'm\nthinking\n</think>\nHello, world!\nWhat's up?")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_NONE)
            .expect_content("<think>\nI'm\nthinking\n</think>\nHello, world!\nWhat's up?")
            .run();

        tst.test("I'm\nthinking\n</think>\nHello, world!\nWhat's up?")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist_thoughts)
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        tst.test(
               "I'm\nthinking\n</think>\n"
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>\n"
               "<tool_call>\n"
               "<function=special_function_with_opt>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "<parameter=arg2>\n2\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Hello, world!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({
                python_tool
        })
            .expect_tool_calls({
                { "python", "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}", {} },
            })
            .run();

        tst.test(
               "I need to output the invoice details in JSON\n"
               "</think>\n"
               R"({"amount": 123.45, "date": "2025-12-03"})")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .json_schema(invoice_schema)
            .expect_reasoning("I need to output the invoice details in JSON")
            .expect_content(R"({"amount": 123.45, "date": "2025-12-03"})")
            .run();

        // tool call segment in reasoning
        tst.test(
               "Let's call a tool: <tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Not the real call!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call></think>\n"
               "<tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Hello, world!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>"
            )
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({
                python_tool
        })
            .expect_reasoning("Let's call a tool: <tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Not the real call!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .expect_tool_calls({
                { "python", "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}", {} },
            })
            .run();

        // No args tool
        tst.test(
               "<tool_call>\n"
               "<function=empty_args>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ empty_args_tool })
            .expect(message_with_tool_calls("empty_args", "{}"))
            .run();

        // No args tool with no properties defined
        tst.test(
               "<tool_call>\n"
               "<function=empty_args_no_props>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ empty_args_tool_no_properties })
            .expect(message_with_tool_calls("empty_args_no_props", "{}"))
            .run();
    }

    {
        // Ministral-3-14B-Reasoning-2512
        auto tst = peg_tester("models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();

        tst.test("[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?")
            .expect_content("[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?")
            .expect_reconstruction()
            .run();

        tst.test("[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .expect(message_assist_thoughts)
            .expect_reconstruction()
            .run();

        tst.test(R"([TOOL_CALLS]special_function[ARGS]{"arg1":1})")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        tst.test(
               "[THINK]I'm\nthinking[/THINK]"
               R"([TOOL_CALLS]special_function[ARGS]{"arg1":1})")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .run();

        tst.test(R"([TOOL_CALLS]special_function[ARGS]{"arg1": 1})"
                 R"([TOOL_CALLS]special_function_with_opt[ARGS]{"arg1": 1, "arg2": 2})")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .expect_reconstruction()
            .run();

        tst.test(
               "[THINK]I need to output the invoice details in JSON[/THINK]"
               "```json\n"
               R"({"amount": 123.45, "date": "2025-12-03"})"
               "\n```")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .json_schema(invoice_schema)
            .expect_reasoning("I need to output the invoice details in JSON")
            .expect_content(R"({"amount": 123.45, "date": "2025-12-03"})")
            .run();

        // fake tool call marker in reasoning
        tst.test(
               "[THINK]Let me think about [TOOL_CALLS]special_function[ARGS]{\"arg1\":1} and more[/THINK]"
               R"([TOOL_CALLS]special_function[ARGS]{"arg1": 1})")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect_reasoning("Let me think about [TOOL_CALLS]special_function[ARGS]{\"arg1\":1} and more")
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
            })
            .expect_reconstruction()
            .run();
    }

    {
        // NVIDIA Nemotron-3 Nano
        auto tst = peg_tester("models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").
            enable_thinking(false).
            reasoning_format(COMMON_REASONING_FORMAT_AUTO).
            expect(message_assist).run();

        tst.test("I'm\nthinking\n</think>\nHello, world!\nWhat's up?")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_NONE)
            .expect_content("<think>\nI'm\nthinking\n</think>\nHello, world!\nWhat's up?")
            .run();

        tst.test("I'm\nthinking\n</think>\nHello, world!\nWhat's up?")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist_thoughts)
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        tst.test(
               "I'm\nthinking\n</think>\n"
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>\n"
               "<tool_call>\n"
               "<function=special_function_with_opt>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "<parameter=arg2>\n2\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Hello, world!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({
                python_tool
        })
            .expect_tool_calls({
                { "python", "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}", {} },
            })
            .run();

        tst.test(
               "I need to output the invoice details in JSON\n"
               "</think>\n"
               R"({"amount": 123.45, "date": "2025-12-03"})")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .json_schema(invoice_schema)
            .expect_reasoning("I need to output the invoice details in JSON")
            .expect_content(R"({"amount": 123.45, "date": "2025-12-03"})")
            .run();

        // tool call segment in reasoning
        tst.test(
               "Let's call a tool: <tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Not the real call!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call></think>\n"
               "<tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Hello, world!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>"
            )
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({
                python_tool
        })
            .expect_reasoning("Let's call a tool: <tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Not the real call!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .expect_tool_calls({
                { "python", "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}", {} },
            })
            .run();

    }

    {
        // CohereForAI Command-R 7B (2024-tool_use)
        auto tst = peg_tester("models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja", detailed_debug);

        tst.test("<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>").expect(message_assist).run();

        tst.test(
               "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
               "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist_thoughts)
            .run();

        tst.test(
               "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
               "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>")
            .expect(message_assist_thoughts_unparsed_r7b)
            .run();

        tst.test(
               "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
               "<|START_ACTION|>[\n"
               "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
               "]<|END_ACTION|>")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool })
            .expect(message_assist_thoughts_call_idx)
            .run();

        tst.test(
               "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
               "<|START_ACTION|>[\n"
               "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", ")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool })
            .is_partial(true)
            .expect(message_assist_thoughts_partial_call)
            .run();

        tst.test(
               "<|START_THINKING|><|END_THINKING|>"
               "<|START_ACTION|>[\n"
               "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
               "]<|END_ACTION|>")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool })
            .expect(message_assist_call_idx)
            .run();
    }

    {
        // Google Gemma 2 2B - does not support tool calling
        auto tst = peg_tester("models/templates/google-gemma-2-2b-it.jinja");

        tst.test("Hello, world!").expect(simple_assist_msg("Hello, world!")).expect_reconstruction().run();

        tst.test("Line 1\nLine 2\nLine 3").expect(simple_assist_msg("Line 1\nLine 2\nLine 3")).expect_reconstruction().run();
    }

    {
        // Google Gemma 4 (tool calling with Gemma4 dict format)
        auto tst = peg_tester("models/templates/google-gemma-4-31B-it.jinja");

        tst.test("Hello, world!").expect(simple_assist_msg("Hello, world!")).run();

        // Reasoning and content
        tst.test(
                "<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist_thoughts)
            .run();

        // Reasoning and content with reasoning_format = none
        tst.test(
                "<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_NONE)
            .expect_content("<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?")
            .run();

        // Simple tool call with string argument
        tst.test(
                "<|tool_call>call:get_time{city:<|\"|>London<|\"|>}<tool_call|>")
            .tools({ get_time_tool })
            .expect(message_with_tool_calls("get_time", R"({"city": "London"})"))
            .run();

        // Tool call with string argument containing special chars
        tst.test(
                "<|tool_call>call:get_time{city:<|\"|>San Francisco<|\"|>}<tool_call|>")
            .tools({ get_time_tool })
            .expect(message_with_tool_calls("get_time", R"({"city": "San Francisco"})"))
            .run();

        // Tool call with empty args
        tst.test(
                "<|tool_call>call:empty_args{}<tool_call|>")
            .tools({ empty_args_tool })
            .expect(message_with_tool_calls("empty_args", "{}"))
            .run();

        // Tool call with string and content
        tst.test(
                "Hello, world!\nWhat's up?<|tool_call>call:get_time{city:<|\"|>Paris<|\"|>}<tool_call|>")
            .tools({ get_time_tool })
            .expect(message_with_content_and_tool_call("Hello, world!\nWhat's up?", "get_time", R"({"city": "Paris"})"))
            .run();

        // Parallel tool calls
        tst.test(
                "<|tool_call>call:get_time{city:<|\"|>London<|\"|>}<tool_call|>"
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>")
            .tools({ get_time_tool, get_weather_tool })
            .parallel_tool_calls(true)
            .expect_tool_calls({
                { "get_time", R"({"city": "London"})", "" },
                { "get_weather", R"({"city": "Paris"})", "" },
            })
            .run();

        // Tool call with integer argument (number type)
        tst.test(
                "<|tool_call>call:special_function{arg1:42}<tool_call|>")
            .tools({ special_function_tool })
            .expect(message_with_tool_calls("special_function", R"({"arg1": 42})"))
            .run();

        // Tool call with negative number argument
        tst.test(
                "<|tool_call>call:special_function{arg1:-7}<tool_call|>")
            .tools({ special_function_tool })
            .expect(message_with_tool_calls("special_function", R"({"arg1": -7})"))
            .run();

        // Tool call with decimal number argument
        tst.test(
                "<|tool_call>call:amount{orig:3.14}<tool_call|>")
            .tools({ amount_tool })
            .expect(message_with_tool_calls("amount", R"({"orig": 3.14})"))
            .run();

        // Tool call with boolean argument (true)
        tst.test(
                "<|tool_call>call:toggle{enabled:true}<tool_call|>")
            .tools({ toggle_tool })
            .expect(message_with_tool_calls("toggle", R"({"enabled": true})"))
            .run();

        // Tool call with boolean argument (false)
        tst.test(
                "<|tool_call>call:toggle{enabled:false}<tool_call|>")
            .tools({ toggle_tool })
            .expect(message_with_tool_calls("toggle", R"({"enabled": false})"))
            .run();

        // Tool call with null argument
        tst.test(
                "<|tool_call>call:set_nullable{value:null}<tool_call|>")
            .tools({ nullable_tool })
            .expect(message_with_tool_calls("set_nullable", R"({"value": null})"))
            .run();

        // Tool call with array argument (todo list)
        tst.test(
                "<|tool_call>call:todo_list{todos:[<|\"|>buy milk<|\"|>,<|\"|>walk dog<|\"|>]}<tool_call|>")
            .tools({ todo_list })
            .expect(message_with_tool_calls("todo_list", R"({"todos":["buy milk","walk dog"]})"))
            .run();

        // Tool call with object/dict argument
        tst.test(
                "<|tool_call>call:set_config{config:{theme:<|\"|>dark<|\"|>,count:3}}<tool_call|>")
            .tools({ config_tool })
            .expect(message_with_tool_calls("set_config", R"({"config":{"theme":"dark","count":3}})"))
            .run();

        // Tool call with empty array
        tst.test(
                "<|tool_call>call:todo_list{todos:[]}<tool_call|>")
            .tools({ todo_list })
            .expect(message_with_tool_calls("todo_list", R"({"todos":[]})"))
            .run();

        // Tool call with empty dict
        tst.test(
                "<|tool_call>call:set_config{config:{}}<tool_call|>")
            .tools({ config_tool })
            .expect(message_with_tool_calls("set_config", R"({"config":{}})"))
            .run();

        // Tool call with scientific notation number
        tst.test(
                "<|tool_call>call:amount{orig:1.5e10}<tool_call|>")
            .tools({ amount_tool })
            .expect(message_with_tool_calls("amount", R"({"orig": 1.5e10})"))
            .run();
    }

    {
        // Qwen-QwQ-32B (reasoning model)
        auto tst = peg_tester("models/templates/Qwen-QwQ-32B.jinja");

        // QwQ always has thinking forced open - input starts after the <think>\n in the prompt
        tst.test("Let me think about this...\n</think>\nThe answer is 42.")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(simple_assist_msg("The answer is 42.", "Let me think about this..."))
            .run();

        tst.test("</think>Hello, world!").reasoning_format(COMMON_REASONING_FORMAT_AUTO).expect(simple_assist_msg("Hello, world!")).run();
    }
    {
        // NousResearch-Hermes-2-Pro and Hermes-3 (tool calling models)
        auto tst = peg_tester("models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja", detailed_debug);

        tst.test(
               "<tool_call>\n"
               "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
               "</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        tst.test(
               "Hello, world!\nWhat's up?<tool_call>\n"
               "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
               "</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call_content)
            .run();

        // Note: Hermes template doesn't support thinking/reasoning natively
        // Note: We only support one tool calling format per template, no alternate formats
    }
    {
        // Test simple content-only template
        auto tst = peg_tester("models/templates/google-gemma-2-2b-it.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
    }
    {
        // IBM Granite (reasoning and tool calling model)
        auto tst = peg_tester("models/templates/ibm-granite-granite-3.3-2B-Instruct.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        tst.test("<think>I'm\nthinking</think>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist_thoughts)
            .run();

        // TODO: pending support for WRAPPED_WITH_REASONING
        // tst.test("<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>")
        //     .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
        //     .expect(message_assist_thoughts)
        //     .run();
    }

    {
        // IBM Granite 4.0 (production template shared by h-tiny, h-small, micro)
        // Uses <tool_call> XML tags for tool calls, tools in system message
        auto tst = peg_tester("models/templates/ibm-granite-granite-4.0.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        tst.test(
               "<tool_call>\n"
               "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
               "</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();
    }

    {
        // ByteDance-Seed-OSS (reasoning and tool calling model)
        auto tst = peg_tester("models/templates/ByteDance-Seed-OSS.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        tst.test("<seed:think>I'm thinking about the answer</seed:think>\nHello, world!")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(simple_assist_msg("Hello, world!", "I'm thinking about the answer"))
            .run();

        tst.test(
               "<seed:tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>1</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        tst.test(
               "<seed:tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>1</parameter>\n"
               "</function>\n"
               "</seed:tool_call>\n"
               "<seed:tool_call>\n"
               "<function=special_function_with_opt>\n"
               "<parameter=arg1>1</parameter>\n"
               "<parameter=arg2>2</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .run();

        tst.test(
               "<seed:tool_call>\n"
               "<function=todo_list>\n"
               "<parameter=todos>[{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .tools({
                todo_list
        })
            .expect_tool_calls({
                { "todo_list", "{\"todos\": [{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]}", {} },
            })
            .run();

        // tool call with inside quotes
        tst.test(
               "<seed:tool_call>\n"
               "<function=edit>\n"
               "<parameter=filename>\n"
               "foo.cpp\n"
               "</parameter>\n"
               "<parameter=oldString>"
               "def foo(arg = \"14\"):\n"
               "    return arg + \"bar\"\n"
               "\n"
               "</parameter>\n"
               "<parameter=newString>"
               "def foo(arg = \"15\"):\n"
               "    pass\n"
               "\n"
               "</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .tools({
                edit_tool
        })
            .expect_tool_calls({
                { "edit", "{\"filename\": \"foo.cpp\", "
                    "\"oldString\": \"def foo(arg = \\\"14\\\"):\\n    return arg + \\\"bar\\\"\\n\", "
                    "\"newString\": \"def foo(arg = \\\"15\\\"):\\n    pass\\n\"}", {}
                }
            })
            .run();

    }

    {
        // Qwen3-Coder (tool calling with XML-style format)
        auto tst = peg_tester("models/templates/Qwen3-Coder.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();

        tst.test(
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n"
               "1\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();

        tst.test(
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
               "</tool_call>")
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .expect_reconstruction()
            .run();

        // Test with code content (multiline)
        tst.test(
               "<tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "def hello():\n"
               "    print(\"Hello, world!\")\n"
               "\n"
               "hello()\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({
                python_tool
        })
            .expect_tool_calls({
                { "python", "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}", {} },
            })
            .expect_reconstruction()
            .run();

        // Test with code content (asian unicode chars)
        tst.test(
               "<tool_call>\n"
               "<function=python>\n"
               "<parameter=code>\n"
               "格\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({
                python_tool
        })
            .expect_tool_calls({
                { "python", "{\"code\": \"格\"}", {} },
            })
            .expect_reconstruction()
            .run();

        // Test with HTML tag content
        tst.test(
               "<tool_call>\n"
               "<function=html>\n"
               "<parameter=markup>\n"
               "<html>\n"
               " <head>\n"
               "  <title>Hello!</title>\n"
               " </head>\n"
               "</html>\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({
                html_tool
        })
            .expect_tool_calls({
                { "html", "{\"markup\": \"<html>\\n <head>\\n  <title>Hello!</title>\\n </head>\\n</html>\"}", {} },
            })
            .expect_reconstruction()
            .run();

        // Test with TODO list (array of objects)
        tst.test(
               "<tool_call>\n"
               "<function=todo_list>\n"
               "<parameter=todos>\n"
               "[{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({
                todo_list
        })
            .expect_tool_calls({
                { "todo_list", "{\"todos\": [{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]}", {} },
            })
            .expect_reconstruction()
            .run();

        // Test flexible optional argument ordering (2 required + 4 optional, reversed optional order)
        tst.test(
               "<tool_call>\n"
               "<function=tool_2req_4opt>\n"
               "<parameter=req1>\nhello\n</parameter>\n"
               "<parameter=req2>\n42\n</parameter>\n"
               "<parameter=opt4>\n100\n</parameter>\n"
               "<parameter=opt2>\n200\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ tool_2req_4opt })
            .expect_tool_calls({
                { "tool_2req_4opt", R"({"req1": "hello", "req2": 42, "opt4": 100, "opt2": 200})", {} },
            })
            .expect_reconstruction()
            .run();

        // Test flexible optional argument ordering (2 required + 5 optional, reversed optional order)
        tst.test(
               "<tool_call>\n"
               "<function=tool_2req_5opt>\n"
               "<parameter=req1>\nworld\n</parameter>\n"
               "<parameter=req2>\n7\n</parameter>\n"
               "<parameter=opt5>\nlast\n</parameter>\n"
               "<parameter=opt3>\nmiddle\n</parameter>\n"
               "<parameter=opt1>\nfirst\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ tool_2req_5opt })
            .expect_tool_calls({
                { "tool_2req_5opt", R"({"req1": "world", "req2": 7, "opt5": "last", "opt3": "middle", "opt1": "first"})", {} },
            })
            .expect_reconstruction()
            .run();

        // Test flexible optional argument ordering (2 required + 5 optional, all 5 in shuffled order)
        tst.test(
               "<tool_call>\n"
               "<function=tool_2req_5opt>\n"
               "<parameter=req1>\ntest\n</parameter>\n"
               "<parameter=req2>\n99\n</parameter>\n"
               "<parameter=opt3>\nc\n</parameter>\n"
               "<parameter=opt1>\na\n</parameter>\n"
               "<parameter=opt5>\ne\n</parameter>\n"
               "<parameter=opt4>\n4\n</parameter>\n"
               "<parameter=opt2>\n2\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ tool_2req_5opt })
            .expect_tool_calls({
                { "tool_2req_5opt", R"({"req1": "test", "req2": 99, "opt3": "c", "opt1": "a", "opt5": "e", "opt4": 4, "opt2": 2})", {} },
            })
            .expect_reconstruction()
            .run();

        // nullable string type ["string", "null"]
        tst.test(
               "<tool_call>\n"
               "<function=set_nullable_str>\n"
               "<parameter=name>\nhello world\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ nullable_string_tool })
            .expect_tool_calls({
                { "set_nullable_str", R"({"name": "hello world"})", {} },
            })
            .run();

        // nullable string with null first in type array ["null", "string"]
        tst.test(
               "<tool_call>\n"
               "<function=set_nullable_str_nf>\n"
               "<parameter=name>\nhello world\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ nullable_string_null_first_tool })
            .expect_tool_calls({
                { "set_nullable_str_nf", R"({"name": "hello world"})", {} },
            })
            .run();

        // nullable integer type ["integer", "null"] - should use JSON value path, not string
        tst.test(
               "<tool_call>\n"
               "<function=set_nullable_int>\n"
               "<parameter=count>\n42\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ nullable_int_tool })
            .expect_tool_calls({
                { "set_nullable_int", R"({"count": 42})", {} },
            })
            .run();

        // enum without explicit type key - should infer string from enum values
        tst.test(
               "<tool_call>\n"
               "<function=set_unit>\n"
               "<parameter=unit>\ncelsius\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .tools({ enum_no_type_tool })
            .expect_tool_calls({
                { "set_unit", R"({"unit": "celsius"})", {} },
            })
            .run();
    }
    {
        auto tst = peg_tester("models/templates/deepseek-ai-DeepSeek-V3.1.jinja", detailed_debug);
        tst.test(
               "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": "
               "\"XYZCITY\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
            .tools({ get_time_tool })
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_with_tool_calls("get_time", "{\"city\":\"XYZCITY\"}"))
            .run();
    }

    {
        auto tst = peg_tester("models/templates/deepseek-ai-DeepSeek-V3.1.jinja", detailed_debug);
        tst.test(
               "REASONING</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": "
               "\"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ get_time_tool })
            .expect(message_with_tool_calls_and_reasoning("get_time", "{\"city\":\"Tokyo\"}", "REASONING"))
            .run();
    }

    {
        auto tst = peg_tester("models/templates/deepseek-ai-DeepSeek-V3.1.jinja", detailed_debug);
        tst.test(
               "REASONING</think>CONTENT<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": "
               "\"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": "
               "\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
            .tools({
                get_time_tool, get_weather_tool
        })
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .parallel_tool_calls(true)
            .expect(message_with_reasoning_content_and_multiple_tool_calls(
                "REASONING", "CONTENT",
                { { "get_time", "{\"city\":\"Paris\"}" }, { "get_weather", "{\"city\":\"Paris\"}" } }))
            .run();
    }

    {
        auto tst = peg_tester("models/templates/deepseek-ai-DeepSeek-V3.1.jinja", detailed_debug);
        tst.test("REASONING</think>\nCONTENT")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(simple_assist_msg("CONTENT", "REASONING\n"))
            .run();
    }

    {
        auto tst = peg_tester("models/templates/deepseek-ai-DeepSeek-V3.1.jinja", detailed_debug);
        tst.test("CONTENT").enable_thinking(false).reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK).
            expect(simple_assist_msg("CONTENT", "")).run();
    }

    // GLM-4.6 tests - format: <tool_call>function_name\n<arg_key>...</arg_key>\n<arg_value>...</arg_value>\n</tool_call>
    {
        auto tst = peg_tester("models/templates/GLM-4.6.jinja", detailed_debug);
        tst.test(
               "<tool_call>special_function\n"
               "<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n"
               "</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();
    }

    // GLM-4.7-Flash tests - format: <tool_call>function_name<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>
    // Note: Template uses forced-open thinking mode (prompt ends with <think>)
    {
        auto tst = peg_tester("models/templates/GLM-4.7-Flash.jinja", detailed_debug);

        // Pure content (no reasoning)
        tst.test("Hello, world!\nWhat's up?")
            .enable_thinking(false)
            .expect(message_assist)
            .expect_reconstruction()
            .run();

        // Reasoning with content (forced-open mode - input starts after <think>)
        tst.test("I'm\nthinking</think>Hello, world!\nWhat's up?")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist_thoughts)
            .expect_reconstruction()
            .run();

        // Tool call without reasoning
        tst.test(
               "<tool_call>special_function"
               "<arg_key>arg1</arg_key><arg_value>1</arg_value>"
               "</tool_call>")
            .enable_thinking(false)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();

        // Tool call with reasoning (forced-open mode)
        tst.test(
               "I'm\nthinking</think>"
               "<tool_call>special_function"
               "<arg_key>arg1</arg_key><arg_value>1</arg_value>"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .expect_reconstruction()
            .run();

        tst.test(
               "<tool_call>special_function"
               "<arg_key>arg1</arg_key><arg_value>1</arg_value>"
               "</tool_call>"
               "<tool_call>special_function_with_opt"
               "<arg_key>arg1</arg_key><arg_value>1</arg_value>"
               "<arg_key>arg2</arg_key><arg_value>2</arg_value>"
               "</tool_call>")
            .enable_thinking(false)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .expect_reconstruction()
            .run();

        // #20650: tool with no required args, model emits <tool_call>name</tool_call> with no arg tags.
        {
            static common_chat_tool no_args_tool{
                "read_file_diff_md", "Reads a file diff",
                R"({"type":"object","properties":{"review_id":{"type":"string"},"file_id":{"type":"string"}}})",
            };
            tst.test(
                   "Let me read the diff content."
                   "</think>"
                   "<tool_call>read_file_diff_md</tool_call>")
                .enable_thinking(true)
                .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
                .tools({ no_args_tool })
                .expect_reasoning("Let me read the diff content.")
                .expect_tool_calls({{ "read_file_diff_md", "{}", {} }})
                .expect_reconstruction()
                .run();
        }
    }

    // Verify the throw path produces a readable error message, not std::out_of_range.
    // #20424 introduced effective_input = generation_prompt + input, but the throw
    // uses input.substr(result.end) where result.end is in effective_input space.
    {
        if (!g_template_filter.empty() && std::string("models/templates/GLM-4.7-Flash.jinja").find(g_template_filter) != std::string::npos) {
            auto tmpls = common_chat_templates_ptr(
                common_chat_templates_init(nullptr, read_file("models/templates/GLM-4.7-Flash.jinja")));

            static common_chat_tool weather_tool{
                "get_weather", "Get weather",
                R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
            };

            common_chat_templates_inputs inputs;
            inputs.tools = { weather_tool };
            inputs.enable_thinking = true;
            inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            inputs.add_generation_prompt = true;
            inputs.use_jinja = true;
            common_chat_msg msg;
            msg.role = "user";
            msg.content = "get_weather";
            inputs.messages = { msg };

            auto params = common_chat_templates_apply(tmpls.get(), inputs);
            common_peg_arena arena;
            arena.load(params.parser);
            common_chat_parser_params pp(params);

            // generation_prompt is non-empty for thinking models, so result.end
            // will be offset by generation_prompt.size() into effective_input space.
            assert(!pp.generation_prompt.empty());

            std::string bad_input =
                "Thinking.\n"
                "</think>"
                "<tool_call>get_weather"
                "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
                "</tool_call>\n";

            bool got_runtime_error = false;
            bool got_out_of_range = false;
            std::string error_msg;
            try {
                common_chat_peg_parse(arena, bad_input, /*is_partial=*/false, pp);
            } catch (const std::out_of_range & e) {
                got_out_of_range = true;
                error_msg = e.what();
            } catch (const std::runtime_error & e) {
                got_runtime_error = true;
                error_msg = e.what();
            }
            GGML_ASSERT(!got_out_of_range && "throw path crashed with out_of_range (input.substr in effective_input space)");
            GGML_ASSERT(got_runtime_error  && "throw path should produce std::runtime_error with parse position");
        }
    }

    // Kimi-K2-Thinking tests - custom parser
    // Unique feature: tool call ID embeds function name as functions.<name>:<counter>
    {
        auto tst = peg_tester("models/templates/Kimi-K2-Thinking.jinja", detailed_debug);

        // Basic content only
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        // Single tool call
        tst.test(
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>"
               "{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>")
            .tools({ special_function_tool })
            .expect(simple_assist_msg("", "", "special_function", "{\"arg1\": 1}", "functions.special_function:0"))
            .run();

        // Single tool call with reasoning
        tst.test(
               "<think>I'm thinking about this</think>"
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>"
               "{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(simple_assist_msg("", "I'm thinking about this", "special_function", "{\"arg1\": 1}", "functions.special_function:0"))
            .run();

        // Tool call with content
        tst.test(
               "Hello, world!\nWhat's up?"
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>"
               "{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>")
            .tools({ special_function_tool })
            .expect(simple_assist_msg("Hello, world!\nWhat's up?", "", "special_function", "{\"arg1\": 1}", "functions.special_function:0"))
            .run();

        // Multiple tool calls (parallel) - tests the indexing behavior
        tst.test(
               "<|tool_calls_section_begin|>"
               "<|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|>"
               "<|tool_call_begin|>functions.special_function_with_opt:1<|tool_call_argument_begin|>{\"arg1\": 1, \"arg2\": 2}<|tool_call_end|>"
               "<|tool_calls_section_end|>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", "functions.special_function:0" },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", "functions.special_function_with_opt:1" },
            })
            .run();

        // Multiple tool calls with reasoning
        tst.test(
               "<think>I need to call two functions</think>"
               "<|tool_calls_section_begin|>"
               "<|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|>"
               "<|tool_call_begin|>functions.python:1<|tool_call_argument_begin|>{\"code\": \"print('hey')\"}<|tool_call_end|>"
               "<|tool_calls_section_end|>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, python_tool
        })
            .expect_reasoning("I need to call two functions")
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", "functions.special_function:0" },
                { "python", "{\"code\": \"print('hey')\"}", "functions.python:1" },
            })
            .run();

        // Python tool with multiline code
        tst.test(
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.python:0<|tool_call_argument_begin|>"
               "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}<|tool_call_end|><|tool_calls_section_end|>")
            .tools({ python_tool })
            .expect_tool_calls({
                { "python", "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}", "functions.python:0" },
            })
            .run();

        // Tool call with empty arguments
        tst.test(
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.empty_args:0<|tool_call_argument_begin|>"
               "{}<|tool_call_end|><|tool_calls_section_end|>")
            .tools({ empty_args_tool })
            .expect(simple_assist_msg("", "", "empty_args", "{}", "functions.empty_args:0"))
            .run();

        // Partial tool call (streaming)
        tst.test(
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>"
               "{\"arg1\": ")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .is_partial(true)
            .expect(simple_assist_msg("", "", "special_function", "{\"arg1\": ", "functions.special_function:0"))
            .run();

        // Three tool calls to verify counter continues incrementing
        tst.test(
               "<|tool_calls_section_begin|>"
               "<|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|>"
               "<|tool_call_begin|>functions.python:1<|tool_call_argument_begin|>{\"code\": \"print(1)\"}<|tool_call_end|>"
               "<|tool_call_begin|>functions.html:2<|tool_call_argument_begin|>{\"markup\": \"<p>test</p>\"}<|tool_call_end|>"
               "<|tool_calls_section_end|>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, python_tool, html_tool
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", "functions.special_function:0" },
                { "python", "{\"code\": \"print(1)\"}", "functions.python:1" },
                { "html", "{\"markup\": \"<p>test</p>\"}", "functions.html:2" },
            })
            .run();

        // Multiple tool calls with reasoning, call *inside thinking block*
        tst.test(
               "<think>I need to call two functions"
               "<|tool_calls_section_begin|>"
               "<|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|>"
               "<|tool_call_begin|>functions.python:1<|tool_call_argument_begin|>{\"code\": \"print('hey')\"}<|tool_call_end|>"
               "<|tool_calls_section_end|>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, python_tool
        })
            .expect_reasoning("I need to call two functions")
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", "functions.special_function:0" },
                { "python", "{\"code\": \"print('hey')\"}", "functions.python:1" },
            })
            .run();

        // Multiple tool calls with reasoning, call *inside thinking block* and *without section markers or end markers
        tst.test(
               "<think>I need to call two functions"
               "<|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}"
               "<|tool_call_begin|>functions.python:1<|tool_call_argument_begin|>{\"code\": \"print('hey')\"}")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, python_tool
        })
            .expect_reasoning("I need to call two functions")
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", "functions.special_function:0" },
                { "python", "{\"code\": \"print('hey')\"}", "functions.python:1" },
            })
            .run();

        // Real life test - execute_command
        tst.test("<|tool_call_begin|>functions.execute_command:0<|tool_call_argument_begin|>{\"command\": \"ls -lah\""
            ", \"cwd\": \"/home/jarvis/development/exllamav3\", \"timeout\": 10}")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .parallel_tool_calls(true)
            .tools({
                {
                    /* .name = */ "execute_command",
                    /* .description = */ "Execute shell command",
                    /* .parameters = */ R"({
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute"
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Working directory"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "The timeout in seconds"
                            }
                        },
                        "required": ["command"]
                    })"
                }
            }).
            expect_tool_calls({
                {
                    "execute_command",
                    R"({"command": "ls -lah", "cwd": "/home/jarvis/development/exllamav3", "timeout": 10})",
                    "functions.execute_command:0"
                }
            })
            .run();
    }

    {
        auto kimi_id_special_func_tool_call =
            simple_assist_msg("", "", "special_function", "{\"arg1\": 1}", "functions.special_function:0");

        // Kimi-K2 old template
        auto tst = peg_tester("models/templates/moonshotai-Kimi-K2.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test(
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>"
               "{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>")
            .tools({ special_function_tool })
            .expect(kimi_id_special_func_tool_call)
            .expect_reconstruction()
            .run();

        // Kimi-K2-Instruct
        auto tst2 = peg_tester("models/templates/Kimi-K2-Instruct.jinja", detailed_debug);
        tst2.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst2.test(
               "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>"
               "{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>")
            .tools({ special_function_tool })
            .expect(kimi_id_special_func_tool_call)
            .expect_reconstruction()
            .run();
    }

    // LFM2-8B-A1B tests - uses <|tool_list_start|>/<|tool_list_end|> and <|tool_call_start|>[name(args)]<|tool_call_end|>
    {
        auto tst = peg_tester("models/templates/LFM2-8B-A1B.jinja", detailed_debug);

        // Basic content only
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        // Single tool call without reasoning
        tst.test("<|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        // Tool call with string argument
        tst.test("<|tool_call_start|>[get_time(city=\"XYZCITY\")]<|tool_call_end|>")
            .tools({ get_time_tool })
            .expect(message_with_tool_calls("get_time", "{\"city\":\"XYZCITY\"}"))
            .run();

        // Tool call with reasoning (enable_thinking=true)
        tst.test("<think>I'm\nthinking</think><|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .run();

        // Multiple tool calls (parallel)
        tst.test("<|tool_call_start|>[special_function(arg1=1), special_function_with_opt(arg1=1, arg2=2)]<|tool_call_end|>")
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
            })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .run();

        // Tool call with reasoning and content
        tst.test("<think>I need to call a function</think>"
                 "Let me check the time.<|tool_call_start|>[get_time(city=\"Paris\")]<|tool_call_end|>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ get_time_tool })
            .expect(message_with_reasoning_content_and_multiple_tool_calls(
                "I need to call a function", "Let me check the time.", { { "get_time", "{\"city\":\"Paris\"}" } }
            ))
            .run();

        // Python tool with multiline code in string
        tst.test("<|tool_call_start|>[python(code=\"def hello():\\n    print('hey')\")]<|tool_call_end|>")
            .tools({ python_tool })
            .expect_tool_calls({
                { "python", R"#({"code": "def hello():\\n    print('hey')"})#", "" }
            })
            .run();

        // Partial tool call (streaming)
        tst.test("<|tool_call_start|>[special_function(arg1=")
            .tools({ special_function_tool })
            .is_partial(true)
            .expect(simple_assist_msg("", "", "special_function", "{\"arg1\": "))
            .run();

        // Tool call with empty arguments
        tst.test("<|tool_call_start|>[empty_args()]<|tool_call_end|>")
            .tools({ empty_args_tool })
            .expect(simple_assist_msg("", "", "empty_args", "{}"))
            .run();

        // fake tool call marker in reasoning
        tst.test(
               "<think>Let me think about <|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|> hmm</think>"
               "<|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect_reasoning("Let me think about <|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|> hmm")
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
            })
            .run();
    }

    // LFM2.5 tests - uses plain "List of tools: [...]" and bare [name(args)] without wrapper tokens
    {
        auto tst = peg_tester("models/templates/LFM2.5-Instruct.jinja", detailed_debug);

        // Basic content only
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        // Single tool call without reasoning
        tst.test("[special_function(arg1=1)]")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        // Tool call with string argument
        tst.test("[get_time(city=\"XYZCITY\")]")
            .tools({ get_time_tool })
            .expect(message_with_tool_calls("get_time", "{\"city\":\"XYZCITY\"}"))
            .run();

        // Tool call with reasoning (enable_thinking=true)
        tst.test("<think>I'm\nthinking</think>[special_function(arg1=1)]")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .run();

        // Multiple tool calls (parallel)
        tst.test("[special_function(arg1=1), special_function_with_opt(arg1=1, arg2=2)]")
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
            })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .run();

        // Tool call with content before tool call
        tst.test("Let me check the time.[get_time(city=\"Paris\")]")
            .tools({ get_time_tool })
            .expect(message_with_reasoning_content_and_multiple_tool_calls(
                "", "Let me check the time.", { { "get_time", "{\"city\":\"Paris\"}" } }
            ))
            .run();

        // Partial tool call (streaming)
        tst.test("[special_function(arg1=")
            .tools({ special_function_tool })
            .is_partial(true)
            .expect(simple_assist_msg("", "", "special_function", "{\"arg1\": "))
            .run();

        // Tool call with empty arguments
        tst.test("[empty_args()]")
            .tools({ empty_args_tool })
            .expect(simple_assist_msg("", "", "empty_args", "{}"))
            .run();
    }

    // Apertus-8B-Instruct tests - FUNC_NAME_AS_KEY format
    // Format: <|tools_prefix|>[{"function_name": {...arguments...}}]<|tools_suffix|>
    {
        auto tst = peg_tester("models/templates/Apertus-8B-Instruct.jinja", detailed_debug);
        tst.test("<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();
    }

    // MiniMax-M2 tests - XML invoke format with parameter tags
    // Format: <minimax:tool_call><invoke name="func"><parameter name="key">value</parameter></invoke></minimax:tool_call>
    {
        auto tst = peg_tester("models/templates/MiniMax-M2.jinja", detailed_debug);
        tst.test(
               "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter "
               "name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .run();
    }

    // NVIDIA-Nemotron-Nano-v2 tests - <TOOLCALL>...</TOOLCALL> format
    // Format: <TOOLCALL>[{"name": "func", "arguments": {...}}]</TOOLCALL>
    {
        auto tst = peg_tester("models/templates/NVIDIA-Nemotron-Nano-v2.jinja", detailed_debug);
        tst.test("<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL><SPECIAL_12>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();
    }

    // CohereForAI-c4ai-command-r7b (uses START_RESPONSE/END_RESPONSE, START_THINKING/END_THINKING, START_ACTION/END_ACTION)
    {
        auto tst = peg_tester("models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja", detailed_debug);
        tst.test("<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>").expect(message_assist).run();
        tst.test(
               "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
               "<|START_ACTION|>[\n"
               "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
               "]<|END_ACTION|>")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool })
            .expect(message_assist_thoughts_call_idx)
            .run();
    }
    // CohereForAI-c4ai-command-r-plus (uses markdown code block format)
    {
        auto tst = peg_tester("models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja", detailed_debug);
        tst.test("<|CHATBOT_TOKEN|>Hello, world!\nWhat's up?<|END_OF_TURN_TOKEN|>").expect(message_assist).run();
        // Tool calls: Action: followed by JSON code block
        tst.test(
               "Action:\n"
               "```json\n"
               "[{\"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}]\n"
               "```")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();
    }

    // mistralai-Mistral-Nemo-Instruct-2407.jinja
    {
        auto tst = peg_tester("models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("[TOOL_CALLS][{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, \"id\": \"123456789\"}]")
            .tools({ special_function_tool })
            .expect(message_assist_call_id)
            .expect_reconstruction()
            .run();
    }
    {
        auto tst = peg_tester("models/templates/meetkai-functionary-medium-v3.1.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("<function=special_function>{\"arg1\": 1}</function>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();
    }
    // Functionary v3.2 - recipient-based format: >>>recipient\n{content}
    {
        auto tst = peg_tester("models/templates/meetkai-functionary-medium-v3.2.jinja", detailed_debug);
        tst.test("all\nHello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("special_function\n{\"arg1\": 1}")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();
    }

    // FireFunction
    {
        auto tst = peg_tester("models/templates/fireworks-ai-llama-3-firefunction-v2.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test(" functools[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();
    }

    // DeepSeek R1 Distill Llama 8B - reasoning tests only (forced open thinking)
    // Note: Template uses forced-open mode (prompt ends with <think>), so input shouldn't include opening tag
    {
        auto tst = peg_tester("models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja", detailed_debug);
        tst.test("</think>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist)
            .run();
        tst.test("I'm\nthinking</think>Hello, world!\nWhat's up?")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist_thoughts)
            .run();
    }
    // llama-cpp DeepSeek R1 template (always forced-open thinking)
    {
        auto tst = peg_tester("models/templates/llama-cpp-deepseek-r1.jinja", detailed_debug);
        tst.test("</think>Hello, world!\nWhat's up?").expect(message_assist).reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK).run();
        tst.test("I'm\nthinking</think>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist_thoughts)
            .run();
        tst.test(
               "</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
               "```json\n{\"arg1\": 1}```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool })
            .parallel_tool_calls(true)
            .expect(message_assist_call)
            .run();
    }
    // DeepSeek R1 Distill Qwen 32B - reasoning tests only (forced open thinking)
    // Note: Template uses forced-open mode (prompt ends with <think>), so input shouldn't include opening tag
    {
        auto tst = peg_tester("models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja", detailed_debug);
        tst.test("</think>Hello, world!\nWhat's up?").enable_thinking(true).
            reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK).
            expect(message_assist).run();
        tst.test("I'm\nthinking</think>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist_thoughts)
            .run();
        tst.test(
               "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
               "```json\n{\"arg1\": 1}```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
            .tools({ special_function_tool })
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(message_assist_call)
            .run();
    }

    // MiMo-VL / Hermes 3 / Qwen 2.5 (Common <tool_call> JSON format)
    for (const auto & path :
         { "models/templates/MiMo-VL.jinja", "models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja",
           "models/templates/Qwen-Qwen2.5-7B-Instruct.jinja" }) {
        auto tst = peg_tester(path, detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("<tool_call>\n{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();
    }

    // Apriel 1.5
    {
        auto tst = peg_tester("models/templates/unsloth-Apriel-1.5.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();
        tst.test("<tool_calls>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</tool_calls>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();
    }

    // Apriel 1.6 Thinker (reasoning-only support)
    {
        auto tst = peg_tester("models/templates/Apriel-1.6-15b-Thinker-fixed.jinja", detailed_debug);

        // Implicit reasoning start (forced open)
        tst.test("I'm\nthinking\n[BEGIN FINAL RESPONSE]\nHello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .expect(simple_assist_msg("Hello, world!\nWhat's up?", "Here are my reasoning steps:\nI'm\nthinking"))
            .expect_reconstruction()
            .run();

        // Reasoning + Tool calls
        tst.test(
               "I'm\nthinking\n[BEGIN FINAL RESPONSE]\n<tool_calls>[{\"name\": \"special_function\", \"arguments\": "
               "{\"arg1\": 1}}]</tool_calls>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect(simple_assist_msg("", "Here are my reasoning steps:\nI'm\nthinking", "special_function", "{\"arg1\":1}"))
            .run();
    }

    // Mistral Small 3.2 - FUNC_BRACKET_TAG format: [TOOL_CALLS]func_name[CALL_ID]id[ARGS]{...}
    {
        auto tst = peg_tester("models/templates/Mistral-Small-3.2-24B-Instruct-2506.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("[TOOL_CALLS]special_function[CALL_ID]123456789[ARGS]{\"arg1\": 1}")
            .tools({ special_function_tool })
            .expect(message_assist_call_id)
            .expect_reconstruction()
            .run();

        tst.test("[TOOL_CALLS]special_function[CALL_ID]000000001[ARGS]{\"arg1\": 1}"
            "[TOOL_CALLS]special_function_with_opt[CALL_ID]000000002[ARGS]{\"arg1\": 1, \"arg2\": 2}")
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
            })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", "000000001" },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", "000000002" },
            })
            .expect_reconstruction()
            .run();


    }
    // Devstral
    {
        auto tst = peg_tester("models/templates/unsloth-mistral-Devstral-Small-2507.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("[TOOL_CALLS]special_function[ARGS]{\"arg1\": 1}")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();
        tst.test("Hello, world!\nWhat's up?[TOOL_CALLS]special_function[ARGS]{\"arg1\": 1}")
            .tools({ special_function_tool })
            .expect(message_assist_call_content)
            .expect_reconstruction()
            .run();
    }

    {
        // Llama 3.1
        auto tst = peg_tester("models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").tools({ special_function_tool }).expect(message_assist).expect_reconstruction().run();
    }

    {
        // Llama 3.2
        auto tst = peg_tester("models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").tools({ special_function_tool }).expect(message_assist).expect_reconstruction().run();
    }

    {
        // Llama 3.3
        auto tst = peg_tester("models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").tools({ python_tool }).expect(message_assist).expect_reconstruction().run();
    }

    // GPT-OSS format tests
    {
        auto tst = peg_tester("models/templates/openai-gpt-oss-120b.jinja", detailed_debug);

        // Basic content only - final channel
        tst.test("<|channel|>final<|message|>Hello, world!\nWhat's up?").expect(message_assist).run();

        // Basic content only - commentary channel
        tst.test("<|channel|>commentary<|message|>Hello, world!\nWhat's up?").expect(message_assist).run();

        // Analysis channel (reasoning) with final channel (content)
        tst.test(
               "<|channel|>analysis<|message|>I'm\nthinking<|end|><|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's "
               "up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist_thoughts)
            .run();

        // Analysis channel (reasoning) with final channel (content) with reasoning_format = none
        tst.test(
               "<|channel|>analysis<|message|>I'm\nthinking<|end|><|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's "
               "up?")
            .reasoning_format(COMMON_REASONING_FORMAT_NONE)
            .expect_content("<|channel|>analysis<|message|>I'm\nthinking<|end|>Hello, world!\nWhat's up?")
            .run();

        // Analysis channel only (partial) - still works when reasoning format is set
        tst.test("<|channel|>analysis<|message|>I'm\nthinking")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .is_partial(true)
            .expect_reasoning("I'm\nthinking")
            .run();

        // Tool call with recipient in role header: " to=functions.NAME<|channel|>analysis<|message|>JSON"
        tst.test(" to=functions.special_function<|channel|>analysis<|message|>{\"arg1\": 1}")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        // Tool call with recipient in channel header: "<|channel|>analysis to=functions.NAME<|message|>JSON"
        tst.test("<|channel|>analysis to=functions.special_function<|message|>{\"arg1\": 1}")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        // Tool call with constraint: " to=functions.NAME<|channel|>analysis <|constrain|>json<|message|>JSON"
        tst.test(" to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        // Tool call in commentary channel (channel header variant)
        tst.test("<|channel|>commentary to=functions.special_function<|message|>{\"arg1\": 1}")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        // Tool call with reasoning + content (analysis first, then tool call)
        tst.test(
               "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
               "<|start|>assistant to=functions.special_function<|channel|>analysis<|message|>{\"arg1\": 1}")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .run();

        // Complex tool calling
        tst.test(
            "<|channel|>analysis<|message|>Thinking about edit...<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.edit <|constrain|>json"
            "<|message|>{\"oldString\": \"if (part < railCount - 1) {\", \"newString\": \"if (part < 4) {\", \"replaceAll\": false}"
            )
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({
                {
                    /* .name = */ "edit",
                    /* .description = */ "Edit a file",
                    /* .parameters = */ R"({
                        "type": "object",
                        "properties": {
                            "oldString": {
                                "type": "string",
                                "description": "Old string to replace."
                            },
                            "newString": {
                                "type": "string",
                                "description": "New replacement string."
                            },
                            "replaceAll": {
                                "type": "boolean",
                                "description": "Whether to replace all occurences."
                            }
                        },
                        "required": ["oldString", "newString"]
                    })",
                }
            })
            .expect_reasoning("Thinking about edit...")
            .expect_tool_calls({
                { "edit", R"({"oldString": "if (part < railCount - 1) {", "newString": "if (part < 4) {", "replaceAll": false})", {} }
            })
            .run();

        // Structured output
        tst.test(
            "<|channel|>analysis<|message|>I need to output the invoice details in JSON<|end|>"
            "<|start|>assistant<|channel|>final <|constrain|>json"
            "<|message|>"
            R"({"amount": 123.45, "date": "2025-12-03"})"
            )
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .json_schema(invoice_schema)
            .expect_reasoning("I need to output the invoice details in JSON")
            .expect_content(R"({"amount": 123.45, "date": "2025-12-03"})")
            .run();


        // Unsolicited tool calls. There is no good way to handle these, so we return empty content.

        // Builtin function - recipient in role
        tst.test(
               "<|channel|>analysis<|message|>I will execute python to say hello<|end|>"
               "<|start|>assistant to=container.exec<|channel|>commentary<|message|>python3 -c 'print(\"hello\")'")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect_reasoning("I will execute python to say hello")
            .expect_content("")
            .run();

        // Builtin function - recipient in channel
        tst.test(
               "<|channel|>analysis<|message|>I will execute python to say hello<|end|>"
               "<|start|>assistant<|channel|>commentary to=python <|constrain|>code<|message|>print(\"hello\")")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect_reasoning("I will execute python to say hello")
            .expect_content("")
            .run();

        // Edge cases

        // "<|channel|>commentary to=assistant" before reasoning
        tst.test(
               "<|channel|>commentary to=assistant<|channel|>analysis<|message|>I'm\nthinking<|end|><|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's "
               "up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist_thoughts)
            .run();

        // "<|channel|>commentary to=assistant" before final message
        tst.test(
               "<|channel|>analysis<|message|>I'm\nthinking<|end|><|start|>assistant<|channel|>commentary to=assistant<|channel|>final<|message|>Hello, world!\nWhat's "
               "up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist_thoughts)
            .run();
    }

    {
        auto tst = peg_tester("models/templates/StepFun3.5-Flash.jinja", detailed_debug);
        tst.test("I was thinking</think>\nNow I'm not.").
            enable_thinking(true).
            reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK).
            expect_reasoning("I was thinking").
            expect_content("Now I'm not.")
        .run();

        // Test that numeric-looking string values are coerced to strings per the schema
        tst.test(
               "Let me call the magic tool\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=magic>\n"
               "<parameter=name>\nfooBar\n</parameter>\n"
               "<parameter=ref>\n5123123\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ magic_tool })
            .expect_reasoning("Let me call the magic tool")
            .expect_tool_calls({
                { "magic", R"({"name": "fooBar", "ref": "5123123"})", {} },
            })
            .run();

        // Test that numeric values are correctly interpreted as numbers when schema calls for number
        tst.test(
               "Let me call the special function\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n42555916\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool })
            .expect_reasoning("Let me call the special function")
            .expect_tool_calls({
                { "special_function", R"({"arg1": 42555916})", {} },
            })
            .run();

        tst.test(
               "Let me call the special function with opt\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=special_function_with_opt>\n"
               "<parameter=arg1>\n42555916\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ special_function_tool_with_optional_param })
            .expect_reasoning("Let me call the special function with opt")
            .expect_tool_calls({
                { "special_function_with_opt", R"({"arg1": 42555916})", {} },
            })
            .run();

        tst.test(
               "Let me call the magic_int function\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=magic_int>\n"
               "<parameter=ref>\n42555916\n</parameter>\n"
               "<parameter=name>\nbaz\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ magic_int_tool })
            .expect_reasoning("Let me call the magic_int function")
            .expect_tool_calls({
                { "magic_int", R"({"ref": 42555916, "name": "baz"})", {} },
            })
            .run();

        tst.test(
               "Call string_param with empty text\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=string_param>\n"
               "<parameter=text>\n\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ string_param_tool })
            .expect_reasoning("Call string_param with empty text")
            .expect_tool_calls({
                { "string_param", R"({"text": ""})", {} },
            })
            .run();

        tst.test(
               "Test simple quoted unquoted\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=quoted_unquoted>\n"
               "<parameter=quoted>\n\"foo\"\n</parameter>\n"
               "<parameter=unquoted>\nfoo\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ quoted_unquoted_tool })
            .expect_reasoning("Test simple quoted unquoted")
            .expect_tool_calls({
                { "quoted_unquoted", R"({"quoted": "\"foo\"", "unquoted": "foo"})", {} },
            })
            .run();

        tst.test(
               "Test complex quoted unquoted\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=quoted_unquoted>\n"
               "<parameter=quoted>\n\"printf(\\\"foo\\\");\"\n</parameter>\n"
               "<parameter=unquoted>\nprintf(\"foo\");\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ quoted_unquoted_tool })
            .expect_reasoning("Test complex quoted unquoted")
            .expect_tool_calls({
                { "quoted_unquoted", R"({ "quoted" : "\"printf(\\\"foo\\\");\"", "unquoted": "printf(\"foo\");" })", {} }
            })
            .run();

            tst.test(
               "Test negative number\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=magic_int>\n"
               "<parameter=ref>\n-14\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ magic_int_tool })
            .expect_reasoning("Test negative number")
            .expect_tool_calls({
                { "magic_int", R"({ "ref" : -14 })", {} }
            })
            .run();

            tst.test(
               "Test decimal number\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=amount>\n"
               "<parameter=orig>\n3.14\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ amount_tool })
            .expect_reasoning("Test decimal number")
            .expect_tool_calls({
                { "amount", R"({ "orig" : 3.14 })", {} }
            })
            .run();

            tst.test(
               "Test imaginary number\n"
               "</think>\n"
               "<tool_call>\n"
               "<function=imaginary_number>\n"
               "<parameter=number>\n"
               "{ \"real\": 3.14, \"imaginary\": 2.71 }\n"
               "</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .tools({ imaginary_number_tool })
            .expect_reasoning("Test imaginary number")
            .expect_tool_calls({
                { "imaginary_number", R"({ "number" : {"real":3.14,"imaginary":2.71 } })", {} }
            })
            .run();

    }

    // GigaChat V3
    {
        auto tst = peg_tester("models/templates/GigaChat3-10B-A1.8B.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("<|message_sep|>\n\nfunction call<|role_sep|>\n{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();

        tst.test(
            "Hello, world!\nWhat's up?"
            "<|message_sep|>\n\nfunction call<|role_sep|>\n{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}"
        )
            .tools({ special_function_tool })
            .expect(message_assist_call_content)
            .expect_reconstruction()
            .run();
    }

    // GigaChat V3.1
    {
        auto tst = peg_tester("models/templates/GigaChat3.1-10B-A1.8B.jinja", detailed_debug);
        tst.test("Hello, world!\nWhat's up?").expect(message_assist).expect_reconstruction().run();
        tst.test("<|function_call|>{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_reconstruction()
            .run();

        tst.test(
            "Hello, world!\nWhat's up?"
            "<|function_call|>{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}"
        )
            .tools({ special_function_tool })
            .expect(message_assist_call_content)
            .expect_reconstruction()
            .run();
    }
}

// Test the developer role to system workaround with a simple mock template
static void test_developer_role_to_system_workaround() {
    LOG_DBG("%s\n", __func__);

    // Simple mock template that supports system role
    const std::string mock_template =
        "{%- for message in messages -%}\n"
        "  {{- '<|' + message.role + '|>' + message.content + '<|end|>' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "  {{- '<|assistant|>' -}}\n"
        "{%- endif -%}";

    auto tmpls = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, mock_template));

    // Test case 1: Developer message - should be changed to system
    // After simplification we only test this case
    {
        common_chat_templates_inputs inputs;
        common_chat_msg developer_msg;
        developer_msg.role = "developer";
        developer_msg.content = "You are a helpful developer assistant.";
        inputs.messages = { developer_msg };
        inputs.add_generation_prompt = false;

        auto params = common_chat_templates_apply(tmpls.get(), inputs);

        // The developer role should have been changed to system
        if (params.prompt.find("<|developer|>") != std::string::npos) {
            throw std::runtime_error("Test failed: developer role was not changed to system");
        }
        if (params.prompt.find("<|system|>You are a helpful developer assistant.<|end|>") == std::string::npos) {
            throw std::runtime_error("Test failed: system message not found in output");
        }
        LOG_ERR("Test 1 passed: developer role changed to system\n");
    }
}

static void test_msg_diffs_compute() {
    LOG_DBG("%s\n", __func__);
    {
        common_chat_msg msg1;

        common_chat_msg msg2;
        msg2.content = "Hello, world!";

        common_chat_msg_diff diff;
        diff.content_delta = "Hello, world!";

        assert_equals({ diff }, common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg1;
        msg1.content = "Hello,";

        common_chat_msg msg2;
        msg2.content = "Hello, world!";

        common_chat_msg_diff diff;
        diff.content_delta = " world!";

        assert_equals({ diff }, common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg0;

        common_chat_msg msg1;
        msg1.tool_calls = {
            { "special_function", "{\"ar", /* .id = */ "123" }
        };

        common_chat_msg msg2;
        msg2.tool_calls = {
            { "special_function", "{\"arg1\": 1}", /* .id = */ "123" }
        };

        common_chat_msg_diff diff01;
        diff01.tool_call_index           = 0;
        diff01.tool_call_delta.name      = "special_function";
        diff01.tool_call_delta.id        = "123";
        diff01.tool_call_delta.arguments = "{\"ar";

        assert_equals({ diff01 }, common_chat_msg_diff::compute_diffs(msg0, msg1));

        common_chat_msg_diff diff12;
        diff12.tool_call_index           = 0;
        // Note: neither id nor name change here.
        diff12.tool_call_delta.arguments = "g1\": 1}";

        assert_equals({ diff12 }, common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg0;

        common_chat_msg msg2;
        msg2.tool_calls = {
            { "f1", "{\"arg1\": 1}", /* .id = */ "123" },
            { "f2", "{\"arg2\": 2}", /* .id = */ "222" },
        };

        common_chat_msg_diff diff1;
        diff1.tool_call_index           = 0;
        diff1.tool_call_delta.name      = "f1";
        diff1.tool_call_delta.id        = "123";
        diff1.tool_call_delta.arguments = "{\"arg1\": 1}";

        common_chat_msg_diff diff2;
        diff2.tool_call_index           = 1;
        diff2.tool_call_delta.name      = "f2";
        diff2.tool_call_delta.id        = "222";
        diff2.tool_call_delta.arguments = "{\"arg2\": 2}";

        assert_equals({ diff1, diff2 }, common_chat_msg_diff::compute_diffs(msg0, msg2));
    }
}

int main(int argc, char ** argv) {
    bool detailed_debug    = false;
    bool only_run_filtered = false;

    // Check for --template flag
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--template" && i + 1 < argc) {
            g_template_filter = argv[++i];
            // Only run PEG parser tests with the filter
            only_run_filtered = true;
        }
        if (arg == "--detailed") {
            detailed_debug = true;
            common_log_set_verbosity_thold(999);
        }
        if (arg == "--force-reconstruction-test") {
            g_force_reconstruction_test = true;
            only_run_filtered          = true;
        }
    }

    if (only_run_filtered) {
        test_template_output_peg_parsers(detailed_debug);
        std::cout << "\n[chat] All template tests passed!" << '\n';
        return 0;
    }

#ifndef _WIN32
    if (argc > 1) {
        common_chat_templates_inputs inputs;
        common_chat_msg              msg;
        msg.role        = "user";
        msg.content     = "Hey";
        inputs.messages = { msg };
        inputs.tools    = { special_function_tool };

        std::cout << "| Template | Format |\n";
        std::cout << "|----------|--------|\n";

        for (int i = 1; i < argc; i++) {
            try {
                std::string path = argv[i];
                if (path.rfind(".jinja") != path.size() - 6) {
                    std::cerr << "Skipping non-jinja file: " << path << '\n';
                    continue;
                }
                auto         tmpls  = read_templates(path);
                auto         parts  = string_split(path, "/");
                const auto & name   = parts[parts.size() - 1];
                const auto * format = common_chat_format_name(common_chat_templates_apply(tmpls.get(), inputs).format);
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
        test_developer_role_to_system_workaround();
        test_template_output_peg_parsers(detailed_debug);
        std::cout << "\n[chat] All tests passed!" << '\n';
    }
    return 0;
}
