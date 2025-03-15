//  Tests chat handling, including grammar generation and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <string>

#include "chat.h"
#include "common.h"
#include "llama-grammar.h"
#include "unicode.h"

using json = nlohmann::ordered_json;

static std::ostream & operator<<(std::ostream & os, const common_chat_msg_diff & diff) {
    // os << "reasoning_content_delta: " << diff.reasoning_content_delta << '\n';
    os << "{ content_delta: " << diff.content_delta << "; ";
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

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
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
static void assert_msg_equals(const common_chat_msg & expected, const common_chat_msg & actual) {
    assert_equals(expected.role, actual.role);
    assert_equals(expected.content, actual.content);
    assert_equals(expected.content_parts.size(), actual.content_parts.size());
    for (size_t i = 0; i < expected.content_parts.size(); i++) {
        const auto & expected_part = expected.content_parts[i];
        const auto & actual_part   = actual.content_parts[i];
        assert_equals(expected_part.type, actual_part.type);
        assert_equals(expected_part.text, actual_part.text);
    }
    assert_equals(expected.reasoning_content, actual.reasoning_content);
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
std::vector<common_chat_tool> tools           { special_function_tool, python_tool };
std::vector<common_chat_tool> llama_3_1_tools { special_function_tool, code_interpreter_tool };

struct delta_data {
    std::string        delta;
    common_chat_params params;
};

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
                          common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE) {
    common_chat_msg user_message;
    user_message.role = "user";
    user_message.content = "Hello, world!";

    for (const auto & tool_choice : std::vector<common_chat_tool_choice> {COMMON_CHAT_TOOL_CHOICE_AUTO, COMMON_CHAT_TOOL_CHOICE_REQUIRED}) {
        auto data = init_delta(tmpls, end_tokens, user_message, test_message, tools, tool_choice);
        if (!expected_delta.empty()) {
            assert_equals(expected_delta, data.delta);
        }

        if (expect_grammar_triggered) {
            common_chat_syntax syntax;
            syntax.format = data.params.format;
            syntax.reasoning_format = reasoning_format;
            const auto msg = common_chat_parse(data.delta, /* is_partial= */ false, syntax);
            assert_msg_equals(test_message, msg);
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
                            pos = match.position(1);
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
const common_chat_msg message_assist {
    "assistant",
    "Hello, world!\nWhat's up?",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_thoughts_unparsed_deepseek {
    "assistant",
    "<think>I'm thinking</think>Hello, world!\nWhat's up?",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_thoughts_unparsed_r7b {
    "assistant",
    "<|START_THINKING|>I'm thinking<|END_THINKING|>Hello, world!\nWhat's up?",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_thoughts {
    "assistant",
    "Hello, world!\nWhat's up?",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "I'm thinking",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_thoughts_unclosed_unparsed {
    "assistant",
    "I'm thinking</think>Hello, world!\nWhat's up?",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const std::vector<common_chat_tool_call> tool_calls {
    { "special_function", "{\"arg1\": 1}", /* .id = */ "" },
};
const std::vector<common_chat_tool_call> tool_calls_cutoff_args {
    { "special_function", "{\"arg", /* .id = */ "" },
};
const std::vector<common_chat_tool_call> tool_calls_empty_args {
    { "special_function", "", /* .id = */ "" },
};
const std::vector<common_chat_tool_call> tool_calls_idx {
    { "special_function", "{\"arg1\": 1}", /* .id = */ "0" },
};
const std::vector<common_chat_tool_call> tool_calls_id {
    { "special_function", "{\"arg1\": 1}", /* .id = */ "123456789" },
};
const std::vector<common_chat_tool_call> tool_calls_python {
    { "python", "{\"code\": \"print('hey')\"}", /* .id = */ "" },
};
const std::vector<common_chat_tool_call> tool_calls_python_lines {
    { "python", "{\"code\": \"# This is a program:\\nprint('hey')\"}", /* .id = */ "" },
};
const std::vector<common_chat_tool_call> tool_calls_python_lines_unclosed {
    { "python", "{\"code\":\"# This is a program:\\nprint('hey')", /* .id = */ "" },
};

const common_chat_msg message_assist_empty {
    "assistant",
    "",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_thoughts_no_content {
    "assistant",
    "",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "I'm\nthinking",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_empty_args {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_empty_args,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_cutoff_args {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_cutoff_args,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_thoughts = {
    "assistant",
    /* .content = */ "",
    /* .content_parts = */ {},
    tool_calls,
    /* .reasoning_content = */ "I'm\nthinking",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_thoughts_unparsed = {
    "assistant",
    /* .content = */ "<think>I'm\nthinking</think>",
    /* .content_parts = */ {},
    tool_calls,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_id {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_id,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_idx {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_idx,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_thoughts_call_idx {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_idx,
    /* .reasoning_content = */ "I'm\nthinking",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_python {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_python,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_python_lines {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_python_lines,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_python_lines_unclosed {
    "assistant",
    "",
    /* .content_parts = */ {},
    tool_calls_python_lines_unclosed,
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};
const common_chat_msg message_assist_call_code_interpreter {
    "assistant",
    "",
    /* .content_parts = */ {},
    { { "code_interpreter", "{\"code\": \"print('hey')\"}", /* .id = */ "" } },
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

static void test_msgs_oaicompat_json_conversion() {
    std::vector<common_chat_msg> msgs{
        message_user,
        message_user_parts,
        message_assist_call,
        message_assist_call_thoughts,
        message_assist_call_thoughts_unparsed,
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
            "          \"arguments\": \"{\\\"code\\\": \\\"print('hey')\\\"}\"\n"
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

static void test_template_output_parsers() {

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
        assert_equals(COMMON_CHAT_FORMAT_GENERIC, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    }
    {
        auto tmpls = read_templates("models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja");
        std::vector<std::string>   end_tokens{ "<|END_OF_TURN_TOKEN|>" };

        assert_equals(COMMON_CHAT_FORMAT_COMMAND_R7B,                   common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_COMMAND_R7B,                   common_chat_templates_apply(tmpls.get(), inputs_tools).format);

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
                "<|START_THINKING|>I'm thinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<|START_THINKING|>I'm thinking<|END_THINKING|>"
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
                "<|START_THINKING|>I'm thinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_COMMAND_R7B}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<|START_THINKING|>I'm thinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
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
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
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
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
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
            message_assist_empty,
            common_chat_parse(
                "{ \"tool_call\" : { \"name\" : \"t",
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
                      "  ]\n"
                      "}");
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
        auto tmpls = read_templates("models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja");
        std::vector<std::string> end_tokens{ "<|im_end|>" };

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
            {
                /* .role = */ "assistant",
                /* .content = */ "",
                /* .content_parts = */ {},
                /* .tool_calls = */ {
                    {
                        /* .name = */ "python",
                        /* .arguments = */ "",
                        /* .id = */ "",
                    }
                },
                /* .reasoning_content = */ "",
                /* .tool_name = */ "",
                /* .tool_call_id = */ "",
            },
            common_chat_parse(
                "```json\n"
                "<function_call> { \"name\" : \"python\"",
                /* is_partial= */ true,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<tool_call>\n"
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

        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<think>I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
        // assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
        //     common_chat_parse(
        //         "I'm thinking</think>Hello, world!\nWhat's up?",
        //         COMMON_CHAT_FORMAT_HERMES_2_PRO));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts_unclosed_unparsed,
            common_chat_parse(
                "I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "I'm thinking</think>Hello, world!\nWhat's up?",
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
        test_templates(tmpls.get(), end_tokens, message_assist_call_python_lines, tools,
                      "<tool_call>\n"
                      "{\"name\": \"python\", \"arguments\": {\"code\": \"# This is a program:\\nprint('hey')\"}}\n"
                      "</tool_call>");
    }
    {
        auto tmpls = read_templates("models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                      common_chat_templates_apply(tmpls.get(), inputs_tools_builtin).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                      common_chat_templates_apply(
                          read_templates("models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja").get(),
                          inputs_tools_builtin)
                          .format);

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

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }
    {
        auto tmpls = read_templates("models/templates/meetkai-functionary-medium-v3.1.jinja");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
                      common_chat_templates_apply(tmpls.get(), inputs_tools).format);

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
            common_chat_msg {
                "assistant",
                "Hello, world!\nnono\nWhat's up?",
                /* .content_parts = */ {},
                /* .tool_calls = */ tool_calls,
                /* .reasoning_content = */ "",
                /* .tool_name = */ "",
                /* .tool_call_id = */ ""
            },
            common_chat_parse(
                "all\n"
                "Hello, world!\n"
                "nono\n"
                "What's up?\n"
                ">>>special_function\n"
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

        assert_equals(COMMON_CHAT_FORMAT_FIREFUNCTION_V2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                      " functools[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]");
    }
    {
        // Original DeepSeek R1 template. Leaves <｜tool▁calls▁begin｜> and others unclosed. Our logic fixes the prompt.
        auto tmpls = read_templates("models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja");
        std::vector<std::string>   end_tokens{ "<｜end▁of▁sentence｜>" };

        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<think>I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts_unclosed_unparsed,
            common_chat_parse(
                "I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));
        assert_msg_equals(message_assist_thoughts,
            // Latest template update (ast of 20250209) adds a trailing <think>\n if add_generation_prompt is true.
            common_chat_parse(
                "I'm thinking</think>Hello, world!\nWhat's up?",
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

        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(tmpls.get(), end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(tmpls.get(), end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<think>I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm thinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "I'm thinking</think>Hello, world!\nWhat's up?",
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
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ false,
                }));
        test_templates(tmpls.get(), end_tokens, message_assist_call, tools,
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>");
    }
}

static void test_msg_diffs_compute() {
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
        diff12.tool_call_delta.name = "special_function";
        diff12.tool_call_delta.id = "123";
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
            std::cout << "\n[chat] All tests passed!" << '\n';
        }
        return 0;
    // } catch (const std::exception & e) {
    //     std::cerr << "Error: " << e.what() << '\n';
    //     return 1;
    // }
}
