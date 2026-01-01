//  Tests chat handling, including grammar generation and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include "chat.h"
#include "test-chat.h"

#include "common.h"
#include "log.h"

#include "../src/llama-grammar.h"

#include <exception>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <functional>
#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

const char * chat_parser_impl_name(chat_parser_impl impl) {
    switch (impl) {
        case chat_parser_impl::LEGACY:      return "legacy";
        case chat_parser_impl::EXPERIMENTAL: return "experimental";
    }
    return "unknown";
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

common_chat_templates_ptr read_templates(const std::string & path) {
    try {
        return common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, path == "chatml" ? "chatml" : read_file(path)));
    } catch (const std::runtime_error &) {
        return nullptr;
    }
}

std::unique_ptr<llama_grammar> build_grammar(const std::string & grammar_str) {
    return std::unique_ptr<llama_grammar>(
        llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0));
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

// Helper to format a message as OpenAI-compatible JSON for error messages
static std::string msg_to_oai_json(const common_chat_msg & msg) {
    return common_chat_msgs_to_json_oaicompat<json>({msg}).at(0).dump(2);
}

void assert_msg_equals(const common_chat_msg & expected, const common_chat_msg & actual, bool ignore_whitespace_differences) {
    try {
        assert_equals(expected.role, actual.role, "role mismatch");
        if (ignore_whitespace_differences) {
            assert_equals(string_strip(expected.content), string_strip(actual.content), "content mismatch");
        } else {
            assert_equals(expected.content, actual.content, "content mismatch");
        }
        assert_equals(expected.content_parts.size(), actual.content_parts.size(), "content_parts count mismatch");
        for (size_t i = 0; i < expected.content_parts.size(); i++) {
            const auto & expected_part = expected.content_parts[i];
            const auto & actual_part   = actual.content_parts[i];
            assert_equals(expected_part.type, actual_part.type, "content_parts[" + std::to_string(i) + "].type mismatch");
            if (ignore_whitespace_differences) {
                assert_equals(string_strip(expected_part.text), string_strip(actual_part.text),
                              "content_parts[" + std::to_string(i) + "].text mismatch");
            } else {
                assert_equals(expected_part.text, actual_part.text,
                              "content_parts[" + std::to_string(i) + "].text mismatch");
            }
        }
        if (ignore_whitespace_differences) {
            assert_equals(string_strip(expected.reasoning_content), string_strip(actual.reasoning_content),
                          "reasoning_content mismatch");
        } else {
            assert_equals(expected.reasoning_content, actual.reasoning_content, "reasoning_content mismatch");
        }
        assert_equals(expected.tool_calls.size(), actual.tool_calls.size(), "tool_calls count mismatch");
        for (size_t i = 0; i < expected.tool_calls.size(); i++) {
            const auto & expected_tool_call = expected.tool_calls[i];
            const auto & actual_tool_call   = actual.tool_calls[i];
            assert_equals(expected_tool_call.name, actual_tool_call.name,
                          "tool_calls[" + std::to_string(i) + "].name mismatch");
            assert_equals(renormalize_json(expected_tool_call.arguments), renormalize_json(actual_tool_call.arguments),
                          "tool_calls[" + std::to_string(i) + "].arguments mismatch");
            assert_equals(expected_tool_call.id, actual_tool_call.id,
                          "tool_calls[" + std::to_string(i) + "].id mismatch");
        }
    } catch (const std::runtime_error & e) {
        // Re-throw with full JSON context
        throw std::runtime_error(
            std::string(e.what()) +
            "\n\nExpected (OpenAI format):\n" + msg_to_oai_json(expected) +
            "\n\nActual (OpenAI format):\n" + msg_to_oai_json(actual));
    }
}

// Helper to create common_chat_syntax from common_chat_params with optional reasoning format override
common_chat_syntax get_syntax(const common_chat_params & params, common_reasoning_format reasoning_format) {
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

static delta_data init_delta(chat_parser_impl impl,
                             const struct common_chat_templates * tmpls, const std::vector<std::string> & end_tokens,
                             const common_chat_msg & user_message,
                             const common_chat_msg & delta_message,
                             const std::vector<common_chat_tool> & tools,
                             const common_chat_tool_choice & tool_choice,
                             common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE,
                             const std::function<void(common_chat_templates_inputs &)> & customize_inputs = {}) {
    common_chat_templates_inputs inputs;
    inputs.parallel_tool_calls = true;
    inputs.messages.push_back(user_message);
    inputs.tools       = tools;
    inputs.tool_choice = tool_choice;
    // Enable thinking when reasoning is expected - this builds the parser with reasoning block support
    inputs.enable_thinking = (reasoning_format != COMMON_REASONING_FORMAT_NONE);
    if (inputs.enable_thinking) {
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
    // printf("PREFIX: %s\n", prefix.c_str());
    // printf("FULL:   %s\n", full.c_str());
    // printf("common_prefix_length: %d\n", common_prefix_length);
    // printf("DELTA:  %s\n", delta.c_str());

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
    // Use params_prefix for the parser since it's built with add_generation_prompt=true,
    // which correctly sets thinking_forced_open when the template ends with <think>.
    // The delta is extracted by stripping this prefix, so the parser should match accordingly.
    return { delta, params_prefix };
}

/*
  Applies the template to 1 user message w/ add_generation_prompt=true, then w/ the test message w/ add_generation_prompt=false,
  gets the diff, removes any end tokens and parses the result w/ the grammar, checking that
  the parsed message is the same as the test_message
*/
void test_templates(chat_parser_impl impl, const struct common_chat_templates * tmpls, const std::vector<std::string> & end_tokens,
                          const common_chat_msg & test_message,
                          const std::vector<common_chat_tool> & tools,
                          const std::string & expected_delta,
                          bool expect_grammar_triggered,
                          bool test_grammar_if_triggered,
                          common_reasoning_format reasoning_format,
                          bool ignore_whitespace_differences,
                          bool expect_parse_failure,
                          const std::function<void(std::string &)> & mutate_delta) {
    common_chat_msg user_message;
    user_message.role = "user";
    user_message.content = "Hello, world!";

    for (const auto & tool_choice : std::vector<common_chat_tool_choice> {COMMON_CHAT_TOOL_CHOICE_AUTO, COMMON_CHAT_TOOL_CHOICE_REQUIRED}) {
        auto data = init_delta(impl, tmpls, end_tokens, user_message, test_message, tools, tool_choice, reasoning_format, {});
        if (!expected_delta.empty()) {
            if (ignore_whitespace_differences) {
                assert_equals(string_strip(expected_delta), string_strip(data.delta), "delta mismatch (ignoring whitespace)");
            } else {
                assert_equals(expected_delta, data.delta, "delta mismatch");
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
            common_chat_syntax syntax = get_syntax(data.params, reasoning_format);
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
                        if (std::regex_match(constrained, match, std::regex(pattern + ".*"))) {
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
                assert_equals(expect_grammar_triggered, grammar_triggered, "Grammar lazy trigger expectation mismatch");
            }

            if (grammar_triggered && test_grammar_if_triggered && !expect_parse_failure && !match_string(constrained, grammar.get())) {
                throw std::runtime_error("Failed to match delta against grammar:\n\n" + data.delta +
                    "\n\nConstrained: " + constrained +
                    "\n\nGrammar: " + data.params.grammar);
            }
        }
    }
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

// JSON schema for json_schema needle tests
static const char * const NEEDLE_JSON_SCHEMA = R"({
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "notes": {"type": "string"}
    },
    "required": ["amount", "notes"]
})";

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
    bool with_json_schema = false;  // Use json_schema mode instead of free text
    size_t tool_call_count = 1;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    bool expect_tool_ids = false;
    bool enable_thinking = false;
    bool force_disable_thinking = false;
    bool require_thinking_support = false;
    bool require_json_schema_support = false;  // Skip if template doesn't support json_schema
    bool parallel_tool_calls = false;
    bool skip_if_thinking_forced = false;
    size_t args_per_tool_call = 2;
    std::string tool_name = "test_function";
    std::vector<std::string> tool_names;  // For parallel calls with different tools
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

static needle_test_context make_needle_context(const needle_scenario & scenario, common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_format legacy_format = COMMON_CHAT_FORMAT_CONTENT_ONLY) {
    needle_test_context ctx;
    ctx.scenario_name = scenario.name;
    ctx.format = format;
    ctx.expected_msg.role = "assistant";

    if (scenario.with_json_schema) {
        // For json_schema mode, content is JSON with needles embedded in string value
        ctx.has_content = true;
        ctx.content_needles = {NEEDLE1_CONTENT, NEEDLE2_CONTENT};
        // Build JSON content: {"amount": 123.45, "notes": "Before $N1C$ middle $N2C$ after"}
        std::string notes_value = ctx.content_needles.first + ctx.content_needles.second;
        ctx.expected_msg.content = R"({"amount": 123.45, "notes": ")" + notes_value + R"("})";
    } else if (scenario.with_content) {
        ctx.has_content = true;
        ctx.content_needles = {NEEDLE1_CONTENT, NEEDLE2_CONTENT};
        ctx.expected_msg.content = ctx.content_needles.first + ctx.content_needles.second;
    }

    if (scenario.with_reasoning) {
        ctx.has_reasoning = true;
        ctx.reasoning_needles = {NEEDLE1_REASONING, NEEDLE2_REASONING};
        ctx.expected_msg.reasoning_content = ctx.reasoning_needles.first + ctx.reasoning_needles.second;
    }

    if (scenario.with_tool_call) {
        for (size_t call_idx = 0; call_idx < scenario.tool_call_count; ++call_idx) {
            needle_tool_expectation expectation;
            json args = json::object();

            // For parallel calls with different tools, each tool has unique arg keys
            // For same-tool calls, use consistent keys across calls
            bool use_different_tools = !scenario.tool_names.empty();

            for (size_t arg_idx = 0; arg_idx < scenario.args_per_tool_call; ++arg_idx) {
                needle_arg_expectation arg_expect;
                // For different tools: each tool has unique key index (call_idx * args + arg_idx)
                // For same tool: all calls share key indices (arg_idx only)
                size_t key_index = use_different_tools
                    ? (call_idx * scenario.args_per_tool_call + arg_idx)
                    : arg_idx;
                size_t value_index = call_idx * scenario.args_per_tool_call + arg_idx;

                arg_expect.key_needles.first  = make_indexed_needle(NEEDLE1_ARG_KEY, key_index);
                arg_expect.key_needles.second = make_indexed_needle(NEEDLE2_ARG_KEY, key_index);
                arg_expect.value_needles.first  = make_indexed_needle(NEEDLE1_ARG_VALUE, value_index);
                arg_expect.value_needles.second = make_indexed_needle(NEEDLE2_ARG_VALUE, value_index);
                arg_expect.key_text = arg_expect.key_needles.first + arg_expect.key_needles.second;
                arg_expect.value_text = arg_expect.value_needles.first + arg_expect.value_needles.second;

                std::string key = arg_expect.key_text;
                std::string value = arg_expect.value_text;

                args[key] = value;
                expectation.args.push_back(arg_expect);
            }

            common_chat_tool_call call;
            // Use tool_names[call_idx] if available, otherwise fall back to tool_name
            call.name = use_different_tools ? scenario.tool_names[call_idx] : scenario.tool_name;
            call.arguments = args.dump();
            if (scenario.expect_tool_ids) {
                // Mistral Nemo requires 9-character alphanumeric IDs
                if (ctx.format == COMMON_CHAT_FORMAT_MISTRAL_NEMO || legacy_format == COMMON_CHAT_FORMAT_MISTRAL_NEMO) {
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
    // Helper to build error message with expected/actual JSON
    auto make_error = [&](const std::string & msg) {
        return msg +
               "\n\nExpected:\n" + msg_to_oai_json(ctx.expected_msg) +
               "\n\nActual:\n" + msg_to_oai_json(result.final_msg);
    };

    if (ctx.has_content) {
        verify_field_state("Content", result.content_state, ctx.content_needles);
    }
    if (ctx.has_reasoning) {
        verify_field_state("Reasoning", result.reasoning_state, ctx.reasoning_needles);
    }

    if (!ctx.tool_expectations.empty()) {
        if (result.unexpected_tool_count) {
            throw std::runtime_error(make_error(
                "Tool call: Parser produced more tool calls than expected (expected " +
                std::to_string(ctx.tool_expectations.size()) + ", got " +
                std::to_string(result.final_msg.tool_calls.size()) + ")"));
        }
        if (result.final_msg.tool_calls.size() != ctx.tool_expectations.size()) {
            throw std::runtime_error(make_error(
                "Tool call: Final tool call count mismatch (expected " +
                std::to_string(ctx.tool_expectations.size()) + ", got " +
                std::to_string(result.final_msg.tool_calls.size()) + ")"));
        }
        for (size_t call_idx = 0; call_idx < ctx.tool_expectations.size(); ++call_idx) {
            const auto & expectation = ctx.tool_expectations[call_idx];
            const auto & state = result.tool_states[call_idx];
            const auto & final_call = result.final_msg.tool_calls[call_idx];

            if (state.args_regressed) {
                throw std::runtime_error(make_error(
                    "Tool call[" + std::to_string(call_idx) + "]: Arguments regressed (got shorter) during streaming"));
            }

            for (size_t arg_idx = 0; arg_idx < expectation.args.size(); ++arg_idx) {
                const auto & arg_expect = expectation.args[arg_idx];
                if (arg_idx >= state.arg_states.size()) {
                    throw std::runtime_error(make_error(
                        "Tool call[" + std::to_string(call_idx) + "]: Missing argument state in tracker for arg " +
                        std::to_string(arg_idx)));
                }
                const auto & arg_state = state.arg_states[arg_idx];

                verify_field_state("Tool arg key", arg_state.key_state, arg_expect.key_needles);
                verify_field_state("Tool arg value", arg_state.value_state, arg_expect.value_needles);

                // Verify keys stream in order (key N completes before key N+1)
                if (arg_idx > 0) {
                    const auto & prev_state = state.arg_states[arg_idx - 1];
                    if (prev_state.key_completion_seq == 0 || arg_state.key_completion_seq == 0 ||
                        prev_state.key_completion_seq > arg_state.key_completion_seq) {
                        throw std::runtime_error(make_error(
                            "Tool call[" + std::to_string(call_idx) + "]: Argument keys streamed out of order at arg " +
                            std::to_string(arg_idx)));
                    }
                }

                if (final_call.arguments.find(arg_expect.key_text) == std::string::npos) {
                    throw std::runtime_error(make_error(
                        "Tool call[" + std::to_string(call_idx) + "]: Final arguments missing expected key '" +
                        arg_expect.key_text + "'"));
                }
                if (final_call.arguments.find(arg_expect.value_text) == std::string::npos) {
                    throw std::runtime_error(make_error(
                        "Tool call[" + std::to_string(call_idx) + "]: Final arguments missing expected value '" +
                        arg_expect.value_text + "'"));
                }
            }
        }
    }

    assert_msg_equals(ctx.expected_msg, result.final_msg, false);
}

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

void test_peg_parser(chat_parser_impl impl, common_chat_templates * tmpls, const std::function<void(peg_test_case &)> & init) {
    peg_test_case tc;
    init(tc);
    if (tc.params.messages.empty()) {
        tc.params.messages = {message_user};
    }
    if (tc.expect.role.empty()) {
        tc.expect.role = "assistant";
    }
    tc.params.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);

    auto parser = make_peg_parser(tmpls, tc.params);

    common_chat_msg msg_accum;
    common_chat_msg msg_prev;
    msg_accum.role = msg_prev.role = "assistant";

    for (size_t i = 1; i <= tc.input.size(); ++i) {
        auto is_partial = i < tc.input.size();
        common_chat_msg msg_current;
        try {
            msg_current = parser.parse(tc.input.substr(0, i), is_partial);
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("PEG parser exception at input size ") + std::to_string(i) + ": " + e.what() + "\nInput so far:\n" + tc.input.substr(0, i) + "\nGrammar:\n" + tc.params.grammar);
        }

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


static void test_format_detection_with_tools(chat_parser_impl impl, const template_capabilities & info, const common_chat_templates_ptr & tmpls) {
    // Apply template with tools and experimental_new_parsers
    common_chat_templates_inputs inputs;
    inputs.messages = {message_user};
    inputs.tools = {python_tool};
    inputs.experimental_new_parsers = impl == chat_parser_impl::EXPERIMENTAL;

    common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);

    auto expected_format = impl == chat_parser_impl::LEGACY ? info.legacy_format : info.experimental_format;
    assert_equals(
        common_chat_format_name(expected_format),
        common_chat_format_name(params.format));

    if (impl == chat_parser_impl::EXPERIMENTAL) {
        assert_equals(false, params.grammar.empty());
        assert_equals(false, params.parser.empty());
    }
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

    {
        needle_scenario tools_disabled;
        tools_disabled.name = "tools-available-but-disabled";
        tools_disabled.provide_tools = true;
        tools_disabled.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        tools_disabled.with_tool_call = false;
        scenarios.push_back(tools_disabled);
    }

    {
        needle_scenario tool_auto;
        tool_auto.name = "tool-auto-single";
        tool_auto.provide_tools = true;
        tool_auto.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        tool_auto.with_tool_call = true;
        tool_auto.with_content = (info.tools_emit_content_with_calls == ToolsEmitContentWithCalls::Yes);
        tool_auto.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
        scenarios.push_back(tool_auto);
    }

    {
        needle_scenario tool_required_only;
        tool_required_only.name = "tool-required-only";
        tool_required_only.provide_tools = true;
        tool_required_only.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        tool_required_only.with_tool_call = true;
        tool_required_only.with_content = false;  // to
        tool_required_only.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
        scenarios.push_back(tool_required_only);
    }

    {
        needle_scenario tool_parallel;
        tool_parallel.name = "parallel-tool-calls";
        tool_parallel.provide_tools = true;
        tool_parallel.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        tool_parallel.with_tool_call = true;
        tool_parallel.tool_call_count = 2;
        tool_parallel.parallel_tool_calls = true;
        // Use two different tools so each has its own schema/args
        // This tests realistic parallel calls and verifies streaming order
        tool_parallel.tool_names = {"tool_alpha", "tool_beta"};
        tool_parallel.args_per_tool_call = 1;  // 1 arg per tool for simpler verification
        tool_parallel.with_content = (info.tools_emit_content_with_calls == ToolsEmitContentWithCalls::Yes);
        tool_parallel.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
        scenarios.push_back(tool_parallel);
    }

    if (info.supports_thinking == ThinkingSupport::Yes) {
        needle_scenario tool_with_reasoning;
        tool_with_reasoning.name = "tool-with-reasoning";
        tool_with_reasoning.provide_tools = true;
        tool_with_reasoning.with_tool_call = true;
        tool_with_reasoning.with_reasoning = true;
        tool_with_reasoning.enable_thinking = true;
        tool_with_reasoning.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        tool_with_reasoning.require_thinking_support = true;
        tool_with_reasoning.with_content = (info.tools_emit_content_with_calls == ToolsEmitContentWithCalls::Yes);
        tool_with_reasoning.expect_tool_ids = (info.tool_calls_have_ids == ToolCallsHaveIds::Yes);
        scenarios.push_back(tool_with_reasoning);
    }

    {
        // Basic json_schema test without reasoning
        needle_scenario json_schema_basic;
        json_schema_basic.name = "json-schema-basic";
        json_schema_basic.with_json_schema = true;
        json_schema_basic.with_content = false;  // content is JSON, handled by with_json_schema
        json_schema_basic.require_json_schema_support = true;
        json_schema_basic.force_disable_thinking = true;
        json_schema_basic.skip_if_thinking_forced = true;
        scenarios.push_back(json_schema_basic);
    }
    // json_schema with reasoning (if supported)
    if (info.supports_thinking == ThinkingSupport::Yes && info.reasoning_requires_tools == ReasoningRequiresTools::No) {
        needle_scenario json_schema_with_reasoning;
        json_schema_with_reasoning.name = "json-schema-with-reasoning";
        json_schema_with_reasoning.with_json_schema = true;
        json_schema_with_reasoning.with_content = false;
        json_schema_with_reasoning.with_reasoning = true;
        json_schema_with_reasoning.enable_thinking = true;
        json_schema_with_reasoning.require_json_schema_support = true;
        json_schema_with_reasoning.require_thinking_support = true;
        scenarios.push_back(json_schema_with_reasoning);
    }

    return scenarios;
}

void run_template_test_suite(chat_parser_impl impl, const template_capabilities & template_caps, const common_chat_templates_ptr & tmpls) {
    test_format_detection_with_tools(impl, template_caps, tmpls);
    
    // The rest of this test is only working / green for new peg parsers
    if (impl != chat_parser_impl::EXPERIMENTAL) {
        return;
    }
    
    if (template_caps.supports_disable_thinking == SupportsDisableThinking::Yes) {
        common_chat_templates_inputs inputs;
        inputs.messages.push_back(message_user);
        inputs.experimental_new_parsers = true;
        inputs.enable_thinking = false;

        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        assert_equals(false, params.thinking_forced_open, "thinking should not be forced open when thinking is disabled");
    }

    // if (template_caps.name != "Command R7B")
    if (false) // TODO(ochafik): debug this!
    {
        // Check that required mode forbids content but allows thoughts
        const auto parse_delta_required = [&](const common_chat_msg & delta_msg, common_reasoning_format reasoning_format) {
            const auto data = init_delta(chat_parser_impl::EXPERIMENTAL, tmpls.get(), template_caps.end_tokens, message_user, delta_msg, {python_tool},
                COMMON_CHAT_TOOL_CHOICE_REQUIRED, reasoning_format, {});
            std::cout << data.delta << "\n" << std::flush;
            return common_chat_parse(data.delta, false, get_syntax(data.params, reasoning_format));
        };

        assert_throws([&]() {
            parse_delta_required(
                simple_assist_msg("Hello, this is just content without any tool call."),
                COMMON_REASONING_FORMAT_NONE);
        }, "required mode forbids content");

        if (template_caps.supports_thinking == ThinkingSupport::Yes) {

            parse_delta_required(
                simple_assist_msg("", "Let me think about this..."),
                COMMON_REASONING_FORMAT_DEEPSEEK);

            assert_throws([&]() {
                parse_delta_required(
                    simple_assist_msg("Here is my response.", "Let me think about this..."),
                    COMMON_REASONING_FORMAT_DEEPSEEK);
            }, "required mode forbids content");
        }
    }

    // TODO(ochafik): unroll these as function calls
    auto scenarios = build_needle_scenarios(template_caps);

    for (const auto & scenario : scenarios) {
        if (scenario.require_thinking_support && template_caps.supports_thinking == ThinkingSupport::No) {
            continue;
        }
        if (scenario.force_disable_thinking && template_caps.supports_disable_thinking == SupportsDisableThinking::No) {
            // Skip scenarios that require disabling thinking when the template doesn't support it
            // (e.g., Kimi template always outputs <think></think> tags regardless of enable_thinking)
            continue;
        }
        if (scenario.parallel_tool_calls && !common_chat_templates_support_parallel_tool_calls(tmpls.get())) {
            continue;
        }

        std::string debug_info;  // Collect debug info to print on failure only
        try {
            // Override tool name if template specifies a custom one
            // auto scenario_copy = scenario;
            // if (template_caps.needle_tool_name != nullptr) {
            //     scenario_copy.tool_name = template_caps.needle_tool_name;
            // }

            auto ctx = make_needle_context(scenario, template_caps.experimental_format, template_caps.legacy_format);
            std::vector<common_chat_tool> scenario_tools;
            if (scenario.provide_tools) {
                // Create dynamic tools with parameter names matching the needle markers
                // This is needed for parsers that use literal_tag for parameter names (e.g., Llama 3.1 builtin tools)
                if (!ctx.expected_msg.tool_calls.empty()) {
                    // For parallel calls with different tools, create one tool per tool_name
                    // For same-tool calls, create a single tool
                    bool use_different_tools = !scenario.tool_names.empty();

                    if (use_different_tools) {
                        // Create separate tools for each tool_name
                        for (size_t i = 0; i < ctx.expected_msg.tool_calls.size(); ++i) {
                            const auto& call = ctx.expected_msg.tool_calls[i];
                            common_chat_tool tool;
                            tool.name = call.name;
                            tool.description = "Dynamic tool for needle testing";

                            json properties = json::object();
                            json required = json::array();

                            if (!call.arguments.empty()) {
                                json args_json = json::parse(call.arguments);
                                for (const auto & [key, value] : args_json.items()) {
                                    properties[key] = {
                                        {"type", "string"},
                                        {"description", "Needle test parameter"}
                                    };
                                    required.push_back(key);
                                }
                            }

                            tool.parameters = json({
                                {"type", "object"},
                                {"properties", properties},
                                {"required", required}
                            }).dump();
                            scenario_tools.push_back(tool);
                        }
                    } else {
                        // Single tool with schema from first call
                        common_chat_tool dynamic_tool;
                        dynamic_tool.name = scenario.tool_name;
                        dynamic_tool.description = "Dynamic tool for needle testing";

                        json properties = json::object();
                        json required = json::array();

                        const auto& first_call = ctx.expected_msg.tool_calls[0];
                        if (!first_call.arguments.empty()) {
                            json args_json = json::parse(first_call.arguments);
                            for (const auto & [key, value] : args_json.items()) {
                                properties[key] = {
                                    {"type", "string"},
                                    {"description", "Needle test parameter"}
                                };
                                required.push_back(key);
                            }
                        }

                        dynamic_tool.parameters = json({
                            {"type", "object"},
                            {"properties", properties},
                            {"required", required}
                        }).dump();
                        scenario_tools = {dynamic_tool};
                    }
                } else {
                    scenario_tools = {python_tool};
                }
            }

            auto reasoning_format = scenario.with_reasoning ? COMMON_REASONING_FORMAT_DEEPSEEK : COMMON_REASONING_FORMAT_NONE;

            auto data = init_delta(chat_parser_impl::EXPERIMENTAL, tmpls.get(), template_caps.end_tokens, message_user, ctx.expected_msg, scenario_tools,
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
                                        // Set json_schema for structured output tests
                                        if (scenario.with_json_schema) {
                                            inputs.json_schema = NEEDLE_JSON_SCHEMA;
                                        }
                                    });

            if (scenario.skip_if_thinking_forced && data.params.thinking_forced_open) {
                continue;
            }
            if (scenario.force_disable_thinking && data.params.thinking_forced_open) {
                continue;
            }

            if (data.params.parser.empty()) {
                throw std::runtime_error("Template returned empty parser definition");
            }

            auto syntax = get_syntax(data.params, reasoning_format);
            if (syntax.parser.empty()) {
                throw std::runtime_error("PEG arena failed to load");
            }

            auto parse_fn = [&](const std::string & msg, bool is_partial) mutable {
                return common_chat_peg_parse(syntax.parser, msg, is_partial, syntax);
            };

            std::string raw_message = data.delta;
            debug_info = "    delta len=" + std::to_string(data.delta.size()) + ": '" + data.delta + "'\n";

            if (template_caps.inject_reasoning_after_format == InjectReasoningAfterFormat::Yes && scenario.with_reasoning &&
                raw_message.find(ctx.reasoning_needles.first) == std::string::npos) {
                const char * open = template_caps.think_open_tag ? template_caps.think_open_tag : "<think>";
                const char * close = template_caps.think_close_tag ? template_caps.think_close_tag : "</think>";
                std::string prefix;
                if (data.params.thinking_forced_open) {
                    // When thinking is forced open, prompt ends with <think> - we need content + closing tag
                    prefix = ctx.expected_msg.reasoning_content + std::string(close);
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

            debug_info += "    raw_message len=" + std::to_string(raw_message.size()) + ": '" + raw_message + "'\n";
            debug_info += "    grammar:\n" + data.params.grammar + "\n";

            auto result = test_streaming_with_needles(ctx, raw_message, parse_fn);
            verify_needle_results(ctx, result);

            // Also test diff computation - this is what the server uses for SSE streaming.
            // This catches bugs that test_streaming_with_needles misses because it exercises
            // common_chat_msg_diff::compute_diffs().
            test_parser_with_streaming(
                ctx.expected_msg,
                raw_message,
                [&](const std::string & msg) {
                    // Use is_partial=true for partial messages, is_partial=false for the full message
                    return parse_fn(msg, msg.size() < raw_message.size());
                });
        } catch (const std::exception & e) {
            throw std::runtime_error(scenario.name + " failed for " + template_caps.name + ": " + e.what() + "\n" + debug_info);
        }
    }
}


static void test_chat_parsers()
{
    printf("[%s]\n", __func__);

    const auto * filter = getenv("TEST");
    
    enum class test_status { Enabled, Disabled };
    enum class test_outcome { Passed, Failed, Skipped };
    struct test_result {
        std::string name;
        test_outcome outcome;
    };
    std::vector<test_result> results;

    auto test_chat_parser = [&](test_status status, const std::string & name, chat_parser_impl impl, const std::function<void(chat_parser_impl)> & test_fn)
    {
        auto full_name = name + ":" + chat_parser_impl_name(impl);
        auto matches_filter = filter && full_name.find(filter) != std::string::npos;
        if (!(filter && filter == std::string("all"))) {
            if (status == test_status::Enabled) {
                if (filter && !matches_filter) {
                    return;
                }
            } else {
                if (!filter) {
                    printf("[%s] ⚠️ Skipping disabled test\n", full_name.c_str());
                    results.push_back({full_name, test_outcome::Skipped});
                    return;
                }
                if  (!matches_filter && filter != std::string("skipped")) {
                    return;
                }
            }
        }
        printf("[%s]\n", full_name.c_str());

        try {
            test_fn(impl);
            printf("[%s] ✅︎ SUCCESS\n", full_name.c_str());
            results.push_back({full_name, test_outcome::Passed});
        } catch (const std::exception & ex) {
            // Print
            printf("[%s] ❌ FAILURE\n%s\n", full_name.c_str(), ex.what());
            results.push_back({full_name, test_outcome::Failed});
        }
    };
    
    test_chat_parser(test_status::Enabled, "apertus", chat_parser_impl::LEGACY, test_apertus_parser);
    test_chat_parser(test_status::Enabled, "apertus", chat_parser_impl::EXPERIMENTAL, test_apertus_parser);

    test_chat_parser(test_status::Enabled, "apriel_1_5", chat_parser_impl::LEGACY, test_apriel_1_5_parser);
    test_chat_parser(test_status::Enabled, "apriel_1_5", chat_parser_impl::EXPERIMENTAL, test_apriel_1_5_parser);

    test_chat_parser(test_status::Enabled, "command_r7b", chat_parser_impl::LEGACY, test_command_r7b_parser);
    test_chat_parser(test_status::Enabled, "command_r7b", chat_parser_impl::EXPERIMENTAL, test_command_r7b_parser);

    test_chat_parser(test_status::Enabled, "deepseek_r1", chat_parser_impl::LEGACY, test_deepseek_r1_parser);
    test_chat_parser(test_status::Enabled, "deepseek_r1", chat_parser_impl::EXPERIMENTAL, test_deepseek_r1_parser);

    test_chat_parser(test_status::Enabled, "deepseek_v3_1", chat_parser_impl::LEGACY, test_deepseek_v3_1_parser);
    test_chat_parser(test_status::Enabled, "deepseek_v3_1", chat_parser_impl::EXPERIMENTAL, test_deepseek_v3_1_parser);

    test_chat_parser(test_status::Enabled, "firefunction_v2", chat_parser_impl::LEGACY, test_firefunction_v2_parser);
    test_chat_parser(test_status::Enabled, "firefunction_v2", chat_parser_impl::EXPERIMENTAL, test_firefunction_v2_parser);

    test_chat_parser(test_status::Enabled, "functionary_v3_1_llama_3_1", chat_parser_impl::LEGACY, test_functionary_v3_1_llama_3_1_parser);
    test_chat_parser(test_status::Enabled, "functionary_v3_1_llama_3_1", chat_parser_impl::EXPERIMENTAL, test_functionary_v3_1_llama_3_1_parser);

    test_chat_parser(test_status::Enabled, "functionary_v3_2", chat_parser_impl::LEGACY, test_functionary_v3_2_parser);
    test_chat_parser(test_status::Enabled, "functionary_v3_2", chat_parser_impl::EXPERIMENTAL, test_functionary_v3_2_parser);

    test_chat_parser(test_status::Enabled, "generic", chat_parser_impl::LEGACY, test_generic_parser);
    test_chat_parser(test_status::Enabled, "generic", chat_parser_impl::EXPERIMENTAL, test_generic_parser);

    test_chat_parser(test_status::Enabled, "glm_4_5", chat_parser_impl::LEGACY, test_glm_4_5_parser);
    test_chat_parser(test_status::Enabled, "glm_4_5", chat_parser_impl::EXPERIMENTAL, test_glm_4_5_parser);

    test_chat_parser(test_status::Enabled, "gpt_oss", chat_parser_impl::LEGACY, test_gpt_oss_parser);
    test_chat_parser(test_status::Enabled, "gpt_oss", chat_parser_impl::EXPERIMENTAL, test_gpt_oss_parser);

    test_chat_parser(test_status::Enabled, "granite", chat_parser_impl::LEGACY, test_granite_parser);
    test_chat_parser(test_status::Enabled, "granite", chat_parser_impl::EXPERIMENTAL, test_granite_parser);

    test_chat_parser(test_status::Enabled, "hermes_2_pro", chat_parser_impl::LEGACY, test_hermes_2_pro_parser);
    test_chat_parser(test_status::Enabled, "hermes_2_pro", chat_parser_impl::EXPERIMENTAL, test_hermes_2_pro_parser);

    test_chat_parser(test_status::Enabled, "kimi_k2", chat_parser_impl::LEGACY, test_kimi_k2_parser);
    // Note: skips run_template_test_suite due to Kimi's reasoning message splitting
    test_chat_parser(test_status::Enabled, "kimi_k2", chat_parser_impl::EXPERIMENTAL, test_kimi_k2_parser);

    test_chat_parser(test_status::Enabled, "lfm2", chat_parser_impl::LEGACY, test_lfm2_parser);
    // TODO
    test_chat_parser(test_status::Disabled, "lfm2", chat_parser_impl::EXPERIMENTAL, test_lfm2_parser);

    test_chat_parser(test_status::Enabled, "llama_3_x", chat_parser_impl::LEGACY, test_llama_3_x_parser);
    // TODO(ochafik): this peg parser needs both TOOL_ARG_NAME (builtins) and TOOL_ARGS (regular) so will need its own mapper
    test_chat_parser(test_status::Disabled, "llama_3_x", chat_parser_impl::EXPERIMENTAL, test_llama_3_x_parser);

    test_chat_parser(test_status::Enabled, "magistral", chat_parser_impl::LEGACY, test_magistral_parser);
    test_chat_parser(test_status::Enabled, "magistral", chat_parser_impl::EXPERIMENTAL, test_magistral_parser);

    test_chat_parser(test_status::Enabled, "minimax_m2", chat_parser_impl::LEGACY, test_minimax_m2_parser);
    test_chat_parser(test_status::Enabled, "minimax_m2", chat_parser_impl::EXPERIMENTAL, test_minimax_m2_parser);

    test_chat_parser(test_status::Enabled, "ministral_3", chat_parser_impl::LEGACY, test_ministral_3_parser);
    test_chat_parser(test_status::Enabled, "ministral_3", chat_parser_impl::EXPERIMENTAL, test_ministral_3_parser);

    test_chat_parser(test_status::Enabled, "mistral_nemo", chat_parser_impl::LEGACY, test_mistral_nemo_parser);
    test_chat_parser(test_status::Enabled, "mistral_nemo", chat_parser_impl::EXPERIMENTAL, test_mistral_nemo_parser);

    test_chat_parser(test_status::Enabled, "nemotron_v2", chat_parser_impl::LEGACY, test_nemotron_v2_parser);
    test_chat_parser(test_status::Enabled, "nemotron_v2", chat_parser_impl::EXPERIMENTAL, test_nemotron_v2_parser);

    test_chat_parser(test_status::Enabled, "nemotron_v3", chat_parser_impl::LEGACY, test_nemotron_v3_parser);
    test_chat_parser(test_status::Enabled, "nemotron_v3", chat_parser_impl::EXPERIMENTAL, test_nemotron_v3_parser);

    test_chat_parser(test_status::Enabled, "qwen3_coder_xml", chat_parser_impl::LEGACY, test_qwen3_coder_xml_parser);
    test_chat_parser(test_status::Enabled, "qwen3_coder_xml", chat_parser_impl::EXPERIMENTAL, test_qwen3_coder_xml_parser);

    test_chat_parser(test_status::Enabled, "seed_oss", chat_parser_impl::LEGACY, test_seed_oss_parser);
    // TODO(ochafik): debug (not sure why we have an experimental-only section, it explodes)
    test_chat_parser(test_status::Disabled, "seed_oss", chat_parser_impl::EXPERIMENTAL, test_seed_oss_parser);

    test_chat_parser(test_status::Enabled, "xiaomi_mimo", chat_parser_impl::LEGACY, test_xiaomi_mimo_parser);
    test_chat_parser(test_status::Enabled, "xiaomi_mimo", chat_parser_impl::EXPERIMENTAL, test_xiaomi_mimo_parser);

    std::cout << std::flush;
    std::cerr << std::flush;

    size_t skipped_count = 0;
    size_t success_count = 0;
    size_t error_count = 0;
    printf("\n[%s] Summary:\n", __func__);
    for (const auto & result : results) {
        std::string icon;
        std::string text;
        if (result.outcome == test_outcome::Skipped) {
            icon = "⚠️";
            text = "SKIPPED";
            skipped_count++;
        } else if (result.outcome == test_outcome::Failed) {
            icon = "❌";
            text = "FAILURE";
            error_count++;
        } else if (result.outcome == test_outcome::Passed) {
            icon = "✅︎";
            text = "SUCCESS";
            success_count++;
        }
        printf("- %s %s (%s)\n", icon.c_str(), result.name.c_str(), text.c_str());
    }
    printf("[%s] %s Passed (%zu / %zu) tests, skipped %zu\n", __func__, error_count ? "❌" : "✅︎", success_count, success_count + error_count, skipped_count);
    if (error_count) {
        throw std::runtime_error("Test failed");
    }
}

static const char * tool_choice_name(common_chat_tool_choice choice) {
    switch (choice) {
        case COMMON_CHAT_TOOL_CHOICE_AUTO: return "auto";
        case COMMON_CHAT_TOOL_CHOICE_REQUIRED: return "required";
        case COMMON_CHAT_TOOL_CHOICE_NONE: return "none";
    }
    return "unknown";
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
    if (scenario.with_json_schema) {
        oss << ", json_schema";
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
    // common_log_set_verbosity_thold(999);

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
                    const auto & name = parts[parts.size() - 1];
                    const auto & format = common_chat_format_name(common_chat_templates_apply(tmpls.get(), inputs).format);
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
            test_chat_parsers();
            
            std::cout << "\n[chat] All tests passed!" << '\n';
        }
        return 0;
}
