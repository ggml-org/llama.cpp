// Offline MiniCPM5 tool-call parse + OpenAI JSON serialization tests.
// Isolates peg-minicpm5 from llama-server to narrow segfault root cause.

#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "testing.h"

#include "nlohmann/json.hpp"

#include <iostream>
#include <string>

using json = nlohmann::ordered_json;

static json minicpm5_tools() {
    return json::parse(
        R"([{"type":"function","function":{"name":"python","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}},{"type":"function","function":{"name":"get_current_timestamp","parameters":{"type":"object","properties":{}}}}])");
}

static common_chat_parser_params make_minicpm5_parser_params(const json & tools, bool parallel_tool_calls) {
    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        const std::string GEN_PROMPT = "<|im_start|>assistant\n";
        auto tool_calls              = p.minicpm5_xml_tool_calls(tools, parallel_tool_calls);
        return p.literal(GEN_PROMPT) + p.content(p.until_one_of({ "<function", " name=\"" })) + tool_calls +
               p.end();
    });

    common_chat_parser_params pp;
    pp.format = COMMON_CHAT_FORMAT_PEG_MINICPM5;
    pp.parser.load(parser.save());
    return pp;
}

static void assert_tool_call(testing & t, const common_chat_msg & msg, const std::string & name) {
    t.assert_equal("tool count", 1u, msg.tool_calls.size());
    if (msg.tool_calls.empty()) {
        return;
    }
    t.assert_equal("tool name", name, msg.tool_calls[0].name);
    t.assert_true("arguments non-empty", !msg.tool_calls[0].arguments.empty());
    auto j = msg.to_json_oaicompat();
    t.assert_true("json dump", !j.dump().empty());
    t.assert_true("has tool_calls in json", j.contains("tool_calls"));
}

static void test_full_parse_cases(testing & t) {
    const auto tools = minicpm5_tools();

    struct case_t {
        const char * label;
        const char * input;
        const char * tool;
        bool         parallel_tool_calls;
    };
    const case_t cases[] = {
        { "full xml",
          "<|im_start|>assistant\n<function name=\"python\"><param name=\"code\">print('Hello, World!')</param></function>",
          "python",
          false },
        { "stripped tags",
          "<|im_start|>assistant\n name=\"python\"> name=\"code\">print('Hello, World!')",
          "python",
          false },
        { "empty args timestamp",
          "<|im_start|>assistant\n<function name=\"get_current_timestamp\"></function>",
          "get_current_timestamp",
          false },
        { "parallel tool calls",
          "<|im_start|>assistant\n"
          "<function name=\"python\"><param name=\"code\">print('x')</param></function>",
          "python",
          true },
    };

    for (const auto & c : cases) {
        t.test(c.label, [&](testing & t) {
            const auto pp = make_minicpm5_parser_params(tools, c.parallel_tool_calls);
            common_chat_msg msg = common_chat_parse(c.input, false, pp);
            assert_tool_call(t, msg, c.tool);
        });
    }
}

static void test_streaming_diffs(testing & t) {
    t.test("streaming partial -> final diffs", [&](testing & t) {
        const auto tools = minicpm5_tools();
        const auto pp    = make_minicpm5_parser_params(tools, false);

        const std::string full =
            "<|im_start|>assistant\n<function name=\"python\"><param name=\"code\">print('Hello')</param></function>";

        common_chat_msg prv;
        std::string generated;
        for (size_t i = 1; i <= full.size(); ++i) {
            generated = full.substr(0, i);
            common_chat_msg cur = common_chat_parse(generated, i < full.size(), pp);
            auto diffs          = common_chat_msg_diff::compute_diffs(prv, cur);
            (void) diffs;
            prv = cur;
        }

        assert_tool_call(t, prv, "python");
    });
}

static void test_set_tool_call_ids(testing & t) {
    t.test("set_tool_call_ids + to_json", [&](testing & t) {
        const auto pp = make_minicpm5_parser_params(minicpm5_tools(), false);
        const std::string input =
            "<|im_start|>assistant\n<function name=\"python\"><param name=\"code\">print('x')</param></function>";

        common_chat_msg msg = common_chat_parse(input, false, pp);
        std::vector<std::string> ids_cache;
        msg.set_tool_call_ids(ids_cache, []() { return std::string("call_test123"); });
        assert_tool_call(t, msg, "python");
    });
}

int main() {
    testing t(std::cout);

    test_full_parse_cases(t);
    test_streaming_diffs(t);
    test_set_tool_call_ids(t);

    return t.summary();
}
