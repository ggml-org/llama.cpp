//  Tests chat handling, including grammar generation and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include <exception>
#include <iostream>
#include <string>

#include "chat-parser.h"
#include "common.h"
#include "log.h"
#include "regex-partial.h"

template <class T>
static void assert_equals(const std::string_view label, const T & expected, const T & actual) {
    if (expected != actual) {
        std::cerr << label << std::endl;
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

template <class T>
static void assert_equals(const T & expected, const T & actual) {
    assert_equals("", expected, actual);
}
static void assert_equals(const char * expected, const std::string & actual) {
  return assert_equals<std::string>(expected, actual);
}

static void assert_throws(const std::function<void()> & fn, const std::string & expected_exception_pattern = "") {
    try {
        fn();
    } catch (const std::exception & e) {
      if (expected_exception_pattern.empty()) {
          return;
        }
        std::regex expected_exception_regex(expected_exception_pattern);
        std::string actual_message = e.what();
        if (std::regex_search(actual_message, expected_exception_regex)) {
            return;
        }
        throw std::runtime_error("Exception doesn't match expected pattern: " + actual_message + " (pattern: " + expected_exception_pattern + ")");
        throw std::runtime_error("Exception of unexpected type: " + std::string(e.what()));
    }
    throw std::runtime_error("Exception was expected but not thrown");
}

static void test_reasoning() {
  //common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);
  {
    common_chat_msg_parser builder("<tnk>Cogito</tnk>Ergo sum", /* is_partial= */ false, {
        /* .format = */ COMMON_CHAT_FORMAT_CONTENT_ONLY,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_NONE,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ false,
    });
    assert_equals(false, builder.try_parse_reasoning("<tnk>", "</tnk>"));
    assert_equals("<tnk>Cogito</tnk>Ergo sum", builder.consume_rest());
  }
  {
    common_chat_msg_parser builder("<tnk>Cogito</tnk>Ergo sum", /* is_partial= */ false, {
        /* .format = */ COMMON_CHAT_FORMAT_CONTENT_ONLY,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ false,
    });
    assert_equals(true, builder.try_parse_reasoning("<tnk>", "</tnk>"));
    assert_equals(std::string("Cogito"), builder.result().reasoning_content);
    assert_equals("Ergo sum", builder.consume_rest());
  }
  {
    common_chat_msg_parser builder("Cogito</tnk>Ergo sum", /* is_partial= */ false, {
        /* .format = */ COMMON_CHAT_FORMAT_CONTENT_ONLY,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_NONE,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ false,
    });
    assert_equals(false, builder.try_parse_reasoning("<tnk>", "</tnk>"));
    assert_equals("Cogito</tnk>Ergo sum", builder.consume_rest());
  }
  {
    common_chat_msg_parser builder("Cogito</tnk>Ergo sum", /* is_partial= */ false, {
        /* .format = */ COMMON_CHAT_FORMAT_CONTENT_ONLY,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ true,
    });
    assert_equals(true, builder.try_parse_reasoning("<tnk>", "</tnk>"));
    assert_equals(std::string("Cogito"), builder.result().reasoning_content);
    assert_equals("Ergo sum", builder.consume_rest());
  }
  {
    common_chat_msg_parser builder("Cogito</tnk>Ergo sum", /* is_partial= */ false, {
        /* .format = */ COMMON_CHAT_FORMAT_CONTENT_ONLY,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
        /* .reasoning_in_content = */ true,
        /* .thinking_forced_open = */ true,
    });
    assert_equals(true, builder.try_parse_reasoning("<tnk>", "</tnk>"));
    assert_equals("<think>Cogito</think>", builder.result().content);
    assert_equals("Ergo sum", builder.consume_rest());
  }
  {
    const std::string variant("content_only_inline_think");
    common_chat_syntax syntax = {
        /* .format = */ COMMON_CHAT_FORMAT_CONTENT_ONLY,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ false,
        /* .parse_tool_calls = */ false,
    };
    const std::string input = "<think>Pense</think>Bonjour";
    auto msg = common_chat_parse(input, false, syntax);
    assert_equals(variant, std::string("Pense"), msg.reasoning_content);
    assert_equals(variant, std::string("Bonjour"), msg.content);
  }
  {
    const std::string variant("llama_3_inline_think");
    common_chat_syntax syntax = {
        /* .format = */ COMMON_CHAT_FORMAT_LLAMA_3_X,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ false,
        /* .parse_tool_calls = */ false,
    };
    const std::string input = "<think>Plan</think>Réponse";
    auto msg = common_chat_parse(input, false, syntax);
    assert_equals(variant, std::string("Plan"), msg.reasoning_content);
    assert_equals(variant, std::string("Réponse"), msg.content);
  }
  // Test DeepSeek V3.1 parsing - reasoning content followed by "</think>" and then regular content
  {
    common_chat_syntax syntax = {
        /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ true,
        /* .parse_tool_calls = */ true,
    };
    const std::string variant("deepseek_v3_1_reasoning_format_deepseek");
    common_chat_msg_parser builder("REASONING</think>ok", /* is_partial= */ false, syntax);
    assert_equals(variant, true, builder.try_parse_reasoning("<think>", "</think>"));
    assert_equals(variant, std::string("REASONING"), builder.result().reasoning_content);
    assert_equals(variant, std::string("ok"), builder.consume_rest());
  }
  // Test DeepSeek V3.1 parsing - reasoning_format none - reasoning content followed by "</think>" and then regular content
  {
    common_chat_syntax syntax = {
        /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_NONE,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ true,
        /* .parse_tool_calls = */ true,
    };
    const std::string variant("deepseek_v3_1_reasoning_format_none");
    const std::string input = "REASONING</think>ok";
    auto msg = common_chat_parse(input, false, syntax);
    assert_equals(variant, std::string("REASONING</think>ok"), msg.content);
    assert_equals(variant, std::string(""), msg.reasoning_content);
  }
}

static void test_regex() {
  auto test_throws = [](const std::string & input, const std::string & regex, const std::string & expected_exception_pattern = "") {
    common_chat_msg_parser builder(input, /* is_partial= */ false, {});
    assert_throws([&]() { builder.consume_regex(common_regex(regex)); }, expected_exception_pattern);
  };

  test_throws("Hello, world!", "abc", "^abc$");
  test_throws("Hello, world!", "e", "^e$");

  {
    common_chat_msg_parser builder("Hello, world!", /* is_partial= */ false, {});
    builder.consume_regex(common_regex("Hello"));
    assert_equals(", world!", builder.consume_rest());
  }

  {
    // When in non partial mode, we can say whether the regex was consumed or not.
    common_chat_msg_parser builder("Hello,", /* is_partial= */ false, {});
    assert_equals(false, builder.try_consume_regex(common_regex("Hello, world!")).has_value());
  }
  {
    common_chat_msg_parser builder("Hello,", /* is_partial= */ false, {});
    auto res = builder.try_consume_regex(common_regex("H(el)l(?:o, world!)?"));
    assert_equals(true, res.has_value());
    // Verify captures
    assert_equals<size_t>(2, res->groups.size());
    assert_equals("Hell", builder.str(res->groups[0]));
    assert_equals("el", builder.str(res->groups[1]));
    // Verify position is after the match
    assert_equals<size_t>(4, builder.pos());
    assert_equals("o,", builder.consume_rest());
  }
  {
    // But in partial mode, we have a partial final match / can't decide, so we throw a partial exception.
    common_chat_msg_parser builder("Hello,", /* is_partial= */ true, {});
    assert_throws([&]() {
      builder.try_consume_regex(common_regex("Hello, world!"));
    }, "^Hello, world!$");
  }

  // Now regardless of the mode, we can tell these aren't a match.
  for (const auto is_partial : {false, true}) {
    common_chat_msg_parser builder("Hello,", is_partial, {});
    assert_equals(false, builder.try_consume_regex(common_regex("a(b|c)(d|e)f")).has_value());
  }
  for (const auto is_partial : {false, true}) {
    common_chat_msg_parser builder("Hello,", is_partial, {});
    assert_equals(false, builder.try_consume_literal("Oh"));
  }
}

const std::vector<std::string> barely_healable_jsons = {
  "{",
  "{\"",
  "{\"\\",
  "{\"n",
  "{\"name\"",
  "{\"name\":",
  "{\"name\":\"",
  "{\"name\":\"\\",
  "{\"name\":\"python",
  "{\"name\":\"python\\",
  "{\",",
  "{\":",
  "{\"[",
  "{\"]",
  "{\"{",
  "{\"}",
  "{\"1",
  "{\"name\":\",",
  "{\"name\":\":",
  "{\"name\":\"[",
  "{\"name\":\"]",
  "{\"name\":\"{",
  "{\"name\":\"}",
  "{\"name\":\"1",
};

static void test(const std::string & input, bool is_partial, const std::vector<std::vector<std::string>> & args_paths, const std::vector<std::vector<std::string>> & content_paths, const std::string & expected) {
  common_chat_msg_parser builder(input, is_partial, {});
  auto js = builder.try_consume_json_with_dumped_args(args_paths, content_paths);
  assert_equals(true, js.has_value());
  assert_equals(is_partial, js->is_partial);
  assert_equals(expected, args_paths.size() == 1 && args_paths[0].empty() ? js->value.get<std::string>() : js->value.dump());
}

static void test_deepseek_v3_1_tool_calls() {
    //common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);
    // variant: happy path for when it works as the model card says it should
    const std::string variant("simple");
    common_chat_syntax syntax = {
        /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ false,
        /* .parse_tool_calls = */ true,
    };
    const std::string input = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    auto msg = common_chat_parse(input, false, syntax);
    assert_equals<std::size_t>(variant, 1, msg.tool_calls.size());
    assert_equals(variant, std::string("get_time"), msg.tool_calls[0].name);
    // JSON arguments are dumped without spaces
    assert_equals(variant, std::string("{\"city\":\"Tokyo\"}"), msg.tool_calls[0].arguments);
    assert_equals(variant, std::string(""), msg.content);
    assert_equals(variant, std::string(""), msg.reasoning_content);

    // variant: simple + thinking open
    {
        common_chat_syntax syntax = {
            /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
            /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            /* .reasoning_in_content = */ false,
            /* .thinking_forced_open = */ true,
            /* .parse_tool_calls = */ true,
        };
        const std::string variant("simple_thinking");
        const std::string in = "REASONING</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        auto m = common_chat_parse(in, false, syntax);
        assert_equals<std::size_t>(variant, 1, m.tool_calls.size());
        assert_equals(variant, std::string("get_time"), m.tool_calls[0].name);
        assert_equals(variant, std::string("{\"city\":\"Tokyo\"}"), m.tool_calls[0].arguments);
        assert_equals(variant, std::string(""), m.content);
        assert_equals(variant, std::string("REASONING"), m.reasoning_content);
    }
    // variant: simple + multiple tool calls
    {
        common_chat_syntax syntax = {
            /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
            /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            /* .reasoning_in_content = */ false,
            /* .thinking_forced_open = */ false,
            /* .parse_tool_calls = */ true,
        };
        const std::string variant("simple_multiple_tool_calls");
        const std::string in = "CONTENT<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        auto m = common_chat_parse(in, false, syntax);
        assert_equals<std::size_t>(variant, 2, m.tool_calls.size());
        assert_equals(variant, std::string("get_time"), m.tool_calls[0].name);
        assert_equals(variant, std::string("{\"city\":\"Paris\"}"), m.tool_calls[0].arguments);
        assert_equals(variant, std::string("get_weather"), m.tool_calls[1].name);
        assert_equals(variant, std::string("{\"city\":\"Paris\"}"), m.tool_calls[1].arguments);
        assert_equals(variant, std::string("CONTENT"), m.content);
        assert_equals(variant, std::string(""), m.reasoning_content);
    }


    // variant: thinking forced open + tool call in reasoning content
    {
        common_chat_syntax syntax = {
            /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
            /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            /* .reasoning_in_content = */ false,
            /* .thinking_forced_open = */ true,
            /* .parse_tool_calls = */ true,
        };
        const std::string variant("thinking_forced_open_tool_call_in_reasoning");
        const std::string in = "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time2<｜tool▁sep｜>{\"city\": \"Tokyo2\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>REASONING</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        auto m = common_chat_parse(in, false, syntax);
        assert_equals<std::size_t>(variant, 1, m.tool_calls.size());
        assert_equals(variant, std::string("get_time"), m.tool_calls[0].name);
        assert_equals(variant, std::string("{\"city\":\"Tokyo\"}"), m.tool_calls[0].arguments);
        assert_equals(variant, std::string(""), m.content);
        assert_equals(variant, std::string("REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time2<｜tool▁sep｜>{\"city\": \"Tokyo2\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>REASONING"), m.reasoning_content);
    }

    // variant: thinking forced open + tool call in reasoning content + no closing think + not partial
    //          This is a bit of a fine tuning issue on the model's part IMO. It really should not be attempting
    //          to make tool calls in reasoning content according to the model card, but it does sometimes, so
    //          add the reasoning content as regular content and parse the tool calls.
    {
        common_chat_syntax syntax = {
            /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
            /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            /* .reasoning_in_content = */ false,
            /* .thinking_forced_open = */ true,
            /* .parse_tool_calls = */ true,
        };
        const std::string variant("thinking_forced_open_tool_call_in_reasoning_no_closing_think_not_partial");
        const std::string in = "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        auto m = common_chat_parse(in, false, syntax);
        assert_equals(variant, std::string("REASONING"), m.content);
        assert_equals(variant, std::string(""), m.reasoning_content);
        assert_equals<std::size_t>(variant, 1, m.tool_calls.size());
        assert_equals(variant, std::string("get_time"), m.tool_calls[0].name);
        assert_equals(variant, std::string("{\"city\":\"Tokyo\"}"), m.tool_calls[0].arguments);
    }

    // variant: thinking forced open + tool call in reasoning content + no closing think + partial
    {
        common_chat_syntax syntax = {
            /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
            /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            /* .reasoning_in_content = */ false,
            /* .thinking_forced_open = */ true,
            /* .parse_tool_calls = */ true,
        };
        const std::string variant("thinking_forced_open_tool_call_in_reasoning_no_closing_think_partial");
        const std::string in = "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        auto m = common_chat_parse(in, /* is_partial= */ true, syntax);
        assert_equals(variant, std::string("REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"), m.reasoning_content);
        assert_equals(variant, std::string(""), m.content);
        assert_equals<std::size_t>(variant, 0, m.tool_calls.size());
    }

    // variant: thinking not forced open + reasoning + regular content + no tool calls
    {
        common_chat_syntax syntax = {
            /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
            /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            /* .reasoning_in_content = */ false,
            /* .thinking_forced_open = */ true,
            /* .parse_tool_calls = */ true,
        };
        const std::string variant("thinking_forced_open_reasoning_regular_content_no_tool_calls");
        const std::string in = "REASONING</think>CONTENT";
        auto m = common_chat_parse(in, false, syntax);
        assert_equals<std::size_t>(variant, 0, m.tool_calls.size());
        assert_equals(variant, std::string("CONTENT"), m.content);
        assert_equals(variant, std::string("REASONING"), m.reasoning_content);
    }
    // variant: thinking not forced open + missing reasoning + no tool calls
    {
        common_chat_syntax syntax = {
            /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
            /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            /* .reasoning_in_content = */ false,
            /* .thinking_forced_open = */ false,
            /* .parse_tool_calls = */ true,
        };
        const std::string variant("thinking_not_forced_open_missing_reasoning_no_tool_calls");
        const std::string in = "CONTENT";
        auto m = common_chat_parse(in, false, syntax);
        assert_equals<std::size_t>(variant, 0, m.tool_calls.size());
        assert_equals(variant, std::string("CONTENT"), m.content);
        assert_equals(variant, std::string(""), m.reasoning_content);
    }
}

static void test_with_args(const std::string & input, const std::string & expected, bool parse_as_partial = true, bool is_partial = true) {
  common_chat_msg_parser builder(input, parse_as_partial, {});
  auto js = builder.try_consume_json_with_dumped_args({{"args"}}, {});
  assert_equals(true, js.has_value());
  assert_equals(is_partial, js->is_partial);
  assert_equals(expected, js->value.dump());
}

static void test_json_with_dumped_args_no_args() {
  // Normal JSON, nothing to heal, nothing to dump
  test("{\"name\": \"python\"}", false, {}, {}, "{\"name\":\"python\"}");
  // Full json is args
  test("{\"name\": \"python\"}", false, {{}}, {}, "{\"name\":\"python\"}");

  // If the arguments are further down, don't heal partial content.
  for (const auto & src : barely_healable_jsons) {
    test(src, true, {{"arguments"}}, {}, "{}");
  }
  // But heal content that isn't partial.
  test("{\"name\": \"python\"", true, {{"arguments"}}, {}, "{\"name\":\"python\"}");
}

static void test_json_with_dumped_args() {

  // Partial content.
  test("{\"content\": \"t", true, {}, {{"content"}}, "{\"content\":\"t\"}");
  test("{\"content\": \"", true, {}, {{"content"}}, "{\"content\":\"\"}");
  test("{\"content\": ", true, {}, {{"content"}}, "{}");

  // If the entire JSON is the arguments, healing it them dumping it produces the same output as the input (just reformatted).
  test("{\"name\": \"python", true, {{}}, {}, "{\"name\":\"python");
  for (const auto & src : barely_healable_jsons) {
    test(src, true, {{}}, {}, src);
  }

  // Full JSON w/ args
  for (auto parse_as_partial : {true, false}) {
    test_with_args(
      R"({"name": "python", "args": {"arg1": 1}})",
      R"({"name":"python","args":"{\"arg1\":1}"})",
      parse_as_partial,
      /* is_partial= */ false
    );
  }

  // Partial JSON w/ partial args
  test_with_args(
    R"({"foo": "bar", "args": {")",
    R"({"foo":"bar","args":"{\""})"
  );
  // Partial args broken in object key
  test_with_args(
    R"({"foo": "bar", "args": {"ar)",
    R"({"foo":"bar","args":"{\"ar"})"
  );
  // Partial args broken after object key
  test_with_args(
    R"({"foo": "bar", "args": {"arg1")",
    R"({"foo":"bar","args":"{\"arg1\""})"
  );
  // Partial args broken before object value
  test_with_args(
    R"({"foo": "bar", "args": {"arg1":)",
    R"({"foo":"bar","args":"{\"arg1\":"})"
  );
  // Partial args broken before object value (space)
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": )",
    R"({"foo":"bar","args":"{\"arg1\":"})"
  );
  // Partial args broken in object value that may not be complete (int)
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": 1)",
    R"({"foo":"bar","args":"{\"arg1\":"})"
  );
  // Partial args broken in object value that is complete (int)
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": 1 )",
    R"({"foo":"bar","args":"{\"arg1\":1"})"
  );
  // Partial args broken in object value that is incomplete (string)
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": ")",
    R"({"foo":"bar","args":"{\"arg1\":\""})"
  );
  // Partial args broken in object value that is complete (string)
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "1")",
    R"({"foo":"bar","args":"{\"arg1\":\"1\""})"
  );
  // Partial args broken on array opening
  test_with_args(
    R"({"foo": "bar", "args": [)",
    R"({"foo":"bar","args":"["})"
  );
  // Partial args broken on array value that is incomplete (int)
  test_with_args(
    R"({"foo": "bar", "args": [1)",
    R"({"foo":"bar","args":"["})"
  );
  // Partial args broken on array value that is complete (int)
  test_with_args(
    R"({"foo": "bar", "args": [1 )",
    R"({"foo":"bar","args":"[1"})"
  );
  // Partial args broken on array value that is complete (string)
  test_with_args(
    R"({"foo": "bar", "args": ["1")",
    R"({"foo":"bar","args":"[\"1\""})"
  );
  // Partial args broken after array value
  test_with_args(
    R"({"foo": "bar", "args": [1,)",
    R"({"foo":"bar","args":"[1,"})"
  );
  // Partial args broken on nested array
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": [)",
    R"({"foo":"bar","args":"{\"arg1\":["})"
  );

  // Unicode tests
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\u)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\u"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\u0)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\u0"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\u00)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\u00"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\u000)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\u000"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\u0000)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\u0000"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud8)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud8"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud80)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud80"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud800)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud800"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud800\)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud800\\"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud800\u)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud800\\u"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud800\ud)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud800\\ud"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud800\udc)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud800\\udc"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud800\udc0)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud800\\udc0"})"
  );
  test_with_args(
    R"({"foo": "bar", "args": {"arg1": "\ud800\udc00)",
    R"({"foo":"bar","args":"{\"arg1\":\"\\ud800\\udc00"})"
  );
}

static void test_positions() {
  {
    common_chat_msg_parser builder("Hello, world!", /* is_partial= */ false, {});
    assert_equals<size_t>(0, builder.pos());
    assert_throws([&]() { builder.move_to(100); });
    assert_equals<size_t>(0, builder.pos());
    assert_throws([&]() { builder.move_back(1); });
    assert_equals<size_t>(0, builder.pos());

    builder.move_to(8);
    assert_equals<size_t>(8, builder.pos());
    builder.move_back(1);
    assert_equals<size_t>(7, builder.pos());
    assert_equals("world!", builder.consume_rest());

    builder.move_to(0);
    assert_equals<size_t>(0, builder.pos());

    assert_throws([&]() { builder.finish(); });
    assert_equals<size_t>(0, builder.pos());

    builder.move_to(builder.input().size());
    builder.finish();
  }
  {
    common_chat_msg_parser builder("Hello, world!", /* is_partial= */ true, {});

    builder.move_to(builder.input().size());
    assert_equals<size_t>(builder.input().size(), builder.pos());
    builder.finish();
  }
}

int main() {
    test_positions();
    test_json_with_dumped_args_no_args();
    test_json_with_dumped_args();
    test_reasoning();
    test_regex();
    test_deepseek_v3_1_tool_calls();
    std::cout << "All tests passed!\n";
    return 0;
}
