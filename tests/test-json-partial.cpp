#include "common.h"
#include "json-partial.h"
#include <exception>
#include <iostream>
#include <stdexcept>

template <class T> static void assert_equals(const T & expected, const T & actual) {
  if (expected != actual) {
      std::cerr << "Expected: " << expected << std::endl;
      std::cerr << "Actual: " << actual << std::endl;
      std::cerr << std::flush;
      throw std::runtime_error("Test failed");
  }
}

static void test_json_healing() {
  auto parse = [](const std::string & str) {
      std::cerr << "# Parsing: " << str << '\n';
      std::string::const_iterator it = str.begin();
      const auto end = str.end();
      common_json out;
      std::string healing_marker = "$llama.cpp.json$";
      if (common_json_parse(it, end, healing_marker, out)) {
          auto dump = out.json.dump();
          std::cerr << "Parsed: " << dump << '\n';
          std::cerr << "Magic: " << out.healing_marker.json_dump_marker << '\n';
          std::string result;
          if (!out.healing_marker.json_dump_marker.empty()) {
              auto i = dump.find(out.healing_marker.json_dump_marker);
              if (i == std::string::npos) {
                  throw std::runtime_error("Failed to find magic in dump " + dump + " (magic: " + out.healing_marker.json_dump_marker + ")");
              }
              result = dump.substr(0, i);
          } else {
            result = dump;
          }
          std::cerr << "Result: " << result << '\n';
          if (string_starts_with(str, result)) {
            std::cerr << "Failure!\n";
          }
        //   return dump;
      } else {
        throw std::runtime_error("Failed to parse: " + str);
      }

  };
  auto parse_all = [&](const std::string & str) {
      for (size_t i = 1; i < str.size(); i++) {
          parse(str.substr(0, i));
      }
  };
  parse_all("{\"a\": \"b\"}");
  parse_all("{\"hey\": 1, \"ho\\\"ha\": [1]}");

  parse_all("[{\"a\": \"b\"}]");

  auto test = [&](const std::vector<std::string> & inputs, const std::string & expected, const std::string & expected_marker) {
      for (const auto & input : inputs) {
        common_json out;
        assert_equals(true, common_json_parse(input, "$foo", out));
        assert_equals<std::string>(expected, out.json.dump());
        assert_equals<std::string>(expected_marker, out.healing_marker.json_dump_marker);
      }
  };
  // No healing needed:
  test(
    {
      R"([{"a":"b"}, "y"])",
    },
    R"([{"a":"b"},"y"])",
    ""
  );
  // Partial literals can't be healed:
  test(
    {
      R"([1)",
      R"([tru)",
      R"([n)",
      R"([nul)",
      R"([23.2)",
    },
    R"(["$foo"])",
    R"("$foo)"
  );
  test(
    {
      R"({"a": 1)",
      R"({"a": tru)",
      R"({"a": n)",
      R"({"a": nul)",
      R"({"a": 23.2)",
    },
    R"({"a":"$foo"})",
    R"("$foo)"
  );
  test(
    {
      R"({)",
    },
    R"({"$foo":1})",
    R"("$foo)"
  );
  test(
    {
      R"([)",
    },
    R"(["$foo"])",
    R"("$foo)"
  );
  // Healing right after a full literal
  test(
    {
      R"(1 )",
    },
    R"(1)",
    ""
  );
  test(
    {
      R"(true)",
      R"(true )",
    },
    R"(true)",
    ""
  );
  test(
    {
      R"(null)",
      R"(null )",
    },
    R"(null)",
    ""
  );
  test(
    {
      R"([1 )",
    },
    R"([1,"$foo"])",
    R"(,"$foo)"
  );
  test(
    {
      R"([{})",
      R"([{} )",
    },
    R"([{},"$foo"])",
    R"(,"$foo)"
  );
  test(
    {
      R"([true)",
    },
    // TODO: detect the true/false/null literal was complete
    R"(["$foo"])",
    R"("$foo)"
  );
  test(
    {
      R"([true )",
    },
    R"([true,"$foo"])",
    R"(,"$foo)"
  );
  test(
    {
      R"([true,)",
    },
    R"([true,"$foo"])",
    R"("$foo)"
  );
  // Test nesting
  test(
    {
      R"([{"a": [{"b": [{)",
    },
    R"([{"a":[{"b":[{"$foo":1}]}]}])",
    R"("$foo)"
  );
  test(
    {
      R"([{"a": [{"b": [)",
    },
    R"([{"a":[{"b":["$foo"]}]}])",
    R"("$foo)"
  );

  test(
    {
      R"([{"a": "b"})",
      R"([{"a": "b"} )",
    },
    R"([{"a":"b"},"$foo"])",
    R"(,"$foo)"
  );
  test(
    {
      R"([{"a": "b"},)",
      R"([{"a": "b"}, )",
    },
    R"([{"a":"b"},"$foo"])",
    R"("$foo)"
  );
  test(
    {
      R"({ "code)",
    },
    R"({"code$foo":1})",
    R"($foo)"
  );
  test(
    {
      R"({ "code\)",
    },
    R"({"code\\$foo":1})",
    R"(\$foo)"
  );
  test(
    {
      R"({ "code")",
    },
    R"({"code":"$foo"})",
    R"(:"$foo)"
  );
  test(
    {
      R"({ "key")",
    },
    R"({"key":"$foo"})",
    R"(:"$foo)"
  );
}

int main() {
    test_json_healing();
    std::cerr << "All tests passed.\n";
    return 0;
}
