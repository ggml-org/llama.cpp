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

  common_json out;
  assert_equals(true, common_json_parse("[{\"a\": \"b\"}", "$foo", out));
  assert_equals<std::string>("[{\"a\":\"b\"},\"$foo\"]", out.json.dump());

  assert_equals(true, common_json_parse("{ \"code", "$foo", out));
  assert_equals<std::string>("{\"code$foo\":1}", out.json.dump());
  assert_equals<std::string>("$foo", out.healing_marker.json_dump_marker);

  assert_equals(true, common_json_parse("{ \"code\"", "$foo", out));
  assert_equals<std::string>("{\"code\":\"$foo\"}", out.json.dump());
}

int main() {
    test_json_healing();
    return 0;
}
