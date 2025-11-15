#pragma once

// Common includes for all test files
#include "../testcase.hpp"
#include <nlohmann/json.hpp>
#include "chat-peg-parser.h"
#include <memory>

// Test class declarations
class test_partial_parsing : public compound_test {
public:
    test_partial_parsing();
};

class test_one : public compound_test {
public:
    test_one();
};

class test_optional : public compound_test {
public:
    test_optional();
};

class test_recursive_references : public compound_test {
public:
    test_recursive_references();
};

class test_json_parser : public compound_test {
public:
    test_json_parser();
};

class test_actions : public compound_test {
public:
    test_actions();
};

class test_gbnf_generation : public compound_test {
public:
    test_gbnf_generation();
};

class uses_simple_tokenizer {
protected:
    static std::vector<std::string> simple_tokenize(const std::string &);
};

struct bench_tool_call {
    std::string            id;
    std::string            name;
    nlohmann::ordered_json args;
};

class benchmark_test {
protected:
    std::vector<std::unique_ptr<test_case>> cases;
    long long run_benchmark(size_t which, int iterations);
public:
    benchmark_test(std::vector<std::unique_ptr<test_case>>);
};

class test_command7_parser_compare : public uses_simple_tokenizer, public benchmark_test {
private:
    class common_chat_peg_parser parser;
    common_chat_parse_event_handler handler;

    std::string reasoning;
    std::string content;
    std::vector<bench_tool_call> tool_calls;
    std::vector<std::string> tokens;
    // Helper methods
    static class common_chat_peg_parser create_command_r7b_parser();
    static common_chat_parse_event_handler create_command_r7b_event_handler();
    static void test_command_r7b_parser(const class common_chat_peg_parser & p, const std::string & input, bool need_more_input, bool print_results = false);
    static void test_command_r7b_legacy_parser(const std::string & input, bool need_more_input, bool print_results = false);
public:
    test_command7_parser_compare();
    void run_comparison(int iterations);
};

class test_example_qwen3_coder : public uses_simple_tokenizer, public compound_test {
private:
    class common_chat_peg_parser parser;
public:
    test_example_qwen3_coder();
};
