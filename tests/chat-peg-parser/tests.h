#pragma once

// Common includes for all test files
#include "test_harness.h"
#include <nlohmann/json.hpp>
#include "chat-peg-parser.h"
#include "chat-peg-parser-helper.h"
#include <memory>
#include <vector>
#include <string>

std::vector<std::string> simple_tokenize(const std::string &);

struct bench_tool_call {
    std::string            id;
    std::string            name;
    nlohmann::ordered_json args;
};

// Test function declarations
void test_partial_parsing(testing &t);
void test_one(testing &t);
void test_optional(testing &t);
void test_recursive_references(testing &t);
void test_json_parser(testing &t);
void test_gbnf_generation(testing &t);
void test_example_qwen3_coder(testing &t);
void test_example_seed_oss(testing &t);
void test_example_minimax_m2(testing &t);
void test_command7_parser_compare(testing &t);
void test_unicode(testing &t);
void test_json_serialization(testing &t);
