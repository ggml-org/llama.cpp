#pragma once

// Common includes for all test files
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "testing.h"
#include "peg-parser.h"
#include "chat-peg-parser.h"
#include "simple-tokenize.h"

struct bench_tool_call {
    std::string            id;
    std::string            name;
    nlohmann::ordered_json args;
};

// Test function declarations
void test_basic(testing &t);
void test_json_parser(testing &t);
void test_gbnf_generation(testing &t);
void test_unicode(testing &t);
void test_json_serialization(testing &t);
