#include "chat-peg-parser/tests.h"
#include "log.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

// Struct to hold test information
struct TestEntry {
    std::string codename;
    std::string function_name;
    void (*test_func)(testing&);
};

// Dynamic list of all available tests
static const std::vector<TestEntry> all_tests = {
    {"partial", "test_partial_parsing", test_partial_parsing},
    {"one", "test_one", test_one},
    {"optional", "test_optional", test_optional},
    {"unicode", "test_unicode", test_unicode},
    {"recursive", "test_recursive_references", test_recursive_references},
    {"json", "test_json_parser", test_json_parser},
    {"gbnf", "test_gbnf_generation", test_gbnf_generation},
    {"qwen3_coder", "test_example_qwen3_coder", test_example_qwen3_coder},
    {"seed_oss", "test_example_seed_oss", test_example_seed_oss},
    {"minimax_m2", "test_example_minimax_m2", test_example_minimax_m2},
    {"command7_parser_compare", "test_command7_parser_compare", test_command7_parser_compare},
    {"serialization", "test_json_serialization", test_json_serialization}
};

// Function to list all available tests
static void list_available_tests() {
    LOG_ERR("Available tests:\n");
    for (const auto& test : all_tests) {
        LOG_ERR("  %s", test.codename.c_str());
        // Format spacing for alignment
        for (size_t i = test.codename.length(); i < 25; ++i) {
            LOG_ERR(" ");
        }
        LOG_ERR("- %s\n", test.function_name.c_str());
    }
    LOG_ERR("\n");
    LOG_ERR("Usage:\n");
    LOG_ERR("  test-chat-peg-parser                 # Run all tests\n");
    LOG_ERR("  test-chat-peg-parser test1 test2     # Run specific tests\n");
    LOG_ERR("  test-chat-peg-parser --tests         # List available tests\n");
}

// Function to check if a codename matches the provided arguments
static bool should_run_test(const std::vector<std::string>& args, const std::string& codename) {
    // If no arguments provided, run all tests
    if (args.size() <= 1) {
        return true;
    }

    // Check if codename matches any of the provided arguments
    return std::find(args.begin() + 1, args.end(), codename) != args.end();
}

// Helper to run a test conditionally
static void run_test_conditionally(testing& t, const std::vector<std::string>& args,
                           const std::string& codename, void (*test_func)(testing&)) {
    if (should_run_test(args, codename)) {
        test_func(t);
    }
}

int main(int argc, char *argv[]) {
    // Convert argv to vector of strings for easier handling
    std::vector<std::string> args;
    args.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }

    // Special case: list available tests and exit
    if (argc == 2 && args[1] == "--tests") {
        list_available_tests();
        return 0;
    }

    testing t(std::cout);

    // Dynamically process all tests from the data structure
    for (const auto& test : all_tests) {
        run_test_conditionally(t, args, test.codename, test.test_func);
    }

    return t.summary();
}
