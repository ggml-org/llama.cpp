#include "../src/llama-grammar.h"
#include "../src/llama-vocab.h"
#include <iostream>
#include <string>
#include <vector>

static void test_grammar_recursion_dos() {
    // Create a deeply nested grammar: root ::= (((...)))
    // Each pair of parens adds a stack frame in parse_sequence -> parse_alternates
    int depth = 100000;
    std::string grammar = "root ::= ";
    for (int i = 0; i < depth; ++i) {
        grammar += "(";
    }
    grammar += " \"a\" ";
    for (int i = 0; i < depth; ++i) {
        grammar += ")";
    }

    try {
        std::cout << "Attempting to parse deeply nested grammar (depth=" << depth << ")..." << std::endl;
        
        // We don't need a real vocab for this test if we don't use token references
        // But the parser might need one if we used <token> syntax. We are using string literals.
        
        llama_grammar_parser parser;
        if (!parser.parse(grammar.c_str())) {
            std::cout << "Parser returned false (failed gracefully)" << std::endl;
        } else {
            std::cout << "Parser succeeded (unexpected)" << std::endl;
        }
        
    } catch (const std::exception & e) {
        std::string what = e.what();
        if (what.find("grammar recursion depth exceeded") != std::string::npos) {
            std::cout << "Caught expected exception: " << what << std::endl;
        } else {
            std::cout << "Caught unexpected exception: " << what << std::endl;
            throw;
        }
    }
}

int main() {
    test_grammar_recursion_dos();
    return 0;
}
