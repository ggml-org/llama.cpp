#define LLAMA_API_INTERNAL

#include "grammar-parser.h"
#include "ggml.h"
#include "llama.h"
#include "unicode.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

static bool llama_sample_grammar_string(struct llama_grammar * grammar, const std::string & input_str, size_t & error_pos, std::string & error_msg) {
    auto decoded = decode_utf8(input_str, {});
    const auto & code_points = decoded.first;

    const llama_grammar_rules  & rules      = llama_grammar_get_rules (grammar);
          llama_grammar_stacks & cur_stacks = llama_grammar_get_stacks(grammar);

    size_t pos = 0;
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        const llama_grammar_stacks prev_stacks = llama_grammar_get_stacks(grammar); // copy

        llama_grammar_accept(rules, prev_stacks, *it, cur_stacks);

        if (cur_stacks.empty()) {
            error_pos = pos;
            error_msg = "Unexpected character '" + unicode_cpt_to_utf8(*it) + "'";
            cur_stacks = prev_stacks;
            return false;
        }
        ++pos;
    }

    for (const auto & stack : cur_stacks) {
        if (stack.empty()) {
            return true;
        }
    }

    error_pos = pos;
    error_msg = "Unexpected end of input";
    return false;
}

static void print_error_message(const std::string & input_str, size_t error_pos, const std::string & error_msg) {
    fprintf(stdout, "Input string is invalid according to the grammar.\n");
    fprintf(stdout, "Error: %s at position %zu\n", error_msg.c_str(), error_pos);
    fprintf(stdout, "\n");
    fprintf(stdout, "Input string:\n");
    fprintf(stdout, "%s", input_str.substr(0, error_pos).c_str());
    if (error_pos < input_str.size()) {
        fprintf(stdout, "\033[1;31m%c", input_str[error_pos]);
        if (error_pos+1 < input_str.size()) {
            fprintf(stdout, "\033[0;31m%s", input_str.substr(error_pos+1).c_str());
        }
        fprintf(stdout, "\033[0m\n");
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stdout, "Usage: %s <grammar_filename> <input_filename>\n", argv[0]);
        return 1;
    }

    const std::string grammar_filename = argv[1];
    const std::string input_filename = argv[2];

    // Read the GBNF grammar file
    FILE* grammar_file = fopen(grammar_filename.c_str(), "r");
    if (!grammar_file) {
        fprintf(stdout, "Failed to open grammar file: %s\n", grammar_filename.c_str());
        return 1;
    }

    std::string grammar_str;
    {
        std::ifstream grammar_file(grammar_filename);
        GGML_ASSERT(grammar_file.is_open() && "Failed to open grammar file");
        std::stringstream buffer;
        buffer << grammar_file.rdbuf();
        grammar_str = buffer.str();
    }

    // Parse the GBNF grammar
    auto parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // will be empty (default) if there are parse errors
    if (parsed_grammar.rules.empty()) {
        fprintf(stdout, "%s: failed to parse grammar\n", __func__);
        return 1;
    }

    // Ensure that there is a "root" node.
    if (parsed_grammar.symbol_ids.find("root") == parsed_grammar.symbol_ids.end()) {
        fprintf(stdout, "%s: grammar does not contain a 'root' symbol\n", __func__);
        return 1;
    }

    std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());

    // Create the LLAMA grammar
    auto grammar = llama_grammar_init(
            grammar_rules.data(),
            grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    if (grammar == nullptr) {
        throw std::runtime_error("Failed to initialize llama_grammar");
    }
    // Read the input file
    std::string input_str;
    {
        std::ifstream input_file(input_filename);
        GGML_ASSERT(input_file.is_open() && "Failed to open input file");
        std::stringstream buffer;
        buffer << input_file.rdbuf();
        input_str = buffer.str();
    }

    // Validate the input string against the grammar
    size_t error_pos;
    std::string error_msg;
    bool is_valid = llama_sample_grammar_string(grammar, input_str, error_pos, error_msg);

    if (is_valid) {
        fprintf(stdout, "Input string is valid according to the grammar.\n");
    } else {
        print_error_message(input_str, error_pos, error_msg);
    }

    // Clean up
    llama_grammar_free(grammar);

    return 0;
}
