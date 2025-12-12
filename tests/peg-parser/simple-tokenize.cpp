#include "simple-tokenize.h"

std::vector<std::string> simple_tokenize(const std::string & input) {
    std::vector<std::string> result;
    std::string              current;

    for (size_t i = 0; i < input.size(); i++) {
        switch (input[i]) {
            case ' ':
            case '\n':
            case '\t':
            case '{':
            case '}':
            case ',':
            case '[':
            case '"':
            case ']':
            case '.':
            case '<':
            case '>':
            case '=':
            case '/':
                if (!current.empty()) {
                    result.push_back(current);
                    current.clear();
                }
            default:;
        }
        current += input[i];
    }

    if (!current.empty()) {
        result.push_back(current);
    }

    return result;
}
