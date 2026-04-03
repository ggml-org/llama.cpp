#include "unicode.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Let's copy unicode_regex_split_custom_tekken and test it
// Wait, we can just call it from unicode.h directly since it's already compiled!
int main() {
    std::string text = "This is a test prompt to trigger the tokenizer 🚀 with LOTS of numbers 12345.";
    for (int i = 0; i < 100; i++) {
        text += " The quick brown fox jumps over the lazy dog. ";
    }
    std::string regex_str =
        "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-"
        "z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    std::vector<std::string> regex_exprs = { regex_str };
    try {
        std::vector<std::string> words = unicode_regex_split(text, regex_exprs, false);
        std::cout << "Successfully split into " << words.size() << " words!" << std::endl;
        if (words.size() > 0) {
            std::cout << "First word: " << words[0] << std::endl;
        }
    } catch (const std::exception & e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    return 0;
}
