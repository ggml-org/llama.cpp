#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "../src/unicode.h"
int main() {
    std::string text = "This is a test prompt to trigger the tokenizer 🚀 with LOTS of numbers 12345.";
    // Inflate the string to 40+ KB to guarantee MSVC stack overflow
    for(int i=0; i<1000; i++) text += " The quick brown fox jumps over the lazy dog. ";
    
    std::string regex_str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    std::vector<std::string> regex_exprs = {regex_str};
    
    try {
        std::vector<std::string> words = unicode_regex_split(text, regex_exprs, false);
        std::cout << "Successfully split into " << words.size() << " words!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1; // Explicitly fail the CI runner
    }
    return 0;
}
