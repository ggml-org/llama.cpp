#include "chat-auto-parser.h"
#include "chat.h"
#include "common.h"
#include <minja/chat-template.hpp>
#include <iostream>
#include <fstream>
#include <string>

using json = nlohmann::ordered_json;

bool test_template_pattern_detection(const std::string& template_path, const std::string& template_name) {
    std::cout << "Testing pattern detection for: " << template_name << std::endl;
    
    // Read the template file
    std::ifstream file(template_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open template file: " << template_path << std::endl;
        return false;
    }
    
    std::string template_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    try {
        // Create a chat template instance
        minja::chat_template tmpl(template_content, "", "");
        
        // Analyze the template using our automatic parser
        TemplatePattern pattern = TemplateAnalyzer::analyze_template(tmpl);
        
        std::cout << "  Detected format: ";
        switch (pattern.format) {
            case TemplatePattern::JSON_NATIVE:
                std::cout << "JSON_NATIVE";
                break;
            case TemplatePattern::XML_CONSTRUCTED:
                std::cout << "XML_CONSTRUCTED";
                break;
            case TemplatePattern::UNKNOWN:
                std::cout << "UNKNOWN";
                break;
        }
        std::cout << std::endl;
        
        std::cout << "  Special markers detected:" << std::endl;
        for (const auto& [key, value] : pattern.special_markers) {
            if (!value.empty()) {
                std::cout << "    " << key << ": \"" << value << "\"" << std::endl;
            }
        }
        
        std::cout << "  Preserved tokens: ";
        for (const auto& token : pattern.preserved_tokens) {
            std::cout << "\"" << token << "\" ";
        }
        std::cout << std::endl;
        
        // Verify that we found some meaningful patterns
        bool has_meaningful_patterns = false;
        for (const auto& [key, value] : pattern.special_markers) {
            if (!value.empty() && value.length() > 1) {  // Filter out single chars like "{" or "["
                has_meaningful_patterns = true;
                break;
            }
        }
        
        if (pattern.format != TemplatePattern::UNKNOWN || has_meaningful_patterns) {
            std::cout << "  ✓ Pattern detection successful for " << template_name << std::endl;
            return true;
        } else {
            std::cout << "  ✗ Pattern detection failed for " << template_name << " - no meaningful patterns found" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  ✗ Error analyzing template " << template_name << ": " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "Testing automatic parser pattern detection..." << std::endl;
    
    int passed = 0;
    int total = 0;
    
    // Test Qwen3-Coder template (XML-style)
    if (test_template_pattern_detection("models/templates/Qwen3-Coder.jinja", "Qwen3-Coder")) {
        passed++;
    }
    total++;
    
    // Test ByteDance-Seed-OSS template (XML-style)
    if (test_template_pattern_detection("models/templates/ByteDance-Seed-OSS.jinja", "ByteDance-Seed-OSS")) {
        passed++;
    }
    total++;
    
    // Test NVIDIA-Nemotron-Nano-v2 template (JSON-style)
    if (test_template_pattern_detection("models/templates/NVIDIA-Nemotron-Nano-v2.jinja", "NVIDIA-Nemotron-Nano-v2")) {
        passed++;
    }
    total++;
    
    std::cout << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " templates passed pattern detection" << std::endl;
    
    if (passed == total) {
        std::cout << "✓ All pattern detection tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some pattern detection tests failed!" << std::endl;
        return 1;
    }
}