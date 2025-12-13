#include "common/chat-auto-parser.h"
#include "common/chat.h"
#include <iostream>
#include <fstream>
#include <sstream>

using json = nlohmann::ordered_json;

int main() {
    try {
        std::cout << "=== Automatic Parser Generation Test ===" << std::endl;
        
        // Test Qwen3-Coder template (XML-style)
        std::cout << "\n1. Testing Qwen3-Coder.jinja (XML-style)" << std::endl;
        std::ifstream qwen_template("models/templates/Qwen3-Coder.jinja");
        std::stringstream qwen_buffer;
        qwen_buffer << qwen_template.rdbuf();
        std::string qwen_template_str = qwen_buffer.str();
        
        minja::chat_template qwen_chat_template(qwen_template_str, "", "");
        TemplatePattern qwen_pattern = TemplateAnalyzer::analyze_template(qwen_chat_template);
        
        std::cout << "  Format detected: " << (qwen_pattern.format == TemplatePattern::XML_CONSTRUCTED ? "XML_CONSTRUCTED" : 
                                              qwen_pattern.format == TemplatePattern::JSON_NATIVE ? "JSON_NATIVE" : "UNKNOWN") << std::endl;
        std::cout << "  Special markers found:" << std::endl;
        for (const auto& [key, value] : qwen_pattern.special_markers) {
            if (!value.empty()) {
                std::cout << "    " << key << ": \"" << value << "\"" << std::endl;
            }
        }
        
        // Test ByteDance-Seed-OSS template (XML-style)
        std::cout << "\n2. Testing ByteDance-Seed-OSS.jinja (XML-style)" << std::endl;
        std::ifstream seed_template("models/templates/ByteDance-Seed-OSS.jinja");
        std::stringstream seed_buffer;
        seed_buffer << seed_template.rdbuf();
        std::string seed_template_str = seed_buffer.str();
        
        minja::chat_template seed_chat_template(seed_template_str, "", "");
        TemplatePattern seed_pattern = TemplateAnalyzer::analyze_template(seed_chat_template);
        
        std::cout << "  Format detected: " << (seed_pattern.format == TemplatePattern::XML_CONSTRUCTED ? "XML_CONSTRUCTED" : 
                                              seed_pattern.format == TemplatePattern::JSON_NATIVE ? "JSON_NATIVE" : "UNKNOWN") << std::endl;
        std::cout << "  Special markers found:" << std::endl;
        for (const auto& [key, value] : seed_pattern.special_markers) {
            if (!value.empty()) {
                std::cout << "    " << key << ": \"" << value << "\"" << std::endl;
            }
        }
        
        // Test NVIDIA-Nemotron-Nano-v2 template (JSON-style)
        std::cout << "\n3. Testing NVIDIA-Nemotron-Nano-v2.jinja (JSON-style)" << std::endl;
        std::ifstream nemotron_template("models/templates/NVIDIA-Nemotron-Nano-v2.jinja");
        std::stringstream nemotron_buffer;
        nemotron_buffer << nemotron_template.rdbuf();
        std::string nemotron_template_str = nemotron_buffer.str();
        
        minja::chat_template nemotron_chat_template(nemotron_template_str, "", "");
        TemplatePattern nemotron_pattern = TemplateAnalyzer::analyze_template(nemotron_chat_template);
        
        std::cout << "  Format detected: " << (nemotron_pattern.format == TemplatePattern::XML_CONSTRUCTED ? "XML_CONSTRUCTED" : 
                                              nemotron_pattern.format == TemplatePattern::JSON_NATIVE ? "JSON_NATIVE" : "UNKNOWN") << std::endl;
        std::cout << "  Special markers found:" << std::endl;
        for (const auto& [key, value] : nemotron_pattern.special_markers) {
            if (!value.empty()) {
                std::cout << "    " << key << ": \"" << value << "\"" << std::endl;
            }
        }
        
        std::cout << "\n=== Test completed successfully! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}