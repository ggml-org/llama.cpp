#include "chat-auto-parser.h"
#include "chat.h"
#include "common.h"
#include "log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using json = nlohmann::ordered_json;

// Helper to read file
static std::string read_file(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::ostringstream buf;
    buf << fin.rdbuf();
    return buf.str();
}

struct templates_params {
    json messages;
    json tools;
    common_chat_tool_choice tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    common_reasoning_format reasoning_format;
    bool stream;
    std::string grammar;
    bool add_generation_prompt;
    bool enable_thinking;
    std::chrono::system_clock::time_point now;
    json extra_context;
    bool add_bos;
    bool add_eos;
    bool is_inference;
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <template_path>" << std::endl;
        return 1;
    }

    std::string template_path = argv[1];
    std::string template_source;
    try {
        template_source = read_file(template_path);
    } catch (const std::exception & e) {
        std::cerr << "Error reading template: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Analyzing template: " << template_path << std::endl;

    try {
        minja::chat_template chat_template(template_source, "", "");
        TemplatePattern pattern = TemplateAnalyzer::analyze_template(chat_template);

        std::cout << "\n=== Analysis Results ===" << std::endl;
        std::cout << "Format: " << (int)pattern.format << std::endl;
        
        std::cout << "\n--- Special Markers ---" << std::endl;
        for (const auto & [key, value] : pattern.special_markers) {
            std::cout << key << ": '" << value << "'" << std::endl;
        }

        // Generate Parser
        templates_params params;
        params.messages = json::array();
        params.tools = {
            {
                {"type", "function"},
                {"function", {
                    {"name", "test_tool"},
                    {"description", "A test tool"},
                    {"parameters", {
                        {"type", "object"},
                        {"properties", {
                            {"arg1", {"type", "string"}},
                            {"arg2", {"type", "string"}}
                        }},
                        {"required", json::array({"arg1", "arg2"})}
                    }}
                }}
            }
        };
        params.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        params.parallel_tool_calls = false;
        
        auto parser_data = UniversalPEGGenerator::generate_parser(pattern, chat_template, params);
        
        std::cout << "\n=== Generated Parser ===" << std::endl;
        std::cout << "Parser size: " << parser_data.parser.size() << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Analysis failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
