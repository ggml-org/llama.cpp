#pragma once

#include "chat.h"
#include "chat-peg-parser.h"
#include "common.h"

#include <string>
#include <vector>
#include <map>
#include <chrono>

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

using json = nlohmann::ordered_json;

// Structure to hold detected template patterns
struct TemplatePattern {
    enum ToolCallFormat {
        JSON_NATIVE,      // JSON-style like Nemotron
        XML_CONSTRUCTED,  // XML-style like Qwen3/Seed-OSS
        CONTENT_ONLY,     // No tool call support detected
        UNKNOWN
    };
    
    ToolCallFormat format;
    std::vector<std::string> preserved_tokens;
    std::map<std::string, std::string> special_markers;
    bool has_reasoning_support = false;
    
    // For XML-style templates
    std::string function_open_marker;
    std::string function_close_marker;
    std::string function_name_suffix; // New field
    std::string parameter_open_marker;
    std::string parameter_close_marker;
    
    // Improved parameter parsing
    std::string parameter_key_prefix;
    std::string parameter_key_suffix;
    
    // For JSON-style templates  
    std::string tool_call_start_marker;
    std::string tool_call_end_marker;
    
    // Reasoning
    std::string reasoning_start_marker;
    std::string reasoning_end_marker;
};

// Structure to hold discovered patterns through differential analysis
struct DiscoveredPattern {
    std::string tool_call_opener;
    std::string tool_call_closer;
    std::string function_opener;
    std::string function_closer;
    std::string function_name_suffix; // New field
    std::string parameter_opener;
    std::string parameter_closer;
    std::string argument_separator;
    std::string tool_call_start_marker;
    std::string tool_call_end_marker;
    
    // Improved parameter parsing
    std::string parameter_key_prefix;
    std::string parameter_key_suffix;
    
    // Reasoning
    std::string reasoning_start_marker;
    std::string reasoning_end_marker;
};

// Template analyzer that uses differential analysis of OpenAI-compatible messages
class TemplateAnalyzer {
public:
    static TemplatePattern analyze_template(const minja::chat_template& tmpl);
    
private:
    static TemplatePattern::ToolCallFormat detect_format_by_differential(const minja::chat_template& tmpl);
    static std::vector<std::string> extract_preserved_tokens(const minja::chat_template& tmpl);
    static bool has_reasoning_support(const minja::chat_template& tmpl);
    static std::map<std::string, std::string> extract_special_markers(const minja::chat_template& tmpl);
    
    // Helper methods for differential analysis
    static std::string analyze_tool_call_differences(const minja::chat_template& tmpl);
    static std::string analyze_reasoning_differences(const minja::chat_template& tmpl);
    static std::string analyze_content_differences(const minja::chat_template& tmpl);
    
    // New pure differential analysis methods
    static DiscoveredPattern analyze_by_differential(const minja::chat_template& tmpl);
    static void analyze_reasoning(const minja::chat_template& tmpl, DiscoveredPattern& patterns); // New method
    static DiscoveredPattern extract_patterns_from_differences(
        const std::string& tool1_diff,
        const std::string& tool2_diff,
        const std::string& tool3_diff);
    static std::string find_closing_pattern(const std::string& diff, size_t func_pos);
    static std::string find_tool_call_start(const std::string& diff);
    static std::string find_tool_call_end(const std::string& diff, size_t func_pos);
    static std::string infer_tool_call_opener(const std::string& diff1, const std::string& diff2, const std::string& diff3);
    static std::string infer_tool_call_closer(const std::string& diff1, const std::string& diff2, const std::string& diff3);
    static std::string find_common_substring(const std::vector<std::string>& strings);
    static std::string find_common_suffix(const std::vector<std::string>& strings);
    static std::string find_common_start_pattern(const std::string& diff1, const std::string& diff2, const std::string& diff3);
    static std::string find_common_end_pattern(const std::string& diff1, const std::string& diff2, const std::string& diff3);
    static TemplatePattern::ToolCallFormat determine_format_from_patterns(const DiscoveredPattern& patterns);
};

// Forward declaration of templates_params to match the one in chat.cpp
struct templates_params;

// Universal PEG parser generator
class UniversalPEGGenerator {
public:
    static common_chat_params generate_parser(
        const TemplatePattern& pattern,
        const minja::chat_template& tmpl,
        const struct templates_params& inputs
    );
    
private:
    static common_peg_arena build_native_parser(
        const TemplatePattern& pattern,
        const minja::chat_template& tmpl,
        const struct templates_params& inputs
    );
    
    static common_peg_arena build_constructed_parser(
        const TemplatePattern& pattern,
        const minja::chat_template& tmpl, 
        const struct templates_params& inputs
    );
};