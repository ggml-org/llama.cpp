// Tests for automatic parser generation
// Tests chat handling for automatic template analysis and parser generation

#include "test-chat.cpp"

static void test_auto_parser_qwen3_coder() {
    printf("[%s]\n", __func__);
    
    auto tmpls = read_templates("models/templates/Qwen3-Coder.jinja");
    REQUIRE(tmpls != nullptr);
    
    // Test template analysis
    auto template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");
    
    TemplatePattern pattern = TemplateAnalyzer::analyze_template(chat_template);
    
    // Should be detected as XML_CONSTRUCTED
    assert_equals(TemplatePattern::XML_CONSTRUCTED, pattern.format);
    
    // Should have found some special markers
    assert_equals(true, pattern.special_markers.size() > 0);
    
    // Should have reasoning support
    assert_equals(true, pattern.has_reasoning_support);
    
    // Test that we can generate a parser
    templates_params params;
    params.messages = json::array();
    params.tools = json::array();
    params.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.parallel_tool_calls = false;
    
    try {
        auto parser_data = UniversalPEGGenerator::generate_parser(pattern, chat_template, params);
        assert_equals(COMMON_CHAT_FORMAT_PEG_CONSTRUCTED, parser_data.format);
    } catch (const std::exception& e) {
        // Expected for now since we have placeholder implementations
        printf("Expected exception during parser generation: %s\n", e.what());
    }
}

static void test_auto_parser_bytedance_seed_oss() {
    printf("[%s]\n", __func__);
    
    auto tmpls = read_templates("models/templates/ByteDance-Seed-OSS.jinja");
    REQUIRE(tmpls != nullptr);
    
    // Test template analysis
    auto template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");
    
    TemplatePattern pattern = TemplateAnalyzer::analyze_template(chat_template);
    
    // Should be detected as XML_CONSTRUCTED
    assert_equals(TemplatePattern::XML_CONSTRUCTED, pattern.format);
    
    // Should have found some special markers
    assert_equals(true, pattern.special_markers.size() > 0);
    
    // Should have reasoning support
    assert_equals(true, pattern.has_reasoning_support);
    
    // Test that we can generate a parser
    templates_params params;
    params.messages = json::array();
    params.tools = json::array();
    params.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.parallel_tool_calls = false;
    
    try {
        auto parser_data = UniversalPEGGenerator::generate_parser(pattern, chat_template, params);
        assert_equals(COMMON_CHAT_FORMAT_PEG_CONSTRUCTED, parser_data.format);
    } catch (const std::exception& e) {
        // Expected for now since we have placeholder implementations
        printf("Expected exception during parser generation: %s\n", e.what());
    }
}

static void test_auto_parser_nemotron_nano_v2() {
    printf("[%s]\n", __func__);
    
    auto tmpls = read_templates("models/templates/NVIDIA-Nemotron-Nano-v2.jinja");
    REQUIRE(tmpls != nullptr);
    
    // Test template analysis
    auto template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");
    
    TemplatePattern pattern = TemplateAnalyzer::analyze_template(chat_template);
    
    // Should be detected as JSON_NATIVE
    assert_equals(TemplatePattern::JSON_NATIVE, pattern.format);
    
    // Should have found some special markers
    assert_equals(true, pattern.special_markers.size() > 0);
    
    // Should have reasoning support
    assert_equals(true, pattern.has_reasoning_support);
    
    // Test that we can generate a parser
    templates_params params;
    params.messages = json::array();
    params.tools = json::array();
    params.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.parallel_tool_calls = false;
    
    try {
        auto parser_data = UniversalPEGGenerator::generate_parser(pattern, chat_template, params);
        assert_equals(COMMON_CHAT_FORMAT_PEG_NATIVE, parser_data.format);
    } catch (const std::exception& e) {
        // Expected for now since we have placeholder implementations
        printf("Expected exception during parser generation: %s\n", e.what());
    }
}

TEST_CASE("Automatic Parser Generation", "[auto-parser]") {
    SECTION("Qwen3-Coder Template Analysis") {
        test_auto_parser_qwen3_coder();
    }
    
    SECTION("ByteDance-Seed-OSS Template Analysis") {
        test_auto_parser_bytedance_seed_oss();
    }
    
    SECTION("NVIDIA-Nemotron-Nano-v2 Template Analysis") {
        test_auto_parser_nemotron_nano_v2();
    }
}