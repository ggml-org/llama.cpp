// Unit test for command validator
// Tests validation logic without requiring a model

#include "command-schema.h"
#include "command-validator.h"

#include <iostream>
#include <cassert>

using namespace semantic_server;

void test_schema_registry() {
    std::cout << "Testing SchemaRegistry..." << std::endl;
    
    auto & registry = SchemaRegistry::instance();
    
    // Test PAN schema
    const CommandSchema * pan_schema = registry.get_schema(Verb::PAN);
    assert(pan_schema != nullptr);
    assert(pan_schema->verb == Verb::PAN);
    assert(pan_schema->action_group == ActionGroup::CAMERA_CONTROL);
    assert(pan_schema->parameters.size() == 3); // direction, degrees, speed
    
    std::cout << "  ✓ PAN schema loaded correctly" << std::endl;
    
    // Test ADD schema
    const CommandSchema * add_schema = registry.get_schema(Verb::ADD);
    assert(add_schema != nullptr);
    assert(add_schema->verb == Verb::ADD);
    assert(add_schema->action_group == ActionGroup::OBJECT_MGMT);
    
    std::cout << "  ✓ ADD schema loaded correctly" << std::endl;
    
    std::cout << "✓ SchemaRegistry tests passed\n" << std::endl;
}

void test_command_validator() {
    std::cout << "Testing CommandValidator..." << std::endl;
    
    CommandValidator validator;
    
    // Test valid PAN command
    json pan_cmd = {
        {"verb", "PAN"},
        {"parameters", {
            {"direction", "LEFT"},
            {"degrees", 30.0}
        }}
    };
    
    auto result = validator.validate(pan_cmd);
    assert(result.valid);
    assert(result.validated_command["action_group"] == "CAMERA_CONTROL");
    std::cout << "  ✓ Valid PAN command validated" << std::endl;
    
    // Test invalid command (missing required parameter)
    json invalid_cmd = {
        {"verb", "PAN"},
        {"parameters", {
            {"degrees", 30.0}  // Missing required "direction"
        }}
    };
    
    result = validator.validate(invalid_cmd);
    assert(!result.valid);
    assert(!result.missing_parameters.empty());
    std::cout << "  ✓ Invalid command detected (missing parameter)" << std::endl;
    
    // Test unknown verb
    json unknown_verb = {
        {"verb", "INVALID_VERB"},
        {"parameters", {}}
    };
    
    result = validator.validate(unknown_verb);
    assert(!result.valid);
    std::cout << "  ✓ Unknown verb detected" << std::endl;
    
    std::cout << "✓ CommandValidator tests passed\n" << std::endl;
}

void test_fuzzy_matching() {
    std::cout << "Testing fuzzy matching..." << std::endl;
    
    CommandValidator validator;
    
    // Test typo correction
    std::string suggested = validator.suggest_verb("PIN");
    assert(suggested == "PAN");
    std::cout << "  ✓ 'PIN' matched to 'PAN'" << std::endl;
    
    suggested = validator.suggest_verb("TILE");
    assert(suggested == "TILT");
    std::cout << "  ✓ 'TILE' matched to 'TILT'" << std::endl;
    
    // Test already correct verb
    suggested = validator.suggest_verb("PAN");
    assert(suggested == "PAN");
    std::cout << "  ✓ 'PAN' matched to 'PAN'" << std::endl;
    
    std::cout << "✓ Fuzzy matching tests passed\n" << std::endl;
}

void test_enum_conversions() {
    std::cout << "Testing enum conversions..." << std::endl;
    
    // Test verb conversions
    assert(string_to_verb("PAN") == Verb::PAN);
    assert(verb_to_string(Verb::PAN) == "PAN");
    std::cout << "  ✓ Verb conversions work" << std::endl;
    
    // Test action group conversions
    assert(string_to_action_group("CAMERA_CONTROL") == ActionGroup::CAMERA_CONTROL);
    assert(action_group_to_string(ActionGroup::CAMERA_CONTROL) == "CAMERA_CONTROL");
    std::cout << "  ✓ ActionGroup conversions work" << std::endl;
    
    // Test unknown values
    assert(string_to_verb("UNKNOWN_VERB") == Verb::UNKNOWN);
    assert(string_to_action_group("UNKNOWN_GROUP") == ActionGroup::UNKNOWN);
    std::cout << "  ✓ Unknown value handling works" << std::endl;
    
    std::cout << "✓ Enum conversion tests passed\n" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "Semantic AI Server - C++ Unit Tests" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    
    try {
        test_schema_registry();
        test_command_validator();
        test_fuzzy_matching();
        test_enum_conversions();
        
        std::cout << "============================================================" << std::endl;
        std::cout << "✓ All C++ tests passed!" << std::endl;
        std::cout << "============================================================" << std::endl;
        
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
