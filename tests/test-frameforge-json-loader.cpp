#include "../tools/frameforge/frameforge-schema.h"
#include <cassert>
#include <iostream>
#include <string>

using namespace frameforge;

static void test_load_from_json() {
    std::cout << "Testing JSON loading..." << std::endl;
    
    std::string json_path = "../tools/frameforge/verb-definitions.json";
    bool loaded = load_verb_definitions(json_path);
    
    assert(loaded && "Failed to load verb definitions");
    assert(are_verb_definitions_loaded() && "Definitions not marked as loaded");
    
    std::cout << "  ✓ JSON loaded successfully" << std::endl;
}

static void test_verb_conversion_with_json() {
    std::cout << "Testing verb conversion with JSON..." << std::endl;
    
    // Test basic verb
    assert(string_to_verb("PAN") == Verb::PAN);
    assert(verb_to_string(Verb::PAN) == "PAN");
    
    // Test alias (PIN -> PAN)
    assert(string_to_verb("PIN") == Verb::PAN);
    
    // Test case insensitivity
    assert(string_to_verb("pan") == Verb::PAN);
    
    // Test DELETE alias (REMOVE -> DELETE)
    assert(string_to_verb("REMOVE") == Verb::DELETE);
    
    std::cout << "  ✓ Verb conversion with JSON tests passed" << std::endl;
}

static void test_action_group_with_json() {
    std::cout << "Testing action group mapping with JSON..." << std::endl;
    
    assert(get_action_group_for_verb(Verb::PAN) == ActionGroup::CAMERA_CONTROL);
    assert(get_action_group_for_verb(Verb::SET_POSE) == ActionGroup::ACTOR_POSE);
    assert(get_action_group_for_verb(Verb::ADD) == ActionGroup::OBJECT_MGMT);
    assert(get_action_group_for_verb(Verb::SHOT) == ActionGroup::SHOT_MGMT);
    
    std::cout << "  ✓ Action group with JSON tests passed" << std::endl;
}

static void test_required_parameters_with_json() {
    std::cout << "Testing required parameters with JSON..." << std::endl;
    
    auto pan_params = get_required_parameters(Verb::PAN);
    assert(pan_params.size() == 1);
    assert(pan_params[0] == "direction");
    
    auto dolly_params = get_required_parameters(Verb::DOLLY);
    assert(dolly_params.size() == 2);
    assert(dolly_params[0] == "direction");
    assert(dolly_params[1] == "speed");
    
    auto lean_params = get_required_parameters(Verb::LEAN);
    assert(lean_params.size() == 2);
    assert(lean_params[0] == "direction");
    assert(lean_params[1] == "degrees");
    
    auto add_params = get_required_parameters(Verb::ADD);
    assert(add_params.size() == 1);
    assert(add_params[0] == "target");
    
    std::cout << "  ✓ Required parameters with JSON tests passed" << std::endl;
}

static void test_fallback_without_json() {
    std::cout << "Testing fallback without JSON (hard-coded defaults)..." << std::endl;
    
    // Test that the system still works without loading JSON
    assert(string_to_verb("TILT") == Verb::TILT);
    assert(get_action_group_for_verb(Verb::TILT) == ActionGroup::CAMERA_CONTROL);
    
    auto tilt_params = get_required_parameters(Verb::TILT);
    assert(tilt_params.size() == 1);
    assert(tilt_params[0] == "direction");
    
    std::cout << "  ✓ Fallback tests passed" << std::endl;
}

int main() {
    std::cout << "Running FrameForge JSON Loader Tests..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // First test fallback without JSON
        test_fallback_without_json();
        
        // Then test with JSON loaded
        test_load_from_json();
        test_verb_conversion_with_json();
        test_action_group_with_json();
        test_required_parameters_with_json();
        
        std::cout << "========================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
