#include "../../tools/frameforge/frameforge-schema.h"
#include "../../tools/frameforge/frameforge-validator.h"
#include "../../tools/frameforge/frameforge-json.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace frameforge;

static void test_verb_conversion() {
    std::cout << "Testing verb conversion..." << std::endl;
    
    // Test basic verb
    assert(string_to_verb("PAN") == Verb::PAN);
    assert(verb_to_string(Verb::PAN) == "PAN");
    
    // Test misspelling
    assert(string_to_verb("PIN") == Verb::PAN);
    
    // Test case insensitivity
    assert(string_to_verb("pan") == Verb::PAN);
    assert(string_to_verb("Pan") == Verb::PAN);
    
    std::cout << "  ✓ Verb conversion tests passed" << std::endl;
}

static void test_action_group() {
    std::cout << "Testing action group mapping..." << std::endl;
    
    assert(get_action_group_for_verb(Verb::PAN) == ActionGroup::CAMERA_CONTROL);
    assert(get_action_group_for_verb(Verb::SET_POSE) == ActionGroup::ACTOR_POSE);
    assert(get_action_group_for_verb(Verb::ADD) == ActionGroup::OBJECT_MGMT);
    assert(get_action_group_for_verb(Verb::SHOT) == ActionGroup::SHOT_MGMT);
    
    std::cout << "  ✓ Action group tests passed" << std::endl;
}

static void test_required_parameters() {
    std::cout << "Testing required parameters..." << std::endl;
    
    auto pan_params = get_required_parameters(Verb::PAN);
    assert(pan_params.size() == 1);
    assert(pan_params[0] == "direction");
    
    auto lean_params = get_required_parameters(Verb::LEAN);
    assert(lean_params.size() == 2);
    
    auto add_params = get_required_parameters(Verb::ADD);
    assert(add_params.size() == 1);
    assert(add_params[0] == "target");
    
    std::cout << "  ✓ Required parameters tests passed" << std::endl;
}

static void test_valid_command() {
    std::cout << "Testing valid command validation..." << std::endl;
    
    CommandValidator validator;
    
    // Create a valid PAN command
    Command cmd;
    cmd.verb = Verb::PAN;
    cmd.subject = "Camera1";
    cmd.action_group = ActionGroup::CAMERA_CONTROL;
    cmd.parameters.direction = Direction::LEFT;
    
    ValidationResult result = validator.validate(cmd);
    assert(result.valid);
    
    std::cout << "  ✓ Valid command test passed" << std::endl;
}

static void test_missing_parameters() {
    std::cout << "Testing missing parameter detection..." << std::endl;
    
    CommandValidator validator;
    
    // Create PAN command without direction
    Command cmd;
    cmd.verb = Verb::PAN;
    cmd.subject = "Camera1";
    cmd.action_group = ActionGroup::CAMERA_CONTROL;
    // Missing direction parameter
    
    ValidationResult result = validator.validate(cmd);
    assert(!result.valid);
    assert(!result.missing_parameters.empty());
    
    std::cout << "  ✓ Missing parameter test passed" << std::endl;
}

static void test_json_parsing() {
    std::cout << "Testing JSON parsing and validation..." << std::endl;
    
    CommandValidator validator;
    
    std::string json_str = R"({
        "verb": "PAN",
        "subject": "Camera1",
        "action_group": "CAMERA_CONTROL",
        "parameters": {
            "direction": "LEFT"
        }
    })";
    
    Command cmd;
    ValidationResult result = validator.validate_json(json_str, cmd);
    
    assert(result.valid);
    assert(cmd.verb == Verb::PAN);
    assert(cmd.subject == "Camera1");
    assert(cmd.parameters.direction.has_value());
    assert(cmd.parameters.direction.value() == Direction::LEFT);
    
    std::cout << "  ✓ JSON parsing test passed" << std::endl;
}

static void test_json_serialization() {
    std::cout << "Testing JSON serialization..." << std::endl;
    
    Command cmd;
    cmd.verb = Verb::PAN;
    cmd.subject = "Camera1";
    cmd.action_group = ActionGroup::CAMERA_CONTROL;
    cmd.parameters.direction = Direction::LEFT;
    cmd.valid = true;
    
    std::string json = command_to_json(cmd);
    
    assert(!json.empty());
    assert(json.find("\"PAN\"") != std::string::npos);
    assert(json.find("\"Camera1\"") != std::string::npos);
    assert(json.find("\"LEFT\"") != std::string::npos);
    
    std::cout << "  ✓ JSON serialization test passed" << std::endl;
}

static void test_complex_command() {
    std::cout << "Testing complex command with pose..." << std::endl;
    
    CommandValidator validator;
    
    std::string json_str = R"({
        "verb": "SET_POSE",
        "subject": "Tom",
        "action_group": "ACTOR_POSE",
        "parameters": {
            "pose_description": "arms crossed",
            "joint_rotations": [
                {"name": "shoulder_left", "rotation_x": 0, "rotation_y": 45, "rotation_z": 0},
                {"name": "shoulder_right", "rotation_x": 0, "rotation_y": -45, "rotation_z": 0}
            ]
        }
    })";
    
    Command cmd;
    ValidationResult result = validator.validate_json(json_str, cmd);
    
    assert(result.valid);
    assert(cmd.verb == Verb::SET_POSE);
    assert(cmd.subject == "Tom");
    assert(cmd.parameters.joint_rotations.has_value());
    assert(cmd.parameters.joint_rotations.value().size() == 2);
    
    std::cout << "  ✓ Complex command test passed" << std::endl;
}

static void test_clarification_request() {
    std::cout << "Testing clarification request generation..." << std::endl;
    
    CommandValidator validator;
    
    Command cmd;
    cmd.verb = Verb::PAN;
    cmd.subject = "Camera1";
    cmd.action_group = ActionGroup::CAMERA_CONTROL;
    
    ValidationResult result = validator.validate(cmd);
    assert(!result.valid);
    
    std::string clarification = validator.generate_clarification_request(result, cmd);
    assert(!clarification.empty());
    assert(clarification.find("direction") != std::string::npos);
    
    std::cout << "  ✓ Clarification request test passed" << std::endl;
}

int main() {
    std::cout << "Running FrameForge Validator Tests..." << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        test_verb_conversion();
        test_action_group();
        test_required_parameters();
        test_valid_command();
        test_missing_parameters();
        test_json_parsing();
        test_json_serialization();
        test_complex_command();
        test_clarification_request();
        
        std::cout << "======================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
