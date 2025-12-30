#include "../tools/frameforge/frameforge-schema.h"
#include "../tools/frameforge/frameforge-validator.h"
#include "../tools/frameforge/frameforge-json.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace frameforge;

static void test_master_verb_detection() {
    std::cout << "Testing master verb detection..." << std::endl;
    
    assert(is_master_verb(Verb::START));
    assert(is_master_verb(Verb::BEGIN));
    assert(is_master_verb(Verb::HAVE));
    assert(is_master_verb(Verb::MAKE));
    assert(is_master_verb(Verb::STOP));
    
    assert(!is_master_verb(Verb::PAN));
    assert(!is_master_verb(Verb::MOVE));
    
    std::cout << "  ✓ Master verb detection passed" << std::endl;
}

static void test_timestamp_generation() {
    std::cout << "Testing timestamp generation..." << std::endl;
    
    std::string ts = get_current_timestamp();
    assert(!ts.empty());
    assert(ts.find('T') != std::string::npos);  // Should contain ISO 8601 separator
    assert(ts.find('Z') != std::string::npos);  // Should end with Z
    
    std::cout << "  Generated timestamp: " << ts << std::endl;
    std::cout << "  ✓ Timestamp generation passed" << std::endl;
}

static void test_subject_in_parameters() {
    std::cout << "Testing subject in parameters..." << std::endl;
    
    Command cmd;
    cmd.verb = Verb::PAN;
    cmd.action_group = ActionGroup::CAMERA_CONTROL;
    cmd.timestamp = get_current_timestamp();
    cmd.parameters.subject = "Camera1";
    cmd.parameters.direction = Direction::LEFT;
    cmd.valid = true;
    
    // Serialize to JSON
    std::string json = command_to_json(cmd);
    
    // Check that subject is in parameters, not at root
    assert(json.find("\"parameters\"") != std::string::npos);
    assert(json.find("\"subject\": \"Camera1\"") != std::string::npos || 
           json.find("\"subject\":\"Camera1\"") != std::string::npos);
    
    // Parse back
    Command parsed = json_to_command(json);
    assert(parsed.parameters.subject.has_value());
    assert(parsed.parameters.subject.value() == "Camera1");
    
    std::cout << "  ✓ Subject in parameters passed" << std::endl;
}

static void test_timestamp_in_json() {
    std::cout << "Testing timestamp in JSON..." << std::endl;
    
    Command cmd;
    cmd.verb = Verb::TILT;
    cmd.action_group = ActionGroup::CAMERA_CONTROL;
    cmd.timestamp = "2024-01-01T12:00:00.000Z";
    cmd.parameters.direction = Direction::UP;
    cmd.valid = true;
    
    std::string json = command_to_json(cmd);
    
    assert(json.find("timestamp") != std::string::npos);
    assert(json.find("2024-01-01T12:00:00.000Z") != std::string::npos);
    
    std::cout << "  ✓ Timestamp in JSON passed" << std::endl;
}

static void test_master_verb_command() {
    std::cout << "Testing master verb command..." << std::endl;
    
    // Create a command like "START PANNING LEFT"
    Command cmd;
    cmd.verb = Verb::PAN;
    cmd.master_verb = Verb::START;
    cmd.action_group = ActionGroup::CAMERA_CONTROL;
    cmd.timestamp = get_current_timestamp();
    cmd.parameters.direction = Direction::LEFT;
    cmd.parameters.speed = 5.0f;
    cmd.valid = true;
    
    std::string json = command_to_json(cmd);
    
    // Should have both verb and master_verb
    assert(json.find("\"verb\": \"PAN\"") != std::string::npos || 
           json.find("\"verb\":\"PAN\"") != std::string::npos);
    assert(json.find("\"master_verb\": \"START\"") != std::string::npos || 
           json.find("\"master_verb\":\"START\"") != std::string::npos);
    
    // Parse back
    Command parsed = json_to_command(json);
    assert(parsed.verb == Verb::PAN);
    assert(parsed.master_verb.has_value());
    assert(parsed.master_verb.value() == Verb::START);
    
    std::cout << "  ✓ Master verb command passed" << std::endl;
}

static void test_verb_aliases() {
    std::cout << "Testing verb aliases..." << std::endl;
    
    // Load definitions to get aliases
    std::string json_path = "../tools/frameforge/verb-definitions.json";
    bool loaded = load_verb_definitions(json_path);
    assert(loaded);
    
    // Test aliases
    assert(string_to_verb("PIN") == Verb::PAN);
    assert(string_to_verb("ROOM") == Verb::ZOOM);
    assert(string_to_verb("PUSH") == Verb::DOLLY);
    assert(string_to_verb("REMOVE") == Verb::DELETE);
    assert(string_to_verb("WALK") == Verb::MOVE);
    assert(string_to_verb("RUN") == Verb::MOVE);
    assert(string_to_verb("TURN") == Verb::ROTATE);
    
    std::cout << "  ✓ Verb aliases passed" << std::endl;
}

static void test_optional_parameters() {
    std::cout << "Testing optional parameters..." << std::endl;
    
    // Ensure definitions are loaded
    if (!are_verb_definitions_loaded()) {
        load_verb_definitions("../tools/frameforge/verb-definitions.json");
    }
    
    auto pan_optional = get_optional_parameters(Verb::PAN);
    assert(!pan_optional.empty());
    
    auto start_optional = get_optional_parameters(Verb::START);
    assert(!start_optional.empty());
    
    std::cout << "  PAN optional params: " << pan_optional.size() << std::endl;
    std::cout << "  START optional params: " << start_optional.size() << std::endl;
    std::cout << "  ✓ Optional parameters passed" << std::endl;
}

static void test_have_command() {
    std::cout << "Testing HAVE command (HAVE TOM WALK FORWARD)..." << std::endl;
    
    CommandValidator validator;
    
    std::string json_str = R"({
        "verb": "MOVE",
        "master_verb": "HAVE",
        "action_group": "OBJECT_MGMT",
        "timestamp": "2024-01-01T12:00:00.000Z",
        "parameters": {
            "subject": "Tom",
            "target": "Tom",
            "direction": "FORWARD",
            "speed": 5.0
        }
    })";
    
    Command cmd;
    ValidationResult result = validator.validate_json(json_str, cmd);
    
    assert(result.valid);
    assert(cmd.verb == Verb::MOVE);
    assert(cmd.master_verb.has_value());
    assert(cmd.master_verb.value() == Verb::HAVE);
    assert(cmd.parameters.subject.has_value());
    assert(cmd.parameters.subject.value() == "Tom");
    
    std::cout << "  ✓ HAVE command passed" << std::endl;
}

static void test_new_json_format() {
    std::cout << "Testing new Delphi Bridge JSON format..." << std::endl;
    
    CommandValidator validator;
    
    // Test complete JSON with all new features
    std::string json_str = R"({
        "verb": "ZOOM",
        "action_group": "CAMERA_CONTROL",
        "timestamp": "2024-12-30T10:00:00.000Z",
        "parameters": {
            "subject": "MainCamera",
            "direction": "IN",
            "speed": 10.0
        }
    })";
    
    Command cmd;
    ValidationResult result = validator.validate_json(json_str, cmd);
    
    assert(result.valid);
    assert(cmd.verb == Verb::ZOOM);
    assert(!cmd.timestamp.empty());
    assert(cmd.parameters.subject.has_value());
    assert(cmd.parameters.direction.has_value());
    
    // Serialize back and ensure all fields present
    std::string output_json = command_to_json(cmd);
    assert(output_json.find("timestamp") != std::string::npos);
    assert(output_json.find("parameters") != std::string::npos);
    assert(output_json.find("subject") != std::string::npos);
    
    std::cout << "  ✓ New JSON format passed" << std::endl;
}

int main() {
    std::cout << "Running FrameForge New Features Tests..." << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        test_master_verb_detection();
        test_timestamp_generation();
        test_subject_in_parameters();
        test_timestamp_in_json();
        test_master_verb_command();
        test_verb_aliases();
        test_optional_parameters();
        test_have_command();
        test_new_json_format();
        
        std::cout << "==========================================" << std::endl;
        std::cout << "All new feature tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
