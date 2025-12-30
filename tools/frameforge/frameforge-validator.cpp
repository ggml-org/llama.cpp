#include "frameforge-validator.h"
#include <sstream>
#include <algorithm>

// Use vendored nlohmann/json library
#include "../../vendor/nlohmann/json.hpp"

using json = nlohmann::json;

namespace frameforge {

CommandValidator::CommandValidator() {
}

CommandValidator::~CommandValidator() {
}

bool CommandValidator::check_required_parameters(
    const Command & cmd,
    std::vector<std::string> & missing
) const {
    missing.clear();
    
    std::vector<std::string> required = get_required_parameters(cmd.verb);
    
    for (const auto & param : required) {
        if (param == "direction" && !cmd.parameters.direction.has_value()) {
            missing.push_back("direction");
        } else if (param == "degrees" && !cmd.parameters.degrees.has_value()) {
            missing.push_back("degrees");
        } else if (param == "speed" && !cmd.parameters.speed.has_value()) {
            missing.push_back("speed");
        } else if (param == "target" && !cmd.parameters.target.has_value()) {
            missing.push_back("target");
        } else if (param == "subject" && !cmd.parameters.subject.has_value()) {
            missing.push_back("subject");
        } else if (param == "pose_description" && 
                   !cmd.parameters.pose_description.has_value() &&
                   !cmd.parameters.joint_rotations.has_value()) {
            missing.push_back("pose_description or joint_rotations");
        }
    }
    
    return missing.empty();
}

bool CommandValidator::validate_parameter_values(
    const Command & cmd,
    std::string & error
) const {
    // Validate direction if present
    if (cmd.parameters.direction.has_value()) {
        if (cmd.parameters.direction.value() == Direction::UNKNOWN) {
            error = "Invalid direction value";
            return false;
        }
    }
    
    // Validate degrees if present
    if (cmd.parameters.degrees.has_value()) {
        float degrees = cmd.parameters.degrees.value();
        if (degrees < -360.0f || degrees > 360.0f) {
            error = "Degrees must be between -360 and 360";
            return false;
        }
    }
    
    // Validate speed if present
    if (cmd.parameters.speed.has_value()) {
        float speed = cmd.parameters.speed.value();
        if (speed < 0.0f || speed > 100.0f) {
            error = "Speed must be between 0 and 100";
            return false;
        }
    }
    
    // Subject validation removed - it's now optional in parameters
    
    return true;
}

ValidationResult CommandValidator::validate(const Command & cmd) const {
    ValidationResult result;
    result.valid = true;
    
    // Check if verb is valid
    if (cmd.verb == Verb::UNKNOWN) {
        result.valid = false;
        result.error_message = "Unknown or invalid verb";
        return result;
    }
    
    // Check if action group matches verb
    ActionGroup expected_group = get_action_group_for_verb(cmd.verb);
    if (cmd.action_group != expected_group && cmd.action_group != ActionGroup::UNKNOWN) {
        result.valid = false;
        result.error_message = "Action group does not match verb";
        return result;
    }
    
    // Check required parameters
    if (!check_required_parameters(cmd, result.missing_parameters)) {
        result.valid = false;
        std::ostringstream oss;
        oss << "Missing required parameters: ";
        for (size_t i = 0; i < result.missing_parameters.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << result.missing_parameters[i];
        }
        result.error_message = oss.str();
        return result;
    }
    
    // Validate parameter values
    std::string value_error;
    if (!validate_parameter_values(cmd, value_error)) {
        result.valid = false;
        result.error_message = value_error;
        return result;
    }
    
    return result;
}

ValidationResult CommandValidator::validate_json(
    const std::string & json_str,
    Command & out_cmd
) const {
    ValidationResult result;
    result.valid = true;
    
    try {
        json j = json::parse(json_str);
        
        // Parse verb
        if (!j.contains("verb")) {
            result.valid = false;
            result.error_message = "Missing 'verb' field in JSON";
            return result;
        }
        out_cmd.verb = string_to_verb(j["verb"].get<std::string>());
        
        // Parse master_verb (optional)
        if (j.contains("master_verb")) {
            out_cmd.master_verb = string_to_verb(j["master_verb"].get<std::string>());
        }
        
        // Parse timestamp (optional, generate if missing)
        if (j.contains("timestamp")) {
            out_cmd.timestamp = j["timestamp"].get<std::string>();
        } else {
            out_cmd.timestamp = get_current_timestamp();
        }
        
        // Parse action_group (optional, can be inferred)
        if (j.contains("action_group")) {
            out_cmd.action_group = string_to_action_group(j["action_group"].get<std::string>());
        } else {
            out_cmd.action_group = get_action_group_for_verb(out_cmd.verb);
        }
        
        // Parse parameters
        if (j.contains("parameters")) {
            json params = j["parameters"];
            
            if (params.contains("subject")) {
                out_cmd.parameters.subject = params["subject"].get<std::string>();
            }
            
            if (params.contains("direction")) {
                out_cmd.parameters.direction = string_to_direction(params["direction"].get<std::string>());
            }
            
            if (params.contains("degrees")) {
                out_cmd.parameters.degrees = params["degrees"].get<float>();
            }
            
            if (params.contains("speed")) {
                out_cmd.parameters.speed = params["speed"].get<float>();
            }
            
            if (params.contains("target")) {
                out_cmd.parameters.target = params["target"].get<std::string>();
            }
            
            if (params.contains("pose_description")) {
                out_cmd.parameters.pose_description = params["pose_description"].get<std::string>();
            }
            
            if (params.contains("joint_rotations")) {
                std::vector<Joint> joints;
                for (const auto & joint_json : params["joint_rotations"]) {
                    Joint joint;
                    joint.name = joint_json["name"].get<std::string>();
                    joint.rotation_x = joint_json.value("rotation_x", 0.0f);
                    joint.rotation_y = joint_json.value("rotation_y", 0.0f);
                    joint.rotation_z = joint_json.value("rotation_z", 0.0f);
                    joints.push_back(joint);
                }
                out_cmd.parameters.joint_rotations = joints;
            }
            
            // Parse additional parameters
            for (auto & [key, value] : params.items()) {
                if (key != "direction" && key != "degrees" && key != "speed" &&
                    key != "target" && key != "pose_description" && key != "joint_rotations") {
                    if (!out_cmd.parameters.additional_params.has_value()) {
                        out_cmd.parameters.additional_params = std::map<std::string, std::string>();
                    }
                    out_cmd.parameters.additional_params.value()[key] = value.dump();
                }
            }
        }
        
        // Now validate the parsed command
        result = validate(out_cmd);
        
    } catch (const json::parse_error & e) {
        result.valid = false;
        result.error_message = std::string("JSON parse error: ") + e.what();
    } catch (const json::type_error & e) {
        result.valid = false;
        result.error_message = std::string("JSON type error: ") + e.what();
    } catch (const std::exception & e) {
        result.valid = false;
        result.error_message = std::string("Error: ") + e.what();
    }
    
    return result;
}

std::string CommandValidator::generate_clarification_request(
    const ValidationResult & result,
    const Command & cmd
) const {
    if (result.valid) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "I need clarification for the command '" 
        << verb_to_string(cmd.verb) 
        << "'";
    
    // Include subject if present in parameters
    if (cmd.parameters.subject.has_value()) {
        oss << " for '" << cmd.parameters.subject.value() << "'";
    }
    
    oss << ". ";
    
    if (!result.missing_parameters.empty()) {
        oss << "Please provide the following parameters: ";
        for (size_t i = 0; i < result.missing_parameters.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << result.missing_parameters[i];
        }
        oss << ".";
    } else {
        oss << result.error_message;
    }
    
    return oss.str();
}

} // namespace frameforge
