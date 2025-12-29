#include "frameforge-json.h"

#include "../../vendor/nlohmann/json.hpp"

using json = nlohmann::json;

namespace frameforge {

std::string command_to_json(const Command & cmd) {
    json j;
    
    j["verb"] = verb_to_string(cmd.verb);
    j["subject"] = cmd.subject;
    j["action_group"] = action_group_to_string(cmd.action_group);
    j["valid"] = cmd.valid;
    
    if (!cmd.error_message.empty()) {
        j["error_message"] = cmd.error_message;
    }
    
    json params = json::object();
    
    if (cmd.parameters.direction.has_value()) {
        params["direction"] = direction_to_string(cmd.parameters.direction.value());
    }
    
    if (cmd.parameters.degrees.has_value()) {
        params["degrees"] = cmd.parameters.degrees.value();
    }
    
    if (cmd.parameters.speed.has_value()) {
        params["speed"] = cmd.parameters.speed.value();
    }
    
    if (cmd.parameters.target.has_value()) {
        params["target"] = cmd.parameters.target.value();
    }
    
    if (cmd.parameters.pose_description.has_value()) {
        params["pose_description"] = cmd.parameters.pose_description.value();
    }
    
    if (cmd.parameters.joint_rotations.has_value()) {
        json joints = json::array();
        for (const auto & joint : cmd.parameters.joint_rotations.value()) {
            json joint_json;
            joint_json["name"] = joint.name;
            joint_json["rotation_x"] = joint.rotation_x;
            joint_json["rotation_y"] = joint.rotation_y;
            joint_json["rotation_z"] = joint.rotation_z;
            joints.push_back(joint_json);
        }
        params["joint_rotations"] = joints;
    }
    
    if (cmd.parameters.additional_params.has_value()) {
        for (const auto & [key, value] : cmd.parameters.additional_params.value()) {
            params[key] = value;
        }
    }
    
    j["parameters"] = params;
    
    return j.dump(2);
}

Command json_to_command(const std::string & json_str) {
    Command cmd;
    
    try {
        json j = json::parse(json_str);
        
        cmd.verb = string_to_verb(j.value("verb", ""));
        cmd.subject = j.value("subject", "");
        cmd.action_group = string_to_action_group(j.value("action_group", ""));
        cmd.valid = j.value("valid", false);
        cmd.error_message = j.value("error_message", "");
        
        if (j.contains("parameters")) {
            json params = j["parameters"];
            
            if (params.contains("direction")) {
                cmd.parameters.direction = string_to_direction(params["direction"].get<std::string>());
            }
            
            if (params.contains("degrees")) {
                cmd.parameters.degrees = params["degrees"].get<float>();
            }
            
            if (params.contains("speed")) {
                cmd.parameters.speed = params["speed"].get<float>();
            }
            
            if (params.contains("target")) {
                cmd.parameters.target = params["target"].get<std::string>();
            }
            
            if (params.contains("pose_description")) {
                cmd.parameters.pose_description = params["pose_description"].get<std::string>();
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
                cmd.parameters.joint_rotations = joints;
            }
        }
        
    } catch (const std::exception & e) {
        cmd.valid = false;
        cmd.error_message = std::string("Error parsing JSON: ") + e.what();
    }
    
    return cmd;
}

std::string validation_error_to_json(
    const std::string & error_message,
    const std::vector<std::string> & missing_params
) {
    json j;
    j["error"] = error_message;
    j["valid"] = false;
    
    if (!missing_params.empty()) {
        j["missing_parameters"] = missing_params;
    }
    
    return j.dump(2);
}

} // namespace frameforge
