#include "frameforge-schema.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>

// Use vendored nlohmann/json library
#include "../../vendor/nlohmann/json.hpp"

using json = nlohmann::json;

namespace frameforge {

// Internal structure to store verb definition data
struct VerbDefinition {
    std::string name;
    ActionGroup action_group;
    std::vector<std::string> required_parameters;
    std::vector<std::string> optional_parameters;
    std::vector<std::string> aliases;
    std::string description;
    bool is_master_verb;
};

// Global storage for loaded verb definitions
static std::map<Verb, VerbDefinition> g_verb_definitions;
static std::map<std::string, Verb> g_string_to_verb_map;
static bool g_definitions_loaded = false;

// Convert string to uppercase for case-insensitive comparison
static std::string to_upper(const std::string & str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

std::string action_group_to_string(ActionGroup group) {
    switch (group) {
        case ActionGroup::CAMERA_CONTROL: return "CAMERA_CONTROL";
        case ActionGroup::ACTOR_POSE:     return "ACTOR_POSE";
        case ActionGroup::OBJECT_MGMT:    return "OBJECT_MGMT";
        case ActionGroup::SHOT_MGMT:      return "SHOT_MGMT";
        case ActionGroup::MASTER_VERB:    return "MASTER_VERB";
        case ActionGroup::UNKNOWN:        return "UNKNOWN";
    }
    return "UNKNOWN";
}

ActionGroup string_to_action_group(const std::string & str) {
    std::string upper = to_upper(str);
    if (upper == "CAMERA_CONTROL") return ActionGroup::CAMERA_CONTROL;
    if (upper == "ACTOR_POSE")     return ActionGroup::ACTOR_POSE;
    if (upper == "OBJECT_MGMT")    return ActionGroup::OBJECT_MGMT;
    if (upper == "SHOT_MGMT")      return ActionGroup::SHOT_MGMT;
    if (upper == "MASTER_VERB")    return ActionGroup::MASTER_VERB;
    return ActionGroup::UNKNOWN;
}

std::string verb_to_string(Verb verb) {
    switch (verb) {
        case Verb::START:       return "START";
        case Verb::BEGIN:       return "BEGIN";
        case Verb::HAVE:        return "HAVE";
        case Verb::MAKE:        return "MAKE";
        case Verb::STOP:        return "STOP";
        case Verb::PAN:         return "PAN";
        case Verb::TILT:        return "TILT";
        case Verb::DOLLY:       return "DOLLY";
        case Verb::ZOOM:        return "ZOOM";
        case Verb::LEAN:        return "LEAN";
        case Verb::SET_POSE:    return "SET_POSE";
        case Verb::ADJUST_POSE: return "ADJUST_POSE";
        case Verb::ADD:         return "ADD";
        case Verb::DELETE:      return "DELETE";
        case Verb::MOVE:        return "MOVE";
        case Verb::ROTATE:      return "ROTATE";
        case Verb::SHOT:        return "SHOT";
        case Verb::SAVE_SHOT:   return "SAVE_SHOT";
        case Verb::LOAD_SHOT:   return "LOAD_SHOT";
        case Verb::UNKNOWN:     return "UNKNOWN";
    }
    return "UNKNOWN";
}

Verb string_to_verb(const std::string & str) {
    std::string upper = to_upper(str);
    
    // If definitions are loaded, use the map
    if (g_definitions_loaded) {
        auto it = g_string_to_verb_map.find(upper);
        if (it != g_string_to_verb_map.end()) {
            return it->second;
        }
        return Verb::UNKNOWN;
    }
    
    // Fall back to hard-coded defaults
    // Handle common misspellings/alternatives
    if (upper == "START")                  return Verb::START;
    if (upper == "BEGIN")                  return Verb::BEGIN;
    if (upper == "HAVE")                   return Verb::HAVE;
    if (upper == "MAKE")                   return Verb::MAKE;
    if (upper == "STOP")                   return Verb::STOP;
    if (upper == "PIN" || upper == "PAN") return Verb::PAN;
    if (upper == "TILT")                   return Verb::TILT;
    if (upper == "DOLLY" || upper == "PUSH") return Verb::DOLLY;
    if (upper == "ZOOM" || upper == "ROOM") return Verb::ZOOM;
    if (upper == "LEAN")                   return Verb::LEAN;
    if (upper == "SET_POSE")               return Verb::SET_POSE;
    if (upper == "ADJUST_POSE")            return Verb::ADJUST_POSE;
    if (upper == "ADD")                    return Verb::ADD;
    if (upper == "DELETE" || upper == "REMOVE") return Verb::DELETE;
    if (upper == "MOVE" || upper == "WALK" || upper == "RUN") return Verb::MOVE;
    if (upper == "ROTATE" || upper == "TURN") return Verb::ROTATE;
    if (upper == "SHOT")                   return Verb::SHOT;
    if (upper == "SAVE_SHOT")              return Verb::SAVE_SHOT;
    if (upper == "LOAD_SHOT")              return Verb::LOAD_SHOT;
    return Verb::UNKNOWN;
}

std::string direction_to_string(Direction dir) {
    switch (dir) {
        case Direction::LEFT:     return "LEFT";
        case Direction::RIGHT:    return "RIGHT";
        case Direction::UP:       return "UP";
        case Direction::DOWN:     return "DOWN";
        case Direction::FORWARD:  return "FORWARD";
        case Direction::BACKWARD: return "BACKWARD";
        case Direction::UNKNOWN:  return "UNKNOWN";
    }
    return "UNKNOWN";
}

Direction string_to_direction(const std::string & str) {
    std::string upper = to_upper(str);
    if (upper == "LEFT")          return Direction::LEFT;
    if (upper == "RIGHT")         return Direction::RIGHT;
    if (upper == "UP")            return Direction::UP;
    if (upper == "DOWN")          return Direction::DOWN;
    if (upper == "FORWARD")       return Direction::FORWARD;
    if (upper == "BACKWARD")      return Direction::BACKWARD;
    return Direction::UNKNOWN;
}

ActionGroup get_action_group_for_verb(Verb verb) {
    // If definitions are loaded, use them
    if (g_definitions_loaded && g_verb_definitions.find(verb) != g_verb_definitions.end()) {
        return g_verb_definitions[verb].action_group;
    }
    
    // Fall back to hard-coded defaults
    switch (verb) {
        case Verb::START:
        case Verb::BEGIN:
        case Verb::HAVE:
        case Verb::MAKE:
        case Verb::STOP:
            return ActionGroup::MASTER_VERB;
            
        case Verb::PAN:
        case Verb::TILT:
        case Verb::DOLLY:
        case Verb::ZOOM:
        case Verb::LEAN:
            return ActionGroup::CAMERA_CONTROL;
            
        case Verb::SET_POSE:
        case Verb::ADJUST_POSE:
            return ActionGroup::ACTOR_POSE;
            
        case Verb::ADD:
        case Verb::DELETE:
        case Verb::MOVE:
        case Verb::ROTATE:
            return ActionGroup::OBJECT_MGMT;
            
        case Verb::SHOT:
        case Verb::SAVE_SHOT:
        case Verb::LOAD_SHOT:
            return ActionGroup::SHOT_MGMT;
            
        case Verb::UNKNOWN:
            return ActionGroup::UNKNOWN;
    }
    return ActionGroup::UNKNOWN;
}

std::vector<std::string> get_required_parameters(Verb verb) {
    // If definitions are loaded, use them
    if (g_definitions_loaded && g_verb_definitions.find(verb) != g_verb_definitions.end()) {
        return g_verb_definitions[verb].required_parameters;
    }
    
    // Fall back to hard-coded defaults
    switch (verb) {
        case Verb::START:
        case Verb::BEGIN:
        case Verb::STOP:
            return {};
            
        case Verb::HAVE:
        case Verb::MAKE:
            return {"subject"};
            
        case Verb::PAN:
        case Verb::TILT:
            return {"direction"};
            
        case Verb::DOLLY:
            return {"direction"};
            
        case Verb::ZOOM:
            return {"direction"};
            
        case Verb::LEAN:
            return {"direction", "degrees"};
            
        case Verb::SET_POSE:
        case Verb::ADJUST_POSE:
            return {"subject", "pose_description"};
            
        case Verb::ADD:
            return {"target"};
            
        case Verb::DELETE:
            return {"target"};
            
        case Verb::MOVE:
            return {"target", "direction"};
            
        case Verb::ROTATE:
            return {"target", "degrees"};
            
        case Verb::SHOT:
        case Verb::SAVE_SHOT:
        case Verb::LOAD_SHOT:
            return {"target"};
            
        case Verb::UNKNOWN:
            return {};
    }
    return {};
}

std::vector<std::string> get_optional_parameters(Verb verb) {
    // If definitions are loaded, use them
    if (g_definitions_loaded && g_verb_definitions.find(verb) != g_verb_definitions.end()) {
        return g_verb_definitions[verb].optional_parameters;
    }
    
    // Fall back to reasonable defaults for all verbs
    return {"speed", "degrees", "target", "subject"};
}

bool is_master_verb(Verb verb) {
    // If definitions are loaded, use them
    if (g_definitions_loaded && g_verb_definitions.find(verb) != g_verb_definitions.end()) {
        return g_verb_definitions[verb].is_master_verb;
    }
    
    // Fall back to hard-coded check
    return verb == Verb::START || verb == Verb::BEGIN || 
           verb == Verb::HAVE || verb == Verb::MAKE || verb == Verb::STOP;
}

std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif
    
    char buffer[30];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &tm_now);
    
    // Add milliseconds
    char result[35];
    std::snprintf(result, sizeof(result), "%s.%03dZ", buffer, static_cast<int>(ms.count()));
    
    return std::string(result);
}

bool load_verb_definitions(const std::string & json_path) {
    try {
        // Read JSON file
        std::ifstream file(json_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open verb definitions file: " << json_path << std::endl;
            return false;
        }
        
        json j;
        file >> j;
        file.close();
        
        // Clear existing definitions
        g_verb_definitions.clear();
        g_string_to_verb_map.clear();
        
        // Load verbs
        if (!j.contains("verbs") || !j["verbs"].is_array()) {
            std::cerr << "Error: JSON must contain a 'verbs' array" << std::endl;
            return false;
        }
        
        for (const auto & verb_json : j["verbs"]) {
            // Parse verb name
            std::string verb_name = verb_json["name"].get<std::string>();
            Verb verb = string_to_verb(verb_name); // Use existing enum mapping
            
            if (verb == Verb::UNKNOWN) {
                std::cerr << "Warning: Unknown verb '" << verb_name << "' in JSON, skipping" << std::endl;
                continue;
            }
            
            VerbDefinition def;
            def.name = verb_name;
            
            // Parse action group
            std::string action_group_str = verb_json["action_group"].get<std::string>();
            def.action_group = string_to_action_group(action_group_str);
            
            // Parse required parameters
            if (verb_json.contains("required_parameters")) {
                for (const auto & param : verb_json["required_parameters"]) {
                    def.required_parameters.push_back(param.get<std::string>());
                }
            }
            
            // Parse optional parameters
            if (verb_json.contains("optional_parameters")) {
                for (const auto & param : verb_json["optional_parameters"]) {
                    def.optional_parameters.push_back(param.get<std::string>());
                }
            }
            
            // Parse is_master_verb
            if (verb_json.contains("is_master_verb")) {
                def.is_master_verb = verb_json["is_master_verb"].get<bool>();
            } else {
                def.is_master_verb = false;
            }
            
            // Parse aliases
            if (verb_json.contains("aliases")) {
                for (const auto & alias : verb_json["aliases"]) {
                    def.aliases.push_back(alias.get<std::string>());
                }
            }
            
            // Parse description (optional)
            if (verb_json.contains("description")) {
                def.description = verb_json["description"].get<std::string>();
            }
            
            // Store the definition
            g_verb_definitions[verb] = def;
            
            // Add to string-to-verb map (main name)
            g_string_to_verb_map[to_upper(verb_name)] = verb;
            
            // Add aliases to map
            for (const auto & alias : def.aliases) {
                g_string_to_verb_map[to_upper(alias)] = verb;
            }
        }
        
        g_definitions_loaded = true;
        std::cerr << "Successfully loaded " << g_verb_definitions.size() << " verb definitions from " << json_path << std::endl;
        return true;
        
    } catch (const json::parse_error & e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return false;
    } catch (const std::exception & e) {
        std::cerr << "Error loading verb definitions: " << e.what() << std::endl;
        return false;
    }
}

bool are_verb_definitions_loaded() {
    return g_definitions_loaded;
}

} // namespace frameforge
