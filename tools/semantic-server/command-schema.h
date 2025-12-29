#pragma once

#include "ggml.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace semantic_server {

// Action Groups for Intent Classification
enum class ActionGroup {
    CAMERA_CONTROL,
    ACTOR_POSE,
    OBJECT_MGMT,
    SHOT_MGMT,
    UNKNOWN
};

// Command Verbs
enum class Verb {
    PAN,
    TILT,
    LEAN,
    ROLL,
    DOLLY,
    TRUCK,
    PEDESTAL,
    ZOOM,
    FOCUS,
    ADD,
    DELETE,
    MODIFY,
    SELECT,
    SHOT,
    CUT,
    UNKNOWN
};

// Parameter types for validation
struct ParameterSpec {
    std::string name;
    std::string type; // "string", "number", "boolean", "array", "object"
    bool required;
    json default_value;
};

// Command Schema Definition
struct CommandSchema {
    Verb verb;
    ActionGroup action_group;
    std::vector<ParameterSpec> parameters;
};

// Convert enum to string
inline std::string action_group_to_string(ActionGroup group) {
    switch (group) {
        case ActionGroup::CAMERA_CONTROL: return "CAMERA_CONTROL";
        case ActionGroup::ACTOR_POSE: return "ACTOR_POSE";
        case ActionGroup::OBJECT_MGMT: return "OBJECT_MGMT";
        case ActionGroup::SHOT_MGMT: return "SHOT_MGMT";
        default: return "UNKNOWN";
    }
}

inline std::string verb_to_string(Verb verb) {
    switch (verb) {
        case Verb::PAN: return "PAN";
        case Verb::TILT: return "TILT";
        case Verb::LEAN: return "LEAN";
        case Verb::ROLL: return "ROLL";
        case Verb::DOLLY: return "DOLLY";
        case Verb::TRUCK: return "TRUCK";
        case Verb::PEDESTAL: return "PEDESTAL";
        case Verb::ZOOM: return "ZOOM";
        case Verb::FOCUS: return "FOCUS";
        case Verb::ADD: return "ADD";
        case Verb::DELETE: return "DELETE";
        case Verb::MODIFY: return "MODIFY";
        case Verb::SELECT: return "SELECT";
        case Verb::SHOT: return "SHOT";
        case Verb::CUT: return "CUT";
        default: return "UNKNOWN";
    }
}

inline Verb string_to_verb(const std::string & str) {
    static const std::unordered_map<std::string, Verb> verb_map = {
        {"PAN", Verb::PAN},
        {"TILT", Verb::TILT},
        {"LEAN", Verb::LEAN},
        {"ROLL", Verb::ROLL},
        {"DOLLY", Verb::DOLLY},
        {"TRUCK", Verb::TRUCK},
        {"PEDESTAL", Verb::PEDESTAL},
        {"ZOOM", Verb::ZOOM},
        {"FOCUS", Verb::FOCUS},
        {"ADD", Verb::ADD},
        {"DELETE", Verb::DELETE},
        {"MODIFY", Verb::MODIFY},
        {"SELECT", Verb::SELECT},
        {"SHOT", Verb::SHOT},
        {"CUT", Verb::CUT},
    };
    
    auto it = verb_map.find(str);
    return (it != verb_map.end()) ? it->second : Verb::UNKNOWN;
}

inline ActionGroup string_to_action_group(const std::string & str) {
    static const std::unordered_map<std::string, ActionGroup> group_map = {
        {"CAMERA_CONTROL", ActionGroup::CAMERA_CONTROL},
        {"ACTOR_POSE", ActionGroup::ACTOR_POSE},
        {"OBJECT_MGMT", ActionGroup::OBJECT_MGMT},
        {"SHOT_MGMT", ActionGroup::SHOT_MGMT},
    };
    
    auto it = group_map.find(str);
    return (it != group_map.end()) ? it->second : ActionGroup::UNKNOWN;
}

// Schema registry - defines all valid commands and their parameters
class SchemaRegistry {
public:
    static SchemaRegistry & instance() {
        static SchemaRegistry registry;
        return registry;
    }

    const CommandSchema * get_schema(Verb verb) const {
        auto it = schemas.find(verb);
        return (it != schemas.end()) ? &it->second : nullptr;
    }

    const std::vector<std::string> & get_verbs_for_group(ActionGroup group) const {
        auto it = group_verbs.find(group);
        static const std::vector<std::string> empty;
        return (it != group_verbs.end()) ? it->second : empty;
    }

private:
    SchemaRegistry() {
        initialize_schemas();
    }

    void initialize_schemas() {
        // CAMERA_CONTROL commands
        schemas[Verb::PAN] = {
            Verb::PAN,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "LEFT", "RIGHT"
                {"degrees", "number", false, 0.0},
                {"speed", "number", false, 1.0}
            }
        };

        schemas[Verb::TILT] = {
            Verb::TILT,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "UP", "DOWN"
                {"degrees", "number", false, 0.0},
                {"speed", "number", false, 1.0}
            }
        };

        schemas[Verb::LEAN] = {
            Verb::LEAN,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "LEFT", "RIGHT"
                {"degrees", "number", false, 0.0}
            }
        };

        schemas[Verb::ROLL] = {
            Verb::ROLL,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "LEFT", "RIGHT", "CW", "CCW"
                {"degrees", "number", false, 0.0}
            }
        };

        schemas[Verb::DOLLY] = {
            Verb::DOLLY,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "IN", "OUT"
                {"distance", "number", false, 0.0},
                {"speed", "number", false, 1.0}
            }
        };

        schemas[Verb::TRUCK] = {
            Verb::TRUCK,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "LEFT", "RIGHT"
                {"distance", "number", false, 0.0}
            }
        };

        schemas[Verb::PEDESTAL] = {
            Verb::PEDESTAL,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "UP", "DOWN"
                {"distance", "number", false, 0.0}
            }
        };

        schemas[Verb::ZOOM] = {
            Verb::ZOOM,
            ActionGroup::CAMERA_CONTROL,
            {
                {"direction", "string", true, nullptr},     // "IN", "OUT"
                {"factor", "number", false, 1.0},
                {"speed", "number", false, 1.0}
            }
        };

        schemas[Verb::FOCUS] = {
            Verb::FOCUS,
            ActionGroup::CAMERA_CONTROL,
            {
                {"target", "string", false, ""},
                {"distance", "number", false, 0.0}
            }
        };

        // ACTOR_POSE commands
        schemas[Verb::MODIFY] = {
            Verb::MODIFY,
            ActionGroup::ACTOR_POSE,
            {
                {"subject", "string", true, nullptr},
                {"pose_description", "string", true, nullptr},
                {"joint_rotations", "array", false, json::array()}
            }
        };

        // OBJECT_MGMT commands
        schemas[Verb::ADD] = {
            Verb::ADD,
            ActionGroup::OBJECT_MGMT,
            {
                {"object_type", "string", true, nullptr},
                {"name", "string", false, ""},
                {"position", "object", false, json::object()},
                {"properties", "object", false, json::object()}
            }
        };

        schemas[Verb::DELETE] = {
            Verb::DELETE,
            ActionGroup::OBJECT_MGMT,
            {
                {"target", "string", true, nullptr}
            }
        };

        schemas[Verb::SELECT] = {
            Verb::SELECT,
            ActionGroup::OBJECT_MGMT,
            {
                {"target", "string", true, nullptr}
            }
        };

        // SHOT_MGMT commands
        schemas[Verb::SHOT] = {
            Verb::SHOT,
            ActionGroup::SHOT_MGMT,
            {
                {"shot_type", "string", true, nullptr},      // "WIDE", "MEDIUM", "CLOSE", "ECU"
                {"subject", "string", false, ""},
                {"duration", "number", false, 0.0}
            }
        };

        schemas[Verb::CUT] = {
            Verb::CUT,
            ActionGroup::SHOT_MGMT,
            {
                {"transition", "string", false, "CUT"},      // "CUT", "DISSOLVE", "FADE"
                {"duration", "number", false, 0.0}
            }
        };

        // Build reverse mapping: ActionGroup -> Verbs
        group_verbs[ActionGroup::CAMERA_CONTROL] = {
            "PAN", "TILT", "LEAN", "ROLL", "DOLLY", "TRUCK", "PEDESTAL", "ZOOM", "FOCUS"
        };
        group_verbs[ActionGroup::ACTOR_POSE] = {"MODIFY"};
        group_verbs[ActionGroup::OBJECT_MGMT] = {"ADD", "DELETE", "SELECT"};
        group_verbs[ActionGroup::SHOT_MGMT] = {"SHOT", "CUT"};
    }

    std::unordered_map<Verb, CommandSchema> schemas;
    std::unordered_map<ActionGroup, std::vector<std::string>> group_verbs;
};

} // namespace semantic_server
