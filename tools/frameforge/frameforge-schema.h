#ifndef FRAMEFORGE_SCHEMA_H
#define FRAMEFORGE_SCHEMA_H

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace frameforge {

// Action Groups
enum class ActionGroup {
    CAMERA_CONTROL,
    ACTOR_POSE,
    OBJECT_MGMT,
    SHOT_MGMT,
    UNKNOWN
};

// Command Verbs
enum class Verb {
    // Camera Control
    PAN,
    TILT,
    DOLLY,
    ZOOM,
    LEAN,
    
    // Actor Pose
    SET_POSE,
    ADJUST_POSE,
    
    // Object Management
    ADD,
    DELETE,
    MOVE,
    ROTATE,
    
    // Shot Management
    SHOT,
    SAVE_SHOT,
    LOAD_SHOT,
    
    UNKNOWN
};

// Direction enum
enum class Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN,
    FORWARD,
    BACKWARD,
    UNKNOWN
};

// Joint definition for pose descriptions
struct Joint {
    std::string name;        // e.g., "shoulder_left", "elbow_right"
    float rotation_x;        // rotation in degrees
    float rotation_y;
    float rotation_z;
};

// Command Parameters structure
struct CommandParameters {
    std::optional<Direction> direction;
    std::optional<float> degrees;
    std::optional<float> speed;
    std::optional<std::string> target;
    std::optional<std::string> pose_description;
    std::optional<std::vector<Joint>> joint_rotations;
    std::optional<std::map<std::string, std::string>> additional_params;
};

// Main Command structure
struct Command {
    Verb verb;
    std::string subject;           // e.g., "Camera1", "Tom"
    ActionGroup action_group;
    CommandParameters parameters;
    bool valid;
    std::string error_message;
};

// Helper functions to convert enums to/from strings
std::string action_group_to_string(ActionGroup group);
ActionGroup string_to_action_group(const std::string & str);

std::string verb_to_string(Verb verb);
Verb string_to_verb(const std::string & str);

std::string direction_to_string(Direction dir);
Direction string_to_direction(const std::string & str);

// Get action group for a verb
ActionGroup get_action_group_for_verb(Verb verb);

// Get required parameters for a verb
std::vector<std::string> get_required_parameters(Verb verb);

// Load verb definitions from JSON file
bool load_verb_definitions(const std::string & json_path);

// Check if verb definitions are loaded
bool are_verb_definitions_loaded();

} // namespace frameforge

#endif // FRAMEFORGE_SCHEMA_H
