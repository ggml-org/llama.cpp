#include "frameforge-schema.h"
#include <algorithm>
#include <cctype>

namespace frameforge {

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
    return ActionGroup::UNKNOWN;
}

std::string verb_to_string(Verb verb) {
    switch (verb) {
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
    // Handle common misspellings/alternatives
    if (upper == "PIN" || upper == "PAN") return Verb::PAN;
    if (upper == "TILT")                   return Verb::TILT;
    if (upper == "DOLLY")                  return Verb::DOLLY;
    if (upper == "ZOOM")                   return Verb::ZOOM;
    if (upper == "LEAN")                   return Verb::LEAN;
    if (upper == "SET_POSE")               return Verb::SET_POSE;
    if (upper == "ADJUST_POSE")            return Verb::ADJUST_POSE;
    if (upper == "ADD")                    return Verb::ADD;
    if (upper == "DELETE" || upper == "REMOVE") return Verb::DELETE;
    if (upper == "MOVE")                   return Verb::MOVE;
    if (upper == "ROTATE")                 return Verb::ROTATE;
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
    switch (verb) {
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
    switch (verb) {
        case Verb::PAN:
        case Verb::TILT:
            return {"direction"};
            
        case Verb::DOLLY:
            return {"direction", "speed"};
            
        case Verb::ZOOM:
            return {"direction"};
            
        case Verb::LEAN:
            return {"direction", "degrees"};
            
        case Verb::SET_POSE:
        case Verb::ADJUST_POSE:
            return {"pose_description"};
            
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

} // namespace frameforge
