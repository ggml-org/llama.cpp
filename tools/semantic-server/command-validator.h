#pragma once

#include "command-schema.h"
#include <string>
#include <vector>

namespace semantic_server {

// Validation result
struct ValidationResult {
    bool valid;
    std::string error_message;
    std::vector<std::string> missing_parameters;
    json validated_command;
    
    ValidationResult() : valid(true) {}
    
    static ValidationResult success(const json & cmd) {
        ValidationResult result;
        result.valid = true;
        result.validated_command = cmd;
        return result;
    }
    
    static ValidationResult error(const std::string & msg, const std::vector<std::string> & missing = {}) {
        ValidationResult result;
        result.valid = false;
        result.error_message = msg;
        result.missing_parameters = missing;
        return result;
    }
    
    json to_json() const {
        if (valid) {
            return validated_command;
        } else {
            json error_obj = {
                {"error", true},
                {"message", error_message}
            };
            if (!missing_parameters.empty()) {
                error_obj["missing_parameters"] = missing_parameters;
            }
            return error_obj;
        }
    }
};

// Command Validator class
class CommandValidator {
public:
    CommandValidator() : registry(SchemaRegistry::instance()) {}
    
    // Validate a command JSON object against the schema
    ValidationResult validate(const json & command) const {
        // Check if command is an object
        if (!command.is_object()) {
            return ValidationResult::error("Command must be a JSON object");
        }
        
        // Check for required "verb" field
        if (!command.contains("verb")) {
            return ValidationResult::error("Command must contain a 'verb' field");
        }
        
        std::string verb_str = command["verb"].get<std::string>();
        Verb verb = string_to_verb(verb_str);
        
        if (verb == Verb::UNKNOWN) {
            return ValidationResult::error("Unknown verb: " + verb_str);
        }
        
        // Get schema for this verb
        const CommandSchema * schema = registry.get_schema(verb);
        if (!schema) {
            return ValidationResult::error("No schema found for verb: " + verb_str);
        }
        
        // Validate parameters
        std::vector<std::string> missing_params;
        json validated_cmd = command;
        
        // Ensure action_group is set
        if (!validated_cmd.contains("action_group")) {
            validated_cmd["action_group"] = action_group_to_string(schema->action_group);
        }
        
        // Initialize parameters object if not present
        if (!validated_cmd.contains("parameters")) {
            validated_cmd["parameters"] = json::object();
        }
        
        json & params = validated_cmd["parameters"];
        
        // Check each parameter in schema
        for (const auto & param_spec : schema->parameters) {
            if (!params.contains(param_spec.name)) {
                if (param_spec.required) {
                    missing_params.push_back(param_spec.name);
                } else if (!param_spec.default_value.is_null()) {
                    // Set default value
                    params[param_spec.name] = param_spec.default_value;
                }
            } else {
                // Validate parameter type
                if (!validate_parameter_type(params[param_spec.name], param_spec.type)) {
                    return ValidationResult::error(
                        "Parameter '" + param_spec.name + "' has incorrect type. Expected: " + param_spec.type
                    );
                }
            }
        }
        
        if (!missing_params.empty()) {
            std::string msg = "Missing required parameters for " + verb_str + ": ";
            for (size_t i = 0; i < missing_params.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += missing_params[i];
            }
            return ValidationResult::error(msg, missing_params);
        }
        
        return ValidationResult::success(validated_cmd);
    }
    
    // Fuzzy matching for typo correction (e.g., "PIN" -> "PAN")
    std::string suggest_verb(const std::string & input) const {
        std::string upper_input = to_upper(input);
        
        // Direct match
        if (string_to_verb(upper_input) != Verb::UNKNOWN) {
            return upper_input;
        }
        
        // Simple edit distance check for common typos
        std::vector<std::string> all_verbs = {
            "PAN", "TILT", "LEAN", "ROLL", "DOLLY", "TRUCK", "PEDESTAL", "ZOOM", "FOCUS",
            "ADD", "DELETE", "MODIFY", "SELECT", "SHOT", "CUT"
        };
        
        int min_distance = 999;
        std::string best_match;
        
        for (const auto & verb : all_verbs) {
            int dist = levenshtein_distance(upper_input, verb);
            if (dist < min_distance && dist <= 2) {  // Allow up to 2 edits
                min_distance = dist;
                best_match = verb;
            }
        }
        
        return best_match;
    }

private:
    const SchemaRegistry & registry;
    
    bool validate_parameter_type(const json & value, const std::string & expected_type) const {
        if (expected_type == "string") return value.is_string();
        if (expected_type == "number") return value.is_number();
        if (expected_type == "boolean") return value.is_boolean();
        if (expected_type == "array") return value.is_array();
        if (expected_type == "object") return value.is_object();
        return true; // Unknown type, allow it
    }
    
    static std::string to_upper(const std::string & str) {
        std::string result = str;
        for (char & c : result) {
            c = std::toupper(c);
        }
        return result;
    }
    
    // Simple Levenshtein distance for fuzzy matching
    static int levenshtein_distance(const std::string & s1, const std::string & s2) {
        const size_t m = s1.size();
        const size_t n = s2.size();
        
        if (m == 0) return n;
        if (n == 0) return m;
        
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
        
        for (size_t i = 0; i <= m; ++i) dp[i][0] = i;
        for (size_t j = 0; j <= n; ++j) dp[0][j] = j;
        
        for (size_t i = 1; i <= m; ++i) {
            for (size_t j = 1; j <= n; ++j) {
                int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
                dp[i][j] = std::min({
                    dp[i-1][j] + 1,      // deletion
                    dp[i][j-1] + 1,      // insertion
                    dp[i-1][j-1] + cost  // substitution
                });
            }
        }
        
        return dp[m][n];
    }
};

} // namespace semantic_server
