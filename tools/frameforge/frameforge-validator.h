#ifndef FRAMEFORGE_VALIDATOR_H
#define FRAMEFORGE_VALIDATOR_H

#include "frameforge-schema.h"
#include <string>

namespace frameforge {

// Validation result structure
struct ValidationResult {
    bool valid;
    std::string error_message;
    std::vector<std::string> missing_parameters;
};

// CommandValidator class
class CommandValidator {
public:
    CommandValidator();
    ~CommandValidator();
    
    // Validate a command against the schema
    ValidationResult validate(const Command & cmd) const;
    
    // Validate JSON string and parse into Command
    ValidationResult validate_json(const std::string & json_str, Command & out_cmd) const;
    
    // Generate clarification request for missing parameters
    std::string generate_clarification_request(const ValidationResult & result, const Command & cmd) const;
    
private:
    // Check if required parameters are present
    bool check_required_parameters(const Command & cmd, std::vector<std::string> & missing) const;
    
    // Validate parameter values
    bool validate_parameter_values(const Command & cmd, std::string & error) const;
};

} // namespace frameforge

#endif // FRAMEFORGE_VALIDATOR_H
