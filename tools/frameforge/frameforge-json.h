#ifndef FRAMEFORGE_JSON_H
#define FRAMEFORGE_JSON_H

#include "frameforge-schema.h"
#include <string>

namespace frameforge {

// Convert Command to JSON string
std::string command_to_json(const Command & cmd);

// Parse JSON string to Command
Command json_to_command(const std::string & json_str);

// Convert ValidationResult to JSON error object
std::string validation_error_to_json(const std::string & error_message, 
                                      const std::vector<std::string> & missing_params);

} // namespace frameforge

#endif // FRAMEFORGE_JSON_H
