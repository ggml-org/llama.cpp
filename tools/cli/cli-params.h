#pragma once

#include "common.h"

// apply CLI-specific post-parse adjustments
void cli_params_finalize(common_params & params);

// non-conversation mode needs a predefined prompt
bool cli_run_requires_prompt(const common_params & params);
