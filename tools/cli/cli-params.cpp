#include "cli-params.h"

void cli_params_finalize(common_params & params) {
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED && !params.prompt.empty()) {
        params.single_turn = true;
    }
}

bool cli_run_requires_prompt(const common_params & params) {
    return params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED && params.prompt.empty();
}
