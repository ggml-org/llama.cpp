#pragma once

#include "log.h"
#include "llama.h"

// Initialize the shared logger for the router tool. This routes llama.cpp
// internal logs through the common logger and ensures they only go to
// stdout/stderr (no log files are created). We mirror the llama-server
// defaults directly (console-only with auto colors, no prefix or timestamp),
// so no runtime JSON option is required.
void router_log_init();

