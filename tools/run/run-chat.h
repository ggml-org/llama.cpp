#pragma once

#include "common.h"
#include <functional>

// Forward declarations
struct server_context;

// Run interactive chat mode
void run_chat_mode(const common_params & params, server_context & ctx_server);

// Setup platform-specific signal handlers for console interruption
void setup_signal_handlers(std::function<void(int)> handler);
