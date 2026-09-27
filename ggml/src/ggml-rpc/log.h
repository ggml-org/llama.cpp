#pragma once

// Log infrastructure for the RPC backend: all RPC logging goes through this header.
//
// LOG_ERROR / LOG_WARN / LOG_INFO are unconditional severity logs.
// LOG_DBG* are debug logs gated by the GGML_RPC_DEBUG verbosity variable:
//   unset / 0 - disabled (only warnings and errors are printed)
//   1         - high-level events: connections, handshake, buffer operations, tensor transfers, graph computes
//   2         - per-command trace: every RPC message sent/received, cache hits/misses, queue events
//   3         - transport detail: byte counts, per-command timings, graph node details
// non-numeric values fall back to 1

#include "ggml-impl.h"

#include <cstdlib>

static inline int rpc_debug_level() {
    static const int level = [] {
        const char * env = std::getenv("GGML_RPC_DEBUG");
        if (env == nullptr) {
            return 0;
        }
        if (env[0] == '\0') {
            return 1;
        }
        int value = 0;
        for (const char * p = env; *p != '\0'; p++) {
            if (*p < '0' || *p > '9') {
                return 1;
            }
            value = value * 10 + (*p - '0');
            if (value > 3) {
                value = 3;
            }
        }
        return value;
    }();
    return level;
}

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  do { if (rpc_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)
#define LOG_DBG2(...) do { if (rpc_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)
#define LOG_DBG3(...) do { if (rpc_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)
