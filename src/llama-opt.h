#pragma once

// CPUOPTI: Deterministic runtime optimization layer — configuration and statistics

#include "llama.h"

#include <cstdint>
#include <string>

//
// Compile-time feature detection
// These are defined by CMake options (LLAMA_OPT_DEDUP, LLAMA_OPT_KV_DIFF, etc.)
// If not defined, the corresponding code paths are compiled out entirely.
//

//
// Runtime configuration
//

struct llama_opt_config {
    // Block hashing
    uint32_t block_size = 64;

    // Phase 1 — Context block deduplication
    bool     dedup_enabled      = true;
    uint32_t dedup_pool_max     = 16384;

    // Phase 1 — Structural KV cache diffing
    bool     diff_enabled       = true;
    uint32_t diff_min_unchanged = 8;

    // Phase 1 — Schema-aware token skipping
    bool     schema_skip_enabled = true;

    // Statistics
    bool     stats_enabled       = false;
};

//
// Statistics tracking
//

struct llama_opt_stats {
    // Phase 1 — Context block dedup
    uint64_t dedup_blocks_total    = 0;
    uint64_t dedup_blocks_hit      = 0;
    uint64_t dedup_tokens_saved    = 0;

    // Phase 1 — KV cache diffing
    uint64_t diff_tokens_total     = 0;
    uint64_t diff_tokens_unchanged = 0;
    uint64_t diff_tokens_recomputed = 0;

    // Phase 1 — Schema-aware skipping
    uint64_t schema_tokens_total   = 0;
    uint64_t schema_tokens_skipped = 0;
    uint64_t schema_tokens_inferred = 0;

    void reset();
    void print() const;
};

//
// Global initialization / shutdown
//

// Initialize the optimization config from compile-time flags and environment variables
llama_opt_config llama_opt_config_init();

// Helper: read an environment variable as int, or return default
int         llama_opt_env_int(const char * name, int default_val);
bool        llama_opt_env_bool(const char * name, bool default_val);
std::string llama_opt_env_str(const char * name, const std::string & default_val);
