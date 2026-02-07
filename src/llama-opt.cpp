// CPUOPTI: Deterministic runtime optimization layer — configuration and statistics

#include "llama-opt.h"
#include "llama-impl.h"

#include <cstdlib>
#include <cstring>
#include <cinttypes>

//
// Environment variable helpers
//

int llama_opt_env_int(const char * name, int default_val) {
    const char * val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return default_val;
    }
    return atoi(val);
}

bool llama_opt_env_bool(const char * name, bool default_val) {
    const char * val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return default_val;
    }
    // "0", "false", "no", "off" → false; anything else → true
    if (strcmp(val, "0") == 0 || strcmp(val, "false") == 0 ||
        strcmp(val, "no") == 0 || strcmp(val, "off") == 0) {
        return false;
    }
    return true;
}

std::string llama_opt_env_str(const char * name, const std::string & default_val) {
    const char * val = getenv(name);
    if (val == nullptr) {
        return default_val;
    }
    return std::string(val);
}

//
// Configuration initialization
//

llama_opt_config llama_opt_config_init() {
    llama_opt_config cfg;

    cfg.block_size = (uint32_t) llama_opt_env_int("LLAMA_OPT_BLOCK_SIZE", 64);

#ifdef LLAMA_OPT_DEDUP
    cfg.dedup_enabled  = llama_opt_env_bool("LLAMA_OPT_DEDUP_ENABLED", true);
    cfg.dedup_pool_max = (uint32_t) llama_opt_env_int("LLAMA_OPT_DEDUP_POOL_MAX", 16384);
#else
    cfg.dedup_enabled  = false;
    cfg.dedup_pool_max = 0;
#endif

#ifdef LLAMA_OPT_KV_DIFF
    cfg.diff_enabled       = llama_opt_env_bool("LLAMA_OPT_DIFF_ENABLED", true);
    cfg.diff_min_unchanged = (uint32_t) llama_opt_env_int("LLAMA_OPT_DIFF_MIN_UNCHANGED", 8);
#else
    cfg.diff_enabled       = false;
    cfg.diff_min_unchanged = 0;
#endif

#ifdef LLAMA_OPT_SCHEMA_SKIP
    cfg.schema_skip_enabled = llama_opt_env_bool("LLAMA_OPT_SCHEMA_SKIP_ENABLED", true);
#else
    cfg.schema_skip_enabled = false;
#endif

    cfg.stats_enabled = llama_opt_env_bool("LLAMA_OPT_STATS", false);

    // Log configuration if stats are enabled
    if (cfg.stats_enabled) {
        LLAMA_LOG_INFO("%s: CPUOPTI runtime optimization config:\n", __func__);
        LLAMA_LOG_INFO("%s:   block_size        = %u\n", __func__, cfg.block_size);
        LLAMA_LOG_INFO("%s:   dedup_enabled     = %s\n", __func__, cfg.dedup_enabled ? "true" : "false");
        LLAMA_LOG_INFO("%s:   dedup_pool_max    = %u\n", __func__, cfg.dedup_pool_max);
        LLAMA_LOG_INFO("%s:   diff_enabled      = %s\n", __func__, cfg.diff_enabled ? "true" : "false");
        LLAMA_LOG_INFO("%s:   diff_min_unchanged = %u\n", __func__, cfg.diff_min_unchanged);
        LLAMA_LOG_INFO("%s:   schema_skip       = %s\n", __func__, cfg.schema_skip_enabled ? "true" : "false");
    }

    return cfg;
}

//
// Statistics
//

void llama_opt_stats::reset() {
    dedup_blocks_total     = 0;
    dedup_blocks_hit       = 0;
    dedup_tokens_saved     = 0;

    diff_tokens_total      = 0;
    diff_tokens_unchanged  = 0;
    diff_tokens_recomputed = 0;

    schema_tokens_total    = 0;
    schema_tokens_skipped  = 0;
    schema_tokens_inferred = 0;
}

void llama_opt_stats::print() const {
    LLAMA_LOG_INFO("%s: === CPUOPTI optimization statistics ===\n", __func__);

    if (dedup_blocks_total > 0) {
        const float hit_rate = (dedup_blocks_total > 0)
            ? (100.0f * (float)dedup_blocks_hit / (float)dedup_blocks_total)
            : 0.0f;
        LLAMA_LOG_INFO("%s: dedup: blocks_total=%" PRIu64 " blocks_hit=%" PRIu64
                       " (%.1f%%) tokens_saved=%" PRIu64 "\n",
                       __func__, dedup_blocks_total, dedup_blocks_hit, hit_rate, dedup_tokens_saved);
    }

    if (diff_tokens_total > 0) {
        const float reuse_rate = (diff_tokens_total > 0)
            ? (100.0f * (float)diff_tokens_unchanged / (float)diff_tokens_total)
            : 0.0f;
        LLAMA_LOG_INFO("%s: diff: tokens_total=%" PRIu64 " unchanged=%" PRIu64
                       " (%.1f%%) recomputed=%" PRIu64 "\n",
                       __func__, diff_tokens_total, diff_tokens_unchanged, reuse_rate, diff_tokens_recomputed);
    }

    if (schema_tokens_total > 0) {
        const float skip_rate = (schema_tokens_total > 0)
            ? (100.0f * (float)schema_tokens_skipped / (float)schema_tokens_total)
            : 0.0f;
        LLAMA_LOG_INFO("%s: schema: tokens_total=%" PRIu64 " skipped=%" PRIu64
                       " (%.1f%%) inferred=%" PRIu64 "\n",
                       __func__, schema_tokens_total, schema_tokens_skipped, skip_rate, schema_tokens_inferred);
    }

    LLAMA_LOG_INFO("%s: =======================================\n", __func__);
}
