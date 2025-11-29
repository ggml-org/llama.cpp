#include "logging.h"

void router_log_init() {
    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);

    // Always reset to console-only logging for the router tool. This mirrors
    // the llama-server defaults (stdout for info, stderr otherwise) and keeps
    // formatting consistent with llama.cpp so router diagnostics and the core
    // library share the same console sink.
    auto * log = common_log_main();
    common_log_set_file(log, nullptr);
    common_log_set_colors(log, LOG_COLORS_AUTO);
    common_log_set_prefix(log, false);
    common_log_set_timestamps(log, false);

    // Forward llama.cpp internals through the common logger so any backend
    // message (tokenization, sampling, etc.) shows up alongside router
    // messages without a second logging pipeline.
    llama_log_set(common_log_default_callback, nullptr);
}
