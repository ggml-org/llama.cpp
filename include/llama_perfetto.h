// Lightweight C-callable shim for Perfetto tracepoints used across C and C++.
// If Perfetto SDK is not available or tracing is disabled, these become no-ops.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Begin a CPU trace span for ML operation `name`.
void llama_perfetto_trace_begin(const char * name);

// Begin a CPU trace span and attach a string argument named "text".
// Useful for spans like "decode" where we want to see the token string.
void llama_perfetto_trace_begin_with_text(const char * name, const char * text);

// End the most recent CPU trace span started with begin.
void llama_perfetto_trace_end(void);

// GPU span helpers (Vulkan compute regions)
void llama_perfetto_gpu_begin(const char * name);
void llama_perfetto_gpu_end(void);

// Optional: start/stop a trace session writing TrackEvent data to a file.
// If you prefer env control, call `llama_perfetto_try_start_from_env()`.
void llama_perfetto_start_trace(const char * path);
void llama_perfetto_stop_flush(void);
// Flushes the active tracing session (if any) and writes Vulkan stats
// without stopping the tracing session. Safe to call multiple times.
void llama_perfetto_flush_dump_stats(void);
// Prints Vulkan GPU counters to stdout if available.
void llama_perfetto_print_gpu_stats(void);
// Emits a GPU timeline track into the Perfetto trace (if tracing).
// Uses the latest Vulkan timestamp batch and anchors it to the current
// trace clock so spans end at "now" preserving relative shape.
void llama_perfetto_emit_gpu_timeline(void);
void llama_perfetto_try_start_from_env(void);

// Counter helpers
// Emits a Perfetto counter sample for tokens per second (throughput).
void llama_perfetto_counter_tokens_per_s(double tokens_per_s);
// Optionally, emit GPU busy percent [0..100]. Usually called internally.
void llama_perfetto_counter_gpu_busy(double percent);

#ifdef __cplusplus
}
#endif
